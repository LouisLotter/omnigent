"""Add team invitations, the verified address of a binding and the admission revision.

Revision ID: ti1a2b3c4d5e
Revises: at1a2b3c4d5e
Create Date: 2026-11-03 00:00:00.000000

Adds:

- ``team_invitations``: single-use invitations to join the team with a role,
  or to link an existing account, for one email address. Only the token's
  digest is stored.
- ``team_identity_bindings.email``: the address an identity was verified
  with, for a front proxy's admission list; existing bindings keep NULL.
- ``teams.admission_revision``: increases with each change to who a front
  proxy should admit; existing teams start at 0.
- ``teams.founding_member_id``: the founding owner membership, which the
  proxy's sign-in of the installation's operator may bind once. Backfilled
  for a team whose only active owner has no identity binding yet.

Deployment: additive. Older application code ignores the new table and
columns. Each step checks what already exists, so an upgrade interrupted after
MySQL or CockroachDB committed part of it resumes on the next start.
Downgrading drops the table and the three columns, including invitations made since.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.mysql import BINARY as MySQLBinary

revision: str = "ti1a2b3c4d5e"
down_revision: str | None = "at1a2b3c4d5e"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_UUID = sa.LargeBinary(16).with_variant(MySQLBinary(16), "mysql")
_OWNER_ROLE = 1
_ACTIVE_STATE = 2
_TEAMS = sa.table(
    "teams",
    sa.column("workspace_id", sa.BigInteger()),
    sa.column("founding_member_id", _UUID),
)
_MEMBERS = sa.table(
    "team_members",
    sa.column("workspace_id", sa.BigInteger()),
    sa.column("id", _UUID),
    sa.column("role", sa.SmallInteger()),
    sa.column("state", sa.SmallInteger()),
)
_BINDINGS = sa.table(
    "team_identity_bindings",
    sa.column("workspace_id", sa.BigInteger()),
    sa.column("member_id", _UUID),
)
_CKSUM32 = sa.LargeBinary(32).with_variant(MySQLBinary(32), "mysql")
_INDEXES = (
    (
        "team_invitations",
        "ix_team_invitations_state_expiry",
        ("workspace_id", "state", "expires_at"),
    ),
    ("team_invitations", "ix_team_invitations_email", ("workspace_id", "email", "state")),
)


def _columns(bind: sa.Connection, table: str) -> set[str]:
    return {column["name"] for column in sa.inspect(bind).get_columns(table)}


def upgrade() -> None:
    """Create the table, its indexes and the two columns, skipping what exists."""
    bind = op.get_bind()
    if not sa.inspect(bind).has_table("team_invitations"):
        op.create_table(
            "team_invitations",
            sa.Column("workspace_id", sa.BigInteger(), nullable=False),
            sa.Column("id", _UUID, nullable=False),
            sa.Column("token_digest", _CKSUM32, nullable=False),
            sa.Column("email", sa.String(320), nullable=False),
            sa.Column("role", sa.SmallInteger(), nullable=False),
            sa.Column("target_user_id", sa.String(128), nullable=True),
            sa.Column("created_by_member_id", _UUID, nullable=True),
            sa.Column("created_at", sa.BigInteger(), nullable=False),
            sa.Column("expires_at", sa.BigInteger(), nullable=False),
            sa.Column("state", sa.SmallInteger(), nullable=False),
            sa.Column("closed_at", sa.BigInteger(), nullable=True),
            sa.Column("redeemed_member_id", _UUID, nullable=True),
            sa.PrimaryKeyConstraint("workspace_id", "id"),
            sa.CheckConstraint("role IN (2, 3)", name="ck_team_invitations_role"),
            sa.CheckConstraint("state IN (1, 2, 3, 4)", name="ck_team_invitations_state"),
            sa.UniqueConstraint("workspace_id", "token_digest", name="uq_team_invitations_token"),
        )
    for table, name, columns in _INDEXES:
        if name not in {index["name"] for index in sa.inspect(bind).get_indexes(table)}:
            op.create_index(name, table, list(columns))
    if "email" not in _columns(bind, "team_identity_bindings"):
        op.add_column("team_identity_bindings", sa.Column("email", sa.String(320), nullable=True))
    if "admission_revision" not in _columns(bind, "teams"):
        op.add_column(
            "teams",
            sa.Column("admission_revision", sa.BigInteger(), nullable=False, server_default="0"),
        )
    if "founding_member_id" not in _columns(bind, "teams"):
        op.add_column("teams", sa.Column("founding_member_id", _UUID, nullable=True))
    if bind.dialect.name == "cockroachdb":
        # CockroachDB publishes new tables only at commit.
        bind.commit()
        bind.execute(sa.text("SET TRANSACTION ISOLATION LEVEL SERIALIZABLE"))
    _backfill_founding_members(bind)


def _backfill_founding_members(bind: sa.Connection) -> None:
    """Name each team's founding member: its only active owner, while that owner is unbound."""
    teams = bind.execute(
        sa.select(_TEAMS.c.workspace_id).where(_TEAMS.c.founding_member_id.is_(None))
    ).scalars()
    for workspace_id in list(teams):
        owners = bind.execute(
            sa.select(_MEMBERS.c.id)
            .where(
                _MEMBERS.c.workspace_id == workspace_id,
                _MEMBERS.c.role == _OWNER_ROLE,
                _MEMBERS.c.state == _ACTIVE_STATE,
            )
            .limit(2)
        ).scalars()
        found = list(owners)
        if len(found) != 1:
            continue
        bound = bind.execute(
            sa.select(_BINDINGS.c.member_id)
            .where(
                _BINDINGS.c.workspace_id == workspace_id,
                _BINDINGS.c.member_id == found[0],
            )
            .limit(1)
        ).first()
        if bound is None:
            bind.execute(
                _TEAMS.update()
                .where(_TEAMS.c.workspace_id == workspace_id)
                .values(founding_member_id=found[0])
            )


def downgrade() -> None:
    """Drop the table and both columns; memberships and bindings stay."""
    bind = op.get_bind()
    if sa.inspect(bind).has_table("team_invitations"):
        op.drop_table("team_invitations")
    if "email" in _columns(bind, "team_identity_bindings"):
        with op.batch_alter_table("team_identity_bindings") as batch_op:
            batch_op.drop_column("email")
    for column in ("admission_revision", "founding_member_id"):
        if column in _columns(bind, "teams"):
            with op.batch_alter_table("teams") as batch_op:
                batch_op.drop_column(column)
