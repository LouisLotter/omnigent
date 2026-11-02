"""Add the team, membership and external-identity tables.

Revision ID: tm1a2b3c4d5e
Revises: c0201a2b3c4d
Create Date: 2026-11-02 00:00:00.000000

Every workspace gets its team. Where exactly one live account holds the admin
flag, that account becomes the team's active owner under its existing
username; no other account, thread, host or schedule changes. A workspace
with no admin, or with several, keeps an ownerless team, and the team
inventory reports it instead of guessing.

Deployment: additive. Older application code ignores the new tables. Each
step checks what already exists, so an upgrade interrupted after MySQL or
CockroachDB committed part of it resumes on the next start. Downgrading drops
the tables, including memberships and identity bindings made since.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.mysql import BINARY as MySQLBinary

revision: str = "tm1a2b3c4d5e"
down_revision: str | None = "c0201a2b3c4d"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_UUID = sa.LargeBinary(16).with_variant(MySQLBinary(16), "mysql")
_CKSUM32 = sa.LargeBinary(32).with_variant(MySQLBinary(32), "mysql")
_RESERVED_USERS = ("local", "__public__")
_OWNER_ROLE = 1
_ACTIVE_STATE = 2
_REMOVED_STATE = 4
_PAGE_SIZE = 500

_USERS = sa.table(
    "users",
    sa.column("workspace_id", sa.BigInteger()),
    sa.column("id", sa.String(128)),
    sa.column("is_admin", sa.Boolean()),
    sa.column("deleted_at", sa.Integer()),
)
_TEAMS = sa.table(
    "teams",
    sa.column("workspace_id", sa.BigInteger()),
    sa.column("id", _UUID),
    sa.column("name", sa.String(128)),
    sa.column("created_at", sa.BigInteger()),
)
_MEMBERS = sa.table(
    "team_members",
    sa.column("workspace_id", sa.BigInteger()),
    sa.column("id", _UUID),
    sa.column("user_id", sa.String(128)),
    sa.column("current_user_id", sa.String(128)),
    sa.column("role", sa.SmallInteger()),
    sa.column("state", sa.SmallInteger()),
    sa.column("created_at", sa.BigInteger()),
    sa.column("updated_at", sa.BigInteger()),
)
_INDEXES = (
    ("team_members", "ix_team_members_role_state", ("workspace_id", "role", "state", "id")),
    (
        "team_identity_bindings",
        "ix_team_identity_bindings_member",
        ("workspace_id", "member_id", "identity_cksum"),
    ),
)


def _create_tables(bind: sa.Connection) -> None:
    """Create each missing table, then each missing index.

    MySQL commits every statement, so a retry can find a table without its
    index; constraints are part of ``CREATE TABLE`` and arrive with it.
    """
    inspector = sa.inspect(bind)
    if not inspector.has_table("teams"):
        op.create_table(
            "teams",
            # A lone integer key must not become AUTO_INCREMENT or a sequence.
            sa.Column("workspace_id", sa.BigInteger(), nullable=False, autoincrement=False),
            sa.Column("id", _UUID, nullable=False),
            sa.Column("name", sa.String(128), nullable=True),
            sa.Column("created_at", sa.BigInteger(), nullable=False),
            sa.PrimaryKeyConstraint("workspace_id"),
        )
    if not inspector.has_table("team_members"):
        op.create_table(
            "team_members",
            sa.Column("workspace_id", sa.BigInteger(), nullable=False),
            sa.Column("id", _UUID, nullable=False),
            sa.Column("user_id", sa.String(128), nullable=False),
            sa.Column("current_user_id", sa.String(128), nullable=True),
            sa.Column("role", sa.SmallInteger(), nullable=False),
            sa.Column("state", sa.SmallInteger(), nullable=False),
            sa.Column("created_at", sa.BigInteger(), nullable=False),
            sa.Column("updated_at", sa.BigInteger(), nullable=False),
            sa.PrimaryKeyConstraint("workspace_id", "id"),
            sa.CheckConstraint("role IN (1, 2, 3)", name="ck_team_members_role"),
            sa.CheckConstraint("state IN (1, 2, 3, 4)", name="ck_team_members_state"),
            sa.CheckConstraint(
                "(state = 4 AND current_user_id IS NULL) "
                "OR (state <> 4 AND current_user_id = user_id)",
                name="ck_team_members_current_user",
            ),
            sa.UniqueConstraint(
                "workspace_id", "current_user_id", name="uq_team_members_current_user"
            ),
        )
    if not inspector.has_table("team_identity_bindings"):
        op.create_table(
            "team_identity_bindings",
            sa.Column("workspace_id", sa.BigInteger(), nullable=False),
            sa.Column("identity_cksum", _CKSUM32, nullable=False),
            sa.Column("issuer", sa.String(512), nullable=False),
            sa.Column("subject", sa.String(255), nullable=False),
            sa.Column("audience", sa.String(255), nullable=False),
            sa.Column("member_id", _UUID, nullable=False),
            sa.Column("provenance", sa.SmallInteger(), nullable=False),
            sa.Column("bound_by_member_id", _UUID, nullable=True),
            sa.Column("created_at", sa.BigInteger(), nullable=False),
            sa.PrimaryKeyConstraint("workspace_id", "identity_cksum"),
            sa.CheckConstraint(
                "provenance IN (1, 2, 3)", name="ck_team_identity_bindings_provenance"
            ),
        )
    for table, name, columns in _INDEXES:
        if name not in {index["name"] for index in sa.inspect(bind).get_indexes(table)}:
            op.create_index(name, table, list(columns))


def _publish_crdb_changes(bind: sa.Connection) -> None:
    if bind.dialect.name == "cockroachdb":
        # CockroachDB publishes new tables only at commit.
        bind.commit()
        bind.execute(sa.text("SET TRANSACTION ISOLATION LEVEL SERIALIZABLE"))


def _workspace_ids(bind: sa.Connection) -> list[int]:
    """Return workspace 0 and every workspace holding a user row, in order."""
    found = [0]
    cursor = 0
    while True:
        next_id = bind.execute(
            sa.select(_USERS.c.workspace_id)
            .where(_USERS.c.workspace_id > cursor)
            .order_by(_USERS.c.workspace_id)
            .limit(1)
        ).scalar_one_or_none()
        if next_id is None:
            return found
        found.append(next_id)
        cursor = next_id


def _sole_admin(bind: sa.Connection, workspace_id: int) -> str | None:
    """Return the workspace's only live admin, or ``None`` for none or several."""
    admins: list[str] = []
    after = ""
    while len(admins) < 2:
        rows = bind.execute(
            sa.select(_USERS.c.id, _USERS.c.is_admin, _USERS.c.deleted_at)
            .where(_USERS.c.workspace_id == workspace_id, _USERS.c.id > after)
            .order_by(_USERS.c.id)
            .limit(_PAGE_SIZE)
        ).all()
        if not rows:
            break
        admins.extend(
            row.id
            for row in rows
            if row.is_admin and row.deleted_at is None and row.id not in _RESERVED_USERS
        )
        after = rows[-1].id
    return admins[0] if len(admins) == 1 else None


def _backfill_team(bind: sa.Connection, workspace_id: int, now: int) -> None:
    """Create the workspace's team, and its owner when that is unambiguous."""
    has_team = bind.execute(
        sa.select(_TEAMS.c.workspace_id).where(_TEAMS.c.workspace_id == workspace_id)
    ).first()
    if has_team is None:
        bind.execute(
            _TEAMS.insert().values(
                workspace_id=workspace_id, id=uuid.uuid4().bytes, name=None, created_at=now
            )
        )
    has_member = bind.execute(
        sa.select(_MEMBERS.c.id)
        .where(_MEMBERS.c.workspace_id == workspace_id, _MEMBERS.c.state != _REMOVED_STATE)
        .limit(1)
    ).first()
    if has_member is not None:
        return
    owner = _sole_admin(bind, workspace_id)
    if owner is None:
        return
    bind.execute(
        _MEMBERS.insert().values(
            workspace_id=workspace_id,
            id=uuid.uuid4().bytes,
            user_id=owner,
            current_user_id=owner,
            role=_OWNER_ROLE,
            state=_ACTIVE_STATE,
            created_at=now,
            updated_at=now,
        )
    )


def upgrade() -> None:
    """Create the tables and give each workspace its team of one."""
    bind = op.get_bind()
    _create_tables(bind)
    _publish_crdb_changes(bind)
    now = int(time.time())
    for workspace_id in _workspace_ids(bind):
        _backfill_team(bind, workspace_id, now)


def downgrade() -> None:
    """Drop the tables; accounts and every resource they own are untouched."""
    bind = op.get_bind()
    for table in ("team_identity_bindings", "team_members", "teams"):
        if sa.inspect(bind).has_table(table):
            op.drop_table(table)
