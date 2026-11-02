"""Add approver grants, protected policies and the approval verdict ledger.

Revision ID: ap1a2b3c4d5e
Revises: ri1a2b3c4d5e
Create Date: 2026-11-02 00:00:00.000000

Adds four tables:

- ``approval_resources``: per-resource approval settings (whether the person
  who started an action may approve it) for a thread, a service or the team.
- ``approver_grants``: people named to answer approvals for a resource,
  separate from thread read/edit/manage/owner grants.
- ``protected_policies``: team rules administrators keep and personal rules
  their person keeps, evaluated after every other policy.
- ``approval_verdicts``: one row per protected approval, binding what was
  asked, recording answers, and consuming an approval once.

Deployment: additive. Older application code ignores the new tables; no
existing row changes. Each table and index is created only when missing, so an
upgrade interrupted after MySQL or CockroachDB committed part of it resumes on
the next start. Downgrading drops the tables, including grants, rules and
verdicts made since.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.mysql import BINARY as MySQLBinary

from omnigent.db.db_models import Uuid16

revision: str = "ap1a2b3c4d5e"
down_revision: str | None = "ri1a2b3c4d5e"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_CKSUM32 = sa.LargeBinary(32).with_variant(MySQLBinary(32), "mysql")
_TABLES = ("approval_resources", "approver_grants", "protected_policies", "approval_verdicts")
_INDEXES = (
    (
        "approver_grants",
        "ix_approver_grants_approver",
        ("workspace_id", "approver_user_id", "resource_cksum"),
    ),
    (
        "protected_policies",
        "ix_protected_policies_domain_owner",
        ("workspace_id", "domain", "owner_user_id", "id"),
    ),
    (
        "approval_verdicts",
        "ix_approval_verdicts_state_expiry",
        ("workspace_id", "state", "expires_at"),
    ),
)


def _create_tables(bind: sa.Connection) -> None:
    """Create each missing table, then each missing index."""
    inspector = sa.inspect(bind)
    if not inspector.has_table("approval_resources"):
        op.create_table(
            "approval_resources",
            sa.Column("workspace_id", sa.BigInteger(), nullable=False),
            sa.Column("resource_cksum", _CKSUM32, nullable=False),
            sa.Column("kind", sa.SmallInteger(), nullable=False),
            sa.Column("resource_id", sa.String(256), nullable=False),
            sa.Column("allow_initiator", sa.Boolean(), nullable=True),
            sa.Column("updated_at", sa.BigInteger(), nullable=False),
            sa.Column("updated_by", sa.String(128), nullable=True),
            sa.PrimaryKeyConstraint("workspace_id", "resource_cksum"),
            sa.CheckConstraint("kind IN (1, 2, 3)", name="ck_approval_resources_kind"),
        )
    if not inspector.has_table("approver_grants"):
        op.create_table(
            "approver_grants",
            sa.Column("workspace_id", sa.BigInteger(), nullable=False),
            sa.Column("resource_cksum", _CKSUM32, nullable=False),
            sa.Column("approver_user_id", sa.String(128), nullable=False),
            sa.Column("kind", sa.SmallInteger(), nullable=False),
            sa.Column("resource_id", sa.String(256), nullable=False),
            sa.Column("granted_by", sa.String(128), nullable=True),
            sa.Column("created_at", sa.BigInteger(), nullable=False),
            sa.PrimaryKeyConstraint("workspace_id", "resource_cksum", "approver_user_id"),
            sa.CheckConstraint("kind IN (1, 2, 3)", name="ck_approver_grants_kind"),
        )
    if not inspector.has_table("protected_policies"):
        op.create_table(
            "protected_policies",
            sa.Column("workspace_id", sa.BigInteger(), nullable=False),
            sa.Column("id", Uuid16(), nullable=False),
            sa.Column("domain", sa.SmallInteger(), nullable=False),
            sa.Column("owner_user_id", sa.String(128), nullable=True),
            sa.Column("name", sa.String(256), nullable=False),
            # CompressedText columns are opaque binary in the database.
            sa.Column("handler", sa.LargeBinary(), nullable=False),
            sa.Column("factory_params", sa.LargeBinary(), nullable=True),
            sa.Column("approval_kind", sa.SmallInteger(), nullable=True),
            sa.Column("approval_resource_id", sa.String(256), nullable=True),
            sa.Column("enabled", sa.Boolean(), nullable=False),
            sa.Column("created_by", sa.String(128), nullable=True),
            sa.Column("created_at", sa.BigInteger(), nullable=False),
            sa.Column("updated_at", sa.BigInteger(), nullable=False),
            sa.PrimaryKeyConstraint("workspace_id", "id"),
            sa.CheckConstraint("domain IN (1, 2)", name="ck_protected_policies_domain"),
            sa.CheckConstraint(
                "(domain = 1 AND owner_user_id IS NULL) "
                "OR (domain = 2 AND owner_user_id IS NOT NULL)",
                name="ck_protected_policies_owner",
            ),
            sa.CheckConstraint(
                "approval_kind IS NULL OR approval_kind IN (2, 3)",
                name="ck_protected_policies_approval_kind",
            ),
        )
    if not inspector.has_table("approval_verdicts"):
        op.create_table(
            "approval_verdicts",
            sa.Column("workspace_id", sa.BigInteger(), nullable=False),
            sa.Column("elicitation_id", sa.String(128), nullable=False),
            sa.Column("session_id", Uuid16(), nullable=False),
            sa.Column("binding_digest", _CKSUM32, nullable=False),
            sa.Column("binding", sa.LargeBinary(), nullable=False),
            sa.Column("requirements", sa.LargeBinary(), nullable=False),
            sa.Column("answers", sa.LargeBinary(), nullable=True),
            sa.Column("state", sa.SmallInteger(), nullable=False),
            sa.Column("decided_by", sa.String(128), nullable=True),
            sa.Column("decided_at", sa.BigInteger(), nullable=True),
            sa.Column("expires_at", sa.BigInteger(), nullable=False),
            sa.Column("consumed_at", sa.BigInteger(), nullable=True),
            sa.Column("operation_id", sa.String(128), nullable=True),
            sa.Column("outcome", sa.SmallInteger(), nullable=True),
            sa.Column("outcome_body", sa.LargeBinary(), nullable=True),
            sa.Column("created_at", sa.BigInteger(), nullable=False),
            sa.PrimaryKeyConstraint("workspace_id", "elicitation_id"),
            sa.CheckConstraint("state IN (1, 2, 3, 4)", name="ck_approval_verdicts_state"),
            sa.CheckConstraint(
                "outcome IS NULL OR outcome IN (1, 2, 3)", name="ck_approval_verdicts_outcome"
            ),
        )
    for table, name, columns in _INDEXES:
        if name not in {index["name"] for index in sa.inspect(bind).get_indexes(table)}:
            op.create_index(name, table, list(columns))


def upgrade() -> None:
    """Create the four tables and their indexes, skipping what already exists."""
    bind = op.get_bind()
    _create_tables(bind)
    if bind.dialect.name == "cockroachdb":
        # CockroachDB publishes new tables only at commit.
        bind.commit()
        bind.execute(sa.text("SET TRANSACTION ISOLATION LEVEL SERIALIZABLE"))


def downgrade() -> None:
    """Drop the tables; no other table is touched."""
    bind = op.get_bind()
    for table in reversed(_TABLES):
        if sa.inspect(bind).has_table(table):
            op.drop_table(table)
