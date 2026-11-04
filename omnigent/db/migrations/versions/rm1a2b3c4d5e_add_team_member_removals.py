"""Add the record of each team member's removal.

Revision ID: rm1a2b3c4d5e
Revises: ti1a2b3c4d5e
Create Date: 2026-11-04 00:00:00.000000

Adds:

- ``team_member_removals``: one row per removed member, written in the
  removal's transaction: who removed them, when, and the admission revision
  it committed, so the team page can say which steps are done.
- ``team_member_removal_items``: the threads the member owned and the cloud
  computers handed to their provider for removal.

Deployment: additive. Older application code ignores both tables, and a
removal made by it simply has no record. Each step checks what already
exists, so an upgrade interrupted after MySQL or CockroachDB committed part of
it resumes on the next start. Downgrading drops both tables and the records
in them; the removals themselves stay.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.mysql import BINARY as MySQLBinary

revision: str = "rm1a2b3c4d5e"
down_revision: str | None = "ti1a2b3c4d5e"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_UUID = sa.LargeBinary(16).with_variant(MySQLBinary(16), "mysql")
_REMOVALS = "team_member_removals"
_ITEMS = "team_member_removal_items"
_INDEX = "ix_team_member_removals_removed_at"


def upgrade() -> None:
    """Create both tables and the index, skipping what exists."""
    bind = op.get_bind()
    if not sa.inspect(bind).has_table(_REMOVALS):
        op.create_table(
            _REMOVALS,
            sa.Column("workspace_id", sa.BigInteger(), nullable=False),
            sa.Column("member_id", _UUID, nullable=False),
            sa.Column("user_id", sa.String(128), nullable=False),
            sa.Column("removed_by_member_id", _UUID, nullable=True),
            sa.Column("removed_at", sa.BigInteger(), nullable=False),
            sa.Column("admission_revision", sa.BigInteger(), nullable=False),
            sa.PrimaryKeyConstraint("workspace_id", "member_id"),
        )
    if _INDEX not in {index["name"] for index in sa.inspect(bind).get_indexes(_REMOVALS)}:
        op.create_index(_INDEX, _REMOVALS, ["workspace_id", "removed_at", "member_id"])
    if not sa.inspect(bind).has_table(_ITEMS):
        op.create_table(
            _ITEMS,
            sa.Column("workspace_id", sa.BigInteger(), nullable=False),
            sa.Column("member_id", _UUID, nullable=False),
            sa.Column("kind", sa.SmallInteger(), nullable=False),
            sa.Column("item_id", _UUID, nullable=False),
            sa.PrimaryKeyConstraint("workspace_id", "member_id", "kind", "item_id"),
            sa.CheckConstraint("kind IN (1, 2)", name="ck_team_member_removal_items_kind"),
        )


def downgrade() -> None:
    """Drop both tables; memberships and the accounts they name stay as they are."""
    bind = op.get_bind()
    for table in (_ITEMS, _REMOVALS):
        if sa.inspect(bind).has_table(table):
            op.drop_table(table)
