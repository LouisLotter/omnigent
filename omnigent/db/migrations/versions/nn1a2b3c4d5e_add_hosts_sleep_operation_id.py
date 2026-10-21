"""Add a persistent host's pending sleep operation to hosts.

Revision ID: nn1a2b3c4d5e
Revises: mm1a2b3c4d5e
Create Date: 2026-10-20 00:00:00.000000

A persistent managed host's sleep fence records its operation id before the
provider stops the sandbox, so a server restart before the outcome is known
keeps admission closed and asks the provider for the recorded outcome.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "nn1a2b3c4d5e"
down_revision: str | None = "mm1a2b3c4d5e"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add the pending-sleep slot."""
    with op.batch_alter_table("hosts") as batch_op:
        batch_op.add_column(sa.Column("sleep_operation_id", sa.String(64), nullable=True))


def downgrade() -> None:
    """Remove the pending-sleep slot."""
    with op.batch_alter_table("hosts") as batch_op:
        batch_op.drop_column("sleep_operation_id")
