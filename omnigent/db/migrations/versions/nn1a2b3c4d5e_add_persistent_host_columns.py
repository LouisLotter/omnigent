"""Add persistent managed host columns to hosts.

Revision ID: nn1a2b3c4d5e
Revises: mm1a2b3c4d5e
Create Date: 2026-10-20 00:00:00.000000

A persistent managed host outlives sessions and sleeps when idle. Its row
records that lifecycle, the operation id of a sleep whose outcome is not yet
known (so a restart keeps admission closed and asks the provider for the
recorded outcome), and why its last start failed.
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
    """Add the lifecycle, pending-sleep and start-failure columns."""
    with op.batch_alter_table("hosts") as batch_op:
        batch_op.add_column(sa.Column("lifecycle", sa.SmallInteger(), nullable=True))
        batch_op.add_column(sa.Column("sleep_operation_id", sa.String(64), nullable=True))
        batch_op.add_column(sa.Column("start_failure", sa.String(512), nullable=True))


def downgrade() -> None:
    """Remove the persistent managed host columns."""
    with op.batch_alter_table("hosts") as batch_op:
        batch_op.drop_column("start_failure")
        batch_op.drop_column("sleep_operation_id")
        batch_op.drop_column("lifecycle")
