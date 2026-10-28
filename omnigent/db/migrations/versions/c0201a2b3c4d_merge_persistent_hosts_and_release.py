"""Join persistent-host identity and the 0.20 release schema.

Revision ID: c0201a2b3c4d
Revises: lt1b2c3d4e5f, b2e4f6a8c013

Both published branches remain unchanged. The merge only reconciles Alembic's
version rows; each parent owns its schema and data changes.
"""

from __future__ import annotations

revision: str = "c0201a2b3c4d"
down_revision: tuple[str, ...] = ("lt1b2c3d4e5f", "b2e4f6a8c013")
branch_labels: tuple[str, ...] | None = None
depends_on: tuple[str, ...] | None = None


def upgrade() -> None:
    """Join both parent revisions without changing their data."""


def downgrade() -> None:
    """Restore both parent version markers without undoing either schema."""
