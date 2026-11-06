"""Join the team schema and the 0.21 release schema.

Revision ID: c0211a2b3c4d
Revises: rm1a2b3c4d5e, d0a1f879a01d

Both published branches remain unchanged. The merge only reconciles Alembic's
version rows; each parent owns its schema and data changes.
"""

from __future__ import annotations

revision: str = "c0211a2b3c4d"
down_revision: tuple[str, ...] = ("rm1a2b3c4d5e", "d0a1f879a01d")
branch_labels: tuple[str, ...] | None = None
depends_on: tuple[str, ...] | None = None


def upgrade() -> None:
    """Join both parent revisions without changing their data."""


def downgrade() -> None:
    """Restore both parent version markers without undoing either schema."""
