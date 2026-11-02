"""Record the host sandbox a thread's runner was bound into.

Revision ID: ri1a2b3c4d5e
Revises: tm1a2b3c4d5e
Create Date: 2026-11-02 00:00:00.000000

Adds the nullable ``omnigent_conversation_metadata.runner_host_incarnation``:
the bound host's ``sandbox_id`` at the time the thread's runner was bound
(NULL for a host without a sandbox). Runner tokens are honoured only while the
host still runs that sandbox, so replacing or detaching a managed host's
sandbox retires the runners launched into the old one.

Existing bindings are stamped with their host's current sandbox, so runners
bound before the upgrade keep working until that sandbox changes.

Deployment: additive; older application code ignores the column. The column
is added only when missing, so an upgrade interrupted after MySQL or
CockroachDB committed it resumes on the next start. Downgrading drops it.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "ri1a2b3c4d5e"
down_revision: str | None = "tm1a2b3c4d5e"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLE = "omnigent_conversation_metadata"
_COLUMN = "runner_host_incarnation"

_METADATA = sa.table(
    _TABLE,
    sa.column("workspace_id"),
    sa.column("host_id"),
    sa.column("runner_id"),
    sa.column(_COLUMN),
)
_HOSTS = sa.table(
    "hosts",
    sa.column("workspace_id"),
    sa.column("host_id"),
    sa.column("sandbox_id"),
)


def _has_column(bind: sa.Connection) -> bool:
    return _COLUMN in {column["name"] for column in sa.inspect(bind).get_columns(_TABLE)}


def upgrade() -> None:
    """Add the column and stamp existing runner bindings with their host's sandbox."""
    bind = op.get_bind()
    if not _has_column(bind):
        op.add_column(_TABLE, sa.Column(_COLUMN, sa.String(256), nullable=True))
    if bind.dialect.name == "cockroachdb":
        # CockroachDB makes a new column writable only after its schema change commits.
        bind.commit()
        bind.execute(sa.text("SET TRANSACTION ISOLATION LEVEL SERIALIZABLE"))
    current_sandbox = (
        sa.select(_HOSTS.c.sandbox_id)
        .where(
            _HOSTS.c.workspace_id == _METADATA.c.workspace_id,
            _HOSTS.c.host_id == _METADATA.c.host_id,
        )
        .scalar_subquery()
    )
    bind.execute(
        _METADATA.update()
        .where(_METADATA.c.runner_id.is_not(None), _METADATA.c.host_id.is_not(None))
        .values({_COLUMN: current_sandbox})
    )


def downgrade() -> None:
    """Drop the column; runner bindings themselves are untouched."""
    bind = op.get_bind()
    if _has_column(bind):
        with op.batch_alter_table(_TABLE) as batch_op:
            batch_op.drop_column(_COLUMN)
