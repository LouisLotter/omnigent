"""Mark server-managed hosts that register only with their launch token.

Revision ID: lt1b2c3d4e5f
Revises: mm1a2b3c4d5e
Create Date: 2026-10-21 00:00:00.000000

Adds ``hosts.launch_token_only``. The host tunnel refuses an owner bearer for a
marked row, so only the host's own launch token can register it.

The backfill marks only rows it can classify: those holding a launch token,
which only server-managed arming writes. A row tagged with a sandbox provider
but holding no token is either a managed host whose token was already revoked
or detached, or a self-registering sandbox, which registers with its owner's
bearer by design; nothing in the row tells them apart. Such rows stay unmarked
and keep the bearer path until their next server-managed arming sets the
marker. The upgrade logs how many there are.

Deployment: additive, with a server default so older application code keeps
writing valid rows. Roll back by downgrading this revision; upgrading again
re-runs the backfill.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "lt1b2c3d4e5f"
down_revision: str | None = "mm1a2b3c4d5e"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_COLUMN = "launch_token_only"
_BATCH_SIZE = 500
_logger = logging.getLogger(__name__)


def upgrade() -> None:
    """Add the marker, mark armed rows and report the rows left unclassified."""
    bind = op.get_bind()
    # MySQL commits DDL, so a retried upgrade may find the column already added.
    if _COLUMN not in {column["name"] for column in sa.inspect(bind).get_columns("hosts")}:
        with op.batch_alter_table("hosts") as batch_op:
            batch_op.add_column(
                sa.Column(_COLUMN, sa.Boolean(), nullable=False, server_default=sa.false())
            )
    if bind.dialect.name == "cockroachdb":
        # CRDB publishes a new column only at commit.
        bind.commit()
        bind.execute(sa.text("SET TRANSACTION ISOLATION LEVEL SERIALIZABLE"))
    hosts = sa.table(
        "hosts",
        sa.column("workspace_id", sa.BigInteger()),
        sa.column("host_id"),
        sa.column("token_hash", sa.String(64)),
        sa.column("sandbox_provider", sa.String(32)),
        sa.column("deleted_at", sa.BigInteger()),
        sa.column(_COLUMN, sa.Boolean()),
    )
    marked = 0
    while True:
        armed = bind.execute(
            sa.select(hosts.c.workspace_id, hosts.c.host_id)
            .where(hosts.c.token_hash.is_not(None), hosts.c[_COLUMN] == sa.false())
            .limit(_BATCH_SIZE)
        ).all()
        if not armed:
            break
        for workspace_id, host_id in armed:
            bind.execute(
                hosts.update()
                .where(hosts.c.workspace_id == workspace_id, hosts.c.host_id == host_id)
                .values({_COLUMN: True})
            )
        marked += len(armed)
    unclassified = bind.execute(
        sa.select(sa.func.count())
        .select_from(hosts)
        .where(
            hosts.c.sandbox_provider.is_not(None),
            hosts.c.token_hash.is_(None),
            hosts.c.deleted_at.is_(None),
            hosts.c[_COLUMN] == sa.false(),
        )
    ).scalar_one()
    _logger.info("Marked %d armed managed hosts as launch-token-only", marked)
    if unclassified:
        _logger.warning(
            "%d sandbox hosts hold no launch token and could not be classified; they keep "
            "registering with their owner's bearer until the server next arms them",
            unclassified,
        )


def downgrade() -> None:
    """Remove the marker."""
    with op.batch_alter_table("hosts") as batch_op:
        batch_op.drop_column(_COLUMN)
