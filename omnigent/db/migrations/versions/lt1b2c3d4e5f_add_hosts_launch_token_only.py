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

The backfill walks the primary key in bounded pages and commits each full page,
so a retried upgrade resumes safely. Deployment: additive, with a server
default so older application code keeps writing valid rows. Roll back by
downgrading this revision; upgrading again re-runs the backfill.
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

_TABLE = "hosts"
_COLUMN = "launch_token_only"
_BATCH_SIZE = 500
# Alembic loads revision files under generated module names; name the logger explicitly.
_logger = logging.getLogger("omnigent.db.migrations.hosts_launch_token_only")


def _commit(bind: sa.Connection) -> None:
    """Commit the work so far and continue in a new transaction."""
    if bind.dialect.name == "cockroachdb":
        bind.commit()
        bind.execute(sa.text("SET TRANSACTION ISOLATION LEVEL SERIALIZABLE"))
    else:
        with op.get_context().autocommit_block():
            pass


def upgrade() -> None:
    """Add the marker, mark armed rows and report the rows left unclassified."""
    bind = op.get_bind()
    # MySQL commits DDL, so a retried upgrade may find the column already added.
    if _COLUMN not in {column["name"] for column in sa.inspect(bind).get_columns(_TABLE)}:
        with op.batch_alter_table(_TABLE) as batch_op:
            batch_op.add_column(
                sa.Column(_COLUMN, sa.Boolean(), nullable=False, server_default=sa.false())
            )
    # CockroachDB publishes a new column only at commit.
    _commit(bind)
    hosts = sa.table(
        _TABLE,
        sa.column("workspace_id", sa.BigInteger()),
        sa.column("host_id"),
        sa.column("token_hash", sa.String(64)),
        sa.column("sandbox_provider", sa.String(32)),
        sa.column("deleted_at", sa.BigInteger()),
        sa.column(_COLUMN, sa.Boolean()),
    )
    marked = unclassified = 0
    cursor: tuple[int, bytes | str] | None = None
    while True:
        query = sa.select(
            hosts.c.workspace_id,
            hosts.c.host_id,
            hosts.c.token_hash.is_not(None).label("armed"),
            sa.and_(hosts.c.sandbox_provider.is_not(None), hosts.c.deleted_at.is_(None)).label(
                "live_sandbox"
            ),
            hosts.c[_COLUMN],
        ).order_by(hosts.c.workspace_id, hosts.c.host_id)
        if cursor is not None:
            if bind.dialect.name == "mysql":
                # MySQL treats tuple inequalities as filters over a full index scan.
                query = query.where(
                    sa.or_(
                        hosts.c.workspace_id > cursor[0],
                        sa.and_(hosts.c.workspace_id == cursor[0], hosts.c.host_id > cursor[1]),
                    )
                )
            else:
                query = query.where(sa.tuple_(hosts.c.workspace_id, hosts.c.host_id) > cursor)
        rows = bind.execute(query.limit(_BATCH_SIZE)).all()
        if not rows:
            break
        for workspace_id, host_id, armed, live_sandbox, already_marked in rows:
            if armed and not already_marked:
                bind.execute(
                    hosts.update()
                    .where(hosts.c.workspace_id == workspace_id, hosts.c.host_id == host_id)
                    .values({_COLUMN: True})
                )
                marked += 1
            elif not armed and live_sandbox and not already_marked:
                unclassified += 1
        last_id = rows[-1].host_id
        cursor = (rows[-1].workspace_id, last_id if isinstance(last_id, str) else bytes(last_id))
        if len(rows) == _BATCH_SIZE:
            _commit(bind)
    _logger.info("Marked %d armed managed hosts as launch-token-only", marked)
    if unclassified:
        _logger.warning(
            "%d sandbox hosts hold no launch token and could not be classified; they keep "
            "registering with their owner's bearer until the server next arms them",
            unclassified,
        )


def downgrade() -> None:
    """Remove the marker."""
    with op.batch_alter_table(_TABLE) as batch_op:
        batch_op.drop_column(_COLUMN)
