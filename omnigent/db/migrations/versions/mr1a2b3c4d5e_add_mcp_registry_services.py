"""Add MCP services an administrator manages at run time.

Revision ID: mr1a2b3c4d5e
Revises: c0201a2b3c4d
Create Date: 2026-11-02 00:00:00.000000

Adds ``mcp_registry_services`` (one row per service added in Settings, beside the
registry file's services) and ``mcp_registry_service_tools`` (its tool allowlist,
one row per tool). Neither holds a secret: logins stay in ``connections``, sealed by
the credential cipher. Adds ``ix_connections_provider`` so a service's logins can be
removed in bounded batches when an administrator changes or removes the service.

Deployment: additive. The new tables have no readers in older code, and the index
only adds a lookup path, so older application code keeps working against the new
schema. A retried upgrade skips what already exists (MySQL commits each DDL
statement). The downgrade first removes the logins saved for these services, in
bounded batches that commit at full pages: older code does not check their service
version, and would otherwise send them to a registry-file service that later takes
the same ID. It then drops the index and both tables.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from omnigent.db.db_models import Uuid16

revision: str = "mr1a2b3c4d5e"
down_revision: str | None = "c0201a2b3c4d"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_SERVICES = "mcp_registry_services"
_TOOLS = "mcp_registry_service_tools"
_CONNECTIONS = "connections"
_PROVIDER_INDEX = "ix_connections_provider"
_BATCH_SIZE = 500
# The ``connections.provider`` prefixes of a run-time service's logins: members'
# logins, an OAuth sign-in in progress, and the OAuth client's own secret.
_PROVIDER_PREFIXES = ("mcp:", "mcp-pending:", "mcp-client:")


def _commit(bind: sa.Connection) -> None:
    """Commit the work so far and continue in a new transaction."""
    if bind.dialect.name == "cockroachdb":
        bind.commit()
        bind.execute(sa.text("SET TRANSACTION ISOLATION LEVEL SERIALIZABLE"))
    else:
        with op.get_context().autocommit_block():
            pass


def upgrade() -> None:
    """Create both tables and the provider index, skipping what already exists."""
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if not inspector.has_table(_SERVICES):
        op.create_table(
            _SERVICES,
            sa.Column("workspace_id", sa.BigInteger(), nullable=False),
            sa.Column("service_id", sa.String(40), nullable=False),
            sa.Column("title", sa.String(120), nullable=False),
            sa.Column("description", sa.String(1000), nullable=True),
            sa.Column("url", sa.String(2048), nullable=False),
            # Login type code (enum_codecs.MCP_SERVICE_AUTH: none=1, bearer=2, oauth=3).
            sa.Column("auth", sa.SmallInteger(), nullable=False),
            sa.Column("timeout_seconds", sa.SmallInteger(), nullable=False),
            # Random; replaced whenever the destination or login settings change.
            sa.Column("credential_version", Uuid16(), nullable=False),
            sa.Column("oauth_authorize_url", sa.String(2048), nullable=True),
            sa.Column("oauth_token_url", sa.String(2048), nullable=True),
            sa.Column("oauth_client_id", sa.String(256), nullable=True),
            sa.Column("oauth_scopes", sa.String(1024), nullable=True),
            sa.Column("oauth_resource", sa.String(2048), nullable=True),
            # enum_codecs.MCP_OAUTH_CLIENT_REGISTRATION: manual=1, dynamic=2.
            sa.Column("oauth_client_registration", sa.SmallInteger(), nullable=True),
            # enum_codecs.MCP_OAUTH_TOKEN_ENDPOINT_AUTH: none=1, post=2, basic=3.
            sa.Column("oauth_token_endpoint_auth", sa.SmallInteger(), nullable=True),
            sa.Column("pending_sweep", sa.Boolean(), nullable=False),
            sa.Column("created_by", sa.String(128), nullable=True),
            sa.Column("created_at", sa.BigInteger(), nullable=False),
            sa.Column("updated_at", sa.BigInteger(), nullable=False),
            sa.Column("deleted_at", sa.BigInteger(), nullable=True),
            sa.CheckConstraint("auth IN (1, 2, 3)", name="ck_mcp_registry_services_auth"),
            sa.CheckConstraint(
                "timeout_seconds BETWEEN 1 AND 300", name="ck_mcp_registry_services_timeout"
            ),
            sa.CheckConstraint(
                "oauth_client_registration IS NULL OR oauth_client_registration IN (1, 2)",
                name="ck_mcp_registry_services_oauth_registration",
            ),
            sa.CheckConstraint(
                "oauth_token_endpoint_auth IS NULL OR oauth_token_endpoint_auth IN (1, 2, 3)",
                name="ck_mcp_registry_services_oauth_token_auth",
            ),
            sa.PrimaryKeyConstraint("workspace_id", "service_id"),
        )
    if not inspector.has_table(_TOOLS):
        op.create_table(
            _TOOLS,
            sa.Column("workspace_id", sa.BigInteger(), nullable=False),
            sa.Column("service_id", sa.String(40), nullable=False),
            sa.Column("tool_name", sa.String(128), nullable=False),
            sa.PrimaryKeyConstraint("workspace_id", "service_id", "tool_name"),
        )
    if _PROVIDER_INDEX not in {index["name"] for index in inspector.get_indexes(_CONNECTIONS)}:
        op.create_index(
            _PROVIDER_INDEX,
            _CONNECTIONS,
            ["workspace_id", "provider", "user_id", "account_id"],
            unique=False,
        )


def _remove_service_logins(bind: sa.Connection) -> None:
    """Delete every login saved for a run-time service, in bounded batches."""
    services = sa.table(
        _SERVICES,
        sa.column("workspace_id", sa.BigInteger()),
        sa.column("service_id", sa.String(40)),
    )
    connections = sa.table(
        _CONNECTIONS,
        sa.column("workspace_id", sa.BigInteger()),
        sa.column("user_id", sa.String(128)),
        sa.column("provider", sa.String(64)),
        sa.column("account_id", sa.String(128)),
    )
    cursor: tuple[int, str] | None = None
    while True:
        query = sa.select(services.c.workspace_id, services.c.service_id).order_by(
            services.c.workspace_id, services.c.service_id
        )
        if cursor is not None:
            query = query.where(
                sa.or_(
                    services.c.workspace_id > cursor[0],
                    sa.and_(
                        services.c.workspace_id == cursor[0],
                        services.c.service_id > cursor[1],
                    ),
                )
            )
        page = bind.execute(query.limit(_BATCH_SIZE)).all()
        if not page:
            return
        for workspace_id, service_id in page:
            providers = [prefix + service_id for prefix in _PROVIDER_PREFIXES]
            while True:
                keys = bind.execute(
                    sa.select(
                        connections.c.user_id, connections.c.provider, connections.c.account_id
                    )
                    .where(
                        connections.c.workspace_id == workspace_id,
                        connections.c.provider.in_(providers),
                    )
                    .limit(_BATCH_SIZE)
                ).all()
                for user_id, provider, account_id in keys:
                    bind.execute(
                        connections.delete().where(
                            connections.c.workspace_id == workspace_id,
                            connections.c.user_id == user_id,
                            connections.c.provider == provider,
                            connections.c.account_id == account_id,
                        )
                    )
                if len(keys) < _BATCH_SIZE:
                    break
                _commit(bind)
        cursor = (page[-1].workspace_id, page[-1].service_id)


def downgrade() -> None:
    """Remove the run-time services' logins, then the index and both tables."""
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if inspector.has_table(_SERVICES):
        _remove_service_logins(bind)
    if _PROVIDER_INDEX in {index["name"] for index in inspector.get_indexes(_CONNECTIONS)}:
        op.drop_index(_PROVIDER_INDEX, table_name=_CONNECTIONS)
    if inspector.has_table(_TOOLS):
        op.drop_table(_TOOLS)
    if inspector.has_table(_SERVICES):
        op.drop_table(_SERVICES)
