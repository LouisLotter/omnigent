"""Tests for the run-time MCP services migration (mr1a2b3c4d5e)."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import command

from omnigent.db.cockroachdb import _crdb_server_version, _prepare_crdb_schema_transaction
from omnigent.db.utils import _build_alembic_config, get_or_create_engine
from omnigent.stores.credential_store.sqlalchemy_store import CredentialStore
from omnigent.stores.mcp_service_store import (
    SERVICE_PRINCIPAL,
    McpServiceSettings,
    McpServiceStore,
    client_provider,
    member_provider,
    stamped,
)
from tests.server.test_credential_store import _FakeCipher

_PREVIOUS = "c0201a2b3c4d"
_REVISION = "mr1a2b3c4d5e"


def _migrate(engine: sa.Engine, revision: str, *, downgrade: bool = False) -> None:
    config = _build_alembic_config(engine.url.render_as_string(hide_password=False))
    with engine.connect() as connection:
        if engine.dialect.name == "cockroachdb":
            _prepare_crdb_schema_transaction(connection, _crdb_server_version(engine))
        config.attributes["connection"] = connection
        (command.downgrade if downgrade else command.upgrade)(config, revision)
        connection.commit()


def _tables(engine: sa.Engine) -> set[str]:
    return set(sa.inspect(engine).get_table_names())


def _connection_indexes(engine: sa.Engine) -> set[str]:
    return {index["name"] for index in sa.inspect(engine).get_indexes("connections")}


def test_upgrade_and_downgrade_keep_other_data_and_remove_service_logins(db_uri: str) -> None:
    engine = get_or_create_engine(db_uri)
    credentials = CredentialStore(db_uri, _FakeCipher())
    services = McpServiceStore(db_uri)
    # A login of a registry-file service and an unrelated integration survive both ways.
    credentials.upsert("alice", "mcp:shipped", secret={"access_token": "keep"}, metadata={})
    credentials.upsert("alice", "github", secret={"token": "keep"}, metadata={})
    record = services.create(
        "tracker",
        McpServiceSettings(
            title="Tracker", url="https://mcp.example.com/mcp", auth="bearer", timeout=60
        ),
        created_by="admin",
    )
    stamp = stamped({}, record.credential_version)
    credentials.upsert("alice", member_provider("tracker"), secret={"a": "1"}, metadata=stamp)
    credentials.upsert(
        SERVICE_PRINCIPAL, client_provider("tracker"), secret={"b": "2"}, metadata=stamp
    )

    _migrate(engine, _PREVIOUS, downgrade=True)

    assert not {"mcp_registry_services", "mcp_registry_service_tools"} & _tables(engine)
    assert "ix_connections_provider" not in _connection_indexes(engine)
    with engine.connect() as connection:
        providers = sorted(
            row.provider for row in connection.execute(sa.text("SELECT provider FROM connections"))
        )
    # Older code would not check a service version, so these logins went with the tables.
    assert providers == ["github", "mcp:shipped"]

    _migrate(engine, _REVISION)

    assert {"mcp_registry_services", "mcp_registry_service_tools"} <= _tables(engine)
    assert "ix_connections_provider" in _connection_indexes(engine)
    assert credentials.get("alice", "mcp:shipped") is not None
    assert credentials.get("alice", "github") is not None
    assert services.scan() == ([], [])


def test_upgrade_skips_what_a_retried_run_already_created(db_uri: str) -> None:
    """MySQL commits each DDL statement, so a retry may find part of the upgrade done."""
    engine = get_or_create_engine(db_uri)
    _migrate(engine, _PREVIOUS, downgrade=True)
    # A pooled SQLite connection keeps its parsed schema until its next read.
    engine.dispose()
    with engine.begin() as connection:
        connection.execute(
            sa.text(
                "CREATE INDEX ix_connections_provider ON connections "
                "(workspace_id, provider, user_id, account_id)"
            )
        )
    _migrate(engine, _REVISION)
    assert "mcp_registry_services" in _tables(engine)
