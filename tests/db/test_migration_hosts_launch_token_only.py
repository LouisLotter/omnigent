"""Tests for the hosts launch-token-only migration (lt1b2c3d4e5f)."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from importlib import import_module

import pytest
import sqlalchemy as sa
from alembic import command

from omnigent.db.cockroachdb import _crdb_server_version, _prepare_crdb_schema_transaction
from omnigent.db.utils import _build_alembic_config, get_or_create_engine, now_epoch
from omnigent.stores.host_store import HostRegistrationRefusedError, HostStore

_PREVIOUS = "mm1a2b3c4d5e"
_REVISION = "lt1b2c3d4e5f"
_MIGRATION = "omnigent.db.migrations.versions.lt1b2c3d4e5f_add_hosts_launch_token_only"
_LOGGER = "omnigent.db.migrations.hosts_launch_token_only"
_OWNER = "alice@example.com"
_ARMED = "a1a1a1a1a1a14a1a8a1a1a1a1a1a1a1a"
_CLEARED = "b2b2b2b2b2b24b2b8b2b2b2b2b2b2b2b"
_SELF_REGISTERING = "c3c3c3c3c3c34c3c8c3c3c3c3c3c3c3c"
_LAPTOP = "d4d4d4d4d4d44d4d8d4d4d4d4d4d4d4d"
_DELETED = "e5e5e5e5e5e54e5e8e5e5e5e5e5e5e5e"


def _migrate(engine: sa.Engine, revision: str, *, downgrade: bool = False) -> None:
    config = _build_alembic_config(engine.url.render_as_string(hide_password=False))
    with engine.connect() as connection:
        if engine.dialect.name == "cockroachdb":
            _prepare_crdb_schema_transaction(connection, _crdb_server_version(engine))
        config.attributes["connection"] = connection
        (command.downgrade if downgrade else command.upgrade)(config, revision)
        connection.commit()


@contextmanager
def _migration_log(caplog: pytest.LogCaptureFixture) -> Iterator[None]:
    """Capture the migration's report; Alembic's logging setup replaces root handlers."""
    logger = logging.getLogger(_LOGGER)
    level = logger.level
    logger.addHandler(caplog.handler)
    logger.setLevel(logging.INFO)
    try:
        yield
    finally:
        logger.removeHandler(caplog.handler)
        logger.setLevel(level)


def _register(store: HostStore, host_id: str, *, token: str = "launch-token") -> None:
    store.register_managed_host(
        host_id=host_id,
        name=f"managed-{host_id[:8]}",
        user_id=_OWNER,
        token=token,
        provider="modal",
        sandbox_id=f"sb-{host_id[:8]}",
        token_expires_at=now_epoch() + 3600,
    )


def _marked(engine: sa.Engine) -> dict[str, bool]:
    with engine.connect() as connection:
        rows = connection.execute(sa.text("SELECT host_id, launch_token_only FROM hosts")).all()
    return {bytes(host_id).hex(): bool(marked) for host_id, marked in rows}


def _columns(engine: sa.Engine) -> set[str]:
    return {column["name"] for column in sa.inspect(engine).get_columns("hosts")}


def test_upgrade_marks_only_armed_hosts_and_reports_the_rest(
    db_uri: str, caplog: pytest.LogCaptureFixture
) -> None:
    engine = get_or_create_engine(db_uri)
    store = HostStore(db_uri)
    _register(store, _ARMED)
    _register(store, _CLEARED)
    store.revoke_launch_token(_CLEARED)
    store.reserve_sandbox_host(
        host_id=_SELF_REGISTERING, name="self-registering", user_id=_OWNER, provider="modal"
    )
    store.upsert_on_connect(_LAPTOP, "alices-laptop", _OWNER)
    _register(store, _DELETED)
    assert store.delete_host(_DELETED) is not None
    _migrate(engine, _PREVIOUS, downgrade=True)
    assert "launch_token_only" not in _columns(engine)

    with _migration_log(caplog):
        _migrate(engine, _REVISION)

    assert _marked(engine) == {
        _ARMED: True,
        _CLEARED: False,
        _SELF_REGISTERING: False,
        _LAPTOP: False,
        _DELETED: False,
    }
    assert "Marked 1 armed managed hosts" in caplog.text
    # The cleared managed host and the self-registering sandbox look the same.
    assert "2 sandbox hosts hold no launch token" in caplog.text


def test_upgrade_below_one_page_runs_inside_a_caller_transaction(db_uri: str) -> None:
    """Below one page the upgrade never commits, so it runs in a transaction the caller began."""
    engine = get_or_create_engine(db_uri)
    if engine.dialect.name == "cockroachdb":
        pytest.skip("CockroachDB publishes the new column only at commit")
    _register(HostStore(db_uri), _ARMED)
    _migrate(engine, _PREVIOUS, downgrade=True)
    config = _build_alembic_config(engine.url.render_as_string(hide_password=False))

    with engine.begin() as connection:
        config.attributes["connection"] = connection
        command.upgrade(config, _REVISION)

    assert _marked(engine) == {_ARMED: True}


def test_backfill_commits_bounded_pages_and_resumes_after_an_interruption(db_uri: str) -> None:
    batch_size = import_module(_MIGRATION)._BATCH_SIZE
    engine = get_or_create_engine(db_uri)
    _migrate(engine, _PREVIOUS, downgrade=True)
    hosts = sa.table(
        "hosts",
        sa.column("workspace_id", sa.BigInteger()),
        sa.column("host_id", sa.LargeBinary()),
        sa.column("user_id", sa.String()),
        sa.column("name", sa.String()),
        sa.column("status", sa.SmallInteger()),
        sa.column("created_at", sa.Integer()),
        sa.column("updated_at", sa.Integer()),
        sa.column("token_hash", sa.String()),
        sa.column("sandbox_provider", sa.String()),
    )
    count = 2 * batch_size + 1
    with engine.begin() as connection:
        connection.execute(
            hosts.insert(),
            [
                {
                    "workspace_id": 0,
                    "host_id": index.to_bytes(16, "big"),
                    "user_id": _OWNER,
                    "name": f"managed-{index}",
                    "status": 2,
                    "created_at": 1,
                    "updated_at": 1,
                    "token_hash": "0" * 64,
                    "sandbox_provider": "modal",
                }
                for index in range(1, count + 1)
            ],
        )
    updates = 0

    def fail_second_page(conn, cursor, statement, parameters, context, executemany):
        nonlocal updates
        if statement.startswith("UPDATE hosts SET launch_token_only"):
            updates += 1
            if updates == batch_size + 1:
                raise RuntimeError("interrupted second page")

    sa.event.listen(engine, "before_cursor_execute", fail_second_page)
    try:
        with pytest.raises(RuntimeError, match="interrupted second page"):
            _migrate(engine, _REVISION)
    finally:
        sa.event.remove(engine, "before_cursor_execute", fail_second_page)
    # The column and the first page were committed before the interruption.
    assert sum(_marked(engine).values()) == batch_size

    _migrate(engine, _REVISION)

    assert _marked(engine) == {f"{index:032x}": True for index in range(1, count + 1)}


@pytest.mark.parametrize("arming", ["replace", "rearm"])
def test_arming_a_migrated_legacy_host_closes_the_bearer_path(db_uri: str, arming: str) -> None:
    """A legacy managed host keeps the bearer path only until the server arms it again."""
    engine = get_or_create_engine(db_uri)
    store = HostStore(db_uri)
    _register(store, _CLEARED, token="old-token")
    store.revoke_launch_token(_CLEARED)
    _migrate(engine, _PREVIOUS, downgrade=True)
    _migrate(engine, _REVISION)
    name = f"managed-{_CLEARED[:8]}"
    assert _marked(engine) == {_CLEARED: False}
    # The documented residual: unclassified, it still registers with the bearer.
    store.upsert_on_connect(_CLEARED, name, _OWNER)
    store.set_offline(_CLEARED)
    snapshot = store.get_host(_CLEARED)
    assert snapshot is not None

    if arming == "replace":
        armed = store.replace_managed_host_sandbox(
            host_id=_CLEARED,
            user_id=_OWNER,
            token="new-token",
            provider="modal",
            sandbox_id="sb-next",
            token_expires_at=now_epoch() + 3600,
        )
    else:
        armed = store.rearm_managed_host(
            _CLEARED,
            sandbox_id=f"sb-{_CLEARED[:8]}",
            expected_updated_at=snapshot.updated_at,
            token="new-token",
            token_expires_at=now_epoch() + 3600,
        )

    assert armed is not None and armed.launch_token_only
    with pytest.raises(HostRegistrationRefusedError):
        store.upsert_on_connect(_CLEARED, name, _OWNER)
    with pytest.raises(HostRegistrationRefusedError):
        store.upsert_on_connect(_LAPTOP, name, _OWNER)
    with pytest.raises(HostRegistrationRefusedError):
        store.upsert_on_connect(_CLEARED, "elsewhere", "local", allow_host_id_reown=True)
    assert store.resolve_launch_token(_CLEARED, "old-token") is None
    connected = store.upsert_on_connect(_CLEARED, name, _OWNER, managed_token="new-token")
    assert connected.status == "online"
