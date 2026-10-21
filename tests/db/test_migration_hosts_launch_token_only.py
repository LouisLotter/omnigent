"""Tests for the hosts launch-token-only migration (lt1b2c3d4e5f)."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from importlib import import_module
from pathlib import Path
from types import ModuleType

import pytest
import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations

from omnigent.db.utils import clear_engine_cache, get_or_create_engine, now_epoch
from omnigent.stores.host_store import HostRegistrationRefusedError, HostStore

_OWNER = "alice@example.com"
_ARMED = "a1a1a1a1a1a14a1a8a1a1a1a1a1a1a1a"
_CLEARED = "b2b2b2b2b2b24b2b8b2b2b2b2b2b2b2b"
_SELF_REGISTERING = "c3c3c3c3c3c34c3c8c3c3c3c3c3c3c3c"
_LAPTOP = "d4d4d4d4d4d44d4d8d4d4d4d4d4d4d4d"
_DELETED = "e5e5e5e5e5e54e5e8e5e5e5e5e5e5e5e"


@pytest.fixture
def migration() -> ModuleType:
    return import_module(
        "omnigent.db.migrations.versions.lt1b2c3d4e5f_add_hosts_launch_token_only"
    )


@pytest.fixture
def db_uri(tmp_path: Path) -> Iterator[str]:
    """A SQLite database at head; each test moves it back across the migration."""
    uri = f"sqlite:///{tmp_path / 'hosts.db'}"
    get_or_create_engine(uri)
    try:
        yield uri
    finally:
        clear_engine_cache()


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


def _seed_hosts(store: HostStore) -> None:
    """One host of each kind the backfill must tell apart."""
    _register(store, _ARMED)
    _register(store, _CLEARED)
    store.revoke_launch_token(_CLEARED)
    store.reserve_sandbox_host(
        host_id=_SELF_REGISTERING, name="self-registering", user_id=_OWNER, provider="modal"
    )
    store.upsert_on_connect(_LAPTOP, "alices-laptop", _OWNER)
    _register(store, _DELETED)
    assert store.delete_host(_DELETED) is not None


def _migrate(uri: str, migration: ModuleType, *steps: str) -> None:
    """Run *steps* (``"downgrade"``, ``"upgrade"`` or raw SQL) on one connection.

    One connection, because SQLite checks ``ADD COLUMN`` against a connection's
    cached schema, which another connection's DDL can leave stale.
    """
    with get_or_create_engine(uri).begin() as connection:
        with Operations.context(MigrationContext.configure(connection)):
            for step in steps:
                if step in {"downgrade", "upgrade"}:
                    getattr(migration, step)()
                else:
                    connection.execute(sa.text(step))


def _marked(uri: str) -> dict[str, bool]:
    with get_or_create_engine(uri).connect() as connection:
        rows = connection.execute(sa.text("SELECT host_id, launch_token_only FROM hosts")).all()
    return {bytes(host_id).hex(): bool(marked) for host_id, marked in rows}


def _columns(uri: str) -> set[str]:
    return {
        column["name"] for column in sa.inspect(get_or_create_engine(uri)).get_columns("hosts")
    }


def test_upgrade_marks_only_armed_hosts_and_reports_the_rest(
    db_uri: str, migration: ModuleType, caplog: pytest.LogCaptureFixture
) -> None:
    _seed_hosts(HostStore(db_uri))
    _migrate(db_uri, migration, "downgrade")
    assert "launch_token_only" not in _columns(db_uri)

    with caplog.at_level(logging.INFO, logger=migration.__name__):
        _migrate(db_uri, migration, "upgrade")

    assert _marked(db_uri) == {
        _ARMED: True,
        _CLEARED: False,
        _SELF_REGISTERING: False,
        _LAPTOP: False,
        _DELETED: False,
    }
    assert "Marked 1 armed managed hosts" in caplog.text
    # The cleared managed host and the self-registering sandbox look the same.
    assert "2 sandbox hosts hold no launch token" in caplog.text


def test_upgrade_resumes_after_the_column_was_committed(
    db_uri: str, migration: ModuleType
) -> None:
    """MySQL commits DDL, so a retried upgrade can find the column already added."""
    _register(HostStore(db_uri), _ARMED)
    _migrate(
        db_uri,
        migration,
        "downgrade",
        "ALTER TABLE hosts ADD COLUMN launch_token_only BOOLEAN NOT NULL DEFAULT 0",
        "upgrade",
    )

    assert _marked(db_uri) == {_ARMED: True}


@pytest.mark.parametrize("arming", ["replace", "rearm"])
def test_arming_a_migrated_legacy_host_closes_the_bearer_path(
    db_uri: str, migration: ModuleType, arming: str
) -> None:
    """A legacy managed host keeps the bearer path only until the server arms it again."""
    store = HostStore(db_uri)
    _register(store, _CLEARED, token="old-token")
    store.revoke_launch_token(_CLEARED)
    _migrate(db_uri, migration, "downgrade", "upgrade")
    name = f"managed-{_CLEARED[:8]}"
    assert _marked(db_uri) == {_CLEARED: False}
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
