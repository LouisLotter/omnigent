"""The runner host incarnation migration (ri1a2b3c4d5e).

Nothing records which sandbox a pre-upgrade runner was launched into, so the
upgrade retires every runner binding on a sandbox-backed host: its old binding
token can neither mint nor authenticate, and the thread binds a new runner on
the current sandbox. Personal hosts (no sandbox) keep their bindings.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass

import pytest
import sqlalchemy as sa
from alembic import command

from omnigent.db.cockroachdb import _crdb_server_version, _prepare_crdb_schema_transaction
from omnigent.db.utils import (
    _build_alembic_config,
    _get_current_db_heads,
    _initialize_or_verify_schema,
    get_or_create_engine,
)
from omnigent.runner.identity import token_bound_runner_id
from omnigent.server.accounts_store import SqlAlchemyAccountStore
from omnigent.server.auth import LEVEL_OWNER
from omnigent.stores.conversation_store.sqlalchemy_store import SqlAlchemyConversationStore
from omnigent.stores.host_store import RETIRED_RUNNER_INCARNATION, HostStore
from omnigent.stores.permission_store.sqlalchemy_store import SqlAlchemyPermissionStore

_PREVIOUS = "tm1a2b3c4d5e"
_REVISION = "ri1a2b3c4d5e"
_TABLE = "omnigent_conversation_metadata"
_COLUMN = "runner_host_incarnation"


@dataclass
class _Installation:
    """Alice's threads: one on a managed host's sandbox, one on her laptop."""

    managed_host: str
    managed_thread: str
    managed_runner: str
    personal_thread: str
    personal_runner: str


def _migrate(engine: sa.Engine, revision: str, *, downgrade: bool = False) -> None:
    config = _build_alembic_config(engine.url.render_as_string(hide_password=False))
    with engine.connect() as connection:
        if engine.dialect.name == "cockroachdb":
            _prepare_crdb_schema_transaction(connection, _crdb_server_version(engine))
        config.attributes["connection"] = connection
        (command.downgrade if downgrade else command.upgrade)(config, revision)
        connection.commit()
    # As after a restart: SQLite connections pooled before a table rebuild keep its old schema.
    engine.dispose()


def _has_column(engine: sa.Engine) -> bool:
    return _COLUMN in {column["name"] for column in sa.inspect(engine).get_columns(_TABLE)}


@pytest.fixture
def engine(db_uri: str):
    engine = get_or_create_engine(db_uri)
    try:
        yield engine
    finally:
        _initialize_or_verify_schema(engine, db_uri)


def _issues(db_uri: str, runner_id: str) -> bool:
    """Whether the runner's binding token can mint a token and authenticate."""
    minted = SqlAlchemyAccountStore(db_uri).with_runner_authority(runner_id, lambda b: b)
    authenticated = HostStore(db_uri).runner_binding(runner_id)
    assert (minted is None) == (authenticated is None)
    return minted is not None


def _seed(db_uri: str) -> _Installation:
    SqlAlchemyAccountStore(db_uri).create_user_with_password("alice", "password-hash")
    hosts = HostStore(db_uri)
    conversations = SqlAlchemyConversationStore(db_uri)
    permissions = SqlAlchemyPermissionStore(db_uri)
    managed_host, laptop = uuid.uuid4().hex, uuid.uuid4().hex
    hosts.upsert_on_connect(managed_host, "cloud-computer", "alice")
    hosts.upsert_on_connect(laptop, "laptop", "alice")
    assert hosts.bind_sandbox(managed_host, provider="fake", sandbox_id="sandbox-1") is not None

    def bound_thread(host: str) -> tuple[str, str]:
        thread = conversations.create_conversation().id
        permissions.grant("alice", thread, LEVEL_OWNER)
        conversations.set_host_id(thread, host, workspace="/tmp/workspace")
        runner = token_bound_runner_id(f"binding-{uuid.uuid4().hex}")
        conversations.replace_runner_id(thread, runner)
        return thread, runner

    managed_thread, managed_runner = bound_thread(managed_host)
    personal_thread, personal_runner = bound_thread(laptop)
    # At the current revision both runners work.
    assert _issues(db_uri, managed_runner) and _issues(db_uri, personal_runner)
    return _Installation(
        managed_host, managed_thread, managed_runner, personal_thread, personal_runner
    )


def _rebind(db_uri: str, thread: str) -> str:
    """A fresh launch binds a new runner on the host's current sandbox."""
    runner = token_bound_runner_id(f"binding-{uuid.uuid4().hex}")
    SqlAlchemyConversationStore(db_uri).replace_runner_id(thread, runner)
    return runner


def _replace_sandbox(db_uri: str, host: str) -> None:
    replaced = HostStore(db_uri).replace_managed_host_sandbox(
        host_id=host,
        user_id="alice",
        token=uuid.uuid4().hex,
        provider="fake",
        sandbox_id="sandbox-2",
        token_expires_at=int(time.time()) + 3600,
    )
    assert replaced is not None and replaced.host_id == host


def _detach_sandbox(db_uri: str, host: str) -> None:
    hosts = HostStore(db_uri)
    current = hosts.get_host(host)
    assert current is not None
    assert hosts.detach_stale_managed_sandbox(
        host, sandbox_id="sandbox-1", expected_updated_at=current.updated_at
    )


def test_the_migration_writes_the_runtime_retirement_marker() -> None:
    from importlib import import_module

    migration = import_module(
        "omnigent.db.migrations.versions.ri1a2b3c4d5e_add_runner_host_incarnation"
    )
    assert migration._RETIRED == RETIRED_RUNNER_INCARNATION


def test_upgrade_retires_a_binding_from_a_sandbox_replaced_before_it(db_uri, engine) -> None:
    seeded = _seed(db_uri)
    _migrate(engine, _PREVIOUS, downgrade=True)
    _replace_sandbox(db_uri, seeded.managed_host)

    _migrate(engine, _REVISION)

    assert not _issues(db_uri, seeded.managed_runner)
    assert _issues(db_uri, _rebind(db_uri, seeded.managed_thread))


def test_upgrade_retires_a_binding_from_a_sandbox_detached_before_it(db_uri, engine) -> None:
    seeded = _seed(db_uri)
    _migrate(engine, _PREVIOUS, downgrade=True)
    _detach_sandbox(db_uri, seeded.managed_host)

    _migrate(engine, _REVISION)

    assert not _issues(db_uri, seeded.managed_runner)
    # No sandbox runs: a runner can't be issued on the host until one does.
    assert not _issues(db_uri, _rebind(db_uri, seeded.managed_thread))


def test_upgrade_requires_a_rebind_on_an_unchanged_managed_host(db_uri, engine) -> None:
    seeded = _seed(db_uri)
    _migrate(engine, _PREVIOUS, downgrade=True)

    _migrate(engine, _REVISION)

    assert _get_current_db_heads(engine) == (_REVISION,)
    assert not _issues(db_uri, seeded.managed_runner)
    assert _issues(db_uri, _rebind(db_uri, seeded.managed_thread))


def test_upgrade_keeps_a_personal_hosts_runner(db_uri, engine) -> None:
    seeded = _seed(db_uri)
    _migrate(engine, _PREVIOUS, downgrade=True)

    _migrate(engine, _REVISION)

    assert _issues(db_uri, seeded.personal_runner)


def test_upgrade_resumes_after_the_column_was_committed(db_uri, engine) -> None:
    seeded = _seed(db_uri)
    _migrate(engine, _PREVIOUS, downgrade=True)
    _replace_sandbox(db_uri, seeded.managed_host)
    assert not _has_column(engine)
    # MySQL and CockroachDB commit the column on their own; the backfill then failed.
    with engine.begin() as connection:
        connection.execute(sa.text(f"ALTER TABLE {_TABLE} ADD COLUMN {_COLUMN} VARCHAR(256)"))
    engine.dispose()

    _migrate(engine, _REVISION)

    assert _get_current_db_heads(engine) == (_REVISION,)
    assert not _issues(db_uri, seeded.managed_runner)
    assert _issues(db_uri, seeded.personal_runner)
    assert _issues(db_uri, _rebind(db_uri, seeded.managed_thread))


def test_downgrade_drops_only_the_column(db_uri, engine) -> None:
    seeded = _seed(db_uri)
    with engine.connect() as connection:
        before = connection.execute(sa.text(f"SELECT COUNT(*) FROM {_TABLE}")).scalar_one()

    _migrate(engine, _PREVIOUS, downgrade=True)
    _migrate(engine, _PREVIOUS, downgrade=True)

    assert not _has_column(engine)
    with engine.connect() as connection:
        assert connection.execute(sa.text(f"SELECT COUNT(*) FROM {_TABLE}")).scalar_one() == before
        bound = connection.execute(
            sa.text(f"SELECT COUNT(*) FROM {_TABLE} WHERE runner_id IS NOT NULL")
        ).scalar_one()
    assert bound == 2
    _migrate(engine, _REVISION)
    assert not _issues(db_uri, seeded.managed_runner)
    assert _issues(db_uri, seeded.personal_runner)
