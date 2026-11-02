"""The runner host incarnation migration (ri1a2b3c4d5e): additive, backfilled, resumable."""

from __future__ import annotations

import uuid

import pytest
import sqlalchemy as sa
from alembic import command

from omnigent.db.cockroachdb import _crdb_server_version, _prepare_crdb_schema_transaction
from omnigent.db.db_models import SqlConversationMetadata
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
from omnigent.stores.host_store import HostStore
from omnigent.stores.permission_store.sqlalchemy_store import SqlAlchemyPermissionStore

_PREVIOUS = "tm1a2b3c4d5e"
_REVISION = "ri1a2b3c4d5e"
_TABLE = "omnigent_conversation_metadata"
_COLUMN = "runner_host_incarnation"


def _migrate(engine: sa.Engine, revision: str, *, downgrade: bool = False) -> None:
    config = _build_alembic_config(engine.url.render_as_string(hide_password=False))
    with engine.connect() as connection:
        if engine.dialect.name == "cockroachdb":
            _prepare_crdb_schema_transaction(connection, _crdb_server_version(engine))
        config.attributes["connection"] = connection
        (command.downgrade if downgrade else command.upgrade)(config, revision)
        connection.commit()


def _has_column(engine: sa.Engine) -> bool:
    return _COLUMN in {column["name"] for column in sa.inspect(engine).get_columns(_TABLE)}


def _incarnations(engine: sa.Engine, threads: dict[str, str]) -> dict[str, str | None]:
    with engine.connect() as connection:
        return {
            name: connection.execute(
                sa.select(SqlConversationMetadata.runner_host_incarnation).where(
                    SqlConversationMetadata.id == thread
                )
            ).scalar_one()
            for name, thread in threads.items()
        }


@pytest.fixture
def engine(db_uri: str):
    engine = get_or_create_engine(db_uri)
    try:
        yield engine
    finally:
        _initialize_or_verify_schema(engine, db_uri)


def _seed(db_uri: str) -> tuple[dict[str, str], dict[str, str]]:
    """Threads bound to a managed host's sandbox, a personal host, and one without a runner."""
    SqlAlchemyAccountStore(db_uri).create_user_with_password("alice", "password-hash")
    hosts = HostStore(db_uri)
    conversations = SqlAlchemyConversationStore(db_uri)
    permissions = SqlAlchemyPermissionStore(db_uri)
    managed, personal = uuid.uuid4().hex, uuid.uuid4().hex
    hosts.upsert_on_connect(managed, "cloud-computer", "alice")
    hosts.upsert_on_connect(personal, "laptop", "alice")
    assert hosts.bind_sandbox(managed, provider="fake", sandbox_id="sandbox-1") is not None
    threads: dict[str, str] = {}
    runners: dict[str, str] = {}
    for name, host in (("managed", managed), ("personal", personal), ("idle", managed)):
        thread = conversations.create_conversation()
        permissions.grant("alice", thread.id, LEVEL_OWNER)
        conversations.set_host_id(thread.id, host, workspace="/tmp/workspace")
        threads[name] = thread.id
        if name != "idle":
            runners[name] = token_bound_runner_id(f"binding-{name}-{uuid.uuid4().hex}")
            conversations.replace_runner_id(thread.id, runners[name])
    return threads, runners


def test_upgrade_stamps_existing_bindings_so_their_runners_keep_working(db_uri, engine) -> None:
    threads, runners = _seed(db_uri)
    _migrate(engine, _PREVIOUS, downgrade=True)
    assert not _has_column(engine)

    _migrate(engine, _REVISION)

    assert _get_current_db_heads(engine) == (_REVISION,)
    assert _incarnations(engine, threads) == {
        "managed": "sandbox-1",
        "personal": None,
        "idle": None,
    }
    hosts = HostStore(db_uri)
    for runner in runners.values():
        assert hosts.runner_binding(runner) is not None


def test_upgrade_resumes_after_the_column_was_committed(db_uri, engine) -> None:
    threads, runners = _seed(db_uri)
    _migrate(engine, _PREVIOUS, downgrade=True)
    assert not _has_column(engine)
    # A new process: SQLite connections pooled before the table rebuild keep its old schema.
    engine.dispose()
    # MySQL and CockroachDB commit the column on their own; the backfill then failed.
    with engine.begin() as connection:
        connection.execute(sa.text(f"ALTER TABLE {_TABLE} ADD COLUMN {_COLUMN} VARCHAR(256)"))

    _migrate(engine, _REVISION)

    assert _get_current_db_heads(engine) == (_REVISION,)
    assert _incarnations(engine, threads)["managed"] == "sandbox-1"
    assert HostStore(db_uri).runner_binding(runners["managed"]) is not None


def test_downgrade_drops_only_the_column(db_uri, engine) -> None:
    threads, _runners = _seed(db_uri)
    with engine.connect() as connection:
        before = connection.execute(sa.text(f"SELECT COUNT(*) FROM {_TABLE}")).scalar_one()

    _migrate(engine, _PREVIOUS, downgrade=True)
    _migrate(engine, _PREVIOUS, downgrade=True)

    assert not _has_column(engine)
    with engine.connect() as connection:
        assert connection.execute(sa.text(f"SELECT COUNT(*) FROM {_TABLE}")).scalar_one() == before
    _migrate(engine, _REVISION)
    assert _incarnations(engine, threads)["managed"] == "sandbox-1"
