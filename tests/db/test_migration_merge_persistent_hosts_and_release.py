"""Upgrade either deployed schema and a restored fork snapshot to one head."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic import command
from alembic.script import ScriptDirectory

from deploy.docker.entrypoint import run_migrations
from omnigent.db.utils import (
    _build_alembic_config,
    _get_current_db_heads,
    _run_migrations,
)

_FORK = "lt1b2c3d4e5f"
_RELEASE = "b2e4f6a8c013"
_MERGE = "c0201a2b3c4d"
_ID = bytes.fromhex("0123456789abcdef0123456789abcdef")


def _migrate(engine: sa.Engine, revision: str, *, downgrade: bool = False) -> None:
    config = _build_alembic_config(str(engine.url))
    with engine.begin() as connection:
        config.attributes["connection"] = connection
        (command.downgrade if downgrade else command.upgrade)(config, revision)


def _insert_canaries(engine: sa.Engine, starting_revision: str) -> None:
    """Store one host, history, schedule, run and connection at *starting_revision*."""
    metadata = sa.MetaData()
    hosts = sa.Table("hosts", metadata, autoload_with=engine)
    conversations = sa.Table("conversations", metadata, autoload_with=engine)
    session_metadata = sa.Table("omnigent_conversation_metadata", metadata, autoload_with=engine)
    tasks = sa.Table("scheduled_tasks", metadata, autoload_with=engine)
    runs = sa.Table("scheduled_task_runs", metadata, autoload_with=engine)
    connections = sa.Table("connections", metadata, autoload_with=engine)
    with engine.begin() as connection:
        host = {
            "workspace_id": 0,
            "host_id": _ID,
            "user_id": "owner@example.test",
            "name": "persistent-computer",
            "status": 2,
            "created_at": 1,
            "updated_at": 2,
            "token_hash": "a" * 64,
            "sandbox_provider": "example",
        }
        if starting_revision == _FORK:
            host.update(launch_token_only=True, lifecycle=2, sleep_operation_id="sleep-canary")
        connection.execute(hosts.insert().values(**host))
        connection.execute(
            conversations.insert().values(
                workspace_id=0,
                id=_ID,
                root_conversation_id=_ID,
                title="history-canary",
                created_at=1,
                updated_at=2,
            )
        )
        connection.execute(
            session_metadata.insert().values(
                workspace_id=0,
                id=_ID,
                host_id=_ID,
                workspace="/workspace",
                kind=1,
                **({"pull_requests": b"pr-canary"} if starting_revision == _RELEASE else {}),
            )
        )
        connection.execute(
            tasks.insert().values(
                workspace_id=0,
                id=_ID,
                name="schedule-canary",
                prompt=b"resume-canary",
                rrule="FREQ=DAILY;COUNT=2",
                agent_id=_ID,
                timezone="UTC",
                state=4 if starting_revision == _RELEASE else 1,
                created_at=1,
            )
        )
        connection.execute(
            runs.insert().values(
                workspace_id=0,
                id=_ID,
                scheduled_task_id=_ID,
                status=3,
                scheduled_at=1,
                fired_at=2,
            )
        )
        connection.execute(
            connections.insert().values(
                workspace_id=0,
                user_id="owner@example.test",
                provider="github",
                account_id="connection-canary",
                secret_enc="secret-canary",
                metadata_json="{}",
                created_at=1,
                updated_at=2,
            )
        )


def _assert_canaries(engine: sa.Engine, starting_revision: str) -> None:
    """Every row :func:`_insert_canaries` stored survived the upgrade unchanged."""
    with engine.connect() as connection:
        assert (
            connection.execute(sa.text("SELECT title FROM conversations")).scalar_one()
            == "history-canary"
        )
        assert (
            connection.execute(
                sa.text("SELECT host_id FROM omnigent_conversation_metadata")
            ).scalar_one()
            == _ID
        )
        assert connection.execute(
            sa.text("SELECT pull_requests FROM omnigent_conversation_metadata")
        ).scalar_one() == (b"pr-canary" if starting_revision == _RELEASE else None)
        assert connection.execute(
            sa.text("SELECT name, prompt, state FROM scheduled_tasks")
        ).one() == (
            "schedule-canary",
            b"resume-canary",
            4 if starting_revision == _RELEASE else 1,
        )
        assert connection.execute(
            sa.text("SELECT scheduled_task_id, status, fired_at FROM scheduled_task_runs")
        ).one() == (_ID, 3, 2)
        assert connection.execute(
            sa.text("SELECT account_id, secret_enc FROM connections")
        ).one() == ("connection-canary", "secret-canary")
        assert connection.execute(
            sa.text("SELECT token_hash, launch_token_only FROM hosts")
        ).one() == ("a" * 64, True)
        if starting_revision == _FORK:
            assert connection.execute(
                sa.text("SELECT lifecycle, sleep_operation_id FROM hosts")
            ).one() == (2, "sleep-canary")


def _create_fork_database(path: Path) -> str:
    """Create a 0.19 fork database holding canaries; return its URL."""
    uri = f"sqlite:///{path}"
    engine = sa.create_engine(uri)
    try:
        _migrate(engine, _FORK)
        _insert_canaries(engine, _FORK)
    finally:
        engine.dispose()
    return uri


def _heads(uri: str) -> tuple[str, ...]:
    engine = sa.create_engine(uri)
    try:
        return _get_current_db_heads(engine)
    finally:
        engine.dispose()


def _assert_upgraded_with_canaries(uri: str) -> None:
    assert _heads(uri) == (_MERGE,)
    engine = sa.create_engine(uri)
    try:
        _assert_canaries(engine, _FORK)
    finally:
        engine.dispose()


@pytest.mark.parametrize("starting_revision", ["base", _FORK, _RELEASE])
def test_merge_preserves_deployed_data_and_joins_heads(
    tmp_path: Path, starting_revision: str
) -> None:
    """Keep identities, histories and schedules through either branch's upgrade."""
    source = tmp_path / "deployed.db"
    engine = sa.create_engine(f"sqlite:///{source}")
    try:
        _migrate(engine, starting_revision)
        if starting_revision != "base":
            _insert_canaries(engine, starting_revision)

        if starting_revision == _FORK:
            engine.dispose()
            restored = tmp_path / "restored.db"
            with sqlite3.connect(source) as original, sqlite3.connect(restored) as copy:
                original.backup(copy)
            engine = sa.create_engine(f"sqlite:///{restored}")

        _migrate(engine, "head")
        config = _build_alembic_config(str(engine.url))
        assert ScriptDirectory.from_config(config).get_heads() == [_MERGE]
        columns = {column["name"] for column in sa.inspect(engine).get_columns("hosts")}
        assert {"launch_token_only", "lifecycle", "sleep_operation_id", "start_failure"} <= columns
        assert "pull_requests" in {
            column["name"]
            for column in sa.inspect(engine).get_columns("omnigent_conversation_metadata")
        }
        with engine.connect() as connection:
            assert connection.execute(
                sa.text("SELECT version_num FROM alembic_version")
            ).one() == (_MERGE,)
        if starting_revision != "base":
            _assert_canaries(engine, starting_revision)

        # Undoing only the merge must preserve the schemas and expose both parents.
        _migrate(engine, _FORK, downgrade=True)
        with engine.connect() as connection:
            assert set(
                connection.execute(sa.text("SELECT version_num FROM alembic_version")).scalars()
            ) == {
                _FORK,
                _RELEASE,
            }
        _migrate(engine, "head")
    finally:
        engine.dispose()


def _debug_db_upgrade(uri: str) -> None:
    """The body of ``omnigent debug db-upgrade``, which a failed start suggests."""
    engine = sa.create_engine(uri)
    try:
        _run_migrations(engine, uri)
    finally:
        engine.dispose()


@pytest.mark.parametrize(
    "resume", [run_migrations, _debug_db_upgrade], ids=["start", "debug-db-upgrade"]
)
@pytest.mark.parametrize(
    "stopped_at",
    [("nf12ab34cd56",), ("d3f7a1c9e2b4", "9e82957bd115"), (_RELEASE,)],
    ids=["first-step", "both-release-branches", "release-merged"],
)
def test_upgrade_resumes_from_database_between_merge_branches(
    tmp_path: Path, stopped_at: tuple[str, ...], resume: Callable[[str], None]
) -> None:
    """A database an earlier upgrade left holding several revisions upgrades to head."""
    uri = _create_fork_database(tmp_path / "between-branches.db")
    engine = sa.create_engine(uri)
    try:
        for revision in stopped_at:
            _migrate(engine, revision)
    finally:
        engine.dispose()
    assert _heads(uri) == tuple(sorted({_FORK, *stopped_at}))

    resume(uri)

    _assert_upgraded_with_canaries(uri)
