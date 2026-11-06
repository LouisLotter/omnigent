"""Upgrade the live team schema to 0.21, on every database and through a crash.

``c0211a2b3c4d`` joins the team chain with upstream's ``d0a1f879a01d``. A live
database stands at ``rm1a2b3c4d5e``, so upgrading it applies 0.21's one revision,
whose check constraint makes SQLite rebuild ``scheduled_tasks``. Everything the
table held must survive that, and on SQLite the upgrade is one transaction
(UC-122): a failure anywhere leaves the database as it was.
"""

from __future__ import annotations

import os
import re
import shutil
import signal
import sqlite3
import subprocess
import sys
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy as sa
from alembic import command

from deploy.docker.entrypoint import run_migrations
from omnigent.db.utils import (
    _build_alembic_config,
    _create_engine,
    _get_current_db_heads,
    _initialize_or_verify_schema,
)
from tests.db.team_chain import RELEASE_BRANCH_HEAD

_TEAM_HEAD = "rm1a2b3c4d5e"
_JOIN = "c0211a2b3c4d"
_TASK_ID = bytes.fromhex("0123456789abcdef0123456789abcdef")
_RUN_ID = bytes.fromhex("fedcba9876543210fedcba9876543210")
_NEW_RUN_INDEX = "ix_scheduled_task_runs_conversation_status"

# Starts the server's migration and SIGKILLs itself right after a matching statement.
_KILLED_START = """
import os, signal, sys

import sqlalchemy as sa

from omnigent.db.utils import _create_engine, _initialize_or_verify_schema

uri, killed_after = sys.argv[1:]
engine = _create_engine(uri)


@sa.event.listens_for(engine, "after_cursor_execute")
def kill(conn, cursor, statement, parameters, context, executemany):
    if statement.lstrip().startswith(killed_after):
        os.kill(os.getpid(), signal.SIGKILL)


_initialize_or_verify_schema(engine, uri)
"""


def _with_database(base_uri: str, name: str) -> str:
    return re.sub(r"/[^/]*(\?.*)?$", f"/{name}", base_uri)


@pytest.fixture
def fresh_uri(tmp_path: Path) -> Iterator[str]:
    """An empty database: a file, or one of its own on ``OMNIGENT_TEST_DB_URI``'s server."""
    base_uri = os.environ.get("OMNIGENT_TEST_DB_URI", "")
    if not base_uri:
        yield f"sqlite:///{tmp_path / 'join.db'}"
        return
    name = f"omnigent_join_{uuid.uuid4().hex[:16]}"
    root = sa.create_engine(base_uri, isolation_level="AUTOCOMMIT")
    quoted = f"`{name}`" if root.dialect.name == "mysql" else f'"{name}"'
    with root.connect() as connection:
        connection.execute(sa.text(f"CREATE DATABASE {quoted}"))
    try:
        yield _with_database(base_uri, name)
    finally:
        with root.connect() as connection:
            suffix = " WITH (FORCE)" if root.dialect.name == "postgresql" else ""
            connection.execute(sa.text(f"DROP DATABASE IF EXISTS {quoted}{suffix}"))
        root.dispose()


@pytest.fixture
def engine(fresh_uri: str) -> Iterator[sa.Engine]:
    created = sa.create_engine(fresh_uri)
    try:
        yield created
    finally:
        created.dispose()


def _migrate(engine: sa.Engine, revision: str, *, downgrade: bool = False) -> None:
    config = _build_alembic_config(engine.url.render_as_string(hide_password=False))
    with engine.connect() as connection:
        config.attributes["connection"] = connection
        (command.downgrade if downgrade else command.upgrade)(config, revision)
        connection.commit()
    # As after a restart: pooled SQLite connections keep an old schema.
    engine.dispose()


def _insert_task(engine: sa.Engine) -> None:
    metadata = sa.MetaData()
    tasks = sa.Table("scheduled_tasks", metadata, autoload_with=engine)
    runs = sa.Table("scheduled_task_runs", metadata, autoload_with=engine)
    with engine.begin() as connection:
        connection.execute(
            tasks.insert().values(
                workspace_id=0,
                id=_TASK_ID,
                name="nightly",
                prompt=b"check the build",
                rrule="FREQ=DAILY",
                agent_id=_TASK_ID,
                timezone="UTC",
                state=1,
                host_id=_TASK_ID,
                user_id="alice",
                created_at=1,
            )
        )
        connection.execute(
            runs.insert().values(
                workspace_id=0,
                id=_RUN_ID,
                scheduled_task_id=_TASK_ID,
                conversation_id=_TASK_ID,
                status=3,
                scheduled_at=1,
                fired_at=2,
            )
        )


def _assert_task_kept(engine: sa.Engine) -> None:
    with engine.connect() as connection:
        task = connection.execute(
            sa.text(
                "SELECT name, prompt, state, host_id, user_id, session_mode, session_id "
                "FROM scheduled_tasks"
            )
        ).one()
        run = connection.execute(
            sa.text("SELECT scheduled_task_id, status, fired_at FROM scheduled_task_runs")
        ).one()
    # A task saved before the upgrade starts a new session each run.
    assert tuple(task) == ("nightly", b"check the build", 1, _TASK_ID, "alice", 1, None)
    assert tuple(run) == (_TASK_ID, 3, 2)


def _structure(engine: sa.Engine, table: str) -> dict[str, Any]:
    """What a rebuild could lose: columns, keys, indexes and constraints of *table*."""
    inspector = sa.inspect(engine)
    return {
        "columns": {
            column["name"]: (str(column["type"]), column["nullable"], column.get("default"))
            for column in inspector.get_columns(table)
        },
        "primary_key": tuple(inspector.get_pk_constraint(table)["constrained_columns"]),
        "indexes": {
            index["name"]: (tuple(index["column_names"]), bool(index.get("unique")))
            for index in inspector.get_indexes(table)
        },
        "uniques": {
            unique["name"]: tuple(unique["column_names"])
            for unique in inspector.get_unique_constraints(table)
        },
        "foreign_keys": sorted(
            (
                tuple(foreign_key["constrained_columns"]),
                foreign_key["referred_table"],
                tuple(foreign_key["referred_columns"]),
            )
            for foreign_key in inspector.get_foreign_keys(table)
        ),
        "checks": {
            check["name"]: re.sub(r"[\s`\"()]", "", check["sqltext"]).lower()
            for check in inspector.get_check_constraints(table)
        },
    }


def _without_session_mode(structure: dict[str, Any]) -> dict[str, Any]:
    """*structure* less what the 0.21 revision adds to ``scheduled_tasks``."""
    trimmed = {**structure}
    trimmed["columns"] = {
        name: column
        for name, column in structure["columns"].items()
        if name not in {"session_mode", "session_id"}
    }
    trimmed["checks"] = {
        name: check
        for name, check in structure["checks"].items()
        if name != "ck_scheduled_tasks_session_mode"
    }
    return trimmed


def test_a_team_database_upgrades_to_the_join_and_keeps_every_table_detail(
    engine: sa.Engine,
) -> None:
    _migrate(engine, _TEAM_HEAD)
    assert _get_current_db_heads(engine) == (_TEAM_HEAD,)
    _insert_task(engine)
    tasks_before = _structure(engine, "scheduled_tasks")
    runs_before = _structure(engine, "scheduled_task_runs")
    engine.dispose()

    _migrate(engine, "head")

    assert _get_current_db_heads(engine) == (_JOIN,)
    tasks_after = _structure(engine, "scheduled_tasks")
    assert _without_session_mode(tasks_after) == tasks_before
    assert {"session_mode", "session_id"} <= tasks_after["columns"].keys()
    assert "ck_scheduled_tasks_session_mode" in tasks_after["checks"]
    runs_after = _structure(engine, "scheduled_task_runs")
    assert runs_after["indexes"].pop(_NEW_RUN_INDEX) == (
        ("workspace_id", "conversation_id", "status", "id"),
        False,
    )
    assert runs_after == runs_before
    _assert_task_kept(engine)


def test_the_new_check_refuses_an_unknown_session_mode(engine: sa.Engine) -> None:
    _migrate(engine, _TEAM_HEAD)
    _insert_task(engine)
    _migrate(engine, "head")

    # MySQL reports a violated check as an operational error.
    with (
        pytest.raises((sa.exc.IntegrityError, sa.exc.OperationalError)),
        engine.begin() as connection,
    ):
        connection.execute(sa.text("UPDATE scheduled_tasks SET session_mode = 3"))


def test_a_fresh_database_reaches_the_join_with_the_new_columns(engine: sa.Engine) -> None:
    _migrate(engine, "head")

    assert _get_current_db_heads(engine) == (_JOIN,)
    fresh = _structure(engine, "scheduled_tasks")
    assert {"session_mode", "session_id"} <= fresh["columns"].keys()
    assert "ck_scheduled_tasks_session_mode" in fresh["checks"]
    assert _NEW_RUN_INDEX in _structure(engine, "scheduled_task_runs")["indexes"]


def test_undoing_the_join_exposes_both_parents_and_changes_no_schema(engine: sa.Engine) -> None:
    _migrate(engine, "head")
    structure = _structure(engine, "scheduled_tasks")

    _migrate(engine, _TEAM_HEAD, downgrade=True)

    assert set(_get_current_db_heads(engine)) == {_TEAM_HEAD, RELEASE_BRANCH_HEAD}
    assert _structure(engine, "scheduled_tasks") == structure
    _migrate(engine, "head")
    assert _get_current_db_heads(engine) == (_JOIN,)


def _sqlite_only(uri: str) -> None:
    if not uri.startswith("sqlite"):
        pytest.skip("the rebuild and its transaction are SQLite's")


def _team_database(uri: str) -> None:
    engine = sa.create_engine(uri)
    try:
        _migrate(engine, _TEAM_HEAD)
        _insert_task(engine)
    finally:
        engine.dispose()


def _schema(path: Path) -> list[tuple[str, ...]]:
    with sqlite3.connect(path) as connection:
        return connection.execute(
            "SELECT type, name, tbl_name, sql FROM sqlite_master ORDER BY type, name"
        ).fetchall()


def _heads(uri: str) -> tuple[str, ...]:
    engine = sa.create_engine(uri)
    try:
        return _get_current_db_heads(engine)
    finally:
        engine.dispose()


def _assert_upgraded(uri: str) -> None:
    assert _heads(uri) == (_JOIN,)
    engine = sa.create_engine(uri)
    try:
        _assert_task_kept(engine)
    finally:
        engine.dispose()


def _interrupt_after_write(engine: sa.Engine, write_number: int) -> list[str]:
    """Raise right after the engine's *write_number*-th writing statement; return them all."""
    writes: list[str] = []

    @sa.event.listens_for(engine, "after_cursor_execute")
    def interrupt(conn, cursor, statement, parameters, context, executemany):
        if statement.split(None, 1)[0].upper() not in {"SELECT", "PRAGMA", "BEGIN"}:
            writes.append(statement)
            if len(writes) == write_number:
                raise RuntimeError("injected interruption")

    return writes


def test_an_upgrade_interrupted_after_any_statement_leaves_the_database_as_it_was(
    tmp_path: Path, fresh_uri: str
) -> None:
    """The rebuild runs inside the upgrade's one transaction (UC-122)."""
    _sqlite_only(fresh_uri)
    template = tmp_path / "join.db"
    _team_database(fresh_uri)
    team_schema = _schema(template)
    interrupted_write = 0
    while True:
        interrupted_write += 1
        database = tmp_path / f"interrupted-{interrupted_write}.db"
        shutil.copyfile(template, database)
        uri = f"sqlite:///{database}"
        engine = _create_engine(uri)
        writes = _interrupt_after_write(engine, interrupted_write)
        try:
            _initialize_or_verify_schema(engine, uri)
        except RuntimeError as exc:
            assert "automatic migration failed" in str(exc)
            assert str(exc.__cause__) == "injected interruption"
        else:
            # The upgrade wrote fewer statements than that, so every one was interrupted.
            break
        finally:
            engine.dispose()

        statement = writes[interrupted_write - 1]
        assert (_heads(uri), _schema(database)) == ((_TEAM_HEAD,), team_schema), statement
        run_migrations(uri)
        _assert_upgraded(uri)

    _assert_upgraded(uri)
    # The rebuild alone is several statements.
    assert interrupted_write > 5


@pytest.mark.skipif(sys.platform == "win32", reason="SIGKILL is POSIX-only")
@pytest.mark.parametrize(
    "killed_after",
    [
        "CREATE TABLE _alembic_tmp_scheduled_tasks",
        "INSERT INTO _alembic_tmp_scheduled_tasks",
        "DROP TABLE scheduled_tasks",
        "ALTER TABLE _alembic_tmp_scheduled_tasks",
        "CREATE INDEX ix_scheduled_task_runs_conversation_status",
        f"UPDATE alembic_version SET version_num='{_JOIN}'",
    ],
    ids=["temp-created", "rows-copied", "table-dropped", "renamed", "run-index", "last-statement"],
)
def test_an_upgrade_killed_during_the_rebuild_resumes_on_next_start(
    tmp_path: Path, fresh_uri: str, killed_after: str
) -> None:
    """A process killed mid-rebuild leaves the team database the next start upgrades."""
    _sqlite_only(fresh_uri)
    database = tmp_path / "join.db"
    _team_database(fresh_uri)
    team_schema = _schema(database)

    killed = subprocess.run(
        [sys.executable, "-c", _KILLED_START, fresh_uri, killed_after],
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert killed.returncode == -signal.SIGKILL, killed.stderr
    assert (_heads(fresh_uri), _schema(database)) == ((_TEAM_HEAD,), team_schema)
    run_migrations(fresh_uri)
    _assert_upgraded(fresh_uri)
