"""The team-of-one migration (tm1a2b3c4d5e): additive, unambiguous and resumable."""

from __future__ import annotations

import shutil
import signal
import sqlite3
import subprocess
import sys
import uuid
from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic import command

from omnigent.db.cockroachdb import _crdb_server_version, _prepare_crdb_schema_transaction
from omnigent.db.utils import (
    _build_alembic_config,
    _create_engine,
    _get_current_db_heads,
    _initialize_or_verify_schema,
    get_or_create_engine,
)
from omnigent.server.accounts_store import SqlAlchemyAccountStore
from omnigent.server.auth import LEVEL_EDIT, LEVEL_OWNER
from omnigent.stores.conversation_store.sqlalchemy_store import SqlAlchemyConversationStore
from omnigent.stores.host_store import HostStore
from omnigent.stores.permission_store.sqlalchemy_store import SqlAlchemyPermissionStore
from omnigent.stores.scheduled_task_store.sqlalchemy_store import SqlAlchemyScheduledTaskStore
from tests.db.team_chain import team_chain_heads

_PREVIOUS = "mr1a2b3c4d5e"
_REVISION = "tm1a2b3c4d5e"
_TEAM_TABLES = ("teams", "team_members", "team_identity_bindings")
# Tables whose rows the upgrade must leave exactly as they were.
_PRESERVED = (
    "users",
    "hosts",
    "session_permissions",
    "omnigent_conversation_metadata",
    "scheduled_tasks",
    "connections",
)

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


def _migrate(engine: sa.Engine, revision: str, *, downgrade: bool = False) -> None:
    config = _build_alembic_config(engine.url.render_as_string(hide_password=False))
    with engine.connect() as connection:
        if engine.dialect.name == "cockroachdb":
            _prepare_crdb_schema_transaction(connection, _crdb_server_version(engine))
        config.attributes["connection"] = connection
        (command.downgrade if downgrade else command.upgrade)(config, revision)
        connection.commit()


def _normalize(value: object) -> object:
    return bytes(value) if isinstance(value, memoryview) else value


def _snapshot(
    engine: sa.Engine, columns: dict[str, list[str]] | None = None
) -> dict[str, list[tuple[object, ...]]]:
    """Every preserved table's rows, in primary-key order.

    :param columns: Compare only these columns per table, e.g. those that
        existed before a later revision added more.
    """
    metadata = sa.MetaData()
    snapshot = {}
    with engine.connect() as connection:
        for name in _PRESERVED:
            table = sa.Table(name, metadata, autoload_with=connection)
            selected = [table.c[column] for column in columns[name]] if columns else list(table.c)
            rows = connection.execute(
                sa.select(*selected).order_by(*table.primary_key.columns)
            ).all()
            snapshot[name] = [tuple(_normalize(value) for value in row) for row in rows]
    return snapshot


def _columns(engine: sa.Engine) -> dict[str, list[str]]:
    inspector = sa.inspect(engine)
    return {name: [c["name"] for c in inspector.get_columns(name)] for name in _PRESERVED}


def _members(engine: sa.Engine) -> list[tuple[int, str, str | None, int, int]]:
    with engine.connect() as connection:
        return [
            tuple(row)
            for row in connection.execute(
                sa.text(
                    "SELECT workspace_id, user_id, current_user_id, role, state "
                    "FROM team_members ORDER BY workspace_id, user_id"
                )
            )
        ]


def _team_workspaces(engine: sa.Engine) -> list[int]:
    with engine.connect() as connection:
        return list(
            connection.execute(sa.text("SELECT workspace_id FROM teams ORDER BY workspace_id"))
            .scalars()
            .all()
        )


def _has_team_tables(engine: sa.Engine) -> dict[str, bool]:
    inspector = sa.inspect(engine)
    return {name: inspector.has_table(name) for name in _TEAM_TABLES}


def _seed_installation(db_uri: str) -> None:
    """A single-owner installation: the owner's work plus a second account's share."""
    accounts = SqlAlchemyAccountStore(db_uri)
    accounts.create_user_with_password("owner", "owner-hash", is_admin=True)
    accounts.create_user_with_password("bob", "bob-hash")
    accounts.create_user_with_password("gone", "gone-hash", is_admin=True)
    assert accounts.delete_user("gone") is True
    permissions = SqlAlchemyPermissionStore(db_uri)
    permissions.ensure_user("local", is_admin=True)
    conversations = SqlAlchemyConversationStore(db_uri)
    host_id = uuid.uuid4().hex
    HostStore(db_uri).upsert_on_connect(host_id, "owner-laptop", "owner")
    thread = conversations.create_conversation(agent_id=None, host_id=host_id, workspace="/w")
    permissions.grant("owner", thread.id, LEVEL_OWNER)
    permissions.grant("bob", thread.id, LEVEL_EDIT)
    SqlAlchemyScheduledTaskStore(db_uri).create(
        uuid.uuid4().hex, "nightly", "p", "FREQ=DAILY", "owner", uuid.uuid4().hex, "UTC"
    )
    with get_or_create_engine(db_uri).begin() as connection:
        connection.execute(
            sa.text(
                "INSERT INTO connections (workspace_id, user_id, provider, account_id, "
                "secret_enc, metadata_json, created_at, updated_at) VALUES "
                "(0, 'owner', 'github', '', 'ciphertext', '{}', 1, 1)"
            )
        )


@pytest.fixture
def engine(db_uri: str):
    engine = get_or_create_engine(db_uri)
    try:
        yield engine
    finally:
        _initialize_or_verify_schema(engine, db_uri)


def test_upgrade_makes_the_sole_admin_owner_and_changes_nothing_else(db_uri, engine) -> None:
    _seed_installation(db_uri)
    _migrate(engine, _PREVIOUS, downgrade=True)
    assert not any(_has_team_tables(engine).values())
    before = _snapshot(engine)

    _migrate(engine, _REVISION)

    assert team_chain_heads(engine) == (_REVISION,)
    assert _team_workspaces(engine) == [0]
    # Owner role (1), active state (2), mapped to the existing username.
    assert _members(engine) == [(0, "owner", "owner", 1, 2)]
    assert _snapshot(engine) == before
    owner = SqlAlchemyAccountStore(db_uri).get_user("owner")
    assert owner.is_admin
    with engine.connect() as connection:
        # The owner membership belongs to the account's current registration.
        assert connection.execute(
            sa.text("SELECT account_generation FROM team_members")
        ).scalar_one() == (owner.account_generation)

    # Upgrading again is a no-op; downgrading drops only the new tables.
    _migrate(engine, _REVISION)
    assert _members(engine) == [(0, "owner", "owner", 1, 2)]
    _migrate(engine, _PREVIOUS, downgrade=True)
    assert not any(_has_team_tables(engine).values())
    assert _snapshot(engine) == before
    _migrate(engine, _REVISION)
    assert _members(engine) == [(0, "owner", "owner", 1, 2)]


@pytest.mark.parametrize(
    "admins",
    [(), ("alice", "bob")],
    ids=["no-admin", "two-admins"],
)
def test_upgrade_assigns_no_owner_when_it_would_have_to_guess(db_uri, engine, admins) -> None:
    accounts = SqlAlchemyAccountStore(db_uri)
    for user_id in ("alice", "bob", "carol"):
        accounts.create_user_with_password(user_id, "hash", is_admin=user_id in admins)
    # The reserved legacy user never counts, even with its admin flag.
    SqlAlchemyPermissionStore(db_uri).ensure_user("local", is_admin=True)
    _migrate(engine, _PREVIOUS, downgrade=True)

    _migrate(engine, _REVISION)

    assert _team_workspaces(engine) == [0]
    assert _members(engine) == []


def test_upgrade_gives_every_workspace_its_own_team(db_uri, engine) -> None:
    from omnigent.db.db_models import workspace_scope

    for workspace_id, owner in ((0, "zero-owner"), (17, "other-owner")):
        with workspace_scope(workspace_id):
            SqlAlchemyAccountStore(db_uri).create_user_with_password(owner, "h", is_admin=True)
    _migrate(engine, _PREVIOUS, downgrade=True)

    _migrate(engine, _REVISION)

    assert _team_workspaces(engine) == [0, 17]
    assert _members(engine) == [
        (0, "zero-owner", "zero-owner", 1, 2),
        (17, "other-owner", "other-owner", 1, 2),
    ]


def test_upgrade_resumes_after_a_committed_part(db_uri, engine) -> None:
    """MySQL and CockroachDB commit schema steps: a retry finds some already done."""
    from omnigent.db.db_models import SqlTeam, SqlTeamMember

    _seed_installation(db_uri)
    _migrate(engine, _PREVIOUS, downgrade=True)
    team_id = uuid.uuid4().hex
    with engine.connect() as connection:
        if engine.dialect.name == "cockroachdb":
            _prepare_crdb_schema_transaction(connection, _crdb_server_version(engine))
        SqlTeam.__table__.create(connection)
        SqlTeamMember.__table__.create(connection)
        suffix = " ON team_members" if engine.dialect.name == "mysql" else ""
        connection.execute(sa.text("DROP INDEX ix_team_members_role_state" + suffix))
        connection.commit()
        connection.execute(
            SqlTeam.__table__.insert().values(workspace_id=0, id=team_id, created_at=1)
        )
        connection.commit()

    _migrate(engine, _REVISION)

    assert {index["name"] for index in sa.inspect(engine).get_indexes("team_members")} >= {
        "ix_team_members_role_state"
    }
    assert _has_team_tables(engine) == dict.fromkeys(_TEAM_TABLES, True)
    with engine.connect() as connection:
        stored = connection.execute(sa.text("SELECT id FROM teams")).scalar_one()
    assert bytes(stored).hex() == team_id
    assert _members(engine) == [(0, "owner", "owner", 1, 2)]


def test_upgrade_keeps_an_existing_membership(db_uri, engine) -> None:
    from omnigent.db.db_models import SqlTeamMember

    _seed_installation(db_uri)
    with engine.begin() as connection:
        connection.execute(
            SqlTeamMember.__table__.insert().values(
                workspace_id=0,
                id=uuid.uuid4().hex,
                user_id="bob",
                current_user_id="bob",
                role=1,
                state=2,
                created_at=1,
                updated_at=1,
            )
        )
        # A resumed upgrade re-runs the data step over tables that already hold members.
        connection.execute(sa.text(f"UPDATE alembic_version SET version_num = '{_PREVIOUS}'"))

    _migrate(engine, _REVISION)

    assert [member[1] for member in _members(engine)] == ["bob"]


def _fork_database(path: Path) -> str:
    """A SQLite installation at the previous revision; returns its URL."""
    uri = f"sqlite:///{path}"
    engine = sa.create_engine(uri)
    try:
        _migrate(engine, "head")
        _seed_installation(uri)
        _migrate(engine, _PREVIOUS, downgrade=True)
    finally:
        engine.dispose()
        get_or_create_engine(uri).dispose()
    return uri


def _sqlite_schema(path: Path) -> list[tuple[str, ...]]:
    with sqlite3.connect(path) as connection:
        return connection.execute(
            "SELECT type, name, tbl_name, sql FROM sqlite_master ORDER BY type, name"
        ).fetchall()


def _assert_upgraded(
    uri: str, before: dict[str, list[tuple[object, ...]]], columns: dict[str, list[str]]
) -> None:
    """A start-up upgrade reached the newest revision, past this one."""
    from alembic.script import ScriptDirectory

    engine = sa.create_engine(uri)
    try:
        head = ScriptDirectory.from_config(_build_alembic_config(uri)).get_current_head()
        assert _get_current_db_heads(engine) == (head,)
        assert _members(engine) == [(0, "owner", "owner", 1, 2)]
        # Later revisions may add columns; the ones that existed are unchanged.
        assert _snapshot(engine, columns) == before
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


def test_sqlite_upgrade_interrupted_anywhere_leaves_the_previous_database(tmp_path) -> None:
    template = tmp_path / "previous.db"
    template_uri = _fork_database(template)
    engine = sa.create_engine(template_uri)
    before, columns = _snapshot(engine), _columns(engine)
    engine.dispose()
    schema = _sqlite_schema(template)
    interrupted = 0
    while True:
        interrupted += 1
        database = tmp_path / f"interrupted-{interrupted}.db"
        shutil.copyfile(template, database)
        uri = f"sqlite:///{database}"
        engine = _create_engine(uri)
        writes = _interrupt_after_write(engine, interrupted)
        try:
            _initialize_or_verify_schema(engine, uri)
        except RuntimeError as exc:
            assert str(exc.__cause__) == "injected interruption"
        else:
            break
        finally:
            engine.dispose()
        check = sa.create_engine(uri)
        try:
            assert team_chain_heads(check) == (_PREVIOUS,), writes[-1]
            assert _snapshot(check) == before
        finally:
            check.dispose()
        assert _sqlite_schema(database) == schema, writes[-1]
        upgraded = _create_engine(uri)
        try:
            _initialize_or_verify_schema(upgraded, uri)
        finally:
            upgraded.dispose()
        _assert_upgraded(uri, before, columns)
    _assert_upgraded(uri, before, columns)
    # Three tables, three indexes, a team and its owner.
    assert interrupted > 8


@pytest.mark.skipif(sys.platform == "win32", reason="SIGKILL is POSIX-only")
@pytest.mark.parametrize(
    "killed_after",
    ["CREATE TABLE team_members", "INSERT INTO teams", "INSERT INTO team_members"],
)
def test_sqlite_upgrade_killed_mid_way_resumes_on_next_start(tmp_path, killed_after) -> None:
    database = tmp_path / "killed.db"
    uri = _fork_database(database)
    engine = sa.create_engine(uri)
    before, columns = _snapshot(engine), _columns(engine)
    engine.dispose()

    killed = subprocess.run(
        [sys.executable, "-c", _KILLED_START, uri, killed_after],
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert killed.returncode == -signal.SIGKILL, killed.stderr
    check = sa.create_engine(uri)
    try:
        assert team_chain_heads(check) == (_PREVIOUS,)
        assert not any(_has_team_tables(check).values())
    finally:
        check.dispose()
    engine = _create_engine(uri)
    try:
        _initialize_or_verify_schema(engine, uri)
    finally:
        engine.dispose()
    _assert_upgraded(uri, before, columns)
