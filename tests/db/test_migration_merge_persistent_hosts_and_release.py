"""Upgrade either deployed schema and a restored fork snapshot to one head."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic import command
from alembic.script import ScriptDirectory

from omnigent.db.utils import _build_alembic_config

_FORK = "lt1b2c3d4e5f"
_RELEASE = "b2e4f6a8c013"
_MERGE = "c0201a2b3c4d"
_ID = bytes.fromhex("0123456789abcdef0123456789abcdef")


def _migrate(engine: sa.Engine, revision: str, *, downgrade: bool = False) -> None:
    config = _build_alembic_config(str(engine.url))
    with engine.begin() as connection:
        config.attributes["connection"] = connection
        (command.downgrade if downgrade else command.upgrade)(config, revision)


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
            metadata = sa.MetaData()
            hosts = sa.Table("hosts", metadata, autoload_with=engine)
            conversations = sa.Table("conversations", metadata, autoload_with=engine)
            session_metadata = sa.Table(
                "omnigent_conversation_metadata", metadata, autoload_with=engine
            )
            tasks = sa.Table("scheduled_tasks", metadata, autoload_with=engine)
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
                    host.update(
                        launch_token_only=True, lifecycle=2, sleep_operation_id="sleep-canary"
                    )
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
                        **(
                            {"pull_requests": b"pr-canary"}
                            if starting_revision == _RELEASE
                            else {}
                        ),
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
                    sa.text("SELECT token_hash, launch_token_only FROM hosts")
                ).one() == ("a" * 64, True)
                if starting_revision == _FORK:
                    assert connection.execute(
                        sa.text("SELECT lifecycle, sleep_operation_id FROM hosts")
                    ).one() == (2, "sleep-canary")

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
