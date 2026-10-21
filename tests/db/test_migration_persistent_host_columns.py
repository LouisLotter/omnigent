"""Tests for the persistent managed host columns migration (nn1a2b3c4d5e)."""

from __future__ import annotations

from pathlib import Path

import sqlalchemy as sa
from alembic import command

from omnigent.db.utils import _build_alembic_config, clear_engine_cache

_COLUMNS = {"lifecycle", "sleep_operation_id", "start_failure"}


def _migrate(uri: str, engine: sa.Engine, revision: str, *, down: bool = False) -> None:
    config = _build_alembic_config(uri)
    with engine.begin() as connection:
        config.attributes["connection"] = connection
        (command.downgrade if down else command.upgrade)(config, revision)


def _host_columns(engine: sa.Engine) -> dict[str, dict[str, object]]:
    return {column["name"]: column for column in sa.inspect(engine).get_columns("hosts")}


def test_upgrade_adds_nullable_persistent_columns_and_downgrade_removes_them(
    tmp_path: Path,
) -> None:
    """Existing hosts keep working: every new column is nullable, NULL meaning "not persistent"."""
    uri = f"sqlite:///{tmp_path / 'persistent-hosts.db'}"
    engine = sa.create_engine(uri)
    try:
        _migrate(uri, engine, "mm1a2b3c4d5e")
        assert not _COLUMNS & set(_host_columns(engine))

        _migrate(uri, engine, "nn1a2b3c4d5e")
        columns = _host_columns(engine)
        assert set(columns) >= _COLUMNS
        assert all(columns[name]["nullable"] is True for name in _COLUMNS)
        assert isinstance(columns["lifecycle"]["type"], sa.SmallInteger)

        _migrate(uri, engine, "mm1a2b3c4d5e", down=True)
        assert not _COLUMNS & set(_host_columns(engine))
    finally:
        engine.dispose()
        clear_engine_cache()
