"""The team-member-removals migration (rm1a2b3c4d5e): additive and resumable.

It adds ``team_member_removals`` and ``team_member_removal_items`` and changes
no existing row.
"""

from __future__ import annotations

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
from omnigent.server.accounts_store import SqlAlchemyAccountStore
from omnigent.stores.team_store import SqlAlchemyTeamStore

_PREVIOUS = "ti1a2b3c4d5e"
_REVISION = "rm1a2b3c4d5e"
_TABLES = ("team_member_removals", "team_member_removal_items")


def _migrate(engine: sa.Engine, revision: str, *, downgrade: bool = False) -> None:
    config = _build_alembic_config(engine.url.render_as_string(hide_password=False))
    with engine.connect() as connection:
        if engine.dialect.name == "cockroachdb":
            _prepare_crdb_schema_transaction(connection, _crdb_server_version(engine))
        config.attributes["connection"] = connection
        (command.downgrade if downgrade else command.upgrade)(config, revision)
        connection.commit()
    # As after a restart: pooled SQLite connections keep an old schema.
    engine.dispose()


@pytest.fixture
def engine(db_uri: str):
    engine = get_or_create_engine(db_uri)
    try:
        yield engine
    finally:
        _initialize_or_verify_schema(engine, db_uri)


def _has_new_schema(engine: sa.Engine) -> bool:
    inspector = sa.inspect(engine)
    return all(inspector.has_table(table) for table in _TABLES)


def _team(db_uri: str) -> tuple[str, str]:
    """An owner and a member, as before the migration; returns their member ids."""
    accounts = SqlAlchemyAccountStore(db_uri)
    accounts.create_user_with_password("owner", "password-hash", is_admin=True, team_owner=True)
    accounts.create_user_with_password("bob", "password-hash")
    teams = SqlAlchemyTeamStore(db_uri)
    owner = teams.get_member_for_account("owner")
    assert owner is not None
    bob = teams.add_member("bob", role="member", state="active")
    return owner.id, bob.id


def _remove_works(db_uri: str, owner_id: str, member_id: str) -> None:
    removal = SqlAlchemyAccountStore(db_uri).remove_member(member_id, actor_member_id=owner_id)
    assert removal.member_id == member_id
    assert SqlAlchemyTeamStore(db_uri).get_removal(member_id) == removal


def test_upgrade_adds_the_tables_and_keeps_the_team(db_uri, engine) -> None:
    owner_id, bob_id = _team(db_uri)
    _migrate(engine, _PREVIOUS, downgrade=True)
    assert not _has_new_schema(engine)
    with engine.connect() as connection:
        members = connection.execute(
            sa.text("SELECT user_id, role, state FROM team_members ORDER BY user_id")
        ).all()

    _migrate(engine, _REVISION)

    assert _get_current_db_heads(engine) == (_REVISION,)
    assert _has_new_schema(engine)
    with engine.connect() as connection:
        assert (
            connection.execute(
                sa.text("SELECT user_id, role, state FROM team_members ORDER BY user_id")
            ).all()
            == members
        )
    _remove_works(db_uri, owner_id, bob_id)


def test_upgrade_resumes_after_a_committed_part(db_uri, engine) -> None:
    """MySQL and CockroachDB commit each step; a retried upgrade adds the rest."""
    owner_id, bob_id = _team(db_uri)
    _migrate(engine, _PREVIOUS, downgrade=True)
    _migrate(engine, _REVISION)
    suffix = " ON team_member_removals" if engine.dialect.name == "mysql" else ""
    for statement, parameters in (
        ("DROP TABLE team_member_removal_items", {}),
        ("DROP INDEX ix_team_member_removals_removed_at" + suffix, {}),
        ("UPDATE alembic_version SET version_num = :rev", {"rev": _PREVIOUS}),
    ):
        with engine.begin() as connection:
            connection.execute(sa.text(statement), parameters)
    engine.dispose()

    _migrate(engine, _REVISION)

    assert _get_current_db_heads(engine) == (_REVISION,)
    assert _has_new_schema(engine)
    assert "ix_team_member_removals_removed_at" in {
        index["name"] for index in sa.inspect(engine).get_indexes("team_member_removals")
    }
    _remove_works(db_uri, owner_id, bob_id)


def test_downgrade_drops_only_the_new_tables(db_uri, engine) -> None:
    owner_id, bob_id = _team(db_uri)
    _remove_works(db_uri, owner_id, bob_id)

    _migrate(engine, _PREVIOUS, downgrade=True)

    assert not sa.inspect(engine).has_table("team_member_removals")
    assert not sa.inspect(engine).has_table("team_member_removal_items")
    with engine.connect() as connection:
        # The removal itself stays: the membership is still removed.
        assert (
            connection.execute(
                sa.text("SELECT COUNT(*) FROM team_members WHERE current_user_id IS NULL")
            ).scalar_one()
            == 1
        )
    _migrate(engine, _REVISION)
