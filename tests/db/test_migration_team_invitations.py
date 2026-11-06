"""The team-invitations migration (ti1a2b3c4d5e): additive and resumable.

It adds ``team_invitations``, ``team_identity_bindings.email`` and
``teams.admission_revision`` and changes no existing row.
"""

from __future__ import annotations

import pytest
import sqlalchemy as sa
from alembic import command

from omnigent.db.cockroachdb import _crdb_server_version, _prepare_crdb_schema_transaction
from omnigent.db.utils import (
    _build_alembic_config,
    _initialize_or_verify_schema,
    get_or_create_engine,
)
from omnigent.server.accounts_store import SqlAlchemyAccountStore
from omnigent.stores.team_store import SqlAlchemyTeamStore
from tests.db.team_chain import team_chain_heads

_PREVIOUS = "at1a2b3c4d5e"
_REVISION = "ti1a2b3c4d5e"


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


def _columns(engine: sa.Engine, table: str) -> set[str]:
    return {column["name"] for column in sa.inspect(engine).get_columns(table)}


def _has_new_schema(engine: sa.Engine) -> bool:
    return (
        sa.inspect(engine).has_table("team_invitations")
        and "email" in _columns(engine, "team_identity_bindings")
        and "admission_revision" in _columns(engine, "teams")
        and "founding_member_id" in _columns(engine, "teams")
    )


def _use_new_schema(db_uri: str) -> None:
    """Invitations, a binding's address and the admission revision all work."""
    store = SqlAlchemyTeamStore(db_uri)
    owner = store.get_member_for_account("owner")
    assert owner is not None
    store.bind_external_identity(
        owner.id,
        issuer="https://team.example",
        subject="owner-sub",
        audience="aud",
        provenance="migration",
        email="owner@example.test",
    )
    invitation, token = store.create_invitation(
        "alice@example.test", "member", created_by_member_id=owner.id, ttl_seconds=3600
    )
    assert store.invitation_for_token(token) == invitation
    snapshot = store.admission_snapshot()
    assert snapshot.emails == ("alice@example.test", "owner@example.test")
    assert snapshot.revision >= 2


def test_upgrade_adds_the_schema_and_keeps_the_owner(db_uri, engine) -> None:
    SqlAlchemyAccountStore(db_uri).create_user_with_password(
        "owner", "password-hash", is_admin=True, team_owner=True
    )
    _migrate(engine, _PREVIOUS, downgrade=True)
    assert not _has_new_schema(engine)
    with engine.connect() as connection:
        members = connection.execute(
            sa.text("SELECT user_id, role, state FROM team_members")
        ).all()

    _migrate(engine, _REVISION)

    assert team_chain_heads(engine) == (_REVISION,)
    assert _has_new_schema(engine)
    with engine.connect() as connection:
        assert (
            connection.execute(sa.text("SELECT user_id, role, state FROM team_members")).all()
            == members
        )
        assert (
            connection.execute(sa.text("SELECT admission_revision FROM teams")).scalar_one() == 0
        )
        # The only active owner, unbound, is named the founding member.
        assert (
            connection.execute(sa.text("SELECT founding_member_id FROM teams")).scalar_one()
            == connection.execute(sa.text("SELECT id FROM team_members")).scalar_one()
        )
    _use_new_schema(db_uri)


def test_upgrade_resumes_after_a_committed_part(db_uri, engine) -> None:
    """MySQL and CockroachDB commit each step; a retried upgrade adds the rest."""
    SqlAlchemyAccountStore(db_uri).create_user_with_password(
        "owner", "password-hash", is_admin=True, team_owner=True
    )
    _migrate(engine, _PREVIOUS, downgrade=True)
    _migrate(engine, _REVISION)
    suffix = " ON team_invitations" if engine.dialect.name == "mysql" else ""
    for statement, parameters in (
        ("DROP INDEX ix_team_invitations_email" + suffix, {}),
        # The two columns stay: the retry must not add them twice.
        ("UPDATE alembic_version SET version_num = :rev", {"rev": _PREVIOUS}),
    ):
        with engine.begin() as connection:
            connection.execute(sa.text(statement), parameters)
    engine.dispose()

    _migrate(engine, _REVISION)

    assert team_chain_heads(engine) == (_REVISION,)
    assert "ix_team_invitations_email" in {
        index["name"] for index in sa.inspect(engine).get_indexes("team_invitations")
    }
    _use_new_schema(db_uri)


def test_downgrade_drops_only_the_new_schema(db_uri, engine) -> None:
    SqlAlchemyAccountStore(db_uri).create_user_with_password(
        "owner", "password-hash", is_admin=True, team_owner=True
    )
    _use_new_schema(db_uri)

    _migrate(engine, _PREVIOUS, downgrade=True)

    assert not _has_new_schema(engine)
    with engine.connect() as connection:
        assert connection.execute(sa.text("SELECT COUNT(*) FROM team_members")).scalar_one() == 1
        assert (
            connection.execute(sa.text("SELECT COUNT(*) FROM team_identity_bindings")).scalar_one()
            == 1
        )
    _migrate(engine, _REVISION)
