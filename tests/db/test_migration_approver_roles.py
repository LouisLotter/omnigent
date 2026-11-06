"""The approver-roles migration (ap1a2b3c4d5e): additive and resumable.

It adds the approver-grant, approval-setting, protected-policy and
approval-ledger tables and changes no existing row.
"""

from __future__ import annotations

import hashlib
import time
import uuid

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
from omnigent.stores.approval_store import (
    ApprovalResource,
    PolicyDomain,
    ResourceKind,
    SqlAlchemyApprovalStore,
)
from tests.db.team_chain import team_chain_heads

_PREVIOUS = "ri1a2b3c4d5e"
_REVISION = "ap1a2b3c4d5e"
_TABLES = ("approval_resources", "approver_grants", "protected_policies", "approval_verdicts")
# Tables later revisions add; a downgrade below this revision drops them too.
_LATER_TABLES = (
    "team_invitations",
    "session_inputs",
    "session_turns",
    "team_member_removals",
    "team_member_removal_items",
)


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


def _tables(engine: sa.Engine) -> set[str]:
    return set(sa.inspect(engine).get_table_names())


def _has_login_id(engine: sa.Engine) -> bool:
    return "login_id" in {c["name"] for c in sa.inspect(engine).get_columns("connections")}


def _use_every_table(db_uri: str) -> None:
    """The store works against the migrated schema.

    The stores read the current schema, which later revisions extend, so the
    database is upgraded to head first.
    """
    engine = get_or_create_engine(db_uri)
    _migrate(engine, "head")
    store = SqlAlchemyApprovalStore(db_uri)
    service = ApprovalResource(ResourceKind.SERVICE, "tracker")
    assert store.add_approver(service, "carol", granted_by="admin")
    store.set_allow_initiator(service, True, updated_by="admin")
    store.create_rule(
        domain=PolicyDomain.TEAM,
        owner_user_id=None,
        name="team",
        handler="omnigent.policies.builtins.cel.cel_policy",
        factory_params={"expression": "{}"},
        approval=service,
        created_by="admin",
    )
    digest = hashlib.sha256(b"bound").digest()
    store.record_request(
        elicitation_id=f"elicit_{uuid.uuid4().hex}",
        session_id=uuid.uuid4().hex,
        binding_digest=digest,
        binding={"owner": "alice"},
        requirements=[{"kind": "team", "resource": service.as_json()}],
        expires_at=int(time.time()) + 600,
    )
    assert store.approvers_for([service])[0].approvers == ("carol",)
    assert len(store.rules_for(None)) == 1
    assert len(store.pending_verdicts()) == 1


def test_upgrade_adds_the_tables_and_changes_nothing_else(db_uri, engine) -> None:
    SqlAlchemyAccountStore(db_uri).create_user_with_password("alice", "password-hash")
    _migrate(engine, _PREVIOUS, downgrade=True)
    assert not _tables(engine) & set(_TABLES)
    with engine.connect() as connection:
        users = connection.execute(sa.text("SELECT id FROM users ORDER BY id")).all()

    _migrate(engine, _REVISION)

    assert team_chain_heads(engine) == (_REVISION,)
    assert _tables(engine) >= set(_TABLES)
    assert _has_login_id(engine)
    with engine.connect() as connection:
        assert connection.execute(sa.text("SELECT id FROM users ORDER BY id")).all() == users
    _use_every_table(db_uri)


def test_upgrade_resumes_after_a_committed_part(db_uri, engine) -> None:
    """MySQL and CockroachDB commit each table; a retried upgrade creates the rest."""
    _migrate(engine, _PREVIOUS, downgrade=True)
    _migrate(engine, _REVISION)
    suffix = " ON approver_grants" if engine.dialect.name == "mysql" else ""
    # One statement per transaction: CockroachDB refuses multi-statement DDL here.
    for statement, parameters in (
        ("DROP TABLE approval_verdicts", {}),
        ("DROP TABLE protected_policies", {}),
        ("DROP INDEX ix_approver_grants_approver" + suffix, {}),
        # The login id column stays: the retry must not add it twice.
        (
            "UPDATE alembic_version SET version_num = :rev WHERE version_num = :current",
            {"rev": _PREVIOUS, "current": _REVISION},
        ),
    ):
        with engine.begin() as connection:
            connection.execute(sa.text(statement), parameters)
    engine.dispose()

    _migrate(engine, _REVISION)

    assert team_chain_heads(engine) == (_REVISION,)
    assert _tables(engine) >= set(_TABLES)
    assert "ix_approver_grants_approver" in {
        index["name"] for index in sa.inspect(engine).get_indexes("approver_grants")
    }
    assert _has_login_id(engine)
    _use_every_table(db_uri)


def test_downgrade_drops_only_the_new_tables(db_uri, engine) -> None:
    SqlAlchemyAccountStore(db_uri).create_user_with_password("alice", "password-hash")
    _use_every_table(db_uri)
    # Downgrading through the later revisions drops their tables too.
    before = _tables(engine) - set(_TABLES) - set(_LATER_TABLES)

    _migrate(engine, _PREVIOUS, downgrade=True)

    assert _tables(engine) == before
    assert not _has_login_id(engine)
    with engine.connect() as connection:
        assert connection.execute(sa.text("SELECT COUNT(*) FROM users")).scalar_one() >= 1
    _migrate(engine, _REVISION)
    _use_every_table(db_uri)
