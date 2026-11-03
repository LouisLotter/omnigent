"""The turn-attribution migration (at1a2b3c4d5e): additive and resumable.

It adds ``conversation_items.attribution``, ``approval_verdicts.attribution``
and the per-thread approval index, and changes no existing row.
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
    _get_current_db_heads,
    _initialize_or_verify_schema,
    get_or_create_engine,
)
from omnigent.entities import MessageData, NewConversationItem
from omnigent.entities.attribution import InitiatorAttribution, OwnerAttribution, TurnAttribution
from omnigent.stores.approval_store import SqlAlchemyApprovalStore
from omnigent.stores.conversation_store.sqlalchemy_store import SqlAlchemyConversationStore

_PREVIOUS = "ap1a2b3c4d5e"
_REVISION = "at1a2b3c4d5e"
_INDEX = "ix_approval_verdicts_session"


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


def _has_index(engine: sa.Engine) -> bool:
    return _INDEX in {
        index["name"] for index in sa.inspect(engine).get_indexes("approval_verdicts")
    }


def _added(engine: sa.Engine) -> tuple[bool, bool, bool]:
    return (
        "attribution" in _columns(engine, "conversation_items"),
        "attribution" in _columns(engine, "approval_verdicts"),
        _has_index(engine),
    )


def _seed_old_item(db_uri: str) -> str:
    """A thread with an item saved at the previous schema, by a store opened then."""
    store = SqlAlchemyConversationStore(db_uri)
    thread = store.create_conversation()
    store.append(
        thread.id,
        [
            NewConversationItem(
                type="message",
                response_id="turn_0",
                data=MessageData(role="user", content=[{"type": "input_text", "text": "old"}]),
                created_by="alice",
            )
        ],
    )
    return thread.id


def _use_both_columns(db_uri: str) -> None:
    """The stores read and write attribution against the migrated schema."""
    conversations = SqlAlchemyConversationStore(db_uri)
    thread = conversations.create_conversation()
    attribution = TurnAttribution(
        initiator=InitiatorAttribution(kind="person", user="bob", source="signed_in"),
        owner=OwnerAttribution(user="alice"),
        recorded_at=int(time.time()),
    )
    conversations.append(
        thread.id,
        [
            NewConversationItem(
                type="message",
                response_id="turn_1",
                data=MessageData(role="user", content=[{"type": "input_text", "text": "go"}]),
                attribution=attribution,
            )
        ],
    )
    [item] = conversations.list_items(thread.id).data
    assert item.attribution is not None and item.attribution.initiator.user == "bob"
    approvals = SqlAlchemyApprovalStore(db_uri)
    elicitation_id = f"elicit_{uuid.uuid4().hex}"
    approvals.record_request(
        elicitation_id=elicitation_id,
        session_id=thread.id,
        binding_digest=hashlib.sha256(b"bound").digest(),
        binding={"owner": "alice"},
        requirements=[{"kind": "personal", "resource": [1, thread.id]}],
        expires_at=int(time.time()) + 600,
        attribution={"initiator": "bob", "contributors": []},
    )
    [record] = approvals.session_verdicts([thread.id])
    assert record.attribution == {"initiator": "bob", "contributors": []}


def test_upgrade_adds_the_columns_and_index_and_changes_nothing_else(db_uri, engine) -> None:
    _migrate(engine, _PREVIOUS, downgrade=True)
    assert _added(engine) == (False, False, False)
    thread_id = _seed_old_item(db_uri)

    _migrate(engine, _REVISION)

    assert _get_current_db_heads(engine) == (_REVISION,)
    assert _added(engine) == (True, True, True)
    # The item saved before the revision is unchanged and reads as unknown.
    [old] = SqlAlchemyConversationStore(db_uri).list_items(thread_id).data
    assert (old.created_by, old.attribution) == ("alice", None)
    _use_both_columns(db_uri)


def test_upgrade_resumes_after_a_committed_part(db_uri, engine) -> None:
    """MySQL and CockroachDB commit each step; a retried upgrade adds the rest."""
    _migrate(engine, _PREVIOUS, downgrade=True)
    _migrate(engine, _REVISION)
    suffix = " ON approval_verdicts" if engine.dialect.name == "mysql" else ""
    drop_index = (
        f"DROP INDEX approval_verdicts@{_INDEX}"
        if engine.dialect.name == "cockroachdb"
        else f"DROP INDEX {_INDEX}{suffix}"
    )
    # One statement per transaction: CockroachDB refuses multi-statement DDL here.
    for statement, parameters in (
        (drop_index, {}),
        ("ALTER TABLE approval_verdicts DROP COLUMN attribution", {}),
        # The item column stays: the retry must not add it twice.
        ("UPDATE alembic_version SET version_num = :rev", {"rev": _PREVIOUS}),
    ):
        with engine.begin() as connection:
            connection.execute(sa.text(statement), parameters)
    engine.dispose()

    _migrate(engine, _REVISION)

    assert _get_current_db_heads(engine) == (_REVISION,)
    assert _added(engine) == (True, True, True)
    _use_both_columns(db_uri)


def test_downgrade_drops_only_what_it_added(db_uri, engine) -> None:
    _use_both_columns(db_uri)
    tables = set(sa.inspect(engine).get_table_names())

    _migrate(engine, _PREVIOUS, downgrade=True)

    assert _added(engine) == (False, False, False)
    assert set(sa.inspect(engine).get_table_names()) == tables
    with engine.connect() as connection:
        assert (
            connection.execute(sa.text("SELECT COUNT(*) FROM conversation_items")).scalar_one()
            >= 1
        )
        assert (
            connection.execute(sa.text("SELECT COUNT(*) FROM approval_verdicts")).scalar_one() >= 1
        )
    _migrate(engine, _REVISION)
    _use_both_columns(db_uri)
