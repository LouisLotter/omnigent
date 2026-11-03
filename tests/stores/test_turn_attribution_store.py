"""Turn and approval attribution in the stores (WP8.8).

A turn's attribution is saved in the same insert as the item that starts it,
so it lasts exactly as long as the thread. A separately hosted conversation
database created before the column existed reads it as unknown and never
fails a turn. Approval records last until their thread is deleted.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from pathlib import Path

import pytest
import sqlalchemy as sa

from omnigent.db.utils import get_or_create_engine
from omnigent.entities import MessageData, NewConversationItem
from omnigent.entities.attribution import (
    HostAttribution,
    InitiatorAttribution,
    OwnerAttribution,
    TurnAttribution,
)
from omnigent.stores.approval_store import SqlAlchemyApprovalStore, VerdictState
from omnigent.stores.conversation_store.sqlalchemy_store import SqlAlchemyConversationStore

_CANARY = "PROMPT-CANARY-7f3a"


def _attribution(user: str = "bob") -> TurnAttribution:
    return TurnAttribution(
        initiator=InitiatorAttribution(kind="person", user=user, source="signed_in"),
        owner=OwnerAttribution(user="alice"),
        host=HostAttribution(id="h1", name="alice-laptop", owner="alice", kind="personal"),
        harness="claude-native",
        recorded_at=int(time.time()),
    )


def _message(text: str, attribution: TurnAttribution | None, stable_id: str | None = None):
    return NewConversationItem(
        type="message",
        response_id="turn_1",
        data=MessageData(role="user", content=[{"type": "input_text", "text": text}]),
        created_by=attribution.responsible_person() if attribution else None,
        attribution=attribution,
        stable_id=stable_id,
    )


def test_attribution_is_saved_with_its_item_and_read_back(db_uri: str) -> None:
    store = SqlAlchemyConversationStore(db_uri)
    conversation = store.create_conversation()
    [saved] = store.append(conversation.id, [_message(_CANARY, _attribution())])

    assert saved.attribution == _attribution().model_copy(
        update={"recorded_at": saved.attribution.recorded_at}
    )
    [listed] = store.list_items(conversation.id).data
    assert listed.attribution is not None
    assert listed.attribution.initiator.user == "bob"
    assert store.get_item(conversation.id, saved.id).attribution == listed.attribution
    api = listed.to_api_dict()["attribution"]
    assert api["initiator"]["kind"] == "person"
    # Unknown values are explicit nulls in the API, never invented.
    assert api["subscription"]["account"] is None and api["model"]["reported"] is None

    # The stored attribution is names, ids and settings: never the prompt.
    with get_or_create_engine(db_uri).connect() as connection:
        raw = connection.execute(
            sa.text("SELECT attribution FROM conversation_items WHERE attribution IS NOT NULL")
        ).scalar_one()
    assert _CANARY not in raw
    assert json.loads(raw)["initiator"]["user"] == "bob"


def test_a_retried_turn_keeps_its_first_attribution(db_uri: str) -> None:
    """T12: a resent message (same stable id) neither duplicates nor relabels the turn."""
    store = SqlAlchemyConversationStore(db_uri)
    conversation = store.create_conversation()
    stable = uuid.uuid4().hex
    [first] = store.append(conversation.id, [_message("go", _attribution("bob"), stable)])
    [retry] = store.append(conversation.id, [_message("go", _attribution("carol"), stable)])

    assert retry.deduplicated
    assert retry.attribution is not None and retry.attribution.initiator.user == "bob"
    assert [item.id for item in store.list_items(conversation.id).data] == [first.id]


def test_a_copied_thread_keeps_who_did_each_turn(db_uri: str) -> None:
    store = SqlAlchemyConversationStore(db_uri)
    conversation = store.create_conversation()
    store.append(conversation.id, [_message("go", _attribution("bob"))])

    copy = store.fork_conversation(conversation.id)

    [copied] = store.list_items(copy.id).data
    assert copied.attribution is not None and copied.attribution.initiator.user == "bob"


def test_a_separate_conversation_database_without_the_column_reads_unknown(
    tmp_path: Path,
) -> None:
    """A conversation database created before the column: turns still save; attribution unknown."""
    main_uri = f"sqlite:///{tmp_path / 'main.db'}"
    conversations_uri = f"sqlite:///{tmp_path / 'conversations.db'}"
    from omnigent.db import ConversationBase

    old = sa.create_engine(conversations_uri)
    ConversationBase.metadata.create_all(old)
    with old.begin() as connection:
        connection.execute(sa.text("ALTER TABLE conversation_items DROP COLUMN attribution"))
    old.dispose()

    store = SqlAlchemyConversationStore(main_uri, conversation_storage_location=conversations_uri)
    conversation = store.create_conversation()
    [saved] = store.append(conversation.id, [_message("go", _attribution())])

    assert saved.attribution is None
    [listed] = store.list_items(conversation.id).data
    assert listed.created_by == "bob" and listed.attribution is None
    assert store.get_item(conversation.id, saved.id).attribution is None
    assert store.fork_conversation(conversation.id) is not None


def _record(store: SqlAlchemyApprovalStore, session_id: str, *, expires_in: int = 600) -> str:
    elicitation_id = f"elicit_{uuid.uuid4().hex}"
    store.record_request(
        elicitation_id=elicitation_id,
        session_id=session_id,
        binding_digest=hashlib.sha256(b"bound").digest(),
        binding={"owner": "alice", "initiator": "alice"},
        requirements=[
            {"kind": "personal", "resource": [1, session_id]},
            {"kind": "team", "resource": [3, ""]},
        ],
        expires_at=int(time.time()) + expires_in,
        attribution={"v": 1, "initiator": "alice", "contributors": ["bob"]},
    )
    return elicitation_id


def test_an_approval_record_is_kept_until_its_thread_is_deleted(db_uri: str) -> None:
    conversations = SqlAlchemyConversationStore(db_uri)
    approvals = SqlAlchemyApprovalStore(db_uri)
    thread = conversations.create_conversation()
    child = conversations.create_conversation(kind="sub_agent", parent_conversation_id=thread.id)
    other = conversations.create_conversation()
    asked = _record(approvals, thread.id, expires_in=-10)
    approvals.record_answer(
        asked,
        answer={"by": "alice", "action": "accept", "satisfies": [0]},
        decline=False,
        now=int(time.time()) - 20,
    )
    in_child = _record(approvals, child.id)
    elsewhere = _record(approvals, other.id)

    # Answered rows survive the expiry purge; the record says who asked.
    approvals.purge_expired(before=int(time.time()))
    kept = approvals.get_verdict(asked)
    assert kept is not None and kept.attribution == {
        "v": 1,
        "initiator": "alice",
        "contributors": ["bob"],
    }
    listed = approvals.session_verdicts([thread.id, child.id])
    assert {record.elicitation_id for record in listed} == {asked, in_child}

    import asyncio

    assert asyncio.run(conversations.delete_conversation(thread.id))
    assert approvals.get_verdict(asked) is None
    assert approvals.get_verdict(in_child) is None
    assert approvals.get_verdict(elsewhere) is not None


def test_two_approvers_answering_at_once_are_both_recorded_once(db_uri: str) -> None:
    """T12: parallel answers to different requirements: both kept, one decision."""
    approvals = SqlAlchemyApprovalStore(db_uri)
    asked = _record(approvals, uuid.uuid4().hex)
    barrier = threading.Barrier(2)

    def answer(by: str, index: int) -> None:
        barrier.wait()
        approvals.record_answer(
            asked,
            answer={"by": by, "action": "accept", "satisfies": [index], "as": [{"role": by}]},
            decline=False,
        )

    threads = [
        threading.Thread(target=answer, args=("alice", 0)),
        threading.Thread(target=answer, args=("dave", 1)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    record = approvals.get_verdict(asked)
    assert record is not None and record.state is VerdictState.APPROVED
    assert sorted(answer["by"] for answer in record.answers) == ["alice", "dave"]
    assert record.decided_by == record.answers[-1]["by"]


@pytest.mark.parametrize("answered", [False, True])
def test_only_unanswered_expired_approvals_are_purged(db_uri: str, answered: bool) -> None:
    approvals = SqlAlchemyApprovalStore(db_uri)
    asked = _record(approvals, uuid.uuid4().hex, expires_in=-10)
    if answered:
        approvals.record_answer(
            asked,
            answer={"by": "alice", "action": "accept", "satisfies": [0]},
            decline=False,
            now=int(time.time()) - 20,
        )
    approvals.purge_expired(before=int(time.time()))
    assert (approvals.get_verdict(asked) is not None) is answered
