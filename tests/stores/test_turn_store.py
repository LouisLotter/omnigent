"""Accepted inputs and registered turns: who took part in a turn, best effort (WP8.8)."""

from __future__ import annotations

import asyncio
import uuid

from omnigent.stores.conversation_store.sqlalchemy_store import SqlAlchemyConversationStore
from omnigent.stores.turn_store import InputKind, SqlAlchemyTurnStore, text_digest


def _thread(db_uri: str) -> str:
    return SqlAlchemyConversationStore(db_uri).create_conversation().id


def test_an_input_is_recorded_once_and_a_retry_keeps_the_first(db_uri: str) -> None:
    store = SqlAlchemyTurnStore(db_uri)
    thread = _thread(db_uri)
    first = store.record_input(thread, "item_1", kind=InputKind.PERSON, person="bob")
    again = store.record_input(thread, "item_1", kind=InputKind.PERSON, person="mallory")

    assert first == again
    assert again.person == "bob"


def test_a_turn_names_its_first_inputs_person_and_a_retry_returns_it(db_uri: str) -> None:
    store = SqlAlchemyTurnStore(db_uri)
    thread = _thread(db_uri)
    store.record_input(thread, "pending_b", kind=InputKind.PERSON, person="bob")
    store.record_input(thread, "pending_c", kind=InputKind.PERSON, person="carol")

    turn = store.register_turn(
        thread, "turn_1", runner_id="runner_1", input_ids=["pending_b", "pending_c"]
    )
    retried = store.register_turn(thread, "turn_1", runner_id="runner_1", input_ids=["other"])

    assert (turn.initiator, turn.unresolved) == ("bob", False)
    assert retried == turn
    assert store.latest_turn(thread) == turn


def test_an_input_named_nowhere_leaves_the_turn_unresolved(db_uri: str) -> None:
    store = SqlAlchemyTurnStore(db_uri)
    thread = _thread(db_uri)

    unknown = store.register_turn(thread, "turn_1", runner_id=None, input_ids=["item_unknown"])
    from_item = store.register_turn(
        thread,
        "turn_2",
        runner_id=None,
        input_ids=["item_old"],
        resolve=lambda value: (value == "item_old", "alice"),
    )

    assert unknown.unresolved and unknown.initiator is None
    assert (from_item.unresolved, from_item.initiator) == (False, "alice")


def test_authority_covers_every_input_since_the_turn_started_without_a_cap(
    db_uri: str,
) -> None:
    """R1: a long turn is read completely; the turn's own earlier input counts too."""
    store = SqlAlchemyTurnStore(db_uri)
    thread = _thread(db_uri)
    store.record_input(thread, "item_old", kind=InputKind.PERSON, person="dave", accepted_at=1)
    store.record_input(thread, "item_alice", kind=InputKind.PERSON, person="alice", accepted_at=2)
    store.register_turn(thread, "turn_1", runner_id=None, input_ids=["item_alice"])
    turn_start = store.latest_turn(thread).started_at
    store.record_input(
        thread, "item_bob", kind=InputKind.PERSON, person="bob", accepted_at=turn_start + 1
    )
    for index in range(700):
        store.record_input(
            thread,
            f"item_agent_{index}",
            kind=InputKind.AGENT,
            person="alice",
            accepted_at=turn_start + 2 + index,
        )
    store.record_terminal_activity(thread, "terminal_s1", "admin", at=turn_start + 800)

    authority = store.turn_authority(thread)

    assert authority.initiator == "alice"
    assert authority.people == {"alice", "bob", "admin"}
    assert not authority.unresolved


def test_without_a_registered_turn_nobody_is_named(db_uri: str) -> None:
    """No window to read; the thread's history is not scanned (D59: best effort)."""
    store = SqlAlchemyTurnStore(db_uri)
    thread = _thread(db_uri)
    store.record_input(thread, "item_1", kind=InputKind.PERSON, person="alice", accepted_at=1)
    store.record_input(thread, "pending_2", kind=InputKind.PERSON, person="bob", accepted_at=2)

    authority = store.turn_authority(thread)

    assert (authority.initiator, authority.turn_id, authority.people) == (None, None, frozenset())


def test_ambiguous_terminal_typing_counts_every_possible_typist(db_uri: str) -> None:
    store = SqlAlchemyTurnStore(db_uri)
    thread = _thread(db_uri)
    store.register_turn(thread, "turn_1", runner_id=None, input_ids=[])
    store.record_input(
        thread, "item_typed", kind=InputKind.TERMINAL, person=None, possible=["alice", "admin"]
    )

    assert store.turn_authority(thread).people == {"alice", "admin"}


def test_terminal_typists_are_recorded_per_terminal_and_person(db_uri: str) -> None:
    store = SqlAlchemyTurnStore(db_uri)
    thread = _thread(db_uri)
    store.record_terminal_activity(thread, "terminal_1", "alice", at=10)
    store.record_terminal_activity(thread, "terminal_2", "admin", at=20)
    store.record_terminal_activity(thread, "terminal_1", "alice", at=30)

    assert store.typists_since(thread, 15) == ["admin", "alice"]
    assert store.typists_since(thread, 25) == ["alice"]
    # After a typed prompt is recorded, only later typing names its typist.
    store.record_input(
        thread, "item_typed", kind=InputKind.TERMINAL, person="alice", accepted_at=35
    )
    store.record_terminal_activity(thread, "terminal_2", "admin", at=40)
    assert store.typists_for_prompt(thread) == ["admin"]


def test_a_queued_input_is_claimed_for_its_transcript_record_by_text(db_uri: str) -> None:
    """R2: after a restart the durable record, not memory, names the sender."""
    store = SqlAlchemyTurnStore(db_uri)
    thread = _thread(db_uri)
    store.record_input(
        thread,
        "pending_1",
        kind=InputKind.PERSON,
        person="bob",
        digest=text_digest("deploy  the\nservice"),
    )

    claimed, same_text = store.claim_unmapped_input(
        thread, text_digest("deploy the service"), "claim:a"
    )
    assert claimed is not None and claimed.person == "bob" and same_text == {"bob"}
    store.settle_claim(thread, "pending_1", "claim:a", "item_9")
    assert store.get_input(thread, "pending_1").item_id == "item_9"
    assert store.claim_unmapped_input(thread, text_digest("deploy the service"), "claim:b") == (
        None,
        frozenset(),
    )


def test_equal_text_from_two_people_is_claimed_once_each_and_marked_ambiguous(
    db_uri: str,
) -> None:
    """F7: two replicas claiming "continue" never take the same input; nor say who sent it."""
    store = SqlAlchemyTurnStore(db_uri)
    thread = _thread(db_uri)
    digest = text_digest("continue")
    store.record_input(thread, "pending_a", kind=InputKind.PERSON, person="alice", digest=digest)
    store.record_input(thread, "pending_b", kind=InputKind.PERSON, person="bob", digest=digest)

    first, first_people = store.claim_unmapped_input(thread, digest, "claim:replica_1")
    second, second_people = store.claim_unmapped_input(thread, digest, "claim:replica_2")

    assert {first.input_id, second.input_id} == {"pending_a", "pending_b"}
    assert first_people == {"alice", "bob"}
    assert second_people == {second.person}
    # A save that failed releases its claim for a later record.
    store.settle_claim(thread, second.input_id, "claim:replica_2", None)
    third, _people = store.claim_unmapped_input(thread, digest, "claim:replica_3")
    assert third.input_id == second.input_id


def test_an_undelivered_input_never_matches_a_later_equal_text(db_uri: str) -> None:
    """F7: the first delivery failed; Bob's identical later message is Bob's alone."""
    store = SqlAlchemyTurnStore(db_uri)
    thread = _thread(db_uri)
    digest = text_digest("continue")
    store.record_input(thread, "pending_a", kind=InputKind.PERSON, person="alice", digest=digest)
    store.retire_text_match(thread, "pending_a")
    store.record_input(thread, "pending_b", kind=InputKind.PERSON, person="bob", digest=digest)

    claimed, same_text = store.claim_unmapped_input(thread, digest, "claim:x")

    assert (claimed.input_id, same_text) == ("pending_b", {"bob"})
    # Alice's record stays: she did send it, it just never ran.
    assert store.get_input(thread, "pending_a").person == "alice"


def test_a_threads_inputs_and_turns_are_deleted_with_it(db_uri: str) -> None:
    conversations = SqlAlchemyConversationStore(db_uri)
    store = SqlAlchemyTurnStore(db_uri)
    thread = conversations.create_conversation().id
    other = conversations.create_conversation().id
    for session in (thread, other):
        store.record_input(session, "item_1", kind=InputKind.PERSON, person="bob")
        store.register_turn(
            session, f"turn_{uuid.uuid4().hex}", runner_id=None, input_ids=["item_1"]
        )

    assert asyncio.run(conversations.delete_conversation(thread))

    assert store.get_input(thread, "item_1") is None and store.latest_turn(thread) is None
    assert store.get_input(other, "item_1") is not None
