"""A dead runner's interrupted work keeps a visible, named, unknown outcome."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Iterator

import httpx
import pytest

from omnigent.db.utils import generate_agent_id
from omnigent.entities import (
    FunctionCallData,
    FunctionCallOutputData,
    MessageData,
    NewConversationItem,
)
from omnigent.runner.session_init_protocol import RUNNER_INCARNATION_LABEL_KEY
from omnigent.server.interrupted_work import (
    INTERRUPTED_OUTCOME_KNOWN_CODE,
    INTERRUPTED_OUTCOME_UNKNOWN_CODE,
    keep_interrupted_outcome,
    record_runner_incarnation,
    settle_late_tool_result,
    unanswered_tool_call,
)
from omnigent.server.routes._sessions.helpers import (
    _last_task_error_from_labels,
    _session_status_cache,
    reconcile_orphaned_running_status,
)
from omnigent.stores.agent_store.sqlalchemy_store import SqlAlchemyAgentStore
from omnigent.stores.conversation_store.sqlalchemy_store import SqlAlchemyConversationStore


@pytest.fixture
def store(db_uri: str) -> SqlAlchemyConversationStore:
    return SqlAlchemyConversationStore(db_uri)


@pytest.fixture
def session_id(db_uri: str, store: SqlAlchemyConversationStore) -> Iterator[str]:
    agent_id = generate_agent_id()
    SqlAlchemyAgentStore(db_uri).create(
        agent_id, name=f"agent-{agent_id}", bundle_location="test:///bundle"
    )
    conversation = store.create_conversation(agent_id=agent_id)
    assert store.set_runner_id(conversation.id, "runner_dead")
    yield conversation.id
    _session_status_cache.pop(conversation.id, None)


def _append(store: SqlAlchemyConversationStore, session_id: str, *items: object) -> list[str]:
    persisted = store.append(
        session_id,
        [
            NewConversationItem(type=kind, response_id="resp_1", data=data)  # type: ignore[arg-type]
            for kind, data in items  # type: ignore[misc]
        ],
    )
    return [item.id for item in persisted]


def _user(text: str) -> tuple[str, MessageData]:
    return "message", MessageData(role="user", content=[{"type": "input_text", "text": text}])


def _call(call_id: str, name: str) -> tuple[str, FunctionCallData]:
    return "function_call", FunctionCallData(agent="a", name=name, arguments="{}", call_id=call_id)


def _output(call_id: str) -> tuple[str, FunctionCallOutputData]:
    return "function_call_output", FunctionCallOutputData(call_id=call_id, output="done")


def _labels(store: SqlAlchemyConversationStore, session_id: str) -> dict[str, str]:
    conversation = store.get_conversation(session_id)
    assert conversation is not None
    return conversation.labels


def _init_answer(**fields: object) -> httpx.Response:
    return httpx.Response(201, json=fields, request=httpx.Request("POST", "http://runner"))


@pytest.mark.asyncio
async def test_unanswered_tool_call_is_the_current_turns_call_without_a_result(
    store: SqlAlchemyConversationStore, session_id: str
) -> None:
    _append(store, session_id, _user("old"), _call("call_old", "Read"))
    _append(store, session_id, _user("new turn"), _call("call_1", "Read"), _output("call_1"))
    assert await unanswered_tool_call(store, session_id) is None, (
        "a call from an earlier turn is not this turn's"
    )
    (effect_id,) = _append(store, session_id, _call("call_2", "Bash"))

    assert await unanswered_tool_call(store, session_id) == (effect_id, "Bash")


@pytest.mark.asyncio
async def test_runner_incarnation_is_recorded_once(
    store: SqlAlchemyConversationStore, session_id: str
) -> None:
    conversation = store.get_conversation(session_id)
    assert conversation is not None

    await record_runner_incarnation(conversation, _init_answer(), store)
    assert RUNNER_INCARNATION_LABEL_KEY not in _labels(store, session_id)
    answer = _init_answer(runner_incarnation_id="b" * 32)
    await record_runner_incarnation(conversation, answer, store)

    assert _labels(store, session_id)[RUNNER_INCARNATION_LABEL_KEY] == "b" * 32


@pytest.mark.asyncio
async def test_replacement_that_declined_to_continue_keeps_the_unknown_outcome(
    store: SqlAlchemyConversationStore, session_id: str
) -> None:
    (effect_id,) = _append(store, session_id, _user("append a line"), _call("call_1", "Bash"))[1:]
    _session_status_cache[session_id] = "running"

    kept = await keep_interrupted_outcome(
        session_id, _init_answer(continuation_blocked=True), store, runner_id="runner_new"
    )

    assert kept is True
    assert _session_status_cache[session_id] == "failed"
    error = _last_task_error_from_labels(_labels(store, session_id))
    assert error is not None
    assert error["code"] == INTERRUPTED_OUTCOME_UNKNOWN_CODE
    assert "Bash" in error["message"]
    assert error["item_id"] == effect_id


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "answer"),
    [("idle", {"continuation_blocked": True}), ("running", {"continuation_blocked": False})],
    ids=["nothing-interrupted", "same-runner"],
)
async def test_unknown_outcome_only_for_interrupted_work_of_another_runner(
    store: SqlAlchemyConversationStore,
    session_id: str,
    status: str,
    answer: dict[str, object],
) -> None:
    _session_status_cache[session_id] = status

    assert not await keep_interrupted_outcome(
        session_id, _init_answer(**answer), store, runner_id="runner_new"
    )
    assert _last_task_error_from_labels(store.get_conversation(session_id).labels) is None  # type: ignore[union-attr]


def test_list_read_settles_an_orphaned_turn_as_interrupted_not_idle(
    store: SqlAlchemyConversationStore, session_id: str
) -> None:
    """A passive read must not hide that the dead runner's turn was interrupted."""
    _append(store, session_id, _user("append a line"), _call("call_1", "Bash"))
    store.set_session_live_status(session_id, "running")
    _session_status_cache.pop(session_id, None)

    assert reconcile_orphaned_running_status(
        session_id, store, int(time.time()) - 90, as_interrupted=True
    )

    assert _session_status_cache[session_id] == "failed"
    error = _last_task_error_from_labels(_labels(store, session_id))
    assert error is not None and error["code"] == "runner_disconnected"
    assert "Bash was running and its outcome is unknown" in error["message"]


@pytest.mark.asyncio
async def test_an_unknown_outcome_already_recorded_is_kept_on_every_later_reattach(
    store: SqlAlchemyConversationStore, session_id: str
) -> None:
    """The second reconnect of the replacement must not clear what the first recorded."""
    _append(store, session_id, _user("append a line"), _call("call_1", "Bash"))
    _session_status_cache[session_id] = "running"
    first = _init_answer(continuation_blocked=True, runner_incarnation_id="b" * 32)
    assert await keep_interrupted_outcome(session_id, first, store, runner_id="runner_new")

    second = _init_answer(continuation_blocked=False, runner_incarnation_id="b" * 32)
    assert await keep_interrupted_outcome(session_id, second, store, runner_id="runner_new")
    error = _last_task_error_from_labels(_labels(store, session_id))
    assert error is not None and error["code"] == INTERRUPTED_OUTCOME_UNKNOWN_CODE


@pytest.mark.asyncio
async def test_a_late_result_settles_the_unknown_outcome_for_good(
    store: SqlAlchemyConversationStore, session_id: str
) -> None:
    """The call's late result makes the outcome known, still interrupted, never unknown again."""
    (effect_id,) = _append(store, session_id, _user("append a line"), _call("call_1", "Bash"))[1:]
    _session_status_cache[session_id] = "running"
    answer = _init_answer(continuation_blocked=True, runner_incarnation_id="b" * 32)
    assert await keep_interrupted_outcome(session_id, answer, store, runner_id="runner_new")

    _append(store, session_id, _output("call_1"))
    await settle_late_tool_result(session_id, store)

    assert _session_status_cache[session_id] == "failed"
    error = _last_task_error_from_labels(_labels(store, session_id))
    assert error is not None
    assert error["code"] == INTERRUPTED_OUTCOME_KNOWN_CODE
    assert "Bash" in error["message"] and "not continued" in error["message"]
    assert error["item_id"] == effect_id
    # Every later re-attach keeps it, and nothing turns it back into unknown.
    assert await keep_interrupted_outcome(session_id, answer, store, runner_id="runner_new")
    await settle_late_tool_result(session_id, store)
    error = _last_task_error_from_labels(_labels(store, session_id))
    assert error is not None and error["code"] == INTERRUPTED_OUTCOME_KNOWN_CODE


@pytest.mark.asyncio
async def test_a_late_result_of_one_call_names_the_call_still_without_one(
    store: SqlAlchemyConversationStore, session_id: str
) -> None:
    """With two calls left unanswered, the outcome stays unknown for the other."""
    first_id, _second_id = _append(
        store, session_id, _user("two effects"), _call("call_1", "Read"), _call("call_2", "Bash")
    )[1:]
    _session_status_cache[session_id] = "running"
    answer = _init_answer(continuation_blocked=True)
    assert await keep_interrupted_outcome(session_id, answer, store, runner_id="runner_new")

    _append(store, session_id, _output("call_2"))
    await settle_late_tool_result(session_id, store)

    error = _last_task_error_from_labels(_labels(store, session_id))
    assert error is not None
    assert error["code"] == INTERRUPTED_OUTCOME_UNKNOWN_CODE
    assert "Read" in error["message"]
    assert error["item_id"] == first_id


@pytest.mark.asyncio
async def test_a_result_after_the_person_acted_changes_nothing(
    store: SqlAlchemyConversationStore, session_id: str
) -> None:
    """Only a recorded unknown outcome is settled; a cleared one stays cleared."""
    _append(store, session_id, _user("append a line"), _call("call_1", "Bash"))
    _append(store, session_id, _output("call_1"))
    await settle_late_tool_result(session_id, store)
    assert _last_task_error_from_labels(_labels(store, session_id)) is None


async def _record_unknown_outcome(store: SqlAlchemyConversationStore, session_id: str) -> str:
    (effect_id,) = _append(store, session_id, _user("append a line"), _call("call_1", "Bash"))[1:]
    _session_status_cache[session_id] = "running"
    answer = _init_answer(continuation_blocked=True)
    assert await keep_interrupted_outcome(session_id, answer, store, runner_id="runner_new")
    return effect_id


@pytest.mark.asyncio
@pytest.mark.parametrize("evidence", ["unreadable", "another call's result", "incomplete scan"])
async def test_no_positive_evidence_keeps_the_outcome_unknown(
    store: SqlAlchemyConversationStore,
    session_id: str,
    monkeypatch: pytest.MonkeyPatch,
    evidence: str,
) -> None:
    """Only the recorded call's own result, found in a complete scan, settles the record."""
    from omnigent.server import interrupted_work

    await _record_unknown_outcome(store, session_id)
    if evidence == "another call's result":
        _append(store, session_id, _output("call_other"))
    else:
        _append(store, session_id, _output("call_1"))
    if evidence == "unreadable":

        def unreadable(*_args: object, **_kwargs: object) -> object:
            raise OSError("database is locked")

        monkeypatch.setattr(store, "list_items", unreadable)
    elif evidence == "incomplete scan":
        _append(store, session_id, _call("call_2", "Read"), _output("call_2"))
        monkeypatch.setattr(interrupted_work, "_TOOL_CALL_LOOKBACK_ITEMS", 2)

    await settle_late_tool_result(session_id, store)

    monkeypatch.undo()
    error = _last_task_error_from_labels(_labels(store, session_id))
    assert error is not None and error["code"] == INTERRUPTED_OUTCOME_UNKNOWN_CODE


@pytest.mark.asyncio
async def test_a_record_cleared_during_settlement_is_not_resurrected(
    store: SqlAlchemyConversationStore, session_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The person's message clears the record while the settlement reads the transcript."""
    from omnigent.server import interrupted_work
    from omnigent.server.routes._sessions.helpers import _persist_session_status_error_labels

    await _record_unknown_outcome(store, session_id)
    _append(store, session_id, _output("call_1"))
    scan = interrupted_work._scan_current_turn

    def scan_while_the_person_acts(
        conversation_store: SqlAlchemyConversationStore, sid: str
    ) -> object:
        asyncio.run(_persist_session_status_error_labels(sid, None, conversation_store))
        return scan(conversation_store, sid)

    monkeypatch.setattr(interrupted_work, "_scan_current_turn", scan_while_the_person_acts)
    _session_status_cache[session_id] = "idle"
    await settle_late_tool_result(session_id, store)

    assert _last_task_error_from_labels(_labels(store, session_id)) is None
    assert _session_status_cache[session_id] == "idle"
