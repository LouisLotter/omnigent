"""A re-parked harness elicitation keeps the first park's deadline and popup.

Host-side long-polls re-post their stable elicitation id whenever a poll goes
unanswered for one period, so a half-open connection costs one period instead
of hanging the turn. Each re-post re-parks the same question on the server.
The question must still time out when its first park's ``ask_timeout`` runs
out, and the native-terminal popup must not be raised again on every re-post.
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from typing import Any

import pytest

from omnigent.server import _elicitation_registry
from omnigent.server.routes import sessions as S
from omnigent.server.routes._sessions import orchestration
from omnigent.server.schemas import ElicitationRequestParams, ElicitationResult
from omnigent.spec.types import Phase


class _Request:
    """Request whose client hangs up when ``hang_up`` is set."""

    def __init__(self) -> None:
        self.hang_up = asyncio.Event()

    async def receive(self) -> dict[str, str]:
        await self.hang_up.wait()
        return {"type": "http.disconnect"}


def _params() -> ElicitationRequestParams:
    return ElicitationRequestParams(mode="form", message="Run `date`?", requestedSchema={})


@pytest.fixture(autouse=True)
def _clean_registries() -> None:
    _elicitation_registry.reset_for_tests()
    yield
    _elicitation_registry.reset_for_tests()


async def _park(eid: str, timeout_s: float, *, hang_up_after: float | None = None) -> Any:
    request = _Request()
    if hang_up_after is not None:
        asyncio.get_running_loop().call_later(hang_up_after, request.hang_up.set)
    return await S._publish_and_wait_for_harness_elicitation(
        request,  # type: ignore[arg-type]
        session_id="conv_repark",
        params=_params(),
        timeout_s=timeout_s,
        elicitation_id=eid,
    )


@pytest.mark.asyncio
async def test_repark_waits_only_for_what_is_left_of_the_first_deadline() -> None:
    eid = "elicit_evaluate_11111111111111111111111111111111"
    assert await _park(eid, 0.6, hang_up_after=0.2) is None

    started = time.monotonic()
    assert await _park(eid, 0.6) is None
    waited = time.monotonic() - started

    assert 0.2 < waited < 0.55, "a re-post must not restart the question's timeout"
    # The expired deadline stays, so a later re-post is answered at once.
    assert eid in _elicitation_registry._harness_elicitation_deadlines


@pytest.mark.asyncio
async def test_repark_after_the_deadline_ends_at_once() -> None:
    eid = "elicit_evaluate_22222222222222222222222222222222"
    assert await _park(eid, 0.2, hang_up_after=0.05) is None
    await asyncio.sleep(0.25)

    started = time.monotonic()
    assert await _park(eid, 5.0) is None
    assert time.monotonic() - started < 0.1


@pytest.mark.asyncio
async def test_a_verdict_settles_the_deadline_so_the_id_can_be_asked_afresh() -> None:
    eid = "elicit_evaluate_33333333333333333333333333333333"
    waiter = asyncio.create_task(_park(eid, 5.0))
    while eid not in _elicitation_registry._harness_elicitation_registry:
        await asyncio.sleep(0)
    _elicitation_registry._harness_elicitation_registry[eid].set_result(
        ElicitationResult(action="accept")
    )

    verdict = await asyncio.wait_for(waiter, timeout=2.0)

    assert verdict is not None and verdict.action == "accept"
    assert eid not in _elicitation_registry._harness_elicitation_deadlines


@pytest.mark.asyncio
async def test_native_popup_is_raised_once_per_question(monkeypatch: pytest.MonkeyPatch) -> None:
    popups: list[str] = []
    monkeypatch.setattr(
        orchestration,
        "_spawn_native_approval_popup_forward",
        lambda session_id, elicitation_id, message, policy_name=None: popups.append(
            elicitation_id
        ),
    )
    monkeypatch.setattr(orchestration, "resolve_ask_timeout", lambda engine, result: 5.0)
    eid = "elicit_evaluate_44444444444444444444444444444444"
    result = SimpleNamespace(
        reason="Approve?", deciding_policy="ask-bash", set_labels=None, state_updates=None
    )

    async def ask(hang_up_after: float) -> bool:
        request = _Request()
        asyncio.get_running_loop().call_later(hang_up_after, request.hang_up.set)
        return await S._hold_native_ask_gate(
            request,  # type: ignore[arg-type]
            session_id="conv_repark",
            phase=Phase.TOOL_CALL,
            data={"name": "Bash", "arguments": {"command": "date"}},
            engine=SimpleNamespace(),  # type: ignore[arg-type]
            result=result,  # type: ignore[arg-type]
            conversation_store=None,  # type: ignore[arg-type]
            elicitation_id=eid,
        )

    assert await ask(0.05) is False
    assert await ask(0.05) is False

    assert popups == [eid]


@pytest.mark.asyncio
async def test_repost_after_the_question_timed_out_is_answered_at_once() -> None:
    """A waiter that reached its timeout leaves the timeout for its re-posts."""
    eid = "elicit_evaluate_55555555555555555555555555555555"
    assert await _park(eid, 0.2) is None

    started = time.monotonic()
    assert await _park(eid, 5.0) is None
    assert time.monotonic() - started < 0.1


@pytest.mark.asyncio
async def test_repost_of_an_answered_question_reuses_its_outcome_without_reapplying(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The approval's writes and the popup happen once, however often the poll is re-posted."""
    popups: list[str] = []
    monkeypatch.setattr(
        orchestration,
        "_spawn_native_approval_popup_forward",
        lambda session_id, elicitation_id, message, policy_name=None: popups.append(
            elicitation_id
        ),
    )
    monkeypatch.setattr(orchestration, "resolve_ask_timeout", lambda engine, result: 5.0)
    applied: list[object] = []
    engine = SimpleNamespace(apply_label_writes=applied.append, apply_state_updates=applied.append)
    result = SimpleNamespace(
        reason="Approve?",
        deciding_policy="ask-bash",
        set_labels={"approved": "1"},
        state_updates=["checkpoint"],
    )
    eid = "elicit_evaluate_66666666666666666666666666666666"

    async def ask() -> bool:
        return await S._hold_native_ask_gate(
            _Request(),  # type: ignore[arg-type]
            session_id="conv_repark",
            phase=Phase.TOOL_CALL,
            data={"name": "Bash", "arguments": {"command": "date"}},
            engine=engine,  # type: ignore[arg-type]
            result=result,  # type: ignore[arg-type]
            conversation_store=None,  # type: ignore[arg-type]
            elicitation_id=eid,
        )

    first = asyncio.create_task(ask())
    while eid not in _elicitation_registry._harness_elicitation_registry:
        await asyncio.sleep(0)
    _elicitation_registry._harness_elicitation_registry[eid].set_result(
        ElicitationResult(action="accept")
    )
    assert await asyncio.wait_for(first, timeout=2.0) is True

    assert await asyncio.wait_for(ask(), timeout=2.0) is True
    assert applied == [{"approved": "1"}, ["checkpoint"]], "writes applied once"
    assert popups == [eid]


@pytest.mark.asyncio
async def test_remembered_outcome_matches_the_whole_request_and_only_after_its_writes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A different command reusing the id asks again; a failed write is never remembered."""
    monkeypatch.setattr(
        orchestration, "_spawn_native_approval_popup_forward", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(orchestration, "resolve_ask_timeout", lambda engine, result: 5.0)

    def fail(_updates: object) -> None:
        raise RuntimeError("state write failed")

    engine = SimpleNamespace(apply_label_writes=lambda labels: None, apply_state_updates=fail)
    result = SimpleNamespace(
        reason="Approve?", deciding_policy="ask-bash", set_labels=None, state_updates=["x"]
    )
    eid = "elicit_evaluate_77777777777777777777777777777777"
    long_prefix = "echo " + "a" * 2000
    data = {"name": "Bash", "arguments": {"command": long_prefix + " one"}}

    task = asyncio.create_task(
        S._hold_native_ask_gate(
            _Request(),  # type: ignore[arg-type]
            session_id="conv_repark",
            phase=Phase.TOOL_CALL,
            data=data,
            engine=engine,  # type: ignore[arg-type]
            result=result,  # type: ignore[arg-type]
            conversation_store=None,  # type: ignore[arg-type]
            elicitation_id=eid,
        )
    )
    while eid not in _elicitation_registry._harness_elicitation_registry:
        await asyncio.sleep(0)
    _elicitation_registry._harness_elicitation_registry[eid].set_result(
        ElicitationResult(action="accept")
    )
    with pytest.raises(RuntimeError, match="state write failed"):
        await asyncio.wait_for(task, timeout=2.0)

    assert orchestration.answered_ask_verdict("conv_repark", eid, Phase.TOOL_CALL, data) is None

    orchestration._remember_answered_harness_elicitation(
        "conv_repark",
        eid,
        orchestration.ask_request_fingerprint(Phase.TOOL_CALL, data),
        approved=True,
        declined_reason=None,
    )
    other = {"name": "Bash", "arguments": {"command": long_prefix + " two"}}
    assert orchestration.answered_ask_verdict("conv_repark", eid, Phase.TOOL_CALL, data)
    assert orchestration.answered_ask_verdict("conv_repark", eid, Phase.TOOL_CALL, other) is None
    assert orchestration.answered_ask_verdict("conv_other", eid, Phase.TOOL_CALL, data) is None
