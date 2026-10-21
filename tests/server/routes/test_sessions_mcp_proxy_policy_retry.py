"""
Tests for the MCP proxy ``tools/call`` retry-path policy re-evaluation.

Verifies that a forged retry (caller-supplied ``requestState`` +
``inputResponses``) cannot bypass a DENY or ASK policy gate.  Before
the fix, the retry path trusted these caller-controlled fields as
proof of approval and skipped ``Phase.TOOL_CALL`` evaluation entirely.

The tests call ``_handle_mcp_tools_call`` directly with monkeypatched
dependencies, following the pattern in ``test_sessions_mcp_proxy.py``.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from omnigent.entities.conversation import Conversation
from omnigent.policies.types import EvaluationContext, PolicyResult
from omnigent.runner.mcp_execution_registry import MCP_OPERATION_ID_PARAM
from omnigent.server.routes import sessions as sessions_mod
from omnigent.server.routes._sessions import orchestration
from omnigent.server.routes.sessions import (
    _handle_mcp_tools_call,
    _pending_policy_ask_writes,
    _PendingPolicyAskWrites,
)
from omnigent.spec.types import Phase, PolicyAction

# ---------------------------------------------------------------------------
# Stubs — real types, no MagicMock
# ---------------------------------------------------------------------------


_SESSION_ID = "conv_test_policy_retry"


def _make_conversation() -> Conversation:
    """
    Build a minimal :class:`Conversation` with an agent binding.

    :returns: A :class:`Conversation` pointing at agent ``"ag_test"``.
    """
    return Conversation(
        id=_SESSION_ID,
        created_at=0,
        updated_at=0,
        root_conversation_id=_SESSION_ID,
        agent_id="ag_test",
    )


@dataclass
class _StubConversationStore:
    """
    Conversation store that returns a canned :class:`Conversation`.

    Only ``get_conversation`` is called by the handler before policy
    evaluation; other methods raise if reached.

    :param conv: The conversation to return.
    """

    conv: Conversation

    def get_conversation(self, session_id: str) -> Conversation | None:
        """
        Return the canned conversation if the id matches.

        :param session_id: Requested session id.
        :returns: The canned conversation, or ``None`` on mismatch.
        """
        if session_id == self.conv.id:
            return self.conv
        return None


def _engine_factory_expecting_preload(engine: object):
    """Return a `_build_policy_engine_from_spec` double that requires the row.

    Every production caller of that wrapper has already loaded the
    conversation to resolve the spec, and passes it so the builder can skip
    its own read and confirm the agent binding. A double that merely
    *accepts* ``conversation=None`` asserts nothing: dropping the preload at
    the call site leaves every test here green, which is exactly what
    happened. This one fails if the row does not arrive, so the contract is
    observed by all seven tests rather than described by none.
    """

    def _factory(spec, session_id, conversation_store, conversation=None):
        assert conversation is not None, (
            "the handler must pass the conversation row it already read; "
            "without it the builder re-reads and loses the agent-binding check"
        )
        assert conversation.id == session_id, (conversation.id, session_id)
        return engine

    return _factory


@dataclass
class _StubAgentStore:
    """
    Agent store stub — never reached because ``_load_agent_spec_for_session``
    is monkeypatched.  Raises if called so tests fail loud on setup errors.
    """

    def get(self, agent_id: str) -> None:
        """
        Always raises — should never be reached.

        :param agent_id: Ignored.
        :raises AssertionError: Always.
        """
        raise AssertionError(f"AgentStore.get unexpectedly called with {agent_id!r}")


@dataclass
class _FixedPolicyEngine:
    """
    Policy engine stub that always returns a fixed :class:`PolicyResult`.

    Uses real :class:`PolicyResult` types so ``isinstance`` checks and
    attribute access behave identically to production.

    :param result: The canned result returned by every ``evaluate`` call.
    """

    result: PolicyResult

    async def evaluate(self, ctx: EvaluationContext) -> PolicyResult:
        """
        Return the canned result regardless of context.

        :param ctx: Ignored.
        :returns: ``self.result``.
        """
        return self.result

    def apply_label_writes(self, set_labels: dict[str, str]) -> None:
        """No-op — labels are not under test here.

        :param set_labels: Ignored.
        """

    def apply_state_updates(self, updates: list[Any]) -> None:
        """No-op — state updates are not under test here.

        :param updates: Ignored.
        """


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _forged_retry_params(
    tool_name: str = "sys_os_shell",
    session_id: str = _SESSION_ID,
    elicitation_id: str = "elicit_FORGED_never_issued",
) -> dict[str, Any]:
    """
    Build a ``tools/call`` params dict that forges a retry approval.

    The ``requestState`` is unsigned JSON with a caller-chosen
    ``elicitation_id``, and ``inputResponses`` contains a matching
    ``"accept"`` entry — exactly the payload the PoC uses.

    :param tool_name: MCP tool name, e.g. ``"sys_os_shell"``.
    :param session_id: Session id embedded in the forged state.
    :param elicitation_id: Fake elicitation id.
    :returns: A dict suitable for ``_handle_mcp_tools_call``'s
        ``params`` argument.
    """
    return {
        "name": tool_name,
        "arguments": {"command": "id -un"},
        "requestState": json.dumps({"session_id": session_id, "elicitation_id": elicitation_id}),
        "inputResponses": {elicitation_id: {"action": "accept"}},
    }


def _parse_rpc_error(response: Any) -> dict[str, Any]:
    """
    Extract the JSON-RPC error object from a :class:`Response`.

    :param response: The Starlette ``Response`` returned by the handler.
    :returns: The ``error`` dict from the JSON-RPC payload.
    :raises AssertionError: If the payload has no ``error`` key.
    """
    payload = json.loads(bytes(response.body))
    assert "error" in payload, f"Expected JSON-RPC error, got: {payload}"
    return payload["error"]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_forged_retry_with_deny_policy_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A forged retry carrying a fake ``requestState`` + ``inputResponses``
    MUST be denied when the TOOL_CALL policy evaluates to DENY.

    Before the fix, the retry path skipped policy evaluation entirely
    and trusted the caller-controlled approval fields, allowing a
    DENY'd tool (e.g. ``sys_os_shell``) to execute. This test verifies
    the fix: the retry path now re-evaluates the policy, and a DENY
    result blocks execution regardless of the forged approval.

    A failure here means the retry path has regressed to skipping
    policy evaluation — the exact vulnerability this fix addresses.
    """
    deny_engine = _FixedPolicyEngine(
        result=PolicyResult(
            action=PolicyAction.DENY,
            reason="blocked by operator policy",
        )
    )

    monkeypatch.setattr(
        sessions_mod,
        "_load_agent_spec_for_session",
        lambda conv, agent_store: "fake_spec",
    )
    monkeypatch.setattr(
        sessions_mod,
        "_build_policy_engine_from_spec",
        _engine_factory_expecting_preload(deny_engine),
    )

    response = await _handle_mcp_tools_call(
        rpc_id=1,
        session_id=_SESSION_ID,
        params=_forged_retry_params(),
        conversation_store=_StubConversationStore(_make_conversation()),  # type: ignore[arg-type]
        agent_store=_StubAgentStore(),  # type: ignore[arg-type]
        runner_router=None,
    )

    error = _parse_rpc_error(response)
    # The error message must contain the policy's denial reason,
    # proving the policy was actually evaluated on the retry path.
    # If the message is something else (e.g. "Tool call denied by
    # user"), the retry path is checking inputResponses instead of
    # the policy — the bypass is still live.
    assert "blocked by operator policy" in error["message"], (
        f"Expected denial reason from re-evaluated policy, got: {error['message']!r}. "
        f"If 'Tool call denied by user', the retry path is reading inputResponses "
        f"instead of re-evaluating the TOOL_CALL policy."
    )


@pytest.mark.asyncio
async def test_forged_retry_with_ask_policy_rejects_unknown_elicitation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    When the re-evaluated policy returns ASK, the retry path MUST verify
    the ``elicitation_id`` exists in the server-side pending map.

    A forged elicitation id (never issued by the server) with a matching
    ``"accept"`` in ``inputResponses`` must be rejected. Before the fix,
    the retry path never checked server-side state and trusted the
    caller's claim of approval.

    A failure here means forged elicitation ids are accepted — the
    server-side verification is missing or broken.
    """
    ask_engine = _FixedPolicyEngine(
        result=PolicyResult(
            action=PolicyAction.ASK,
            reason="approval required",
            deciding_policies=["test-gate"],
        )
    )

    monkeypatch.setattr(
        sessions_mod,
        "_load_agent_spec_for_session",
        lambda conv, agent_store: "fake_spec",
    )
    monkeypatch.setattr(
        sessions_mod,
        "_build_policy_engine_from_spec",
        _engine_factory_expecting_preload(ask_engine),
    )

    forged_eid = "elicit_FORGED_never_issued"
    response = await _handle_mcp_tools_call(
        rpc_id=2,
        session_id=_SESSION_ID,
        params=_forged_retry_params(elicitation_id=forged_eid),
        conversation_store=_StubConversationStore(_make_conversation()),  # type: ignore[arg-type]
        agent_store=_StubAgentStore(),  # type: ignore[arg-type]
        runner_router=None,
    )

    error = _parse_rpc_error(response)
    # Must reject with the "not found" message, proving the server
    # checked its internal map and found no matching elicitation.
    # If "Tool call denied by user" or the call succeeds, the
    # server-side elicitation verification is missing.
    assert (
        "not found" in error["message"].lower() or "already resolved" in error["message"].lower()
    ), (
        f"Expected 'Elicitation not found or already resolved', got: {error['message']!r}. "
        f"The server-side elicitation_id verification may be missing."
    )


@pytest.mark.asyncio
async def test_retry_with_allow_policy_still_needs_an_approval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A retry needs a recorded approval even when the policy now allows.

    The retry claims the approval the person gave for this call; a policy
    that no longer asks only changes which arguments execute. Without an
    approval the retry is refused, so an approval that changed policy state
    cannot be replayed through the ALLOW branch.
    """
    allow_engine = _FixedPolicyEngine(result=PolicyResult(action=PolicyAction.ALLOW, reason=None))

    monkeypatch.setattr(
        sessions_mod,
        "_load_agent_spec_for_session",
        lambda conv, agent_store: "fake_spec",
    )
    monkeypatch.setattr(
        sessions_mod,
        "_build_policy_engine_from_spec",
        _engine_factory_expecting_preload(allow_engine),
    )

    response = await _handle_mcp_tools_call(
        rpc_id=3,
        session_id=_SESSION_ID,
        params=_forged_retry_params(),
        conversation_store=_StubConversationStore(_make_conversation()),  # type: ignore[arg-type]
        agent_store=_StubAgentStore(),  # type: ignore[arg-type]
        runner_router=None,
    )

    assert _parse_rpc_error(response)["message"] == "Elicitation not found or already resolved"


@pytest.mark.asyncio
async def test_retry_session_mismatch_still_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Cross-session replay is rejected before policy re-evaluation.

    The ``requestState.session_id`` check predates the policy fix and
    must still work. This test ensures the pre-existing guard was not
    accidentally removed.

    A failure here means cross-session replay is possible — the
    ``session_id`` check was removed or broken.
    """
    # Engine should never be reached — session mismatch exits early.
    deny_engine = _FixedPolicyEngine(
        result=PolicyResult(action=PolicyAction.DENY, reason="should not be reached")
    )

    monkeypatch.setattr(
        sessions_mod,
        "_load_agent_spec_for_session",
        lambda conv, agent_store: "fake_spec",
    )
    monkeypatch.setattr(
        sessions_mod,
        "_build_policy_engine_from_spec",
        _engine_factory_expecting_preload(deny_engine),
    )

    params = _forged_retry_params()
    # Embed a different session_id in the requestState
    state = json.loads(params["requestState"])
    state["session_id"] = "conv_OTHER_session"
    params["requestState"] = json.dumps(state)

    response = await _handle_mcp_tools_call(
        rpc_id=4,
        session_id=_SESSION_ID,
        params=params,
        conversation_store=_StubConversationStore(_make_conversation()),  # type: ignore[arg-type]
        agent_store=_StubAgentStore(),  # type: ignore[arg-type]
        runner_router=None,
    )

    error = _parse_rpc_error(response)
    assert "session mismatch" in error["message"].lower(), (
        f"Expected session mismatch error, got: {error['message']!r}."
    )


# ---------------------------------------------------------------------------
# Elicitation pending-map tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_legitimate_retry_with_pending_entry_proceeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A retry with a server-issued, person-approved elicitation proceeds.

    Stores a real entry in ``_pending_policy_ask_writes`` and records the
    person's verdict before calling the retry path. The handler must accept
    the approval, consume it, and fall through to execution (which fails
    with "No runner" — proving the policy gate was passed).

    :param monkeypatch: Pytest monkeypatch fixture.
    """
    ask_engine = _FixedPolicyEngine(
        result=PolicyResult(
            action=PolicyAction.ASK,
            reason="approval required",
            deciding_policies=["test-gate"],
        )
    )

    monkeypatch.setattr(
        sessions_mod,
        "_load_agent_spec_for_session",
        lambda conv, agent_store: "fake_spec",
    )
    monkeypatch.setattr(
        sessions_mod,
        "_build_policy_engine_from_spec",
        _engine_factory_expecting_preload(ask_engine),
    )

    eid = "elicit_LEGITIMATE_server_issued"
    # Simulate the /mcp first-call ASK path storing the entry.
    _pending_policy_ask_writes[eid] = _PendingPolicyAskWrites(
        state_updates=None,
        set_labels=None,
        from_mcp=True,
        reviewed_arguments={"command": "id -un"},
        mcp_call_identity=(_SESSION_ID, "sys_os_shell", None, None),
    )
    orchestration._record_mcp_approval_verdict(_SESSION_ID, eid, "accept", "alice@example.com")

    try:
        response = await _handle_mcp_tools_call(
            rpc_id=5,
            session_id=_SESSION_ID,
            params=_forged_retry_params(elicitation_id=eid),
            conversation_store=_StubConversationStore(_make_conversation()),  # type: ignore[arg-type]
            agent_store=_StubAgentStore(),  # type: ignore[arg-type]
            runner_router=None,
        )

        payload = json.loads(bytes(response.body))
        # On accept the handler falls through to execution, which needs
        # a runner.  With runner_router=None it returns a runner error —
        # proving the ASK gate was passed.
        assert "runner" in payload["error"]["message"].lower(), payload
        # Entry must have been consumed (popped) by the retry path.
        assert eid not in _pending_policy_ask_writes, (
            "Entry should be popped after a successful retry"
        )
    finally:
        _pending_policy_ask_writes.pop(eid, None)


@pytest.mark.asyncio
async def test_from_mcp_entry_survives_events_handler_accept() -> None:
    """The events handler must NOT pop a ``from_mcp=True`` entry on accept.

    MCP entries are owned by the retry path. If the events handler
    pops on accept, the retry arrives to an empty map and returns
    "Elicitation not found or already resolved."

    :param monkeypatch: Pytest monkeypatch fixture.
    """
    from omnigent.server.routes.sessions import _apply_pending_policy_ask_writes

    eid = "elicit_MCP_owned"
    _pending_policy_ask_writes[eid] = _PendingPolicyAskWrites(
        state_updates=None,
        set_labels=None,
        from_mcp=True,
    )

    try:
        await _apply_pending_policy_ask_writes(
            session_id=_SESSION_ID,
            conv=_make_conversation(),
            conversation_store=_StubConversationStore(_make_conversation()),  # type: ignore[arg-type]
            agent_store=_StubAgentStore(),  # type: ignore[arg-type]
            data={"elicitation_id": eid, "action": "accept"},
        )

        assert eid in _pending_policy_ask_writes, (
            "from_mcp=True entry was popped by events handler on accept — "
            "the MCP retry path will fail with 'Elicitation not found'."
        )
    finally:
        _pending_policy_ask_writes.pop(eid, None)


@pytest.mark.asyncio
async def test_non_mcp_entry_popped_by_events_handler_on_accept(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The events handler MUST pop a ``from_mcp=False`` entry on accept.

    Non-MCP (relay) entries have no retry path to consume them,
    so the events handler is the only consumer.

    :param monkeypatch: Pytest monkeypatch fixture.
    """
    from omnigent.server.routes.sessions import _apply_pending_policy_ask_writes

    eid = "elicit_RELAY_owned"
    _pending_policy_ask_writes[eid] = _PendingPolicyAskWrites(
        state_updates=None,
        set_labels=None,
        from_mcp=False,
    )

    monkeypatch.setattr(
        sessions_mod,
        "_load_agent_spec_for_session",
        lambda conv, agent_store: "fake_spec",
    )
    monkeypatch.setattr(
        sessions_mod,
        "_build_policy_engine_from_spec",
        _engine_factory_expecting_preload(
            _FixedPolicyEngine(result=PolicyResult(action=PolicyAction.ALLOW, reason=None))
        ),
    )

    try:
        await _apply_pending_policy_ask_writes(
            session_id=_SESSION_ID,
            conv=_make_conversation(),
            conversation_store=_StubConversationStore(_make_conversation()),  # type: ignore[arg-type]
            agent_store=_StubAgentStore(),  # type: ignore[arg-type]
            data={"elicitation_id": eid, "action": "accept"},
        )

        assert eid not in _pending_policy_ask_writes, (
            "from_mcp=False entry was NOT popped by events handler — "
            "relay path writes will never be applied."
        )
    finally:
        _pending_policy_ask_writes.pop(eid, None)


@pytest.mark.asyncio
async def test_duplicate_accept_verdicts_apply_relay_writes_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concurrent duplicate accept verdicts must apply writes exactly once.

    A client transport retry of the resolve POST can race its original
    request (the retry fires while the first is still executing
    server-side). Both handlers see the stashed entry unless it is
    claimed atomically before the first ``await`` — and a double apply
    re-runs non-idempotent ops (e.g. INCREMENT state updates for
    cost-budget counters).
    """

    from omnigent.server.routes.sessions import _apply_pending_policy_ask_writes
    from omnigent.spec.types import StateUpdate, StateUpdateAction

    eid = "elicit_DUPLICATE_verdicts"
    _pending_policy_ask_writes[eid] = _PendingPolicyAskWrites(
        state_updates=[
            StateUpdate(key="budget.checkpoint", action=StateUpdateAction.INCREMENT, value=1)
        ],
        set_labels=None,
        from_mcp=False,
    )

    applied: list[list[Any]] = []

    @dataclass
    class _CountingEngine(_FixedPolicyEngine):
        def apply_state_updates(self, updates: list[Any]) -> None:
            applied.append(updates)

    engine = _CountingEngine(result=PolicyResult(action=PolicyAction.ALLOW, reason=None))
    monkeypatch.setattr(
        sessions_mod,
        "_load_agent_spec_for_session",
        lambda conv, agent_store: "fake_spec",
    )
    monkeypatch.setattr(
        sessions_mod,
        "_build_policy_engine_from_spec",
        _engine_factory_expecting_preload(engine),
    )

    conv = _make_conversation()
    store = _StubConversationStore(conv)
    data = {"elicitation_id": eid, "action": "accept"}
    try:
        await asyncio.gather(
            _apply_pending_policy_ask_writes(
                session_id=_SESSION_ID,
                conv=conv,
                conversation_store=store,  # type: ignore[arg-type]
                agent_store=_StubAgentStore(),  # type: ignore[arg-type]
                data=data,
            ),
            _apply_pending_policy_ask_writes(
                session_id=_SESSION_ID,
                conv=conv,
                conversation_store=store,  # type: ignore[arg-type]
                agent_store=_StubAgentStore(),  # type: ignore[arg-type]
                data=data,
            ),
        )
        assert len(applied) == 1, (
            f"stashed relay writes applied {len(applied)} times for duplicate "
            "verdicts — non-idempotent ops (INCREMENT counters) double-apply."
        )
        assert eid not in _pending_policy_ask_writes
    finally:
        _pending_policy_ask_writes.pop(eid, None)


# ---------------------------------------------------------------------------
# Actor threading tests
# ---------------------------------------------------------------------------


@dataclass
class _CapturingPolicyEngine:
    """
    Policy engine stub that records the :class:`EvaluationContext` it
    receives and returns ALLOW so execution proceeds.

    :param captured: Mutable list; the first ``evaluate`` call appends
        the context it received. Using a list avoids ``nonlocal``.
    """

    captured: list[EvaluationContext]

    async def evaluate(self, ctx: EvaluationContext) -> PolicyResult:
        """Record ``ctx`` and return ALLOW.

        :param ctx: The evaluation context passed by the handler.
        :returns: ``ALLOW`` so the handler proceeds past the gate.
        """
        self.captured.append(ctx)
        return PolicyResult(action=PolicyAction.ALLOW)

    def apply_label_writes(self, set_labels: dict[str, str]) -> None:
        """No-op."""

    def apply_state_updates(self, updates: list[Any]) -> None:
        """No-op."""


@pytest.mark.asyncio
async def test_mcp_proxy_runner_supplied_actor_reaches_policy_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    When the runner includes ``actor`` in the JSON-RPC body, the policy
    engine receives that identity in ``EvaluationContext.actor`` rather
    than ``None`` (the request has no authenticated user in this path).

    Verifies the fix for shared multi-user sessions where the MCP proxy
    must gate on the human who triggered the turn, not the runner's
    service-account credential.
    """
    captured: list[EvaluationContext] = []
    engine = _CapturingPolicyEngine(captured=captured)

    monkeypatch.setattr(
        sessions_mod,
        "_load_agent_spec_for_session",
        lambda conv, agent_store: "fake_spec",
    )
    monkeypatch.setattr(
        sessions_mod,
        "_build_policy_engine_from_spec",
        _engine_factory_expecting_preload(engine),
    )

    params = {"name": "sys_os_shell", "arguments": {"command": "echo hi"}}
    await _handle_mcp_tools_call(
        rpc_id=1,
        session_id=_SESSION_ID,
        params=params,
        conversation_store=_StubConversationStore(_make_conversation()),  # type: ignore[arg-type]
        agent_store=_StubAgentStore(),  # type: ignore[arg-type]
        runner_router=None,
        actor={"run_as": "alice@example.com"},
    )

    assert captured, "policy engine was not called"
    assert captured[0].actor == {"run_as": "alice@example.com"}, (
        f"expected actor from runner body, got: {captured[0].actor!r}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("retry_action", "reviewed_transform", "retry_transform", "changed_arguments", "approval"),
    [
        (PolicyAction.ASK, {"value": "reviewed"}, {"value": "later"}, False, "accept"),
        (PolicyAction.ASK, None, {"value": "later"}, False, "accept"),
        (PolicyAction.ASK, {}, {"value": "later"}, False, "accept"),
        (PolicyAction.ASK, {"value": "reviewed"}, None, True, "accept"),
        (PolicyAction.ASK, {"value": "reviewed"}, None, False, "decline"),
        (PolicyAction.DENY, {"value": "reviewed"}, None, False, "accept"),
        (PolicyAction.ALLOW, {"value": "reviewed"}, {"value": "current"}, False, "accept"),
        (PolicyAction.ALLOW, {"value": "reviewed"}, None, False, "accept"),
        (PolicyAction.ALLOW, {"value": "reviewed"}, {}, False, "accept"),
    ],
)
async def test_retry_dispatches_policy_owned_arguments(
    monkeypatch: pytest.MonkeyPatch,
    retry_action: PolicyAction,
    reviewed_transform: dict[str, str] | None,
    retry_transform: dict[str, str] | None,
    changed_arguments: bool,
    approval: str,
) -> None:
    """Approval preserves the reviewed call; ALLOW uses the current policy result."""
    eid = ""
    original_arguments = {"value": "original"}
    dispatched: list[dict[str, Any]] = []
    label_writes: list[dict[str, str]] = []
    evaluations: list[EvaluationContext] = []

    class Engine(_FixedPolicyEngine):
        def apply_label_writes(self, set_labels: dict[str, str]) -> None:
            label_writes.append(set_labels)

    engine = Engine(PolicyResult(action=PolicyAction.ALLOW))
    call_results = iter(
        [
            PolicyResult(
                action=PolicyAction.ASK,
                data=reviewed_transform,
                set_labels={"review": "accepted"},
                deciding_policies=["review"],
            ),
            PolicyResult(action=retry_action, data=retry_transform),
        ]
    )

    async def evaluate(session_id, spec, store, conv, ctx):
        evaluations.append(ctx)
        result = (
            next(call_results)
            if ctx.phase == Phase.TOOL_CALL
            else PolicyResult(action=PolicyAction.ALLOW)
        )
        return engine, result

    async def register(*args, elicitation_id, **kwargs):
        return elicitation_id

    def execute(request: httpx.Request) -> httpx.Response:
        dispatched.append(json.loads(request.content))
        return httpx.Response(200, json={"result": {"output": "ok"}})

    monkeypatch.setattr(sessions_mod, "_load_agent_spec_for_session", lambda *args: "spec")
    monkeypatch.setattr(orchestration, "_evaluate_policy_with_fresh_engine", evaluate)
    monkeypatch.setattr(orchestration, "_register_policy_elicitation", register)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(execute), base_url="http://runner.test"
    ) as runner:

        async def get_runner(*args):
            return runner

        monkeypatch.setattr(sessions_mod, "_get_runner_client", get_runner)

        async def call(params):
            response = await _handle_mcp_tools_call(
                rpc_id=1,
                session_id=_SESSION_ID,
                params=params,
                conversation_store=_StubConversationStore(_make_conversation()),  # type: ignore[arg-type]
                agent_store=_StubAgentStore(),  # type: ignore[arg-type]
                runner_router=None,
            )
            return json.loads(bytes(response.body))

        try:
            initial = await call({"name": "example__lookup", "arguments": original_arguments})
            assert not dispatched
            assert not label_writes
            eid = next(iter(initial["result"]["inputRequests"]))
            # The person answers in the browser; the runner relays that answer.
            orchestration._record_mcp_approval_verdict(_SESSION_ID, eid, approval, "alice")
            state = json.loads(initial["result"]["requestState"])
            state["transformed_arguments"] = {"value": "client"}
            result = await call(
                {
                    "name": "example__lookup",
                    "arguments": {"value": "changed"} if changed_arguments else original_arguments,
                    "requestState": json.dumps(state),
                    "inputResponses": {eid: {"action": approval}},
                }
            )
            rejected = retry_action == PolicyAction.DENY or (
                retry_action == PolicyAction.ASK and (changed_arguments or approval != "accept")
            )
            if rejected:
                assert "error" in result
                assert not dispatched
                assert not label_writes
                if changed_arguments:
                    assert "do not match" in result["error"]["message"]
            else:
                assert "error" not in result
                transform = (
                    reviewed_transform if retry_action == PolicyAction.ASK else retry_transform
                )
                expected = transform if transform is not None else original_arguments
                assert len(dispatched) == 1
                assert dispatched[0]["params"]["arguments"] == expected
                assert evaluations[-1].request_data == {
                    "name": "example__lookup",
                    "arguments": expected,
                }
                if retry_action == PolicyAction.ASK:
                    assert label_writes == [{"review": "accepted"}]
                    assert eid not in _pending_policy_ask_writes
            assert "transformed_arguments" not in json.loads(initial["result"]["requestState"])
        finally:
            _pending_policy_ask_writes.pop(eid, None)


# ---------------------------------------------------------------------------
# Recorded human verdicts
# ---------------------------------------------------------------------------


_OTHER_SESSION_ID = "conv_test_policy_retry_other"
_RUNNER_ID = "runner_policy_retry"


@dataclass
class _StubConversations:
    """Conversation store holding several canned conversations by id."""

    conversations: dict[str, Conversation]

    def get_conversation(self, session_id: str) -> Conversation | None:
        """Return the canned conversation for *session_id*, if any."""
        return self.conversations.get(session_id)


@dataclass
class _StubTunnelRegistry:
    """Tunnel registry exposing one connected runner and its generation."""

    generation: int

    def get(self, runner_id: str) -> SimpleNamespace | None:
        """Return the connected runner's session, or ``None`` for others."""
        return SimpleNamespace(generation=self.generation) if runner_id == _RUNNER_ID else None


def _runner_bound_conversation(session_id: str) -> Conversation:
    """Build a conversation bound to :data:`_RUNNER_ID`."""
    return Conversation(
        id=session_id,
        created_at=0,
        updated_at=0,
        root_conversation_id=session_id,
        agent_id="ag_test",
        runner_id=_RUNNER_ID,
    )


@dataclass
class _ApprovalHarness:
    """Drive the MCP ASK and retry through ``_handle_mcp_tools_call``.

    :param registry: Tunnel registry whose generation a test may change.
    :param dispatched: Every tool call the runner was asked to execute.
    :param dispatch_generations: The tunnel generation each execution was
        pinned to, or ``None`` when unpinned.
    """

    registry: _StubTunnelRegistry
    dispatched: list[dict[str, Any]]
    dispatch_generations: list[int | None]

    async def call(
        self,
        params: dict[str, Any],
        *,
        session_id: str = _SESSION_ID,
        actor: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """POST one ``tools/call`` and return the JSON-RPC payload."""
        response = await _handle_mcp_tools_call(
            rpc_id=1,
            session_id=session_id,
            params=params,
            conversation_store=_StubConversations(  # type: ignore[arg-type]
                {sid: _runner_bound_conversation(sid) for sid in (_SESSION_ID, _OTHER_SESSION_ID)}
            ),
            agent_store=_StubAgentStore(),  # type: ignore[arg-type]
            runner_router=None,
            actor=actor or {"run_as": "alice@example.com"},
            request=SimpleNamespace(  # type: ignore[arg-type]
                app=SimpleNamespace(state=SimpleNamespace(tunnel_registry=self.registry))
            ),
        )
        return json.loads(bytes(response.body))

    async def ask(self) -> tuple[str, dict[str, Any]]:
        """Make the first call; return the elicitation id and retry params."""
        initial = await self.call(
            {
                "name": "linkedin__post",
                "arguments": {"text": "hello"},
                MCP_OPERATION_ID_PARAM: "mcpop_1",
            }
        )
        elicitation_id = next(iter(initial["result"]["inputRequests"]))
        return elicitation_id, {
            "name": "linkedin__post",
            "arguments": {"text": "hello"},
            MCP_OPERATION_ID_PARAM: "mcpop_1",
            "requestState": initial["result"]["requestState"],
            "inputResponses": {elicitation_id: {"action": "accept"}},
        }


@pytest.fixture
def approval_harness(monkeypatch: pytest.MonkeyPatch) -> Iterator[_ApprovalHarness]:
    """An ASK-gated MCP tool whose runner execution is recorded."""
    harness = _ApprovalHarness(_StubTunnelRegistry(generation=7), [], [])
    engine = _FixedPolicyEngine(PolicyResult(action=PolicyAction.ALLOW))

    async def evaluate(session_id, spec, store, conv, ctx):
        if ctx.phase == Phase.TOOL_CALL:
            return engine, PolicyResult(
                action=PolicyAction.ASK, reason="post?", deciding_policies=["review"]
            )
        return engine, PolicyResult(action=PolicyAction.ALLOW)

    def execute(request: httpx.Request) -> httpx.Response:
        harness.dispatched.append(json.loads(request.content))
        harness.dispatch_generations.append(request.extensions.get("runner_tunnel_generation"))
        return httpx.Response(200, json={"result": {"output": "posted"}})

    runner = httpx.AsyncClient(transport=httpx.MockTransport(execute), base_url="http://runner")

    async def get_runner(*args):
        return runner

    monkeypatch.setattr(sessions_mod, "_load_agent_spec_for_session", lambda *args: "spec")
    monkeypatch.setattr(orchestration, "_evaluate_policy_with_fresh_engine", evaluate)
    monkeypatch.setattr(sessions_mod, "_get_runner_client", get_runner)
    before = set(_pending_policy_ask_writes.keys())
    yield harness
    for key in set(_pending_policy_ask_writes.keys()) - before:
        _pending_policy_ask_writes.pop(key, None)


@pytest.mark.asyncio
async def test_person_approval_is_consumed_by_one_runner_retry(
    approval_harness: _ApprovalHarness,
) -> None:
    """The normal path: a person approves, the runner's retry executes once."""
    eid, retry = await approval_harness.ask()
    assert not approval_harness.dispatched

    orchestration._record_mcp_approval_verdict(_SESSION_ID, eid, "accept", "alice@example.com")
    executed = await approval_harness.call(retry)
    duplicate = await approval_harness.call(retry)

    assert "error" not in executed, executed
    assert len(approval_harness.dispatched) == 1
    assert approval_harness.dispatched[0]["params"]["arguments"] == {"text": "hello"}
    assert duplicate["error"]["message"] == "Elicitation not found or already resolved"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("scenario", "message", "still_pending"),
    [
        ("forged_accept", "Tool call has not been approved", True),
        ("declined", "Tool call denied by user", False),
        ("changed_arguments", "Retry arguments do not match the approved request", False),
        ("other_operation", "Approval does not match this call", False),
        ("runner_reconnected", "Approval was granted before the runner reconnected", False),
        ("other_session", "Approval does not match this call", True),
        ("other_actor", "Approval does not match this call", True),
        ("server_restart", "Elicitation not found or already resolved", False),
    ],
)
async def test_retry_without_a_matching_person_approval_never_executes(
    approval_harness: _ApprovalHarness, scenario: str, message: str, still_pending: bool
) -> None:
    """Every retry the person did not approve as-is is refused before execution."""
    eid, retry = await approval_harness.ask()
    if scenario != "forged_accept":
        verdict = "decline" if scenario == "declined" else "accept"
        orchestration._record_mcp_approval_verdict(_SESSION_ID, eid, verdict, "alice")
    session_id = _SESSION_ID
    actor = None
    if scenario == "changed_arguments":
        retry["arguments"] = {"text": "something else"}
    elif scenario == "other_operation":
        retry[MCP_OPERATION_ID_PARAM] = "mcpop_2"
    elif scenario == "runner_reconnected":
        approval_harness.registry.generation += 1
    elif scenario == "other_session":
        session_id = _OTHER_SESSION_ID
        retry["requestState"] = json.dumps({"session_id": session_id, "elicitation_id": eid})
    elif scenario == "other_actor":
        actor = {"run_as": "mallory@example.com"}
    elif scenario == "server_restart":
        _pending_policy_ask_writes.pop(eid)

    refused = await approval_harness.call(retry, session_id=session_id, actor=actor)

    assert refused["error"]["message"] == message
    assert not approval_harness.dispatched
    assert (eid in _pending_policy_ask_writes) is still_pending


@pytest.mark.asyncio
async def test_first_person_verdict_is_recorded_with_who_gave_it(
    approval_harness: _ApprovalHarness,
) -> None:
    """Only this session's first answer is recorded, with the answering user."""
    eid, _retry = await approval_harness.ask()

    orchestration._record_mcp_approval_verdict(_OTHER_SESSION_ID, eid, "accept", "mallory")
    assert _pending_policy_ask_writes[eid].verdict is None
    orchestration._record_mcp_approval_verdict(_SESSION_ID, eid, "decline", "alice")
    orchestration._record_mcp_approval_verdict(_SESSION_ID, eid, "accept", "bob")
    recorded = _pending_policy_ask_writes[eid]

    assert (recorded.verdict, recorded.decided_by) == ("decline", "alice")


@pytest.mark.asyncio
async def test_verdict_is_recorded_before_the_runner_hears_it(
    approval_harness: _ApprovalHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The runner retries as soon as it hears the verdict, so the record comes first."""
    eid, _retry = await approval_harness.ask()
    seen_by_runner: list[str | None] = []

    async def forward(session_id, data, runner_router):
        seen_by_runner.append(_pending_policy_ask_writes[eid].verdict)

    monkeypatch.setattr(orchestration, "_forward_approval_to_runner", forward)
    await orchestration._resolve_elicitation(
        _SESSION_ID, {"elicitation_id": eid, "action": "accept"}, None, decided_by="alice"
    )

    assert seen_by_runner == ["accept"]
    assert _pending_policy_ask_writes[eid].decided_by == "alice"


@pytest.mark.asyncio
async def test_approved_retry_cannot_replay_once_policy_allows(
    approval_harness: _ApprovalHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An approval that makes the policy allow (e.g. a recorded checkpoint) runs once."""
    eid, retry = await approval_harness.ask()
    orchestration._record_mcp_approval_verdict(_SESSION_ID, eid, "accept", "alice")
    assert "error" not in await approval_harness.call(retry)

    engine = _FixedPolicyEngine(PolicyResult(action=PolicyAction.ALLOW))

    async def allow(session_id, spec, store, conv, ctx):
        return engine, PolicyResult(action=PolicyAction.ALLOW)

    monkeypatch.setattr(orchestration, "_evaluate_policy_with_fresh_engine", allow)
    replayed = await approval_harness.call(retry)

    assert replayed["error"]["message"] == "Elicitation not found or already resolved"
    assert len(approval_harness.dispatched) == 1


@pytest.mark.asyncio
async def test_approved_call_executes_only_on_the_approved_connection(
    approval_harness: _ApprovalHarness,
) -> None:
    """Execution is pinned to the runner connection the approval was granted on."""
    eid, retry = await approval_harness.ask()
    orchestration._record_mcp_approval_verdict(_SESSION_ID, eid, "accept", "alice")

    executed = await approval_harness.call(retry)

    assert "error" not in executed, executed
    assert approval_harness.dispatch_generations == [7]


@pytest.mark.asyncio
async def test_answer_given_while_the_prompt_is_published_is_kept(
    approval_harness: _ApprovalHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The approval record exists before anyone can see, and answer, the prompt."""

    def answer_during_fan_out(store, session_id, event):
        orchestration._record_mcp_approval_verdict(
            session_id, event["elicitation_id"], "accept", "alice"
        )

    monkeypatch.setattr(
        orchestration, "_publish_elicitation_request_to_ancestors", answer_during_fan_out
    )
    _eid, retry = await approval_harness.ask()

    assert "error" not in await approval_harness.call(retry)
    assert len(approval_harness.dispatched) == 1
