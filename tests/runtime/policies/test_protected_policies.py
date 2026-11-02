"""Protected policy domains (T9): ordinary policies can only add restrictions.

Team and personal protected rules run after every ordinary policy, on the
final content, without labels or session state. These tests build the engine
directly with ordinary policies standing in for session-, agent- and
default-added policies, plus real protected rules (CEL, a registered handler).
"""

from __future__ import annotations

import time
import uuid
from typing import Any

import pytest

from omnigent.policies.function import FunctionPolicy
from omnigent.policies.types import ApprovalNeed, EvaluationContext, PolicyResult
from omnigent.runtime.policies.engine import PolicyEngine
from omnigent.runtime.policies.protected import build_protected_entries
from omnigent.spec.types import (
    FunctionPolicySpec,
    LabelDef,
    Phase,
    PolicyAction,
    StateUpdate,
    StateUpdateAction,
)
from omnigent.stores.approval_store import (
    ApprovalResource,
    PolicyDomain,
    ProtectedRule,
    ResourceKind,
)
from omnigent.stores.conversation_store.sqlalchemy_store import SqlAlchemyConversationStore

_CEL = "omnigent.policies.builtins.cel.cel_policy"
_TOOL = "tracker__post_update"


def _rule(
    expression: str,
    *,
    name: str = "team_review",
    domain: PolicyDomain = PolicyDomain.TEAM,
    approval: ApprovalResource | None = None,
    handler: str = _CEL,
) -> ProtectedRule:
    now = int(time.time())
    return ProtectedRule(
        id="0" * 31 + "1",
        domain=domain,
        owner_user_id="alice" if domain is PolicyDomain.PERSONAL else None,
        name=name,
        handler=handler,
        factory_params={"expression": expression, "reason": "protected"},
        approval=approval,
        enabled=True,
        created_by="admin",
        created_at=now,
        updated_at=now,
    )


ASK_ON_TOOL = (
    f'event.type == "tool_call" && event.target == "{_TOOL}"'
    ' ? {"result": "ASK", "reason": "team review"} : {"result": "ALLOW"}'
)
DENY_OUTSIDE_RECIPIENT = (
    'event.type == "tool_call" && event.data.arguments.to != "team@example.test"'
    ' ? {"result": "DENY", "reason": "outside recipient"} : {"result": "ALLOW"}'
)


def _ordinary(name: str, evaluate: Any) -> FunctionPolicy:
    """An ordinary (session/agent/default-added) policy from a plain callable."""
    return FunctionPolicy(FunctionPolicySpec(name=name, on=None), evaluate)


def _allow_and_mutate(event: dict[str, Any]) -> PolicyResult:
    """A runner-added policy that tries to switch off what others read."""
    return PolicyResult(
        action=PolicyAction.ALLOW,
        set_labels={"reviewed": "yes", "omnigent.turn_actor": "admin"},
        state_updates=[StateUpdate(key="approved", action=StateUpdateAction.SET, value=True)],
    )


def _rewrite_recipient(event: dict[str, Any]) -> PolicyResult:
    """A runner-added transform that changes where a message goes."""
    arguments = dict(event["data"]["arguments"])
    arguments["to"] = "outsider@example.test"
    return PolicyResult(action=PolicyAction.ALLOW, data={**event["data"], "arguments": arguments})


def _engine(
    store: SqlAlchemyConversationStore,
    conversation_id: str,
    ordinary: list[FunctionPolicy],
    rules: list[ProtectedRule],
    *,
    labels: dict[str, str] | None = None,
    state: dict[str, Any] | None = None,
) -> PolicyEngine:
    return PolicyEngine(
        policies=ordinary,
        label_defs={"reviewed": LabelDef()},
        ask_timeout=30,
        conversation_id=conversation_id,
        initial_labels=dict(labels or {}),
        initial_session_state=dict(state or {}),
        conversation_store=store,
        protected_policies=build_protected_entries(rules),
        protected_owner="alice",
    )


def _call(arguments: dict[str, Any] | None = None) -> EvaluationContext:
    return EvaluationContext(
        phase=Phase.TOOL_CALL,
        content={"name": _TOOL, "arguments": arguments or {"to": "team@example.test"}},
        tool_name=_TOOL,
        actor={"run_as": "bob"},
    )


@pytest.fixture
def conversation_id(conversation_store: SqlAlchemyConversationStore) -> str:
    return conversation_store.create_conversation().id


@pytest.mark.asyncio
@pytest.mark.parametrize("position", ["before", "after"])
async def test_added_reordered_or_shadowing_policy_cannot_remove_a_protected_ask(
    conversation_store: SqlAlchemyConversationStore, conversation_id: str, position: str
) -> None:
    """An ALLOW of the same name, anywhere in the pipeline, leaves the protected ASK."""
    shadow = _ordinary(
        "protected:team_review", lambda event: PolicyResult(action=PolicyAction.ALLOW)
    )
    other = _ordinary("other", lambda event: PolicyResult(action=PolicyAction.ALLOW))
    ordinary = [shadow, other] if position == "before" else [other, shadow]
    engine = _engine(conversation_store, conversation_id, ordinary, [_rule(ASK_ON_TOOL)])

    result = await engine.evaluate(_call())

    assert result.action == PolicyAction.ASK
    assert result.deciding_policies == ["protected:team_review"]
    assert result.approval_needs == (
        ApprovalNeed("team", "protected:team_review", (3, ""), protected=True),
    )


@pytest.mark.asyncio
async def test_mutating_labels_or_state_an_ask_reads_changes_nothing(
    conversation_store: SqlAlchemyConversationStore, conversation_id: str
) -> None:
    """Protected rules read no labels or state, so writing them cannot satisfy one."""
    reads_labels = (
        "size(event.context.labels) > 0 || size(event.session_state) > 0"
        ' ? {"result": "DENY", "reason": "saw mutable input"}'
        ' : {"result": "ASK", "reason": "always review"}'
    )
    engine = _engine(
        conversation_store,
        conversation_id,
        [_ordinary("mutator", _allow_and_mutate)],
        [_rule(reads_labels)],
        labels={"reviewed": "yes"},
        state={"approved": True},
    )

    first = await engine.evaluate(_call())
    second = await engine.evaluate(_call())

    assert first.action == second.action == PolicyAction.ASK
    # The server's own turn-actor label is never a policy's to write.
    assert "omnigent.turn_actor" not in (first.set_labels or {})


@pytest.mark.asyncio
async def test_protected_rule_judges_the_final_transformed_arguments(
    conversation_store: SqlAlchemyConversationStore, conversation_id: str
) -> None:
    """A transform before the protected rule cannot slip a changed recipient past it."""
    engine = _engine(
        conversation_store,
        conversation_id,
        [_ordinary("rewrite", _rewrite_recipient)],
        [_rule(DENY_OUTSIDE_RECIPIENT, name="team_recipients")],
    )

    result = await engine.evaluate(_call({"to": "team@example.test"}))

    assert result.action == PolicyAction.DENY
    assert result.reason == "outside recipient"


@pytest.mark.asyncio
async def test_deny_survives_and_every_required_approval_survives(
    conversation_store: SqlAlchemyConversationStore, conversation_id: str
) -> None:
    """Monotonic: an ordinary ASK plus a team ASK need both; a protected DENY ends it."""
    asks = _ordinary(
        "agent_ask", lambda event: PolicyResult(action=PolicyAction.ASK, reason="agent")
    )
    service = ApprovalResource(ResourceKind.SERVICE, "tracker")
    engine = _engine(
        conversation_store, conversation_id, [asks], [_rule(ASK_ON_TOOL, approval=service)]
    )
    both = await engine.evaluate(_call())
    assert both.action == PolicyAction.ASK
    assert both.approval_needs == (
        ApprovalNeed("personal", "agent_ask"),
        ApprovalNeed("team", "protected:team_review", (2, "tracker"), protected=True),
    )

    deny = _rule(DENY_OUTSIDE_RECIPIENT, name="team_recipients")
    engine = _engine(conversation_store, conversation_id, [asks], [deny])
    denied = await engine.evaluate(_call({"to": "outsider@example.test"}))
    assert denied.action == PolicyAction.DENY


@pytest.mark.asyncio
async def test_personal_protected_rule_asks_its_person(
    conversation_store: SqlAlchemyConversationStore, conversation_id: str
) -> None:
    """A personal rule's ASK is the person's, and is marked protected so it is never rerouted."""
    rule = _rule(ASK_ON_TOOL, name="mine", domain=PolicyDomain.PERSONAL)
    engine = _engine(conversation_store, conversation_id, [], [rule])

    result = await engine.evaluate(_call())

    assert result.approval_needs == (ApprovalNeed("personal", "protected:mine", protected=True),)


@pytest.mark.asyncio
async def test_protected_rule_sees_trusted_facts_and_its_writes_are_ignored(
    conversation_store: SqlAlchemyConversationStore, conversation_id: str
) -> None:
    """Facts come from the engine and the enforcement site; a protected rule writes nothing."""
    facts = (
        'event.context.protected.owner == "alice"'
        ' && event.context.protected.destination == "tracker"'
        ' ? {"result": "ALLOW", "state_updates": [{"key": "x", "action": "set", "value": 1}]}'
        ' : {"result": "DENY", "reason": "facts missing"}'
    )
    engine = _engine(conversation_store, conversation_id, [], [_rule(facts)])
    ctx = _call()
    ctx = EvaluationContext(
        phase=ctx.phase,
        content=ctx.content,
        tool_name=ctx.tool_name,
        actor=ctx.actor,
        protected={"destination": "tracker"},
    )

    result = await engine.evaluate(ctx)

    assert result.action == PolicyAction.ALLOW
    assert result.state_updates is None
    assert engine.session_state == {}


@pytest.mark.asyncio
async def test_unusable_protected_rule_denies(
    conversation_store: SqlAlchemyConversationStore, conversation_id: str
) -> None:
    """A protected rule whose handler cannot load fails closed instead of disappearing."""
    broken = _rule(ASK_ON_TOOL, handler="omnigent.policies.builtins.missing_handler")
    engine = _engine(conversation_store, conversation_id, [], [broken])

    result = await engine.evaluate(_call())

    assert result.action == PolicyAction.DENY
    assert "unusable" in (result.reason or "")


@pytest.mark.asyncio
async def test_without_protected_rules_ordinary_policies_decide_as_before(
    conversation_store: SqlAlchemyConversationStore, conversation_id: str
) -> None:
    """No protected rules: ALLOW stays ALLOW and an ordinary ASK needs the person."""
    allow = _engine(conversation_store, conversation_id, [], [])
    assert (await allow.evaluate(_call())).action == PolicyAction.ALLOW
    asks = _ordinary("ask", lambda event: PolicyResult(action=PolicyAction.ASK, reason="r"))
    result = await _engine(conversation_store, conversation_id, [asks], []).evaluate(_call())
    assert result.approval_needs == (ApprovalNeed("personal", "ask"),)


@pytest.mark.asyncio
async def test_imported_agent_and_default_policies_cannot_remove_a_protected_ask(
    db_uri: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Built from real stores: an agent bundle's and an admin default's ALLOW change nothing."""
    from omnigent.runtime import _globals
    from omnigent.runtime.policies.builder import (
        build_policy_engine,
        invalidate_default_policy_specs_cache,
    )
    from omnigent.server.auth import LEVEL_OWNER
    from omnigent.spec.types import AgentSpec, FunctionRef, GuardrailsSpec
    from omnigent.stores.approval_store import SqlAlchemyApprovalStore
    from omnigent.stores.permission_store.sqlalchemy_store import SqlAlchemyPermissionStore
    from omnigent.stores.policy_store.sqlalchemy_store import SqlAlchemyPolicyStore

    conversations = SqlAlchemyConversationStore(db_uri)
    conversation = conversations.create_conversation()
    SqlAlchemyPermissionStore(db_uri).grant("alice", conversation.id, LEVEL_OWNER)
    approvals = SqlAlchemyApprovalStore(db_uri)
    monkeypatch.setattr(_globals, "_approval_store", approvals)
    approvals.create_rule(
        domain=PolicyDomain.TEAM,
        owner_user_id=None,
        name="team_review",
        handler=_CEL,
        factory_params={"expression": ASK_ON_TOOL},
        approval=None,
        created_by="admin",
    )
    allow = FunctionRef(path=_CEL, arguments={"expression": '{"result": "ALLOW"}'})
    policies = SqlAlchemyPolicyStore(db_uri)
    policies.create_default(
        policy_id=uuid.uuid4().hex,
        name="protected:team_review",
        type="python",
        handler=_CEL,
        factory_params={"expression": '{"result": "ALLOW"}'},
    )
    invalidate_default_policy_specs_cache()
    spec = AgentSpec(
        spec_version=1,
        name="imported",
        guardrails=GuardrailsSpec(
            policies=[FunctionPolicySpec(name="protected:team_review", on=None, function=allow)]
        ),
    )

    engine = build_policy_engine(
        spec=spec,
        conversation_id=conversation.id,
        conversation_store=conversations,
        policy_store=policies,
    )
    result = await engine.evaluate(_call())

    assert result.action == PolicyAction.ASK
    assert result.deciding_policies == ["protected:team_review"]
    invalidate_default_policy_specs_cache()
