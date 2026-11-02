"""Protected policy domains: rules no session- or agent-added policy can weaken.

Team rules are kept by human administrators; personal rules by their person.
They run after every ordinary policy (session, agent, default and server
policies), on the final arguments after every transform, and see only
server-derived facts: no labels, no session state, no usage counters. Their
own label writes, state updates and transforms are ignored, so evaluating a
protected rule has no side effect. Their verdicts only add restrictions: a
protected DENY ends the decision, a protected ASK always survives, and an
ordinary ALLOW never erases either.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from omnigent.policies.function import resolve_function_policy
from omnigent.policies.types import ApprovalNeed, EvaluationContext, PolicyResult
from omnigent.spec.types import FunctionPolicySpec, FunctionRef, PolicyAction

if TYPE_CHECKING:
    from omnigent.policies.base import Policy
    from omnigent.stores.approval_store import ProtectedRule

PERSONAL_NEED = "personal"
TEAM_NEED = "team"
# ``(kind_code, resource_id)`` of the team as a whole (ResourceKind.TEAM).
TEAM_RESOURCE_KEY = (3, "")


@dataclasses.dataclass(frozen=True)
class ProtectedPolicyEntry:
    """A protected rule ready to evaluate.

    :param rule: The stored rule.
    :param policy: Its evaluator, or ``None`` when the handler could not be
        loaded; such a rule denies, so a broken rule never stops enforcing.
    :param load_error: Why the handler could not be loaded.
    """

    rule: ProtectedRule
    policy: Policy | None
    load_error: str | None = None

    @property
    def name(self) -> str:
        """The name shown on its verdicts."""
        return f"protected:{self.rule.name}"

    def need(self) -> ApprovalNeed:
        """Who must answer this rule's ASK."""
        from omnigent.stores.approval_store import PolicyDomain

        if self.rule.domain is PolicyDomain.PERSONAL:
            return ApprovalNeed(PERSONAL_NEED, self.name, protected=True)
        approval = self.rule.approval
        resource = (
            (int(approval.kind), approval.resource_id)
            if approval is not None
            else TEAM_RESOURCE_KEY
        )
        return ApprovalNeed(TEAM_NEED, self.name, resource, protected=True)


def build_protected_entries(rules: Sequence[ProtectedRule]) -> list[ProtectedPolicyEntry]:
    """Instantiate protected rules; a rule that fails to load still denies."""
    entries = []
    for rule in rules:
        spec = FunctionPolicySpec(
            name=f"protected:{rule.name}",
            on=None,
            function=FunctionRef(path=rule.handler, arguments=rule.factory_params),
        )
        try:
            entries.append(ProtectedPolicyEntry(rule, resolve_function_policy(spec)))
        except Exception as exc:
            entries.append(ProtectedPolicyEntry(rule, None, f"{type(exc).__name__}: {exc}"))
    return entries


def protected_fingerprint(rules: Sequence[ProtectedRule]) -> str:
    """Digest of the protected rules in force, for binding a verdict to them."""
    payload = json.dumps(sorted(rule.fingerprint() for rule in rules))
    return hashlib.sha256(payload.encode()).hexdigest()


def protected_context(
    ctx: EvaluationContext, content: object, facts: dict[str, object]
) -> EvaluationContext:
    """The context a protected rule sees: final content and trusted facts only."""
    return EvaluationContext(
        phase=ctx.phase,
        content=content,
        tool_name=ctx.tool_name,
        actor=ctx.actor,
        request_data=ctx.request_data,
        model=ctx.model,
        harness=ctx.harness,
        conversation_id=ctx.conversation_id,
        protected=facts,
    )


@dataclasses.dataclass(frozen=True)
class ProtectedVerdict:
    """What the protected rules decided.

    :param deny: The first protected DENY, or ``None``.
    :param asks: ``(name, reason, need)`` of every protected ASK.
    """

    deny: tuple[str, str] | None
    asks: list[tuple[str, str, ApprovalNeed]]


async def evaluate_protected(
    entries: Sequence[ProtectedPolicyEntry],
    ctx: EvaluationContext,
    context: dict[str, Any],
) -> ProtectedVerdict:
    """Run every protected rule on *ctx*; any failure is a DENY.

    :param entries: The rules in force.
    :param ctx: A context built by :func:`protected_context`.
    :param context: The engine's context bundle.
    """
    asks: list[tuple[str, str, ApprovalNeed]] = []
    for entry in entries:
        if entry.policy is None:
            return ProtectedVerdict(
                (entry.name, f"protected rule {entry.rule.name!r} is unusable"), asks
            )
        try:
            result: PolicyResult = await entry.policy.evaluate(ctx, context)
        except Exception as exc:
            return ProtectedVerdict(
                (entry.name, f"protected rule {entry.rule.name!r} failed: {exc}"), asks
            )
        if result.action == PolicyAction.DENY:
            return ProtectedVerdict(
                (entry.name, result.reason or "Denied by a protected rule"), asks
            )
        if result.action == PolicyAction.ASK:
            asks.append((entry.name, result.reason or "approval required", entry.need()))
    return ProtectedVerdict(None, asks)


__all__ = [
    "PERSONAL_NEED",
    "TEAM_NEED",
    "TEAM_RESOURCE_KEY",
    "ProtectedPolicyEntry",
    "ProtectedVerdict",
    "build_protected_entries",
    "evaluate_protected",
    "protected_context",
    "protected_fingerprint",
]
