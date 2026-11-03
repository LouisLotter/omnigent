"""Who may answer a protected approval, and what an approval is bound to.

Every policy ASK the server raises is recorded in the approval ledger
(:mod:`omnigent.stores.approval_store`) with:

- its requirements: the thread's person (or named thread approvers) for
  personal actions; the named approvers of a service or of the team for
  team-shared resources, by default not the person who started the action;
- a binding: the requesting principal, initiator, owner, thread, action,
  destination, content digest, connection generations, protected-rule and
  approver fingerprints, operation id and expiry.

Only a person signed in through a browser (or a trusted proxy) answers;
runners, hosts, service principals, device grants and CLI or host sign-ins
never do. An administrator is not an approver unless named. Each answer
satisfies the requirements its author may answer; the approval completes
when all are satisfied, and any eligible decline declines it.

An approval is consumed once, by the operation it was bound to, and only
while every bound fact still holds: changed arguments, rules, approvers,
owner or connection need a new decision. A retry of the consuming
operation inspects that operation's recorded result instead.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from omnigent.errors import ErrorCode, OmnigentError
from omnigent.policies.types import ApprovalNeed, PolicyResult
from omnigent.runtime import get_approval_store
from omnigent.runtime.policies.protected import (
    PERSONAL_NEED,
    TEAM_NEED,
)
from omnigent.server.auth import LEVEL_READ, AuthProvider
from omnigent.server.permissions import check_session_access
from omnigent.server.principals import CredentialKind, Principal, recorded_principal
from omnigent.stores.approval_store import (
    TEAM_RESOURCE,
    ApprovalResource,
    ConsumeResult,
    Outcome,
    ResourceApprovers,
    ResourceKind,
    SqlAlchemyApprovalStore,
    VerdictRecord,
    VerdictState,
    canonical_digest,
    requirement_resources,
)

if TYPE_CHECKING:
    from starlette.requests import HTTPConnection

    from omnigent.stores import ConversationStore
    from omnigent.stores.permission_store import PermissionStore

_logger = logging.getLogger(__name__)

# How long an approved verdict stays usable after the prompt's own timeout.
CONSUME_GRACE_SECONDS = 300
_BINDING_VERSION = 1
# Ledger rows are purged a while after they expire, at most this often.
_PURGE_INTERVAL_SECONDS = 600.0
_PURGE_AFTER_SECONDS = 7 * 24 * 3600
_last_purge = {"at": 0.0}


@dataclasses.dataclass(frozen=True)
class ApprovalBinding:
    """The canonical facts an approval is bound to, and their digest."""

    digest: bytes
    fields: dict[str, Any]


@dataclasses.dataclass(frozen=True)
class ApprovalSubject:
    """What one protected approval is about; inputs to its binding.

    :param session_id: The thread asking.
    :param phase: Enforcement phase, e.g. ``"tool_call"``.
    :param tool_name: Canonical action: the namespaced tool, or ``None``.
    :param content: The final content that would take effect.
    :param initiator: The person who started the turn, or ``None`` (the owner).
    :param destination: The service the action reaches, or ``None``.
    :param connection: The service connection's generation, or ``None``.
    :param runner_connection: The runner tunnel's generation, or ``None``.
    :param operation_id: The operation that would consume it, or ``None``.
    :param request: The request as posted, before any transform, for a gate
        that a retry re-posts (the native and request gates); ``None``
        elsewhere.
    """

    session_id: str
    phase: str
    tool_name: str | None
    content: object
    initiator: str | None = None
    destination: str | None = None
    connection: str | None = None
    runner_connection: int | None = None
    operation_id: str | None = None
    request: object | None = None


def _request_principal(request: object) -> Principal | None:
    """The principal recorded on a request; ``None`` for a tunnel frame or none."""
    scope = getattr(request, "scope", None)
    return recorded_principal(request) if isinstance(scope, dict) else None  # type: ignore[arg-type]


def turn_initiator(
    request: object, turn_actor: str | None, caller: str | None = None
) -> str | None:
    """The person who started the turn an action belongs to.

    A runner or a runner tunnel acts for the thread's owner; the person who
    sent the turn is the server-recorded turn actor (a label only the server
    writes). A person calling directly is the initiator themselves.
    """
    principal = _request_principal(request)
    if principal is None or principal.is_machine:
        return turn_actor or caller
    return caller or principal.user_id


def _principal_summary(request: object) -> dict[str, Any]:
    principal = _request_principal(request)
    if principal is None:
        return {"kind": "runner_tunnel" if request is not None else None, "user": None}
    return {
        "kind": principal.kind.value,
        "user": principal.user_id,
        "runner": principal.runner_id,
        "host": principal.host_id,
    }


def _owner(conversation_store: ConversationStore, session_id: str) -> str | None:
    authority = conversation_store.get_actual_session_owner(session_id)
    return authority.user_id if authority is not None else None


def _root(conversation_store: ConversationStore, session_id: str) -> str:
    conversation = conversation_store.get_conversation(session_id)
    if conversation is not None and conversation.root_conversation_id:
        return conversation.root_conversation_id
    return session_id


def _requirements(
    needs: Sequence[ApprovalNeed],
    *,
    root_session_id: str,
    destination: ApprovalResource | None,
) -> list[dict[str, Any]]:
    """Requirement records for an ASK, deduplicated, in order.

    An ordinary policy's ASK about an action on a service with named
    approvers is a team-shared action, answered by those approvers. A
    personal protected rule's ASK always goes to the person.
    """
    found: list[dict[str, Any]] = []
    for need in needs:
        entry = _requirement_for(need, root_session_id, destination)
        if entry not in found:
            found.append(entry)
    return found


def _requirement_for(
    need: ApprovalNeed, root_session_id: str, destination: ApprovalResource | None
) -> dict[str, Any]:
    if need.kind == TEAM_NEED and need.resource is not None:
        kind, resource_id = need.resource
        return {"kind": TEAM_NEED, "resource": [kind, resource_id]}
    if need.kind == PERSONAL_NEED and destination is not None and not need.protected:
        return {"kind": TEAM_NEED, "resource": destination.as_json()}
    personal = ApprovalResource(ResourceKind.SESSION, root_session_id)
    return {"kind": PERSONAL_NEED, "resource": personal.as_json()}


def final_check_problem(
    store: SqlAlchemyApprovalStore,
    record: VerdictRecord | None,
    needs: Sequence[ApprovalNeed],
    *,
    root_session_id: str,
) -> str | None:
    """Whether a consumed approval covers the protected ASKs found just before an effect.

    Every requirement those ASKs need must be one the approval was given for,
    under the same protected rules and approvers. Without a consumed approval,
    any protected ASK refuses the effect.

    :param record: The approval this request consumed, or ``None``.
    :param needs: The approvals the final protected evaluation requires.
    :returns: Why the effect may not happen, or ``None``.
    """
    if not needs:
        return None
    if record is None:
        return "A protected rule now requires approval for this call; retry it"
    required = [_requirement_for(need, root_session_id, None) for need in needs]
    if any(entry not in record.requirements for entry in required):
        return "A protected rule requires an approval this call was not given; ask again"
    if store.authority_fingerprints(record.requirements, record.binding.get("owner")) != (
        record.binding.get("grants"),
        record.binding.get("policies"),
    ):
        return "The approvers or protected rules changed after approval; ask again"
    return None


def _replay_matches_sync(
    store: SqlAlchemyApprovalStore,
    conversation_store: ConversationStore,
    request: HTTPConnection | None,
    subject: ApprovalSubject,
    record: VerdictRecord,
) -> bool:
    if record.state is not VerdictState.CONSUMED or "request" not in record.binding:
        return False
    current = _binding(store, conversation_store, request, subject, record.requirements).fields
    # The final content needs a fresh evaluation; the posted request stands in for it.
    ignored = {"content"}
    keys = (set(current) | set(record.binding)) - ignored
    changed = sorted(key for key in keys if current.get(key) != record.binding.get(key))
    if changed:
        _logger.info(
            "replayed approval %s refused: %s changed", record.elicitation_id, ", ".join(changed)
        )
    return not changed


async def replay_matches(
    request: HTTPConnection | None,
    conversation_store: ConversationStore,
    subject: ApprovalSubject,
    *,
    elicitation_id: str,
) -> bool | None:
    """Whether a re-posted, already approved question may reuse its answer.

    A gate's retry gets its recorded result only while every fact the approval
    was bound to still holds: principal, initiator, owner, thread, request,
    protected rules and approvers. A changed context needs a new decision.

    :returns: ``None`` when the question was never recorded in the ledger
        (no store, or not a protected approval); otherwise whether it matches.
    """
    store = get_approval_store()
    if store is None:
        return None
    record = await asyncio.to_thread(store.get_verdict, elicitation_id)
    if record is None:
        return None
    if record.session_id != subject.session_id:
        return False
    return await asyncio.to_thread(
        _replay_matches_sync, store, conversation_store, request, subject, record
    )


def _destination_resource(
    store: SqlAlchemyApprovalStore, destination: str | None
) -> ApprovalResource | None:
    """The destination service as a resource, when it has named approvers."""
    if not destination:
        return None
    resource = ApprovalResource(ResourceKind.SERVICE, destination)
    (entry,) = store.approvers_for([resource])
    return resource if entry.approvers else None


def _binding(
    store: SqlAlchemyApprovalStore,
    conversation_store: ConversationStore,
    request: HTTPConnection | None,
    subject: ApprovalSubject,
    requirements: Sequence[dict[str, Any]],
) -> ApprovalBinding:
    owner = _owner(conversation_store, subject.session_id)
    grants, policies = store.authority_fingerprints(requirements, owner)
    fields: dict[str, Any] = {
        "v": _BINDING_VERSION,
        "principal": _principal_summary(request),
        "initiator": subject.initiator or owner,
        "owner": owner,
        "session": subject.session_id,
        "phase": subject.phase,
        "action": subject.tool_name,
        "destination": subject.destination,
        "content": canonical_digest({"tool": subject.tool_name, "content": subject.content}),
        "connection": subject.connection,
        "runner_connection": subject.runner_connection,
        "operation": subject.operation_id,
        "policies": policies,
        "grants": grants,
        "requirements": list(requirements),
    }
    if subject.request is not None:
        fields["request"] = canonical_digest(subject.request)
    return ApprovalBinding(bytes.fromhex(canonical_digest(fields)), fields)


def _maybe_purge(store: SqlAlchemyApprovalStore, now: int) -> None:
    if time.monotonic() - _last_purge["at"] < _PURGE_INTERVAL_SECONDS:
        return
    _last_purge["at"] = time.monotonic()
    try:
        store.purge_expired(before=now - _PURGE_AFTER_SECONDS)
    except Exception:
        _logger.debug("approval ledger purge failed", exc_info=True)


def _record_sync(
    store: SqlAlchemyApprovalStore,
    conversation_store: ConversationStore,
    request: HTTPConnection | None,
    subject: ApprovalSubject,
    elicitation_id: str,
    result: PolicyResult,
    ask_timeout: float,
) -> VerdictRecord:
    needs = result.approval_needs or (
        ApprovalNeed(PERSONAL_NEED, result.deciding_policy or "unknown"),
    )
    requirements = _requirements(
        needs,
        root_session_id=_root(conversation_store, subject.session_id),
        destination=_destination_resource(store, subject.destination),
    )
    binding = _binding(store, conversation_store, request, subject, requirements)
    now = int(time.time())
    _maybe_purge(store, now)
    return store.record_request(
        elicitation_id=elicitation_id,
        session_id=subject.session_id,
        binding_digest=binding.digest,
        binding=binding.fields,
        requirements=requirements,
        expires_at=now + int(ask_timeout) + CONSUME_GRACE_SECONDS,
    )


async def record_protected_approval(
    request: HTTPConnection | None,
    conversation_store: ConversationStore,
    subject: ApprovalSubject,
    *,
    elicitation_id: str,
    result: PolicyResult,
    ask_timeout: float,
) -> VerdictRecord | None:
    """Record a policy ASK in the approval ledger before it is shown.

    :returns: The ledger row, or ``None`` when the server has no approval store.
    """
    store = get_approval_store()
    if store is None:
        return None
    return await asyncio.to_thread(
        _record_sync,
        store,
        conversation_store,
        request,
        subject,
        elicitation_id,
        result,
        ask_timeout,
    )


def current_binding(
    store: SqlAlchemyApprovalStore,
    conversation_store: ConversationStore,
    request: HTTPConnection | None,
    subject: ApprovalSubject,
    record: VerdictRecord,
) -> ApprovalBinding:
    """The binding the action would have now, under the recorded requirements."""
    return _binding(store, conversation_store, request, subject, record.requirements)


@dataclasses.dataclass(frozen=True)
class ConsumedApproval:
    """The outcome of consuming a protected approval for an action.

    :param consumed: This call consumed the approval; the action may run.
    :param record: The ledger row, when one exists.
    :param refusal: Why it may not run, when it may not.
    :param replay: The consuming operation's recorded result, for a retry
        of the same operation after it already ran.
    """

    consumed: bool
    record: VerdictRecord | None
    refusal: str | None = None
    replay: str | None = None


def _consume_sync(
    store: SqlAlchemyApprovalStore,
    conversation_store: ConversationStore,
    request: HTTPConnection | None,
    subject: ApprovalSubject,
    elicitation_id: str,
) -> ConsumedApproval:
    record = store.get_verdict(elicitation_id)
    if record is None or record.session_id != subject.session_id:
        return ConsumedApproval(False, None, "Approval not found")
    if (
        record.state is VerdictState.CONSUMED
        and subject.operation_id is not None
        and record.operation_id == subject.operation_id
    ):
        # A retry of the operation that already used this approval: report its
        # result rather than running it again.
        if record.outcome is Outcome.EXECUTING:
            return ConsumedApproval(
                False, record, "The approved call already ran; its result is unknown"
            )
        return ConsumedApproval(
            False,
            record,
            None if record.outcome_body is not None else "The approved call already ran",
            replay=record.outcome_body,
        )
    binding = current_binding(store, conversation_store, request, subject, record)
    attempt: ConsumeResult = store.consume(
        elicitation_id, binding_digest=binding.digest, operation_id=subject.operation_id
    )
    if attempt.consumed:
        return ConsumedApproval(True, attempt.record)
    if attempt.record is not None and attempt.record.state is VerdictState.APPROVED:
        changed = sorted(
            key
            for key, value in binding.fields.items()
            if attempt.record.binding.get(key) != value
        )
        if changed:
            _logger.info(
                "approval %s no longer matches its action: %s changed",
                elicitation_id,
                ", ".join(changed),
            )
    return ConsumedApproval(False, attempt.record, attempt.reason)


async def consume_protected_approval(
    request: HTTPConnection | None,
    conversation_store: ConversationStore,
    subject: ApprovalSubject,
    *,
    elicitation_id: str,
) -> ConsumedApproval | None:
    """Consume a recorded approval for exactly this action, once.

    :returns: ``None`` when the approval was never recorded (no store, or an
        ASK raised before the ledger existed); otherwise the outcome.
    """
    store = get_approval_store()
    if store is None:
        return None
    record = await asyncio.to_thread(store.get_verdict, elicitation_id)
    if record is None:
        return None
    return await asyncio.to_thread(
        _consume_sync, store, conversation_store, request, subject, elicitation_id
    )


async def record_operation_outcome(
    elicitation_id: str, *, succeeded: bool, body: str | None
) -> None:
    """Record the consuming operation's result for later retries."""
    store = get_approval_store()
    if store is None:
        return
    await asyncio.to_thread(
        store.record_outcome,
        elicitation_id,
        outcome=Outcome.SUCCEEDED if succeeded else Outcome.FAILED,
        body=body,
    )


# ── Answers ────────────────────────────────────────────────────────


def answer_credential_problem(
    request: HTTPConnection,
    auth_provider: AuthProvider | None,
    action: str = "answer approvals",
) -> str | None:
    """Why this request's credential may not answer a protected approval.

    :param action: What the caller tried, completing "cannot ...".
    :returns: ``None`` when it may: auth is off (one local user), or a person
        is signed in through a browser or a trusted proxy.
    """
    if auth_provider is None:
        return None
    principal = auth_provider.principal(request)
    if principal is None:
        return f"Sign in to the Omnigent web page to {action}."
    if principal.is_interactive_human:
        return None
    instead = f"{action[:1].upper()}{action[1:]} in the Omnigent web page instead."
    if principal.credential is CredentialKind.SESSION_COOKIE:
        return (
            f"Sign in again in the browser to {action}. This browser session was not "
            "started by a sign-in in the browser: it predates approval sign-ins, or it "
            "came from a link the CLI opened."
        )
    if principal.is_human:
        return f"A CLI or host sign-in cannot {action}. {instead}"
    return f"A {principal.kind.value.replace('_', ' ')} credential cannot {action}. {instead}"


@dataclasses.dataclass(frozen=True)
class AnswerGate:
    """How an answer to an approval prompt is handled.

    :param protected: The prompt is a recorded protected approval.
    :param final: The approval is now decided; resolve the prompt.
    :param action: The decision to deliver when final: ``"accept"`` or
        ``"decline"``.
    :param awaiting: Requirements still unanswered, e.g. ``["team"]``.
    """

    protected: bool
    final: bool = False
    action: str | None = None
    awaiting: tuple[str, ...] = ()


def _eligible(
    requirement: dict[str, Any],
    answerer: str | None,
    *,
    owner: str | None,
    initiator: str | None,
    approvers: dict[ApprovalResource, ResourceApprovers],
) -> bool:
    resource = ApprovalResource.from_json(requirement["resource"])
    named = approvers[resource]
    if requirement["kind"] == PERSONAL_NEED:
        is_owner = answerer is not None and answerer == owner
        if not is_owner and answerer not in named.approvers:
            return False
        # The owner approves their own work by default; a thread that refuses
        # its initiator refuses its owner's own actions too.
        return answerer != initiator or named.initiator_may_approve()
    if resource.kind is ResourceKind.SERVICE and not named.approvers:
        named = approvers[TEAM_RESOURCE]
    if answerer not in named.approvers:
        return False
    return answerer != initiator or named.initiator_may_approve()


def _gate_sync(
    store: SqlAlchemyApprovalStore,
    conversation_store: ConversationStore,
    permission_store: PermissionStore | None,
    record: VerdictRecord,
    answerer: str | None,
    action: object,
    auth_on: bool,
) -> AnswerGate:
    now = int(time.time())
    if record.state is not VerdictState.PENDING:
        raise OmnigentError("This approval was already decided.", code=ErrorCode.CONFLICT)
    if record.expires_at <= now:
        raise OmnigentError(
            "This approval expired; it must be asked again.", code=ErrorCode.CONFLICT
        )
    requirements = record.requirements
    if not auth_on:
        satisfied = set(range(len(requirements)))
    else:
        owner = _owner(conversation_store, record.session_id)
        if owner != record.binding.get("owner"):
            raise OmnigentError(
                "The thread's owner changed; the approval must be asked again.",
                code=ErrorCode.CONFLICT,
            )
        resources = requirement_resources(requirements)
        approvers = {entry.resource: entry for entry in store.approvers_for(resources)}
        initiator = record.binding.get("initiator")
        satisfied = {
            index
            for index, requirement in enumerate(requirements)
            if _eligible(
                requirement, answerer, owner=owner, initiator=initiator, approvers=approvers
            )
        }
        if not satisfied:
            can_see = permission_store is not None and check_session_access(
                answerer, record.session_id, LEVEL_READ, permission_store, conversation_store
            )
            if not can_see:
                raise OmnigentError("Conversation not found", code=ErrorCode.NOT_FOUND)
            raise OmnigentError(
                "You are not an approver for this request.", code=ErrorCode.FORBIDDEN
            )
        # An answer the changed approvers or rules would refuse at use is refused now.
        if store.authority_fingerprints(requirements, owner) != (
            record.binding.get("grants"),
            record.binding.get("policies"),
        ):
            raise OmnigentError(
                "The approvers or protected rules changed; the approval must be asked again.",
                code=ErrorCode.CONFLICT,
            )
    decline = action != "accept"
    answer = {
        "by": answerer,
        "action": "accept" if not decline else "decline",
        "satisfies": sorted(satisfied),
        "at": now,
    }
    # Completion is decided from the locked row's answers, not this snapshot.
    updated = store.record_answer(record.elicitation_id, answer=answer, decline=decline, now=now)
    if updated is None:
        raise OmnigentError(
            "This approval was already decided or expired.", code=ErrorCode.CONFLICT
        )
    if updated.state is VerdictState.PENDING:
        done = {index for item in updated.answers for index in item.get("satisfies", [])}
        awaiting = tuple(
            requirements[index]["kind"] for index in range(len(requirements)) if index not in done
        )
        return AnswerGate(True, final=False, awaiting=awaiting)
    return AnswerGate(
        True,
        final=True,
        action="accept" if updated.state is VerdictState.APPROVED else "decline",
    )


async def gate_approval_answer(
    request: HTTPConnection,
    *,
    session_id: str,
    elicitation_id: object,
    action: object,
    auth_provider: AuthProvider | None,
    permission_store: PermissionStore | None,
    conversation_store: ConversationStore,
) -> AnswerGate:
    """Check and record an answer to an approval prompt.

    Prompts that are not recorded protected approvals (questions, forms,
    plan reviews, harness permission prompts) are left to the caller's
    ordinary checks.

    :raises OmnigentError: ``FORBIDDEN`` for a credential that may not
        answer, an answerer who is not an approver, or an answer posted
        through another thread than the one that asked; ``NOT_FOUND`` when they
        cannot see the thread either; ``CONFLICT`` when the approval was
        already decided, expired, or its thread changed owner.
    """
    store = get_approval_store()
    if store is None or not isinstance(elicitation_id, str) or not elicitation_id:
        return AnswerGate(False)
    record = await asyncio.to_thread(store.get_verdict, elicitation_id)
    if record is None:
        return AnswerGate(False)
    if record.session_id != session_id:
        # A runner resolves approvals by id alone; an answer posted through
        # another thread would reach it without these checks.
        raise OmnigentError(
            "This approval belongs to another thread; answer it there.",
            code=ErrorCode.FORBIDDEN,
        )
    problem = answer_credential_problem(request, auth_provider)
    if problem is not None:
        raise OmnigentError(problem, code=ErrorCode.FORBIDDEN)
    answerer = auth_provider.get_user_id(request) if auth_provider is not None else None
    return await asyncio.to_thread(
        _gate_sync,
        store,
        conversation_store,
        permission_store,
        record,
        answerer,
        action,
        auth_provider is not None,
    )


def may_view_approval(
    store: SqlAlchemyApprovalStore,
    conversation_store: ConversationStore,
    record: VerdictRecord,
    user_id: str | None,
) -> bool:
    """Whether *user_id* may answer some requirement of a pending approval."""
    if record.state is not VerdictState.PENDING:
        return False
    owner = _owner(conversation_store, record.session_id)
    resources = requirement_resources(record.requirements)
    approvers = {entry.resource: entry for entry in store.approvers_for(resources)}
    return any(
        _eligible(
            requirement,
            user_id,
            owner=owner,
            initiator=record.binding.get("initiator"),
            approvers=approvers,
        )
        for requirement in record.requirements
    )


__all__ = [
    "CONSUME_GRACE_SECONDS",
    "AnswerGate",
    "ApprovalBinding",
    "ApprovalSubject",
    "ConsumedApproval",
    "answer_credential_problem",
    "canonical_digest",
    "consume_protected_approval",
    "current_binding",
    "final_check_problem",
    "gate_approval_answer",
    "may_view_approval",
    "record_operation_outcome",
    "record_protected_approval",
    "replay_matches",
    "turn_initiator",
]
