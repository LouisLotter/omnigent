"""Approver grants, protected policies and the protected-approval ledger.

Approval authority is separate from a thread's read/edit/manage/owner levels.
A person approves their own work; named approvers answer for a thread, a
service or the team. Protected policies are rules a person or the team's
administrators keep, which session- or agent-added policies cannot weaken.

The ledger holds one row per protected approval. The row binds what was asked
by digest, records each answer, and lets exactly one operation consume an
approval: consumption is a conditional update, so of several concurrent
attempts only one succeeds, on every backend.

Every change to who approves or what is protected, and every consumption,
first locks the workspace's team row (the lock membership writes take). A
consumption rereads the approvers and rules under that lock, so a grant
removed or a rule changed before it commits always refuses it.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import time
import uuid
from collections.abc import Iterable, Sequence
from enum import IntEnum
from typing import Any, Literal, cast

from sqlalchemy import CursorResult, and_, delete, or_, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from omnigent.db.account_authority import (
    current_account_user,
    lock_account,
    require_active_account,
)
from omnigent.db.db_models import (
    SqlApprovalResource,
    SqlApprovalVerdict,
    SqlApproverGrant,
    SqlProtectedPolicy,
    SqlSessionPermission,
    SqlUser,
    current_workspace_id,
)
from omnigent.db.utils import (
    get_or_create_engine,
    make_named_managed_session_maker,
    run_write_transaction,
)
from omnigent.errors import ErrorCode, OmnigentError
from omnigent.stores.team_store import lock_team

# A recorded operation result larger than this is not kept; a retry then
# learns the operation completed, without its result.
MAX_OUTCOME_BODY_CHARS = 48_000


class ResourceKind(IntEnum):
    """What an approver grant or approval setting is for."""

    SESSION = 1
    SERVICE = 2
    TEAM = 3


class PolicyDomain(IntEnum):
    """Who keeps a protected policy."""

    TEAM = 1
    PERSONAL = 2


class VerdictState(IntEnum):
    """Where a protected approval stands."""

    PENDING = 1
    APPROVED = 2
    DECLINED = 3
    CONSUMED = 4


class Outcome(IntEnum):
    """The consuming operation's result."""

    EXECUTING = 1
    SUCCEEDED = 2
    FAILED = 3


@dataclasses.dataclass(frozen=True)
class ApprovalResource:
    """A thread, a service, or the whole team.

    :param kind: The resource kind.
    :param resource_id: Thread id or service id; ``""`` for the team.
    """

    kind: ResourceKind
    resource_id: str = ""

    def __post_init__(self) -> None:
        if self.kind is ResourceKind.TEAM and self.resource_id:
            raise ValueError("the team resource has no id")
        if self.kind is not ResourceKind.TEAM and not self.resource_id:
            raise ValueError("a thread or service resource needs an id")

    @property
    def cksum(self) -> bytes:
        """The resource's storage key."""
        return hashlib.sha256(f"{int(self.kind)}:{self.resource_id}".encode()).digest()

    def as_json(self) -> list[object]:
        """``[kind, resource_id]`` for a JSON record."""
        return [int(self.kind), self.resource_id]

    @classmethod
    def from_json(cls, value: object) -> ApprovalResource:
        """Inverse of :meth:`as_json`."""
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError("invalid approval resource")
        kind, resource_id = value
        if not isinstance(kind, int) or not isinstance(resource_id, str):
            raise ValueError("invalid approval resource")
        return cls(ResourceKind(kind), resource_id)


TEAM_RESOURCE = ApprovalResource(ResourceKind.TEAM)


def canonical_digest(value: object) -> str:
    """sha256 of *value* as canonical JSON."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def requirement_resources(requirements: Sequence[dict[str, Any]]) -> list[ApprovalResource]:
    """The resources whose approvers decide *requirements*.

    A service without named approvers falls back to the team's, so the team is
    included whenever a service is.
    """
    resources = [ApprovalResource.from_json(req["resource"]) for req in requirements]
    if any(r.kind is ResourceKind.SERVICE for r in resources):
        resources.append(TEAM_RESOURCE)
    return list(dict.fromkeys(resources))


@dataclasses.dataclass(frozen=True)
class ResourceApprovers:
    """A resource's named approvers and initiator setting.

    :param resource: The resource.
    :param approvers: Named approvers, sorted.
    :param allow_initiator: The saved initiator setting, or ``None`` for the
        default (allowed).
    """

    resource: ApprovalResource
    approvers: tuple[str, ...]
    allow_initiator: bool | None

    def initiator_may_approve(self) -> bool:
        """Whether the person who started an action may approve it.

        Yes by default, for a thread, a service and the team alike (D59): an
        approval records a person's consent, not a separation of duties. An
        administrator (or a thread's owner) can turn it off per resource; it
        then excludes, as best effort, the people attribution recorded as
        having taken part in the turn.
        """
        if self.allow_initiator is not None:
            return self.allow_initiator
        return True

    def fingerprint(self) -> str:
        """Digest of who may answer, for binding a verdict to it."""
        payload = json.dumps(
            [self.resource.as_json(), list(self.approvers), self.initiator_may_approve()]
        )
        return hashlib.sha256(payload.encode()).hexdigest()


@dataclasses.dataclass(frozen=True)
class ProtectedRule:
    """A protected policy.

    :param id: Rule id (32-hex).
    :param domain: Team or personal.
    :param owner_user_id: The person a personal rule belongs to.
    :param name: Display name.
    :param handler: Registered policy handler path.
    :param factory_params: Factory arguments, or ``None``.
    :param approval: Who answers a team rule's ASK, or ``None`` for the team's
        approvers (personal rules always go to their person).
    :param enabled: Whether it runs.
    :param created_by: Who created it.
    :param created_at: Creation time, epoch seconds.
    :param updated_at: Last change, epoch seconds.
    """

    id: str
    domain: PolicyDomain
    owner_user_id: str | None
    name: str
    handler: str
    factory_params: dict[str, Any] | None
    approval: ApprovalResource | None
    enabled: bool
    created_by: str | None
    created_at: int
    updated_at: int

    def fingerprint(self) -> str:
        """Digest of everything that decides this rule's verdicts."""
        payload = json.dumps(
            [
                self.id,
                int(self.domain),
                self.owner_user_id,
                self.handler,
                self.factory_params,
                self.approval.as_json() if self.approval else None,
                self.enabled,
                self.updated_at,
            ],
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(payload.encode()).hexdigest()


def rules_fingerprint(rules: Sequence[ProtectedRule]) -> str:
    """Digest of the protected rules in force, for binding a verdict to them."""
    return canonical_digest(sorted(rule.fingerprint() for rule in rules))


def grants_fingerprint(approvers: Sequence[ResourceApprovers]) -> str:
    """Digest of who may answer for some resources, for binding a verdict to them."""
    return canonical_digest([entry.fingerprint() for entry in approvers])


@dataclasses.dataclass(frozen=True)
class VerdictRecord:
    """One protected approval in the ledger.

    :param elicitation_id: The approval prompt's id.
    :param session_id: The thread that asked.
    :param binding_digest: Digest of the canonical binding.
    :param binding: Non-secret binding fields.
    :param requirements: Approvals required, as JSON objects.
    :param answers: Answers recorded so far.
    :param state: Pending, approved, declined or consumed.
    :param decided_by: Who completed the decision.
    :param decided_at: When, epoch seconds.
    :param expires_at: After this no answer or consumption counts.
    :param consumed_at: When the one consumption happened.
    :param operation_id: The consuming operation.
    :param outcome: The consuming operation's result state.
    :param outcome_body: The recorded result, when kept.
    :param created_at: When it was asked.
    :param attribution: Who asked: the turn's initiator and contributors, the
        asking principal, the thread's owner, computer and harness login.
        ``None`` for rows recorded before attribution existed.
    """

    elicitation_id: str
    session_id: str
    binding_digest: bytes
    binding: dict[str, Any]
    requirements: list[dict[str, Any]]
    answers: list[dict[str, Any]]
    state: VerdictState
    decided_by: str | None
    decided_at: int | None
    expires_at: int
    consumed_at: int | None
    operation_id: str | None
    outcome: Outcome | None
    outcome_body: str | None
    created_at: int
    attribution: dict[str, Any] | None = None


@dataclasses.dataclass(frozen=True)
class ConsumeResult:
    """The result of trying to consume an approval.

    :param consumed: Whether this caller consumed it.
    :param record: The ledger row after the attempt, or ``None`` when absent.
    :param reason: Why it was not consumed, when it was not.
    """

    consumed: bool
    record: VerdictRecord | None
    reason: str | None = None


def _now() -> int:
    return int(time.time())


def _json_list(text: str | None) -> list[Any]:
    if not text:
        return []
    value = json.loads(text)
    return value if isinstance(value, list) else []


def _to_rule(row: SqlProtectedPolicy) -> ProtectedRule:
    approval = (
        ApprovalResource(ResourceKind(row.approval_kind), row.approval_resource_id or "")
        if row.approval_kind is not None
        else None
    )
    params = json.loads(row.factory_params) if row.factory_params else None
    return ProtectedRule(
        id=row.id,
        domain=PolicyDomain(row.domain),
        owner_user_id=row.owner_user_id,
        name=row.name,
        handler=row.handler,
        factory_params=params if isinstance(params, dict) else None,
        approval=approval,
        enabled=bool(row.enabled),
        created_by=row.created_by,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _to_verdict(row: SqlApprovalVerdict) -> VerdictRecord:
    binding = json.loads(row.binding)
    return VerdictRecord(
        elicitation_id=row.elicitation_id,
        session_id=row.session_id,
        binding_digest=bytes(row.binding_digest),
        binding=binding if isinstance(binding, dict) else {},
        requirements=_json_list(row.requirements),
        answers=_json_list(row.answers),
        state=VerdictState(row.state),
        decided_by=row.decided_by,
        decided_at=row.decided_at,
        expires_at=row.expires_at,
        consumed_at=row.consumed_at,
        operation_id=row.operation_id,
        outcome=Outcome(row.outcome) if row.outcome is not None else None,
        outcome_body=row.outcome_body,
        created_at=row.created_at,
        attribution=_json_object(row.attribution),
    )


def _json_object(text: str | None) -> dict[str, Any] | None:
    if not text:
        return None
    try:
        value = json.loads(text)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def _validate_rule_target(
    domain: PolicyDomain, owner_user_id: str | None, approval: ApprovalResource | None
) -> None:
    if domain is PolicyDomain.PERSONAL and not owner_user_id:
        raise OmnigentError("a personal rule needs its person", code=ErrorCode.INVALID_INPUT)
    if domain is PolicyDomain.TEAM and owner_user_id is not None:
        raise OmnigentError("a team rule has no person", code=ErrorCode.INVALID_INPUT)
    if approval is not None and (
        domain is PolicyDomain.PERSONAL or approval.kind is ResourceKind.SESSION
    ):
        raise OmnigentError(
            "only a team rule names its approvers, as a service's or the team's",
            code=ErrorCode.INVALID_INPUT,
        )


def remove_account_approval_authority(session: Session, user_id: str) -> None:
    """Inside account deletion: drop the account's approver grants and personal rules.

    The deletion transaction already holds the team lock, so a concurrent grant
    write either commits first (and is removed here) or waits and then finds
    the account deleted.
    """
    workspace = current_workspace_id()
    session.execute(
        delete(SqlApproverGrant).where(
            SqlApproverGrant.workspace_id == workspace,
            SqlApproverGrant.approver_user_id == user_id,
        )
    )
    session.execute(
        delete(SqlProtectedPolicy).where(
            SqlProtectedPolicy.workspace_id == workspace,
            SqlProtectedPolicy.domain == int(PolicyDomain.PERSONAL),
            SqlProtectedPolicy.owner_user_id == user_id,
        )
    )


def delete_session_verdicts(session: Session, session_ids: Sequence[str]) -> None:
    """Delete the approvals of threads being deleted, inside their deletion."""
    if session_ids:
        session.execute(
            delete(SqlApprovalVerdict).where(
                SqlApprovalVerdict.workspace_id == current_workspace_id(),
                SqlApprovalVerdict.session_id.in_(list(session_ids)),
            )
        )


def _lock_authority(session: Session) -> None:
    """Serialize approval authority changes and consumptions (the team row lock)."""
    lock_team(session)


@dataclasses.dataclass(frozen=True)
class Acting:
    """The authority the acting person must still hold when a write commits.

    The route checks it first; the write checks it again inside its
    transaction, after the team lock, so a request authorized before its
    actor's account was deleted (or demoted) cannot complete afterwards.

    :param role: ``"admin"`` (a team service or rule), ``"thread_owner"``
        (the thread's actual owner) or ``"rule_owner"`` (a personal rule's
        person).
    :param thread_id: The thread, for ``"thread_owner"``.
    :param owner_user_id: The rule's person, for ``"rule_owner"``.
    """

    role: Literal["admin", "thread_owner", "rule_owner"]
    thread_id: str | None = None
    owner_user_id: str | None = None


def _require_acting(session: Session, acting: Acting | None) -> None:
    """Validate the acting person's registration and authority under the locks.

    The captured account generation of the request's actor is checked with
    the ordered account lock discipline; then the role is read again. A
    request without an accounts identity (authentication off, or a trusted
    proxy) relies on the route's check.

    :raises OmnigentError: ``UNAUTHORIZED`` when the actor's registration was
        revoked; ``FORBIDDEN`` when the actor no longer holds the role.
    """
    require_active_account(session, None)
    actor = current_account_user()
    if acting is None or actor is None:
        return
    if acting.role == "admin":
        row = session.get(SqlUser, (current_workspace_id(), actor))
        if row is not None and (row.deleted_at is not None or not row.is_admin):
            raise OmnigentError(
                "Administrator privileges required to manage approvals",
                code=ErrorCode.FORBIDDEN,
            )
        return
    if acting.role == "thread_owner":
        from omnigent.server.auth import LEVEL_OWNER, RESERVED_USER_PUBLIC

        owners = session.execute(
            select(SqlSessionPermission.user_id)
            .where(
                SqlSessionPermission.workspace_id == current_workspace_id(),
                SqlSessionPermission.conversation_id == acting.thread_id,
                SqlSessionPermission.level >= LEVEL_OWNER,
                SqlSessionPermission.user_id != RESERVED_USER_PUBLIC,
            )
            .limit(2)
        ).scalars()
        if list(owners) != [actor]:
            raise OmnigentError(
                "Only the thread's owner names its approvers.", code=ErrorCode.FORBIDDEN
            )
        return
    if acting.owner_user_id != actor:
        raise OmnigentError("Protected policy not found", code=ErrorCode.NOT_FOUND)


def _approvers_in(
    session: Session, resources: Iterable[ApprovalResource]
) -> list[ResourceApprovers]:
    wanted = list(dict.fromkeys(resources))
    if not wanted:
        return []
    keys = [resource.cksum for resource in wanted]
    workspace = current_workspace_id()
    grants = session.execute(
        select(SqlApproverGrant.resource_cksum, SqlApproverGrant.approver_user_id).where(
            SqlApproverGrant.workspace_id == workspace,
            SqlApproverGrant.resource_cksum.in_(keys),
        )
    ).all()
    settings = session.execute(
        select(SqlApprovalResource.resource_cksum, SqlApprovalResource.allow_initiator).where(
            SqlApprovalResource.workspace_id == workspace,
            SqlApprovalResource.resource_cksum.in_(keys),
        )
    ).all()
    by_key: dict[bytes, list[str]] = {}
    for key, user in grants:
        by_key.setdefault(bytes(key), []).append(user)
    allow = {bytes(key): value for key, value in settings}
    return [
        ResourceApprovers(
            resource, tuple(sorted(by_key.get(resource.cksum, []))), allow.get(resource.cksum)
        )
        for resource in wanted
    ]


def _rules_in(session: Session, owner_user_id: str | None) -> list[ProtectedRule]:
    scope = SqlProtectedPolicy.domain == int(PolicyDomain.TEAM)
    if owner_user_id:
        scope = or_(
            scope,
            and_(
                SqlProtectedPolicy.domain == int(PolicyDomain.PERSONAL),
                SqlProtectedPolicy.owner_user_id == owner_user_id,
            ),
        )
    rows = session.execute(
        select(SqlProtectedPolicy)
        .where(
            SqlProtectedPolicy.workspace_id == current_workspace_id(),
            SqlProtectedPolicy.enabled.is_(True),
            scope,
        )
        .order_by(SqlProtectedPolicy.domain, SqlProtectedPolicy.created_at, SqlProtectedPolicy.id)
    ).scalars()
    return [_to_rule(row) for row in rows]


def _authority_fingerprints(
    session: Session, requirements: Sequence[dict[str, Any]], owner: str | None
) -> tuple[str, str]:
    approvers = _approvers_in(session, requirement_resources(requirements))
    return grants_fingerprint(approvers), rules_fingerprint(_rules_in(session, owner))


def _acting_for(row: SqlProtectedPolicy, acting: Acting | None) -> Acting | None:
    """The authority a change to *row* needs: its person's, or an administrator's."""
    if acting is None:
        return None
    if row.domain == int(PolicyDomain.PERSONAL):
        return Acting("rule_owner", owner_user_id=row.owner_user_id)
    return Acting("admin")


class SqlAlchemyApprovalStore:
    """SQLAlchemy persistence for approver grants, protected rules and verdicts.

    :param storage_location: SQLAlchemy database URI; shares the engine of the
        other stores.
    """

    def __init__(self, storage_location: str) -> None:
        self.storage_location = storage_location
        self._engine = get_or_create_engine(storage_location)
        self._session = make_named_managed_session_maker(
            self._engine, query_name_prefix="omnigent.approval_store"
        )
        self._session_immediate = make_named_managed_session_maker(
            self._engine, query_name_prefix="omnigent.approval_store", immediate=True
        )

    # ── Approvers ──────────────────────────────────────────────────

    def approvers_for(self, resources: Iterable[ApprovalResource]) -> list[ResourceApprovers]:
        """Each resource's named approvers and initiator setting, in order."""
        with self._session("select_resource_approvers") as session:
            return _approvers_in(session, resources)

    def authority_fingerprints(
        self, requirements: Sequence[dict[str, Any]], owner: str | None
    ) -> tuple[str, str]:
        """``(grants, rules)`` fingerprints for a binding, read in one transaction."""
        with self._session("select_approval_authority") as session:
            return _authority_fingerprints(session, requirements, owner)

    def add_approver(
        self,
        resource: ApprovalResource,
        user_id: str,
        *,
        granted_by: str | None,
        acting: Acting | None = None,
    ) -> bool:
        """Name *user_id* an approver for *resource*; ``False`` if already named.

        A deleted account cannot be named: the grant would pass to whoever
        registers the username next. The account and team locks order this
        write with account deletion, which removes the account's grants.

        :param acting: The authority the actor must still hold (see :class:`Acting`).
        :raises OmnigentError: ``INVALID_INPUT`` for a deleted account; see
            :func:`_require_acting` for a revoked actor.
        """

        def write(session: Session) -> bool:
            _lock_authority(session)
            _require_acting(session, acting)
            account = lock_account(session, user_id)
            if account is not None and account.deleted_at is not None:
                raise OmnigentError(f"no account {user_id!r}", code=ErrorCode.INVALID_INPUT)
            existing = session.get(
                SqlApproverGrant, (current_workspace_id(), resource.cksum, user_id)
            )
            if existing is not None:
                return False
            session.add(
                SqlApproverGrant(
                    resource_cksum=resource.cksum,
                    approver_user_id=user_id,
                    kind=int(resource.kind),
                    resource_id=resource.resource_id,
                    granted_by=granted_by,
                    created_at=_now(),
                )
            )
            try:
                session.flush()
            except IntegrityError:
                return False
            return True

        return run_write_transaction(self._session_immediate, "insert_approver_grant", write)

    def remove_approver(
        self, resource: ApprovalResource, user_id: str, *, acting: Acting | None = None
    ) -> bool:
        """Stop *user_id* approving for *resource*; ``False`` if they were not named."""

        def write(session: Session) -> bool:
            _lock_authority(session)
            _require_acting(session, acting)
            result = cast(
                "CursorResult[Any]",
                session.execute(
                    delete(SqlApproverGrant).where(
                        SqlApproverGrant.workspace_id == current_workspace_id(),
                        SqlApproverGrant.resource_cksum == resource.cksum,
                        SqlApproverGrant.approver_user_id == user_id,
                    )
                ),
            )
            return result.rowcount > 0

        return run_write_transaction(self._session_immediate, "delete_approver_grant", write)

    def set_allow_initiator(
        self,
        resource: ApprovalResource,
        value: bool | None,
        *,
        updated_by: str | None,
        acting: Acting | None = None,
    ) -> None:
        """Save whether the initiator may approve for *resource*; ``None`` restores the default."""

        def write(session: Session) -> None:
            _lock_authority(session)
            _require_acting(session, acting)
            row = session.get(SqlApprovalResource, (current_workspace_id(), resource.cksum))
            if row is None:
                session.add(
                    SqlApprovalResource(
                        resource_cksum=resource.cksum,
                        kind=int(resource.kind),
                        resource_id=resource.resource_id,
                        allow_initiator=value,
                        updated_at=_now(),
                        updated_by=updated_by,
                    )
                )
                return
            row.allow_initiator = value
            row.updated_at = _now()
            row.updated_by = updated_by

        run_write_transaction(self._session_immediate, "upsert_approval_resource", write)

    def resources_for_approver(self, user_id: str) -> list[ApprovalResource]:
        """Every resource *user_id* is a named approver for."""
        with self._session("select_approver_resources") as session:
            rows = session.execute(
                select(SqlApproverGrant.kind, SqlApproverGrant.resource_id)
                .where(
                    SqlApproverGrant.workspace_id == current_workspace_id(),
                    SqlApproverGrant.approver_user_id == user_id,
                )
                .order_by(SqlApproverGrant.resource_cksum)
            ).all()
        return [ApprovalResource(ResourceKind(kind), resource_id) for kind, resource_id in rows]

    # ── Protected rules ────────────────────────────────────────────

    def create_rule(
        self,
        *,
        domain: PolicyDomain,
        owner_user_id: str | None,
        name: str,
        handler: str,
        factory_params: dict[str, Any] | None,
        approval: ApprovalResource | None,
        created_by: str | None,
        enabled: bool = True,
        acting: Acting | None = None,
    ) -> ProtectedRule:
        """Create a protected rule.

        A personal rule's person must be a live account: a rule written for a
        deleted username would govern whoever registers it next.
        """
        _validate_rule_target(domain, owner_user_id, approval)

        def write(session: Session) -> ProtectedRule:
            _lock_authority(session)
            _require_acting(session, acting)
            if owner_user_id is not None:
                owner = lock_account(session, owner_user_id)
                if owner is not None and owner.deleted_at is not None:
                    raise OmnigentError(
                        f"no account {owner_user_id!r}", code=ErrorCode.INVALID_INPUT
                    )
            now = _now()
            row = SqlProtectedPolicy(
                id=uuid.uuid4().hex,
                domain=int(domain),
                owner_user_id=owner_user_id,
                name=name,
                handler=handler,
                factory_params=json.dumps(factory_params) if factory_params is not None else None,
                approval_kind=int(approval.kind) if approval else None,
                approval_resource_id=approval.resource_id if approval else None,
                enabled=enabled,
                created_by=created_by,
                created_at=now,
                updated_at=now,
            )
            session.add(row)
            session.flush()
            return _to_rule(row)

        return run_write_transaction(self._session_immediate, "insert_protected_policy", write)

    def get_rule(self, rule_id: str) -> ProtectedRule | None:
        """A protected rule by id."""
        with self._session("select_protected_policy") as session:
            row = session.get(SqlProtectedPolicy, (current_workspace_id(), rule_id))
            return _to_rule(row) if row is not None else None

    def update_rule(
        self,
        rule_id: str,
        *,
        name: str | None = None,
        factory_params: dict[str, Any] | None = None,
        enabled: bool | None = None,
        acting: Acting | None = None,
    ) -> ProtectedRule | None:
        """Change a rule's name, arguments or enabled flag; ``None`` if missing."""

        def write(session: Session) -> ProtectedRule | None:
            _lock_authority(session)
            row = session.get(SqlProtectedPolicy, (current_workspace_id(), rule_id))
            if row is None:
                return None
            _require_acting(session, _acting_for(row, acting))
            if name is not None:
                row.name = name
            if factory_params is not None:
                row.factory_params = json.dumps(factory_params)
            if enabled is not None:
                row.enabled = enabled
            # Strictly increasing, so a verdict bound to the old rule never matches.
            row.updated_at = max(_now(), row.updated_at + 1)
            session.flush()
            return _to_rule(row)

        return run_write_transaction(self._session_immediate, "update_protected_policy", write)

    def delete_rule(self, rule_id: str, *, acting: Acting | None = None) -> bool:
        """Delete a rule; ``False`` if it did not exist."""

        def write(session: Session) -> bool:
            _lock_authority(session)
            row = session.get(SqlProtectedPolicy, (current_workspace_id(), rule_id))
            if row is None:
                return False
            _require_acting(session, _acting_for(row, acting))
            result = cast(
                "CursorResult[Any]",
                session.execute(
                    delete(SqlProtectedPolicy).where(
                        SqlProtectedPolicy.workspace_id == current_workspace_id(),
                        SqlProtectedPolicy.id == rule_id,
                    )
                ),
            )
            return result.rowcount > 0

        return run_write_transaction(self._session_immediate, "delete_protected_policy", write)

    def list_rules(
        self, *, domain: PolicyDomain | None = None, owner_user_id: str | None = None
    ) -> list[ProtectedRule]:
        """Rules of one domain (and person), oldest first."""
        query = select(SqlProtectedPolicy).where(
            SqlProtectedPolicy.workspace_id == current_workspace_id()
        )
        if domain is not None:
            query = query.where(SqlProtectedPolicy.domain == int(domain))
        if owner_user_id is not None:
            query = query.where(SqlProtectedPolicy.owner_user_id == owner_user_id)
        with self._session("select_protected_policies") as session:
            rows = session.execute(
                query.order_by(SqlProtectedPolicy.created_at, SqlProtectedPolicy.id)
            ).scalars()
            return [_to_rule(row) for row in rows]

    def rules_for(self, owner_user_id: str | None) -> list[ProtectedRule]:
        """The enabled rules governing a thread: the team's, then its owner's."""
        with self._session("select_governing_protected_policies") as session:
            return _rules_in(session, owner_user_id)

    def has_enabled_rules(self) -> bool:
        """Whether any protected rule is enabled in this workspace."""
        with self._session("select_any_protected_policy") as session:
            return (
                session.execute(
                    select(SqlProtectedPolicy.id)
                    .where(
                        SqlProtectedPolicy.workspace_id == current_workspace_id(),
                        SqlProtectedPolicy.enabled.is_(True),
                    )
                    .limit(1)
                ).first()
                is not None
            )

    # ── Verdict ledger ─────────────────────────────────────────────

    def record_request(
        self,
        *,
        elicitation_id: str,
        session_id: str,
        binding_digest: bytes,
        binding: dict[str, Any],
        requirements: Sequence[dict[str, Any]],
        expires_at: int,
        attribution: dict[str, Any] | None = None,
    ) -> VerdictRecord:
        """Record a protected approval request.

        Re-recording the same id with the same binding (a hook retry) returns
        the existing row. The same id with another binding is refused.
        *attribution* (who asked) is written in the same insert.

        :raises OmnigentError: ``CONFLICT`` when the id is bound to another request.
        """

        def write(session: Session) -> VerdictRecord:
            existing = session.get(SqlApprovalVerdict, (current_workspace_id(), elicitation_id))
            if existing is not None:
                if bytes(existing.binding_digest) != binding_digest:
                    raise OmnigentError(
                        "approval id is bound to another request", code=ErrorCode.CONFLICT
                    )
                return _to_verdict(existing)
            row = SqlApprovalVerdict(
                elicitation_id=elicitation_id,
                session_id=session_id,
                binding_digest=binding_digest,
                binding=json.dumps(binding, sort_keys=True),
                requirements=json.dumps(list(requirements), sort_keys=True),
                answers=None,
                state=int(VerdictState.PENDING),
                expires_at=expires_at,
                created_at=_now(),
                attribution=(
                    json.dumps(attribution, sort_keys=True) if attribution is not None else None
                ),
            )
            session.add(row)
            session.flush()
            return _to_verdict(row)

        return run_write_transaction(self._session_immediate, "insert_approval_verdict", write)

    def get_verdict(self, elicitation_id: str) -> VerdictRecord | None:
        """The ledger row for an approval prompt, or ``None``."""
        with self._session("select_approval_verdict") as session:
            row = session.get(SqlApprovalVerdict, (current_workspace_id(), elicitation_id))
            return _to_verdict(row) if row is not None else None

    def session_verdicts(
        self, session_ids: Sequence[str], *, limit: int = 200
    ) -> list[VerdictRecord]:
        """A thread's approvals (pass a sub-agent tree's ids for all of it), newest first."""
        if not session_ids:
            return []
        with self._session("select_session_approval_verdicts") as session:
            rows = session.execute(
                select(SqlApprovalVerdict)
                .where(
                    SqlApprovalVerdict.workspace_id == current_workspace_id(),
                    SqlApprovalVerdict.session_id.in_(list(session_ids)),
                )
                .order_by(SqlApprovalVerdict.created_at.desc(), SqlApprovalVerdict.elicitation_id)
                .limit(limit)
            ).scalars()
            return [_to_verdict(row) for row in rows]

    def pending_verdicts(self, *, now: int | None = None, limit: int = 200) -> list[VerdictRecord]:
        """Unexpired pending approvals, newest first."""
        at = _now() if now is None else now
        with self._session("select_pending_approval_verdicts") as session:
            rows = session.execute(
                select(SqlApprovalVerdict)
                .where(
                    SqlApprovalVerdict.workspace_id == current_workspace_id(),
                    SqlApprovalVerdict.state == int(VerdictState.PENDING),
                    SqlApprovalVerdict.expires_at > at,
                )
                .order_by(SqlApprovalVerdict.expires_at.desc())
                .limit(limit)
            ).scalars()
            return [_to_verdict(row) for row in rows]

    def record_answer(
        self,
        elicitation_id: str,
        *,
        answer: dict[str, Any],
        decline: bool,
        now: int | None = None,
    ) -> VerdictRecord | None:
        """Record one answer to a pending, unexpired approval.

        The row is locked, so concurrent answers apply one after another, and
        completion is decided from the answers the locked row holds: two
        approvers satisfying different requirements at once approve it.

        :param answer: The answer to append: who, the action, and the indexes
            of the requirements it satisfies (``"satisfies"``).
        :param decline: The answer declines; the approval is then declined.
        :returns: The updated row, or ``None`` when it is missing, no longer
            pending, or expired.
        """
        at = _now() if now is None else now

        def write(session: Session) -> VerdictRecord | None:
            row = session.execute(
                select(SqlApprovalVerdict)
                .where(
                    SqlApprovalVerdict.workspace_id == current_workspace_id(),
                    SqlApprovalVerdict.elicitation_id == elicitation_id,
                )
                .with_for_update()
                .execution_options(populate_existing=True)
            ).scalar_one_or_none()
            if row is None or row.state != int(VerdictState.PENDING) or row.expires_at <= at:
                return None
            answers = [*_json_list(row.answers), answer]
            row.answers = json.dumps(answers, sort_keys=True)
            satisfied = {index for item in answers for index in item.get("satisfies", [])}
            complete = satisfied >= set(range(len(_json_list(row.requirements))))
            if decline or complete:
                row.state = int(VerdictState.DECLINED if decline else VerdictState.APPROVED)
                row.decided_by = cast("str | None", answer.get("by"))
                row.decided_at = at
            session.flush()
            return _to_verdict(row)

        return run_write_transaction(self._session_immediate, "answer_approval_verdict", write)

    def consume(
        self,
        elicitation_id: str,
        *,
        binding_digest: bytes,
        operation_id: str | None,
        now: int | None = None,
    ) -> ConsumeResult:
        """Consume an approved, unexpired approval for exactly this binding, once.

        Under the authority lock, the approvers and protected rules the
        approval was bound to are read again: a grant removed or a rule changed
        before this commits refuses it.

        :returns: Whether this call consumed it; when not, the row and reason.
        """
        at = _now() if now is None else now

        def write(session: Session) -> str | None:
            _lock_authority(session)
            row = session.execute(
                select(SqlApprovalVerdict)
                .where(
                    SqlApprovalVerdict.workspace_id == current_workspace_id(),
                    SqlApprovalVerdict.elicitation_id == elicitation_id,
                )
                .with_for_update()
                .execution_options(populate_existing=True)
            ).scalar_one_or_none()
            if (
                row is None
                or row.state != int(VerdictState.APPROVED)
                or bytes(row.binding_digest) != binding_digest
                or row.expires_at <= at
            ):
                return _refusal(_to_verdict(row) if row is not None else None, binding_digest, at)
            binding = json.loads(row.binding)
            bound = (binding.get("grants"), binding.get("policies"))
            current = _authority_fingerprints(
                session, _json_list(row.requirements), binding.get("owner")
            )
            if current != bound:
                return "The approvers or protected rules changed after approval; ask again"
            row.state = int(VerdictState.CONSUMED)
            row.consumed_at = at
            row.operation_id = operation_id
            row.outcome = int(Outcome.EXECUTING)
            return None

        refusal = run_write_transaction(self._session_immediate, "consume_approval_verdict", write)
        record = self.get_verdict(elicitation_id)
        return ConsumeResult(refusal is None, record, refusal)

    def record_outcome(self, elicitation_id: str, *, outcome: Outcome, body: str | None) -> None:
        """Record the consuming operation's result, for retries to inspect."""
        kept = body if body is not None and len(body) <= MAX_OUTCOME_BODY_CHARS else None

        def write(session: Session) -> None:
            session.execute(
                update(SqlApprovalVerdict)
                .where(
                    SqlApprovalVerdict.workspace_id == current_workspace_id(),
                    SqlApprovalVerdict.elicitation_id == elicitation_id,
                    SqlApprovalVerdict.state == int(VerdictState.CONSUMED),
                )
                .values(outcome=int(outcome), outcome_body=kept)
            )

        run_write_transaction(self._session_immediate, "record_approval_outcome", write)

    def purge_expired(self, *, before: int) -> int:
        """Delete unanswered ledger rows that expired before *before*; returns how many.

        A row anyone answered is the record of who decided; it stays until its
        thread is deleted (see :func:`delete_session_verdicts`).
        """

        def write(session: Session) -> int:
            result = cast(
                "CursorResult[Any]",
                session.execute(
                    delete(SqlApprovalVerdict).where(
                        SqlApprovalVerdict.workspace_id == current_workspace_id(),
                        SqlApprovalVerdict.expires_at < before,
                        SqlApprovalVerdict.answers.is_(None),
                    )
                ),
            )
            return int(result.rowcount)

        return run_write_transaction(self._session_immediate, "purge_approval_verdicts", write)


def _refusal(record: VerdictRecord | None, binding_digest: bytes, now: int) -> str:
    """Why an approval could not be consumed."""
    if record is None:
        return "Approval not found"
    if record.state is VerdictState.CONSUMED:
        return "Approval was already used"
    if record.state is VerdictState.DECLINED:
        return "Tool call denied by user"
    if record.expires_at <= now:
        return "Approval expired; ask again"
    if record.state is VerdictState.PENDING:
        return "Tool call has not been approved"
    if record.binding_digest != binding_digest:
        return "The action changed after it was approved; ask again"
    return "Approval cannot be used"


__all__ = [
    "MAX_OUTCOME_BODY_CHARS",
    "TEAM_RESOURCE",
    "Acting",
    "ApprovalResource",
    "ConsumeResult",
    "Outcome",
    "PolicyDomain",
    "ProtectedRule",
    "ResourceApprovers",
    "ResourceKind",
    "SqlAlchemyApprovalStore",
    "VerdictRecord",
    "VerdictState",
    "canonical_digest",
    "delete_session_verdicts",
    "grants_fingerprint",
    "remove_account_approval_authority",
    "requirement_resources",
    "rules_fingerprint",
]
