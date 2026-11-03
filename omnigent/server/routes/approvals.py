"""Approver inbox, approver grants and protected policies.

- ``GET /v1/approvals``: pending protected approvals the caller may answer.
- ``GET /v1/sessions/{id}/approvals``: a thread's approvals (its sub-agents'
  included) with who asked and who answered under which grant, for anyone who
  can read the thread. ``GET /v1/sessions/{id}/approvals/{elicitation_id}``:
  one of them, also for the approvers named to answer it.
- ``GET/POST/DELETE /v1/approvers``: named approvers for a thread, a service
  or the team; ``PUT /v1/approval-settings``: whether the person who started
  an action may approve it. A thread's actual owner manages its approvers
  (they only add people who may approve that owner's own work); services and
  the team are managed by human administrators only. Neither the admin flag
  nor edit access manages another person's thread.
- ``GET/POST/PATCH/DELETE /v1/protected-policies``: team rules (human
  administrators) and personal rules (their person only).

Writes need a person signed in through a browser: a CLI sign-in, device
grant or machine credential cannot change who approves or what is protected.
Administrator status grants these powers, never approval itself.
"""

from __future__ import annotations

import asyncio
from typing import Any, Literal

from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict, Field

from omnigent.errors import ErrorCode, OmnigentError
from omnigent.policies.registry import is_registered_handler, validate_factory_params
from omnigent.server.approval_authority import (
    answer_credential_problem,
    approval_view,
    may_read_approval,
    may_view_approval,
)
from omnigent.server.auth import (
    LEVEL_READ,
    RESERVED_USER_LOCAL,
    RESERVED_USER_PUBLIC,
    AuthProvider,
)
from omnigent.server.permissions import check_session_access
from omnigent.server.routes._auth_helpers import get_user_id
from omnigent.stores import ConversationStore
from omnigent.stores.approval_store import (
    TEAM_RESOURCE,
    Acting,
    ApprovalResource,
    PolicyDomain,
    ProtectedRule,
    ResourceKind,
    SqlAlchemyApprovalStore,
)
from omnigent.stores.permission_store import PermissionStore

ResourceKindName = Literal["session", "service", "team"]
_KINDS: dict[str, ResourceKind] = {
    "session": ResourceKind.SESSION,
    "service": ResourceKind.SERVICE,
    "team": ResourceKind.TEAM,
}
_KIND_NAMES = {value: key for key, value in _KINDS.items()}
_RESERVED_APPROVERS = frozenset({RESERVED_USER_LOCAL, RESERVED_USER_PUBLIC})


class _Resource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: ResourceKindName
    resource_id: str = Field(default="", max_length=256)


class ApproverRequest(_Resource):
    """Name an approver for a resource."""

    user_id: str = Field(min_length=1, max_length=128)


class ApprovalSettingsRequest(_Resource):
    """Whether the initiator may approve; ``null`` restores the default."""

    allow_initiator: bool | None = None


class ProtectedPolicyRequest(BaseModel):
    """Create a protected rule."""

    model_config = ConfigDict(extra="forbid")

    domain: Literal["team", "personal"]
    name: str = Field(min_length=1, max_length=256)
    handler: str = Field(min_length=1, max_length=512)
    factory_params: dict[str, Any] | None = None
    approval: _Resource | None = None
    enabled: bool = True


class ProtectedPolicyUpdate(BaseModel):
    """Change a protected rule."""

    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(default=None, min_length=1, max_length=256)
    factory_params: dict[str, Any] | None = None
    enabled: bool | None = None


def _resource(kind: str, resource_id: str) -> ApprovalResource:
    try:
        return ApprovalResource(_KINDS[kind], resource_id)
    except (KeyError, ValueError) as exc:
        raise OmnigentError(
            "a thread or service needs an id; the team has none", code=ErrorCode.INVALID_INPUT
        ) from exc


def _rule_response(rule: ProtectedRule) -> dict[str, Any]:
    return {
        "id": rule.id,
        "object": "protected_policy",
        "domain": "team" if rule.domain is PolicyDomain.TEAM else "personal",
        "owner": rule.owner_user_id,
        "name": rule.name,
        "handler": rule.handler,
        "factory_params": rule.factory_params,
        "approval": (
            {"kind": _KIND_NAMES[rule.approval.kind], "resource_id": rule.approval.resource_id}
            if rule.approval is not None
            else None
        ),
        "enabled": rule.enabled,
        "created_by": rule.created_by,
        "created_at": rule.created_at,
        "updated_at": rule.updated_at,
    }


def _validate_handler(handler: str, factory_params: dict[str, Any] | None) -> None:
    if not is_registered_handler(handler):
        raise OmnigentError(
            f"Policy handler {handler!r} is not registered; protected rules use "
            "handlers from the policy registry.",
            code=ErrorCode.INVALID_INPUT,
        )
    problem = validate_factory_params(handler, factory_params)
    if problem:
        raise OmnigentError(problem, code=ErrorCode.INVALID_INPUT)


# A thread tree's approvals are read for at most this many threads.
_MAX_TREE_THREADS = 256


def _thread_tree(conversation_store: ConversationStore, session_id: str) -> list[str]:
    """The thread and its sub-agents, breadth first, bounded."""
    tree = [session_id]
    frontier = [session_id]
    while frontier and len(tree) < _MAX_TREE_THREADS:
        children = conversation_store.list_child_conversation_ids_by_parent(frontier)
        frontier = [
            child for parent in frontier for child in children.get(parent, []) if child not in tree
        ][: _MAX_TREE_THREADS - len(tree)]
        tree.extend(frontier)
    return tree


def _acting(resource: ApprovalResource) -> Acting:
    """The authority a change to *resource* needs, rechecked inside the write."""
    if resource.kind is ResourceKind.SESSION:
        return Acting("thread_owner", thread_id=resource.resource_id)
    return Acting("admin")


def create_approvals_router(
    store: SqlAlchemyApprovalStore,
    conversation_store: ConversationStore,
    *,
    auth_provider: AuthProvider | None,
    permission_store: PermissionStore | None,
) -> APIRouter:
    """Build the approvals router (mounted under ``/v1``)."""
    router = APIRouter()

    def _person(request: Request, *, write: bool) -> str | None:
        user_id = get_user_id(request, auth_provider)
        if auth_provider is not None and user_id is None:
            raise OmnigentError("Authentication required", code=ErrorCode.UNAUTHORIZED)
        if write:
            problem = answer_credential_problem(request, auth_provider, "change approvals")
            if problem is not None:
                raise OmnigentError(problem, code=ErrorCode.FORBIDDEN)
        return user_id

    async def _admin(request: Request, *, write: bool) -> str | None:
        user_id = _person(request, write=write)
        if permission_store is None or user_id is None:
            return user_id
        if not await asyncio.to_thread(permission_store.is_admin, user_id):
            raise OmnigentError(
                "Administrator privileges required to manage approvals",
                code=ErrorCode.FORBIDDEN,
            )
        return user_id

    async def _resource_manager(
        request: Request, resource: ApprovalResource, *, write: bool
    ) -> str | None:
        """Who may manage *resource*: its owner for a thread, else an administrator."""
        if resource.kind is not ResourceKind.SESSION:
            return await _admin(request, write=write)
        user_id = _person(request, write=write)
        if permission_store is None or user_id is None:
            return user_id
        thread = await asyncio.to_thread(conversation_store.get_conversation, resource.resource_id)
        if thread is None:
            raise OmnigentError("Conversation not found", code=ErrorCode.NOT_FOUND)
        owner = await asyncio.to_thread(
            conversation_store.get_actual_session_owner, resource.resource_id
        )
        if owner is not None and owner.user_id == user_id:
            if thread.root_conversation_id not in (None, thread.id):
                raise OmnigentError(
                    "A sub-agent's approvals belong to its top-level thread; name them there.",
                    code=ErrorCode.INVALID_INPUT,
                )
            return user_id
        # Reading a thread's approvers needs only access to the thread.
        can_see = await asyncio.to_thread(
            check_session_access,
            user_id,
            resource.resource_id,
            LEVEL_READ,
            permission_store,
            conversation_store,
        )
        if not can_see:
            raise OmnigentError("Conversation not found", code=ErrorCode.NOT_FOUND)
        if write:
            raise OmnigentError(
                "Only the thread's owner names its approvers.", code=ErrorCode.FORBIDDEN
            )
        return user_id

    @router.get("/approvals")
    async def list_pending_approvals(request: Request) -> dict[str, Any]:
        """Pending protected approvals the caller may answer, newest first."""
        user_id = _person(request, write=False)
        records = await asyncio.to_thread(store.pending_verdicts)
        data = []
        for record in records:
            if auth_provider is not None and not await asyncio.to_thread(
                may_view_approval, store, conversation_store, record, user_id
            ):
                continue
            data.append(approval_view(record))
        return {"object": "list", "data": data}

    @router.get("/sessions/{session_id}/approvals")
    async def list_session_approvals(request: Request, session_id: str) -> dict[str, Any]:
        """A thread's approvals, newest first: who asked and who answered as what."""
        user_id = _person(request, write=False)
        thread = await asyncio.to_thread(conversation_store.get_conversation, session_id)
        if thread is None:
            raise OmnigentError("Conversation not found", code=ErrorCode.NOT_FOUND)
        if permission_store is not None and not await asyncio.to_thread(
            check_session_access,
            user_id,
            session_id,
            LEVEL_READ,
            permission_store,
            conversation_store,
        ):
            raise OmnigentError("Conversation not found", code=ErrorCode.NOT_FOUND)
        tree = await asyncio.to_thread(_thread_tree, conversation_store, session_id)
        records = await asyncio.to_thread(store.session_verdicts, tree)
        return {"object": "list", "data": [approval_view(record) for record in records]}

    @router.get("/sessions/{session_id}/approvals/{elicitation_id}")
    async def get_session_approval(
        request: Request, session_id: str, elicitation_id: str
    ) -> dict[str, Any]:
        """One approval of a thread (or its sub-agents): for readers and its approvers."""
        user_id = _person(request, write=False)
        record = await asyncio.to_thread(store.get_verdict, elicitation_id)
        if record is None:
            raise OmnigentError("Approval not found", code=ErrorCode.NOT_FOUND)
        tree = await asyncio.to_thread(_thread_tree, conversation_store, session_id)
        if record.session_id not in tree:
            raise OmnigentError("Approval not found", code=ErrorCode.NOT_FOUND)
        if permission_store is not None:
            reads = await asyncio.to_thread(
                check_session_access,
                user_id,
                session_id,
                LEVEL_READ,
                permission_store,
                conversation_store,
            )
            if not reads and not await asyncio.to_thread(
                may_read_approval, store, conversation_store, record, user_id
            ):
                raise OmnigentError("Approval not found", code=ErrorCode.NOT_FOUND)
        return approval_view(record)

    @router.get("/approvers")
    async def list_approvers(
        request: Request, kind: ResourceKindName, resource_id: str = ""
    ) -> dict[str, Any]:
        """A resource's named approvers and initiator setting."""
        resource = _resource(kind, resource_id)
        await _resource_manager(request, resource, write=False)
        resources = [resource] if resource == TEAM_RESOURCE else [resource, TEAM_RESOURCE]
        entries = await asyncio.to_thread(store.approvers_for, resources)
        entry = entries[0]
        return {
            "kind": kind,
            "resource_id": resource.resource_id,
            "approvers": list(entry.approvers),
            "allow_initiator": entry.initiator_may_approve(),
            "allow_initiator_saved": entry.allow_initiator,
            # A service with no named approvers falls back to the team's.
            "effective_approvers": list(
                entry.approvers
                or (entries[1].approvers if resource.kind is ResourceKind.SERVICE else ())
            ),
        }

    @router.post("/approvers", status_code=201)
    async def add_approver(request: Request, body: ApproverRequest) -> dict[str, Any]:
        """Name an approver for a thread, a service or the team."""
        if body.user_id in _RESERVED_APPROVERS:
            raise OmnigentError(f"{body.user_id!r} cannot approve", code=ErrorCode.INVALID_INPUT)
        resource = _resource(body.kind, body.resource_id)
        manager = await _resource_manager(request, resource, write=True)
        added = await asyncio.to_thread(
            lambda: store.add_approver(
                resource, body.user_id, granted_by=manager, acting=_acting(resource)
            )
        )
        return {"added": added}

    @router.delete("/approvers")
    async def remove_approver(
        request: Request, kind: ResourceKindName, user_id: str, resource_id: str = ""
    ) -> dict[str, Any]:
        """Stop someone approving for a resource; pending approvals bound to it lapse."""
        resource = _resource(kind, resource_id)
        await _resource_manager(request, resource, write=True)
        removed = await asyncio.to_thread(
            lambda: store.remove_approver(resource, user_id, acting=_acting(resource))
        )
        return {"removed": removed}

    @router.put("/approval-settings")
    async def save_approval_settings(
        request: Request, body: ApprovalSettingsRequest
    ) -> dict[str, Any]:
        """Save whether the person who started an action may approve it."""
        resource = _resource(body.kind, body.resource_id)
        manager = await _resource_manager(request, resource, write=True)
        await asyncio.to_thread(
            lambda: store.set_allow_initiator(
                resource, body.allow_initiator, updated_by=manager, acting=_acting(resource)
            )
        )
        (entry,) = await asyncio.to_thread(store.approvers_for, [resource])
        return {"allow_initiator": entry.initiator_may_approve()}

    async def _may_manage(request: Request, rule: ProtectedRule, *, write: bool) -> None:
        if rule.domain is PolicyDomain.TEAM:
            await _admin(request, write=write)
            return
        user_id = _person(request, write=write)
        if auth_provider is not None and user_id != rule.owner_user_id:
            raise OmnigentError("Protected policy not found", code=ErrorCode.NOT_FOUND)

    @router.get("/protected-policies")
    async def list_protected_policies(
        request: Request, domain: Literal["team", "personal"] = "personal"
    ) -> dict[str, Any]:
        """The team's rules (administrators) or the caller's own personal rules."""
        if domain == "team":
            await _admin(request, write=False)
            rules = await asyncio.to_thread(store.list_rules, domain=PolicyDomain.TEAM)
        else:
            user_id = _person(request, write=False) or RESERVED_USER_LOCAL
            rules = await asyncio.to_thread(
                store.list_rules, domain=PolicyDomain.PERSONAL, owner_user_id=user_id
            )
        return {"object": "list", "data": [_rule_response(rule) for rule in rules]}

    @router.post("/protected-policies", status_code=201)
    async def create_protected_policy(
        request: Request, body: ProtectedPolicyRequest
    ) -> dict[str, Any]:
        """Create a team rule (administrators) or a personal rule for the caller."""
        _validate_handler(body.handler, body.factory_params)
        approval = (
            _resource(body.approval.kind, body.approval.resource_id) if body.approval else None
        )
        if body.domain == "team":
            creator = await _admin(request, write=True)
            domain, owner = PolicyDomain.TEAM, None
        else:
            creator = _person(request, write=True)
            domain, owner = PolicyDomain.PERSONAL, creator or RESERVED_USER_LOCAL
        rule = await asyncio.to_thread(
            lambda: store.create_rule(
                domain=domain,
                owner_user_id=owner,
                name=body.name,
                handler=body.handler,
                factory_params=body.factory_params,
                approval=approval,
                created_by=creator,
                enabled=body.enabled,
                acting=(
                    Acting("admin")
                    if domain is PolicyDomain.TEAM
                    else Acting("rule_owner", owner_user_id=owner)
                ),
            )
        )
        return _rule_response(rule)

    async def _existing(policy_id: str) -> ProtectedRule:
        rule = await asyncio.to_thread(store.get_rule, policy_id)
        if rule is None:
            raise OmnigentError("Protected policy not found", code=ErrorCode.NOT_FOUND)
        return rule

    @router.patch("/protected-policies/{policy_id}")
    async def update_protected_policy(
        request: Request, policy_id: str, body: ProtectedPolicyUpdate
    ) -> dict[str, Any]:
        """Rename, re-parameterise, enable or disable a protected rule."""
        rule = await _existing(policy_id)
        await _may_manage(request, rule, write=True)
        if body.factory_params is not None:
            _validate_handler(rule.handler, body.factory_params)
        updated = await asyncio.to_thread(
            lambda: store.update_rule(
                policy_id,
                name=body.name,
                factory_params=body.factory_params,
                enabled=body.enabled,
                # The store derives the authority from the rule it locked.
                acting=Acting("admin"),
            )
        )
        if updated is None:
            raise OmnigentError("Protected policy not found", code=ErrorCode.NOT_FOUND)
        return _rule_response(updated)

    @router.delete("/protected-policies/{policy_id}")
    async def delete_protected_policy(request: Request, policy_id: str) -> dict[str, Any]:
        """Delete a protected rule."""
        rule = await _existing(policy_id)
        await _may_manage(request, rule, write=True)
        deleted = await asyncio.to_thread(
            lambda: store.delete_rule(policy_id, acting=Acting("admin"))
        )
        return {"deleted": deleted}

    return router


__all__ = ["create_approvals_router"]
