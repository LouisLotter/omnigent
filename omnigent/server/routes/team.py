"""The team page's API: members, roles, suspension and invitations.

- ``GET /v1/team``: the team, the caller's membership, members and pending
  invitations; owners and admins only.
- ``POST /v1/team/invitations``: invite an email address with a role; the
  single-use link is returned once, for the administrator to hand over.
- ``DELETE /v1/team/invitations/{invitation_id}``: revoke a pending invitation.
- ``PUT /v1/team/members/{member_id}/role``: change a role (owners only).
- ``POST /v1/team/members/{member_id}/suspend`` and ``/reinstate``.
- ``POST /v1/team/members/{member_id}/remove``: remove a member for good, or
  repeat an earlier removal's remaining steps; returns where it stands.
- ``GET /v1/team/members/{member_id}/removal``: a removal's steps and the
  threads the member owned.
- ``GET /v1/team/invitations/preview``: what an invitation offers, for the
  person holding its link; no sign-in needed.

Who may do what: an owner invites admins and members, appoints and removes
owners and admins, and suspends or reinstates anyone but themselves; an admin
invites members and suspends or reinstates members only. The last active owner
is always kept. Every change needs a person signed in through a browser, so
no runner, host, CLI sign-in or device grant can administer the team. With an
identity-aware proxy in front (:mod:`omnigent.server.external_identity`), the
caller must also be the member the proxy's sign-in is bound to.

Suspension blocks the member's new work at once: their credentials stop
authenticating, their open streams close, their schedules skip, and their
registration rotates so every token issued before stays dead after
reinstatement.

Removal does the same at once and for good (:mod:`omnigent.server.member_removal`):
the account is removed, their cloud computers are handed to their provider,
and their threads stay readable by owners and admins. Who may remove whom
follows suspension; the last active owner and the installation's own owner
(``OMNIGENT_INSTALLATION_OWNER``) are kept.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Literal

from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict, Field

from omnigent.entities.team import MemberState, TeamInvitation, TeamMember
from omnigent.errors import ErrorCode, OmnigentError
from omnigent.server.accounts_store import SqlAlchemyAccountStore
from omnigent.server.approval_authority import answer_credential_problem
from omnigent.server.auth import AuthProvider
from omnigent.server.authority_changes import authority_changed
from omnigent.server.external_identity import (
    ExternalIdentityMode,
    external_identity_mode,
    resolved_external_identity,
)
from omnigent.server.member_removal import MemberRemovals, installation_owner
from omnigent.server.routes._auth_helpers import require_user
from omnigent.server.team_admission import AdmissionSync
from omnigent.stores.team_store import SqlAlchemyTeamStore

_logger = logging.getLogger(__name__)

InvitationRoleName = Literal["admin", "member"]
RoleName = Literal["owner", "admin", "member"]


class InvitationRequest(BaseModel):
    """Invite one address."""

    model_config = ConfigDict(extra="forbid")

    email: str = Field(min_length=3, max_length=320)
    role: InvitationRoleName = "member"
    # Link this existing account instead of creating one (owners only).
    user_id: str | None = Field(default=None, min_length=1, max_length=128)


class RoleRequest(BaseModel):
    """A member's new role."""

    model_config = ConfigDict(extra="forbid")

    role: RoleName


def _member_json(member: TeamMember) -> dict[str, object]:
    return {
        "member_id": member.id,
        "user_id": member.user_id,
        "role": member.role,
        "state": member.state,
        "created_at": member.created_at,
        "updated_at": member.updated_at,
        # The deployment's own owner is never removed; the page hides Remove for them.
        "installation_owner": member.user_id == installation_owner(),
    }


def _invitation_json(invitation: TeamInvitation) -> dict[str, object]:
    return {
        "invitation_id": invitation.id,
        "email": invitation.email,
        "role": invitation.role,
        "user_id": invitation.target_user_id,
        "state": invitation.state,
        "created_at": invitation.created_at,
        "expires_at": invitation.expires_at,
    }


def team_administrator(
    request: Request,
    team_store: SqlAlchemyTeamStore,
    auth_provider: AuthProvider | None,
    action: str | None,
) -> TeamMember:
    """The calling active owner or admin; reads the store (call from a worker thread).

    :param action: A change or support action that needs a browser sign-in,
        completing "cannot ..."; ``None`` for a read any of their sign-ins may make.
    :raises OmnigentError: ``FORBIDDEN`` for anyone else, or another credential.
    """
    user_id = require_user(request, auth_provider)
    if user_id is None:
        raise OmnigentError("team administration needs accounts", code=ErrorCode.FORBIDDEN)
    if action is not None:
        problem = answer_credential_problem(request, auth_provider, action)
        if problem is not None:
            raise OmnigentError(problem, code=ErrorCode.FORBIDDEN)
    member = team_store.get_member_for_account(user_id)
    if member is None or member.state != "active" or member.role not in ("owner", "admin"):
        raise OmnigentError(
            "only the team's owners and admins manage it", code=ErrorCode.FORBIDDEN
        )
    return member


def create_team_router(
    team_store: SqlAlchemyTeamStore,
    account_store: SqlAlchemyAccountStore,
    *,
    auth_provider: AuthProvider | None,
    base_url: str,
    invite_ttl_seconds: int,
    admission_sync: AdmissionSync | None = None,
    removals: MemberRemovals | None = None,
    support_access: bool = False,
) -> APIRouter:
    """Build the ``/v1/team`` router.

    :param team_store: The team's memberships and invitations.
    :param account_store: Suspension rotates the account's registration.
    :param auth_provider: Resolves the caller.
    :param base_url: The installation's public address, for invitation links.
    :param invite_ttl_seconds: How long an invitation stays usable.
    :param admission_sync: Pushes the change to a front proxy, when one is wired.
    :param removals: Removes members; one is built from the stores when omitted.
    :param support_access: Whether owners and admins manage members' computers and
        connections (:mod:`omnigent.server.routes.member_support`); reported to the page.
    """
    router = APIRouter()
    member_removals = removals or MemberRemovals(
        account_store, team_store, auth_provider=auth_provider, admission_sync=admission_sync
    )

    def _caller(request: Request, action: str | None) -> TeamMember:
        """The calling owner or admin; *action* names a change that needs a browser."""
        return team_administrator(request, team_store, auth_provider, action)

    def _require_bound(request: Request, caller: TeamMember) -> None:
        """Behind a proxy, people are invited only once the caller's sign-in is bound."""
        if external_identity_mode(request) is ExternalIdentityMode.OFF:
            return
        resolved = resolved_external_identity(request)
        if resolved is None or resolved.member is None or resolved.member.id != caller.id:
            raise OmnigentError(
                "open this page through the installation's address to invite people",
                code=ErrorCode.FORBIDDEN,
            )

    def _target(member_id: str) -> TeamMember:
        target = team_store.get_member(member_id)
        if target is None or target.state == "removed":
            raise OmnigentError("no such team member", code=ErrorCode.NOT_FOUND)
        return target

    def _may_change_state(caller: TeamMember, target: TeamMember) -> None:
        if target.id == caller.id:
            raise OmnigentError("you cannot suspend yourself", code=ErrorCode.FORBIDDEN)
        if caller.role != "owner" and target.role != "member":
            raise OmnigentError(
                "only an owner suspends or reinstates owners and admins",
                code=ErrorCode.FORBIDDEN,
            )

    def _changed(user_id: str | None) -> None:
        if user_id is not None and auth_provider is not None:
            revoke = getattr(auth_provider, "revoke_user_sessions", None)
            if callable(revoke):
                revoke(user_id)
        authority_changed()
        if admission_sync is not None:
            admission_sync.request()

    @router.get("/team")
    async def get_team(request: Request) -> dict[str, object]:
        """The team, the caller's membership, members and pending invitations."""
        caller = await asyncio.to_thread(_caller, request, None)
        team = await asyncio.to_thread(team_store.get_team)
        members, _ = await asyncio.to_thread(team_store.list_members, limit=1000)
        invitations = await asyncio.to_thread(team_store.list_invitations)
        removed = await asyncio.to_thread(_removal_list)
        return {
            "team": {"team_id": team.id, "name": team.name} if team is not None else None,
            "me": _member_json(caller),
            "members": [_member_json(member) for member in members if member.is_current],
            "invitations": [_invitation_json(invitation) for invitation in invitations],
            "removals": removed,
            "external_identity": external_identity_mode(request).value,
            "support_access": support_access,
        }

    def _removal_list() -> list[dict[str, object]]:
        return [member_removals.status(removal) for removal in team_store.list_removals()]

    @router.post("/team/invitations")
    async def invite(request: Request, body: InvitationRequest) -> dict[str, object]:
        """Invite an address; the link is shown once."""
        caller = await asyncio.to_thread(_caller, request, "invite people")
        _require_bound(request, caller)
        if caller.role != "owner" and (body.role != "member" or body.user_id is not None):
            raise OmnigentError(
                "only an owner invites admins or links existing accounts",
                code=ErrorCode.FORBIDDEN,
            )
        invitation, token = await asyncio.to_thread(
            team_store.create_invitation,
            body.email,
            body.role,
            created_by_member_id=caller.id,
            ttl_seconds=invite_ttl_seconds,
            target_user_id=body.user_id,
        )
        _changed(None)
        _logger.info("team: %s invited a %s", caller.id, invitation.role)
        return {
            "invitation": _invitation_json(invitation),
            "invite_url": f"{base_url}/register?invite={token}",
        }

    @router.delete("/team/invitations/{invitation_id}")
    async def revoke(request: Request, invitation_id: str) -> dict[str, object]:
        """Revoke a pending invitation."""
        caller = await asyncio.to_thread(_caller, request, "revoke invitations")
        invitation = await asyncio.to_thread(team_store.get_invitation, invitation_id)
        if invitation is None:
            raise OmnigentError("no such invitation", code=ErrorCode.NOT_FOUND)
        if caller.role != "owner" and invitation.role != "member":
            raise OmnigentError(
                "only an owner revokes an admin's invitation", code=ErrorCode.FORBIDDEN
            )
        revoked = await asyncio.to_thread(
            team_store.revoke_invitation, invitation_id, actor_member_id=caller.id
        )
        _changed(None)
        return {"invitation": _invitation_json(revoked)}

    @router.put("/team/members/{member_id}/role")
    async def change_role(
        request: Request, member_id: str, body: RoleRequest
    ) -> dict[str, object]:
        """Change a member's role; owners only, keeping an active owner."""
        caller = await asyncio.to_thread(_caller, request, "change roles")
        if caller.role != "owner":
            raise OmnigentError("only an owner changes roles", code=ErrorCode.FORBIDDEN)
        target = await asyncio.to_thread(_target, member_id)
        changed = await asyncio.to_thread(
            team_store.change_role, target.id, body.role, actor_member_id=caller.id
        )
        _changed(changed.user_id)
        _logger.info("team: %s made %s %s", caller.id, changed.id, changed.role)
        return {"member": _member_json(changed)}

    async def _set_state(
        request: Request, member_id: str, state: MemberState, action: str
    ) -> TeamMember:
        caller = await asyncio.to_thread(_caller, request, action)
        target = await asyncio.to_thread(_target, member_id)
        _may_change_state(caller, target)
        changed = await asyncio.to_thread(
            account_store.change_member_state, target.id, state, actor_member_id=caller.id
        )
        _changed(changed.user_id)
        _logger.info("team: %s set %s %s", caller.id, changed.id, changed.state)
        return changed

    @router.post("/team/members/{member_id}/suspend")
    async def suspend(request: Request, member_id: str) -> dict[str, object]:
        """Suspend a member: their sign-ins, tokens, streams and schedules stop now."""
        changed = await _set_state(request, member_id, "suspended", "suspend members")
        return {"member": _member_json(changed)}

    @router.post("/team/members/{member_id}/reinstate")
    async def reinstate(request: Request, member_id: str) -> dict[str, object]:
        """Reinstate a suspended member; they sign in again on every device."""
        changed = await _set_state(request, member_id, "active", "reinstate members")
        return {"member": _member_json(changed)}

    @router.post("/team/members/{member_id}/remove")
    async def remove(request: Request, member_id: str) -> dict[str, object]:
        """Remove a member for good; repeated, it retries the steps not yet done."""
        caller = await asyncio.to_thread(_caller, request, "remove members")
        removal = await member_removals.remove(request, member_id, actor_member_id=caller.id)
        return {"removal": await asyncio.to_thread(member_removals.status, removal)}

    @router.get("/team/members/{member_id}/removal")
    async def removal_detail(request: Request, member_id: str) -> dict[str, object]:
        """A removal's steps and the threads the member owned."""
        await asyncio.to_thread(_caller, request, None)
        removal = await asyncio.to_thread(team_store.get_removal, member_id)
        if removal is None:
            raise OmnigentError("no such removal", code=ErrorCode.NOT_FOUND)
        status = await asyncio.to_thread(member_removals.status, removal)
        threads = await asyncio.to_thread(member_removals.threads, member_id)
        return {"removal": status, "threads": threads}

    @router.get("/team/invitations/preview")
    async def preview(token: str = "") -> dict[str, object]:
        """What an invitation offers, for the person holding its link."""
        invitation = await asyncio.to_thread(team_store.invitation_for_token, token)
        if invitation is None or invitation.state != "pending":
            raise OmnigentError(
                "this invitation is invalid or has expired", code=ErrorCode.NOT_FOUND
            )
        team = await asyncio.to_thread(team_store.get_team)
        return {
            "team_name": team.name if team is not None else None,
            "email": invitation.email,
            "role": invitation.role,
            "links_account": invitation.target_user_id is not None,
            # The account a link invitation is for, so its page asks for that password.
            "user_id": invitation.target_user_id,
            "expires_at": invitation.expires_at,
        }

    return router
