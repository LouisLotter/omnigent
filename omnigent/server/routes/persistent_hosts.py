"""Create, start and delete persistent managed hosts ("computers") without a session.

With support access (:mod:`omnigent.server.support_access`), a team's owners and
admins also start and delete members' computers and create one for a member:
the computer is the member's, and its provider is called for the member.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from omnigent.db.account_authority import target_account_scope
from omnigent.entities.team import TeamMember
from omnigent.server import persistent_hosts
from omnigent.server.auth import RESERVED_USER_LOCAL, AuthProvider
from omnigent.server.host_registry import HostRegistry
from omnigent.server.routes._auth_helpers import require_user
from omnigent.server.routes._host_launch import managed_host
from omnigent.server.routes.hosts import host_payload
from omnigent.server.support_access import SupportAccess, log_support_action
from omnigent.stores.host_store import Host, HostStore


class PersistentHostCreate(BaseModel):
    """Body of ``POST /v1/hosts``.

    :param name: Display name shown in host pickers, e.g. ``"Work computer"``.
    :param provider: A configured persistent provider, or ``None`` for the first.
    :param owner_user_id: Whose computer it is, when a team owner or admin
        creates one for an active member, e.g. ``"alice"``; ``None`` for the caller.
    """

    name: str = Field(min_length=1, max_length=64)
    provider: str | None = None
    owner_user_id: str | None = Field(default=None, min_length=1, max_length=128)

    @field_validator("name")
    @classmethod
    def _clean_name(cls, value: str) -> str:
        name = value.strip()
        if not name or any(not char.isprintable() for char in name):
            raise ValueError("name must be printable text")
        return name


def create_persistent_hosts_router(
    host_store: HostStore,
    host_registry: HostRegistry,
    *,
    auth_provider: AuthProvider | None = None,
    support: SupportAccess | None = None,
) -> APIRouter:
    """Build the router that creates, starts and deletes persistent managed hosts.

    :param support: Support access, or ``None`` to keep every computer its owner's.
    """
    router = APIRouter()

    async def _managed_persistent_host(request: Request, host_id: str, action: str) -> Host:
        host = await managed_host(
            request,
            host_id,
            action=action,
            host_store=host_store,
            auth_provider=auth_provider,
            support=support,
        )
        if not persistent_hosts.is_persistent_host(host):
            raise HTTPException(status_code=400, detail="not a persistent host")
        return host

    def _member_to_create_for(
        request: Request, user_id: str | None, owner_user_id: str
    ) -> tuple[TeamMember, TeamMember, str | None]:
        """The supporting caller, the active member the computer is for, and their registration."""
        actor = None
        if support is not None and auth_provider is not None:
            actor = support.supporter(auth_provider.principal(request), user_id)
        if support is None or actor is None:
            raise HTTPException(
                status_code=403,
                detail="only a team owner or admin, in the browser, adds a computer for someone",
            )
        member = support.team_store.get_member_for_account(owner_user_id)
        if member is None or member.state != "active":
            raise HTTPException(status_code=404, detail="no such active team member")
        try:
            generation = support.registration(member)
        except LookupError:
            raise HTTPException(status_code=404, detail="no such active team member") from None
        return actor, member, generation

    @router.post("/hosts", status_code=201)
    async def create_persistent_host(
        request: Request, body: PersistentHostCreate
    ) -> dict[str, Any]:
        """Create a persistent host; its first start runs in the background.

        The host is the caller's, or with ``owner_user_id`` and support access,
        the named active member's: a team owner or admin adds it for them.

        :returns: The new host, ``waking`` while it starts.
        :raises OmnigentError: 400 when no such persistent provider is
            configured; 409 when the owner already has a host with the name.
        :raises HTTPException: 403 when the caller may not add a computer for
            someone else; 404 when ``owner_user_id`` names no active member.
        """
        user_id = require_user(request, auth_provider)
        owner = user_id if user_id is not None else RESERVED_USER_LOCAL
        if body.owner_user_id is None or body.owner_user_id == owner:
            host = await persistent_hosts.create_host(
                owner=owner, name=body.name, provider=body.provider
            )
            return host_payload(host, host_registry)
        actor, member, generation = await asyncio.to_thread(
            _member_to_create_for, request, user_id, body.owner_user_id
        )
        # Pinned so the row is written for the registration that was checked.
        with (
            target_account_scope(member.user_id, generation)
            if generation is not None
            else contextlib.nullcontext()
        ):
            host = await persistent_hosts.create_host(
                owner=member.user_id, name=body.name, provider=body.provider
            )
        log_support_action(actor, "create_host", member, subject=host.host_id)
        return host_payload(host, host_registry)

    @router.post("/hosts/{host_id}/start", status_code=202)
    async def start_persistent_host(request: Request, host_id: str) -> dict[str, Any]:
        """Start a persistent host again, e.g. after its first start failed.

        :returns: The host, ``waking`` while it starts.
        """
        host = await _managed_persistent_host(request, host_id, "start_host")
        persistent_hosts.start_host_now(host.host_id)
        return host_payload(host, host_registry)

    @router.delete("/hosts/{host_id}")
    async def delete_persistent_host(request: Request, host_id: str) -> dict[str, Any]:
        """Delete a persistent host and ask its provider to remove the sandbox.

        The host is removed from Omnigent first; a provider removal that fails
        is retried by the managed-sandbox reaper.
        """
        host = await _managed_persistent_host(request, host_id, "delete_host")
        await persistent_hosts.delete_host(host)
        return {"host_id": host.host_id, "deleted": True}

    return router
