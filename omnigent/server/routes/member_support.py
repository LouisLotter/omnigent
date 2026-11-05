"""A team member's computers, connections and threads, for the team's owners and admins.

Mounted beside ``/v1/team`` by ``create_app(admin_support_access=True)``:

- ``GET /v1/team/members/{member_id}/hosts``: the member's hosts, as their
  owner's ``GET /v1/hosts`` lists them.
- ``GET /v1/team/members/{member_id}/connections``: for each MCP service the
  member may use and each connection provider (GitHub, ...), whether they are
  connected. Never a login or a token.
- ``DELETE /v1/team/members/{member_id}/connections/{provider}``: remove the
  member's connection to a provider, e.g. ``github``.
- ``DELETE /v1/team/members/{member_id}/mcp-services/{service_id}/connection``:
  remove the member's login to an MCP service.
- ``GET /v1/team/members/{member_id}/threads``: the threads the member owns, or
  owned when they were removed.

There is no connect: logins stay the person's own (:mod:`omnigent.server.support_access`).
Every route needs an active owner or admin signed in through a browser. The
member comes from the path alone, never from anything a runner or host sends,
and each route logs the acting member and the member by member id.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Request

from omnigent.db.utils import now_epoch
from omnigent.errors import ErrorCode, OmnigentError
from omnigent.server.connections_registry import connection_providers
from omnigent.server.routes.connections_base import remove_connection
from omnigent.server.routes.hosts import host_payload
from omnigent.server.routes.mcp_registry import (
    forget_login,
    holds_own_login,
    service_connection,
)
from omnigent.server.routes.team import team_administrator
from omnigent.server.support_access import SUPPORTED_STATES, SupportAccess, log_support_action

if TYPE_CHECKING:
    from omnigent.entities.team import TeamMember
    from omnigent.server.auth import AuthProvider
    from omnigent.server.host_registry import HostRegistry
    from omnigent.server.mcp_registry import McpRegistry
    from omnigent.server.member_removal import MemberRemovals
    from omnigent.stores.host_store import Host, HostStore

# Completes "cannot ...", for a CLI, device, runner or host credential.
_ACTION = "manage team members' computers and connections"


def create_member_support_router(
    support: SupportAccess,
    host_store: HostStore,
    host_registry: HostRegistry,
    *,
    auth_provider: AuthProvider | None,
    removals: MemberRemovals,
    mcp_registry: McpRegistry | None = None,
) -> APIRouter:
    """Build the support routes under ``/v1/team/members/{member_id}``.

    :param support: Support access: the team and the accounts.
    :param host_store: The members' hosts.
    :param host_registry: Live host tunnels on this replica, for host status.
    :param auth_provider: Resolves the caller.
    :param removals: Lists a member's threads, as a removal does.
    :param mcp_registry: The MCP services, when the server has any.
    """
    router = APIRouter()
    team_store = support.team_store

    def _actor_and_member(
        request: Request, member_id: str, *, removed: bool = False
    ) -> tuple[TeamMember, TeamMember]:
        actor = team_administrator(request, team_store, auth_provider, _ACTION)
        member = team_store.get_member(member_id)
        states = (SUPPORTED_STATES | {"removed"}) if removed else SUPPORTED_STATES
        if member is None or member.state not in states:
            raise OmnigentError("no such team member", code=ErrorCode.NOT_FOUND)
        return actor, member

    def _member_hosts(member: TeamMember) -> list[Host]:
        try:
            generation = support.registration(member)
        except LookupError:
            raise OmnigentError("no such team member", code=ErrorCode.NOT_FOUND) from None
        return [
            host
            for host in host_store.list_hosts(member.user_id)
            if host.account_generation in (None, generation)
        ]

    @router.get("/team/members/{member_id}/hosts")
    async def member_hosts(request: Request, member_id: str) -> dict[str, list[dict[str, Any]]]:
        """The member's hosts, with the fields ``GET /v1/hosts`` gives their owner."""
        actor, member = await asyncio.to_thread(_actor_and_member, request, member_id)
        hosts = await asyncio.to_thread(_member_hosts, member)
        log_support_action(actor, "list_hosts", member)
        now = now_epoch()
        return {"hosts": [host_payload(host, host_registry, now=now) for host in hosts]}

    @router.get("/team/members/{member_id}/connections")
    async def member_connections(request: Request, member_id: str) -> dict[str, Any]:
        """Whether the member is connected to each MCP service and connection provider."""
        actor, member = await asyncio.to_thread(_actor_and_member, request, member_id)
        services: list[dict[str, Any]] = []
        if mcp_registry is not None:
            await mcp_registry.refresh()
            catalog = mcp_registry.catalog()
            for service_id in sorted(catalog):
                entry = catalog[service_id]
                if not entry.permits(member.user_id):
                    continue
                connected, rejected = await service_connection(
                    mcp_registry, entry, member.user_id, request.app.state
                )
                services.append(
                    {
                        "id": entry.id,
                        "title": entry.title,
                        "auth": entry.auth,
                        "connected": connected,
                        "needs_reconnect": rejected,
                        "disconnectable": connected
                        and holds_own_login(entry)
                        and mcp_registry.store is not None,
                    }
                )
        providers: list[dict[str, Any]] = []
        for provider in connection_providers():
            store = getattr(request.app.state, f"{provider.name}_store", None)
            if store is None:
                continue
            connection = await asyncio.to_thread(store.get, member.user_id)
            providers.append(
                {
                    "provider": provider.name,
                    "connected": connection is not None,
                    "connected_at": getattr(connection, "created_at", None),
                }
            )
        log_support_action(actor, "list_connections", member)
        return {"services": services, "providers": providers}

    @router.delete("/team/members/{member_id}/connections/{provider}")
    async def disconnect_member_provider(
        request: Request, member_id: str, provider: str
    ) -> dict[str, bool]:
        """Remove the member's connection to a provider, e.g. ``github``."""
        actor, member = await asyncio.to_thread(_actor_and_member, request, member_id)
        names = {known.name for known in connection_providers()}
        store = (
            getattr(request.app.state, f"{provider}_store", None) if provider in names else None
        )
        if store is None:
            raise OmnigentError("no such connection", code=ErrorCode.NOT_FOUND)
        removed = await remove_connection(store, member.user_id)
        log_support_action(actor, "disconnect", member, subject=provider)
        return {"disconnected": removed}

    @router.delete("/team/members/{member_id}/mcp-services/{service_id}/connection")
    async def disconnect_member_service(
        request: Request, member_id: str, service_id: str
    ) -> dict[str, bool]:
        """Remove the member's login to an MCP service, and any sign-in they had started."""
        actor, member = await asyncio.to_thread(_actor_and_member, request, member_id)
        entry = None
        if mcp_registry is not None:
            await mcp_registry.refresh()
            entry = mcp_registry.catalog().get(service_id)
        if mcp_registry is None or entry is None:
            raise OmnigentError("no such MCP service", code=ErrorCode.NOT_FOUND)
        if not holds_own_login(entry) or mcp_registry.store is None:
            raise OmnigentError(
                "this service uses a connection provider; remove that connection instead",
                code=ErrorCode.INVALID_INPUT,
            )
        await asyncio.to_thread(forget_login, mcp_registry, entry, member.user_id)
        log_support_action(actor, "disconnect", member, subject=f"mcp:{entry.id}")
        return {"disconnected": True}

    @router.get("/team/members/{member_id}/threads")
    async def member_threads(request: Request, member_id: str) -> dict[str, Any]:
        """The threads the member owns, or owned when they were removed, with titles."""
        actor, member = await asyncio.to_thread(
            _actor_and_member, request, member_id, removed=True
        )
        threads = await asyncio.to_thread(removals.member_threads, member)
        log_support_action(actor, "list_threads", member)
        return {"threads": threads}

    return router
