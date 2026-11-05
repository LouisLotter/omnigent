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
and each route logs the acting member and the member by member id. A disconnect
is one write that checks the actor and the member again, so it acts only for
the membership and registration the request checked.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Request

from omnigent.db.utils import now_epoch
from omnigent.errors import ErrorCode, OmnigentError
from omnigent.server.connections_registry import connection_providers
from omnigent.server.routes.hosts import host_payload
from omnigent.server.routes.mcp_registry import (
    connection_key,
    connection_state,
    connection_store_configured,
    holds_own_login,
)
from omnigent.server.routes.team import team_administrator
from omnigent.server.support_access import (
    SUPPORTED_STATES,
    SupportAccess,
    SupportedMember,
    log_support_action,
)

if TYPE_CHECKING:
    from omnigent.entities.team import TeamMember
    from omnigent.server.auth import AuthProvider
    from omnigent.server.host_registry import HostRegistry
    from omnigent.server.mcp_registry import McpRegistry
    from omnigent.server.member_removal import MemberRemovals
    from omnigent.stores.host_store import Host, HostStore

# Completes "cannot ...", for a CLI, device, runner or host credential.
_ACTION = "manage team members' computers and connections"
# The member ID now names another registration, e.g. a new account took the username.
_CHANGED = "this member's account changed since the page loaded; reload it"


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

    def _actor(request: Request) -> TeamMember:
        return team_administrator(request, team_store, auth_provider, _ACTION)

    def _actor_and_member(request: Request, member_id: str) -> tuple[TeamMember, SupportedMember]:
        """The acting owner or admin, and the current member with their registration."""
        actor = _actor(request)
        member = support.member(member_id=member_id)
        if member is None:
            raise OmnigentError("no such team member", code=ErrorCode.NOT_FOUND)
        return actor, member

    def _member_hosts(member: SupportedMember) -> list[Host]:
        return [
            host
            for host in host_store.list_hosts(member.member.user_id)
            if host.account_generation in (None, member.generation)
        ]

    @router.get("/team/members/{member_id}/hosts")
    async def member_hosts(request: Request, member_id: str) -> dict[str, list[dict[str, Any]]]:
        """The member's hosts, with the fields ``GET /v1/hosts`` gives their owner."""
        actor, member = await asyncio.to_thread(_actor_and_member, request, member_id)
        hosts = await asyncio.to_thread(_member_hosts, member)
        log_support_action(actor, "list_hosts", member.member)
        now = now_epoch()
        return {"hosts": [host_payload(host, host_registry, now=now) for host in hosts]}

    @router.get("/team/members/{member_id}/connections")
    async def member_connections(request: Request, member_id: str) -> dict[str, Any]:
        """Whether the member is connected to each MCP service and connection provider."""
        actor, supported = await asyncio.to_thread(_actor_and_member, request, member_id)
        member = supported.member
        # Read bound to the member ID and its registration: never a new account's rows.
        rows = await asyncio.to_thread(
            support.account_store.member_connections,
            member.id,
            supported.generation,
            SUPPORTED_STATES,
        )
        if rows is None:
            raise OmnigentError(_CHANGED, code=ErrorCode.CONFLICT)
        by_key = {row.provider: row for row in rows if row.account_id == ""}
        services: list[dict[str, Any]] = []
        if mcp_registry is not None:
            await mcp_registry.refresh()
            catalog = mcp_registry.catalog()
            for service_id in sorted(catalog):
                entry = catalog[service_id]
                if not entry.permits(member.user_id):
                    continue
                key = connection_key(entry)
                connected, rejected = connection_state(
                    entry,
                    by_key.get(key) if key is not None else None,
                    store_configured=connection_store_configured(
                        mcp_registry, entry, request.app.state
                    ),
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
            connection = by_key.get(store.provider_key)
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
        removed = await asyncio.to_thread(
            support.account_store.remove_member_connections,
            member.action(actor),
            [store.provider_key],
        )
        log_support_action(actor, "disconnect", member.member, subject=provider)
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
        # The login and the sign-in they had started go in the same write.
        await asyncio.to_thread(
            support.account_store.remove_member_connections,
            member.action(actor),
            [f"mcp:{entry.id}", f"mcp-pending:{entry.id}"],
        )
        log_support_action(actor, "disconnect", member.member, subject=f"mcp:{entry.id}")
        return {"disconnected": True}

    @router.get("/team/members/{member_id}/threads")
    async def member_threads(request: Request, member_id: str) -> dict[str, Any]:
        """The threads the member owns, or owned when they were removed, with titles."""

        def actor_member_and_threads() -> tuple[TeamMember, TeamMember, list[dict[str, object]]]:
            actor = _actor(request)
            current = support.member(member_id=member_id)
            if current is not None:
                # Read bound to the member ID and its registration.
                threads = removals.member_threads(current.member.id, current.generation)
                if threads is not None:
                    return actor, current.member, threads
            # A removed member's threads are their removal's record, whoever has the name now.
            removed = team_store.get_member(member_id)
            if removed is not None and removed.state == "removed":
                return actor, removed, removals.threads(member_id)
            if current is not None:
                raise OmnigentError(_CHANGED, code=ErrorCode.CONFLICT)
            raise OmnigentError("no such team member", code=ErrorCode.NOT_FOUND)

        actor, member, threads = await asyncio.to_thread(actor_member_and_threads)
        log_support_action(actor, "list_threads", member)
        return {"threads": threads}

    return router
