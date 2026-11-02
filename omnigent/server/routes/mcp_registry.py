"""Catalog and per-user account management for server-side MCP services.

With a service store, administrators also add, change and remove services here
(``/mcp-registry/admin/...``) without a restart. Only a person signed in as an
administrator may: a runner's delegated token is refused even when its owner is
an administrator. No response ever carries a login or a client secret.
"""

from __future__ import annotations

import asyncio
import json
import re
from bisect import bisect_right
from typing import Any, Literal, TypeVar

from fastapi import APIRouter, HTTPException, Query, Request, Response
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from omnigent.server.auth import RESERVED_USER_LOCAL, AuthProvider
from omnigent.server.mcp_destinations import McpDestinationError, check_destination
from omnigent.server.mcp_oauth_discovery import (
    DiscoveredOAuth,
    McpOAuthDiscoveryError,
    discover_oauth,
    register_client,
)
from omnigent.server.mcp_registry import (
    ManagedMcpService,
    McpOAuthHooks,
    McpRegistry,
    McpService,
    McpUpstreamError,
    TokenEndpointAuth,
    login_matches,
    needs_reconnect,
)
from omnigent.server.routes._auth_helpers import require_user
from omnigent.server.routes.connections_base import (
    ConnectionError,
    create_connection_router,
    create_resolved_connection_router,
)
from omnigent.stores.credential_store.secret_cipher import SecretTooLargeError
from omnigent.stores.credential_store.sqlalchemy_store import SealedSecret
from omnigent.stores.mcp_service_store import (
    SERVICE_PRINCIPAL,
    McpClientRegistration,
    McpClientSecretMissing,
    McpServiceConflict,
    McpServiceNotFound,
    McpServiceOAuth,
    McpServiceSettings,
    McpServiceVersionChanged,
    client_provider,
)
from omnigent.stores.permission_store import PermissionStore

_SERVICE_ID = r"^[a-z][a-z0-9_-]{0,39}$"
_TOOL_NAME = re.compile(r"[A-Za-z0-9_.\-/]{1,128}")
_SCOPE = re.compile(r"[!#-\[\]-~]{1,256}")  # printable ASCII without space, '"' or '\'
_CLIENT_ID = re.compile(r"[!-~]{1,256}")


class ConnectMcpToken(BaseModel):
    token: str = Field(min_length=1, max_length=16000, repr=False)


class AdminOAuthSettings(BaseModel):
    """OAuth settings an administrator enters; endpoints left out are discovered."""

    model_config = ConfigDict(extra="forbid")
    client_id: str | None = Field(default=None, max_length=256)
    #: Write-only. Left out on a change that keeps the client: the saved one stays.
    client_secret: str | None = Field(default=None, min_length=1, max_length=4096, repr=False)
    token_endpoint_auth: Literal["none", "client_secret_post", "client_secret_basic"] | None = None
    authorize_url: str | None = Field(default=None, max_length=2048)
    token_url: str | None = Field(default=None, max_length=2048)
    scopes: list[str] = Field(default_factory=list, max_length=50)
    resource: str | None = Field(default=None, max_length=2048)

    @field_validator("client_id")
    @classmethod
    def _client_id(cls, value: str | None) -> str | None:
        if value is not None and not _CLIENT_ID.fullmatch(value):
            raise ValueError("client_id must be printable text without spaces")
        return value

    @field_validator("scopes")
    @classmethod
    def _scopes(cls, value: list[str]) -> list[str]:
        if not all(_SCOPE.fullmatch(scope) for scope in value):
            raise ValueError("scopes must be scope names without spaces or quotes")
        return value


class AdminServiceSettings(BaseModel):
    """A run-time service's settings, as the Add/Edit MCP server form sends them."""

    model_config = ConfigDict(extra="forbid")
    title: str = Field(min_length=1, max_length=120)
    description: str = Field(default="", max_length=1000)
    url: str = Field(min_length=1, max_length=2048)
    auth: Literal["none", "bearer", "oauth"]
    timeout: int = Field(default=60, ge=1, le=300)
    tools: list[str] = Field(default_factory=list, max_length=200)
    oauth: AdminOAuthSettings | None = None

    @field_validator("title", "description")
    @classmethod
    def _plain(cls, value: str) -> str:
        if any(not ch.isprintable() for ch in value):
            raise ValueError("must be printable text")
        return value.strip()

    @field_validator("tools")
    @classmethod
    def _tools(cls, value: list[str]) -> list[str]:
        if not all(_TOOL_NAME.fullmatch(tool) for tool in value) or len(set(value)) != len(value):
            raise ValueError("tools must list distinct tool names")
        return value


class AdminServiceCreate(AdminServiceSettings):
    id: str = Field(pattern=_SERVICE_ID)

    @field_validator("id")
    @classmethod
    def _id(cls, value: str) -> str:
        if "__" in value:
            raise ValueError("ids cannot contain '__', the tool namespace separator")
        return value


#: Largest request body these routes read (a service's settings, or one login).
_MAX_BODY_BYTES = 64 * 1024
_Body = TypeVar("_Body", bound=BaseModel)


async def _read_body(request: Request, model: type[_Body]) -> _Body:
    """Validate a JSON body that can carry a secret.

    FastAPI's own 422 response repeats the submitted input, including a client
    secret or a token in a field that failed validation, or in the object around
    it. These errors keep only each problem's location, type and message.
    """
    body = await request.body()
    if len(body) > _MAX_BODY_BYTES:
        raise HTTPException(413, "The request body is too large")

    def invalid(message: str) -> HTTPException:
        return HTTPException(422, [{"type": "invalid_body", "loc": ["body"], "msg": message}])

    try:
        data = json.loads(body)
    except ValueError:
        raise invalid("The request body must be JSON") from None
    if not isinstance(data, dict):
        raise invalid("The request body must be a JSON object")
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        raise HTTPException(
            422,
            [
                {"type": error["type"], "loc": ["body", *error["loc"]], "msg": error["msg"]}
                for error in exc.errors(
                    include_url=False, include_input=False, include_context=False
                )
            ],
        ) from None


def _admin_view(service: McpService, *, source: str, has_client_secret: bool) -> dict[str, Any]:
    """A service as administrators see it: settings, never a secret."""
    oauth = service.oauth
    view: dict[str, Any] = {
        "id": service.id,
        "title": service.title,
        "description": service.description,
        "url": service.url,
        "auth": service.auth,
        "timeout": service.timeout,
        "tools": list(service.tools),
        "source": source,
        "editable": source == "admin",
        "oauth": (
            {
                "authorize_url": oauth.authorize_url,
                "token_url": oauth.token_url,
                "client_id": oauth.client_id,
                "scopes": list(oauth.scopes),
                "resource": oauth.resource,
                "has_client_secret": has_client_secret,
            }
            if oauth is not None
            else None
        ),
    }
    if isinstance(service, ManagedMcpService):
        view["draft"] = not service.tools
        view["created_by"] = service.created_by
        view["created_at"] = service.created_at
        view["updated_at"] = service.updated_at
        if view["oauth"] is not None:
            view["oauth"]["registration"] = service.client_registration
            view["oauth"]["token_endpoint_auth"] = service.token_endpoint_auth
    return view


def create_mcp_registry_router(
    registry: McpRegistry,
    auth_provider: AuthProvider | None,
    permission_store: PermissionStore | None = None,
) -> APIRouter:
    router = APIRouter()

    def user(request: Request) -> str:
        return require_user(request, auth_provider) or RESERVED_USER_LOCAL

    async def service(request: Request, service_id: str) -> McpService:
        await registry.refresh()
        try:
            return registry.service(service_id, user(request))
        except ConnectionError as exc:
            raise HTTPException(404, str(exc)) from exc

    @router.get("/mcp-registry/services")
    async def catalog(
        request: Request,
        response: Response,
        limit: int = Query(default=50, ge=1, le=100),
        after: str = Query(default="", max_length=64),
    ) -> dict[str, Any]:
        response.headers["Cache-Control"] = "no-store"
        user_id = user(request)
        await registry.refresh()
        entries = registry.catalog()
        data = []
        ids = sorted(entry.id for entry in entries.values() if entry.permits(user_id))
        start = bisect_right(ids, after)
        page = ids[start : start + limit]
        for service_id in page:
            entry = entries[service_id]
            connected = entry.auth == "none"
            rejected = False
            if entry.auth in {"github", "databricks"}:
                store = getattr(request.app.state, f"{entry.auth}_store", None)
                connected = (
                    store is not None and await asyncio.to_thread(store.get, user_id) is not None
                )
            elif entry.auth in {"oauth", "bearer"} and registry.store is not None:
                connection = await asyncio.to_thread(
                    registry.store.get, user_id, f"mcp:{entry.id}"
                )
                connected = login_matches(entry, connection)
                rejected = connected and needs_reconnect(connection)
            data.append(
                {
                    "id": entry.id,
                    "title": entry.title,
                    "description": entry.description,
                    "auth": entry.auth,
                    "connected": connected,
                    "needs_reconnect": rejected,
                    "tools": entry.tools,
                    "connect_provider": f"mcp-{entry.id}" if entry.oauth else entry.auth,
                }
            )
        return {
            "data": data,
            "next_cursor": page[-1] if start + len(page) < len(ids) else None,
        }

    @router.put("/mcp-registry/services/{service_id}/connection")
    async def connect_token(request: Request, service_id: str) -> dict[str, bool]:
        body = await _read_body(request, ConnectMcpToken)
        entry = await service(request, service_id)
        if entry.auth != "bearer" or registry.store is None:
            raise HTTPException(400, "This service does not accept a personal bearer token")
        if any(c.isspace() for c in body.token):
            raise HTTPException(400, "Bearer tokens must not contain whitespace")
        try:
            await asyncio.to_thread(
                registry.save_login,
                entry,
                user(request),
                f"mcp:{entry.id}",
                {"access_token": body.token},
                {},
            )
        except ConnectionError as exc:
            raise HTTPException(400, str(exc)) from None
        return {"connected": True}

    @router.delete("/mcp-registry/services/{service_id}/connection")
    async def disconnect(request: Request, service_id: str) -> dict[str, bool]:
        entry = await service(request, service_id)
        if entry.auth not in {"oauth", "bearer"} or registry.store is None:
            raise HTTPException(
                400, "Manage the shared provider connection in Sandbox Integrations"
            )
        user_id = user(request)
        await asyncio.to_thread(registry.store.delete, user_id, f"mcp:{entry.id}")
        await asyncio.to_thread(registry.store.delete, user_id, f"mcp-pending:{entry.id}")
        return {"disconnected": True}

    @router.post("/mcp-registry/services/{service_id}/test")
    async def test_connection(request: Request, service_id: str) -> dict[str, Any]:
        entry = await service(request, service_id)
        try:
            tools = await registry.execute(entry.id, user(request), request.app.state)
        except McpUpstreamError as exc:
            raise HTTPException(502, str(exc)) from None
        except ConnectionError as exc:
            raise HTTPException(400, str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                502, "MCP connection failed. Check the account and server configuration."
            ) from exc
        return {"tools": [t.name for t in tools]}

    for entry in registry.services.values():
        if entry.oauth:
            router.include_router(
                create_connection_router(
                    McpOAuthHooks(registry, entry), auth_provider=auth_provider
                )
            )
    if registry.manages_services:
        _add_admin_routes(router, registry, auth_provider, permission_store, user)
        if registry.store is not None:

            async def resolve_hooks(request: Request) -> McpOAuthHooks:
                await registry.refresh()
                service_id = request.path_params.get("service_id", "")
                entry = registry.managed_services().get(service_id)
                if entry is None or entry.oauth is None or service_id in registry.services:
                    raise HTTPException(404, "MCP service is unavailable for this user")
                if len(registry.state_secret) < 32:
                    raise HTTPException(
                        503, "Set OMNIGENT_MCP_OAUTH_STATE_SECRET to sign in to MCP services"
                    )
                return McpOAuthHooks(registry, entry)

            router.include_router(
                create_resolved_connection_router(
                    "/connections/mcp-{service_id}", resolve_hooks, auth_provider=auth_provider
                )
            )
    return router


def _add_admin_routes(
    router: APIRouter,
    registry: McpRegistry,
    auth_provider: AuthProvider | None,
    permission_store: PermissionStore | None,
    user: Any,
) -> None:
    store = registry.service_store
    assert store is not None

    async def require_admin(request: Request) -> str:
        """A person signed in as an administrator; never a delegated credential."""
        user_id = user(request)
        if auth_provider is not None:
            delegation = auth_provider.credential_delegation(request)
            if delegation is not None:
                raise HTTPException(
                    403,
                    f"A {delegation} credential cannot manage MCP services; "
                    "sign in as yourself to do this.",
                )
        if permission_store is not None and not await asyncio.to_thread(
            permission_store.is_admin, user_id
        ):
            raise HTTPException(403, "Only administrators can manage MCP services")
        return user_id

    def managed(service_id: str) -> ManagedMcpService:
        if service_id in registry.services:
            raise HTTPException(
                409, "This service is defined in the server's registry file and is read-only"
            )
        entry = registry.managed_services().get(service_id)
        if entry is None:
            raise HTTPException(404, "No MCP service has this ID")
        return entry

    async def has_client_secret(entry: McpService) -> bool:
        if entry.oauth is None:
            return False
        if not isinstance(entry, ManagedMcpService):
            return bool(entry.oauth.client_secret_env)
        if registry.store is None or entry.token_endpoint_auth == "none":
            return False
        saved = await asyncio.to_thread(
            registry.store.get, SERVICE_PRINCIPAL, client_provider(entry.id)
        )
        return login_matches(entry, saved)

    async def checked_url(url: str, what: str) -> str:
        try:
            await check_destination(registry.destination_policy, url, what=what)
        except McpDestinationError as exc:
            raise HTTPException(400, str(exc)) from None
        return url

    async def endpoints(
        body: AdminServiceSettings,
    ) -> tuple[str, str, str, tuple[str, ...], DiscoveredOAuth | None]:
        """Authorization and token URLs, resource and scopes: as entered, else discovered."""
        given = body.oauth or AdminOAuthSettings()
        if given.authorize_url and given.token_url:
            authorize_url, token_url, discovered = given.authorize_url, given.token_url, None
            resource, scopes = given.resource or body.url, tuple(given.scopes)
        elif given.authorize_url or given.token_url:
            raise HTTPException(400, "Enter both the authorization and token URLs, or neither")
        else:
            discovered = await discover_oauth(body.url, registry.destination_policy)
            authorize_url, token_url = discovered.authorize_url, discovered.token_url
            resource = given.resource or discovered.resource
            scopes = tuple(given.scopes) or discovered.scopes
        problem = registry.destination_policy.url_problem(authorize_url)
        if problem is not None:
            raise HTTPException(
                400, f"The authorization URL breaks a destination rule: {problem}."
            )
        await checked_url(token_url, "The token URL")
        return authorize_url, token_url, resource, scopes, discovered

    async def resolve_oauth(
        service_id: str, body: AdminServiceSettings, current: ManagedMcpService | None
    ) -> tuple[McpServiceOAuth, str | None, bool]:
        """The OAuth settings for *body*, a new client secret to seal for them, and
        whether to keep the saved one instead.

        A secret is sent only to the token URL, so a saved one is kept only while the
        token URL and client ID stay the same; any other change needs a new one. The
        same client keeps its authentication method unless the request names one.
        """
        if registry.store is None:
            raise HTTPException(
                400, "OAuth services need the server's credential store (a credential cipher)"
            )
        if len(registry.state_secret) < 32:
            raise HTTPException(
                400, "Set OMNIGENT_MCP_OAUTH_STATE_SECRET (32+ characters) for OAuth services"
            )
        redirect_base = registry.public_url()
        if redirect_base is None:
            raise HTTPException(
                400, "Set the server's public URL (OMNIGENT_ACCOUNTS_BASE_URL) for OAuth services"
            )
        given = body.oauth or AdminOAuthSettings()
        authorize_url, token_url, resource, scopes, discovered = await endpoints(body)
        previous = current.oauth if current is not None else None
        same_client_endpoint = previous is not None and previous.token_url == token_url

        def oauth(
            client_id: str, registration: McpClientRegistration, method: TokenEndpointAuth
        ) -> McpServiceOAuth:
            return McpServiceOAuth(
                authorize_url=authorize_url,
                token_url=token_url,
                client_id=client_id,
                scopes=scopes,
                resource=resource,
                registration=registration,
                token_endpoint_auth=method,
            )

        if given.client_id:
            same_client = previous is not None and previous.client_id == given.client_id
            method: TokenEndpointAuth
            if given.token_endpoint_auth is not None:
                method = given.token_endpoint_auth
            elif same_client and current is not None:
                method = current.token_endpoint_auth
            else:
                method = "client_secret_post" if given.client_secret else "none"
            if method == "none" and given.client_secret and given.token_endpoint_auth is None:
                method = "client_secret_post"
            if method == "none":
                return oauth(given.client_id, "manual", method), None, False
            if given.client_secret:
                return oauth(given.client_id, "manual", method), given.client_secret, False
            if same_client and same_client_endpoint and current is not None:
                if current.token_endpoint_auth != "none":
                    return oauth(given.client_id, "manual", method), None, True
            raise HTTPException(
                400,
                "Enter the client secret for this client ID"
                + (
                    " again: the token URL changed"
                    if same_client and not same_client_endpoint
                    else ""
                ),
            )
        if (
            current is not None
            and previous is not None
            and current.client_registration == "dynamic"
            and same_client_endpoint
            and previous.authorize_url == authorize_url
        ):
            # Same authorization server: keep the client registered earlier.
            keep = current.token_endpoint_auth != "none"
            return oauth(previous.client_id, "dynamic", current.token_endpoint_auth), None, keep
        if discovered is None:
            discovered = await discover_oauth(body.url, registry.destination_policy)
        registered = await register_client(
            DiscoveredOAuth(
                authorize_url=authorize_url,
                token_url=token_url,
                resource=resource,
                scopes=scopes,
                registration_endpoint=discovered.registration_endpoint,
                token_endpoint_auth_methods=discovered.token_endpoint_auth_methods,
            ),
            f"{redirect_base}/v1/connections/mcp-{service_id}/callback",
            registry.destination_policy,
        )
        return (
            oauth(registered.client_id, "dynamic", registered.token_endpoint_auth),
            registered.client_secret,
            False,
        )

    async def settings_for(
        service_id: str, body: AdminServiceSettings, current: ManagedMcpService | None
    ) -> tuple[McpServiceSettings, SealedSecret | None, bool]:
        """The settings to store, the client secret sealed for them, and whether to
        keep the saved client secret."""
        if body.auth != "none" and registry.store is None:
            raise HTTPException(
                400, "Logins need the server's credential store (a credential cipher)"
            )
        if body.auth != "oauth" and body.oauth is not None:
            raise HTTPException(400, "OAuth settings apply only to the OAuth login type")
        unchanged = current is not None and (current.url, current.auth) == (body.url, body.auth)
        if not unchanged:
            await checked_url(body.url, "The server URL")
        oauth, client_secret, keep = None, None, False
        if body.auth == "oauth":
            try:
                oauth, client_secret, keep = await resolve_oauth(service_id, body, current)
            except McpOAuthDiscoveryError as exc:
                raise HTTPException(400, str(exc)) from None
        sealed = None
        if client_secret is not None:
            assert registry.store is not None
            try:
                sealed = await asyncio.to_thread(
                    registry.store.seal,
                    SERVICE_PRINCIPAL,
                    client_provider(service_id),
                    {"client_secret": client_secret},
                )
            except SecretTooLargeError as exc:
                raise HTTPException(400, str(exc)) from None
        return (
            McpServiceSettings(
                title=body.title,
                description=body.description,
                url=body.url,
                auth=body.auth,
                timeout=body.timeout,
                tools=tuple(body.tools),
                oauth=oauth,
            ),
            sealed,
            keep,
        )

    async def view(service_id: str) -> dict[str, Any]:
        await registry.refresh(force=True)
        entry = registry.managed_services().get(service_id)
        if entry is None:
            raise HTTPException(404, "No MCP service has this ID")
        return _admin_view(entry, source="admin", has_client_secret=await has_client_secret(entry))

    @router.get("/mcp-registry/admin/services")
    async def admin_list(
        request: Request,
        response: Response,
        limit: int = Query(default=50, ge=1, le=100),
        after: str = Query(default="", max_length=64),
    ) -> dict[str, Any]:
        await require_admin(request)
        response.headers["Cache-Control"] = "no-store"
        await registry.refresh()
        entries = registry.catalog()
        ids = sorted(entries)
        start = bisect_right(ids, after)
        page = ids[start : start + limit]
        data = []
        for service_id in page:
            entry = entries[service_id]
            source = "admin" if isinstance(entry, ManagedMcpService) else "config"
            data.append(
                _admin_view(entry, source=source, has_client_secret=await has_client_secret(entry))
            )
        return {"data": data, "next_cursor": page[-1] if start + len(page) < len(ids) else None}

    @router.get("/mcp-registry/admin/presets")
    async def admin_presets(request: Request) -> dict[str, Any]:
        await require_admin(request)
        return {"data": [preset.model_dump() for preset in registry.config.presets]}

    @router.post("/mcp-registry/admin/services", status_code=201)
    async def admin_create(request: Request) -> dict[str, Any]:
        admin = await require_admin(request)
        body = await _read_body(request, AdminServiceCreate)
        await registry.refresh(force=True)
        if registry.lookup(body.id) is not None:
            raise HTTPException(409, "An MCP service with this ID already exists")
        settings, sealed, _keep = await settings_for(body.id, body, None)
        try:
            # The settings and their client secret are saved in one transaction.
            await asyncio.to_thread(
                store.create, body.id, settings, created_by=admin, client_secret=sealed
            )
        except McpServiceConflict:
            raise HTTPException(409, "An MCP service with this ID already exists") from None
        except McpClientSecretMissing:
            raise HTTPException(400, "Enter the client secret for this client ID") from None
        return await view(body.id)

    @router.put("/mcp-registry/admin/services/{service_id}")
    async def admin_update(request: Request, service_id: str) -> dict[str, Any]:
        await require_admin(request)
        body = await _read_body(request, AdminServiceSettings)
        await registry.refresh(force=True)
        current = managed(service_id)
        settings, sealed, keep = await settings_for(service_id, body, current)
        try:
            # Applies only to the version read above; the client secret is saved, or
            # the saved one kept, in the same transaction.
            result = await asyncio.to_thread(
                store.update,
                service_id,
                settings,
                expected_version=current.credential_version,
                client_secret=sealed,
                keep_client_secret=keep,
            )
        except McpServiceNotFound:
            raise HTTPException(404, "No MCP service has this ID") from None
        except McpServiceVersionChanged:
            raise HTTPException(
                409, "The service changed meanwhile; reload and try again"
            ) from None
        except McpClientSecretMissing:
            raise HTTPException(400, "Enter the client secret for this client ID") from None
        return {**await view(service_id), "credentials_reset": result.credentials_reset}

    @router.delete("/mcp-registry/admin/services/{service_id}")
    async def admin_delete(request: Request, service_id: str) -> dict[str, bool]:
        await require_admin(request)
        await registry.refresh(force=True)
        managed(service_id)
        removed = await asyncio.to_thread(store.delete, service_id)
        await registry.refresh(force=True)
        if not removed:
            raise HTTPException(404, "No MCP service has this ID")
        return {"deleted": True}

    @router.post("/mcp-registry/admin/services/{service_id}/discover")
    async def admin_discover(request: Request, service_id: str) -> dict[str, Any]:
        """Connect with the administrator's own login and list every tool the server offers."""
        admin = await require_admin(request)
        await registry.refresh(force=True)
        entry = managed(service_id)
        try:
            tools = await registry.execute(entry.id, admin, request.app.state, all_tools=True)
        except McpUpstreamError as exc:
            raise HTTPException(502, str(exc)) from None
        except ConnectionError as exc:
            raise HTTPException(400, str(exc)) from None
        except Exception:  # noqa: BLE001 — upstream errors may carry credentials
            raise HTTPException(
                502, "MCP connection failed. Check the login and the server URL."
            ) from None
        return {
            "tools": [
                {
                    "name": tool.name,
                    "title": getattr(tool, "title", None),
                    "description": (tool.description or "")[:1000],
                }
                for tool in tools[:500]
            ],
            "allowed": list(entry.tools),
        }
