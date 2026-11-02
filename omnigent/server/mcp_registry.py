"""Administrator-configured MCP services using the shared per-user credential store."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import math
import os
import secrets
import threading
import time
from builtins import ExceptionGroup
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote, urlencode, urlsplit
from weakref import WeakValueDictionary

import httpx
import yaml
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from pydantic import BaseModel, ConfigDict, Field, model_validator

from omnigent.entities import ProviderConnection
from omnigent.server.auth import RESERVED_USER_LOCAL
from omnigent.server.mcp_destinations import (
    DefaultMcpDestinationPolicy,
    McpDestinationPolicy,
    destination_refused,
    guarded_client,
    redirect_refused,
)
from omnigent.server.routes.connections_base import ConnectionError, ConnectStart
from omnigent.stores.credential_store.secret_cipher import SecretTooLargeError
from omnigent.stores.credential_store.sqlalchemy_store import CredentialStore
from omnigent.stores.mcp_service_store import (
    SERVICE_PRINCIPAL,
    SERVICE_VERSION_KEY,
    McpServiceRecord,
    McpServiceStore,
    McpServiceVersionChanged,
    client_provider,
    stamped,
    version_guard,
)

_logger = logging.getLogger(__name__)

#: Connection metadata key set when the provider rejects a saved login; the value
#: names the rejection. Saving a new grant replaces the metadata and clears it.
NEEDS_RECONNECT = "needs_reconnect"
#: ``0`` turns off MCP services administrators add in Settings (on by default
#: wherever the server can seal logins or has a registry file).
ADMIN_SERVICES_ENV = "OMNIGENT_MCP_ADMIN_SERVICES"


def admin_services_enabled() -> bool:
    """Whether administrators may add MCP services at run time."""
    return os.environ.get(ADMIN_SERVICES_ENV, "").strip() != "0"


def needs_reconnect(connection: ProviderConnection | None) -> bool:
    """Whether the provider rejected this saved login since it was stored."""
    return connection is not None and bool(connection.metadata.get(NEEDS_RECONNECT))


class McpGrantRejected(ConnectionError):
    """The token endpoint answered ``invalid_grant``: the grant is revoked or expired."""


class McpUpstreamError(ConnectionError):
    """Credential-safe diagnosis of an upstream request failure."""

    def __init__(self, message: str, *, kind: str, status_code: int | None = None) -> None:
        super().__init__(message + " The operation was not retried.")
        self.kind = kind
        self.status_code = status_code


def _upstream_error(exc: Exception) -> McpUpstreamError:
    # MCP's task groups wrap HTTP failures, sometimes alongside closed-stream errors.
    errors = [exc]
    for error in errors:
        if isinstance(error, ExceptionGroup):
            errors.extend(error.exceptions)
    if any(redirect_refused(error) for error in errors):
        return McpUpstreamError(
            "MCP service answered with a redirect, which Omnigent does not follow. "
            "Use the exact URL the service expects (a trailing slash can matter).",
            kind="redirect",
        )
    if any(destination_refused(error) for error in errors):
        return McpUpstreamError(
            "MCP service address is refused by the server's destination rules.",
            kind="destination",
        )
    for error in errors:
        if isinstance(error, httpx.HTTPStatusError):
            status = error.response.status_code
            detail = {
                401: "MCP service rejected the credential (HTTP 401). Reconnect the account.",
                403: (
                    "MCP service denied access (HTTP 403). Check the account's scopes, app "
                    "permissions, and organization access; reconnect if the grant changes."
                ),
                429: "MCP service rate limit reached (HTTP 429). Wait before trying again.",
            }.get(
                status,
                f"MCP service returned HTTP {status}. "
                "Check service configuration and availability.",
            )
            return McpUpstreamError(detail, kind="http", status_code=status)
    if any(isinstance(error, (TimeoutError, httpx.TimeoutException)) for error in errors):
        return McpUpstreamError("MCP request timed out; completion is unknown.", kind="timeout")
    if any(isinstance(error, httpx.RequestError) for error in errors):
        return McpUpstreamError(
            "MCP transport failed; completion is unknown. Check service connectivity.",
            kind="transport",
        )
    return McpUpstreamError(
        "MCP request failed. Ask the server operator to check diagnostics.", kind="unexpected"
    )


class OAuthConfig(BaseModel):
    """Explicit OAuth configuration for an approved MCP resource."""

    model_config = ConfigDict(extra="forbid")
    authorize_url: str
    token_url: str
    client_id: str
    client_secret_env: str | None = None
    scopes: list[str] = Field(default_factory=list)
    resource: str | None = None


class McpService(BaseModel):
    """A catalog entry. Only the server operator can choose upstream destinations."""

    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,39}$")
    title: str
    description: str = ""
    url: str
    auth: Literal["none", "bearer", "oauth", "github", "databricks"] = "bearer"
    oauth: OAuthConfig | None = None
    allowed_users: list[str] | None = None
    tools: list[str] = Field(min_length=1)
    allow_http: bool = False
    timeout: int = Field(default=60, ge=1, le=300)

    @model_validator(mode="after")
    def validate_service(self) -> McpService:
        if "__" in self.id:
            raise ValueError("MCP service ids cannot contain the tool namespace separator '__'")
        if (self.auth == "oauth") != (self.oauth is not None):
            raise ValueError("oauth configuration is required only for auth=oauth")
        urls = [self.url]
        if self.oauth:
            urls += [self.oauth.authorize_url, self.oauth.token_url]
        for url in urls:
            parsed = urlsplit(url)
            if (
                parsed.scheme not in ({"http", "https"} if self.allow_http else {"https"})
                or not parsed.hostname
                or parsed.username
                or parsed.password
                or parsed.fragment
            ):
                raise ValueError(
                    "MCP/OAuth URLs require HTTPS (allow_http is for local development)"
                )
        return self

    def permits(self, user_id: str) -> bool:
        return self.allowed_users is None or user_id in self.allowed_users


TokenEndpointAuth = Literal["none", "client_secret_post", "client_secret_basic"]


class ManagedMcpService(McpService):
    """A catalog entry an administrator added at run time (Settings > MCP).

    Logins saved for it carry its :attr:`credential_version`; one saved for other
    settings is never sent. With no tools allowed yet it is a draft, which only its
    creator can connect to while setting it up.
    """

    tools: list[str] = Field(default_factory=list)
    credential_version: str
    token_endpoint_auth: TokenEndpointAuth = "none"
    client_registration: Literal["manual", "dynamic"] | None = None
    created_by: str | None = None
    created_at: int = 0
    updated_at: int = 0

    @classmethod
    def from_record(cls, record: McpServiceRecord) -> ManagedMcpService:
        """The catalog entry for a stored service. Its URLs passed the destination
        policy when saved, and are checked again whenever they are used."""
        oauth = record.oauth
        urls = [record.url, *([oauth.authorize_url, oauth.token_url] if oauth else [])]
        return cls(
            id=record.service_id,
            title=record.title,
            description=record.description,
            url=record.url,
            auth=record.auth,
            oauth=(
                OAuthConfig(
                    authorize_url=oauth.authorize_url,
                    token_url=oauth.token_url,
                    client_id=oauth.client_id,
                    scopes=list(oauth.scopes),
                    resource=oauth.resource,
                )
                if oauth is not None
                else None
            ),
            # A draft is visible to its creator only, until an allowlist is saved.
            allowed_users=None if record.tools else [record.created_by or RESERVED_USER_LOCAL],
            tools=list(record.tools),
            allow_http=any(url.startswith("http:") for url in urls),
            timeout=record.timeout,
            credential_version=record.credential_version,
            token_endpoint_auth=oauth.token_endpoint_auth if oauth else "none",
            client_registration=oauth.registration if oauth else None,
            created_by=record.created_by,
            created_at=record.created_at,
            updated_at=record.updated_at,
        )


class McpOAuthPreset(BaseModel):
    """A preset's OAuth hints. Never a client secret."""

    model_config = ConfigDict(extra="forbid")
    authorize_url: str | None = Field(default=None, max_length=2048)
    token_url: str | None = Field(default=None, max_length=2048)
    client_id: str | None = Field(default=None, max_length=256)
    scopes: list[str] = Field(default_factory=list, max_length=50)
    resource: str | None = Field(default=None, max_length=2048)


class McpServicePreset(BaseModel):
    """A starting point the Add MCP server form offers; nothing is on until an
    administrator saves it, and the saved service is checked like any other."""

    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,39}$")
    title: str = Field(max_length=120)
    description: str = Field(default="", max_length=1000)
    #: May hold a placeholder such as ``https://<your-host>/mcp`` the admin replaces.
    url: str = Field(default="", max_length=2048)
    auth: Literal["none", "bearer", "oauth"] = "bearer"
    oauth: McpOAuthPreset | None = None
    tools: list[str] = Field(default_factory=list, max_length=200)
    timeout: int = Field(default=60, ge=1, le=300)


class McpRegistryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    public_url: str | None = None
    services: list[McpService] = Field(default_factory=list)
    presets: list[McpServicePreset] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_catalog(self) -> McpRegistryConfig:
        if len({s.id for s in self.services}) != len(self.services):
            raise ValueError("MCP service ids must be unique")
        if any(s.oauth for s in self.services) and not self.public_url:
            raise ValueError("public_url is required for OAuth callbacks")
        if self.public_url:
            parsed = urlsplit(self.public_url)
            if (
                parsed.scheme not in {"http", "https"}
                or not parsed.hostname
                or parsed.username
                or parsed.password
                or parsed.query
                or parsed.fragment
                or (parsed.scheme == "http" and parsed.hostname not in {"localhost", "127.0.0.1"})
            ):
                raise ValueError("public_url must be an HTTPS base URL (HTTP only on localhost)")
        return self

    @classmethod
    def from_env(cls) -> McpRegistryConfig | None:
        path = os.environ.get("OMNIGENT_MCP_REGISTRY")
        return cls.model_validate(yaml.safe_load(Path(path).read_text())) if path else None


def _public_base_url(configured: str | None) -> str | None:
    """The server's public URL for OAuth callbacks: the registry's, else the accounts base URL."""
    url = configured or os.environ.get("OMNIGENT_ACCOUNTS_BASE_URL") or None
    if url is None:
        return None
    parsed = urlsplit(url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or (parsed.scheme == "http" and parsed.hostname not in {"localhost", "127.0.0.1"})
    ):
        return None
    return url.rstrip("/")


def login_matches(service: McpService, connection: ProviderConnection | None) -> bool:
    """Whether a saved login belongs to *service* as it is configured now.

    A run-time service's logins carry its credential version; a registry-file
    service never accepts a login stamped for a run-time service of the same ID.
    """
    if connection is None:
        return False
    stamp = connection.metadata.get(SERVICE_VERSION_KEY)
    if isinstance(service, ManagedMcpService):
        return isinstance(stamp, str) and secrets.compare_digest(stamp, service.credential_version)
    return stamp is None


class McpConnectionStore:
    """Typed facade sharing the existing encrypted connections table."""

    def __init__(
        self, store: CredentialStore, service_id: str, service: McpService | None = None
    ) -> None:
        self._store = store
        self.provider = f"mcp:{service_id}"
        self._service = service

    def get(self, user_id: str, *, with_tokens: bool = False) -> ProviderConnection | None:
        connection = self._store.get(user_id, self.provider, with_secret=with_tokens)
        if self._service is not None and not login_matches(self._service, connection):
            return None
        return connection

    def delete(self, user_id: str) -> bool:
        return self._store.delete(user_id, self.provider)


class McpRegistry:
    """Resolve credentials and execute approved catalog tools on the server.

    The catalog is the registry file's services plus, with a *service_store*, the
    services administrators manage at run time. Those are cached per workspace and
    re-read every *refresh_seconds* (at once after a change through this process),
    so other server processes see a change within that interval without a restart.
    """

    def __init__(
        self,
        config: McpRegistryConfig,
        store: CredentialStore | None,
        *,
        service_store: McpServiceStore | None = None,
        destination_policy: McpDestinationPolicy | None = None,
        refresh_seconds: float = 5.0,
    ) -> None:
        self.config = config
        self.services = {service.id: service for service in config.services}
        self.store = store
        self.service_store = service_store
        self.destination_policy = destination_policy or DefaultMcpDestinationPolicy.from_env()
        self.refresh_seconds = refresh_seconds
        self._managed: dict[int, tuple[float, dict[str, ManagedMcpService]]] = {}
        self._reload_lock = threading.Lock()
        self._reported: set[tuple[str, str]] = set()
        self._locks: WeakValueDictionary[tuple[int, str, str], asyncio.Lock] = (
            WeakValueDictionary()
        )
        if any(s.auth in {"oauth", "bearer"} for s in config.services) and store is None:
            raise ValueError("MCP account connections require the existing KMS or Vault cipher")
        self.state_secret = os.environ.get("OMNIGENT_MCP_OAUTH_STATE_SECRET", "")
        if any(s.oauth for s in config.services) and len(self.state_secret) < 32:
            raise ValueError(
                "Set OMNIGENT_MCP_OAUTH_STATE_SECRET to at least 32 random characters"
            )

    # ── Catalog ─────────────────────────────────────────

    @property
    def manages_services(self) -> bool:
        """Whether administrators can add services at run time."""
        return self.service_store is not None

    def public_url(self) -> str | None:
        return _public_base_url(self.config.public_url)

    def reload(self) -> dict[str, ManagedMcpService]:
        """Re-read the run-time services of the current workspace (blocking)."""
        from omnigent.db.db_models import current_workspace_id

        if self.service_store is None:
            return {}
        workspace_id = current_workspace_id()
        with self._reload_lock:
            records, pending = self.service_store.scan()
            managed: dict[str, ManagedMcpService] = {}
            for record in records:
                problem = None
                if record.service_id in self.services:
                    problem = "is hidden by the registry file's service with the same ID"
                else:
                    try:
                        managed[record.service_id] = ManagedMcpService.from_record(record)
                    except ValueError:
                        problem = "is not usable"
                if problem and (record.service_id, problem) not in self._reported:
                    self._reported.add((record.service_id, problem))
                    _logger.warning("Run-time MCP service %s %s", record.service_id, problem)
            self._managed[workspace_id] = (time.monotonic(), managed)
        if pending:
            try:
                self.service_store.finish_sweeps(pending)
            except Exception:  # noqa: BLE001 — retried at the next load
                _logger.warning("Could not finish removing old MCP logins; will retry")
        return managed

    async def refresh(self, *, force: bool = False) -> None:
        """Re-read run-time services when the cached copy is older than the interval."""
        from omnigent.db.db_models import current_workspace_id

        if self.service_store is None:
            return
        cached = self._managed.get(current_workspace_id())
        if force or cached is None or time.monotonic() - cached[0] >= self.refresh_seconds:
            await asyncio.to_thread(self.reload)

    def managed_services(self) -> dict[str, ManagedMcpService]:
        """The current workspace's run-time services, from the cache (loaded on first use)."""
        from omnigent.db.db_models import current_workspace_id

        if self.service_store is None:
            return {}
        cached = self._managed.get(current_workspace_id())
        return cached[1] if cached is not None else self.reload()

    def catalog(self) -> dict[str, McpService]:
        """Every service by ID; the registry file's win over run-time ones."""
        return {**self.managed_services(), **self.services}

    def lookup(self, service_id: str) -> McpService | None:
        """A service by ID regardless of who may use it."""
        return self.services.get(service_id) or self.managed_services().get(service_id)

    def service(self, service_id: str, user_id: str) -> McpService:
        service = self.lookup(service_id)
        if service is None or not service.permits(user_id):
            raise ConnectionError("MCP service is unavailable for this user")
        return service

    def http_client(self, service: McpService, url: str, **kwargs: Any) -> httpx.AsyncClient:
        """A client for *service* to call *url*.

        A run-time service's URL is checked against the current destination rules,
        and its client connects only to addresses they allow.
        """
        if isinstance(service, ManagedMcpService):
            if self.destination_policy.url_problem(url) is not None:
                raise ConnectionError(
                    "MCP service address is refused by the server's destination rules"
                )
            return guarded_client(self.destination_policy, **kwargs)
        return httpx.AsyncClient(follow_redirects=False, **kwargs)

    # ── Logins ──────────────────────────────────────────

    def login_metadata(self, service: McpService, metadata: dict[str, Any]) -> dict[str, Any]:
        """Metadata for a login saved now: stamped for a run-time service."""
        if isinstance(service, ManagedMcpService):
            return stamped(metadata, service.credential_version)
        return metadata

    @staticmethod
    def _version_guard(service: McpService) -> Any:
        """For a run-time service, a write precondition requiring its current version."""
        if isinstance(service, ManagedMcpService):
            return version_guard(service.id, service.credential_version)
        return None

    def save_login(
        self,
        service: McpService,
        user_id: str,
        provider: str,
        secret: dict[str, Any],
        metadata: dict[str, Any],
    ) -> None:
        """Save a login, refused if a run-time service changed since it was read."""
        assert self.store is not None
        guard = self._version_guard(service)
        try:
            self.store.upsert(
                user_id,
                provider,
                secret=secret,
                metadata=self.login_metadata(service, metadata),
                precondition=guard,
            )
        except McpServiceVersionChanged:
            raise ConnectionError(
                "The MCP service changed while connecting; connect again"
            ) from None
        except SecretTooLargeError as exc:
            raise ConnectionError(str(exc)) from None

    async def token(self, service: McpService, user_id: str, app_state: Any) -> str | None:
        if service.auth == "none":
            return None
        if service.auth in {"github", "databricks"}:
            from omnigent.server.connections_registry import connection_providers

            provider = next(p for p in connection_providers() if p.name == service.auth)
            store = getattr(app_state, f"{provider.name}_store", None)
            client = getattr(app_state, f"{provider.name}_client", None)
            if store is None or provider.credential_resolver is None:
                raise ConnectionError(f"Connect {provider.name} in Sandbox Integrations")
            payload = await provider.credential_resolver(user_id, store=store, client=client)
            if not payload or not isinstance(payload.get("token"), str):
                raise ConnectionError(f"Reconnect {provider.name} in Sandbox Integrations")
            return payload["token"]
        from omnigent.db.db_models import current_workspace_id

        assert self.store is not None
        key = (current_workspace_id(), user_id, service.id)
        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:
            conn = await asyncio.to_thread(
                self.store.get, user_id, f"mcp:{service.id}", with_secret=True
            )
            if (
                conn is None
                or not login_matches(service, conn)
                or not conn.secret
                or not conn.secret.get("access_token")
            ):
                raise ConnectionError("Connect this MCP account in Settings > MCP")
            expiry = conn.metadata.get("expires_at")
            refresh = conn.secret.get("refresh_token")
            expiring = expiry is not None and float(expiry) <= time.time() + 30
            # A rejected login refreshes before use even if its access token looks current.
            if service.oauth and (expiring or (refresh and needs_reconnect(conn))):
                if not refresh:
                    raise ConnectionError("Reconnect this MCP account in Settings > MCP")
                try:
                    tokens = await self.exchange(
                        service, {"grant_type": "refresh_token", "refresh_token": refresh}
                    )
                except McpGrantRejected:
                    await self._mark_needs_reconnect(
                        service, user_id, "refresh_token", refresh, "invalid_grant"
                    )
                    raise
                if tokens.get("refresh_token") is None:
                    tokens["refresh_token"] = refresh
                try:
                    # Saved only while the login is the one refreshed and, for a
                    # run-time service, its settings are unchanged (under the lock
                    # edits take); otherwise the response is discarded.
                    updated = await asyncio.to_thread(
                        self.store.update_secret,
                        user_id,
                        f"mcp:{service.id}",
                        secret=tokens,
                        metadata=self.login_metadata(service, self.metadata(tokens)),
                        if_secret_matches={"refresh_token": refresh},
                        precondition=self._version_guard(service),
                    )
                except SecretTooLargeError as exc:
                    raise ConnectionError(str(exc)) from None
                except McpServiceVersionChanged:
                    raise ConnectionError(
                        "The MCP login changed while it was refreshed; try again"
                    ) from None
                if not updated:
                    current = await asyncio.to_thread(self.store.get, user_id, f"mcp:{service.id}")
                    if current is not None:
                        # Replaced by a reconnect or another server's refresh: that one stays.
                        raise ConnectionError(
                            "The MCP login changed while it was refreshed; try again"
                        )
                    raise ConnectionError("MCP account disconnected")
                return str(tokens["access_token"])
            return str(conn.secret["access_token"])

    async def _mark_needs_reconnect(
        self, service: McpService, user_id: str, field: str, rejected: str, reason: str
    ) -> None:
        """Mark the saved login, unless it no longer holds the *rejected* credential."""
        assert self.store is not None
        try:
            await asyncio.to_thread(
                self.store.merge_metadata,
                user_id,
                f"mcp:{service.id}",
                {NEEDS_RECONNECT: reason},
                if_secret_matches={field: rejected},
            )
        except Exception:  # noqa: BLE001 — keep the caller's failure; store errors are not useful here
            _logger.warning("Could not mark MCP service %s for reconnection", service.id)

    @staticmethod
    def metadata(tokens: dict[str, Any]) -> dict[str, Any]:
        expiry = tokens.get("expires_in")
        if expiry is None:
            return {"expires_at": None}
        try:
            seconds = float(expiry)
            if isinstance(expiry, bool) or not math.isfinite(seconds) or seconds < 0:
                raise ValueError
        except (TypeError, ValueError, OverflowError):
            raise ConnectionError("MCP provider returned an invalid token expiry") from None
        return {"expires_at": time.time() + seconds}

    def client_secret(self, service: McpService) -> str | None:
        """The OAuth client secret: from the environment, or sealed for a run-time service."""
        oauth = service.oauth
        assert oauth is not None
        if not isinstance(service, ManagedMcpService):
            if not oauth.client_secret_env:
                return None
            secret = os.environ.get(oauth.client_secret_env)
            if not secret:
                raise ConnectionError("MCP OAuth client secret is not configured")
            return secret
        if service.token_endpoint_auth == "none":
            return None
        if self.store is None:
            raise ConnectionError("MCP OAuth client secret is not configured")
        saved = self.store.get(SERVICE_PRINCIPAL, client_provider(service.id), with_secret=True)
        value = saved.secret.get("client_secret") if saved and saved.secret else None
        if not login_matches(service, saved) or not isinstance(value, str) or not value:
            raise ConnectionError("MCP OAuth client secret is not configured")
        return value

    def save_client_secret(self, service: ManagedMcpService, client_secret: str) -> None:
        """Seal a run-time service's OAuth client secret for its current version."""
        self.save_login(
            service,
            SERVICE_PRINCIPAL,
            client_provider(service.id),
            {"client_secret": client_secret},
            {},
        )

    async def exchange(self, service: McpService, data: dict[str, Any]) -> dict[str, Any]:
        oauth = service.oauth
        assert oauth is not None
        data = {**data, "client_id": oauth.client_id}
        headers = {"Accept": "application/json"}
        secret = await asyncio.to_thread(self.client_secret, service)
        if secret is not None:
            if (
                isinstance(service, ManagedMcpService)
                and service.token_endpoint_auth == "client_secret_basic"
            ):
                credentials = f"{quote(oauth.client_id, safe='')}:{quote(secret, safe='')}"
                headers["Authorization"] = (
                    "Basic " + base64.b64encode(credentials.encode()).decode()
                )
            else:
                data["client_secret"] = secret
        if oauth.resource:
            data["resource"] = oauth.resource
        try:
            async with self.http_client(service, oauth.token_url, timeout=20) as client:
                response = await client.post(oauth.token_url, data=data, headers=headers)
        except httpx.RequestError:
            raise ConnectionError(
                "MCP authorization could not reach the provider; try again"
            ) from None
        if response.status_code != 200:
            try:
                error = response.json().get("error")
            except (ValueError, AttributeError):
                error = None
            failure = McpGrantRejected if error == "invalid_grant" else ConnectionError
            raise failure("MCP authorization failed; reconnect the account")
        try:
            tokens = response.json()
        except ValueError:
            raise ConnectionError("MCP provider returned an invalid token response") from None
        if not isinstance(tokens, dict):
            raise ConnectionError("MCP provider returned an invalid token response")
        access_token = tokens.get("access_token")
        token_type = tokens.get("token_type", "Bearer")
        refresh_token = tokens.get("refresh_token")
        if (
            not isinstance(access_token, str)
            or not access_token
            or any(c.isspace() for c in access_token)
            or not isinstance(token_type, str)
            or token_type.lower() != "bearer"
            or (
                refresh_token is not None
                and (not isinstance(refresh_token, str) or not refresh_token)
            )
        ):
            raise ConnectionError("MCP provider did not return a valid bearer grant")
        self.metadata(tokens)
        return tokens

    async def execute(
        self,
        service_id: str,
        user_id: str,
        app_state: Any,
        *,
        tool: str | None = None,
        arguments: dict[str, Any] | None = None,
        all_tools: bool = False,
    ) -> Any:
        """Call *tool*, or list the allowed tools (every tool with *all_tools*, for discovery)."""
        service = self.service(service_id, user_id)
        if tool is not None and tool not in service.tools:
            raise ConnectionError("Tool is not allowed by the MCP registry")
        token = await self.token(service, user_id, app_state)
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        try:
            async with asyncio.timeout(service.timeout):
                async with (
                    self.http_client(
                        service, service.url, headers=headers, timeout=service.timeout
                    ) as client,
                    streamable_http_client(service.url, http_client=client) as (
                        read,
                        write,
                        _,
                    ),
                    ClientSession(read, write) as session,
                ):
                    await session.initialize()
                    if tool is not None:
                        return await session.call_tool(tool, arguments or {})
                    tools = []
                    cursor = None
                    for _ in range(100):
                        result = await session.list_tools(cursor=cursor)
                        tools.extend(
                            t for t in result.tools if all_tools or t.name in service.tools
                        )
                        cursor = result.nextCursor
                        if not cursor:
                            return tools
                    raise ConnectionError("MCP tool catalog exceeded the pagination limit")
        except ConnectionError:
            raise
        except Exception as exc:  # noqa: BLE001 — transport errors may contain credentials
            failure = _upstream_error(exc)
            _logger.warning(
                "Registry MCP request failed service=%s tool=%s failure=%s http_status=%s",
                service.id,
                tool or "tools/list",
                failure.kind,
                failure.status_code,
            )
            if failure.status_code == 401 and token and service.auth in {"oauth", "bearer"}:
                await self._mark_needs_reconnect(
                    service, user_id, "access_token", token, "http_401"
                )
            raise failure from None


class McpOAuthHooks:
    """Reuse the existing user-bound OAuth connect/callback/disconnect routes."""

    def __init__(self, registry: McpRegistry, service: McpService) -> None:
        assert registry.store is not None
        self.registry = registry
        self.service = service
        self.provider = f"mcp-{service.id}"
        self.store = McpConnectionStore(registry.store, service.id, service)

    def signing_key(self) -> str:
        return self.registry.state_secret

    def status_fields(self, connection: ProviderConnection | None) -> dict[str, Any]:
        return {"needs_reconnect": needs_reconnect(connection)}

    def _version(self) -> str | None:
        service = self.service
        return service.credential_version if isinstance(service, ManagedMcpService) else None

    def begin(self, request: Any, build_state: Any) -> ConnectStart:
        oauth = self.service.oauth
        assert oauth is not None
        from omnigent.server.routes._auth_helpers import require_user

        user_id = require_user(request, request.app.state.mcp_registry_auth) or RESERVED_USER_LOCAL
        self.registry.service(self.service.id, user_id)
        if isinstance(self.service, ManagedMcpService) and (
            self.registry.destination_policy.url_problem(oauth.authorize_url) is not None
        ):
            raise ConnectionError("MCP sign-in address is refused by the destination rules")
        verifier = secrets.token_urlsafe(48)
        nonce = secrets.token_urlsafe(32)
        self.registry.save_login(
            self.service,
            user_id,
            f"mcp-pending:{self.service.id}",
            {"verifier": verifier, "nonce": nonce},
            {},
        )
        claims: dict[str, Any] = {"nonce": nonce, "service": self.service.id}
        if self._version() is not None:
            claims[SERVICE_VERSION_KEY] = self._version()
        params = {
            "client_id": oauth.client_id,
            "response_type": "code",
            "redirect_uri": self.redirect_uri(),
            "scope": " ".join(oauth.scopes),
            "state": build_state(claims),
            "code_challenge": base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
            .rstrip(b"=")
            .decode(),
            "code_challenge_method": "S256",
        }
        if not oauth.scopes and isinstance(self.service, ManagedMcpService):
            # Without scopes the authorization server grants its default ones.
            del params["scope"]
        if oauth.resource:
            params["resource"] = oauth.resource
        separator = "&" if "?" in oauth.authorize_url else "?"
        return ConnectStart(authorize_url=oauth.authorize_url + separator + urlencode(params))

    def redirect_uri(self) -> str:
        base = self.registry.public_url()
        if base is None:
            raise ConnectionError("Set the server's public URL to sign in to MCP services")
        return f"{base}/v1/connections/{self.provider}/callback"

    async def complete(self, user_id: str, code: str, claims: dict[str, Any]) -> None:
        self.registry.service(self.service.id, user_id)
        store = self.registry.store
        assert store is not None
        pending = await asyncio.to_thread(
            store.get, user_id, f"mcp-pending:{self.service.id}", with_secret=True
        )
        if (
            not pending
            or not pending.secret
            or claims.get("service") != self.service.id
            or claims.get(SERVICE_VERSION_KEY) != self._version()
            or not login_matches(self.service, pending)
            or not secrets.compare_digest(
                str(pending.secret.get("nonce", "")), str(claims.get("nonce", ""))
            )
        ):
            raise ConnectionError("MCP authorization expired; try connecting again")
        await asyncio.to_thread(store.delete, user_id, f"mcp-pending:{self.service.id}")
        tokens = await self.registry.exchange(
            self.service,
            {
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": self.redirect_uri(),
                "code_verifier": pending.secret["verifier"],
            },
        )
        await asyncio.to_thread(
            self.registry.save_login,
            self.service,
            user_id,
            f"mcp:{self.service.id}",
            tokens,
            self.registry.metadata(tokens),
        )
