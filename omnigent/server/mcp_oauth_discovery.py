"""OAuth discovery and dynamic client registration for run-time MCP services.

Follows the MCP authorization specification: the server's protected-resource
metadata (RFC 9728, found through the ``WWW-Authenticate`` challenge or the
well-known path) names its authorization server, whose metadata (RFC 8414 or
OpenID discovery) names the endpoints; a server without either is assumed to
use the default endpoints at its origin. Where the authorization server offers
registration (RFC 7591), Omnigent registers itself as a client, preferring a
public client (PKCE, no secret to keep) when the server supports one.

Every URL is checked against the destination policy before it is fetched, and
fetched through the guarded client, so discovery cannot be steered to a private
address by the metadata it reads.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import urlsplit, urlunsplit

import httpx
from mcp.shared.auth import OAuthMetadata, ProtectedResourceMetadata
from pydantic import ValidationError

from omnigent.server.mcp_destinations import (
    McpDestinationError,
    McpDestinationPolicy,
    check_destination,
    guarded_client,
)

_logger = logging.getLogger(__name__)

#: Largest metadata or registration response read.
MAX_RESPONSE_BYTES = 64 * 1024
_TIMEOUT_S = 15.0
_MCP_PROTOCOL_VERSION = "2025-06-18"


class McpOAuthDiscoveryError(Exception):
    """Discovery or registration failed; the message is safe to show an administrator."""


@dataclass(frozen=True)
class DiscoveredOAuth:
    """What the service's metadata says about signing in."""

    authorize_url: str
    token_url: str
    resource: str
    scopes: tuple[str, ...] = ()
    registration_endpoint: str | None = None
    token_endpoint_auth_methods: tuple[str, ...] | None = None


TokenEndpointAuth = Literal["none", "client_secret_post", "client_secret_basic"]


@dataclass(frozen=True)
class RegisteredClient:
    """An OAuth client registered for Omnigent. ``client_secret`` is a credential."""

    client_id: str
    client_secret: str | None
    token_endpoint_auth: TokenEndpointAuth

    def __repr__(self) -> str:
        return f"RegisteredClient(client_id={self.client_id!r}, auth={self.token_endpoint_auth!r})"


def _origin(url: str) -> str:
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, "", "", ""))


def _path(url: str) -> str:
    return urlsplit(url).path.rstrip("/")


async def _get_json(
    client: httpx.AsyncClient, policy: McpDestinationPolicy, url: str
) -> dict[str, Any] | None:
    """GET *url* as JSON, or ``None`` when it is absent or not a JSON object."""
    try:
        await check_destination(policy, url, what="A discovery URL")
    except McpDestinationError:
        return None
    try:
        async with client.stream("GET", url, headers={"Accept": "application/json"}) as response:
            if response.status_code != 200:
                return None
            body = b""
            async for chunk in response.aiter_bytes():
                body += chunk
                if len(body) > MAX_RESPONSE_BYTES:
                    return None
    except httpx.HTTPError:
        return None
    try:
        data = json.loads(body)
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def _challenge_metadata_url(header: str) -> str | None:
    """The ``resource_metadata`` parameter of a ``WWW-Authenticate: Bearer`` challenge."""
    marker = "resource_metadata="
    index = header.find(marker)
    if index < 0:
        return None
    value = header[index + len(marker) :].lstrip()
    if value.startswith('"'):
        end = value.find('"', 1)
        return value[1:end] if end > 0 else None
    return value.split(",", 1)[0].strip() or None


async def _protected_resource(
    client: httpx.AsyncClient, policy: McpDestinationPolicy, url: str
) -> ProtectedResourceMetadata | None:
    candidates: list[str] = []
    try:
        probe = await client.post(
            url,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": _MCP_PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "omnigent", "version": "1"},
                },
            },
            headers={"Accept": "application/json, text/event-stream"},
        )
        if probe.status_code == 401:
            challenged = _challenge_metadata_url(probe.headers.get("WWW-Authenticate", ""))
            if challenged:
                candidates.append(challenged)
    except httpx.HTTPError:
        pass
    path = _path(url)
    candidates.append(f"{_origin(url)}/.well-known/oauth-protected-resource{path}")
    if path:
        candidates.append(f"{_origin(url)}/.well-known/oauth-protected-resource")
    for candidate in candidates:
        data = await _get_json(client, policy, candidate)
        if data is None:
            continue
        try:
            return ProtectedResourceMetadata.model_validate(data)
        except ValidationError:
            continue
    return None


async def _authorization_server(
    client: httpx.AsyncClient, policy: McpDestinationPolicy, issuer: str
) -> OAuthMetadata | None:
    origin, path = _origin(issuer), _path(issuer)
    candidates = [f"{origin}/.well-known/oauth-authorization-server{path}"]
    candidates.append(f"{origin}/.well-known/openid-configuration{path}")
    if path:
        candidates.append(f"{origin}{path}/.well-known/openid-configuration")
    for candidate in candidates:
        data = await _get_json(client, policy, candidate)
        if data is None:
            continue
        try:
            return OAuthMetadata.model_validate(data)
        except ValidationError:
            continue
    return None


async def discover_oauth(url: str, policy: McpDestinationPolicy) -> DiscoveredOAuth:
    """Find a service's authorization endpoints, starting from its MCP URL."""
    async with guarded_client(policy, timeout=_TIMEOUT_S) as client:
        resource_metadata = await _protected_resource(client, policy, url)
        issuer = (
            str(resource_metadata.authorization_servers[0])
            if resource_metadata is not None
            else _origin(url)
        )
        server = await _authorization_server(client, policy, issuer)
    resource = str(resource_metadata.resource) if resource_metadata is not None else url
    if server is None:
        if resource_metadata is not None:
            raise McpOAuthDiscoveryError(
                "The service names an authorization server that publishes no metadata. "
                "Enter its authorization and token URLs."
            )
        # The MCP specification's defaults for a server without metadata.
        origin = _origin(url)
        return DiscoveredOAuth(
            authorize_url=f"{origin}/authorize",
            token_url=f"{origin}/token",
            resource=resource,
            registration_endpoint=f"{origin}/register",
        )
    if server.code_challenge_methods_supported is not None and (
        "S256" not in server.code_challenge_methods_supported
    ):
        raise McpOAuthDiscoveryError(
            "The service's authorization server does not support PKCE (S256), which "
            "Omnigent requires."
        )
    scopes = (resource_metadata.scopes_supported if resource_metadata else None) or []
    return DiscoveredOAuth(
        authorize_url=str(server.authorization_endpoint),
        token_url=str(server.token_endpoint),
        resource=resource,
        scopes=tuple(scopes),
        registration_endpoint=(
            str(server.registration_endpoint) if server.registration_endpoint else None
        ),
        token_endpoint_auth_methods=(
            tuple(server.token_endpoint_auth_methods_supported)
            if server.token_endpoint_auth_methods_supported is not None
            else None
        ),
    )


def _preferred_auth_method(supported: tuple[str, ...] | None) -> TokenEndpointAuth:
    if supported is None:
        return "client_secret_post"
    methods: tuple[TokenEndpointAuth, ...] = ("none", "client_secret_post", "client_secret_basic")
    for method in methods:
        if method in supported:
            return method
    raise McpOAuthDiscoveryError(
        "The service's authorization server supports no client authentication Omnigent "
        "can use. Enter a client ID and secret."
    )


async def register_client(
    discovered: DiscoveredOAuth,
    redirect_uri: str,
    policy: McpDestinationPolicy,
    *,
    client_name: str = "Omnigent",
) -> RegisteredClient:
    """Register Omnigent as an OAuth client of the service's authorization server."""
    endpoint = discovered.registration_endpoint
    if endpoint is None:
        raise McpOAuthDiscoveryError(
            "The service does not offer automatic client registration. Register Omnigent "
            "with the service and enter its client ID and secret."
        )
    try:
        await check_destination(policy, endpoint, what="The registration URL")
    except McpDestinationError as exc:
        raise McpOAuthDiscoveryError(str(exc)) from None
    method = _preferred_auth_method(discovered.token_endpoint_auth_methods)
    body: dict[str, Any] = {
        "client_name": client_name,
        "redirect_uris": [redirect_uri],
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
        "token_endpoint_auth_method": method,
    }
    if discovered.scopes:
        body["scope"] = " ".join(discovered.scopes)
    try:
        async with guarded_client(policy, timeout=_TIMEOUT_S) as client:
            response = await client.post(
                endpoint, json=body, headers={"Accept": "application/json"}
            )
    except httpx.HTTPError:
        raise McpOAuthDiscoveryError(
            "Omnigent could not reach the service's registration endpoint."
        ) from None
    if response.status_code not in (200, 201) or len(response.content) > MAX_RESPONSE_BYTES:
        _logger.warning("MCP client registration refused http_status=%s", response.status_code)
        raise McpOAuthDiscoveryError(
            f"The service refused automatic client registration (HTTP {response.status_code}). "
            "Enter a client ID and secret instead."
        )
    try:
        data = response.json()
    except ValueError:
        data = None
    client_id = data.get("client_id") if isinstance(data, dict) else None
    if not isinstance(client_id, str) or not client_id or len(client_id) > 256:
        raise McpOAuthDiscoveryError(
            "The service's registration response has no usable client ID."
        )
    assert isinstance(data, dict)
    secret = data.get("client_secret")
    granted = data.get("token_endpoint_auth_method", method)
    if granted == "none":
        return RegisteredClient(
            client_id=client_id, client_secret=None, token_endpoint_auth="none"
        )
    if granted not in ("client_secret_post", "client_secret_basic") or (
        not isinstance(secret, str) or not secret
    ):
        raise McpOAuthDiscoveryError(
            "The service registered a client Omnigent cannot use. Enter a client ID and secret."
        )
    return RegisteredClient(
        client_id=client_id,
        client_secret=secret,
        token_endpoint_auth="client_secret_basic"
        if granted == "client_secret_basic"
        else "client_secret_post",
    )
