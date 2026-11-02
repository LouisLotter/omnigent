"""Where a run-time MCP service may send the server or a member's browser.

Services an administrator adds in Settings are checked against a destination
policy twice: when they are saved (the URL's form and every address its host
resolves to), and on every connection the server opens for them, where the
resolved address is checked again and the connection is pinned to it, so a name
that later resolves elsewhere (DNS rebinding) is refused. Redirects are never
followed and no proxy from the environment is used, since either would connect
somewhere the check did not see.

:class:`DefaultMcpDestinationPolicy` is Omnigent's rule set: ``https`` only, no
credentials, query or fragment in the URL, and only globally routable addresses
(no loopback, private, link-local, shared or reserved ranges). A deployment can
pass a stricter :class:`McpDestinationPolicy` to the registry. Services from the
server's registry file are the operator's own choice and are not checked here.
"""

from __future__ import annotations

import ipaddress
import os
import socket
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Protocol, cast
from urllib.parse import urlsplit

import anyio
import httpcore
import httpx

IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address

#: Allow ``http`` and private or loopback destinations, for a server on a trusted
#: network or a developer's machine. Never set it on an internet-facing server.
ALLOW_PRIVATE_DESTINATIONS_ENV = "OMNIGENT_MCP_ALLOW_PRIVATE_DESTINATIONS"
MAX_URL_LENGTH = 2048
_NAT64_PREFIXES = (
    ipaddress.ip_network("64:ff9b::/96"),
    ipaddress.ip_network("64:ff9b:1::/48"),
)
_INTERNAL_SUFFIXES = ("localhost", "local", "internal", "localdomain", "home.arpa")


class McpDestinationError(ValueError):
    """A destination breaks the policy. The message never repeats the URL."""


class DestinationRefused(httpcore.ConnectError):
    """A connection's resolved address is not allowed; raised before connecting."""


class McpDestinationPolicy(Protocol):
    """Rules for the destinations of run-time MCP services."""

    def url_problem(self, url: str) -> str | None:
        """Why *url* is refused, in words that never repeat it; ``None`` if allowed."""

    def address_allowed(self, address: IPAddress) -> bool:
        """Whether the server may connect to *address*."""


def _embedded_ipv4(address: ipaddress.IPv6Address) -> ipaddress.IPv4Address | None:
    if address.ipv4_mapped is not None:
        return address.ipv4_mapped
    if address.sixtofour is not None:
        return address.sixtofour
    if address.teredo is not None:
        return address.teredo[1]
    if any(address in prefix for prefix in _NAT64_PREFIXES):
        return ipaddress.IPv4Address(int(address) & 0xFFFFFFFF)
    return None


def globally_routable(address: IPAddress) -> bool:
    """Whether *address* is a public unicast address, including any IPv4 it embeds."""
    if isinstance(address, ipaddress.IPv6Address):
        embedded = _embedded_ipv4(address)
        if embedded is not None and not globally_routable(embedded):
            return False
    return address.is_global and not address.is_multicast


def _ip_literal(host: str) -> IPAddress | None:
    try:
        return ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        return None


@dataclass(frozen=True)
class DefaultMcpDestinationPolicy:
    """Omnigent's destination rules for run-time MCP services.

    :param allow_private: Also allow ``http`` and loopback, private or internal
        hosts (``OMNIGENT_MCP_ALLOW_PRIVATE_DESTINATIONS=1``).
    """

    allow_private: bool = False

    @classmethod
    def from_env(cls) -> DefaultMcpDestinationPolicy:
        return cls(allow_private=os.environ.get(ALLOW_PRIVATE_DESTINATIONS_ENV, "") == "1")

    def url_problem(self, url: str) -> str | None:
        if not url or len(url) > MAX_URL_LENGTH:
            return f"it must be a URL of at most {MAX_URL_LENGTH} characters"
        if not url.isascii() or any(
            ch.isspace() or ch == "\\" or not ch.isprintable() for ch in url
        ):
            return "it must be a plain ASCII URL"
        try:
            parts = urlsplit(url)
            port = parts.port
        except ValueError:
            return "it must be a well-formed URL"
        if "@" in parts.netloc:
            return "it must not carry credentials"
        if parts.query or parts.fragment or "?" in url or "#" in url:
            return "it must not carry a query or a fragment"
        if parts.scheme not in ({"https", "http"} if self.allow_private else {"https"}):
            return "it must be https"
        host = (parts.hostname or "").rstrip(".")
        if not host:
            return "it must name a host"
        if port == 0:
            return "it must not use port 0"
        if self.allow_private:
            return None
        literal = _ip_literal(host)
        if literal is not None:
            return None if self.address_allowed(literal) else "it must be a public address"
        labels = host.split(".")
        if len(labels) < 2:
            return "it must name a public host (a name with a top-level domain)"
        if host == "localhost" or any(
            host == suffix or host.endswith("." + suffix) for suffix in _INTERNAL_SUFFIXES
        ):
            return "it must not be a local or internal host"
        return None

    def address_allowed(self, address: IPAddress) -> bool:
        return self.allow_private or globally_routable(address)


async def resolve(host: str, port: int) -> list[IPAddress]:
    """Every address *host* resolves to for a TCP connection, in resolver order."""
    literal = _ip_literal(host)
    if literal is not None:
        return [literal]
    try:
        infos = await anyio.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError:
        return []
    addresses: list[IPAddress] = []
    for _family, _type, _proto, _name, sockaddr in infos:
        address = ipaddress.ip_address(str(sockaddr[0]).split("%", 1)[0])
        if address not in addresses:
            addresses.append(address)
    return addresses


async def check_destination(policy: McpDestinationPolicy, url: str, *, what: str) -> None:
    """Check *url*'s form and every address its host resolves to.

    :param what: Names the URL in the error, e.g. ``"The server URL"``.
    :raises McpDestinationError: When the policy refuses it or the host does not resolve.
    """
    problem = policy.url_problem(url)
    if problem is not None:
        raise McpDestinationError(f"{what} breaks a destination rule: {problem}.")
    parts = urlsplit(url)
    host = (parts.hostname or "").rstrip(".")
    port = parts.port or (443 if parts.scheme == "https" else 80)
    addresses = await resolve(host, port)
    if not addresses:
        raise McpDestinationError(f"{what}'s host name does not resolve.")
    if not all(policy.address_allowed(address) for address in addresses):
        raise McpDestinationError(
            f"{what}'s host resolves to an address the destination rules refuse "
            "(for example a private or loopback address)."
        )


class _GuardedNetworkBackend(httpcore.AsyncNetworkBackend):
    """Resolve, check and connect to an allowed address; refuse everything else."""

    def __init__(self, policy: McpDestinationPolicy) -> None:
        self._policy = policy
        # httpcore types AnyIOBackend as a placeholder when anyio is optional.
        self._backend = cast(httpcore.AsyncNetworkBackend, httpcore.AnyIOBackend())

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Iterable[Any] | None = None,
    ) -> httpcore.AsyncNetworkStream:
        addresses = await resolve(host, port)
        if not addresses:
            raise httpcore.ConnectError("The MCP server's host name does not resolve.")
        # One refused answer refuses the name: a rebinding resolver mixes them.
        if not all(self._policy.address_allowed(address) for address in addresses):
            raise DestinationRefused(
                "The MCP server's address is refused by the destination rules."
            )
        failure: Exception | None = None
        for address in addresses:
            try:
                # TLS still verifies and sends the host name: httpcore passes it separately.
                return await self._backend.connect_tcp(
                    str(address),
                    port,
                    timeout=timeout,
                    local_address=local_address,
                    socket_options=socket_options,
                )
            except (httpcore.ConnectError, httpcore.ConnectTimeout, OSError) as exc:
                failure = exc
        assert failure is not None
        raise failure

    async def connect_unix_socket(
        self,
        path: str,  # noqa: ARG002 — the backend interface's signature
        timeout: float | None = None,  # noqa: ARG002
        socket_options: Iterable[Any] | None = None,  # noqa: ARG002
    ) -> httpcore.AsyncNetworkStream:
        raise DestinationRefused("Unix sockets are not MCP destinations.")

    async def sleep(self, seconds: float) -> None:
        await self._backend.sleep(seconds)


class _GuardedTransport(httpx.AsyncHTTPTransport):
    """httpx's transport over :class:`_GuardedNetworkBackend`, without environment proxies."""

    def __init__(self, policy: McpDestinationPolicy) -> None:
        super().__init__(trust_env=False)
        limits = httpx.Limits(max_connections=100, max_keepalive_connections=20)
        # httpx exposes no network-backend option; its transport sends through ``_pool``.
        self._pool = httpcore.AsyncConnectionPool(
            ssl_context=httpx.create_ssl_context(),
            max_connections=limits.max_connections,
            max_keepalive_connections=limits.max_keepalive_connections,
            keepalive_expiry=5.0,
            network_backend=_GuardedNetworkBackend(policy),
        )


def guarded_client(policy: McpDestinationPolicy, **kwargs: Any) -> httpx.AsyncClient:
    """An HTTP client that connects only where *policy* allows and never redirects."""
    return httpx.AsyncClient(
        transport=_GuardedTransport(policy), trust_env=False, follow_redirects=False, **kwargs
    )


def destination_refused(exc: BaseException) -> bool:
    """Whether *exc* (or what caused it) is a refusal by the connection guard."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        if isinstance(current, DestinationRefused):
            return True
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return False
