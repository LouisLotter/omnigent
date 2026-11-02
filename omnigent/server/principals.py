"""The kind of caller behind an authenticated request.

Authentication resolves the account a credential speaks for. A runner's
token, a host connection and a person's browser can all resolve to the same
account, so decisions that need the person themselves (approvals, team
administration) must also look at the principal kind recorded here.

Runners and hosts are machine principals. Their tokens are minted from the
server's current binding (runner -> thread -> owner, thread -> host -> owner),
carry their own audience and signing key, and never act as an administrator.
"""

from __future__ import annotations

import dataclasses
import hashlib
import hmac
import time
import uuid
from collections.abc import Iterable
from enum import Enum

from starlette.requests import HTTPConnection

_PRINCIPAL_SCOPE_KEY = "omnigent.principal"


class PrincipalKind(str, Enum):
    """What holds the credential.

    - ``HUMAN``: the person's own sign-in, such as a browser session, a CLI
      login or a trusted proxy's identity header.
    - ``DELEGATED_CLIENT``: an application the person approved to act for
      them, such as a device-authorization client.
    - ``RUNNER``: a runner acting for its thread's owner.
    - ``HOST``: a host connection acting for the host's owner.
    - ``SERVICE``: a machine client with no person behind it.
    """

    HUMAN = "human"
    DELEGATED_CLIENT = "delegated_client"
    RUNNER = "runner"
    HOST = "host"
    SERVICE = "service"


MACHINE_KINDS = frozenset({PrincipalKind.RUNNER, PrincipalKind.HOST})

# Operations a runner token may perform. Each machine-reachable route names one
# (see ``omnigent.server.route_authority``); a token without it is refused.
SCOPE_SESSION = "session"
SCOPE_SESSIONS_CREATE = "sessions.create"
SCOPE_SESSIONS_LIST = "sessions.list"
SCOPE_SCHEDULES = "schedules"
SCOPE_CATALOG = "catalog"
SCOPE_AGENTS = "agents"
SCOPE_MCP = "mcp"
SCOPE_TUNNEL = "tunnel"
RUNNER_SCOPES = frozenset(
    {
        SCOPE_SESSION,
        SCOPE_SESSIONS_CREATE,
        SCOPE_SESSIONS_LIST,
        SCOPE_SCHEDULES,
        SCOPE_CATALOG,
        SCOPE_AGENTS,
        SCOPE_MCP,
        SCOPE_TUNNEL,
    }
)
SCOPE_HOST_TUNNEL = "host.tunnel"
HOST_SCOPES = frozenset({SCOPE_HOST_TUNNEL})

RUNNER_AUDIENCE = "omnigent:runner"
HOST_AUDIENCE = "omnigent:host"
_AUDIENCES = {PrincipalKind.RUNNER: RUNNER_AUDIENCE, PrincipalKind.HOST: HOST_AUDIENCE}
_DEFAULT_SCOPES = {PrincipalKind.RUNNER: RUNNER_SCOPES, PrincipalKind.HOST: HOST_SCOPES}
# Domain separation: machine tokens are signed with a key derived from the
# server secret, so no session-cookie decoder can ever accept one.
_MACHINE_KEY_LABEL = b"omnigent machine principal v1"


@dataclasses.dataclass(frozen=True)
class Principal:
    """An authenticated caller.

    :param kind: What holds the credential.
    :param user_id: The account it acts for, or ``None`` for a service.
    :param runner_id: The runner, for a runner principal. A runner id names one
        launch (its binding token), so it is also the runner's incarnation.
    :param host_id: The host, for a runner or host principal.
    :param session_id: The thread a runner token was minted for.
    :param account_generation: The account registration the token was minted
        for, when the account has one.
    :param token_id: The machine token's unique id.
    :param audience: The machine token's audience.
    :param expires_at: The machine token's expiry, in epoch seconds.
    :param scopes: Operations a machine principal may perform.
    """

    kind: PrincipalKind
    user_id: str | None
    runner_id: str | None = None
    host_id: str | None = None
    session_id: str | None = None
    account_generation: str | None = None
    token_id: str | None = None
    audience: str | None = None
    expires_at: int | None = None
    scopes: frozenset[str] = frozenset()

    @property
    def is_human(self) -> bool:
        """Whether the person is acting in person rather than through software."""
        return self.kind is PrincipalKind.HUMAN

    @property
    def is_machine(self) -> bool:
        """Whether a runner or host holds the credential."""
        return self.kind in MACHINE_KINDS

    def allows(self, scope: str) -> bool:
        """Whether this machine principal holds *scope*."""
        return scope in self.scopes


def default_scopes(kind: PrincipalKind) -> frozenset[str]:
    """The scopes a machine principal of *kind* is minted with."""
    return _DEFAULT_SCOPES.get(kind, frozenset())


@dataclasses.dataclass(frozen=True)
class MachineBinding:
    """The server-side binding a machine token is minted from and checked against.

    :param kind: ``RUNNER`` or ``HOST``.
    :param user_id: The person the binding acts for: the thread's single owner,
        who also owns the host.
    :param account_generation: That person's account registration, or ``None``
        for an identity without an accounts row.
    :param host_id: The host the thread is bound to (or the host itself).
    :param session_id: The thread, for a runner binding.
    :param runner_id: The runner, for a runner binding.
    """

    kind: PrincipalKind
    user_id: str
    account_generation: str | None
    host_id: str
    session_id: str | None = None
    runner_id: str | None = None

    def matches(self, principal: Principal) -> bool:
        """Whether *principal* was minted from this binding."""
        return (
            principal.kind is self.kind
            and principal.user_id == self.user_id
            and principal.account_generation == self.account_generation
            and principal.host_id == self.host_id
            and principal.session_id == self.session_id
            and principal.runner_id == self.runner_id
        )


def _machine_key(secret: bytes) -> bytes:
    return hmac.new(secret, _MACHINE_KEY_LABEL, hashlib.sha256).digest()


def mint_machine_token(
    binding: MachineBinding,
    secret: bytes,
    ttl_seconds: int,
    *,
    scopes: Iterable[str] | None = None,
) -> tuple[str, int]:
    """Sign a machine token for *binding*.

    :param binding: The current server-side binding, never caller-supplied.
    :param secret: The server's session secret; the token uses a derived key.
    :param ttl_seconds: Lifetime in seconds.
    :param scopes: Operations granted; defaults to the kind's standard set.
        Only attenuation is possible: unknown scopes are dropped.
    :returns: ``(token, expires_at)``.
    """
    import jwt

    allowed = default_scopes(binding.kind)
    granted = allowed if scopes is None else allowed & frozenset(scopes)
    now = int(time.time())
    expires_at = now + ttl_seconds
    claims: dict[str, object] = {
        "aud": _AUDIENCES[binding.kind],
        "kind": binding.kind.value,
        "sub": binding.user_id,
        "host": binding.host_id,
        "jti": uuid.uuid4().hex,
        "iat": now,
        "exp": expires_at,
        "scope": sorted(granted),
    }
    if binding.account_generation is not None:
        claims["gen"] = binding.account_generation
    if binding.session_id is not None:
        claims["sid"] = binding.session_id
    if binding.runner_id is not None:
        claims["rid"] = binding.runner_id
    return jwt.encode(claims, _machine_key(secret), algorithm="HS256"), expires_at


def _optional_str(claims: dict[str, object], name: str) -> str | None:
    value = claims.get(name)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(name)
    return value


def decode_machine_token(token: str, secret: bytes) -> Principal | None:
    """Verify a machine token's signature, audience and expiry.

    This proves only that this server minted the token. The caller must still
    check the claims against the current binding before trusting them.

    :param token: The bearer token.
    :param secret: The server's session secret.
    :returns: The principal the token names, or ``None`` when it is not a
        valid machine token.
    """
    import jwt

    try:
        claims = jwt.decode(
            token,
            _machine_key(secret),
            algorithms=["HS256"],
            audience=list(_AUDIENCES.values()),
            options={"require": ["aud", "exp", "iat", "jti", "sub"]},
        )
        kind = PrincipalKind(claims.get("kind"))
        if kind not in MACHINE_KINDS or claims.get("aud") != _AUDIENCES[kind]:
            return None
        user_id = _optional_str(claims, "sub")
        host_id = _optional_str(claims, "host")
        session_id = _optional_str(claims, "sid")
        runner_id = _optional_str(claims, "rid")
        token_id = _optional_str(claims, "jti")
        generation = _optional_str(claims, "gen")
        scope = claims.get("scope")
        if not isinstance(scope, list) or not all(isinstance(item, str) for item in scope):
            return None
        expires_at = claims["exp"]
        if not isinstance(expires_at, int):
            return None
    except (jwt.InvalidTokenError, ValueError):
        return None
    if user_id is None or host_id is None:
        return None
    if kind is PrincipalKind.RUNNER and (session_id is None or runner_id is None):
        return None
    return Principal(
        kind,
        user_id,
        runner_id=runner_id,
        host_id=host_id,
        session_id=session_id,
        account_generation=generation,
        token_id=token_id,
        audience=_AUDIENCES[kind],
        expires_at=expires_at,
        scopes=frozenset(scope) & default_scopes(kind),
    )


def record_principal(connection: HTTPConnection, principal: Principal) -> None:
    """Attach the authenticated principal to a request or WebSocket scope."""
    connection.scope[_PRINCIPAL_SCOPE_KEY] = principal


def clear_principal(connection: HTTPConnection) -> None:
    """Drop any principal before authenticating the connection again."""
    connection.scope.pop(_PRINCIPAL_SCOPE_KEY, None)


def recorded_principal(connection: HTTPConnection) -> Principal | None:
    """Return the principal recorded for this request, if any."""
    principal = connection.scope.get(_PRINCIPAL_SCOPE_KEY)
    return principal if isinstance(principal, Principal) else None
