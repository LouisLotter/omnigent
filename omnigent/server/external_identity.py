"""A person's identity verified by a trusted front proxy, and the same-person rule.

A deployment behind an identity-aware proxy (Cloudflare Access, a cloud IAP, an
SSO gateway) can have the proxy sign the identity it verified into a short-lived
envelope on every request it forwards. Omnigent binds that ``(issuer, subject)``
to one team member and then refuses an Omnigent credential that belongs to
anyone else: a valid sign-in at the proxy for Alice with Bob's session cookie,
bearer, refresh grant, runner token or WebSocket handshake fails.

The envelope is ``v1.<payload>.<mac>``: base64url JSON and an HMAC-SHA256 under
a key the proxy and the server share (``OMNIGENT_EXTERNAL_IDENTITY_KEY``). It
is not a credential on its own: it never signs anyone in, it only says who the
proxy admitted. The proxy must strip any envelope a client sends.

Modes (``OMNIGENT_EXTERNAL_IDENTITY_MODE``):

- ``off`` (default): the header is ignored; behaviour is upstream's.
- ``bind``: an envelope, when present, must verify and must name the member
  whose credential the request carries; a request without one is unchanged.
- ``required``: additionally, a person's credential is honoured only together
  with an envelope, so people reach Omnigent only through the proxy. Machine
  principals (runners, hosts) and anonymous requests are unaffected.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import dataclasses
import hashlib
import hmac
import json
import os
import time
from collections.abc import Mapping
from enum import Enum
from typing import TYPE_CHECKING

from starlette.requests import HTTPConnection
from starlette.types import ASGIApp, Receive, Scope, Send

from omnigent.server.principals import Principal, PrincipalKind, recorded_principal

if TYPE_CHECKING:
    from omnigent.entities.team import TeamMember
    from omnigent.server.auth import AuthProvider
    from omnigent.stores.team_store import SqlAlchemyTeamStore

ENVELOPE_HEADER = "x-omnigent-external-identity"
MODE_ENV = "OMNIGENT_EXTERNAL_IDENTITY_MODE"
KEY_ENV = "OMNIGENT_EXTERNAL_IDENTITY_KEY"
ISSUER_ENV = "OMNIGENT_EXTERNAL_IDENTITY_ISSUER"
AUDIENCE_ENV = "OMNIGENT_EXTERNAL_IDENTITY_AUDIENCE"

_VERSION = "v1"
_MAC_LABEL = b"omnigent-external-identity/v1."
MAX_ENVELOPE_CHARS = 4096
# An envelope covers one forwarded request; a long lifetime would only widen replay.
MAX_LIFETIME_S = 300
CLOCK_SKEW_S = 30
MIN_KEY_CHARS = 32
_MAX_ISSUER_CHARS = 512
_MAX_SUBJECT_CHARS = 255
_MAX_AUDIENCE_CHARS = 255
_MAX_EMAIL_CHARS = 320

_RESOLVED_SCOPE_KEY = "omnigent.external_identity"
_MODE_SCOPE_KEY = "omnigent.external_identity_mode"

# Routes that establish or end a sign-in check the identity themselves: their
# incoming credential (an old cookie of someone else) is about to be replaced.
SELF_CHECKING_PATHS = frozenset(
    {
        "/auth/login",
        "/auth/logout",
        "/auth/register",
        "/auth/setup",
        "/auth/magic/redeem",
        "/oauth/token",
    }
)

INVALID_CODE = "external_identity_invalid"
REQUIRED_CODE = "external_identity_required"
UNBOUND_CODE = "external_identity_unbound"
MISMATCH_CODE = "external_identity_mismatch"
SUSPENDED_CODE = "member_suspended"

REFUSAL_MESSAGES: Mapping[str, str] = {
    INVALID_CODE: "The sign-in in front of Omnigent could not be verified. Reload the page.",
    REQUIRED_CODE: "Sign in through this installation's web address.",
    UNBOUND_CODE: (
        "This sign-in is not linked to a member of this team. "
        "Ask a team administrator for an invitation."
    ),
    MISMATCH_CODE: (
        "You are signed in to Omnigent as someone other than the person signed in "
        "in front of it. Sign out of Omnigent, then sign in again."
    ),
    SUSPENDED_CODE: "Your membership of this team is suspended. Ask a team administrator.",
}


class ExternalIdentityMode(str, Enum):
    """How strictly requests are tied to the proxy's identity."""

    OFF = "off"
    BIND = "bind"
    REQUIRED = "required"


@dataclasses.dataclass(frozen=True)
class ExternalIdentity:
    """The identity a trusted proxy verified for one request.

    :param issuer: The identity provider's issuer, e.g.
        ``"https://example.cloudflareaccess.com"``.
    :param subject: The provider's stable subject for the person.
    :param audience: The application the proxy admitted the person to.
    :param email: The verified address, lowercase, when the provider has one.
        Evidence for redeeming an invitation, never the binding key.
    :param installation_owner: The proxy asserts this is the installation's
        operator (for example its owner pin matched); used only to bind the
        existing owner account once.
    :param issued_at: Epoch seconds the envelope was signed.
    :param expires_at: Epoch seconds after which the envelope is refused.
    """

    issuer: str
    subject: str
    audience: str
    email: str | None
    installation_owner: bool
    issued_at: int
    expires_at: int


@dataclasses.dataclass(frozen=True)
class ResolvedExternalIdentity:
    """A verified envelope and the member its identity is bound to, if any."""

    identity: ExternalIdentity
    member: TeamMember | None


@dataclasses.dataclass(frozen=True)
class ExternalIdentityConfig:
    """The server's envelope settings.

    :param mode: How strictly requests are tied to the proxy's identity.
    :param key: The shared HMAC key, required unless the mode is ``off``.
    :param issuer: When set, envelopes for another issuer are refused.
    :param audience: When set, envelopes for another audience are refused.
    """

    mode: ExternalIdentityMode = ExternalIdentityMode.OFF
    key: bytes | None = None
    issuer: str | None = None
    audience: str | None = None

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> ExternalIdentityConfig:
        """Read the settings; a misconfiguration fails the start rather than opening up.

        :raises ValueError: For an unknown mode, or a mode other than ``off``
            without a key of at least :data:`MIN_KEY_CHARS` characters.
        """
        source = os.environ if env is None else env
        raw_mode = (source.get(MODE_ENV) or "off").strip().lower()
        try:
            mode = ExternalIdentityMode(raw_mode)
        except ValueError:
            raise ValueError(
                f"{MODE_ENV} must be off, bind or required, not {raw_mode!r}"
            ) from None
        if mode is ExternalIdentityMode.OFF:
            return cls()
        key = source.get(KEY_ENV) or ""
        if len(key) < MIN_KEY_CHARS:
            raise ValueError(f"{MODE_ENV}={mode.value} needs {KEY_ENV} of {MIN_KEY_CHARS}+ chars")
        return cls(
            mode=mode,
            key=key.encode(),
            issuer=(source.get(ISSUER_ENV) or "").strip() or None,
            audience=(source.get(AUDIENCE_ENV) or "").strip() or None,
        )


def _b64encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64decode(text: str) -> bytes | None:
    """Decode canonical unpadded base64url, or ``None``: a token has one spelling."""
    if not text or len(text) % 4 == 1 or not all(c.isalnum() or c in "-_" for c in text):
        return None
    try:
        data = base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))
    except (binascii.Error, ValueError):
        return None
    return data if _b64encode(data) == text else None


def _mac(key: bytes, payload: str) -> bytes:
    return hmac.new(key, _MAC_LABEL + payload.encode("ascii"), hashlib.sha256).digest()


def sign_envelope(identity: ExternalIdentity, key: bytes) -> str:
    """Return the envelope for *identity*; what a proxy sends in :data:`ENVELOPE_HEADER`."""
    claims: dict[str, object] = {
        "v": 1,
        "iss": identity.issuer,
        "sub": identity.subject,
        "aud": identity.audience,
        "owner": identity.installation_owner,
        "iat": identity.issued_at,
        "exp": identity.expires_at,
    }
    if identity.email is not None:
        claims["email"] = identity.email
    payload = _b64encode(json.dumps(claims, separators=(",", ":"), sort_keys=True).encode())
    return f"{_VERSION}.{payload}.{_b64encode(_mac(key, payload))}"


def _text(claims: Mapping[str, object], name: str, limit: int) -> str | None:
    value = claims.get(name)
    return value if isinstance(value, str) and 0 < len(value) <= limit else None


def _epoch(claims: Mapping[str, object], name: str) -> int | None:
    value = claims.get(name)
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def verify_envelope(
    token: str,
    key: bytes,
    *,
    now: float,
    expected_issuer: str | None = None,
    expected_audience: str | None = None,
) -> ExternalIdentity | None:
    """Return the identity in a valid envelope, or ``None``.

    The MAC is checked before anything in the payload is read.

    :param token: The header's value.
    :param key: The shared key.
    :param now: The current time, epoch seconds.
    :param expected_issuer: Refuse another issuer when set.
    :param expected_audience: Refuse another audience when set.
    """
    if len(token) > MAX_ENVELOPE_CHARS:
        return None
    parts = token.split(".")
    if len(parts) != 3 or parts[0] != _VERSION:
        return None
    _, payload, mac_text = parts
    mac = _b64decode(mac_text)
    payload_bytes = _b64decode(payload)
    if mac is None or payload_bytes is None:
        return None
    if not hmac.compare_digest(mac, _mac(key, payload)):
        return None
    try:
        claims = json.loads(payload_bytes.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None
    if not isinstance(claims, dict) or claims.get("v") != 1:
        return None
    issuer = _text(claims, "iss", _MAX_ISSUER_CHARS)
    subject = _text(claims, "sub", _MAX_SUBJECT_CHARS)
    audience = _text(claims, "aud", _MAX_AUDIENCE_CHARS)
    issued_at, expires_at = _epoch(claims, "iat"), _epoch(claims, "exp")
    owner = claims.get("owner")
    email = claims.get("email")
    if issuer is None or subject is None or audience is None or not isinstance(owner, bool):
        return None
    if email is not None and (
        not isinstance(email, str)
        or not 0 < len(email) <= _MAX_EMAIL_CHARS
        or email != email.lower()
    ):
        return None
    if issued_at is None or expires_at is None or not issued_at < expires_at:
        return None
    if expires_at - issued_at > MAX_LIFETIME_S:
        return None
    if now >= expires_at + CLOCK_SKEW_S or issued_at > now + CLOCK_SKEW_S:
        return None
    if expected_issuer is not None and issuer != expected_issuer:
        return None
    if expected_audience is not None and audience != expected_audience:
        return None
    return ExternalIdentity(
        issuer=issuer,
        subject=subject,
        audience=audience,
        email=email,
        installation_owner=owner,
        issued_at=issued_at,
        expires_at=expires_at,
    )


def resolved_external_identity(connection: HTTPConnection) -> ResolvedExternalIdentity | None:
    """The request's verified external identity, when it carried one."""
    resolved = connection.scope.get(_RESOLVED_SCOPE_KEY)
    return resolved if isinstance(resolved, ResolvedExternalIdentity) else None


def external_identity_mode(connection: HTTPConnection) -> ExternalIdentityMode:
    """The mode the request was checked under; ``off`` when the middleware is absent."""
    mode = connection.scope.get(_MODE_SCOPE_KEY)
    return mode if isinstance(mode, ExternalIdentityMode) else ExternalIdentityMode.OFF


def refusal_for_user(connection: HTTPConnection, user_id: str) -> str | None:
    """Why this request may not establish or use a sign-in as *user_id*, or ``None``.

    For the routes in :data:`SELF_CHECKING_PATHS`, which the middleware lets
    through: login, magic-link redemption and token refresh call it with the
    account they are about to sign in.

    :returns: A refusal code from :data:`REFUSAL_MESSAGES`, or ``None``.
    """
    mode = external_identity_mode(connection)
    if mode is ExternalIdentityMode.OFF:
        return None
    resolved = resolved_external_identity(connection)
    if resolved is None:
        return REQUIRED_CODE if mode is ExternalIdentityMode.REQUIRED else None
    if resolved.member is None:
        return UNBOUND_CODE
    if resolved.member.user_id != user_id:
        return MISMATCH_CODE
    if resolved.member.state != "active":
        return SUSPENDED_CODE
    return None


def decide(
    mode: ExternalIdentityMode,
    resolved: ResolvedExternalIdentity | None,
    principal: Principal | None,
    path: str,
) -> str | None:
    """The same-person rule for one request: a refusal code, or ``None`` to proceed.

    :param mode: The configured mode.
    :param resolved: The request's verified identity and its member, if any.
    :param principal: Who the request's Omnigent credential speaks for, if any.
    :param path: The route path, e.g. ``"/v1/sessions"``.
    """
    if mode is ExternalIdentityMode.OFF or path in SELF_CHECKING_PATHS:
        return None
    if resolved is None:
        if (
            mode is ExternalIdentityMode.REQUIRED
            and principal is not None
            and not principal.is_machine
        ):
            return REQUIRED_CODE
        return None
    if principal is None:
        # Anonymous: the sign-in and invitation pages, static files, public info.
        return None
    member = resolved.member
    if member is None:
        return UNBOUND_CODE
    if principal.user_id != member.user_id:
        return MISMATCH_CODE
    if member.state != "active":
        return SUSPENDED_CODE
    return None


def _route_path(scope: Scope) -> str:
    path = scope.get("path") or ""
    root_path = scope.get("root_path") or ""
    if root_path and path.startswith(root_path + "/"):
        return path[len(root_path) :]
    return path


class ExternalIdentityMiddleware:
    """Verify each request's envelope and apply the same-person rule.

    Runs inside the authentication middleware, so the request's principal is
    already known. An owner signing in through the proxy for the first time
    with their own account is bound here (provenance ``migration``).
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        config: ExternalIdentityConfig,
        team_store: SqlAlchemyTeamStore,
        auth_provider: AuthProvider | None,
    ) -> None:
        self._app = app
        self._config = config
        self._team_store = team_store
        self._auth_provider = auth_provider

    async def _principal(self, connection: HTTPConnection) -> Principal | None:
        recorded = recorded_principal(connection)
        if recorded is not None or self._auth_provider is None:
            return recorded
        return await asyncio.to_thread(self._auth_provider.principal, connection)

    async def _resolve(
        self, identity: ExternalIdentity, principal: Principal | None
    ) -> ResolvedExternalIdentity:
        member = await asyncio.to_thread(
            self._team_store.member_for_external_identity,
            identity.issuer,
            identity.subject,
            audience=identity.audience,
        )
        if (
            member is None
            and identity.installation_owner
            and principal is not None
            and principal.kind is PrincipalKind.HUMAN
            and principal.user_id
        ):
            member = await asyncio.to_thread(
                self._team_store.bind_installation_owner, principal.user_id, identity
            )
        return ResolvedExternalIdentity(identity, member)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        mode = self._config.mode
        if scope["type"] not in ("http", "websocket") or mode is ExternalIdentityMode.OFF:
            await self._app(scope, receive, send)
            return
        from omnigent.server.route_authority import _deny

        scope[_MODE_SCOPE_KEY] = mode
        connection = HTTPConnection(scope)
        principal = await self._principal(connection)
        resolved: ResolvedExternalIdentity | None = None
        raw = connection.headers.get(ENVELOPE_HEADER)
        if raw is not None:
            key = self._config.key or b""
            identity = verify_envelope(
                raw,
                key,
                now=time.time(),
                expected_issuer=self._config.issuer,
                expected_audience=self._config.audience,
            )
            if identity is None:
                await _deny(scope, send, 403, INVALID_CODE, REFUSAL_MESSAGES[INVALID_CODE])
                return
            resolved = await self._resolve(identity, principal)
            scope[_RESOLVED_SCOPE_KEY] = resolved
        code = decide(mode, resolved, principal, _route_path(scope))
        if code is not None:
            await _deny(scope, send, 403, code, REFUSAL_MESSAGES[code])
            return
        await self._app(scope, receive, send)
