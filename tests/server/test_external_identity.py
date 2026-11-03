"""The front proxy's signed identity: the envelope, its settings and the same-person rule."""

from __future__ import annotations

import base64
import json
import time

import pytest

from omnigent.entities.team import TeamMember
from omnigent.server.external_identity import (
    INVALID_CODE,
    MISMATCH_CODE,
    REQUIRED_CODE,
    SUSPENDED_CODE,
    UNBOUND_CODE,
    ExternalIdentity,
    ExternalIdentityConfig,
    ExternalIdentityMode,
    ResolvedExternalIdentity,
    decide,
    sign_envelope,
    verify_envelope,
)
from omnigent.server.principals import CredentialKind, Principal, PrincipalKind

_KEY = b"cfo-test-envelope-key-0123456789abcdef"
# The same vector is checked by the platform's signer (contracts fixture
# `ExternalIdentityEnvelope/alice.json`), so both sides agree byte for byte.
_VECTOR_IDENTITY = ExternalIdentity(
    issuer="https://example-team.cloudflareaccess.com",
    subject="7335d417-61da-459d-899c-0a01c76a2f94",
    audience="32eafc7626e974616deaf0dc3ce63d7bcbed58a2731e84d06bc3cdf1b53c4228",
    email="alice@example.test",
    installation_owner=False,
    issued_at=1800000000,
    expires_at=1800000060,
)
_VECTOR = (
    "v1.eyJhdWQiOiIzMmVhZmM3NjI2ZTk3NDYxNmRlYWYwZGMzY2U2M2Q3YmNiZWQ1OGEyNzMxZTg0ZDA2YmMzY2Rm"
    "MWI1M2M0MjI4IiwiZW1haWwiOiJhbGljZUBleGFtcGxlLnRlc3QiLCJleHAiOjE4MDAwMDAwNjAsImlhdCI6MTgw"
    "MDAwMDAwMCwiaXNzIjoiaHR0cHM6Ly9leGFtcGxlLXRlYW0uY2xvdWRmbGFyZWFjY2Vzcy5jb20iLCJvd25lciI6"
    "ZmFsc2UsInN1YiI6IjczMzVkNDE3LTYxZGEtNDU5ZC04OTljLTBhMDFjNzZhMmY5NCIsInYiOjF9"
    ".kvKB20Zx20Fk5ptJO4wGFT04BEWmRXmPCVsgaFVY3I8"
)


def _identity(**changes: object) -> ExternalIdentity:
    now = int(time.time())
    values: dict[str, object] = {
        "issuer": "https://team.example",
        "subject": "alice-sub",
        "audience": "aud",
        "email": "alice@example.test",
        "installation_owner": False,
        "issued_at": now,
        "expires_at": now + 60,
    }
    values.update(changes)
    return ExternalIdentity(**values)  # type: ignore[arg-type]


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


# ── The envelope ───────────────────────────────────────────────────


def test_the_shared_vector_signs_and_verifies() -> None:
    assert sign_envelope(_VECTOR_IDENTITY, _KEY) == _VECTOR
    assert verify_envelope(_VECTOR, _KEY, now=1800000030) == _VECTOR_IDENTITY


def test_a_valid_envelope_verifies_and_every_forgery_fails() -> None:
    identity = _identity()
    token = sign_envelope(identity, _KEY)
    assert verify_envelope(token, _KEY, now=time.time()) == identity
    version, payload, mac = token.split(".")
    claims = json.loads(base64.urlsafe_b64decode(payload + "=="))
    claims["sub"] = "bob-sub"
    forged_payload = _b64(json.dumps(claims, separators=(",", ":"), sort_keys=True).encode())
    for forged in (
        f"{version}.{forged_payload}.{mac}",  # another person, the old MAC
        sign_envelope(identity, b"another-key-of-the-same-length-000000"),
        f"v2.{payload}.{mac}",
        f"{version}.{payload}.{mac}x",
        f"{version}.{payload}",
        token + "." + mac,
        "x" * 5000,
    ):
        assert verify_envelope(forged, _KEY, now=time.time()) is None, forged


def test_an_expired_early_or_long_lived_envelope_fails() -> None:
    now = int(time.time())
    assert verify_envelope(sign_envelope(_identity(), _KEY), _KEY, now=now + 30) is not None
    expired = _identity(issued_at=now - 70, expires_at=now - 15)
    assert verify_envelope(sign_envelope(expired, _KEY), _KEY, now=now) is None
    early = _identity(issued_at=now + 20, expires_at=now + 60)
    assert verify_envelope(sign_envelope(early, _KEY), _KEY, now=now) is None
    # The platform signs for at most 60 seconds; anything longer is refused.
    for lifetime in (61, 300, 3600):
        long_lived = _identity(issued_at=now, expires_at=now + lifetime)
        assert verify_envelope(sign_envelope(long_lived, _KEY), _KEY, now=now) is None
    assert verify_envelope(sign_envelope(_identity(expires_at=now + 60), _KEY), _KEY, now=now)


def test_the_expected_issuer_and_audience_are_enforced_when_set() -> None:
    token = sign_envelope(_identity(), _KEY)
    now = time.time()
    assert verify_envelope(token, _KEY, now=now, expected_issuer="https://team.example")
    assert verify_envelope(token, _KEY, now=now, expected_issuer="https://other.example") is None
    assert verify_envelope(token, _KEY, now=now, expected_audience="aud")
    assert verify_envelope(token, _KEY, now=now, expected_audience="other-aud") is None


def test_an_uppercase_or_missing_email_is_handled_strictly() -> None:
    now = time.time()
    assert verify_envelope(sign_envelope(_identity(email=None), _KEY), _KEY, now=now)
    assert (
        verify_envelope(sign_envelope(_identity(email="Alice@X.test"), _KEY), _KEY, now=now)
        is None
    )


# ── Settings ───────────────────────────────────────────────────────


def test_settings_default_off_and_refuse_a_mode_without_a_key() -> None:
    assert ExternalIdentityConfig.from_env({}).mode is ExternalIdentityMode.OFF
    with pytest.raises(ValueError):
        ExternalIdentityConfig.from_env({"OMNIGENT_EXTERNAL_IDENTITY_MODE": "sometimes"})
    with pytest.raises(ValueError):
        ExternalIdentityConfig.from_env({"OMNIGENT_EXTERNAL_IDENTITY_MODE": "bind"})
    with pytest.raises(ValueError):
        ExternalIdentityConfig.from_env(
            {
                "OMNIGENT_EXTERNAL_IDENTITY_MODE": "required",
                "OMNIGENT_EXTERNAL_IDENTITY_KEY": "short",
            }
        )
    config = ExternalIdentityConfig.from_env(
        {
            "OMNIGENT_EXTERNAL_IDENTITY_MODE": "Required",
            "OMNIGENT_EXTERNAL_IDENTITY_KEY": _KEY.decode(),
            "OMNIGENT_EXTERNAL_IDENTITY_AUDIENCE": "aud",
        }
    )
    assert (config.mode, config.key, config.audience, config.issuer) == (
        ExternalIdentityMode.REQUIRED,
        _KEY,
        "aud",
        None,
    )


# ── The same-person rule ───────────────────────────────────────────


def _member(user_id: str, state: str = "active") -> TeamMember:
    return TeamMember(
        id=f"{user_id}-member",
        user_id=user_id,
        role="member",
        state=state,  # type: ignore[arg-type]
        created_at=1,
        updated_at=1,
    )


def _person(user_id: str, kind: PrincipalKind = PrincipalKind.HUMAN) -> Principal:
    return Principal(kind, user_id, credential=CredentialKind.BROWSER_SESSION)


def _resolved(member: TeamMember | None) -> ResolvedExternalIdentity:
    return ResolvedExternalIdentity(_identity(), member)


_REQUIRED = ExternalIdentityMode.REQUIRED
_BIND = ExternalIdentityMode.BIND


@pytest.mark.parametrize(
    ("mode", "member", "principal", "path", "expected"),
    [
        # Alice's sign-in with Bob's credential fails; with her own it passes.
        (_REQUIRED, _member("alice"), _person("bob"), "/v1/me", MISMATCH_CODE),
        (_REQUIRED, _member("alice"), _person("alice"), "/v1/me", None),
        # Bob's runner or host token under Alice's sign-in (a laptop) fails.
        (
            _REQUIRED,
            _member("alice"),
            _person("bob", PrincipalKind.RUNNER),
            "/v1/me",
            MISMATCH_CODE,
        ),
        (_REQUIRED, _member("alice"), _person("alice", PrincipalKind.HOST), "/v1/me", None),
        # A suspended member's credential fails; an active one passes.
        (_REQUIRED, _member("bob", "suspended"), _person("bob"), "/v1/me", SUSPENDED_CODE),
        # An unbound sign-in holding someone's credential fails; anonymous passes.
        (_REQUIRED, None, _person("alice"), "/v1/me", UNBOUND_CODE),
        (_REQUIRED, None, None, "/v1/info", None),
        # Sign-in routes check the person themselves.
        (_REQUIRED, _member("alice"), _person("bob"), "/auth/login", None),
        (_REQUIRED, _member("alice"), _person("bob"), "/auth/logout", None),
        # Bind, the move to the proxy: an unbound sign-in (a device's) proceeds; a bound one
        # with someone else's credential, or a suspended member's, is refused as in required.
        (_BIND, None, _person("alice"), "/v1/me", None),
        (_BIND, None, _person("alice", PrincipalKind.HOST), "/v1/me", None),
        (_BIND, _member("alice"), _person("bob"), "/v1/me", MISMATCH_CODE),
        (_BIND, _member("bob", "suspended"), _person("bob"), "/v1/me", SUSPENDED_CODE),
        (_BIND, _member("alice"), _person("alice"), "/v1/me", None),
    ],
)
def test_the_same_person_rule_with_an_envelope(mode, member, principal, path, expected) -> None:
    assert decide(mode, _resolved(member), principal, path) == expected


@pytest.mark.parametrize(
    ("mode", "principal", "expected"),
    [
        # Required: a person's credential needs the proxy's sign-in.
        (_REQUIRED, _person("alice"), REQUIRED_CODE),
        (_REQUIRED, _person("alice", PrincipalKind.RUNNER), None),
        (_REQUIRED, _person("alice", PrincipalKind.HOST), None),
        (_REQUIRED, None, None),
        # Bind: a request without the proxy's sign-in is unchanged.
        (_BIND, _person("alice"), None),
        # Off: nothing is checked.
        (ExternalIdentityMode.OFF, _person("alice"), None),
    ],
)
def test_the_same_person_rule_without_an_envelope(mode, principal, expected) -> None:
    assert decide(mode, None, principal, "/v1/me") == expected


def test_an_unbound_sign_in_is_refused_only_in_required() -> None:
    from omnigent.server.external_identity import unbound_refusal

    assert unbound_refusal(ExternalIdentityMode.BIND) is None
    assert unbound_refusal(ExternalIdentityMode.REQUIRED) == UNBOUND_CODE


def test_the_founding_binding_needs_a_browsers_sign_in() -> None:
    from omnigent.server.external_identity import _FOUNDING_CREDENTIALS

    assert {CredentialKind.BROWSER_SESSION} == _FOUNDING_CREDENTIALS
    for credential in (
        CredentialKind.SESSION_COOKIE,
        CredentialKind.SESSION_BEARER,
        CredentialKind.LOGIN_GRANT,
        CredentialKind.DEVICE_GRANT,
        CredentialKind.CLIENT_CREDENTIALS,
        CredentialKind.MACHINE,
    ):
        assert credential not in _FOUNDING_CREDENTIALS


def test_the_invalid_code_has_a_message() -> None:
    from omnigent.server.external_identity import REFUSAL_MESSAGES

    assert {INVALID_CODE, MISMATCH_CODE, REQUIRED_CODE, SUSPENDED_CODE, UNBOUND_CODE} <= set(
        REFUSAL_MESSAGES
    )
