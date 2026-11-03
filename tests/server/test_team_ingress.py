"""Membership behind an identity-aware proxy, end to end (T1, T2, T4).

Synthetic people: the installation's owner (the bootstrap admin), Carol (an
admin), Alice and Bob (members). Every request carries the envelope a front
proxy would sign for that person's sign-in; tests then mix credentials and
sign-ins. Each forbidden action sits next to its allowed neighbour.
"""

from __future__ import annotations

import dataclasses
import json
import secrets
import threading
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from omnigent.entities.team import AdmissionSnapshot
from omnigent.host.identity import MANAGED_HOST_TOKEN_HEADER
from omnigent.runner.identity import RUNNER_TUNNEL_TOKEN_HEADER, token_bound_runner_id
from omnigent.server.authority_changes import authority_changed
from omnigent.server.device_grant_store import hash_secret
from omnigent.server.external_identity import (
    ENVELOPE_HEADER,
    ExternalIdentity,
    ExternalIdentityMode,
    sign_envelope,
)
from omnigent.server.principals import Principal, PrincipalKind
from omnigent.server.route_authority import rule_for
from omnigent.stores.host_store import REGISTRATION_REVOKED_REFUSAL, HostStore
from tests.server.integration.test_host_tunnel_route import _make_hello
from tests.server.test_accounts import _build_accounts_app

_KEY = "cfo-test-envelope-key-0123456789abcdef"
_ISSUER = "https://team.example.cloudflareaccess.com"
_AUDIENCE = "aud-tag-1"
_OWNER_PASSWORD = "admin-pw-12345"
_COOKIE = "ap_session"
# The optional machine client (client-credentials grant), on a dedicated identity.
_MACHINE_CLIENT_ID = "svc-client"
_MACHINE_CLIENT_SECRET = "svc-client-secret-0123456789abcdef"
_SERVICE = "svc-machine"
_MACHINE_GRANT = {
    "grant_type": "client_credentials",
    "client_id": _MACHINE_CLIENT_ID,
    "client_secret": _MACHINE_CLIENT_SECRET,
}


class RecordingAdmission:
    """The front proxy's adapter: records every snapshot it is asked to apply."""

    def __init__(self) -> None:
        self.applied: list[AdmissionSnapshot] = []

    def apply(self, snapshot: AdmissionSnapshot) -> None:
        self.applied.append(snapshot)


def _envelope(subject: str, email: str | None, *, owner: bool = False, key: str = _KEY) -> str:
    now = int(time.time())
    return sign_envelope(
        ExternalIdentity(
            issuer=_ISSUER,
            subject=subject,
            audience=_AUDIENCE,
            email=email,
            installation_owner=owner,
            issued_at=now,
            expires_at=now + 55,
        ),
        key.encode(),
    )


@dataclasses.dataclass
class World:
    app: object
    admission: RecordingAdmission
    people: dict[str, TestClient]
    tokens: dict[str, str]

    def as_(self, subject: str, email: str | None, *, owner: bool = False) -> TestClient:
        """A browser whose proxy sign-in is *subject*, with no Omnigent cookie yet."""
        client = TestClient(self.app)  # type: ignore[arg-type]
        client.headers[ENVELOPE_HEADER] = _envelope(subject, email, owner=owner)
        return client

    def stolen(self, holder: str, thief: str) -> TestClient:
        """*thief*'s proxy sign-in carrying *holder*'s Omnigent session cookie."""
        client = self.as_(f"{thief}-sub", f"{thief}@example.test")
        client.cookies.set(_COOKIE, self.people[holder].cookies[_COOKIE])
        return client

    def member_id(self, name: str) -> str:
        team = self.people["owner"].get("/v1/team").json()
        return next(m["member_id"] for m in team["members"] if m["user_id"] == name)


def _invite(world: World, email: str, role: str = "member", by: str = "owner") -> str:
    response = world.people[by].post("/v1/team/invitations", json={"email": email, "role": role})
    assert response.status_code == 200, response.text
    return parse_qs(urlparse(response.json()["invite_url"]).query)["invite"][0]


def _join(world: World, name: str, role: str = "member") -> None:
    token = _invite(world, f"{name}@example.test", role)
    client = world.as_(f"{name}-sub", f"{name}@example.test")
    response = client.post(
        "/auth/register", json={"invite": token, "username": name, "password": f"{name}-pw-1234"}
    )
    assert response.status_code == 200, response.text
    world.people[name] = client
    world.tokens[name] = response.json()["token"]


@pytest.fixture
def world(
    request: pytest.FixtureRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[World]:
    """The team behind the proxy; an ``indirect`` parameter names the machine client's subject."""
    monkeypatch.setenv("OMNIGENT_EXTERNAL_IDENTITY_MODE", "required")
    monkeypatch.setenv("OMNIGENT_EXTERNAL_IDENTITY_KEY", _KEY)
    monkeypatch.setenv("OMNIGENT_DEVICE_GRANT_ENABLED", "1")
    cookie_secret = secrets.token_hex(32)
    monkeypatch.setenv("OMNIGENT_MACHINE_CLIENT_ID", _MACHINE_CLIENT_ID)
    monkeypatch.setenv(
        "OMNIGENT_MACHINE_CLIENT_SECRET_HASH",
        hash_secret(_MACHINE_CLIENT_SECRET, bytes.fromhex(cookie_secret)),
    )
    monkeypatch.setenv("OMNIGENT_MACHINE_SUB", getattr(request, "param", _SERVICE))
    admission = RecordingAdmission()
    for base in _build_accounts_app(
        tmp_path,
        monkeypatch,
        init_admin_password=_OWNER_PASSWORD,
        cookie_secret=cookie_secret,
        team_admission=admission,
    ):
        world = World(base.app, admission, {}, {})
        owner = world.as_("owner-sub", "owner@example.test", owner=True)
        login = owner.post("/auth/login", json={"username": "admin", "password": _OWNER_PASSWORD})
        assert login.status_code == 200, login.text
        world.people["owner"] = owner
        _join(world, "carol", "admin")
        _join(world, "alice")
        _join(world, "bob")
        yield world


@pytest.fixture
def moving_world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[World]:
    """A team of one moving behind the proxy: ``bind`` mode, the owner not bound yet."""
    monkeypatch.setenv("OMNIGENT_EXTERNAL_IDENTITY_MODE", "bind")
    monkeypatch.setenv("OMNIGENT_EXTERNAL_IDENTITY_KEY", _KEY)
    for base in _build_accounts_app(
        tmp_path, monkeypatch, init_admin_password=_OWNER_PASSWORD, team_admission=None
    ):
        yield World(base.app, RecordingAdmission(), {}, {})


def _code(response) -> str | None:
    body = response.json()
    error = body.get("error")
    return error.get("code") if isinstance(error, dict) else body.get("code")


# ── T1: identity ───────────────────────────────────────────────────


def test_the_owner_is_bound_on_first_entry_and_owns_the_team(world: World) -> None:
    team = world.people["owner"].get("/v1/team").json()
    assert team["me"]["role"] == "owner"
    assert team["external_identity"] == "required"
    assert {m["user_id"]: m["role"] for m in team["members"]} == {
        "admin": "owner",
        "carol": "admin",
        "alice": "member",
        "bob": "member",
    }


def test_alices_sign_in_with_bobs_cookie_fails_and_bobs_own_works(world: World) -> None:
    refused = world.stolen("bob", "alice").get("/v1/me")
    assert refused.status_code == 403 and _code(refused) == "external_identity_mismatch"
    assert "bob" not in refused.text
    assert world.people["bob"].get("/v1/me").status_code == 200


def test_alices_sign_in_with_bobs_bearer_fails_and_bobs_own_works(world: World) -> None:
    bearer = {"Authorization": f"Bearer {world.tokens['bob']}"}
    alice = world.as_("alice-sub", "alice@example.test")
    assert alice.get("/v1/me", headers=bearer).status_code == 403
    bob = world.as_("bob-sub", "bob@example.test")
    assert bob.get("/v1/me", headers=bearer).status_code == 200


def test_alices_sign_in_cannot_log_in_as_bob_but_as_herself(world: World) -> None:
    alice = world.as_("alice-sub", "alice@example.test")
    refused = alice.post("/auth/login", json={"username": "bob", "password": "bob-pw-1234"})
    assert refused.status_code == 403 and _code(refused) == "external_identity_mismatch"
    assert _COOKIE not in refused.cookies
    allowed = alice.post("/auth/login", json={"username": "alice", "password": "alice-pw-1234"})
    assert allowed.status_code == 200


def test_bobs_magic_link_does_not_sign_in_alices_browser(world: World) -> None:
    link = world.people["bob"].post("/auth/magic").json()["redeem_url"]
    path = urlparse(link).path + "?" + urlparse(link).query
    refused = world.as_("alice-sub", "alice@example.test").get(path, follow_redirects=False)
    assert refused.status_code == 302 and "magic=expired" in refused.headers["location"]
    assert _COOKIE not in refused.cookies
    link = world.people["bob"].post("/auth/magic").json()["redeem_url"]
    path = urlparse(link).path + "?" + urlparse(link).query
    allowed = world.as_("bob-sub", "bob@example.test").get(path, follow_redirects=False)
    assert allowed.status_code == 302 and "magic=expired" not in allowed.headers["location"]


def test_bobs_refresh_grant_renews_only_under_his_sign_in(world: World) -> None:
    bob = world.as_("bob-sub", "bob@example.test")
    login = bob.post(
        "/auth/login",
        json={"username": "bob", "password": "bob-pw-1234", "issue_refresh": True},
    )
    refresh = login.json()["refresh_token"]
    form = {"grant_type": "refresh_token", "refresh_token": refresh}
    refused = world.as_("alice-sub", "alice@example.test").post("/oauth/token", data=form)
    assert refused.status_code == 400 and refused.json()["error"] == "invalid_grant"
    assert bob.post("/oauth/token", data=form).status_code == 200


def test_a_websocket_handshake_follows_the_same_rule(world: World) -> None:
    with pytest.raises(WebSocketDisconnect) as closed:
        with world.stolen("bob", "alice").websocket_connect("/v1/sessions/updates"):
            pass
    assert closed.value.code == 4403
    with world.people["bob"].websocket_connect("/v1/sessions/updates"):
        pass


def test_an_oauth_callback_with_bobs_cookie_is_refused_under_alices_sign_in(world: World) -> None:
    path = "/v1/connections/mcp-demo/callback?code=c&state=s"
    refused = world.stolen("bob", "alice").get(path, follow_redirects=False)
    assert refused.status_code == 403 and _code(refused) == "external_identity_mismatch"
    allowed = world.people["bob"].get(path, follow_redirects=False)
    assert allowed.status_code != 403


def test_a_forged_or_missing_sign_in_is_refused(world: World) -> None:
    forged = TestClient(world.app)  # type: ignore[arg-type]
    forged.headers[ENVELOPE_HEADER] = _envelope(
        "bob-sub", "bob@example.test", key="another-key-of-the-same-length-000000"
    )
    forged.cookies.set(_COOKIE, world.people["bob"].cookies[_COOKIE])
    refused = forged.get("/v1/me")
    assert refused.status_code == 403 and _code(refused) == "external_identity_invalid"
    missing = TestClient(world.app)  # type: ignore[arg-type]
    missing.cookies.set(_COOKIE, world.people["bob"].cookies[_COOKIE])
    refused = missing.get("/v1/me")
    assert refused.status_code == 403 and _code(refused) == "external_identity_required"
    # Anonymous requests without a sign-in still reach public routes.
    assert TestClient(world.app).get("/v1/info").status_code == 200  # type: ignore[arg-type]


def test_an_invitation_is_single_use_and_for_its_address(world: World) -> None:
    token = _invite(world, "erin@example.test")
    dave = world.as_("dave-sub", "dave@example.test")
    body = {"invite": token, "username": "dave", "password": "dave-pw-1234"}
    mismatch = dave.post("/auth/register", json=body)
    assert mismatch.status_code == 403 and mismatch.json()["code"] == "invitation_email_mismatch"
    erin = world.as_("erin-sub", "erin@example.test")
    body = {"invite": token, "username": "erin", "password": "erin-pw-1234"}
    assert erin.post("/auth/register", json=body).status_code == 200
    replay = world.as_("frank-sub", "erin@example.test").post(
        "/auth/register", json={"invite": token, "username": "frank", "password": "frank-pw-1234"}
    )
    assert replay.status_code == 400


def test_the_binding_follows_the_subject_never_the_address(world: World) -> None:
    renamed = world.as_("alice-sub", "alice.new@example.test")
    renamed.cookies.set(_COOKIE, world.people["alice"].cookies[_COOKIE])
    assert renamed.get("/v1/me").status_code == 200
    reused = world.as_("someone-else-sub", "alice@example.test")
    reused.cookies.set(_COOKIE, world.people["alice"].cookies[_COOKIE])
    refused = reused.get("/v1/me")
    assert refused.status_code == 403 and _code(refused) == "external_identity_unbound"


def test_an_unbound_sign_in_reaches_only_the_entry_pages(world: World) -> None:
    stranger = world.as_("stranger-sub", "stranger@example.test")
    assert stranger.get("/v1/info").status_code == 200
    assert stranger.get("/v1/sessions").status_code == 401
    refused = stranger.post("/auth/login", json={"username": "alice", "password": "alice-pw-1234"})
    assert refused.status_code == 403 and _code(refused) == "external_identity_unbound"


def test_an_operator_claim_does_not_make_a_member_the_owner(world: World) -> None:
    claim = world.as_("fresh-sub", "alice@example.test", owner=True)
    claim.cookies.set(_COOKIE, world.people["alice"].cookies[_COOKIE])
    refused = claim.get("/v1/me")
    assert refused.status_code == 403 and _code(refused) == "external_identity_unbound"
    login = claim.post("/auth/login", json={"username": "alice", "password": "alice-pw-1234"})
    assert login.status_code == 403


def test_a_bound_person_cannot_redeem_another_invitation(world: World) -> None:
    token = _invite(world, "alice@example.test")
    refused = world.people["alice"].post(
        "/auth/register", json={"invite": token, "username": "alice2", "password": "alice-pw-1234"}
    )
    assert refused.status_code == 409 and refused.json()["code"] == "external_identity_bound"


def test_untargeted_invites_are_closed_behind_the_proxy(world: World) -> None:
    refused = world.people["owner"].post("/auth/invite", json={})
    assert refused.status_code == 409 and refused.json()["code"] == "team_invitations_required"


# ── T2: membership ─────────────────────────────────────────────────


def _wait_for(predicate, timeout_s: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


def test_suspension_stops_every_credential_and_reinstatement_needs_a_new_sign_in(
    world: World,
) -> None:
    bob = world.people["bob"]
    login = bob.post(
        "/auth/login", json={"username": "bob", "password": "bob-pw-1234", "issue_refresh": True}
    )
    refresh = {"grant_type": "refresh_token", "refresh_token": login.json()["refresh_token"]}
    bob_id = world.member_id("bob")

    suspended = world.people["owner"].post(f"/v1/team/members/{bob_id}/suspend")
    assert suspended.status_code == 200 and suspended.json()["member"]["state"] == "suspended"

    assert bob.get("/v1/me").status_code == 401
    assert bob.post("/oauth/token", data=refresh).status_code == 400
    again = bob.post("/auth/login", json={"username": "bob", "password": "bob-pw-1234"})
    assert again.status_code == 403 and again.json()["code"] == "member_suspended"
    # Alice is untouched.
    assert world.people["alice"].get("/v1/me").status_code == 200

    reinstated = world.people["owner"].post(f"/v1/team/members/{bob_id}/reinstate")
    assert reinstated.json()["member"]["state"] == "active"
    # Tokens from before the suspension stay dead; a fresh sign-in works.
    assert bob.post("/oauth/token", data=refresh).status_code == 400
    fresh = bob.post("/auth/login", json={"username": "bob", "password": "bob-pw-1234"})
    assert fresh.status_code == 200
    assert bob.get("/v1/me").status_code == 200


def test_suspension_closes_the_members_open_stream(world: World) -> None:
    bob_id = world.member_id("bob")
    with world.people["bob"].websocket_connect("/v1/sessions/updates") as stream:
        world.people["owner"].post(f"/v1/team/members/{bob_id}/suspend")
        with pytest.raises(WebSocketDisconnect):
            for _ in range(50):
                stream.receive_text()
    with world.people["alice"].websocket_connect("/v1/sessions/updates"):
        pass


def test_the_proxy_is_told_who_to_admit_after_each_change(world: World) -> None:
    assert _wait_for(
        lambda: (
            world.admission.applied and "bob@example.test" in world.admission.applied[-1].emails
        )
    )
    world.people["owner"].post(f"/v1/team/members/{world.member_id('bob')}/suspend")
    assert _wait_for(lambda: "bob@example.test" not in world.admission.applied[-1].emails)
    latest = world.admission.applied[-1]
    assert {"owner@example.test", "carol@example.test", "alice@example.test"} <= set(latest.emails)
    revisions = [snapshot.revision for snapshot in world.admission.applied]
    assert revisions == sorted(revisions)


# ── T4: the human admin ────────────────────────────────────────────


def test_an_admin_suspends_members_but_not_owners_or_herself(world: World) -> None:
    carol = world.people["carol"]
    assert carol.post(f"/v1/team/members/{world.member_id('admin')}/suspend").status_code == 403
    assert carol.post(f"/v1/team/members/{world.member_id('carol')}/suspend").status_code == 403
    assert carol.post(f"/v1/team/members/{world.member_id('bob')}/suspend").status_code == 200


def test_an_admin_invites_members_and_only_an_owner_invites_admins(world: World) -> None:
    carol = world.people["carol"]
    refused = carol.post("/v1/team/invitations", json={"email": "x@example.test", "role": "admin"})
    assert refused.status_code == 403
    allowed = carol.post(
        "/v1/team/invitations", json={"email": "x@example.test", "role": "member"}
    )
    assert allowed.status_code == 200
    owner = world.people["owner"]
    assert (
        owner.post(
            "/v1/team/invitations", json={"email": "y@example.test", "role": "admin"}
        ).status_code
        == 200
    )


def test_only_an_owner_changes_roles(world: World) -> None:
    alice_id = world.member_id("alice")
    body = {"role": "admin"}
    assert (
        world.people["carol"].put(f"/v1/team/members/{alice_id}/role", json=body).status_code
        == 403
    )
    changed = world.people["owner"].put(f"/v1/team/members/{alice_id}/role", json=body)
    assert changed.status_code == 200 and changed.json()["member"]["role"] == "admin"


def test_an_admins_cli_sign_in_cannot_administer_but_her_browser_can(world: World) -> None:
    cli = world.as_("carol-sub", "carol@example.test")
    bearer = {"Authorization": f"Bearer {world.tokens['carol']}"}
    bob_id = world.member_id("bob")
    refused = cli.post(f"/v1/team/members/{bob_id}/suspend", headers=bearer)
    assert refused.status_code == 403
    assert world.people["carol"].post(f"/v1/team/members/{bob_id}/suspend").status_code == 200


def test_a_member_cannot_read_the_team_page_and_an_admin_can(world: World) -> None:
    assert world.people["alice"].get("/v1/team").status_code == 403
    assert world.people["carol"].get("/v1/team").status_code == 200


def test_an_admin_reset_password_is_useless_to_the_admin_and_works_for_the_member(
    world: World,
) -> None:
    reset = world.people["carol"].post("/auth/users/bob/reset")
    assert reset.status_code == 200
    password = reset.json()["new_password"]
    as_carol = world.as_("carol-sub", "carol@example.test")
    refused = as_carol.post("/auth/login", json={"username": "bob", "password": password})
    assert refused.status_code == 403
    as_bob = world.as_("bob-sub", "bob@example.test")
    assert (
        as_bob.post("/auth/login", json={"username": "bob", "password": password}).status_code
        == 200
    )


def test_runners_never_reach_the_team_page() -> None:
    runner = Principal(PrincipalKind.RUNNER, "carol", scopes=frozenset({"session"}))
    host = Principal(PrincipalKind.HOST, "carol")
    person = Principal(PrincipalKind.HUMAN, "carol")
    for method, path in (
        ("GET", "/v1/team"),
        ("POST", "/v1/team/invitations"),
        ("PUT", "/v1/team/members/{member_id}/role"),
        ("POST", "/v1/team/members/{member_id}/suspend"),
        ("POST", "/v1/team/members/{member_id}/reinstate"),
    ):
        rule = rule_for(method, path)
        assert rule is not None
        assert not rule.admits(runner) and not rule.admits(host)
        assert rule.admits(person)


# ── A team of one on today's defaults ──────────────────────────────


def test_a_team_of_one_on_the_defaults_behaves_as_before(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OMNIGENT_EXTERNAL_IDENTITY_MODE", raising=False)
    for client in _build_accounts_app(tmp_path, monkeypatch, init_admin_password=_OWNER_PASSWORD):
        assert client.app.state.external_identity_config.mode is ExternalIdentityMode.OFF
        # No sign-in in front, and a header nobody verifies is ignored.
        client.headers[ENVELOPE_HEADER] = "v1.garbage.garbage"
        login = client.post("/auth/login", json={"username": "admin", "password": _OWNER_PASSWORD})
        assert login.status_code == 200
        assert client.get("/v1/me").status_code == 200
        legacy = client.post("/auth/invite", json={})
        assert legacy.status_code == 200 and "register_url" in legacy.json()
        assert client.get("/v1/team").json()["me"]["role"] == "owner"


# ── R1: credentials resolved inside routes (F-1) and suspension (F-2) ──


def _stores(world: World) -> dict[str, Any]:
    from omnigent.server.accounts_store import SqlAlchemyAccountStore
    from omnigent.stores.conversation_store.sqlalchemy_store import SqlAlchemyConversationStore

    app_state = world.app.state  # type: ignore[attr-defined]
    url = app_state.team_store.storage_location
    return {
        "hosts": app_state.host_store,
        "teams": app_state.team_store,
        "conversations": SqlAlchemyConversationStore(url),
        "accounts": SqlAlchemyAccountStore(url),
    }


def _managed_host(world: World, owner: str) -> tuple[str, str]:
    host_id, token = uuid.uuid4().hex, secrets.token_urlsafe(32)
    _stores(world)["hosts"].register_managed_host(
        host_id=host_id,
        name=f"managed-{host_id[:8]}",
        user_id=owner,
        token=token,
        provider="test",
        sandbox_id=f"sandbox-{host_id[:8]}",
        token_expires_at=int(time.time()) + 3600,
    )
    return host_id, token


def _bound_runner(world: World, owner: str) -> str:
    """Bind a runner to a new thread of *owner* on their own host; return the binding token."""
    from tests.server.test_account_revocation import create_session

    created = create_session(world.people[owner])
    assert created.status_code == 201, created.text
    session_id = created.json()["session_id"]
    stores = _stores(world)
    binding, host_id = f"binding-{uuid.uuid4().hex}", uuid.uuid4().hex
    stores["hosts"].upsert_on_connect(host_id, f"host-{host_id[:8]}", owner)
    stores["conversations"].set_host_id(session_id, host_id, workspace="/tmp/workspace")
    stores["conversations"].replace_runner_id(session_id, token_bound_runner_id(binding))
    return binding


def _opens(client: TestClient, path: str, headers: dict[str, str]) -> bool:
    try:
        with client.websocket_connect(path, headers=headers):
            return True
    except WebSocketDisconnect:
        return False


def test_bobs_device_code_is_not_redeemed_under_alices_sign_in(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    from omnigent.server.routes import device_auth

    monkeypatch.setattr(device_auth, "_POLL_INTERVAL_SECONDS", 0)
    started = TestClient(world.app).post("/oauth/device/authorize", json={"client_id": "cli"})  # type: ignore[arg-type]
    device_code, user_code = started.json()["device_code"], started.json()["user_code"]
    bob = world.people["bob"]
    time.sleep(1.1)  # approval needs a sign-in no older than the grant
    assert (
        bob.post("/auth/login", json={"username": "bob", "password": "bob-pw-1234"}).status_code
        == 200
    )
    approved = bob.post(
        "/oauth/device/approve",
        data={"user_code": user_code},
        headers={"Origin": "http://localhost:8000"},
    )
    assert approved.status_code == 200, approved.text
    form = {
        "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
        "device_code": device_code,
    }
    refused = world.as_("alice-sub", "alice@example.test").post("/oauth/token", data=form)
    assert refused.status_code == 400 and refused.json()["error"] == "access_denied"
    allowed = world.as_("bob-sub", "bob@example.test").post("/oauth/token", data=form)
    assert allowed.status_code == 200 and allowed.json()["access_token"]


def test_a_launch_token_opens_the_host_tunnel_only_for_its_owners_sign_in(world: World) -> None:
    host_id, token = _managed_host(world, "bob")
    path, headers = f"/v1/hosts/{host_id}/tunnel", {MANAGED_HOST_TOKEN_HEADER: token}
    assert not _opens(world.as_("alice-sub", "alice@example.test"), path, headers)
    assert _opens(world.as_("bob-sub", "bob@example.test"), path, headers)
    # A cloud computer's private route carries no sign-in, and keeps working.
    assert _opens(TestClient(world.app), path, headers)  # type: ignore[arg-type]


def test_a_runner_binding_mints_and_tunnels_only_for_its_owners_sign_in(world: World) -> None:
    binding = _bound_runner(world, "bob")
    runner_id = token_bound_runner_id(binding)
    mint, headers = f"/v1/runners/{runner_id}/token", {RUNNER_TUNNEL_TOKEN_HEADER: binding}
    assert (
        world.as_("alice-sub", "alice@example.test").post(mint, headers=headers).status_code == 401
    )
    assert world.as_("bob-sub", "bob@example.test").post(mint, headers=headers).status_code == 200
    assert TestClient(world.app).post(mint, headers=headers).status_code == 200  # type: ignore[arg-type]
    tunnel = f"/v1/runners/{runner_id}/tunnel"
    assert not _opens(world.as_("alice-sub", "alice@example.test"), tunnel, headers)
    assert _opens(TestClient(world.app), tunnel, headers)  # type: ignore[arg-type]


def test_suspension_ends_machine_credentials_and_reinstatement_needs_new_ones(
    world: World,
) -> None:
    host_id, token = _managed_host(world, "bob")
    binding = _bound_runner(world, "bob")
    private = TestClient(world.app)  # type: ignore[arg-type]
    host_path, host_headers = f"/v1/hosts/{host_id}/tunnel", {MANAGED_HOST_TOKEN_HEADER: token}
    mint = f"/v1/runners/{token_bound_runner_id(binding)}/token"
    runner_headers = {RUNNER_TUNNEL_TOKEN_HEADER: binding}
    assert _opens(private, host_path, host_headers)
    assert private.post(mint, headers=runner_headers).status_code == 200

    bob_id = world.member_id("bob")
    world.people["owner"].post(f"/v1/team/members/{bob_id}/suspend")
    assert not _opens(private, host_path, host_headers)
    assert private.post(mint, headers=runner_headers).status_code == 401

    world.people["owner"].post(f"/v1/team/members/{bob_id}/reinstate")
    # Pre-suspension machine credentials stay dead; the records stay.
    assert not _opens(private, host_path, host_headers)
    assert private.post(mint, headers=runner_headers).status_code == 401
    assert _stores(world)["hosts"].get_host(host_id) is not None
    # Fresh machine authority works: Bob signs in again and his new runner mints.
    bob = world.people["bob"]
    assert (
        bob.post("/auth/login", json={"username": "bob", "password": "bob-pw-1234"}).status_code
        == 200
    )
    fresh = _bound_runner(world, "bob")
    fresh_mint = f"/v1/runners/{token_bound_runner_id(fresh)}/token"
    assert private.post(fresh_mint, headers={RUNNER_TUNNEL_TOKEN_HEADER: fresh}).status_code == 200


# ── R2: an open personal-host tunnel and the client-credentials grant ──


def test_an_open_host_tunnel_stays_ended_after_suspension_and_reinstatement(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bob's laptop tunnel, opened with his CLI sign-in, does not outlive his suspension.

    The tunnel's checks are held until the owner has suspended and reinstated
    Bob through the team page, so they run only after the reinstatement, when
    Bob is active again: the tunnel still ends, and only a fresh sign-in
    connects his laptop again.
    """
    real_heartbeat = HostStore.heartbeat
    release = threading.Event()
    checks: list[bool] = []

    def held_heartbeat(self: HostStore, host_id: str, **kwargs: Any) -> bool:
        release.wait(timeout=10)
        kept = real_heartbeat(self, host_id, **kwargs)
        checks.append(kept)
        return kept

    monkeypatch.setattr(HostStore, "heartbeat", held_heartbeat)
    registry = world.app.state.host_registry  # type: ignore[attr-defined]
    host_id = uuid.uuid4().hex
    path = f"/v1/hosts/{host_id}/tunnel"
    laptop = world.as_("bob-sub", "bob@example.test")
    old_sign_in = {"Authorization": f"Bearer {world.tokens['bob']}"}
    bob_id = world.member_id("bob")
    owner = world.people["owner"]

    with laptop.websocket_connect(path, headers=old_sign_in) as tunnel:
        tunnel.send_text(_make_hello("bob-laptop"))
        assert _wait_for(lambda: registry.get(host_id) is not None)
        opened = registry.get(host_id)
        assert owner.post(f"/v1/team/members/{bob_id}/suspend").status_code == 200
        assert owner.post(f"/v1/team/members/{bob_id}/reinstate").status_code == 200
        release.set()
        # Woken by the change, the check ran after the reinstatement and ended the tunnel.
        assert _wait_for(lambda: registry.get(host_id) is not opened)
        assert checks[0] is False
        received: list[str] = []
        with pytest.raises(WebSocketDisconnect) as closed:
            for _ in range(20):
                received.append(tunnel.receive_text())
        assert closed.value.code == 4009
        assert any(REGISTRATION_REVOKED_REFUSAL in json.dumps(json.loads(m)) for m in received)

    # The old sign-in cannot open it again; a fresh one can, and the tunnel stays.
    assert not _opens(laptop, path, old_sign_in)
    fresh = laptop.post("/auth/login", json={"username": "bob", "password": "bob-pw-1234"})
    assert fresh.status_code == 200
    with laptop.websocket_connect(
        path, headers={"Authorization": f"Bearer {fresh.json()['token']}"}
    ) as tunnel:
        tunnel.send_text(_make_hello("bob-laptop"))
        assert _wait_for(lambda: registry.get(host_id) is not None)
        reconnected = registry.get(host_id)
        done = len(checks)
        authority_changed()
        assert _wait_for(lambda: len(checks) > done)
        assert checks[-1] is True and registry.get(host_id) is reconnected


def test_the_client_credentials_grant_follows_the_same_person_rule(world: World) -> None:
    # A machine's own request carries no sign-in: its dedicated identity gets a token.
    issued = TestClient(world.app).post("/oauth/token", data=_MACHINE_GRANT)  # type: ignore[arg-type]
    assert issued.status_code == 200 and issued.json()["access_token"]
    # Under a person's sign-in it gets none: another person, an unbound one, a suspended one.
    bob_id = world.member_id("bob")
    assert world.people["owner"].post(f"/v1/team/members/{bob_id}/suspend").status_code == 200
    for signed_in in (
        world.as_("alice-sub", "alice@example.test"),
        world.as_("stranger-sub", "stranger@example.test"),
        world.as_("bob-sub", "bob@example.test"),
    ):
        refused = signed_in.post("/oauth/token", data=_MACHINE_GRANT)
        assert refused.status_code == 400 and refused.json()["error"] == "access_denied"
        assert "access_token" not in refused.json()


@pytest.mark.parametrize("world", ["bob"], indirect=True)
def test_a_client_credentials_subject_that_is_a_member_gets_no_token(world: World) -> None:
    """A machine client configured as Bob never acts as Bob, with or without his sign-in."""

    def refused(client: TestClient) -> bool:
        answer = client.post("/oauth/token", data=_MACHINE_GRANT)
        return answer.status_code == 400 and answer.json()["error"] == "unauthorized_client"

    assert refused(TestClient(world.app))  # type: ignore[arg-type]
    assert refused(world.as_("bob-sub", "bob@example.test"))
    bob_id = world.member_id("bob")
    assert world.people["owner"].post(f"/v1/team/members/{bob_id}/suspend").status_code == 200
    assert refused(TestClient(world.app))  # type: ignore[arg-type]
    assert refused(world.as_("bob-sub", "bob@example.test"))


# ── 8.11: moving a team of one behind the proxy (bind, then required) ──


def _team_status(world: World):
    from omnigent.server.team_status import team_status

    state = world.app.state  # type: ignore[attr-defined]
    return team_status(state.team_store, state.host_registry)


def _laptop_tunnel(client: TestClient, bearer: str, host_id: str):
    """Open *host_id*'s tunnel with a CLI sign-in and say hello; the caller closes it."""
    tunnel = client.websocket_connect(
        f"/v1/hosts/{host_id}/tunnel", headers={"Authorization": f"Bearer {bearer}"}
    )
    socket = tunnel.__enter__()
    socket.send_text(_make_hello("owner-laptop"))
    return tunnel


def test_bind_keeps_a_device_whose_sign_in_differs_working_and_reports_it(
    moving_world: World,
) -> None:
    """The owner's laptop connects first, under a device sign-in unlike the browser's."""
    world = moving_world
    registry = world.app.state.host_registry  # type: ignore[attr-defined]
    device = world.as_("device-sub", "owner@example.test", owner=True)
    # The laptop's CLI sign-in (refresh material asked): it works, and binds nothing.
    cli = device.post(
        "/auth/login",
        json={"username": "admin", "password": _OWNER_PASSWORD, "issue_refresh": True},
    )
    assert cli.status_code == 200, cli.text
    device.cookies.clear()  # a CLI keeps the token it was given, not the browser's cookie
    assert _team_status(world).owners_bound == 0
    host_id = uuid.uuid4().hex
    tunnel = _laptop_tunnel(device, cli.json()["token"], host_id)
    try:
        assert _wait_for(lambda: registry.get(host_id) is not None)
        [host] = _team_status(world).hosts
        assert (host.host_id, host.identity) == (host_id, "unbound")

        # The owner's browser signs in: the founding binding is the browser's identity.
        browser = world.as_("browser-sub", "owner@example.test", owner=True)
        signed_in = browser.post(
            "/auth/login", json={"username": "admin", "password": _OWNER_PASSWORD}
        )
        assert signed_in.status_code == 200, signed_in.text
        status = _team_status(world)
        assert (status.owners, status.owners_bound) == (1, 1)
        # The laptop keeps working, and is reported as unbound: the move stops at bind.
        assert registry.get(host_id) is not None
        assert [h.identity for h in status.hosts] == ["unbound"]
        assert (
            device.get(
                "/v1/me", headers={"Authorization": f"Bearer {cli.json()['token']}"}
            ).status_code
            == 200
        )
    finally:
        tunnel.__exit__(None, None, None)


def test_bind_reports_a_laptop_under_the_owners_own_sign_in_as_the_owners(
    moving_world: World,
) -> None:
    world = moving_world
    registry = world.app.state.host_registry  # type: ignore[attr-defined]
    browser = world.as_("owner-sub", "owner@example.test", owner=True)
    login = browser.post("/auth/login", json={"username": "admin", "password": _OWNER_PASSWORD})
    assert login.status_code == 200, login.text
    host_id = uuid.uuid4().hex
    tunnel = _laptop_tunnel(browser, login.json()["token"], host_id)
    try:
        assert _wait_for(lambda: registry.get(host_id) is not None)
        assert [(h.host_id, h.identity) for h in _team_status(world).hosts] == [(host_id, "owner")]
    finally:
        tunnel.__exit__(None, None, None)
    # A cloud computer's private route carries no sign-in at all.
    managed_id, token = _managed_host(world, "admin")
    with TestClient(world.app).websocket_connect(  # type: ignore[arg-type]
        f"/v1/hosts/{managed_id}/tunnel", headers={MANAGED_HOST_TOKEN_HEADER: token}
    ) as cloud:
        cloud.send_text(_make_hello("cloud"))
        assert _wait_for(lambda: registry.get(managed_id) is not None)
        assert [h.identity for h in _team_status(world).hosts if h.host_id == managed_id] == [
            "none"
        ]


def test_bind_never_founds_from_a_cli_and_still_refuses_another_members_sign_in(
    moving_world: World,
) -> None:
    world = moving_world
    device = world.as_("device-sub", "owner@example.test", owner=True)
    for _ in range(2):
        cli = device.post(
            "/auth/login",
            json={"username": "admin", "password": _OWNER_PASSWORD, "issue_refresh": True},
        )
        assert cli.status_code == 200
        device.cookies.clear()  # a CLI keeps the token it was given, not the browser's cookie
    bearer = {"Authorization": f"Bearer {cli.json()['token']}"}
    assert device.get("/v1/me", headers=bearer).status_code == 200
    assert _team_status(world).owners_bound == 0
    browser = world.as_("owner-sub", "owner@example.test", owner=True)
    assert (
        browser.post(
            "/auth/login", json={"username": "admin", "password": _OWNER_PASSWORD}
        ).status_code
        == 200
    )
    # A sign-in bound to the owner carrying another account's credential is still refused.
    _stores(world)["accounts"].create_user_with_password("mallory", "unused")
    from omnigent.server.passwords import hash_password

    _stores(world)["accounts"].update_password("mallory", hash_password("mallory-pw-1234"))
    mallory = TestClient(world.app).post(  # type: ignore[arg-type]
        "/auth/login", json={"username": "mallory", "password": "mallory-pw-1234"}
    )
    assert mallory.status_code == 200
    borrowed = world.as_("owner-sub", "owner@example.test")
    refused = borrowed.get(
        "/v1/me", headers={"Authorization": f"Bearer {mallory.json()['token']}"}
    )
    assert refused.status_code == 403 and _code(refused) == "external_identity_mismatch"


# ── R1: the founding binding is one-time (F-3) ──


def test_the_founding_binding_is_one_time(world: World) -> None:
    # The owner bound on first entry; a new subject with the pinned address
    # and the owner's own cookie is no longer bound by the pin.
    recycled = world.as_("recycled-sub", "owner@example.test", owner=True)
    recycled.cookies.set(_COOKIE, world.people["owner"].cookies[_COOKIE])
    refused = recycled.get("/v1/me")
    assert refused.status_code == 403 and _code(refused) == "external_identity_unbound"
    login = recycled.post("/auth/login", json={"username": "admin", "password": _OWNER_PASSWORD})
    assert login.status_code == 403
    # A second owner gets no pin binding either.
    carol_id = world.member_id("carol")
    world.people["owner"].put(f"/v1/team/members/{carol_id}/role", json={"role": "owner"})
    claim = world.as_("carol-new-sub", "owner@example.test", owner=True)
    claim.cookies.set(_COOKIE, world.people["carol"].cookies[_COOKIE])
    assert claim.get("/v1/me").status_code == 403
    # The founder's own binding still works.
    assert world.people["owner"].get("/v1/me").status_code == 200


# ── R1: authorization inside the write (F-4) ──


def test_a_role_promoted_while_a_request_was_checked_is_seen_by_the_write(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    teams = _stores(world)["teams"]
    bob_id = world.member_id("bob")
    original = teams.get_member

    def promoted_meanwhile(member_id: str):
        stale = original(member_id)
        if member_id == bob_id:
            teams.change_role(bob_id, "admin")  # an owner's promotion commits first
        return stale

    monkeypatch.setattr(teams, "get_member", promoted_meanwhile)
    refused = world.people["carol"].post(f"/v1/team/members/{bob_id}/suspend")
    assert refused.status_code == 403
    monkeypatch.setattr(teams, "get_member", original)
    assert teams.get_member(bob_id).state == "active"
    # The owner may still suspend Bob.
    assert world.people["owner"].post(f"/v1/team/members/{bob_id}/suspend").status_code == 200


# ── R1: existing-account invitations (F-6) ──


def test_a_link_invitation_binds_an_existing_account_proved_by_its_password(
    world: World,
) -> None:
    from omnigent.server.passwords import hash_password

    _stores(world)["accounts"].create_user_with_password("legacy", hash_password("legacy-pw-1234"))
    invited = world.people["owner"].post(
        "/v1/team/invitations",
        json={"email": "legacy@example.test", "role": "member", "user_id": "legacy"},
    )
    assert invited.status_code == 200, invited.text
    token = parse_qs(urlparse(invited.json()["invite_url"]).query)["invite"][0]
    legacy = world.as_("legacy-sub", "legacy@example.test")
    preview = legacy.get(f"/v1/team/invitations/preview?token={token}").json()
    assert preview["links_account"] is True and preview["user_id"] == "legacy"
    wrong = legacy.post(
        "/auth/login", json={"username": "legacy", "password": "wrong-password", "invite": token}
    )
    assert wrong.status_code == 401
    right = legacy.post(
        "/auth/login", json={"username": "legacy", "password": "legacy-pw-1234", "invite": token}
    )
    assert right.status_code == 200, right.text
    assert legacy.get("/v1/me").status_code == 200
    binding = _stores(world)["teams"].get_external_identity(
        "https://team.example.cloudflareaccess.com", "legacy-sub"
    )
    assert binding is not None and binding.provenance == "account_proof"
