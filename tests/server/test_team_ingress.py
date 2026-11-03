"""Membership behind an identity-aware proxy, end to end (T1, T2, T4).

Synthetic people: the installation's owner (the bootstrap admin), Carol (an
admin), Alice and Bob (members). Every request carries the envelope a front
proxy would sign for that person's sign-in; tests then mix credentials and
sign-ins. Each forbidden action sits next to its allowed neighbour.
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from omnigent.entities.team import AdmissionSnapshot
from omnigent.server.external_identity import (
    ENVELOPE_HEADER,
    ExternalIdentity,
    ExternalIdentityMode,
    sign_envelope,
)
from omnigent.server.principals import Principal, PrincipalKind
from omnigent.server.route_authority import rule_for
from tests.server.test_accounts import _build_accounts_app

_KEY = "cfo-test-envelope-key-0123456789abcdef"
_ISSUER = "https://team.example.cloudflareaccess.com"
_AUDIENCE = "aud-tag-1"
_OWNER_PASSWORD = "admin-pw-12345"
_COOKIE = "ap_session"


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
            expires_at=now + 290,
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
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[World]:
    monkeypatch.setenv("OMNIGENT_EXTERNAL_IDENTITY_MODE", "required")
    monkeypatch.setenv("OMNIGENT_EXTERNAL_IDENTITY_KEY", _KEY)
    admission = RecordingAdmission()
    for base in _build_accounts_app(
        tmp_path, monkeypatch, init_admin_password=_OWNER_PASSWORD, team_admission=admission
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
