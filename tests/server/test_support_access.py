"""Support access: a team's owners and admins act on members' computers and connections.

The team behind the proxy (``computer_team`` from the team ingress tests): ``admin``
is the owner, carol an admin, alice and bob members. The fake provider records the
owner Omnigent names on each call. With ``create_app(admin_support_access=True)``,
carol reads, starts, deletes and creates alice's computers, runs the setup actions
on alice's hosts and sees or removes alice's connections, always for alice; logins,
files, runners and new threads stay alice's. A member, or a CLI, device, runner or
host credential, gets none of it, and with the option off nothing changes.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from omnigent.connections.github import GithubConnectionStore
from omnigent.entities.team import TeamMember
from omnigent.host.frames import CAP_REFRESH_HARNESSES, HostHelloFrame
from omnigent.host.identity import MANAGED_HOST_TOKEN_HEADER
from omnigent.runner.identity import RUNNER_TUNNEL_TOKEN_HEADER, token_bound_runner_id
from omnigent.server import persistent_hosts
from omnigent.server.github_app import GitHubTokenSet
from omnigent.server.mcp_registry import McpRegistry, McpRegistryConfig, McpService
from omnigent.server.principals import CredentialKind, Principal, PrincipalKind
from omnigent.server.routes import accounts_auth
from omnigent.server.routes import hosts as hosts_routes
from omnigent.server.routes._host_launch import ManageGrant, may_manage_host
from omnigent.server.support_access import SupportedMember
from omnigent.stores.credential_store.sqlalchemy_store import CredentialStore
from omnigent.stores.host_store import Host
from tests.server.persistent_host_fakes import OwnedComputer
from tests.server.test_account_revocation import create_session
from tests.server.test_credential_store import _FakeCipher
from tests.server.test_team_ingress import (
    World,
    _bound_runner,
    _cloud_computer,
    _join,
    _managed_host,
    _stores,
    computer_team,
)

pytestmark = [pytest.mark.timeout(90)]

_SERVICE = "tracker"


def _db_url(tmp_path: Path) -> str:
    # The accounts app's database (``_build_accounts_app``).
    return f"sqlite:///{tmp_path}/test.db"


def _team(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, support: bool) -> Iterator[World]:
    monkeypatch.setenv("OMNIGENT_FEATURES", "harness_install")
    monkeypatch.setenv("OMNIGENT_DEVICE_GRANT_ENABLED", "1")
    # A wake's status polls shortened, as the sandbox owner tests shorten them.
    monkeypatch.setattr(persistent_hosts, "_STATUS_POLL_S", 0.05)
    monkeypatch.setattr(persistent_hosts, "_UNKNOWN_RETRY_S", 0.05)
    registry = McpRegistry(
        McpRegistryConfig(
            services=[
                McpService.model_validate(
                    {
                        "id": _SERVICE,
                        "title": "Tracker",
                        "url": "https://mcp.example.test/mcp",
                        "tools": ["read_ticket"],
                    }
                )
            ]
        ),
        CredentialStore(_db_url(tmp_path), _FakeCipher()),
    )
    computer = OwnedComputer(status="stopped")
    try:
        for world in computer_team(
            tmp_path,
            monkeypatch,
            computer,
            online_timeout_s=3,
            admin_support_access=support,
            mcp_registry=registry,
        ):
            world.app.state.github_store = GithubConnectionStore(  # type: ignore[attr-defined]
                _db_url(tmp_path), _FakeCipher()
            )
            yield world
    finally:
        for state in persistent_hosts._states.values():
            if state.wake_task is not None:
                state.wake_task.cancel()
        persistent_hosts._states.clear()


@pytest.fixture
def support_world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[World]:
    """The team with support access on."""
    yield from _team(tmp_path, monkeypatch, support=True)


@pytest.fixture
def plain_world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[World]:
    """The same team on Omnigent's defaults: support access off."""
    yield from _team(tmp_path, monkeypatch, support=False)


def _computer(world: World) -> OwnedComputer:
    return world.app.state.fake_computer  # type: ignore[attr-defined,no-any-return]


def _laptop(world: World, owner: str) -> str:
    """A connected (non-cloud) host of *owner*, live on this replica; return its id."""
    host_id = uuid.uuid4().hex
    _stores(world)["hosts"].upsert_on_connect(host_id, f"laptop-{host_id[:8]}", owner)
    world.app.state.host_registry.register(  # type: ignore[attr-defined]
        host_id,
        AsyncMock(),
        HostHelloFrame(
            version="test",
            frame_protocol_version=1,
            name="laptop",
            capabilities=[CAP_REFRESH_HARNESSES],
        ),
        owner=owner,
    )
    return host_id


@pytest.fixture
def answered(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    """Answer the host's setup frames; record ``(request, host_id)`` for each one sent."""
    sent: list[tuple[str, str]] = []

    def answer(name: str, result: dict[str, Any]) -> Callable[..., Any]:
        async def proxy(**kwargs: Any) -> dict[str, Any]:
            sent.append((name, kwargs["host_conn"].host_id))
            return result

        return proxy

    ok = {"status": "ok", "configured_harnesses": {}, "gateway_inference": None}
    monkeypatch.setattr(hosts_routes, "_proxy_install_harness", answer("install", ok))
    monkeypatch.setattr(hosts_routes, "_proxy_refresh_harnesses", answer("refresh", ok))
    monkeypatch.setattr(
        hosts_routes,
        "_proxy_detect_credentials",
        answer("detect", {"credentials": [{"family": "anthropic", "source": "env"}]}),
    )
    monkeypatch.setattr(
        hosts_routes,
        "_proxy_model_options",
        answer("models", {"status": "ok", "models": [{"id": "m"}], "routable_models": ["m"]}),
    )
    return sent


def _setup_actions(host_id: str) -> list[tuple[str, str]]:
    """The setup actions an owner or admin may take on a host: ``(method, path)``."""
    base = f"/v1/hosts/{host_id}"
    return [
        ("POST", f"{base}/harnesses/claude/install"),
        ("POST", f"{base}/harnesses/refresh"),
        ("GET", f"{base}/harnesses/claude/model-options"),
        ("GET", f"{base}/credentials/detected"),
    ]


def _wait(predicate: Callable[[], bool], timeout_s: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


def _member(world: World, name: str) -> TeamMember:
    member = _stores(world)["teams"].get_member_for_account(name)
    assert member is not None
    return member


# ── Computers ────────────────────────────────────────────────────


def test_an_admin_reads_starts_and_deletes_a_members_computer_for_the_member(
    support_world: World,
) -> None:
    world = support_world
    # On the app's loop, so the wake the start begins runs on as it does in a server.
    carol = world.on_app_loop(world.people["carol"])
    computer, alice = _computer(world), _member(world, "alice")
    host_id = _cloud_computer(world, "alice")

    read = carol.get(f"/v1/hosts/{host_id}")
    assert read.status_code == 200, read.text
    assert read.json()["owner"] == "alice"

    started = carol.post(f"/v1/hosts/{host_id}/start")
    assert started.status_code == 202, started.text
    assert _wait(lambda: any(call == "start_host" for call, _ in computer.owners))
    # The provider is told the computer is alice's at every call, never carol's.
    assert computer.owners
    for _call, owner in computer.owners:
        assert owner is not None
        assert (owner.user_id, owner.member_id, owner.host_id) == ("alice", alice.id, host_id)

    deleted = carol.delete(f"/v1/hosts/{host_id}")
    assert deleted.status_code == 200, deleted.text
    assert _wait(lambda: "sb-alice" in computer.terminated)
    terminate = [owner for call, owner in computer.owners if call == "terminate"]
    assert terminate and all(owner is not None and owner.user_id == "alice" for owner in terminate)


def test_an_admin_runs_the_setup_actions_on_a_members_host(
    support_world: World, answered: list[tuple[str, str]]
) -> None:
    world, carol = support_world, support_world.people["carol"]
    host_id = _laptop(world, "alice")
    for method, path in _setup_actions(host_id):
        response = carol.request(method, path)
        assert response.status_code == 200, (path, response.text)
    assert answered == [
        ("install", host_id),
        ("refresh", host_id),
        ("models", host_id),
        ("detect", host_id),
    ]
    host = _stores(world)["hosts"].get_host(host_id)
    assert host is not None and host.user_id == "alice"


def test_an_admin_adds_a_computer_for_an_active_member_and_it_is_theirs(
    support_world: World,
) -> None:
    world = support_world
    carol = world.on_app_loop(world.people["carol"])
    computer, alice = _computer(world), _member(world, "alice")
    created = carol.post("/v1/hosts", json={"name": "For alice", "owner_user_id": "alice"})
    assert created.status_code == 201, created.text
    assert created.json()["owner"] == "alice"
    host = _stores(world)["hosts"].get_host(created.json()["host_id"])
    assert host is not None and host.user_id == "alice"
    assert _wait(lambda: any(call == "provision" for call, _ in computer.owners))
    provision = next(owner for call, owner in computer.owners if call == "provision")
    assert provision is not None and (provision.user_id, provision.member_id) == (
        "alice",
        alice.id,
    )
    # Not for a suspended member, someone outside the team, or by a member.
    owner = world.people["owner"]
    assert owner.post(f"/v1/team/members/{_member(world, 'bob').id}/suspend").status_code == 200
    for name in ("bob", "nobody"):
        refused = carol.post("/v1/hosts", json={"name": "For them", "owner_user_id": name})
        assert refused.status_code == 404, refused.text
    by_member = world.people["bob"].post(
        "/v1/hosts", json={"name": "For alice", "owner_user_id": "alice"}
    )
    assert by_member.status_code in (401, 403)


def test_an_admin_looks_after_a_suspended_members_computer(support_world: World) -> None:
    world, carol = support_world, support_world.people["carol"]
    host_id = _cloud_computer(world, "alice")
    alice = _member(world, "alice")
    assert world.people["owner"].post(f"/v1/team/members/{alice.id}/suspend").status_code == 200
    assert carol.get(f"/v1/hosts/{host_id}").status_code == 200
    listed = carol.get(f"/v1/team/members/{alice.id}/hosts")
    assert [host["host_id"] for host in listed.json()["hosts"]] == [host_id]
    assert carol.delete(f"/v1/hosts/{host_id}").status_code == 200


def test_logins_files_runners_and_new_threads_stay_the_members(
    support_world: World, answered: list[tuple[str, str]]
) -> None:
    world, carol = support_world, support_world.people["carol"]
    host_id = _laptop(world, "alice")
    created = create_session(carol)
    assert created.status_code == 201, created.text
    agent_id, session_id = created.json()["agent_id"], created.json()["session_id"]
    refusals = [
        carol.post(
            f"/v1/hosts/{host_id}/harnesses/claude/credential",
            json={"kind": "key", "secret": "sk-test-not-real"},
        ),
        carol.post(
            f"/v1/hosts/{host_id}/runners",
            json={"session_id": session_id, "workspace": "/tmp/workspace"},
        ),
        carol.post(
            "/v1/sessions",
            json={"agent_id": agent_id, "host_id": host_id, "workspace": "/tmp/workspace"},
        ),
        carol.get(f"/v1/hosts/{host_id}/filesystem"),
        carol.get(f"/v1/hosts/{host_id}/worktrees", params={"path": "/tmp"}),
    ]
    for refused in refusals:
        assert refused.status_code == 403, (refused.request.url, refused.text)
    assert answered == []
    # Nothing connects for alice: the support routes offer no connect at all.
    alice_id = _member(world, "alice").id
    base = f"/v1/team/members/{alice_id}"
    assert carol.get(f"{base}/connections/github/connect").status_code in (404, 405)
    connect = carol.put(f"{base}/mcp-services/{_SERVICE}/connection", json={"token": "t"})
    assert connect.status_code in (404, 405)


def test_a_member_gets_none_of_it_on_another_members_things(
    support_world: World, answered: list[tuple[str, str]]
) -> None:
    world, bob = support_world, support_world.people["bob"]
    computer = _computer(world)
    cloud, laptop = _cloud_computer(world, "alice"), _laptop(world, "alice")
    calls = [
        ("GET", f"/v1/hosts/{cloud}"),
        ("POST", f"/v1/hosts/{cloud}/start"),
        ("DELETE", f"/v1/hosts/{cloud}"),
        *_setup_actions(laptop),
        *_support_routes(_member(world, "alice").id),
    ]
    for method, path in calls:
        response = bob.request(method, path)
        assert response.status_code == 403, (method, path, response.text)
    assert answered == [] and computer.owners == []
    assert _stores(world)["hosts"].get_host(cloud) is not None


# ── The member's page ────────────────────────────────────────────


def _support_routes(member_id: str) -> list[tuple[str, str]]:
    base = f"/v1/team/members/{member_id}"
    return [
        ("GET", f"{base}/hosts"),
        ("GET", f"{base}/connections"),
        ("DELETE", f"{base}/connections/github"),
        ("DELETE", f"{base}/mcp-services/{_SERVICE}/connection"),
        ("GET", f"{base}/threads"),
    ]


def _connect(world: World, name: str) -> None:
    """Give *name* a GitHub connection and a login to the MCP service."""
    world.app.state.github_store.upsert(  # type: ignore[attr-defined]
        name,
        github_login=f"{name}-gh",
        github_user_id=len(name),
        tokens=GitHubTokenSet(
            access_token=f"gh-{name}",
            refresh_token=None,
            expires_at=None,
            refresh_token_expires_at=None,
            scopes="",
        ),
    )
    registry: McpRegistry = world.app.state.mcp_registry  # type: ignore[attr-defined]
    entry = registry.services[_SERVICE]
    registry.save_login(entry, name, f"mcp:{_SERVICE}", {"access_token": f"mcp-{name}"}, {})


def _connected(world: World, name: str) -> tuple[bool, bool]:
    registry: McpRegistry = world.app.state.mcp_registry  # type: ignore[attr-defined]
    github = world.app.state.github_store.get(name) is not None  # type: ignore[attr-defined]
    assert registry.store is not None
    return github, registry.store.get(name, f"mcp:{_SERVICE}") is not None


def test_an_admin_sees_and_removes_only_the_members_connections(support_world: World) -> None:
    world, carol = support_world, support_world.people["carol"]
    for name in ("alice", "bob", "carol"):
        _connect(world, name)
    alice_id = _member(world, "alice").id

    status = carol.get(f"/v1/team/members/{alice_id}/connections")
    assert status.status_code == 200, status.text
    assert status.json()["providers"] == [
        {
            "provider": "github",
            "connected": True,
            "connected_at": pytest.approx(time.time(), abs=60),
        }
    ]
    (service,) = status.json()["services"]
    assert (service["id"], service["connected"], service["disconnectable"]) == (
        _SERVICE,
        True,
        True,
    )
    assert "gh-alice" not in status.text and "mcp-alice" not in status.text

    github = carol.delete(f"/v1/team/members/{alice_id}/connections/github")
    assert github.status_code == 200 and github.json() == {"disconnected": True}
    mcp = carol.delete(f"/v1/team/members/{alice_id}/mcp-services/{_SERVICE}/connection")
    assert mcp.status_code == 200, mcp.text
    assert _connected(world, "alice") == (False, False)
    assert _connected(world, "bob") == (True, True)
    assert _connected(world, "carol") == (True, True)
    after = carol.get(f"/v1/team/members/{alice_id}/connections").json()
    assert after["providers"][0]["connected"] is False
    assert after["services"][0]["connected"] is False


def test_an_admin_lists_a_members_hosts_and_threads(support_world: World) -> None:
    world, carol = support_world, support_world.people["carol"]
    cloud, laptop = _cloud_computer(world, "alice"), _laptop(world, "alice")
    _cloud_computer(world, "bob")
    thread = create_session(world.people["alice"])
    assert thread.status_code == 201, thread.text
    alice_id = _member(world, "alice").id

    hosts = carol.get(f"/v1/team/members/{alice_id}/hosts")
    assert hosts.status_code == 200, hosts.text
    assert {host["host_id"] for host in hosts.json()["hosts"]} == {cloud, laptop}
    assert {host["owner"] for host in hosts.json()["hosts"]} == {"alice"}

    threads = carol.get(f"/v1/team/members/{alice_id}/threads")
    assert threads.status_code == 200, threads.text
    assert [t["session_id"] for t in threads.json()["threads"]] == [thread.json()["session_id"]]
    assert carol.get(f"/v1/sessions/{thread.json()['session_id']}").status_code == 200

    # A removed member's threads come from their removal's record.
    bob_thread = create_session(world.people["bob"])
    bob_id = _member(world, "bob").id
    assert world.people["owner"].post(f"/v1/team/members/{bob_id}/remove").status_code == 200
    removed = carol.get(f"/v1/team/members/{bob_id}/threads")
    assert [t["session_id"] for t in removed.json()["threads"]] == [
        bob_thread.json()["session_id"]
    ]
    assert carol.get(f"/v1/team/members/{bob_id}/hosts").status_code == 404
    assert carol.get("/v1/team/members/no-such-member/threads").status_code == 404


def test_a_members_threads_are_listed_to_nobody_but_owners_and_admins(
    support_world: World,
) -> None:
    world = support_world
    alice_id = _member(world, "alice").id
    assert create_session(world.people["alice"]).status_code == 201
    path = f"/v1/team/members/{alice_id}/threads"
    bob_cli = world.as_("bob-sub", "bob@example.test")
    for response in (
        world.people["bob"].get(path),
        bob_cli.get(path, headers={"Authorization": f"Bearer {world.tokens['bob']}"}),
        # Not even alice herself: the list is the support page's.
        world.people["alice"].get(path),
    ):
        assert response.status_code == 403, response.text
        assert "threads" not in response.json()
    assert world.people["carol"].get(path).status_code == 200


# ── Credentials ──────────────────────────────────────────────────


def _device_token(world: World, name: str, monkeypatch: pytest.MonkeyPatch) -> str:
    """An access token from a device grant *name* approved in their browser."""
    from omnigent.server.routes import device_auth

    monkeypatch.setattr(device_auth, "_POLL_INTERVAL_SECONDS", 0)
    started = TestClient(world.app).post("/oauth/device/authorize", json={"client_id": "cli"})  # type: ignore[arg-type]
    device_code, user_code = started.json()["device_code"], started.json()["user_code"]
    browser = world.people[name]
    time.sleep(1.1)  # approval needs a sign-in no older than the grant
    login = browser.post("/auth/login", json={"username": name, "password": f"{name}-pw-1234"})
    assert login.status_code == 200, login.text
    approved = browser.post(
        "/oauth/device/approve",
        data={"user_code": user_code},
        headers={"Origin": "http://localhost:8000"},
    )
    assert approved.status_code == 200, approved.text
    token = world.as_(f"{name}-sub", f"{name}@example.test").post(
        "/oauth/token",
        data={
            "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            "device_code": device_code,
        },
    )
    assert token.status_code == 200, token.text
    return str(token.json()["access_token"])


def _runner_token(world: World, name: str) -> str:
    binding = _bound_runner(world, name)
    minted = TestClient(world.app).post(  # type: ignore[arg-type]
        f"/v1/runners/{token_bound_runner_id(binding)}/token",
        headers={RUNNER_TUNNEL_TOKEN_HEADER: binding},
    )
    assert minted.status_code == 200, minted.text
    return str(minted.json()["token"])


def test_cli_device_runner_and_host_credentials_never_get_support_access(
    support_world: World, answered: list[tuple[str, str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    world = support_world
    for name in ("alice",):
        _connect(world, name)
    computer = _computer(world)
    cloud, laptop = _cloud_computer(world, "alice"), _laptop(world, "alice")
    _host_id, host_token = _managed_host(world, "carol")
    proxied = world.as_("carol-sub", "carol@example.test")
    machine = TestClient(world.app)  # type: ignore[arg-type]
    credentials = {
        "cli": (proxied, {"Authorization": f"Bearer {world.tokens['carol']}"}),
        "device": (
            proxied,
            {"Authorization": f"Bearer {_device_token(world, 'carol', monkeypatch)}"},
        ),
        "runner": (machine, {"Authorization": f"Bearer {_runner_token(world, 'carol')}"}),
        "host": (machine, {MANAGED_HOST_TOKEN_HEADER: host_token}),
    }
    calls = [
        ("GET", f"/v1/hosts/{cloud}", None),
        ("POST", f"/v1/hosts/{cloud}/start", None),
        ("DELETE", f"/v1/hosts/{cloud}", None),
        ("POST", "/v1/hosts", {"name": "For alice", "owner_user_id": "alice"}),
        *((method, path, None) for method, path in _setup_actions(laptop)),
        *((method, path, None) for method, path in _support_routes(_member(world, "alice").id)),
    ]
    for kind, (client, headers) in credentials.items():
        for method, path, body in calls:
            response = client.request(method, path, headers=headers, json=body)
            assert response.status_code in (401, 403), (kind, method, path, response.text)
    assert answered == [] and computer.owners == []
    assert _stores(world)["hosts"].get_host(cloud) is not None
    assert _connected(world, "alice") == (True, True)
    assert [h.host_id for h in _stores(world)["hosts"].list_hosts("alice")].count(cloud) == 1


def test_support_access_needs_a_browser_sign_in_of_an_active_owner_or_admin(
    support_world: World,
) -> None:
    support = support_world.app.state.support_access  # type: ignore[attr-defined]
    browser = CredentialKind.BROWSER_SESSION
    assert support.supporter(Principal(PrincipalKind.HUMAN, "carol", credential=browser), "carol")
    assert support.supporter(Principal(PrincipalKind.HUMAN, "admin", credential=browser), "admin")
    refused = [
        (Principal(PrincipalKind.HUMAN, "alice", credential=browser), "alice"),
        (
            Principal(PrincipalKind.HUMAN, "carol", credential=CredentialKind.SESSION_BEARER),
            "carol",
        ),
        (Principal(PrincipalKind.HUMAN, "carol", credential=CredentialKind.LOGIN_GRANT), "carol"),
        (
            Principal(
                PrincipalKind.DELEGATED_CLIENT, "carol", credential=CredentialKind.DEVICE_GRANT
            ),
            "carol",
        ),
        (Principal(PrincipalKind.RUNNER, "carol", credential=CredentialKind.MACHINE), "carol"),
        (Principal(PrincipalKind.HOST, "carol", credential=CredentialKind.MACHINE), "carol"),
        (Principal(PrincipalKind.HUMAN, "carol", credential=browser), "alice"),
        (None, "carol"),
    ]
    for principal, user_id in refused:
        assert support.supporter(principal, user_id) is None, principal


def test_may_manage_host_allows_the_owner_or_support_for_someone_else() -> None:
    carol = TeamMember("m-carol", "carol", "admin", "active", 0, 0)
    alice = TeamMember("m-alice", "alice", "member", "active", 0, 0)
    as_carol = SupportedMember(carol, "gen-carol")
    as_alice = SupportedMember(alice, "gen-alice")

    class Support:
        def __init__(self, actor: TeamMember | None, member: SupportedMember | None) -> None:
            self.actor, self.member = actor, member

        def supporter(self, principal: object, user_id: str | None) -> TeamMember | None:
            return self.actor

        def member_for_host(self, host: Host) -> SupportedMember | None:
            return self.member

    host = Host("h1", "laptop", "alice", "online", 0, 0)
    principal = Principal(PrincipalKind.HUMAN, "carol", credential=CredentialKind.BROWSER_SESSION)

    def decide(user_id: str | None, support: Any) -> ManageGrant | None:
        return may_manage_host(principal, host, user_id=user_id, support=support)

    assert decide(None, None) == ManageGrant()  # auth off: one local user
    assert decide("alice", None) == ManageGrant()
    assert decide("alice", Support(None, None)) == ManageGrant()
    assert decide("carol", None) is None
    assert decide("carol", Support(None, as_alice)) is None
    assert decide("carol", Support(carol, None)) is None
    assert decide("carol", Support(carol, as_carol)) is None
    grant = decide("carol", Support(carol, as_alice))
    assert grant == ManageGrant(actor=carol, member=as_alice)
    # The grant carries what a write checks again: who, for whom, which registration.
    action = grant.action()
    assert action is not None
    assert (action.actor_member_id, action.member_id, action.member_generation) == (
        "m-carol",
        "m-alice",
        "gen-alice",
    )
    assert ManageGrant().action() is None


# ── Reset, the option off, and the log ───────────────────────────


def test_reset_follows_the_suspend_rule(plain_world: World) -> None:
    world = plain_world
    _join(world, "dave", "admin")
    carol, owner = world.people["carol"], world.people["owner"]
    for target in ("admin", "dave"):
        refused = carol.post(f"/auth/users/{target}/reset")
        assert refused.status_code == 403, refused.text
        assert "new_password" not in refused.text
    assert carol.post("/auth/users/alice/reset").status_code == 200
    assert carol.post("/auth/users/carol/reset").status_code == 200
    assert owner.post("/auth/users/dave/reset").status_code == 200
    assert owner.post("/auth/users/carol/reset").status_code == 200


def test_with_the_option_off_hosts_and_connections_stay_their_owners(
    plain_world: World, answered: list[tuple[str, str]]
) -> None:
    world, carol = plain_world, plain_world.people["carol"]
    computer = _computer(world)
    cloud, laptop = _cloud_computer(world, "alice"), _laptop(world, "alice")
    assert carol.get("/v1/team").json()["support_access"] is False
    for method, path in [
        ("GET", f"/v1/hosts/{cloud}"),
        ("POST", f"/v1/hosts/{cloud}/start"),
        ("DELETE", f"/v1/hosts/{cloud}"),
        *_setup_actions(laptop),
    ]:
        assert carol.request(method, path).status_code == 403, path
    refused = carol.post("/v1/hosts", json={"name": "For alice", "owner_user_id": "alice"})
    assert refused.status_code == 403
    for method, path in _support_routes(_member(world, "alice").id):
        assert carol.request(method, path).status_code == 404, path
    assert answered == [] and computer.owners == []
    # The owner's own use is unchanged.
    alice = world.people["alice"]
    assert alice.get(f"/v1/hosts/{cloud}").status_code == 200
    created = alice.post("/v1/hosts", json={"name": "Mine", "owner_user_id": "alice"})
    assert created.status_code == 201 and created.json()["owner"] == "alice"


def test_support_actions_are_logged_with_member_ids_only(
    support_world: World, caplog: pytest.LogCaptureFixture
) -> None:
    world, carol = support_world, support_world.people["carol"]
    host_id = _cloud_computer(world, "alice")
    _connect(world, "alice")
    carol_id, alice_id = _member(world, "carol").id, _member(world, "alice").id
    with caplog.at_level(logging.INFO, logger="omnigent.server.support_access"):
        assert carol.get(f"/v1/hosts/{host_id}").status_code == 200
        assert carol.get(f"/v1/team/members/{alice_id}/connections").status_code == 200
        assert carol.delete(f"/v1/team/members/{alice_id}/connections/github").status_code == 200
        assert carol.get("/v1/team").json()["support_access"] is True
        # The owner's own reads are not support actions.
        assert world.people["alice"].get(f"/v1/hosts/{host_id}").status_code == 200
    lines = [r.getMessage() for r in caplog.records if r.name == "omnigent.server.support_access"]
    assert lines == [
        f"support action=read_host actor_member={carol_id} member={alice_id} subject={host_id}",
        f"support action=list_connections actor_member={carol_id} member={alice_id}",
        f"support action=disconnect actor_member={carol_id} member={alice_id} subject=github",
    ]
    for line in lines:
        assert "alice" not in line and "carol" not in line and "@" not in line


# ── A change between the check and the write is seen by the write ─


def _demote(world: World, name: str) -> None:
    _stores(world)["teams"].change_role(_member(world, name).id, "member")


def _suspend(world: World, name: str) -> None:
    _stores(world)["accounts"].change_member_state(_member(world, name).id, "suspended")


@pytest.mark.parametrize("change", ["target_to_admin", "target_to_owner", "actor_demoted"])
def test_a_role_change_before_a_resets_write_is_seen_by_it(
    plain_world: World, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    world, carol = plain_world, plain_world.people["carol"]
    teams, accounts = _stores(world)["teams"], _stores(world)["accounts"]
    alice_id = _member(world, "alice").id
    before = accounts.get_password_hash("alice")
    original = accounts_auth.hash_password

    def changed_meanwhile(password: str) -> str:
        # After the route's checks, before the reset's write.
        if change == "actor_demoted":
            _demote(world, "carol")
        else:
            teams.change_role(alice_id, "admin" if change == "target_to_admin" else "owner")
        return original(password)

    monkeypatch.setattr(accounts_auth, "hash_password", changed_meanwhile)
    refused = carol.post("/auth/users/alice/reset")
    assert refused.status_code == 403, refused.text
    assert "new_password" not in refused.text
    assert accounts.get_password_hash("alice") == before


def _support_store(world: World) -> Any:
    return world.app.state.support_access.account_store  # type: ignore[attr-defined]


def _pending_mcp_login(world: World, name: str) -> None:
    registry: McpRegistry = world.app.state.mcp_registry  # type: ignore[attr-defined]
    entry = registry.services[_SERVICE]
    registry.save_login(entry, name, f"mcp-pending:{_SERVICE}", {"state": "s"}, {})


def _mcp_rows(world: World, name: str) -> tuple[bool, bool]:
    registry: McpRegistry = world.app.state.mcp_registry  # type: ignore[attr-defined]
    assert registry.store is not None
    return (
        registry.store.get(name, f"mcp:{_SERVICE}") is not None,
        registry.store.get(name, f"mcp-pending:{_SERVICE}") is not None,
    )


def test_a_disconnect_removes_the_login_and_its_pending_sign_in_and_nothing_else(
    support_world: World,
) -> None:
    world, carol = support_world, support_world.people["carol"]
    for name in ("alice", "bob"):
        _connect(world, name)
        _pending_mcp_login(world, name)
    alice_id = _member(world, "alice").id
    removed = carol.delete(f"/v1/team/members/{alice_id}/mcp-services/{_SERVICE}/connection")
    assert removed.status_code == 200, removed.text
    assert _mcp_rows(world, "alice") == (False, False)
    # Her GitHub connection, and Bob's login and sign-in, are untouched.
    assert _connected(world, "alice") == (True, False)
    assert _mcp_rows(world, "bob") == (True, True)


@pytest.mark.parametrize("change", ["actor_demoted", "actor_suspended"])
@pytest.mark.parametrize("connection", ["github", "mcp"])
def test_a_disconnect_checks_its_actor_again_in_its_write(
    support_world: World, monkeypatch: pytest.MonkeyPatch, change: str, connection: str
) -> None:
    world, carol = support_world, support_world.people["carol"]
    _connect(world, "alice")
    store = _support_store(world)
    original = store.remove_member_connections

    def changed_meanwhile(*args: Any, **kwargs: Any) -> bool:
        (_demote if change == "actor_demoted" else _suspend)(world, "carol")
        return original(*args, **kwargs)

    monkeypatch.setattr(store, "remove_member_connections", changed_meanwhile)
    alice_id = _member(world, "alice").id
    path = (
        f"/v1/team/members/{alice_id}/connections/github"
        if connection == "github"
        else f"/v1/team/members/{alice_id}/mcp-services/{_SERVICE}/connection"
    )
    refused = carol.delete(path)
    # A demoted admin is refused; a suspended one's own sign-in has ended.
    assert refused.status_code == (403 if change == "actor_demoted" else 401), refused.text
    assert _connected(world, "alice") == (True, True)


def test_a_disconnect_never_reaches_a_new_account_with_the_members_old_username(
    support_world: World, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    world, carol = support_world, support_world.people["carol"]
    accounts = _stores(world)["accounts"]
    _connect(world, "alice")
    alice_id, owner_id = _member(world, "alice").id, _member(world, "admin").id
    store = _support_store(world)
    original = store.remove_member_connections

    def replaced_meanwhile(*args: Any, **kwargs: Any) -> bool:
        # Alice is removed and someone else registers her username, with connections.
        accounts.remove_member(alice_id, actor_member_id=owner_id)
        accounts.create_user_with_password("alice", "another-persons-hash")
        _connect(world, "alice")
        return original(*args, **kwargs)

    monkeypatch.setattr(store, "remove_member_connections", replaced_meanwhile)
    with caplog.at_level(logging.INFO, logger="omnigent.server.support_access"):
        refused = carol.delete(f"/v1/team/members/{alice_id}/connections/github")
    assert refused.status_code in (404, 409), refused.text
    # The new account's connections survive, and no support action is logged for it.
    assert _connected(world, "alice") == (True, True)
    assert not [r for r in caplog.records if "action=disconnect" in r.getMessage()]


@pytest.mark.parametrize("change", ["actor_demoted", "target_suspended"])
def test_adding_a_computer_for_someone_is_checked_again_in_its_write(
    support_world: World, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    world, carol = support_world, support_world.people["carol"]
    computer, hosts = _computer(world), _stores(world)["hosts"]
    original = hosts.create_persistent_host

    def changed_meanwhile(**kwargs: Any) -> Host:
        # After the route's checks, before the row's write.
        if change == "actor_demoted":
            _demote(world, "carol")
        else:
            _suspend(world, "alice")
        return original(**kwargs)

    monkeypatch.setattr(hosts, "create_persistent_host", changed_meanwhile)
    refused = carol.post("/v1/hosts", json={"name": "For alice", "owner_user_id": "alice"})
    assert refused.status_code in (401, 403, 404, 409), refused.text
    assert [h.name for h in hosts.list_hosts("alice")] == []
    assert computer.owners == []


@pytest.mark.parametrize("change", ["actor_demoted", "actor_suspended"])
def test_a_command_after_a_wait_checks_the_admin_again_before_it_is_sent(
    support_world: World,
    answered: list[tuple[str, str]],
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    world, carol = support_world, support_world.people["carol"]
    host_id = _laptop(world, "alice")
    original = hosts_routes.ready_host_connection

    async def waited(*args: Any, **kwargs: Any) -> Any:
        # The host was woken; meanwhile Carol lost her role.
        (_demote if change == "actor_demoted" else _suspend)(world, "carol")
        return await original(*args, **kwargs)

    monkeypatch.setattr(hosts_routes, "ready_host_connection", waited)
    refused = carol.post(f"/v1/hosts/{host_id}/harnesses/claude/install")
    assert refused.status_code == 403, refused.text
    assert answered == []


def test_deleting_a_members_computer_is_checked_again_in_its_write(
    support_world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    world, carol = support_world, support_world.people["carol"]
    computer, hosts = _computer(world), _stores(world)["hosts"]
    host_id = _cloud_computer(world, "alice")
    original = hosts.delete_host

    def demoted_meanwhile(*args: Any, **kwargs: Any) -> Host | None:
        _demote(world, "carol")
        return original(*args, **kwargs)

    monkeypatch.setattr(hosts, "delete_host", demoted_meanwhile)
    refused = carol.delete(f"/v1/hosts/{host_id}")
    assert refused.status_code == 403, refused.text
    assert hosts.get_host(host_id) is not None
    assert computer.terminated == []
