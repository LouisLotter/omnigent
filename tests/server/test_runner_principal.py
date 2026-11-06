"""Runner principals: scoped, bound, never an administrator.

Accounts-mode app, real routes. Alice and Bob are members without the
administrator flag; ``admin`` is an administrator. Each runner holds a token
minted from its binding token for one thread on its owner's host.
"""

from __future__ import annotations

import ast
import dataclasses
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
import jwt
import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from omnigent.errors import ErrorCode, OmnigentError
from omnigent.runner.identity import RUNNER_TUNNEL_TOKEN_HEADER, token_bound_runner_id
from omnigent.server.auth import LEVEL_EDIT, LEVEL_OWNER
from omnigent.server.route_authority import (
    MOUNT_RULES,
    ROUTE_RULES,
    RouteRule,
    route_keys,
    rule_for,
)
from tests.server.test_account_revocation import create_session
from tests.server.test_account_revocation import setup_app as setup_app
from tests.server.test_accounts import header_mode_app as header_mode_app


@dataclasses.dataclass
class _Runner:
    client: TestClient
    runner_id: str
    host_id: str
    token: str


@pytest.fixture
def people(setup_app: Any) -> tuple[TestClient, TestClient, TestClient, dict[str, Any]]:
    """The administrator, Alice and Bob signed in, plus the app's stores."""
    admin, alice, stores, _ = setup_app
    invite = admin.post("/auth/invite", json={}).json()["token"]
    bob = TestClient(admin.app)
    registered = bob.post(
        "/auth/register", json={"invite": invite, "username": "bob", "password": "bob-pw-12345"}
    )
    assert registered.status_code == 200, registered.text
    return admin, alice, bob, stores


def _bind_runner(stores: dict[str, Any], app: Any, session_id: str, host_owner: str) -> str:
    """Bind a fresh runner to *session_id* on a new host of *host_owner*; return the binding."""
    binding = f"binding-{uuid.uuid4().hex}"
    host_id = uuid.uuid4().hex
    stores["host_store"].upsert_on_connect(host_id, f"host-{host_id[:8]}", host_owner)
    stores["conversation_store"].set_host_id(session_id, host_id, workspace="/tmp/workspace")
    stores["conversation_store"].replace_runner_id(session_id, token_bound_runner_id(binding))
    return binding


def _mint(app: Any, binding: str) -> httpx.Response:
    client = TestClient(app)
    runner_id = token_bound_runner_id(binding)
    return client.post(
        f"/v1/runners/{runner_id}/token", headers={RUNNER_TUNNEL_TOKEN_HEADER: binding}
    )


def _runner(stores: dict[str, Any], person: TestClient, session_id: str, owner: str) -> _Runner:
    """A runner for the owner's thread on the owner's own host, holding its token."""
    binding = _bind_runner(stores, person.app, session_id, owner)
    minted = _mint(person.app, binding)
    assert minted.status_code == 200, minted.text
    client = TestClient(person.app)
    client.headers[RUNNER_TUNNEL_TOKEN_HEADER] = binding
    client.headers["Authorization"] = f"Bearer {minted.json()['token']}"
    conversation = stores["conversation_store"].get_conversation(session_id)
    token = minted.json()["token"]
    return _Runner(client, token_bound_runner_id(binding), conversation.host_id, token)


def _session(person: TestClient) -> str:
    created = create_session(person)
    assert created.status_code == 201, created.text
    return created.json()["session_id"]


def _usage(client: TestClient, session_id: str) -> httpx.Response:
    return client.post(
        f"/v1/sessions/{session_id}/events",
        json={"type": "external_session_usage", "data": {"cumulative_cost_usd": 0.01}},
    )


def test_admin_runner_fails_on_another_members_thread_like_any_runner(people: Any) -> None:
    """T3/T4: the admin keeps the human override; the admin's runner never gets it."""
    admin, alice, bob, stores = people
    bob_thread = _session(bob)
    admin_runner = _runner(stores, admin, _session(admin), "admin")
    alice_runner = _runner(stores, alice, _session(alice), "alice")

    assert admin.get(f"/v1/sessions/{bob_thread}").status_code == 200
    assert admin.get("/v1/me").json()["is_admin"] is True
    assert admin_runner.client.get("/v1/me").json() == {"user_id": "admin", "is_admin": False}
    for runner in (admin_runner, alice_runner):
        client = runner.client
        assert client.get(f"/v1/sessions/{bob_thread}").status_code == 404
        assert client.get(f"/v1/sessions/{bob_thread}/items").status_code == 404
        assert client.get(f"/v1/sessions/{bob_thread}/resources").status_code == 404
        assert client.get(f"/v1/sessions/{bob_thread}/agent").status_code == 404
        assert _usage(client, bob_thread).status_code == 404
        assert client.patch(f"/v1/sessions/{bob_thread}", json={"title": "x"}).status_code in (
            403,
            404,
        )
        assert client.post(
            f"/v1/sessions/{bob_thread}/agent/mcp-servers",
            json={"name": "tracker", "transport": "registry"},
        ).status_code in (403, 404)
        assert client.put(
            f"/v1/sessions/{bob_thread}/permissions", json={"user_id": "admin", "level": 2}
        ).status_code in (403, 404)
        assert client.delete(f"/v1/sessions/{bob_thread}").status_code in (403, 404)
        listed = client.get("/v1/sessions", params={"limit": 100})
        assert listed.status_code == 200, listed.text
        assert bob_thread not in {item["id"] for item in listed.json()["data"]}
    assert admin.get(f"/v1/sessions/{bob_thread}").status_code == 200


def test_runner_never_gets_the_admin_override_after_a_promotion(people: Any) -> None:
    """Promoting a member widens their own reach, not their runner's."""
    _admin, alice, bob, stores = people
    bob_thread = _session(bob)
    alice_runner = _runner(stores, alice, _session(alice), "alice")
    stores["permission_store"].set_admin("alice", True)

    assert alice.get(f"/v1/sessions/{bob_thread}").status_code == 200
    assert alice_runner.client.get(f"/v1/sessions/{bob_thread}").status_code == 404
    assert alice_runner.client.get("/v1/me").json()["is_admin"] is False


def test_runner_does_not_use_its_persons_share_grants(people: Any) -> None:
    """A thread shared with Alice is hers to open, not her runner's to drive."""
    _admin, alice, bob, stores = people
    bob_thread = _session(bob)
    shared = bob.put(
        f"/v1/sessions/{bob_thread}/permissions", json={"user_id": "alice", "level": LEVEL_EDIT}
    )
    assert shared.status_code == 200, shared.text
    alice_runner = _runner(stores, alice, _session(alice), "alice")

    assert alice.get(f"/v1/sessions/{bob_thread}").status_code == 200
    assert alice_runner.client.get(f"/v1/sessions/{bob_thread}").status_code == 404
    assert _usage(alice_runner.client, bob_thread).status_code == 404


_SECRET = "zebracrossing"


def _threads_mentioning_the_secret(
    person: TestClient, stores: dict[str, Any], *, archived: bool
) -> tuple[str, str]:
    """One thread naming the secret in its title, one only in its body."""
    from omnigent.entities.conversation import MessageData, NewConversationItem

    by_title, by_body = _session(person), _session(person)
    assert person.patch(f"/v1/sessions/{by_title}", json={"title": f"{_SECRET} plan"}).is_success
    assert person.patch(f"/v1/sessions/{by_body}", json={"title": "notes"}).is_success
    stores["conversation_store"].append(
        by_body,
        [
            NewConversationItem(
                type="message",
                response_id=f"resp_{by_body}",
                data=MessageData(
                    role="user", content=[{"type": "input_text", "text": f"the {_SECRET} notes"}]
                ),
            )
        ],
    )
    if archived:
        for thread in (by_title, by_body):
            assert person.patch(f"/v1/sessions/{thread}", json={"archived": True}).is_success
    return by_title, by_body


@pytest.mark.parametrize("visibility", ["all", "mine", "shared", "archived"])
@pytest.mark.parametrize("search_scope", [None, "all", "title", "content"])
def test_no_visibility_or_search_lists_a_shared_thread_to_a_runner(
    people: Any, visibility: str, search_scope: str | None
) -> None:
    """Every list filter and search stays within the runner's person's own threads."""
    _admin, alice, bob, stores = people
    archived = visibility == "archived"
    bobs = _threads_mentioning_the_secret(bob, stores, archived=archived)
    for thread in bobs:
        granted = bob.put(
            f"/v1/sessions/{thread}/permissions", json={"user_id": "alice", "level": LEVEL_EDIT}
        )
        assert granted.status_code == 200, granted.text
    own = _threads_mentioning_the_secret(alice, stores, archived=archived)
    runner = _runner(stores, alice, _session(alice), "alice")
    params: dict[str, object] = {"visibility": visibility, "limit": 100}
    if search_scope is not None:
        params |= {"search_query": _SECRET, "search_scope": search_scope}

    human = alice.get("/v1/sessions", params=params)
    listed = runner.client.get("/v1/sessions", params=params)
    assert human.status_code == listed.status_code == 200, listed.text

    seen_by_alice = {item["id"] for item in human.json()["data"]}
    if visibility in ("all", "shared", "archived"):
        assert set(bobs) & seen_by_alice, "the person herself still sees what was shared"
    items = listed.json()["data"]
    assert not set(bobs) & {item["id"] for item in items}
    assert {item["id"] for item in items} <= set(own) | {
        thread for thread in seen_by_alice if thread not in bobs
    }
    if visibility == "shared":
        assert items == []
    elif search_scope == "content":
        # Content search still finds the person's own thread, with its snippet.
        by_id = {item["id"]: item for item in items}
        assert _SECRET in by_id[own[1]]["search_snippet"]


def test_runner_works_on_its_own_persons_threads(people: Any) -> None:
    """Same-person positives: the runner reads, writes, lists and creates for its owner."""
    _admin, alice, _bob, stores = people
    thread = _session(alice)
    other_thread = _session(alice)
    runner = _runner(stores, alice, thread, "alice")
    client = runner.client

    assert client.get(f"/v1/sessions/{thread}").status_code == 200
    assert client.get(f"/v1/sessions/{thread}/items").status_code == 200
    assert _usage(client, thread).status_code == 202
    # Its person's other work is reachable under normal authorization.
    assert client.get(f"/v1/sessions/{other_thread}").status_code == 200
    listed = client.get("/v1/sessions", params={"limit": 100})
    assert {thread, other_thread} <= {item["id"] for item in listed.json()["data"]}
    assert create_session(client).status_code == 201
    assert client.get(f"/v1/hosts/{runner.host_id}").status_code == 200
    assert client.get("/v1/scheduled-tasks").status_code == 200
    assert client.get("/v1/me").json() == {"user_id": "alice", "is_admin": False}


def test_runner_is_refused_on_human_only_routes(people: Any) -> None:
    """Account, sharing, host registration and approval routes stay with people."""
    admin, _alice, _bob, stores = people
    thread = _session(admin)
    runner = _runner(stores, admin, thread, "admin")
    refused = [
        runner.client.get("/v1/usage"),
        runner.client.get("/v1/sharing"),
        runner.client.put("/v1/sharing", json={"mode": "off"}),
        runner.client.post("/v1/hosts", json={"name": "rogue"}),
        runner.client.post(
            f"/v1/sessions/{thread}/elicitations/elicit_1/resolve", json={"action": "accept"}
        ),
        # Publishing under the host's GitHub account, and taking a PR off the thread.
        runner.client.post(
            f"/v1/sessions/{thread}/resources/github/review-threads",
            json={"pr_url": "https://github.com/o/r/pull/1", "body": "from the agent"},
        ),
        runner.client.post(
            f"/v1/sessions/{thread}/resources/github/review-threads/thread_1/replies",
            json={"pr_url": "https://github.com/o/r/pull/1", "body": "from the agent"},
        ),
        runner.client.delete(
            f"/v1/sessions/{thread}/pull-requests",
            params={"url": "https://github.com/o/r/pull/1"},
        ),
        # A computer's sign-ins are its owner's, in person.
        runner.client.post(f"/v1/hosts/{runner.host_id}/harnesses/accounts/refresh"),
        runner.client.post(f"/v1/hosts/{runner.host_id}/harnesses/codex/login"),
        runner.client.get(f"/v1/hosts/{runner.host_id}/harnesses/codex/login/login_1"),
        runner.client.delete(f"/v1/hosts/{runner.host_id}/harnesses/codex/login/login_1"),
        runner.client.post(f"/v1/hosts/{runner.host_id}/harnesses/codex/logout"),
        runner.client.get("/v1/sandbox-profiles"),
    ]
    for response in refused:
        assert response.status_code == 403, response.text
        assert response.json()["error"]["code"] == "machine_principal_forbidden"
    # Sign-in and credential routes never accept a runner token.
    assert runner.client.get("/auth/me").status_code == 401
    assert runner.client.post("/auth/invite", json={}).status_code in (401, 403)
    assert admin.get("/v1/usage").status_code == 200


def test_a_runner_reads_the_pull_requests_of_its_persons_threads_only(people: Any) -> None:
    """The read routes take the session scope, and the owner grant still decides."""
    _admin, alice, bob, stores = people
    thread = _session(alice)
    others = _session(bob)
    runner = _runner(stores, alice, thread, "alice")

    for suffix in ("pull-requests", "resources/repository-branches"):
        assert runner.client.get(f"/v1/sessions/{thread}/{suffix}").status_code == 200
        foreign = runner.client.get(f"/v1/sessions/{others}/{suffix}")
        assert foreign.status_code in (403, 404), foreign.text
    review_threads = f"/v1/sessions/{others}/resources/github/review-threads"
    params = {"pr_url": "https://github.com/o/r/pull/1"}
    assert runner.client.get(review_threads, params=params).status_code in (403, 404)
    assert runner.client.get(
        f"{review_threads}/thread_1/comments", params={**params, "after": "cursor"}
    ).status_code in (403, 404)


def test_runner_token_cannot_open_a_host_tunnel_or_another_runners_tunnel(people: Any) -> None:
    """A runner token registers no computer and opens no other runner's tunnel."""
    _admin, alice, _bob, stores = people
    runner = _runner(stores, alice, _session(alice), "alice")
    other_binding = _bind_runner(stores, alice.app, _session(alice), "alice")
    other_id = token_bound_runner_id(other_binding)
    headers = {"Authorization": f"Bearer {runner.token}"}

    with pytest.raises(WebSocketDisconnect) as host_tunnel:
        with runner.client.websocket_connect(
            f"/v1/hosts/{uuid.uuid4().hex}/tunnel", headers=headers
        ) as ws:
            ws.receive_text()
    assert host_tunnel.value.code == 4403
    with pytest.raises(WebSocketDisconnect) as stolen:
        with TestClient(alice.app).websocket_connect(
            f"/v1/runners/{other_id}/tunnel",
            headers={**headers, RUNNER_TUNNEL_TOKEN_HEADER: other_binding},
        ) as ws:
            ws.receive_text()
    assert stolen.value.code == 4004


def test_runner_token_dies_with_its_binding(people: Any) -> None:
    """T7: a replaced runner, a moved thread or a deleted host ends the token."""
    _admin, alice, _bob, stores = people
    conversations = stores["conversation_store"]
    thread = _session(alice)

    replaced = _runner(stores, alice, thread, "alice")
    conversations.replace_runner_id(thread, token_bound_runner_id("next-incarnation"))
    assert replaced.client.get(f"/v1/sessions/{thread}").status_code == 401

    moved = _runner(stores, alice, thread, "alice")
    other_host = uuid.uuid4().hex
    stores["host_store"].upsert_on_connect(other_host, "other", "alice")
    conversations.set_host_id(thread, other_host)
    assert moved.client.get(f"/v1/sessions/{thread}").status_code == 401

    deleted = _runner(stores, alice, thread, "alice")
    stores["host_store"].delete_host(deleted.host_id)
    assert deleted.client.get(f"/v1/sessions/{thread}").status_code == 401


def _bearer(app: Any, token: str) -> TestClient:
    client = TestClient(app)
    client.headers["Authorization"] = f"Bearer {token}"
    return client


def test_replacing_or_detaching_a_hosts_sandbox_retires_its_runners(people: Any) -> None:
    """T7: a new computer incarnation under the same host ID and account ends old runners.

    Neither the old runner's token nor a fresh mint with its binding token
    works after the sandbox is replaced; a runner launched into the new
    sandbox does. Detaching the sandbox retires that one in turn.
    """
    _admin, alice, _bob, stores = people
    hosts, conversations = stores["host_store"], stores["conversation_store"]
    thread = _session(alice)
    host_id = uuid.uuid4().hex
    hosts.upsert_on_connect(host_id, "cloud-computer", "alice")
    assert hosts.bind_sandbox(host_id, provider="fake", sandbox_id="sandbox-1") is not None
    conversations.set_host_id(thread, host_id, workspace="/tmp/workspace")
    old_binding = f"binding-{uuid.uuid4().hex}"
    conversations.replace_runner_id(thread, token_bound_runner_id(old_binding))
    minted = _mint(alice.app, old_binding)
    assert minted.status_code == 200, minted.text
    old_runner = _bearer(alice.app, minted.json()["token"])
    assert old_runner.get(f"/v1/sessions/{thread}").status_code == 200
    registration = stores["account_store"].get_user("alice").account_generation

    replaced = hosts.replace_managed_host_sandbox(
        host_id=host_id,
        user_id="alice",
        token=uuid.uuid4().hex,
        provider="fake",
        sandbox_id="sandbox-2",
        token_expires_at=int(time.time()) + 3600,
    )

    assert replaced is not None and replaced.host_id == host_id
    assert stores["account_store"].get_user("alice").account_generation == registration
    assert old_runner.get(f"/v1/sessions/{thread}").status_code == 401
    assert _mint(alice.app, old_binding).status_code == 401
    new_binding = f"binding-{uuid.uuid4().hex}"
    conversations.replace_runner_id(thread, token_bound_runner_id(new_binding))
    renewed = _mint(alice.app, new_binding)
    assert renewed.status_code == 200, renewed.text
    new_runner = _bearer(alice.app, renewed.json()["token"])
    assert new_runner.get(f"/v1/sessions/{thread}").status_code == 200

    current = hosts.get_host(host_id)
    assert current is not None
    assert hosts.detach_stale_managed_sandbox(
        host_id, sandbox_id="sandbox-2", expected_updated_at=current.updated_at
    )
    assert new_runner.get(f"/v1/sessions/{thread}").status_code == 401
    assert _mint(alice.app, new_binding).status_code == 401


def _retire_binding(stores: dict[str, Any], thread: str) -> None:
    """Mark the thread's runner binding retired, as the incarnation migration does."""
    from sqlalchemy import update

    from omnigent.db.db_models import SqlConversationMetadata, current_workspace_id
    from omnigent.stores.host_store import RETIRED_RUNNER_INCARNATION

    with stores["conversation_store"]._engine.begin() as connection:
        connection.execute(
            update(SqlConversationMetadata)
            .where(
                SqlConversationMetadata.workspace_id == current_workspace_id(),
                SqlConversationMetadata.id == thread,
            )
            .values(runner_host_incarnation=RETIRED_RUNNER_INCARNATION)
        )


@pytest.mark.parametrize("retirement", ["retired-by-migration", "sandbox-replaced"])
def test_a_retired_binding_is_refused_like_a_replaced_one(people: Any, retirement: str) -> None:
    """A binding retired at upgrade gets 401 and no mint; a fresh launch works."""
    _admin, alice, _bob, stores = people
    hosts, conversations = stores["host_store"], stores["conversation_store"]
    thread = _session(alice)
    host_id = uuid.uuid4().hex
    hosts.upsert_on_connect(host_id, "cloud-computer", "alice")
    assert hosts.bind_sandbox(host_id, provider="fake", sandbox_id="sandbox-1") is not None
    conversations.set_host_id(thread, host_id, workspace="/tmp/workspace")
    old_binding = f"binding-{uuid.uuid4().hex}"
    conversations.replace_runner_id(thread, token_bound_runner_id(old_binding))
    minted = _mint(alice.app, old_binding)
    assert minted.status_code == 200, minted.text
    old_runner = _bearer(alice.app, minted.json()["token"])

    if retirement == "retired-by-migration":
        _retire_binding(stores, thread)
    else:
        replaced = hosts.replace_managed_host_sandbox(
            host_id=host_id,
            user_id="alice",
            token=uuid.uuid4().hex,
            provider="fake",
            sandbox_id="sandbox-2",
            token_expires_at=int(time.time()) + 3600,
        )
        assert replaced is not None

    assert old_runner.get(f"/v1/sessions/{thread}").status_code == 401
    refused = _mint(alice.app, old_binding)
    assert refused.status_code == 401
    assert refused.json()["error"]["code"] == "unauthorized"
    new_binding = f"binding-{uuid.uuid4().hex}"
    conversations.replace_runner_id(thread, token_bound_runner_id(new_binding))
    renewed = _mint(alice.app, new_binding)
    assert renewed.status_code == 200, renewed.text
    assert _bearer(alice.app, renewed.json()["token"]).get(f"/v1/sessions/{thread}").is_success


def test_runner_token_from_one_thread_names_only_that_binding(people: Any) -> None:
    """A token whose claims are edited (another thread, host or owner) fails its signature."""
    _admin, alice, bob, stores = people
    runner = _runner(stores, alice, _session(alice), "alice")
    claims = jwt.decode(runner.token, options={"verify_signature": False})
    for field, value in (("sid", _session(bob)), ("host", uuid.uuid4().hex), ("sub", "bob")):
        forged = jwt.encode({**claims, field: value}, b"x" * 32, algorithm="HS256")
        client = TestClient(alice.app)
        client.headers["Authorization"] = f"Bearer {forged}"
        assert client.get("/v1/sessions").status_code == 401


def test_expired_and_legacy_runner_tokens_are_refused(
    people: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An expired token and a pre-cutover owner JWT marked ``delegated=runner`` both fail."""
    import omnigent.server.routes.runner_tunnel as tunnel_module
    from omnigent.server.oidc import mint_session_token

    _admin, alice, _bob, stores = people
    thread = _session(alice)
    monkeypatch.setattr(tunnel_module, "_MANAGED_RUNNER_TOKEN_TTL_S", -1)
    expired = _mint(alice.app, _bind_runner(stores, alice.app, thread, "alice"))
    assert expired.status_code == 200, expired.text
    client = TestClient(alice.app)
    client.headers["Authorization"] = f"Bearer {expired.json()['token']}"
    assert client.get(f"/v1/sessions/{thread}").status_code == 401

    config = stores["auth_provider"]._accounts_config
    generation = stores["account_store"].get_user("alice").account_generation
    legacy = jwt.decode(
        mint_session_token(
            "alice", config.cookie_secret, 1800, "accounts", account_generation=generation
        ),
        options={"verify_signature": False},
    )
    legacy_token = jwt.encode(
        {**legacy, "delegated": "runner"}, config.cookie_secret, algorithm="HS256"
    )
    client.headers["Authorization"] = f"Bearer {legacy_token}"
    assert client.get(f"/v1/sessions/{thread}").status_code == 401


def test_header_mode_server_says_runner_tokens_are_unsupported_before_any_binding(
    header_mode_app: TestClient,
) -> None:
    """A server that can't mint answers with the explicit code, never a refusal.

    The runner treats only this answer (or 404) as "use the older bootstrap",
    so it must not depend on whether the runner's binding resolves.
    """
    binding = "header-mode-runner-binding"
    response = header_mode_app.post(
        f"/v1/runners/{token_bound_runner_id(binding)}/token",
        headers={RUNNER_TUNNEL_TOKEN_HEADER: binding},
    )

    assert response.status_code == 400, response.text
    assert response.json()["error"]["code"] == "runner_tokens_unsupported"


def test_accounts_server_refuses_rather_than_reports_unsupported(people: Any) -> None:
    """An accounts server offers runner tokens: an unbound runner is refused (401)."""
    _admin, alice, _bob, _stores = people
    refused = _mint(alice.app, "never-bound-binding")
    assert refused.status_code == 401
    assert refused.json()["error"]["code"] != "runner_tokens_unsupported"


def test_no_runner_token_for_a_thread_on_someone_elses_computer(people: Any) -> None:
    """An admin's host can't mint for a member's thread, nor a member's host for the admin's."""
    admin, alice, _bob, stores = people
    alice_thread = _session(alice)
    admin_thread = _session(admin)

    on_admin_host = _bind_runner(stores, alice.app, alice_thread, "admin")
    on_member_host = _bind_runner(stores, admin.app, admin_thread, "alice")
    assert _mint(alice.app, on_admin_host).status_code == 401
    assert _mint(admin.app, on_member_host).status_code == 401


def test_launch_admission_refuses_a_host_owner_who_is_not_the_thread_owner(people: Any) -> None:
    """``admit_launch`` binds a runner only for the thread's single owner."""
    _admin, alice, _bob, stores = people
    alice_thread = _session(alice)
    hosts = stores["host_store"]
    admin_host = uuid.uuid4().hex
    hosts.upsert_on_connect(admin_host, "admin-laptop", "admin")
    stores["conversation_store"].set_host_id(alice_thread, admin_host, workspace="/tmp/w")
    generation = stores["account_store"].get_user("admin").account_generation

    with pytest.raises(OmnigentError) as refused:
        hosts.admit_launch(admin_host, alice_thread, "admin", generation)
    assert refused.value.code == ErrorCode.FORBIDDEN


def test_duplicate_owner_grants_name_no_owner_and_mint_nothing(people: Any) -> None:
    """Ambiguous ownership is shown as no owner and binds no runner."""
    _admin, alice, _bob, stores = people
    thread = _session(alice)
    binding = _bind_runner(stores, alice.app, thread, "alice")
    assert _mint(alice.app, binding).status_code == 200
    stores["permission_store"].grant("bob", thread, LEVEL_OWNER)

    assert alice.get(f"/v1/sessions/{thread}/owner").json() == {"owner": None}
    assert _mint(alice.app, binding).status_code == 401


def test_colleague_message_runs_on_the_owners_runner_with_separate_attribution(
    people: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D51: Bob's message in Alice's shared thread goes to Alice's runner; roles stay apart."""
    from omnigent.server.routes.sessions import routes_events

    _admin, alice, bob, stores = people
    thread = _session(alice)
    runner = _runner(stores, alice, thread, "alice")
    granted = alice.put(
        f"/v1/sessions/{thread}/permissions", json={"user_id": "bob", "level": LEVEL_EDIT}
    )
    assert granted.status_code == 200, granted.text
    dispatched: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        dispatched.append(request.url.path)
        return httpx.Response(202, json={"queued": True})

    async def to_owners_runner(session_id: str, *_args: object, **_kwargs: object) -> Any:
        conversation = stores["conversation_store"].get_conversation(session_id)
        assert conversation.runner_id == runner.runner_id
        return httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://r")

    async def relay_ready(*_args: object, **_kwargs: object) -> None:
        return None

    audit: dict[str, object] = {}
    monkeypatch.setattr("omnigent.server.routes.sessions._get_runner_client", to_owners_runner)
    monkeypatch.setattr("omnigent.server.routes.sessions._ensure_runner_relay_ready", relay_ready)
    monkeypatch.setattr(routes_events, "debug_sink_enabled", lambda: True)
    monkeypatch.setattr(routes_events, "add_audit_attrs", lambda **attrs: audit.update(attrs))
    message = {
        "type": "message",
        "data": {"role": "user", "content": [{"type": "input_text", "text": "please check"}]},
    }

    sent = bob.post(f"/v1/sessions/{thread}/events", json=message)
    assert sent.status_code == 202, sent.text
    assert dispatched, "the owner's runner never received the colleague's turn"
    assert audit["initiator"] == "bob"
    assert audit["thread_owner"] == "alice"
    items = alice.get(f"/v1/sessions/{thread}/items").json()["data"]
    assert any(item.get("created_by") == "bob" for item in items)
    # The owner's runner keeps working in its own thread after the shared turn.
    assert _usage(runner.client, thread).status_code == 202


def test_unclassified_route_fails_closed_for_everyone(people: Any) -> None:
    """A route missing from the inventory answers nobody."""
    admin, _alice, _bob, _stores = people

    @admin.app.get("/v1/unclassified-probe")
    async def probe() -> dict[str, bool]:
        return {"ok": True}

    refused = admin.get("/v1/unclassified-probe")
    assert refused.status_code == 403
    assert refused.json()["error"]["code"] == "route_unclassified"


# ── Route inventory ──────────────────────────────────────────────────────


@pytest.fixture
def maximal_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """An accounts-mode app with every optional router this test can enable."""
    import omnigent.server.app as app_module
    from omnigent.server.mcp_destinations import DefaultMcpDestinationPolicy
    from omnigent.server.mcp_registry import McpRegistry, McpRegistryConfig
    from omnigent.stores.credential_store.sqlalchemy_store import CredentialStore
    from omnigent.stores.mcp_service_store import McpServiceStore
    from omnigent.stores.policy_store.sqlalchemy_store import SqlAlchemyPolicyStore
    from omnigent.stores.preference_store.sqlalchemy_store import SqlAlchemyPreferenceStore
    from omnigent.stores.project_store.sqlalchemy_store import SqlAlchemyProjectStore
    from omnigent.stores.scheduled_task_store.sqlalchemy_store import (
        SqlAlchemyScheduledTaskStore,
    )
    from tests.server.test_accounts import _build_accounts_app
    from tests.server.test_credential_store import _FakeCipher

    monkeypatch.setattr(app_module, "_WEB_UI_DIST", tmp_path / "no-web-ui")
    monkeypatch.delenv("OMNIGENT_OIDC_ISSUER", raising=False)
    monkeypatch.setenv("OMNIGENT_DEVICE_GRANT_ENABLED", "1")
    original = app_module.create_app

    def capture(**kwargs: Any) -> Any:
        db_url = str(kwargs["account_store"]._engine.url)
        registry = McpRegistry(
            McpRegistryConfig(services=[]),
            CredentialStore(db_url, _FakeCipher()),
            service_store=McpServiceStore(db_url),
            destination_policy=DefaultMcpDestinationPolicy(),
        )
        return original(
            **kwargs,
            policy_store=SqlAlchemyPolicyStore(db_url),
            scheduled_task_store=SqlAlchemyScheduledTaskStore(db_url),
            project_store=SqlAlchemyProjectStore(db_url),
            preference_store=SqlAlchemyPreferenceStore(db_url, authority_id="inventory"),
            mcp_registry=registry,
        )

    monkeypatch.setattr(app_module, "create_app", capture)
    built = _build_accounts_app(tmp_path, monkeypatch, init_admin_password="admin-pw-12345")
    client = next(built)
    try:
        yield client.app
    finally:
        built.close()


def test_every_mounted_route_is_classified(maximal_app: Any) -> None:
    """Route inventory: a new endpoint must be classified before anyone can reach it."""
    unclassified = []
    for route in maximal_app.router.routes:
        mount = getattr(route, "name", None) if not hasattr(route, "methods") else None
        if mount in MOUNT_RULES:
            continue
        keys = route_keys(route)
        unclassified += [key for key in keys if rule_for(*key) is None]
    assert unclassified == []
    assert sum(isinstance(route, APIRoute) for route in maximal_app.router.routes) > 150


_DECORATOR_METHODS = {"get", "post", "put", "patch", "delete", "websocket"}


def _decorated_routes() -> list[tuple[str, str, str]]:
    """Every route decorator under ``omnigent/server``: (file, method, path pattern)."""
    import omnigent.server

    root = Path(omnigent.server.__file__).parent
    found = []
    for path in sorted(root.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for decorator in node.decorator_list:
                if (
                    isinstance(decorator, ast.Call)
                    and isinstance(decorator.func, ast.Attribute)
                    and decorator.func.attr in _DECORATOR_METHODS
                    and decorator.args
                ):
                    method = decorator.func.attr.upper().replace("WEBSOCKET", "WS")
                    found.append((path.name, method, ast.unparse(decorator.args[0])))
    return found


# Router mount prefixes (see ``create_app``): API routers under ``/v1``; the
# sign-in routers under ``/auth``; device authorization declares full paths.
_SIGN_IN_ROUTERS = frozenset({"auth.py", "accounts_auth.py"})
# Provider connection routes are built from the provider's name at run time.
_RUNTIME_PATH_FAMILIES = {"connections_base.py": "/v1/connections/provider"}


def _classified(file: str, method: str, pattern: str) -> bool:
    """Whether the route this decorator declares is classified wherever it is mounted."""
    if pattern.startswith(("f'", 'f"')):
        family = _RUNTIME_PATH_FAMILIES.get(file)
        return family is not None and rule_for(method, family) is not None
    path = ast.literal_eval(pattern)
    prefixes = ("/auth",) if file in _SIGN_IN_ROUTERS else ("", "/v1")
    if any(rule_for(method, prefix + path) is not None for prefix in prefixes):
        return True
    # Routers with their own prefix (for example a session router's
    # ``/sessions/{session_id}`` routes) end with the declared path.
    return any(m == method and full.endswith(path) for (m, full) in ROUTE_RULES)


def test_every_route_decorator_is_classified() -> None:
    """Routes this test app can't mount (OIDC sign-in, connections) are classified too."""
    unclassified = [
        (file, method, pattern)
        for file, method, pattern in _decorated_routes()
        if not _classified(file, method, pattern)
    ]
    assert unclassified == []


def test_rule_admission_matrix() -> None:
    """Humans pass human rules; runners need the named scope; hosts have no HTTP routes."""
    from omnigent.server.principals import RUNNER_SCOPES, Principal, PrincipalKind

    human = Principal(PrincipalKind.HUMAN, "alice")
    runner = Principal(PrincipalKind.RUNNER, "alice", scopes=RUNNER_SCOPES)
    scopeless = Principal(PrincipalKind.RUNNER, "alice")
    host = Principal(PrincipalKind.HOST, "alice", host_id="h")
    session_rule = rule_for("GET", "/v1/sessions/{session_id}")
    assert isinstance(session_rule, RouteRule)
    assert session_rule.admits(human) and session_rule.admits(runner)
    assert not session_rule.admits(scopeless) and not session_rule.admits(host)
    usage = rule_for("GET", "/v1/usage")
    assert usage is not None and usage.admits(human) and not usage.admits(runner)
    assert rule_for("POST", "/auth/anything") is not None
    assert rule_for("GET", "/v1/not-a-route") is None
