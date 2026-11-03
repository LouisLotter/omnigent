"""Approvals through the real provider, gateway, account and hook paths (review r2).

Each test drives a production path end to end: the real GitHub/Databricks
connection stores and credential resolvers behind the registry's token read,
the gateway's own backend up to the MCP transport, the approval routes with
real account deletion, and the Claude/Codex hook functions with the real
hook converter. Only the outermost transports are replaced: the MCP session
(which records the credential it was handed) and the hook's HTTP client
(which posts to the app).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from mcp.types import CallToolResult, TextContent

from omnigent.server import mcp_registry as mcp_registry_module
from omnigent.server.mcp_registry import McpRegistry, McpRegistryConfig, McpService
from omnigent.server.routes._sessions import orchestration
from omnigent.stores.approval_store import ApprovalResource, PolicyDomain, ResourceKind
from tests.server.test_account_revocation import create_session
from tests.server.test_account_revocation import setup_app as setup_app
from tests.server.test_approver_roles import (
    _CEL,
    _member,
    _name_approver,
    _team_rule,
)
from tests.server.test_approver_roles import custom_handler as custom_handler
from tests.server.test_approver_roles import team as team
from tests.server.test_delegated_runner_tokens import (
    _ASK_ON_TOOL_CALL,
    _delegated_runner,
    _resolve_url,
)
from tests.server.test_delegated_runner_tokens import tracker as tracker

# ── The MCP transport: records the credential each call was sent with ───────


@pytest.fixture
def sent(monkeypatch: pytest.MonkeyPatch) -> list[str | None]:
    """Replace only the MCP session; everything up to it is the registry's own."""
    calls: list[str | None] = []

    @contextlib.asynccontextmanager
    async def streamable_http_client(url: str, *, http_client: httpx.AsyncClient):
        calls.append(http_client.headers.get("Authorization"))
        yield None, None, None

    class _Session:
        def __init__(self, read: Any, write: Any) -> None:
            pass

        async def __aenter__(self) -> _Session:
            return self

        async def __aexit__(self, *exc: object) -> None:
            return None

        async def initialize(self) -> None:
            return None

        async def call_tool(self, tool: str, arguments: dict[str, Any]) -> CallToolResult:
            return CallToolResult(content=[TextContent(type="text", text="posted")])

    monkeypatch.setattr(mcp_registry_module, "streamable_http_client", streamable_http_client)
    monkeypatch.setattr(mcp_registry_module, "ClientSession", _Session)
    return calls


def _session_using(team: dict[str, Any], service: str) -> tuple[str, TestClient]:
    """Alice's thread asking before every call to *service*, with her runner."""
    guardrails = {
        "policies": {
            "approve_posts": {
                "type": "function",
                "function": {"path": _CEL, "arguments": {"expression": _ASK_ON_TOOL_CALL}},
            }
        }
    }
    created = create_session(team["alice"], guardrails=guardrails)
    assert created.status_code == 201, created.text
    session_id = created.json()["session_id"]
    attached = team["alice"].post(
        f"/v1/sessions/{session_id}/agent/mcp-servers",
        json={"name": service, "transport": "registry"},
    )
    assert attached.status_code == 200, attached.text
    return session_id, _delegated_runner(team["alice"], team["stores"], session_id)


def _call(runner: TestClient, session_id: str, service: str, **params: Any) -> dict[str, Any]:
    response = runner.post(
        f"/v1/mcp/{service}",
        headers={"X-Omnigent-Session-Id": session_id},
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "post_update", "arguments": {"text": "hello"}, **params},
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


def _ask_and_approve(
    team: dict[str, Any], runner: TestClient, session_id: str, service: str
) -> dict[str, Any]:
    asked = _call(runner, session_id, service)["result"]
    assert asked["resultType"] == "input_required", asked
    elicitation_id = next(iter(asked["inputRequests"]))
    approved = team["alice"].post(
        _resolve_url(session_id, elicitation_id), json={"action": "accept"}
    )
    assert approved.status_code == 202, approved.text
    return {
        "requestState": asked["requestState"],
        "inputResponses": {elicitation_id: {"action": "accept"}},
    }


def _db_url(team: dict[str, Any]) -> str:
    return team["stores"]["permission_store"].storage_location


# ── r2-1: GitHub and Databricks logins ──────────────────────────────────────


def _provider_login(team: dict[str, Any], provider: str) -> Any:
    """Install the real provider store on the app; return a reconnect function."""
    from tests.server.test_credential_store import _FakeCipher

    app = team["alice"].app
    if provider == "github":
        from omnigent.connections.github import GithubConnectionStore
        from omnigent.server.github_app import GitHubTokenSet

        store = GithubConnectionStore(_db_url(team), _FakeCipher())

        def connect(token: str) -> None:
            store.upsert(
                "alice",
                github_login="alice-gh",
                github_user_id=7,
                tokens=GitHubTokenSet(
                    access_token=token,
                    refresh_token=None,
                    expires_at=None,
                    refresh_token_expires_at=None,
                    scopes="",
                ),
            )
    else:
        from omnigent.connections.databricks import DatabricksConnectionStore
        from omnigent.server.databricks_app import DatabricksTokenSet

        store = DatabricksConnectionStore(_db_url(team), _FakeCipher())

        def connect(token: str) -> None:
            store.upsert(
                "alice",
                workspace_host="https://workspace.example.test",
                databricks_user="alice@example.test",
                databricks_user_id="7",
                tokens=DatabricksTokenSet(
                    access_token=token,
                    refresh_token=None,
                    expires_at=None,
                    refresh_token_expires_at=None,
                    scopes="",
                ),
            )

    setattr(app.state, f"{provider}_store", store)
    setattr(app.state, f"{provider}_client", object())  # no refresh is needed
    app.state.mcp_registry = McpRegistry(
        McpRegistryConfig(
            services=[
                McpService(
                    id="provider-tool",
                    title="Provider tool",
                    url="https://provider.example.test/mcp",
                    auth=provider,
                    tools=["post_update"],
                )
            ]
        ),
        None,
    )
    app.state.mcp_gateway_backend = None  # the registry's own backend
    connect("first-token")
    return connect


@pytest.mark.parametrize("provider", ["github", "databricks"])
@pytest.mark.parametrize("reconnect", ["none", "before_consumption", "before_token_read"])
def test_a_provider_reconnect_never_uses_an_approval_for_the_old_login(
    team, sent, monkeypatch, provider, reconnect
) -> None:
    """r2-1: the approval is bound to the provider login; its token read enforces it."""
    connect = _provider_login(team, provider)
    session_id, runner = _session_using(team, "provider-tool")
    retry = _ask_and_approve(team, runner, session_id, "provider-tool")
    if reconnect == "before_consumption":
        connect("second-token")
    elif reconnect == "before_token_read":
        registry = team["alice"].app.state.mcp_registry
        real_token = registry.token

        async def reconnect_then_read(*args: Any, **kwargs: Any) -> Any:
            await asyncio.to_thread(connect, "second-token")
            return await real_token(*args, **kwargs)

        monkeypatch.setattr(registry, "token", reconnect_then_read)

    response = _call(runner, session_id, "provider-tool", **retry)

    if reconnect == "none":
        assert response["result"]["content"][0]["text"] == "posted", response
        assert sent == ["Bearer first-token"]
        return
    # Each check refuses on its own: the binding at consumption, the pin at the token read.
    expected = {
        "before_consumption": "The action changed after it was approved; ask again",
        "before_token_read": "The MCP login changed after this call was approved; ask again",
    }[reconnect]
    assert response["error"]["message"] == expected, response
    assert sent == []


def test_a_custom_gateway_backend_must_declare_it_honours_the_approved_login(
    team, tracker
) -> None:
    """r2-1: an approved call through a backend with its own credentials is refused."""
    from tests.server.test_approver_roles import _ask, _tools_call

    del tracker.enforces_pinned_login
    session_id, runner = team["session"], team["runner"]
    elicitation_id, retry = _ask(runner, session_id)
    assert (
        team["alice"]
        .post(_resolve_url(session_id, elicitation_id), json={"action": "accept"})
        .status_code
        == 202
    )

    refused = _tools_call(runner, session_id, **retry)

    assert "cannot bind an approved call" in refused["error"]["message"], refused
    tracker.call_tool.assert_not_awaited()


# ── r2-2: the pin is the login that was checked ─────────────────────────────


def test_the_pinned_login_is_the_one_checked_not_a_later_read(team, sent, monkeypatch) -> None:
    """r2-2: a reconnect right after the pre-send check is refused at the token read."""
    from omnigent.stores.credential_store.sqlalchemy_store import CredentialStore
    from tests.server.test_credential_store import _FakeCipher

    app = team["alice"].app
    service = McpService(
        id="bearer-tool",
        title="Bearer tool",
        url="https://bearer.example.test/mcp",
        auth="bearer",
        tools=["post_update"],
    )
    registry = McpRegistry(
        McpRegistryConfig(services=[service]), CredentialStore(_db_url(team), _FakeCipher())
    )
    app.state.mcp_registry = registry
    app.state.mcp_gateway_backend = None
    registry.save_login(service, "alice", "mcp:bearer-tool", {"access_token": "first-token"}, {})
    session_id, runner = _session_using(team, "bearer-tool")
    retry = _ask_and_approve(team, runner, session_id, "bearer-tool")

    real_snapshot = orchestration._registry_login_snapshot
    reads: list[object] = []

    def snapshot_then_reconnect(*args: Any) -> Any:
        snapshot = real_snapshot(*args)
        reads.append(snapshot)
        if len(reads) == 2:
            # The pre-send comparison has just read the approved login.
            registry.save_login(
                service, "alice", "mcp:bearer-tool", {"access_token": "second-token"}, {}
            )
        return snapshot

    monkeypatch.setattr(orchestration, "_registry_login_snapshot", snapshot_then_reconnect)

    response = _call(runner, session_id, "bearer-tool", **retry)

    assert len(reads) >= 2
    assert "ask again" in response["error"]["message"], response
    assert sent == []


# ── r2-3: a request authorized before its actor was deleted ─────────────────


def _admin2(team: dict[str, Any]) -> TestClient:
    invite = team["admin"].post("/auth/invite", json={"is_admin": True}).json()["token"]
    admin2 = TestClient(team["admin"].app)
    registered = admin2.post(
        "/auth/register",
        json={"invite": invite, "username": "admin2", "password": "admin2-pw-12345"},
    )
    assert registered.status_code == 200, registered.text
    return admin2


@contextlib.contextmanager
def _paused(store: Any, method: str) -> Iterator[tuple[threading.Event, threading.Event]]:
    """Hold *method* after the route authorized the request, until released."""
    entered, release = threading.Event(), threading.Event()
    real = getattr(store, method)

    def held(*args: Any, **kwargs: Any) -> Any:
        entered.set()
        assert release.wait(30)
        return real(*args, **kwargs)

    setattr(store, method, held)
    try:
        yield entered, release
    finally:
        setattr(store, method, real)


@pytest.mark.parametrize("operation", ["name_approver", "delete_team_rule"])
def test_a_deleted_administrators_paused_request_cannot_complete(team, operation) -> None:
    """r2-3: authorized, then the actor is deleted, then resumed: refused, nothing changed."""
    store = team["approvals"]
    admin2 = _admin2(team)
    rule = _team_rule(team, '{"result": "ASK", "reason": "keep me"}', name="keep")
    method, request = {
        "name_approver": (
            "add_approver",
            lambda: admin2.post(
                "/v1/approvers",
                json={"kind": "service", "resource_id": "tracker", "user_id": "bob"},
            ),
        ),
        "delete_team_rule": (
            "delete_rule",
            lambda: admin2.delete(f"/v1/protected-policies/{rule['id']}"),
        ),
    }[operation]
    results: list[httpx.Response] = []

    with _paused(store, method) as (entered, release):
        worker = threading.Thread(target=lambda: results.append(request()))
        worker.start()
        assert entered.wait(30), "the request did not reach the store"
        deleted = team["admin"].delete("/auth/users/admin2")
        assert deleted.status_code == 204, deleted.text
        release.set()
        worker.join(30)

    (response,) = results
    assert response.status_code in (401, 403), response.text
    service = ApprovalResource(ResourceKind.SERVICE, "tracker")
    assert store.approvers_for([service])[0].approvers == ()
    assert store.get_rule(rule["id"]) is not None


def test_a_late_personal_rule_never_reaches_a_reused_username(team) -> None:
    """r2-3: Alice's paused rule creation after her deletion fails; a new Alice inherits none."""
    store = team["approvals"]
    body = {
        "domain": "personal",
        "name": "mine",
        "handler": _CEL,
        "factory_params": {"expression": "{}"},
    }
    results: list[httpx.Response] = []

    with _paused(store, "create_rule") as (entered, release):
        worker = threading.Thread(
            target=lambda: results.append(team["alice"].post("/v1/protected-policies", json=body))
        )
        worker.start()
        assert entered.wait(30)
        deleted = team["admin"].delete("/auth/users/alice")
        assert deleted.status_code == 204, deleted.text
        release.set()
        worker.join(30)

    (response,) = results
    assert response.status_code in (400, 401, 403), response.text
    new_alice = _member(team["admin"], "alice")
    listed = new_alice.get("/v1/protected-policies", params={"domain": "personal"})
    assert listed.json()["data"] == []
    assert store.list_rules(domain=PolicyDomain.PERSONAL, owner_user_id="alice") == []


# ── r2-4: what native and relay gates execute ──────────────────────────────


def rewrite_bash_command(event: dict[str, Any]) -> Any:
    """An ordinary policy that transforms the command (registered like a custom handler)."""
    from omnigent.policies.types import PolicyResult
    from omnigent.spec.types import PolicyAction

    arguments = (event.get("data") or {}).get("arguments") or {}
    if event.get("type") != "tool_call" or arguments.get("command") not in (
        "blocked",
        "review-me",
    ):
        return None
    return PolicyResult(action=PolicyAction.ALLOW, data={**arguments, "command": "harmless"})


POLICY_REGISTRY = [
    {"handler": f"{__name__}.rewrite_bash_command", "description": "Test transform."},
]


@pytest.fixture
def rewriting(team: dict[str, Any]) -> Iterator[tuple[str, TestClient]]:
    """Alice's thread whose agent rewrites Bash commands; a team rule judges commands."""
    from omnigent.policies.registry import load_registry

    load_registry([__name__, "tests.server.test_approver_roles"])
    try:
        guardrails = {
            "policies": {
                "rewrite": {
                    "type": "function",
                    "function": {"path": f"{__name__}.rewrite_bash_command"},
                }
            }
        }
        created = create_session(team["alice"], guardrails=guardrails)
        assert created.status_code == 201, created.text
        session_id = created.json()["session_id"]
        _team_rule(
            team,
            'event.type == "tool_call" && event.data.arguments.command == "blocked"'
            ' ? {"result": "DENY", "reason": "blocked command"}'
            ' : event.type == "tool_call" && event.data.arguments.command == "review-me"'
            ' ? {"result": "ASK", "reason": "needs review"} : {"result": "ALLOW"}',
            name="commands",
        )
        _name_approver(team, "team", "dave")
        yield session_id, _delegated_runner(team["alice"], team["stores"], session_id)
    finally:
        load_registry()


class _AppLoop:
    """One event loop serving every request, so a parked gate and its answer share it."""

    def __init__(self, app: Any) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        self._client = self.run(self._open(app))

    @staticmethod
    async def _open(app: Any) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        )

    def run(self, coroutine: Any) -> Any:
        return asyncio.run_coroutine_threadsafe(coroutine, self._loop).result(60)

    def post(self, path: str, **kwargs: Any) -> httpx.Response:
        return self.run(self._client.post(path, **kwargs))

    def close(self) -> None:
        self.run(self._client.aclose())
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(10)


def _cookie_header(team: dict[str, Any], person: str) -> dict[str, str]:
    name = team["stores"]["auth_provider"]._accounts_config.session_cookie_name
    return {"Cookie": f"{name}={team[person].cookies[name]}"}


def _point_hooks_at_the_app(
    monkeypatch: pytest.MonkeyPatch, app_loop: _AppLoop, session_id: str, posted: list[Any]
) -> None:
    """The hooks' HTTP client posts to the app; the hook code itself is unchanged."""
    from omnigent.native import native_policy_hook

    class _Client:
        def __init__(self, headers: dict[str, str] | None = None, timeout: Any = None) -> None:
            self._headers = dict(headers or {})

        def __enter__(self) -> _Client:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def post(self, url: str, json: dict[str, Any]) -> httpx.Response:
            posted.append(json)
            return app_loop.post(
                f"/v1/sessions/{session_id}/policies/evaluate", json=json, headers=self._headers
            )

    class _Httpx:
        Client = _Client

        def __getattr__(self, name: str) -> Any:
            return getattr(httpx, name)

    monkeypatch.setattr(native_policy_hook, "httpx", _Httpx())


def _relay_bridge(tmp_path: Path, session_id: str, runner: TestClient) -> Path:
    """A hook bridge directory whose relay file carries the runner's token."""
    token = runner.headers["Authorization"].removeprefix("Bearer ")
    (tmp_path / "tool_relay.json").write_text(
        json.dumps({"url": "http://relay.invalid", "token": token, "session_id": session_id})
    )
    return tmp_path


def _run_hook(
    harness: str, bridge: Path, session_id: str, payload: dict[str, Any], monkeypatch, capsys
) -> Any:
    """Run the harness's real policy hook on *payload*; return its hook output."""
    import io

    if harness == "codex":
        from omnigent.harnesses.codex_native import bridge as codex_bridge
        from omnigent.harnesses.codex_native.hook import _evaluate_policy_hook_output

        monkeypatch.setattr(codex_bridge, "read_codex_config_model", lambda _dir: None)
        return _evaluate_policy_hook_output(bridge, session_id, payload)
    from omnigent.harnesses.claude_native import hook as claude_hook

    monkeypatch.setattr(claude_hook, "read_active_session_id", lambda _dir: session_id)
    monkeypatch.setattr(claude_hook, "read_claude_status_model", lambda _dir: None)
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))
    capsys.readouterr()
    assert claude_hook._main_evaluate_policy(["--bridge-dir", str(bridge)]) == 0
    out = capsys.readouterr().out
    return json.loads(out) if out.strip() else None


def _bash(command: str) -> dict[str, Any]:
    return {
        "hook_event_name": "PreToolUse",
        "tool_name": "Bash",
        "tool_input": {"command": command},
    }


@pytest.mark.parametrize("harness", ["claude", "codex"])
def test_a_hook_denies_the_posted_call_a_transform_would_have_hidden(
    team, custom_handler, rewriting, monkeypatch, capsys, tmp_path, harness
) -> None:
    """r2-4: the harness executes the posted command, so the team rule judges it."""
    session_id, runner = rewriting
    app_loop = _AppLoop(team["alice"].app)
    try:
        _point_hooks_at_the_app(monkeypatch, app_loop, session_id, [])
        bridge = _relay_bridge(tmp_path, session_id, runner)

        output = _run_hook(harness, bridge, session_id, _bash("blocked"), monkeypatch, capsys)
    finally:
        app_loop.close()

    assert output is not None, "an ALLOW leaves the harness to run the blocked command"
    assert output["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert "blocked command" in output["hookSpecificOutput"]["permissionDecisionReason"]


@pytest.mark.parametrize("harness", ["claude", "codex"])
def test_a_hook_asks_about_the_posted_call_and_its_retry_reuses_the_answer(
    team, custom_handler, rewriting, monkeypatch, capsys, tmp_path, harness
) -> None:
    """r2-4: the approver reviews the command that runs; the hook's re-post gets its answer."""
    from omnigent.server._elicitation_registry import _harness_parked_elicitations

    session_id, runner = rewriting
    app_loop = _AppLoop(team["alice"].app)
    posted: list[Any] = []
    outputs: list[Any] = []
    try:
        _point_hooks_at_the_app(monkeypatch, app_loop, session_id, posted)
        bridge = _relay_bridge(tmp_path, session_id, runner)
        hook = threading.Thread(
            target=lambda: outputs.append(
                _run_hook(harness, bridge, session_id, _bash("review-me"), monkeypatch, capsys)
            )
        )
        hook.start()
        deadline = time.monotonic() + 15
        parked = None
        while parked is None and time.monotonic() < deadline and hook.is_alive():
            parked = next(
                (
                    eid
                    for eid, entry in list(_harness_parked_elicitations.items())
                    if entry.session_id == session_id
                ),
                None,
            )
            time.sleep(0.05)
        assert parked is not None, "the posted command was never put to the approver"
        shown = app_loop.run(
            app_loop._client.get(
                f"/v1/sessions/{session_id}/elicitations/{parked}",
                headers=_cookie_header(team, "dave"),
            )
        ).json()
        assert "review-me" in shown["content_preview"], shown
        answered = app_loop.post(
            _resolve_url(session_id, parked),
            json={"action": "accept"},
            headers=_cookie_header(team, "dave"),
        )
        assert answered.status_code == 202, answered.text
        hook.join(30)
        # The hook's retry re-posts the same question; it gets the recorded answer.
        retried = app_loop.post(
            f"/v1/sessions/{session_id}/policies/evaluate",
            json=posted[-1],
            headers={"Authorization": runner.headers["Authorization"]},
        ).json()
    finally:
        app_loop.close()

    assert outputs == [None]  # allowed: the harness runs the approved, posted command
    assert retried["result"] == "POLICY_ACTION_ALLOW", retried


def test_the_relay_gate_denies_the_posted_call(team, custom_handler, rewriting) -> None:
    """r2-4: the relay returns no transform, so the runner would run the posted command."""
    session_id, runner = rewriting

    verdict = runner.post(
        f"/v1/sessions/{session_id}/events",
        json={
            "type": "function_call",
            "data": {
                "evaluate_policy": True,
                "name": "Bash",
                "arguments": json.dumps({"command": "blocked"}),
                "agent": "review-agent",
                "call_id": "relay-probe",
            },
        },
    )

    assert verdict.status_code == 202, verdict.text
    assert verdict.json() == {"verdict": "deny", "reason": "blocked command"}
