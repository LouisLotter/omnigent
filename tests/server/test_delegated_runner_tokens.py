"""A runner's delegated token keeps its reach but cannot stand in for the person.

Accounts-mode app, real routes. The runner holds the owner JWT minted by
``POST /v1/runners/{id}/token``; the person holds their browser login.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi.testclient import TestClient
from mcp.types import CallToolResult, TextContent, Tool

from omnigent.runner.identity import RUNNER_TUNNEL_TOKEN_HEADER, token_bound_runner_id
from omnigent.server.mcp_registry import McpRegistry, McpRegistryConfig, McpService
from tests.server.helpers import build_agent_bundle
from tests.server.test_account_revocation import create_session
from tests.server.test_account_revocation import setup_app as setup_app

_CEL_POLICY = "omnigent.policies.builtins.cel.cel_policy"
_ASK_ON_TOOL_CALL = (
    'event.type == "tool_call" ? {"result": "ASK", "reason": "Post this?"} : {"result": "ALLOW"}'
)


@pytest.fixture
def tracker(setup_app: Any) -> AsyncMock:
    """A gateway MCP service whose backend records every executed call."""
    _admin, alice, _stores, _ = setup_app
    alice.app.state.mcp_registry = McpRegistry(
        McpRegistryConfig(
            services=[
                McpService(
                    id="tracker",
                    title="Tracker",
                    url="https://example.test/mcp",
                    auth="none",
                    tools=["post_update"],
                )
            ]
        ),
        None,
    )
    backend = AsyncMock()
    backend.list_tools.return_value = [Tool(name="post_update", inputSchema={"type": "object"})]
    backend.call_tool.return_value = CallToolResult(
        content=[TextContent(type="text", text="posted")]
    )
    alice.app.state.mcp_gateway_backend = backend
    return backend


def _delegated_runner(person: TestClient, stores: dict[str, Any], session_id: str) -> TestClient:
    """Bind a host runner to the session; return a client holding its minted token."""
    binding = f"binding-{uuid.uuid4().hex}"
    runner_id = token_bound_runner_id(binding)
    host_id = uuid.uuid4().hex
    owner = stores["conversation_store"].get_session_owner(session_id)
    stores["host_store"].upsert_on_connect(host_id, f"host-{host_id[:8]}", owner)
    stores["conversation_store"].set_host_id(session_id, host_id, workspace="/tmp/workspace")
    stores["conversation_store"].replace_runner_id(session_id, runner_id)
    runner = TestClient(person.app)
    runner.headers[RUNNER_TUNNEL_TOKEN_HEADER] = binding
    minted = runner.post(f"/v1/runners/{runner_id}/token")
    assert minted.status_code == 200, minted.text
    runner.headers["Authorization"] = f"Bearer {minted.json()['token']}"
    return runner


def _session_using_tracker(person: TestClient) -> str:
    """Create a session whose agent asks before every tool call, with tracker attached."""
    guardrails = {
        "policies": {
            "approve_posts": {
                "type": "function",
                "function": {"path": _CEL_POLICY, "arguments": {"expression": _ASK_ON_TOOL_CALL}},
            }
        }
    }
    created = create_session(person, guardrails=guardrails)
    assert created.status_code == 201, created.text
    session_id = created.json()["session_id"]
    attached = person.post(
        f"/v1/sessions/{session_id}/agent/mcp-servers",
        json={"name": "tracker", "transport": "registry"},
    )
    assert attached.status_code == 200, attached.text
    return session_id


def _tools_call_body(**params: Any) -> dict[str, Any]:
    """A gateway ``tools/call`` request posting "hello", with extra params."""
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": "post_update", "arguments": {"text": "hello"}, **params},
    }


def _tools_call(runner: TestClient, session_id: str, **params: Any) -> dict[str, Any]:
    """Send one gateway ``tools/call`` as the runner and return the JSON-RPC body."""
    response = runner.post(
        "/v1/mcp/tracker",
        headers={"X-Omnigent-Session-Id": session_id},
        json=_tools_call_body(**params),
    )
    assert response.status_code == 200, response.text
    return response.json()


def _ask(runner: TestClient, session_id: str) -> tuple[str, dict[str, Any]]:
    """Make the runner's first call; return the elicitation id and its retry params."""
    asked = _tools_call(runner, session_id)["result"]
    assert asked["resultType"] == "input_required", asked
    elicitation_id = next(iter(asked["inputRequests"]))
    return elicitation_id, {
        "requestState": asked["requestState"],
        "inputResponses": {elicitation_id: {"action": "accept"}},
    }


def _resolve_url(session_id: str, elicitation_id: str) -> str:
    return f"/v1/sessions/{session_id}/elicitations/{elicitation_id}/resolve"


def test_browser_approval_runs_the_runners_retry_once(setup_app: Any, tracker: AsyncMock) -> None:
    """The normal path: the person approves in the browser and the runner's retry executes."""
    _admin, alice, stores, _ = setup_app
    session_id = _session_using_tracker(alice)
    runner = _delegated_runner(alice, stores, session_id)
    elicitation_id, retry = _ask(runner, session_id)
    tracker.call_tool.assert_not_awaited()

    approved = alice.post(_resolve_url(session_id, elicitation_id), json={"action": "accept"})
    executed = _tools_call(runner, session_id, **retry)
    repeated = _tools_call(runner, session_id, **retry)

    assert approved.status_code == 202, approved.text
    assert executed["result"]["content"][0]["text"] == "posted", executed
    assert repeated["error"]["message"] == "Elicitation not found or already resolved"
    tracker.call_tool.assert_awaited_once()
    assert tracker.call_tool.await_args.args[2:] == ("post_update", {"text": "hello"})


@pytest.mark.parametrize("route", ["resolve", "approval_event"])
def test_runner_cannot_approve_its_own_call(
    setup_app: Any, tracker: AsyncMock, route: str
) -> None:
    """Self-approval fails by either route, and the runner's own accept does not count."""
    _admin, alice, stores, _ = setup_app
    session_id = _session_using_tracker(alice)
    runner = _delegated_runner(alice, stores, session_id)
    elicitation_id, retry = _ask(runner, session_id)

    if route == "resolve":
        answered = runner.post(_resolve_url(session_id, elicitation_id), json={"action": "accept"})
    else:
        verdict = {"elicitation_id": elicitation_id, "action": "accept"}
        answered = runner.post(
            f"/v1/sessions/{session_id}/events", json={"type": "approval", "data": verdict}
        )
    forged = _tools_call(runner, session_id, **retry)

    assert answered.status_code == 403, answered.text
    assert forged["error"]["message"] == "Tool call has not been approved"
    tracker.call_tool.assert_not_awaited()
    # The approval is still the person's to give.
    approved = alice.post(_resolve_url(session_id, elicitation_id), json={"action": "accept"})
    assert approved.status_code == 202, approved.text
    assert "result" in _tools_call(runner, session_id, **retry)
    tracker.call_tool.assert_awaited_once()


def test_declined_approval_never_executes(setup_app: Any, tracker: AsyncMock) -> None:
    """A person's decline stands even when the runner retries with ``accept``."""
    _admin, alice, stores, _ = setup_app
    session_id = _session_using_tracker(alice)
    runner = _delegated_runner(alice, stores, session_id)
    elicitation_id, retry = _ask(runner, session_id)

    declined = alice.post(_resolve_url(session_id, elicitation_id), json={"action": "decline"})
    refused = _tools_call(runner, session_id, **retry)

    assert declined.status_code == 202, declined.text
    assert "error" in refused
    tracker.call_tool.assert_not_awaited()


def _policy_body(name: str) -> dict[str, Any]:
    return {
        "name": name,
        "type": "python",
        "handler": _CEL_POLICY,
        "factory_params": {"expression": '{"result": "ALLOW"}'},
    }


def test_policy_changes_need_the_person(setup_app: Any) -> None:
    """Only the person adds, changes or removes policies; a runner may cap a sub-agent.

    A new policy can write session state that other policies read, so the
    runner's token cannot add one, except the sub-agent cost budget
    ``sys_session_send`` attaches, which only asks or denies.
    """
    admin, _alice, stores, _ = setup_app
    session_id = create_session(admin).json()["session_id"]
    runner = _delegated_runner(admin, stores, session_id)

    assert runner.post("/v1/policies", json=_policy_body("default-gate")).status_code == 403
    default = admin.post("/v1/policies", json=_policy_body("default-gate"))
    assert default.status_code == 200, default.text
    default_url = f"/v1/policies/{default.json()['id']}"
    assert runner.patch(default_url, json={"enabled": False}).status_code == 403
    assert runner.delete(default_url).status_code == 403
    assert admin.patch(default_url, json={"enabled": False}).status_code == 200
    assert admin.delete(default_url).status_code == 200

    session_policies = f"/v1/sessions/{session_id}/policies"
    assert runner.post(session_policies, json=_policy_body("state-writer")).status_code == 403
    added = admin.post(session_policies, json=_policy_body("person-gate"))
    assert added.status_code == 200, added.text
    session_url = f"{session_policies}/{added.json()['id']}"
    assert runner.patch(session_url, json={"enabled": False}).status_code == 403
    assert runner.delete(session_url).status_code == 403
    assert admin.patch(session_url, json={"enabled": False}).status_code == 200
    assert admin.delete(session_url).status_code == 200

    budget = runner.post(
        session_policies,
        json={
            "name": "__subagent_cost_budget",
            "type": "python",
            "handler": "omnigent.policies.builtins.cost.subagent_cost_budget",
            "factory_params": {"max_cost_usd": 1.0},
        },
    )
    assert budget.status_code == 200, budget.text
    assert runner.delete(f"{session_policies}/{budget.json()['id']}").status_code == 403


def _bundle_upload(name: str) -> dict[str, Any]:
    return {"bundle": ("agent.tar.gz", build_agent_bundle(name=name), "application/gzip")}


def test_agent_swaps_need_the_person(setup_app: Any) -> None:
    """Replacing a session's agent or changing a user agent needs the person."""
    _admin, alice, stores, _ = setup_app
    session_id = create_session(alice).json()["session_id"]
    runner = _delegated_runner(alice, stores, session_id)
    agent_url = f"/v1/sessions/{session_id}/agent"

    assert runner.put(agent_url, files=_bundle_upload("review-agent")).status_code == 403
    assert alice.put(agent_url, files=_bundle_upload("review-agent")).status_code == 200

    assert runner.post("/v1/agents", files=_bundle_upload("my-agent")).status_code == 403
    installed = alice.post("/v1/agents", files=_bundle_upload("my-agent"))
    assert installed.status_code == 200, installed.text
    user_agent_url = f"/v1/agents/{installed.json()['id']}"
    assert runner.patch(user_agent_url, json={"description": "changed"}).status_code == 403
    assert runner.delete(user_agent_url).status_code == 403
    assert alice.delete(user_agent_url).status_code == 200


@pytest.mark.parametrize(
    ("method", "path"), [("GET", "/auth/me"), ("GET", "/auth/users"), ("POST", "/auth/invite")]
)
def test_runner_token_is_not_an_account_sign_in(setup_app: Any, method: str, path: str) -> None:
    """``/auth/*`` answers an administrator's runner token as it answers an outsider."""
    admin, _alice, stores, _ = setup_app
    session_id = create_session(admin).json()["session_id"]
    runner = _delegated_runner(admin, stores, session_id)
    outsider = TestClient(admin.app)
    body = {} if method == "POST" else None

    refused = runner.request(method, path, json=body)

    assert refused.status_code in (401, 403), refused.text
    assert refused.status_code == outsider.request(method, path, json=body).status_code
    assert admin.request(method, path, json=body).status_code == 200


def test_runner_ordinary_traffic_is_unaffected(setup_app: Any, tracker: AsyncMock) -> None:
    """Events, reads, sub-sessions, MCP discovery and re-minting keep working."""
    _admin, alice, stores, _ = setup_app
    session_id = _session_using_tracker(alice)
    runner = _delegated_runner(alice, stores, session_id)

    status = runner.post(
        f"/v1/sessions/{session_id}/events",
        json={"type": "external_session_status", "data": {"status": "idle"}},
    )
    message = runner.post(
        f"/v1/sessions/{session_id}/events",
        json={
            "type": "external_conversation_item",
            "data": {
                "item_type": "message",
                "item_data": {
                    "role": "assistant",
                    "agent": "review-agent",
                    "content": [{"type": "output_text", "text": "done"}],
                },
                "response_id": "resp_runner_1",
            },
        },
    )
    listed = runner.post(
        "/v1/mcp/tracker",
        headers={"X-Omnigent-Session-Id": session_id},
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
    )
    child = create_session(runner)
    runner_id = stores["conversation_store"].get_conversation(session_id).runner_id

    assert status.status_code == 202, status.text
    assert message.status_code == 202, message.text
    assert runner.get(f"/v1/sessions/{session_id}").status_code == 200
    assert [tool["name"] for tool in listed.json()["result"]["tools"]] == ["post_update"]
    assert child.status_code == 201, child.text
    assert runner.post(f"/v1/runners/{runner_id}/token").status_code == 200


def _evaluate_body(elicitation_id: str, text: str) -> dict[str, Any]:
    """Body of the native policy gate a Pi bridged tool waits on before its retry."""
    return {
        "event": {
            "type": "PHASE_TOOL_CALL",
            "target": "",
            "data": {"name": "tracker__post_update", "arguments": {"text": text}},
            "context": {},
        },
        "_omnigent_elicitation_id": elicitation_id,
    }


def _evaluate_tool_call(client: TestClient, session_id: str, elicitation_id: str, text: str):
    """POST the native policy gate as the runner."""
    return client.post(
        f"/v1/sessions/{session_id}/policies/evaluate",
        json=_evaluate_body(elicitation_id, text),
    )


def test_native_gate_reports_an_mcp_approval_already_answered(
    setup_app: Any, tracker: AsyncMock
) -> None:
    """Pi's gate names the /mcp approval itself; an earlier answer is reported, not re-asked."""
    _admin, alice, stores, _ = setup_app
    session_id = _session_using_tracker(alice)
    runner = _delegated_runner(alice, stores, session_id)
    elicitation_id, retry = _ask(runner, session_id)

    alice.post(_resolve_url(session_id, elicitation_id), json={"action": "accept"})
    gate = _evaluate_tool_call(runner, session_id, elicitation_id, "hello")

    assert gate.json()["result"] == "POLICY_ACTION_ALLOW", gate.text
    assert "result" in _tools_call(runner, session_id, **retry)
    assert "error" in _tools_call(runner, session_id, **retry)
    tracker.call_tool.assert_awaited_once()


@pytest.mark.asyncio
async def test_native_gate_waits_on_the_mcp_approval_itself(
    setup_app: Any, tracker: AsyncMock
) -> None:
    """While Pi's gate waits on the /mcp approval, one browser answer releases one call.

    Both requests share the server's event loop, as in production.
    """
    from omnigent.server._elicitation_registry import _harness_parked_elicitations

    _admin, alice, stores, _ = setup_app
    session_id = _session_using_tracker(alice)
    runner = _delegated_runner(alice, stores, session_id)
    elicitation_id, retry = _ask(runner, session_id)
    transport = httpx.ASGITransport(app=alice.app)

    async with (
        httpx.AsyncClient(
            transport=transport, base_url="http://testserver", headers=dict(runner.headers)
        ) as runner_http,
        httpx.AsyncClient(
            transport=transport, base_url="http://testserver", cookies=alice.cookies
        ) as person_http,
    ):
        waiting = asyncio.create_task(
            runner_http.post(
                f"/v1/sessions/{session_id}/policies/evaluate",
                json=_evaluate_body(elicitation_id, "hello"),
            )
        )
        async with asyncio.timeout(15):
            while elicitation_id not in _harness_parked_elicitations:
                await asyncio.sleep(0.05)
        approved = await person_http.post(
            _resolve_url(session_id, elicitation_id), json={"action": "accept"}
        )
        gate = await asyncio.wait_for(waiting, timeout=15)
        executed = await runner_http.post(
            "/v1/mcp/tracker",
            headers={"X-Omnigent-Session-Id": session_id},
            json=_tools_call_body(**retry),
        )

    assert approved.status_code == 202, approved.text
    assert gate.json()["result"] == "POLICY_ACTION_ALLOW", gate.text
    assert executed.json()["result"]["content"][0]["text"] == "posted", executed.text
    tracker.call_tool.assert_awaited_once()


def test_native_gate_cannot_borrow_another_calls_approval(
    setup_app: Any, tracker: AsyncMock
) -> None:
    """A gate naming an MCP approval must describe that exact call."""
    _admin, alice, stores, _ = setup_app
    session_id = _session_using_tracker(alice)
    runner = _delegated_runner(alice, stores, session_id)
    elicitation_id, _retry = _ask(runner, session_id)

    other_arguments = _evaluate_tool_call(runner, session_id, elicitation_id, "something else")
    unknown = _evaluate_tool_call(runner, session_id, "elicit_" + "0" * 32, "hello")

    assert other_arguments.status_code == 400, other_arguments.text
    assert unknown.status_code == 400, unknown.text
