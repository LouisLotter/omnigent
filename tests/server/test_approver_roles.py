"""Approver roles and exact verdict binding through the real routes (T9, T10).

Accounts-mode app. Alice owns a thread whose agent asks before every tracker
call; her runner holds a runner token. Bob is a colleague, Carol and Dave are
members an administrator can name as approvers, and ``admin`` holds the
administrator flag. People are signed in through the browser (the test
client keeps the session cookie the sign-in set).
"""

from __future__ import annotations

import asyncio
import dataclasses
import uuid
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest
import sqlalchemy as sa
from fastapi.testclient import TestClient
from starlette.datastructures import Headers

from omnigent.db.db_models import SqlApprovalVerdict
from omnigent.runner.transports.ws_tunnel.frames import EventBatchFrame
from omnigent.server.auth import LEVEL_EDIT
from omnigent.server.mcp_registry import McpRegistry, McpRegistryConfig, McpService
from omnigent.server.routes._sessions import orchestration
from omnigent.server.routes.device_auth import LOGIN_GRANT_CLIENT_ID, mint_delegated_token
from omnigent.stores.approval_store import VerdictState
from tests.server.test_account_revocation import setup_app as setup_app
from tests.server.test_delegated_runner_tokens import (
    _ask,
    _delegated_runner,
    _resolve_url,
    _session_using_tracker,
    _tools_call,
    _tools_call_body,
)
from tests.server.test_delegated_runner_tokens import tracker as tracker

_CEL = "omnigent.policies.builtins.cel.cel_policy"
_ASK_ON_POST = (
    'event.type == "tool_call" && event.target == "tracker__post_update"'
    ' ? {"result": "ASK", "reason": "review"} : {"result": "ALLOW"}'
)


def _member(admin: TestClient, username: str) -> TestClient:
    invite = admin.post("/auth/invite", json={}).json()["token"]
    person = TestClient(admin.app)
    registered = person.post(
        "/auth/register",
        json={"invite": invite, "username": username, "password": f"{username}-pw-12345"},
    )
    assert registered.status_code == 200, registered.text
    return person


@pytest.fixture
def team(setup_app: Any, tracker: AsyncMock) -> dict[str, Any]:
    admin, alice, stores, _ = setup_app
    people = {name: _member(admin, name) for name in ("bob", "carol", "dave")}
    session_id = _session_using_tracker(alice)
    return {
        "admin": admin,
        "alice": alice,
        **people,
        "stores": stores,
        "tracker": tracker,
        "session": session_id,
        "runner": _delegated_runner(alice, stores, session_id),
        "approvals": alice.app.state.approval_store,
    }


def _answer(person: TestClient, session_id: str, elicitation_id: str, action: str = "accept"):
    return person.post(_resolve_url(session_id, elicitation_id), json={"action": action})


def _name_approver(team: dict[str, Any], kind: str, user: str, resource_id: str = "") -> None:
    named = team["admin"].post(
        "/v1/approvers", json={"kind": kind, "resource_id": resource_id, "user_id": user}
    )
    assert named.status_code == 201, named.text


def _bearer(app: Any, token: str, *, as_cookie: str | None = None) -> TestClient:
    client = TestClient(app)
    if as_cookie is None:
        client.headers["Authorization"] = f"Bearer {token}"
    else:
        client.cookies.set(as_cookie, token)
    return client


def _cookie_name(team: dict[str, Any]) -> str:
    return team["stores"]["auth_provider"]._accounts_config.session_cookie_name


def _cli_token(team: dict[str, Any]) -> str:
    """The token Alice's CLI receives and stores when it signs in."""
    login = TestClient(team["alice"].app).post(
        "/auth/login", json={"username": "alice", "password": "alice-pw-1234"}
    )
    assert login.status_code == 200, login.text
    return login.json()["token"]


def _grant_token(team: dict[str, Any], *, device: bool) -> str:
    """An access token from Alice's device grant, or from her CLI's login grant."""
    config = team["stores"]["auth_provider"]._accounts_config
    generation = team["stores"]["account_store"].get_user("alice").account_generation
    kwargs: dict[str, Any] = {} if device else {"scope": None}
    return mint_delegated_token(
        "alice",
        config.cookie_secret,
        600,
        "accounts",
        account_generation=generation,
        grant_id=f"grant-{uuid.uuid4().hex}",
        client_id="slack" if device else LOGIN_GRANT_CLIENT_ID,
        jti=uuid.uuid4().hex,
        **kwargs,
    )


def _state(team: dict[str, Any], elicitation_id: str) -> VerdictState:
    record = team["approvals"].get_verdict(elicitation_id)
    assert record is not None
    return record.state


# ── Who answers (T9) ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "credential",
    ["runner", "runner_event", "device_grant", "login_grant", "cli_bearer", "cli_as_cookie"],
)
def test_only_a_browser_sign_in_answers_a_protected_approval(team, credential) -> None:
    """Runners, device grants and CLI or host sign-ins never answer; the browser does, once."""
    session_id, runner = team["session"], team["runner"]
    elicitation_id, retry = _ask(runner, session_id)
    app = team["alice"].app
    if credential == "runner":
        answered = _answer(runner, session_id, elicitation_id)
    elif credential == "runner_event":
        answered = runner.post(
            f"/v1/sessions/{session_id}/events",
            json={
                "type": "approval",
                "data": {"elicitation_id": elicitation_id, "action": "accept"},
            },
        )
    elif credential in ("device_grant", "login_grant"):
        token = _grant_token(team, device=credential == "device_grant")
        answered = _answer(_bearer(app, token), session_id, elicitation_id)
    else:
        cookie = _cookie_name(team) if credential == "cli_as_cookie" else None
        answered = _answer(
            _bearer(app, _cli_token(team), as_cookie=cookie), session_id, elicitation_id
        )

    assert answered.status_code == 403, answered.text
    assert _state(team, elicitation_id) is VerdictState.PENDING
    assert _tools_call(runner, session_id, **retry)["error"]["message"] == (
        "Tool call has not been approved"
    )
    if credential == "runner_event":
        # The person's own answer through the same session-event route counts.
        answered = team["alice"].post(
            f"/v1/sessions/{session_id}/events",
            json={
                "type": "approval",
                "data": {"elicitation_id": elicitation_id, "action": "accept"},
            },
        )
    else:
        answered = _answer(team["alice"], session_id, elicitation_id)
    assert answered.status_code == 202, answered.text
    assert "result" in _tools_call(runner, session_id, **retry)
    team["tracker"].call_tool.assert_awaited_once()


def test_a_tunnelled_answer_is_refused(team) -> None:
    """An approval sent over the runner tunnel is not a person's answer."""
    session_id = team["session"]
    elicitation_id, _retry = _ask(team["runner"], session_id)
    ingest = team["alice"].app.state.runner_event_ingest
    conversation = team["stores"]["conversation_store"].get_conversation(session_id)

    ack = asyncio.run(
        ingest(
            app=team["alice"].app,
            headers=Headers({}),
            owner="alice",
            runner_id=conversation.runner_id,
            batch=EventBatchFrame(
                id="b1",
                session_id=session_id,
                events=[
                    {
                        "type": "approval",
                        "data": {"elicitation_id": elicitation_id, "action": "accept"},
                    }
                ],
            ),
        )
    )

    assert ack.applied == 0 and ack.error
    assert _state(team, elicitation_id) is VerdictState.PENDING


def test_an_approval_is_answered_only_on_the_thread_that_asked(team) -> None:
    """Posting another thread's approval id through one's own thread changes nothing."""
    from tests.server.test_account_revocation import create_session

    session_id = team["session"]
    elicitation_id, retry = _ask(team["runner"], session_id)
    other = create_session(team["alice"]).json()["session_id"]

    for route in ("resolve", "event"):
        if route == "resolve":
            answered = _answer(team["alice"], other, elicitation_id)
        else:
            answered = team["alice"].post(
                f"/v1/sessions/{other}/events",
                json={
                    "type": "approval",
                    "data": {"elicitation_id": elicitation_id, "action": "accept"},
                },
            )
        assert answered.status_code == 403, answered.text
    assert _state(team, elicitation_id) is VerdictState.PENDING
    assert "error" in _tools_call(team["runner"], session_id, **retry)


def test_an_administrator_or_editor_is_not_an_approver_unless_named(team) -> None:
    """D52/D53: thread access is not approval; naming someone makes them an approver."""
    session_id = team["session"]
    shared = team["alice"].put(
        f"/v1/sessions/{session_id}/permissions", json={"user_id": "bob", "level": LEVEL_EDIT}
    )
    assert shared.status_code in (200, 201), shared.text
    elicitation_id, retry = _ask(team["runner"], session_id)

    assert _answer(team["admin"], session_id, elicitation_id).status_code == 403
    assert _answer(team["bob"], session_id, elicitation_id).status_code == 403
    assert _answer(team["carol"], session_id, elicitation_id).status_code == 404

    # Approvers named after the ask change who may answer: it must be asked again.
    _name_approver(team, "session", "bob", session_id)
    assert _answer(team["bob"], session_id, elicitation_id).status_code == 409
    second, retry = _ask(team["runner"], session_id)
    assert _answer(team["bob"], session_id, second).status_code == 202
    assert "result" in _tools_call(team["runner"], session_id, **retry)


def test_named_service_approvers_answer_and_the_initiator_does_not_by_default(team) -> None:
    """D53: a team-shared service's actions go to its approvers, not to their initiator."""
    session_id, runner = team["session"], team["runner"]
    _name_approver(team, "service", "carol", "tracker")
    _name_approver(team, "service", "alice", "tracker")
    elicitation_id, retry = _ask(runner, session_id)

    refused = _answer(team["alice"], session_id, elicitation_id)
    assert refused.status_code == 403, refused.text
    # Carol cannot see Alice's thread, yet sees and answers the approval.
    viewed = team["carol"].get(f"/v1/sessions/{session_id}/elicitations/{elicitation_id}")
    assert viewed.json()["status"] == "pending", viewed.text
    listed = team["carol"].get("/v1/approvals").json()["data"]
    assert [item["elicitation_id"] for item in listed] == [elicitation_id]
    assert _answer(team["carol"], session_id, elicitation_id).status_code == 202
    assert "result" in _tools_call(runner, session_id, **retry)

    saved = team["admin"].put(
        "/v1/approval-settings",
        json={"kind": "service", "resource_id": "tracker", "allow_initiator": True},
    )
    assert saved.json() == {"allow_initiator": True}
    second, retry = _ask(runner, session_id)
    assert _answer(team["alice"], session_id, second).status_code == 202
    assert "result" in _tools_call(runner, session_id, **retry)


def test_a_team_rule_and_a_personal_rule_each_need_their_approver(team) -> None:
    """All required approvals survive: Alice's own rule needs Alice, the team rule needs Dave."""
    session_id, runner = team["session"], team["runner"]
    _name_approver(team, "team", "dave")
    for client, domain in ((team["admin"], "team"), (team["alice"], "personal")):
        created = client.post(
            "/v1/protected-policies",
            json={
                "domain": domain,
                "name": f"{domain}-review",
                "handler": _CEL,
                "factory_params": {"expression": _ASK_ON_POST},
            },
        )
        assert created.status_code == 201, created.text
    elicitation_id, retry = _ask(runner, session_id)

    first = _answer(team["alice"], session_id, elicitation_id)
    assert first.json() == {"queued": False, "awaiting": "team"}, first.text
    assert _tools_call(runner, session_id, **retry)["error"]["message"] == (
        "Tool call has not been approved"
    )
    assert _answer(team["dave"], session_id, elicitation_id).json() == {"queued": False}
    assert "result" in _tools_call(runner, session_id, **retry)
    team["tracker"].call_tool.assert_awaited_once()


def test_any_eligible_decline_declines(team) -> None:
    session_id, runner = team["session"], team["runner"]
    _name_approver(team, "service", "carol", "tracker")
    elicitation_id, retry = _ask(runner, session_id)

    assert _answer(team["carol"], session_id, elicitation_id, "decline").status_code == 202
    assert _state(team, elicitation_id) is VerdictState.DECLINED
    assert "error" in _tools_call(runner, session_id, **retry)
    assert _answer(team["carol"], session_id, elicitation_id).status_code == 409
    team["tracker"].call_tool.assert_not_awaited()


# ── What an approval is bound to (T10) ───────────────────────────────────


def test_a_removed_approver_or_a_wrong_resource_cannot_answer(team) -> None:
    session_id, runner = team["session"], team["runner"]
    _name_approver(team, "service", "carol", "tracker")
    _name_approver(team, "service", "dave", "another-service")
    elicitation_id, _retry = _ask(runner, session_id)

    removed = team["admin"].delete(
        "/v1/approvers", params={"kind": "service", "resource_id": "tracker", "user_id": "carol"}
    )
    assert removed.json() == {"removed": True}
    assert _answer(team["carol"], session_id, elicitation_id).status_code == 404
    assert _answer(team["dave"], session_id, elicitation_id).status_code == 404
    assert _state(team, elicitation_id) is VerdictState.PENDING


def test_changed_approvers_after_approval_need_a_new_decision(team) -> None:
    session_id, runner = team["session"], team["runner"]
    _name_approver(team, "service", "carol", "tracker")
    elicitation_id, retry = _ask(runner, session_id)
    assert _answer(team["carol"], session_id, elicitation_id).status_code == 202

    team["admin"].delete(
        "/v1/approvers", params={"kind": "service", "resource_id": "tracker", "user_id": "carol"}
    )

    refused = _tools_call(runner, session_id, **retry)
    assert "ask again" in refused["error"]["message"], refused
    team["tracker"].call_tool.assert_not_awaited()


@pytest.mark.parametrize("change", ["arguments", "tool", "protected_rule", "later_transform"])
def test_a_changed_action_or_policy_needs_a_new_decision(team, change, monkeypatch) -> None:
    """T9/T10: changed content, a swapped tool, a changed rule or a later transform."""
    session_id, runner = team["session"], team["runner"]
    rule = team["admin"].post(
        "/v1/protected-policies",
        json={
            "domain": "team",
            "name": "audit",
            "handler": _CEL,
            "factory_params": {"expression": '{"result": "ALLOW"}'},
        },
    )
    assert rule.status_code == 201, rule.text
    elicitation_id, retry = _ask(runner, session_id)
    assert _answer(team["alice"], session_id, elicitation_id).status_code == 202
    params: dict[str, Any] = dict(retry)
    if change == "arguments":
        params["arguments"] = {"text": "something else"}
    elif change == "tool":
        params["name"] = "delete_everything"
    elif change == "protected_rule":
        team["admin"].patch(
            f"/v1/protected-policies/{rule.json()['id']}",
            json={"factory_params": {"expression": '{"result": "ALLOW", "reason": "v2"}'}},
        )
    else:
        # A policy added after the approval rewrites what would be sent.
        real = orchestration._evaluate_policy_with_fresh_engine

        async def rewritten(session_id, spec, store, conv, ctx):
            engine, result = await real(session_id, spec, store, conv, ctx)
            changed = {**ctx.content["arguments"], "text": "rewritten"}
            return engine, dataclasses.replace(result, data=changed)

        monkeypatch.setattr(orchestration, "_evaluate_policy_with_fresh_engine", rewritten)

    refused = _tools_call(runner, session_id, **params)

    assert "error" in refused, refused
    if change in ("protected_rule", "later_transform"):
        assert refused["error"]["message"].endswith("ask again"), refused
    team["tracker"].call_tool.assert_not_awaited()


def test_an_expired_verdict_is_never_used(team) -> None:
    session_id, runner = team["session"], team["runner"]
    elicitation_id, retry = _ask(runner, session_id)
    assert _answer(team["alice"], session_id, elicitation_id).status_code == 202
    with team["approvals"]._engine.begin() as connection:
        connection.execute(
            sa.update(SqlApprovalVerdict)
            .where(SqlApprovalVerdict.elicitation_id == elicitation_id)
            .values(expires_at=0)
        )

    refused = _tools_call(runner, session_id, **retry)

    assert refused["error"]["message"] == "Approval expired; ask again"
    team["tracker"].call_tool.assert_not_awaited()


def test_a_stale_connection_generation_needs_a_new_decision(team) -> None:
    """The service's destination changed after approval: the approved call never runs."""
    session_id, runner = team["session"], team["runner"]
    elicitation_id, retry = _ask(runner, session_id)
    assert _answer(team["alice"], session_id, elicitation_id).status_code == 202
    team["alice"].app.state.mcp_registry = McpRegistry(
        McpRegistryConfig(
            services=[
                McpService(
                    id="tracker",
                    title="Tracker",
                    url="https://elsewhere.example.test/mcp",
                    auth="none",
                    tools=["post_update"],
                )
            ]
        ),
        None,
    )

    refused = _tools_call(runner, session_id, **retry)

    assert "ask again" in refused["error"]["message"], refused
    team["tracker"].call_tool.assert_not_awaited()


def test_parallel_retries_run_the_approved_call_once(team) -> None:
    session_id, runner = team["session"], team["runner"]
    elicitation_id, retry = _ask(runner, session_id)
    assert _answer(team["alice"], session_id, elicitation_id).status_code == 202

    async def race() -> list[dict[str, Any]]:
        transport = httpx.ASGITransport(app=team["alice"].app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver", headers=dict(runner.headers)
        ) as client:
            responses = await asyncio.gather(
                *(
                    client.post(
                        "/v1/mcp/tracker",
                        headers={"X-Omnigent-Session-Id": session_id},
                        json=_tools_call_body(**retry),
                    )
                    for _ in range(4)
                )
            )
        return [response.json() for response in responses]

    results = asyncio.run(race())

    assert sum("result" in result for result in results) == 1, results
    team["tracker"].call_tool.assert_awaited_once()


def test_a_retry_of_the_consuming_operation_gets_its_result(team) -> None:
    """Retries inspect the operation's recorded result instead of using the approval again."""
    from omnigent.runner.mcp_execution_registry import MCP_OPERATION_ID_PARAM

    session_id, runner = team["session"], team["runner"]
    operation = {MCP_OPERATION_ID_PARAM: f"mcpop_{uuid.uuid4().hex}"}
    asked = _tools_call(runner, session_id, **operation)["result"]
    elicitation_id = next(iter(asked["inputRequests"]))
    retry = {
        **operation,
        "requestState": asked["requestState"],
        "inputResponses": {elicitation_id: {"action": "accept"}},
    }
    assert _answer(team["alice"], session_id, elicitation_id).status_code == 202

    first = _tools_call(runner, session_id, **retry)
    again = _tools_call(runner, session_id, **retry)

    assert first["result"] == again["result"]
    team["tracker"].call_tool.assert_awaited_once()


def test_ordinary_questions_keep_their_rules(team) -> None:
    """A question that is not a protected approval: an editor or Alice's CLI may answer it."""
    session_id, runner = team["session"], team["runner"]
    shared = team["alice"].put(
        f"/v1/sessions/{session_id}/permissions", json={"user_id": "bob", "level": LEVEL_EDIT}
    )
    assert shared.status_code in (200, 201), shared.text
    cli = _bearer(team["alice"].app, _cli_token(team))
    for answerer in (team["bob"], cli):
        asked = runner.post(
            f"/v1/sessions/{session_id}/events",
            json={"type": "mcp_elicitation", "data": {"message": "Which branch?"}},
        )
        question = asked.json()["elicitation_id"]
        answered = answerer.post(
            f"/v1/sessions/{session_id}/events",
            json={
                "type": "approval",
                "data": {"elicitation_id": question, "action": "accept", "content": {"b": "main"}},
            },
        )
        assert answered.status_code == 202, answered.text
        assert team["approvals"].get_verdict(question) is None


# ── Who manages approvals ────────────────────────────────────────────────


def test_only_human_administrators_manage_approvers_and_team_rules(team) -> None:
    body = {"kind": "service", "resource_id": "tracker", "user_id": "carol"}
    assert team["bob"].post("/v1/approvers", json=body).status_code == 403
    assert team["runner"].post("/v1/approvers", json=body).status_code == 403
    admin_cli = TestClient(team["admin"].app)
    login = admin_cli.post("/auth/login", json={"username": "admin", "password": "admin-pw-12345"})
    admin_bearer = _bearer(team["admin"].app, login.json()["token"])
    assert admin_bearer.post("/v1/approvers", json=body).status_code == 403
    assert team["admin"].post("/v1/approvers", json=body).status_code == 201
    listed = team["admin"].get(
        "/v1/approvers", params={"kind": "service", "resource_id": "tracker"}
    )
    assert listed.json()["approvers"] == ["carol"]

    rule = {"domain": "team", "name": "r", "handler": _CEL, "factory_params": {"expression": "{}"}}
    assert team["bob"].post("/v1/protected-policies", json=rule).status_code == 403
    unregistered = {**rule, "handler": "os.system"}
    assert team["admin"].post("/v1/protected-policies", json=unregistered).status_code == 400


def test_personal_rules_belong_to_their_person(team) -> None:
    rule = {
        "domain": "personal",
        "name": "mine",
        "handler": _CEL,
        "factory_params": {"expression": "{}"},
    }
    created = team["alice"].post("/v1/protected-policies", json=rule)
    assert created.status_code == 201, created.text
    rule_id = created.json()["id"]
    assert created.json()["owner"] == "alice"

    assert (
        team["bob"].patch(f"/v1/protected-policies/{rule_id}", json={"enabled": False}).status_code
        == 404
    )
    assert team["admin"].delete(f"/v1/protected-policies/{rule_id}").status_code == 404
    assert team["runner"].delete(f"/v1/protected-policies/{rule_id}").status_code == 403
    mine = team["alice"].get("/v1/protected-policies", params={"domain": "personal"}).json()
    assert [item["id"] for item in mine["data"]] == [rule_id]
    assert (
        team["bob"].get("/v1/protected-policies", params={"domain": "personal"}).json()["data"]
        == []
    )
    assert team["alice"].delete(f"/v1/protected-policies/{rule_id}").json() == {"deleted": True}


# ── Server-held gates ────────────────────────────────────────────────────


@pytest.mark.parametrize("rule_changes", [False, True])
def test_native_gate_needs_a_browser_answer_still_matching_its_action(team, rule_changes) -> None:
    """The native hook gate: a runner's answer is refused; a stale approval is not used."""
    from omnigent.server._elicitation_registry import _harness_parked_elicitations

    session_id, runner = team["session"], team["runner"]
    rule = (
        team["admin"]
        .post(
            "/v1/protected-policies",
            json={
                "domain": "team",
                "name": "audit",
                "handler": _CEL,
                "factory_params": {"expression": '{"result": "ALLOW"}'},
            },
        )
        .json()
    )
    elicitation_id = f"elicit_evaluate_{uuid.uuid4().hex}"
    body = {
        "event": {
            "type": "PHASE_TOOL_CALL",
            "target": "",
            "data": {"name": "Bash", "arguments": {"command": "ls"}},
            "context": {},
        },
        "_omnigent_elicitation_id": elicitation_id,
    }

    async def flow() -> tuple[int, int, dict[str, Any]]:
        transport = httpx.ASGITransport(app=team["alice"].app)
        async with (
            httpx.AsyncClient(
                transport=transport, base_url="http://testserver", headers=dict(runner.headers)
            ) as runner_http,
            httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
                cookies=team["alice"].cookies,
            ) as person_http,
        ):
            waiting = asyncio.create_task(
                runner_http.post(f"/v1/sessions/{session_id}/policies/evaluate", json=body)
            )
            async with asyncio.timeout(15):
                while elicitation_id not in _harness_parked_elicitations:
                    await asyncio.sleep(0.05)
            forged = await runner_http.post(
                _resolve_url(session_id, elicitation_id), json={"action": "accept"}
            )
            approved = await person_http.post(
                _resolve_url(session_id, elicitation_id), json={"action": "accept"}
            )
            gate = await asyncio.wait_for(waiting, timeout=15)
        return forged.status_code, approved.status_code, gate.json()

    if rule_changes:
        # The rule changes between the answer and its use: the gate refuses it.
        real = team["approvals"].record_answer

        def answer_then_change(*args: Any, **kwargs: Any) -> Any:
            answered = real(*args, **kwargs)
            team["approvals"].update_rule(rule["id"], name="audit-v2")
            return answered

        team["approvals"].record_answer = answer_then_change  # type: ignore[method-assign]
    forged, approved, gate = asyncio.run(flow())

    assert forged == 403
    assert approved == 202
    assert gate["result"] == ("POLICY_ACTION_DENY" if rule_changes else "POLICY_ACTION_ALLOW"), (
        gate
    )
