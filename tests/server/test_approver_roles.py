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
import time
import uuid
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest
import sqlalchemy as sa
from fastapi.testclient import TestClient
from starlette.datastructures import Headers

from omnigent.db.db_models import SqlApprovalVerdict
from omnigent.entities import MessageData, NewConversationItem
from omnigent.entities.attribution import (
    InitiatorAttribution,
    OwnerAttribution,
    TurnAttribution,
)
from omnigent.runner.transports.ws_tunnel.frames import EventBatchFrame
from omnigent.runtime import get_turn_store
from omnigent.server.auth import LEVEL_EDIT, LEVEL_READ
from omnigent.server.mcp_registry import McpRegistry, McpRegistryConfig, McpService
from omnigent.server.routes._sessions import orchestration
from omnigent.server.routes.device_auth import LOGIN_GRANT_CLIENT_ID, mint_delegated_token
from omnigent.stores.approval_store import ApprovalResource, ResourceKind, VerdictState
from omnigent.stores.turn_store import InputKind
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
    """Name an approver as whoever manages the resource: Alice for her thread, else the admin."""
    manager = team["alice"] if kind == "session" else team["admin"]
    named = manager.post(
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
    """An access token from Alice's stored device grant, or from her CLI's login grant."""
    from omnigent.server.device_grant_store import DeviceGrantStore

    stores = team["stores"]
    config = stores["auth_provider"]._accounts_config
    generation = stores["account_store"].get_user("alice").account_generation
    grant_id = f"grant-{uuid.uuid4().hex}"
    client_id = "slack" if device else LOGIN_GRANT_CLIENT_ID
    DeviceGrantStore(stores["permission_store"].storage_location).create_redeemed_grant(
        grant_id,
        user_id="alice",
        client_id=client_id,
        refresh_token_hash=uuid.uuid4().hex,
        created_at=int(time.time()),
    )
    kwargs: dict[str, Any] = {} if device else {"scope": None}
    return mint_delegated_token(
        "alice",
        config.cookie_secret,
        600,
        "accounts",
        account_generation=generation,
        grant_id=grant_id,
        client_id=client_id,
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
    else:
        if credential in ("device_grant", "login_grant"):
            token = _grant_token(team, device=credential == "device_grant")
            client = _bearer(app, token)
        else:
            cookie = _cookie_name(team) if credential == "cli_as_cookie" else None
            client = _bearer(app, _cli_token(team), as_cookie=cookie)
        # The credential is a valid sign-in for Alice: it reads her thread.
        assert client.get(f"/v1/sessions/{session_id}").status_code == 200
        answered = _answer(client, session_id, elicitation_id)

    assert answered.status_code == 403, answered.text
    message = answered.json()["error"]["message"]
    if credential == "runner":
        # The route is for people only; the guard refuses a runner before the gate.
        assert answered.json()["error"]["code"] == "machine_principal_forbidden", message
    elif credential == "cli_as_cookie":
        # Indistinguishable from a browser session that predates the mark.
        assert message.startswith("Sign in again in the browser to answer approvals"), message
    else:
        assert "Answer approvals in the Omnigent web page instead." in message, message
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

    first = _answer(team["alice"], session_id, elicitation_id).json()
    assert (first["queued"], first["awaiting"]) == (False, "team"), first
    assert _tools_call(runner, session_id, **retry)["error"]["message"] == (
        "Tool call has not been approved"
    )
    final = _answer(team["dave"], session_id, elicitation_id).json()
    assert final["queued"] is False
    # Each answer records the grant it was given under.
    roles = [(answer["by"], answer["as"][0]["role"]) for answer in final["approval"]["answers"]]
    assert roles == [("alice", "thread_owner"), ("dave", "team_approver")]
    assert final["approval"]["decided_by"] == "dave"
    assert "result" in _tools_call(runner, session_id, **retry)
    team["tracker"].call_tool.assert_awaited_once()


def _turn(team: dict[str, Any], person: str, text: str) -> Any:
    """A message as the forward path saves and records it, before any runner sees it."""
    session_id = team["session"]
    [item] = team["stores"]["conversation_store"].append(
        session_id,
        [
            NewConversationItem(
                type="message",
                response_id=f"turn_{uuid.uuid4().hex}",
                data=MessageData(role="user", content=[{"type": "input_text", "text": text}]),
                created_by=person,
                attribution=TurnAttribution(
                    initiator=InitiatorAttribution(kind="person", user=person, source="signed_in"),
                    owner=OwnerAttribution(user="alice"),
                    recorded_at=int(time.time()),
                ),
            )
        ],
    )
    get_turn_store().record_input(session_id, item.id, kind=InputKind.PERSON, person=person)
    return item


def _register(team: dict[str, Any], *input_ids: str) -> Any:
    """The bound runner registers a turn before its harness acts."""
    registered = team["runner"].post(
        f"/v1/sessions/{team['session']}/turns",
        json={"turn_id": f"turn_{uuid.uuid4().hex}", "input_ids": list(input_ids)},
    )
    assert registered.status_code == 200, registered.text
    return registered.json()


def _service_approvers(team: dict[str, Any], *names: str) -> None:
    for approver in names:
        _name_approver(team, "service", approver, "tracker")


def _refused(team: dict[str, Any], person: str, elicitation_id: str) -> str:
    """The answer is refused; one who cannot read the thread is told it does not exist."""
    refused = _answer(team[person], team["session"], elicitation_id)
    assert refused.status_code in (403, 404, 409), refused.text
    assert _state(team, elicitation_id) is VerdictState.PENDING
    return refused.json()["error"]["message"]


def test_a_message_queued_behind_a_turn_lets_nobody_in_it_approve_its_actions(team) -> None:
    """D53 + WP8.8: Bob queues during Alice's turn; the approval names Alice, and neither
    Alice nor Bob may approve it as a service approver where self-approval is off."""
    session_id, runner = team["session"], team["runner"]
    team["stores"]["permission_store"].grant("bob", session_id, LEVEL_EDIT)
    _service_approvers(team, "alice", "bob", "carol")
    alices = _turn(team, "alice", "post the update")
    assert _register(team, alices.id)["initiator"] == "alice"
    # Queued: the runner has not started Bob's turn, so it is still Alice's.
    _turn(team, "bob", "and mention the outage")

    elicitation_id, retry = _ask(runner, session_id)

    record = team["approvals"].get_verdict(elicitation_id)
    assert record.binding["initiator"] == "alice"
    assert (record.attribution["initiator"], record.attribution["contributors"]) == (
        "alice",
        ["bob"],
    )
    assert record.attribution["authority"]["complete"] is True
    assert record.attribution["principal"]["kind"] == "runner"
    assert record.attribution["owner"]["user"] == "alice"
    for person in ("alice", "bob"):
        _refused(team, person, elicitation_id)
    approved = _answer(team["carol"], session_id, elicitation_id)
    assert approved.status_code == 202, approved.text
    [answer] = approved.json()["approval"]["answers"]
    assert (answer["by"], answer["as"][0]["role"]) == ("carol", "service_approver")
    assert "result" in _tools_call(runner, session_id, **retry)

    # The thread's approval record: anyone who reads the thread; nobody else; no runner.
    [listed] = team["bob"].get(f"/v1/sessions/{session_id}/approvals").json()["data"]
    assert listed["elicitation_id"] == elicitation_id
    assert listed["asked"]["contributors"] == ["bob"]
    assert listed["decided_by"] == "carol"
    assert listed["decision_digest"] == record.binding_digest.hex()
    assert team["dave"].get(f"/v1/sessions/{session_id}/approvals").status_code == 404
    assert runner.get(f"/v1/sessions/{session_id}/approvals").status_code == 403


def test_a_long_turn_is_read_completely_before_anyone_approves(team) -> None:
    """R1: Bob injects early in Alice's turn; 600 items later he still may not approve."""
    session_id, runner = team["session"], team["runner"]
    team["stores"]["permission_store"].grant("bob", session_id, LEVEL_EDIT)
    _service_approvers(team, "bob", "carol")
    _register(team, _turn(team, "alice", "migrate the data").id)
    _turn(team, "bob", "and drop the old table")
    team["stores"]["conversation_store"].append(
        session_id,
        [
            NewConversationItem(
                type="message",
                response_id="resp_long",
                data=MessageData(
                    role="assistant",
                    agent="review-agent",
                    content=[{"type": "output_text", "text": f"step {index}"}],
                ),
            )
            for index in range(600)
        ],
    )

    elicitation_id, _retry = _ask(runner, session_id)

    assert "bob" in team["approvals"].get_verdict(elicitation_id).attribution["contributors"]
    _refused(team, "bob", elicitation_id)
    assert _answer(team["carol"], session_id, elicitation_id).status_code == 202


def test_an_unreadable_record_of_who_took_part_refuses_every_answer(team, monkeypatch) -> None:
    """R1: a failed read never counts as "nobody else took part"."""
    from omnigent.stores.turn_store import SqlAlchemyTurnStore

    session_id, runner = team["session"], team["runner"]
    _service_approvers(team, "carol")
    _register(team, _turn(team, "alice", "post the update").id)

    def unreadable(self: Any, session_id: str) -> Any:
        raise sa.exc.OperationalError("SELECT", {}, Exception("database went away"))

    with monkeypatch.context() as patch:
        patch.setattr(SqlAlchemyTurnStore, "turn_authority", unreadable)
        elicitation_id, retry = _ask(runner, session_id)

    record = team["approvals"].get_verdict(elicitation_id)
    assert record.attribution["authority"] == {"complete": False}
    message = _refused(team, "carol", elicitation_id)
    assert message.startswith("Omnigent could not confirm everyone who took part")
    assert _state(team, elicitation_id) is VerdictState.PENDING
    assert team["alice"].get("/v1/approvals").json()["data"] == []
    assert "error" in _tools_call(runner, session_id, **retry)


def test_a_held_back_turn_registration_falls_back_to_counting_every_sender(team) -> None:
    """R4: a policy call arrives before the turn is registered; Bob still may not approve."""
    session_id, runner = team["session"], team["runner"]
    team["stores"]["permission_store"].grant("bob", session_id, LEVEL_EDIT)
    _service_approvers(team, "bob", "carol")
    # Bob's first turn in Alice's thread: accepted, but its registration has not landed.
    _turn(team, "bob", "post the update")

    elicitation_id, _retry = _ask(runner, session_id)

    record = team["approvals"].get_verdict(elicitation_id)
    assert record.binding["initiator"] == "alice"
    assert record.attribution["contributors"] == ["bob"]
    assert record.attribution["authority"]["turn"] is None
    _refused(team, "bob", elicitation_id)
    _refused(team, "alice", elicitation_id)
    assert _answer(team["carol"], session_id, elicitation_id).status_code == 202


def test_typing_recorded_on_another_replica_counts_as_taking_part(team) -> None:
    """R3: Dave typed into the thread's terminal through another server replica."""
    session_id, runner = team["session"], team["runner"]
    _service_approvers(team, "dave", "carol")
    _register(team, _turn(team, "alice", "post the update").id)
    get_turn_store().record_terminal_activity(session_id, "terminal_s1", "dave")

    elicitation_id, _retry = _ask(runner, session_id)

    _refused(team, "dave", elicitation_id)
    assert _answer(team["carol"], session_id, elicitation_id).status_code == 202


def test_a_turn_started_with_unknown_input_cannot_be_approved(team) -> None:
    session_id, runner = team["session"], team["runner"]
    _service_approvers(team, "carol")
    assert _register(team, "item_nobody_recorded")["unresolved"] is True

    elicitation_id, _retry = _ask(runner, session_id)

    assert _refused(team, "carol", elicitation_id).startswith("The turn that asked started with")


def test_an_approval_record_is_read_by_thread_readers_and_its_approvers_only(team) -> None:
    """R6: readers and named approvers read the record; a runner and outsiders do not."""
    session_id, runner = team["session"], team["runner"]
    team["stores"]["permission_store"].grant("bob", session_id, LEVEL_READ)
    _service_approvers(team, "carol")
    _register(team, _turn(team, "alice", "post the update").id)
    elicitation_id, _retry = _ask(runner, session_id)
    url = f"/v1/sessions/{session_id}/approvals/{elicitation_id}"

    generic = runner.get(f"/v1/sessions/{session_id}/elicitations/{elicitation_id}")
    assert generic.status_code == 200 and "approval" not in generic.json()
    assert runner.get(url).status_code == 403
    assert team["bob"].get(url).json()["elicitation_id"] == elicitation_id
    assert team["carol"].get(f"/v1/sessions/{session_id}").status_code == 404
    assert team["carol"].get(url).json()["initiator"] == "alice"
    assert team["dave"].get(url).status_code == 404


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


def test_a_thread_owner_names_approvers_for_their_own_thread_only(team) -> None:
    """Alice names Carol for her thread; an editor, the admin flag or a CLI sign-in cannot."""
    session_id, runner = team["session"], team["runner"]
    shared = team["alice"].put(
        f"/v1/sessions/{session_id}/permissions", json={"user_id": "bob", "level": LEVEL_EDIT}
    )
    assert shared.status_code in (200, 201), shared.text
    body = {"kind": "session", "resource_id": session_id, "user_id": "dave"}

    by_editor = team["bob"].post("/v1/approvers", json=body)
    by_admin = team["admin"].post("/v1/approvers", json=body)
    by_stranger = team["carol"].post("/v1/approvers", json=body)
    by_cli = _bearer(team["alice"].app, _cli_token(team)).post("/v1/approvers", json=body)
    service = {"kind": "service", "resource_id": "tracker", "user_id": "dave"}
    owner_on_service = team["alice"].post("/v1/approvers", json=service)

    assert (by_editor.status_code, by_admin.status_code) == (403, 403)
    assert by_admin.json()["error"]["message"] == "Only the thread's owner names its approvers."
    assert by_stranger.status_code == 404
    assert by_cli.status_code == 403
    assert "Change approvals in the Omnigent web page instead." in by_cli.text
    assert owner_on_service.status_code == 403
    thread = ApprovalResource(ResourceKind.SESSION, session_id)
    service_entry = ApprovalResource(ResourceKind.SERVICE, "tracker")
    assert [e.approvers for e in team["approvals"].approvers_for([thread, service_entry])] == [
        (),
        (),
    ]

    named = team["alice"].post("/v1/approvers", json={**body, "user_id": "carol"})
    assert named.status_code == 201, named.text
    listed = team["bob"].get(
        "/v1/approvers", params={"kind": "session", "resource_id": session_id}
    )
    assert listed.json()["approvers"] == ["carol"], listed.text
    elicitation_id, retry = _ask(runner, session_id)
    assert _answer(team["carol"], session_id, elicitation_id).status_code == 202
    assert "result" in _tools_call(runner, session_id, **retry)
    team["tracker"].call_tool.assert_awaited_once()

    removed_by_admin = team["admin"].delete(
        "/v1/approvers", params={"kind": "session", "resource_id": session_id, "user_id": "carol"}
    )
    assert removed_by_admin.status_code == 403
    removed = team["alice"].delete(
        "/v1/approvers", params={"kind": "session", "resource_id": session_id, "user_id": "carol"}
    )
    assert removed.json() == {"removed": True}


def test_approvers_are_named_on_the_top_level_thread(team) -> None:
    """A sub-agent's approvals belong to its root thread, so its own id is refused."""
    stores = team["stores"]
    child = stores["conversation_store"].create_conversation(
        parent_conversation_id=team["session"]
    )
    stores["permission_store"].grant("alice", child.id, 4)

    named = team["alice"].post(
        "/v1/approvers", json={"kind": "session", "resource_id": child.id, "user_id": "carol"}
    )

    assert named.status_code == 400, named.text


# ── Review r1 ────────────────────────────────────────────────────────────

_ASK_OUTSIDE = (
    'event.type == "tool_call" && has(event.data.arguments.to)'
    ' && event.data.arguments.to == "outside"'
    ' ? {"result": "ASK", "reason": "outside recipient"} : {"result": "ALLOW"}'
)


def ask_and_send_outside(event: dict[str, Any]) -> Any:
    """An agent policy that asks and rewrites the call's arguments (bare, as transforms do)."""
    from omnigent.policies.types import PolicyResult
    from omnigent.spec.types import PolicyAction

    if event.get("type") != "tool_call":
        return None
    arguments = {**(event["data"].get("arguments") or {}), "to": "outside"}
    return PolicyResult(action=PolicyAction.ASK, reason="send it?", data=arguments)


# Registered the way a server administrator adds a custom handler (``policy_modules``).
POLICY_REGISTRY = [
    {
        "handler": f"{__name__}.ask_and_send_outside",
        "description": "Test transform: asks, and sends the call to an outside recipient.",
    }
]


@pytest.fixture
def custom_handler() -> Any:
    from omnigent.policies.registry import load_registry

    load_registry([__name__])
    try:
        yield
    finally:
        load_registry()


def _session_rewriting_the_recipient(team: dict[str, Any]) -> tuple[str, TestClient]:
    from tests.server.test_account_revocation import create_session

    guardrails = {
        "policies": {
            "rewrite": {
                "type": "function",
                "function": {"path": f"{__name__}.ask_and_send_outside"},
            }
        }
    }
    created = create_session(team["alice"], guardrails=guardrails)
    assert created.status_code == 201, created.text
    session_id = created.json()["session_id"]
    attached = team["alice"].post(
        f"/v1/sessions/{session_id}/agent/mcp-servers",
        json={"name": "tracker", "transport": "registry"},
    )
    assert attached.status_code == 200, attached.text
    return session_id, _delegated_runner(team["alice"], team["stores"], session_id)


def _team_rule(team: dict[str, Any], expression: str, name: str = "outside") -> dict[str, Any]:
    created = team["admin"].post(
        "/v1/protected-policies",
        json={
            "domain": "team",
            "name": name,
            "handler": _CEL,
            "factory_params": {"expression": expression},
        },
    )
    assert created.status_code == 201, created.text
    return created.json()


@pytest.mark.parametrize("rule_changes_before_dispatch", [False, True])
def test_a_transformed_call_needs_both_approvers_and_a_late_rule_refuses_it(
    team, monkeypatch, custom_handler, rule_changes_before_dispatch
) -> None:
    """r1-1: the team rule sees the transformed call; the person's approval is not enough."""
    _name_approver(team, "team", "dave")
    _team_rule(team, _ASK_OUTSIDE)
    session_id, runner = _session_rewriting_the_recipient(team)
    elicitation_id, retry = _ask(runner, session_id)
    record = team["approvals"].get_verdict(elicitation_id)
    assert [r["kind"] for r in record.requirements] == ["personal", "team"]

    first = _answer(team["alice"], session_id, elicitation_id).json()
    assert (first["queued"], first["awaiting"]) == (False, "team")
    assert "error" in _tools_call(runner, session_id, **retry)
    assert _answer(team["dave"], session_id, elicitation_id).status_code == 202
    if rule_changes_before_dispatch:
        real = orchestration.consume_protected_approval

        async def consume_then_add_a_rule(*args: Any, **kwargs: Any) -> Any:
            used = await real(*args, **kwargs)
            _team_rule(team, '{"result": "ASK", "reason": "late rule"}', name="late")
            return used

        monkeypatch.setattr(orchestration, "consume_protected_approval", consume_then_add_a_rule)

    response = _tools_call(runner, session_id, **retry)

    if rule_changes_before_dispatch:
        assert "ask again" in response["error"]["message"], response
        team["tracker"].call_tool.assert_not_awaited()
    else:
        assert "result" in response, response
        assert team["tracker"].call_tool.await_args.args[3]["to"] == "outside"


def _admin_cli(team: dict[str, Any]) -> tuple[str, str]:
    """The admin's CLI sign-in: its session token and its login-grant access token."""
    cli = TestClient(team["admin"].app)
    login = cli.post(
        "/auth/login",
        json={"username": "admin", "password": "admin-pw-12345", "issue_refresh": True},
    )
    assert login.status_code == 200, login.text
    refreshed = cli.post(
        "/oauth/token",
        data={"grant_type": "refresh_token", "refresh_token": login.json()["refresh_token"]},
    )
    assert refreshed.status_code == 200, refreshed.text
    return login.json()["token"], refreshed.json()["access_token"]


def test_a_cli_sign_in_cannot_become_a_browser_sign_in_or_obtain_a_password(team) -> None:
    """r1-2: reset, invitation and password-change chains keep a CLI credential a CLI one."""
    app = team["admin"].app
    session_token, grant_token = _admin_cli(team)
    for token in (session_token, grant_token):
        cli = _bearer(app, token)
        assert cli.get("/auth/me").status_code == 200  # a valid admin sign-in
        reset = cli.post("/auth/users/admin/reset")
        assert reset.status_code == 403, reset.text
        assert "Reset passwords in the Omnigent web page instead." in reset.json()["error"]
        assert cli.post("/auth/users/bob/reset").status_code == 403
        invite = cli.post("/auth/invite", json={"is_admin": True})
        assert invite.status_code == 403, invite.text
    # The old password still signs in: nothing was reset.
    assert (
        TestClient(app)
        .post("/auth/login", json={"username": "admin", "password": "admin-pw-12345"})
        .status_code
        == 200
    )

    # Knowing the old password, a CLI may change it (the restore drill does);
    # the cookie it gets back stays a non-browser one.
    cli = _bearer(app, session_token)
    changed = cli.post(
        "/auth/users/me/password",
        json={"old_password": "admin-pw-12345", "new_password": "admin-pw-67890"},
    )
    assert changed.status_code == 204, changed.text
    reissued = TestClient(app)
    reissued.cookies.set(_cookie_name(team), changed.cookies[_cookie_name(team)])
    refused = reissued.post(
        "/v1/approvers", json={"kind": "team", "resource_id": "", "user_id": "carol"}
    )
    assert refused.status_code == 403
    assert "Sign in again in the browser" in refused.json()["error"]["message"]

    # A browser sign-in keeps every power.
    browser = TestClient(app)
    signed_in = browser.post(
        "/auth/login", json={"username": "admin", "password": "admin-pw-67890"}
    )
    assert signed_in.status_code == 200
    assert browser.post("/auth/users/bob/reset").status_code == 200
    own = browser.post("/auth/users/admin/reset")
    assert own.status_code == 200
    assert (
        browser.post(
            "/v1/approvers", json={"kind": "team", "resource_id": "", "user_id": "carol"}
        ).status_code
        == 201
    )


@pytest.mark.parametrize("tunnel_header", [True, False])
def test_the_initiator_is_the_server_recorded_turn_actor_with_or_without_the_header(
    team, tunnel_header
) -> None:
    """r1-3: Bob's turn in Alice's thread stays Bob's, however the runner calls."""
    from omnigent.runner.identity import RUNNER_TUNNEL_TOKEN_HEADER
    from omnigent.server.routes._sessions.common import _TURN_ACTOR_LABEL

    session_id, runner = team["session"], team["runner"]
    team["alice"].put(
        f"/v1/sessions/{session_id}/permissions", json={"user_id": "bob", "level": LEVEL_EDIT}
    )
    _name_approver(team, "service", "bob", "tracker")
    # The server records the turn's actor when Bob's message is forwarded.
    team["stores"]["conversation_store"].set_labels(session_id, {_TURN_ACTOR_LABEL: "bob"})
    if not tunnel_header:
        runner = TestClient(team["alice"].app)
        runner.headers["Authorization"] = team["runner"].headers["Authorization"]
        assert RUNNER_TUNNEL_TOKEN_HEADER not in runner.headers

    elicitation_id, _retry = _ask(runner, session_id)

    assert team["approvals"].get_verdict(elicitation_id).binding["initiator"] == "bob"
    assert _answer(team["bob"], session_id, elicitation_id).status_code == 403


def test_an_in_place_reconnect_needs_a_new_decision(team) -> None:
    """r1-4: a replaced login (same row, same creation time) is a different connection."""
    from omnigent.stores.credential_store.sqlalchemy_store import CredentialStore
    from tests.server.test_credential_store import _FakeCipher

    app = team["alice"].app
    db_url = team["stores"]["permission_store"].storage_location
    service = McpService(
        id="tracker",
        title="Tracker",
        url="https://example.test/mcp",
        auth="bearer",
        tools=["post_update"],
    )
    registry = McpRegistry(
        McpRegistryConfig(services=[service]), CredentialStore(db_url, _FakeCipher())
    )
    app.state.mcp_registry = registry
    registry.save_login(service, "alice", "mcp:tracker", {"access_token": "first"}, {})
    session_id, runner = team["session"], team["runner"]
    elicitation_id, retry = _ask(runner, session_id)
    assert _answer(team["alice"], session_id, elicitation_id).status_code == 202
    before = registry.store.get("alice", "mcp:tracker")

    registry.save_login(service, "alice", "mcp:tracker", {"access_token": "second"}, {})
    after = registry.store.get("alice", "mcp:tracker")

    assert after.created_at == before.created_at and after.login_id != before.login_id
    refused = _tools_call(runner, session_id, **retry)
    assert "ask again" in refused["error"]["message"], refused
    team["tracker"].call_tool.assert_not_awaited()


def test_a_pinned_login_is_refused_after_a_replacement(team) -> None:
    """r1-4: the token read for an approved call refuses a login other than the approved one."""
    from omnigent.server.mcp_registry import pinned_login
    from omnigent.server.routes.connections_base import ConnectionError as McpConnectionError
    from omnigent.stores.credential_store.sqlalchemy_store import CredentialStore
    from tests.server.test_credential_store import _FakeCipher

    db_url = team["stores"]["permission_store"].storage_location
    service = McpService(
        id="tracker",
        title="Tracker",
        url="https://example.test/mcp",
        auth="bearer",
        tools=["post_update"],
    )
    registry = McpRegistry(
        McpRegistryConfig(services=[service]), CredentialStore(db_url, _FakeCipher())
    )
    registry.save_login(service, "alice", "mcp:tracker", {"access_token": "first"}, {})
    approved = registry.store.get("alice", "mcp:tracker").login_marker()

    async def read() -> str | None:
        with pinned_login("tracker", "alice", approved):
            return await registry.token(service, "alice", None)

    assert asyncio.run(read()) == "first"
    registry.save_login(service, "alice", "mcp:tracker", {"access_token": "second"}, {})
    with pytest.raises(McpConnectionError):
        asyncio.run(read())


@pytest.mark.parametrize("change", ["none", "rule", "grant", "initiator"])
def test_a_reposted_native_question_reuses_its_answer_only_unchanged(team, change) -> None:
    """r1-6: the hook's retry gets its recorded ALLOW only while the approval still matches."""
    from omnigent.server._elicitation_registry import _harness_parked_elicitations
    from omnigent.server.routes._sessions.common import _TURN_ACTOR_LABEL

    session_id, runner = team["session"], team["runner"]
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

    async def ask_and_approve() -> dict[str, Any]:
        transport = httpx.ASGITransport(app=team["alice"].app)
        async with (
            httpx.AsyncClient(
                transport=transport, base_url="http://testserver", headers=dict(runner.headers)
            ) as runner_http,
            httpx.AsyncClient(
                transport=transport, base_url="http://testserver", cookies=team["alice"].cookies
            ) as person_http,
        ):
            waiting = asyncio.create_task(
                runner_http.post(f"/v1/sessions/{session_id}/policies/evaluate", json=body)
            )
            async with asyncio.timeout(15):
                while elicitation_id not in _harness_parked_elicitations:
                    await asyncio.sleep(0.05)
            approved = await person_http.post(
                _resolve_url(session_id, elicitation_id), json={"action": "accept"}
            )
            assert approved.status_code == 202, approved.text
            return (await asyncio.wait_for(waiting, timeout=15)).json()

    assert asyncio.run(ask_and_approve())["result"] == "POLICY_ACTION_ALLOW"
    if change == "rule":
        _team_rule(team, '{"result": "ALLOW"}', name="audit")
    elif change == "grant":
        _name_approver(team, "session", "carol", session_id)
    elif change == "initiator":
        team["stores"]["conversation_store"].set_labels(session_id, {_TURN_ACTOR_LABEL: "bob"})

    reposted = runner.post(f"/v1/sessions/{session_id}/policies/evaluate", json=body).json()

    expected = "POLICY_ACTION_ALLOW" if change == "none" else "POLICY_ACTION_DENY"
    assert reposted["result"] == expected, reposted


def test_a_reregistered_username_inherits_no_approval_authority(team) -> None:
    """r1-7: deleting Carol removes her grants; a new Carol is not her approver."""
    session_id, runner = team["session"], team["runner"]
    _name_approver(team, "service", "carol", "tracker")
    deleted = team["admin"].delete("/auth/users/carol")
    assert deleted.status_code == 204, deleted.text
    gone = team["admin"].post(
        "/v1/approvers", json={"kind": "service", "resource_id": "tracker", "user_id": "carol"}
    )
    assert gone.status_code == 400, gone.text
    new_carol = _member(team["admin"], "carol")
    listed = team["admin"].get(
        "/v1/approvers", params={"kind": "service", "resource_id": "tracker"}
    )
    assert listed.json()["approvers"] == []

    elicitation_id, _retry = _ask(runner, session_id)

    assert _answer(new_carol, session_id, elicitation_id).status_code == 404


@pytest.mark.parametrize("initiator", ["alice", "bob"])
def test_a_thread_refusing_its_initiator_refuses_its_owner_too(team, initiator) -> None:
    """r1-8: with ``allow_initiator=False`` the owner approves others' actions, not her own."""
    from omnigent.server.routes._sessions.common import _TURN_ACTOR_LABEL

    session_id, runner = team["session"], team["runner"]
    saved = team["alice"].put(
        "/v1/approval-settings",
        json={"kind": "session", "resource_id": session_id, "allow_initiator": False},
    )
    assert saved.json() == {"allow_initiator": False}, saved.text
    if initiator == "bob":
        team["alice"].put(
            f"/v1/sessions/{session_id}/permissions", json={"user_id": "bob", "level": LEVEL_EDIT}
        )
        team["stores"]["conversation_store"].set_labels(session_id, {_TURN_ACTOR_LABEL: "bob"})
    elicitation_id, retry = _ask(runner, session_id)

    answered = _answer(team["alice"], session_id, elicitation_id)

    if initiator == "alice":
        assert answered.status_code == 403, answered.text
        assert _state(team, elicitation_id) is VerdictState.PENDING
    else:
        assert answered.status_code == 202, answered.text
        assert "result" in _tools_call(runner, session_id, **retry)


def test_answers_from_one_stale_snapshot_still_complete_the_request(team) -> None:
    """r1-9: two approvers answering from the same snapshot approve it, resolving it once."""
    from omnigent.server import approval_authority

    session_id, runner = team["session"], team["runner"]
    _name_approver(team, "team", "dave")
    _team_rule(team, '{"result": "ASK", "reason": "team review"}', name="team-review")
    elicitation_id, _retry = _ask(runner, session_id)
    snapshot = team["approvals"].get_verdict(elicitation_id)
    assert [r["kind"] for r in snapshot.requirements] == ["personal", "team"]
    stores = team["stores"]

    gates = [
        approval_authority._gate_sync(
            team["approvals"],
            stores["conversation_store"],
            stores["permission_store"],
            snapshot,
            person,
            "accept",
            True,
        )
        for person in ("alice", "dave")
    ]

    assert [gate.final for gate in gates] == [False, True]
    assert gates[1].action == "accept"
    assert _state(team, elicitation_id) is VerdictState.APPROVED
