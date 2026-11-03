"""Who did what on whose behalf: turn attribution against what actually happened (WP8.8).

Accounts-mode app, real routes. Alice owns a thread on her own computer; Bob
is a member she shares it with. Each turn records its initiator, the thread's
owner, the computer that ran it and whose configured harness login it used,
from the server's own records. A runner never names a person.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.testclient import TestClient
from starlette.datastructures import Headers

from omnigent.entities import NewConversationItem
from omnigent.entities.conversation import MessageData
from omnigent.runner.identity import RUNNER_TUNNEL_TOKEN_HEADER, token_bound_runner_id
from omnigent.runtime import pending_inputs
from omnigent.server import terminal_typists
from omnigent.server.approval_authority import turn_contributors
from omnigent.server.auth import LEVEL_EDIT
from omnigent.server.routes._sessions import orchestration
from omnigent.server.schemas import SessionEventInput
from omnigent.server.turn_attribution import (
    TURN_ACTOR_LABEL,
    TURN_STARTED_AT_LABEL,
    person_initiator,
    startup_initiator,
)
from tests.server.integration.test_sessions_attribution import (
    _CaptureRunnerClient,
    _noop_relay_ready,
)
from tests.server.test_account_revocation import setup_app as setup_app
from tests.server.test_runner_principal import _mint, _session
from tests.server.test_runner_principal import people as people
from tests.server.test_runner_surfaces import _attach, _IdleRunnerSocket
from tests.server.test_runner_surfaces import idle_terminal as idle_terminal

_CANARY = "PROMPT-CANARY-51c2"
_READY = {"claude-sdk": True}


@pytest.fixture(autouse=True)
def forwarding(monkeypatch: pytest.MonkeyPatch) -> None:
    """Messages are saved and forwarded to a stand-in runner that accepts them."""
    from omnigent.server.routes import sessions as sessions_mod
    from omnigent.server.routes.sessions import routes_events as events_mod

    async def _client(*_: Any, **__: Any) -> _CaptureRunnerClient:
        return _CaptureRunnerClient()

    monkeypatch.setattr(sessions_mod, "_get_runner_client", _client)
    monkeypatch.setattr(events_mod, "_ensure_runner_relay_ready", _noop_relay_ready)


def _on_computer(
    stores: dict[str, Any],
    session_id: str,
    owner: str,
    readiness: dict[str, Any] | None,
    *,
    host_id: str | None = None,
    name: str | None = None,
) -> str:
    """Bind the thread to a runner on *owner*'s computer; return the runner's binding.

    A new computer is registered unless *host_id* names one; computer names are
    unique per person, so a second computer needs its own *name*.
    """
    binding = f"binding-{uuid.uuid4().hex}"
    if host_id is None:
        host_id = uuid.uuid4().hex
        stores["host_store"].upsert_on_connect(
            host_id, name or f"{owner}-laptop", owner, configured_harnesses=readiness
        )
    stores["conversation_store"].set_host_id(session_id, host_id, workspace="/tmp/workspace")
    stores["conversation_store"].replace_runner_id(session_id, token_bound_runner_id(binding))
    return binding


def _runner(app: Any, binding: str) -> TestClient:
    minted = _mint(app, binding)
    assert minted.status_code == 200, minted.text
    client = TestClient(app)
    client.headers[RUNNER_TUNNEL_TOKEN_HEADER] = binding
    client.headers["Authorization"] = f"Bearer {minted.json()['token']}"
    return client


def _send(client: TestClient, session_id: str, text: str, **extra: Any) -> Any:
    return client.post(
        f"/v1/sessions/{session_id}/events",
        json={
            "type": "message",
            "data": {"role": "user", "content": [{"type": "input_text", "text": text}]},
            **extra,
        },
    )


def _turns(stores: dict[str, Any], session_id: str) -> list[Any]:
    return [
        item
        for item in stores["conversation_store"].list_items(session_id).data
        if isinstance(item.data, MessageData) and item.data.role == "user"
    ]


def _labels(stores: dict[str, Any], session_id: str) -> dict[str, str]:
    return stores["conversation_store"].get_conversation(session_id).labels


def test_a_turn_records_who_and_whose_thread_computer_and_subscription(people: Any) -> None:
    """T18: initiator, owner, computer and subscription are each the server's record."""
    _admin, alice, _bob, stores = people
    thread = _session(alice)
    host_binding = _on_computer(stores, thread, "alice", _READY)

    assert _send(alice, thread, f"summarize {_CANARY}").status_code == 202

    [turn] = _turns(stores, thread)
    attribution = turn.attribution
    alice_registration = stores["account_store"].get_user("alice").account_generation
    assert attribution.initiator.model_dump(exclude_none=True) == {
        "kind": "person",
        "user": "alice",
        "registration": alice_registration,
        "source": "signed_in",
    }
    assert (attribution.owner.user, attribution.owner.registration) == (
        "alice",
        alice_registration,
    )
    host = stores["conversation_store"].get_conversation(thread)
    assert attribution.host.model_dump() == {
        "id": host.host_id,
        "name": "alice-laptop",
        "owner": "alice",
        "kind": "personal",
        "runner_id": token_bound_runner_id(host_binding),
    }
    assert attribution.subscription.model_dump() == {
        "subject": "alice",
        "provider": "anthropic",
        "source": "computer_login",
        "login_state": "ready",
        "login_state_reported_by": "computer",
        "account": None,
        "verified": False,
    }
    assert attribution.harness == "claude-sdk"
    # Never invented: the harness has reported neither, so both are unknown.
    assert attribution.model.reported is None and attribution.effort.reported is None

    listed = alice.get(f"/v1/sessions/{thread}/items").json()["data"]
    api = next(item["attribution"] for item in listed if "attribution" in item)
    assert api["subscription"]["account"] is None and api["subscription"]["verified"] is False
    # No credential, sign-in material or prompt content in the record.
    text = json.dumps(api)
    assert _CANARY not in text
    assert host_binding not in text
    for cookie in alice.cookies.values():
        assert cookie not in text

    def keys(value: Any) -> set[str]:
        if isinstance(value, dict):
            return set(value) | {key for child in value.values() for key in keys(child)}
        return set()

    assert not {
        key
        for key in keys(api)
        if any(word in key for word in ("token", "password", "cookie", "secret", "assertion"))
    }


def test_a_colleague_editing_a_shared_thread_runs_on_the_owners_computer_and_subscription(
    people: Any,
) -> None:
    """T5 under D51: Bob initiates, on Alice's computer and Claude login; he cannot pose as her."""
    _admin, alice, bob, stores = people
    thread = _session(alice)
    _on_computer(stores, thread, "alice", _READY)
    stores["permission_store"].grant("bob", thread, LEVEL_EDIT)

    assert _send(bob, thread, "run the report").status_code == 202

    [turn] = _turns(stores, thread)
    assert turn.created_by == "bob"
    attribution = turn.attribution
    assert (attribution.initiator.kind, attribution.initiator.user) == ("person", "bob")
    assert attribution.initiator.registration == (
        stores["account_store"].get_user("bob").account_generation
    )
    assert attribution.owner.user == "alice"
    assert attribution.host.owner == "alice"
    assert attribution.subscription.subject == "alice"
    assert _labels(stores, thread)[TURN_ACTOR_LABEL] == "bob"

    # Bob cannot name someone else as the sender, nor write the server's record.
    forged = _send(bob, thread, "as alice", created_by="alice")
    assert forged.status_code == 403, forged.text
    for label in (TURN_ACTOR_LABEL, TURN_STARTED_AT_LABEL):
        refused = bob.patch(f"/v1/sessions/{thread}", json={"labels": {label: "alice"}})
        assert refused.status_code == 400, refused.text
    assert len(_turns(stores, thread)) == 1


def test_a_runner_cannot_name_another_initiator_for_a_sub_agent(people: Any) -> None:
    """T6/T8: a sub-agent's prompt is the agent's, for the person recorded on its parent."""
    _admin, alice, _bob, stores = people
    conversations = stores["conversation_store"]
    parent = _session(alice)
    runner = _runner(alice.app, _on_computer(stores, parent, "alice", _READY))
    # The server recorded Bob as the parent's current turn actor.
    conversations.set_labels(parent, {TURN_ACTOR_LABEL: "bob"})
    agent_id = conversations.get_conversation(parent).agent_id
    created = runner.post("/v1/sessions", json={"agent_id": agent_id, "parent_session_id": parent})
    assert created.status_code == 201, created.text
    child = created.json().get("session_id") or created.json()["id"]

    # The runner claims Carol sent it; the server does not take its word.
    assert _send(runner, child, "check the logs", created_by="carol").status_code == 202

    [turn] = _turns(stores, child)
    initiator = turn.attribution.initiator
    assert (initiator.kind, initiator.from_session_id, initiator.on_behalf_of) == (
        "agent",
        parent,
        "bob",
    )
    assert initiator.source == "thread_record"
    assert turn.created_by == "bob"
    assert turn.attribution.owner.user == "alice"
    assert turn.attribution.host is not None and turn.attribution.host.owner == "alice"
    assert _labels(stores, child)[TURN_ACTOR_LABEL] == "bob"


def test_a_saved_scheduled_input_starts_the_schedules_turn() -> None:
    """A managed fire's saved prompt is the schedule's turn; a person's saved input is theirs."""
    scheduled = startup_initiator("alice", "task_9")
    assert (scheduled.kind, scheduled.user, scheduled.schedule_id) == (
        "schedule",
        "alice",
        "task_9",
    )
    assert startup_initiator("alice", None).kind == "person"


def test_a_scheduled_saved_input_carries_its_task_into_delivery(people: Any) -> None:
    from omnigent.server.scheduled.authority import ScheduledOrigin
    from omnigent.server.startup_inputs import ManagedStartupInputs

    _admin, alice, _bob, stores = people
    conversations = stores["conversation_store"]
    startup = ManagedStartupInputs(conversations, alice.app.state.host_registry)
    startup.scheduled_authority = AsyncMock()
    delivered: list[Any] = []

    async def dispatch(context: Any, session_id: str, event: Any) -> dict[str, Any]:
        delivered.append(context)
        return {"queued": True}

    startup.dispatch = dispatch
    thread = _session(alice)
    saved = asyncio.run(
        startup.enqueue(
            thread,
            SessionEventInput(
                type="message",
                data={"role": "user", "content": [{"type": "input_text", "text": "nightly"}]},
            ),
            user_id="alice",
            headers=Headers({}),
            scheduled=ScheduledOrigin("task_9", None),
        )
    )

    asyncio.run(startup._deliver(alice.app, thread, saved))

    assert [context.scheduled_task_id for context in delivered] == ["task_9"]


def _report_turn_start(stores: dict[str, Any], session_id: str, **ids: list[str]) -> None:
    asyncio.run(
        orchestration._record_reported_turn_start(
            session_id,
            {"type": "session.turn_started", **ids},
            stores["conversation_store"],
        )
    )


def test_a_message_queued_behind_a_running_turn_does_not_relabel_it(
    people: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D53: the turn actor changes when the runner starts the queued message's turn, not before."""
    monkeypatch.setattr(orchestration, "_runner_reports_turn_starts", lambda conv: True)
    _admin, alice, bob, stores = people
    thread = _session(alice)
    _on_computer(stores, thread, "alice", _READY)
    stores["permission_store"].grant("bob", thread, LEVEL_EDIT)

    assert _send(alice, thread, "start the migration").status_code == 202
    [alices] = _turns(stores, thread)
    _report_turn_start(stores, thread, input_item_ids=[alices.id])
    assert _labels(stores, thread)[TURN_ACTOR_LABEL] == "alice"

    assert _send(bob, thread, "also update the docs").status_code == 202
    bobs = _turns(stores, thread)[-1]
    assert _labels(stores, thread)[TURN_ACTOR_LABEL] == "alice"
    # Everyone whose input joined the turn counts: Bob's message may be injected.
    assert turn_contributors(stores["conversation_store"], thread) == ["alice", "bob"]

    _report_turn_start(stores, thread, input_item_ids=[bobs.id])
    labels = _labels(stores, thread)
    assert labels[TURN_ACTOR_LABEL] == "bob"
    assert labels[TURN_STARTED_AT_LABEL] == str(bobs.created_at)


def test_a_turn_start_names_people_only_from_the_servers_records(people: Any) -> None:
    """A runner names inputs, never people; ids it cannot back with a record change nothing."""
    _admin, alice, _bob, stores = people
    thread, other = _session(alice), _session(alice)
    _on_computer(stores, thread, "alice", _READY)
    _on_computer(stores, other, "alice", _READY, name="alice-desktop")
    assert _send(alice, other, "elsewhere").status_code == 202
    [foreign] = _turns(stores, other)
    stores["conversation_store"].set_labels(thread, {TURN_ACTOR_LABEL: "alice"})

    _report_turn_start(
        stores, thread, input_item_ids=[foreign.id, "item_missing"], input_stable_ids=["f" * 32]
    )

    labels = _labels(stores, thread)
    assert labels[TURN_ACTOR_LABEL] == "alice"
    assert TURN_STARTED_AT_LABEL not in labels


def test_a_terminal_harness_turn_is_named_from_the_queued_sender(people: Any) -> None:
    _admin, alice, _bob, stores = people
    thread = _session(alice)
    stable = uuid.uuid4().hex
    pending_inputs.record(
        thread,
        [{"type": "input_text", "text": "deploy"}],
        created_by="bob",
        stable_id=stable,
        initiator=person_initiator("bob"),
    )

    _report_turn_start(stores, thread, input_stable_ids=[stable])

    assert _labels(stores, thread)[TURN_ACTOR_LABEL] == "bob"


def test_an_older_runner_keeps_the_forward_time_record(people: Any) -> None:
    """A runner without turn-start reports: recorded at forward time, as before (a known limit)."""
    _admin, alice, bob, stores = people
    thread = _session(alice)
    _on_computer(stores, thread, "alice", _READY)
    stores["permission_store"].grant("bob", thread, LEVEL_EDIT)

    assert _send(alice, thread, "first").status_code == 202
    assert _send(bob, thread, "queued").status_code == 202

    assert _labels(stores, thread)[TURN_ACTOR_LABEL] == "bob"


def _typed_prompt(session_id: str, text: str) -> SessionEventInput:
    return SessionEventInput(
        type="external_conversation_item",
        data={
            "source_id": uuid.uuid4().hex,
            "item_type": "message",
            "item_data": {"role": "user", "content": [{"type": "input_text", "text": text}]},
            "response_id": f"resp_{uuid.uuid4().hex[:8]}",
        },
    )


@pytest.mark.timeout(30)
def test_terminal_typing_names_whoever_attached_to_type(
    people: Any, idle_terminal: _IdleRunnerSocket, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An administrator types into Alice's terminal under D52's override: the turn names them."""
    admin, alice, _bob, stores = people
    conversations = stores["conversation_store"]
    thread = _session(alice)
    _on_computer(stores, thread, "alice", _READY)
    monkeypatch.setattr("omnigent.runtime.session_stream.publish", Mock())

    def persist(text: str) -> Any:
        conv = conversations.get_conversation(thread)
        asyncio.run(
            orchestration._persist_external_conversation_item(
                thread, conv, _typed_prompt(thread, text), conversations, created_by="alice"
            )
        )
        return _turns(stores, thread)[-1]

    with _attach(admin, thread, read_only=False):
        for _ in range(50):
            if terminal_typists.typist(thread) == "admin":
                break
            time.sleep(0.02)
        typed = persist("restart the service")
    assert typed.attribution.initiator.model_dump(exclude_none=True) == {
        "kind": "terminal",
        "user": "admin",
        "source": "terminal_attach",
    }
    assert typed.created_by == "admin"
    assert _labels(stores, thread)[TURN_ACTOR_LABEL] == "admin"

    # With nobody attached here, the typing is assumed to be the owner's, and says so.
    for _ in range(50):
        if terminal_typists.typist(thread) is None:
            break
        time.sleep(0.02)
    assumed = persist("check the status")
    initiator = assumed.attribution.initiator
    assert (initiator.kind, initiator.user, initiator.source) == (
        "terminal",
        "alice",
        "owner_assumed",
    )


def test_a_read_only_attachment_names_nobody(
    people: Any, idle_terminal: _IdleRunnerSocket
) -> None:
    admin, alice, _bob, stores = people
    thread = _session(alice)
    _on_computer(stores, thread, "alice", _READY)
    with _attach(admin, thread, read_only=True):
        time.sleep(0.1)
        assert terminal_typists.typist(thread) is None


def test_a_turns_attribution_reaches_live_viewers(
    people: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    published: list[dict[str, Any]] = []
    monkeypatch.setattr(
        "omnigent.runtime.session_stream.publish",
        lambda session_id, event: published.append(event),
    )
    _admin, alice, bob, stores = people
    thread = _session(alice)
    _on_computer(stores, thread, "alice", _READY)
    stores["permission_store"].grant("bob", thread, LEVEL_EDIT)

    assert _send(bob, thread, "hello").status_code == 202

    [consumed] = [event for event in published if event["type"] == "session.input.consumed"]
    attribution = consumed["data"]["attribution"]
    assert attribution["initiator"]["user"] == "bob"
    assert attribution["host"]["owner"] == "alice"


def test_items_without_attribution_still_list(people: Any) -> None:
    """Items saved before attribution existed read as unknown."""
    _admin, alice, _bob, stores = people
    thread = _session(alice)
    stores["conversation_store"].append(
        thread,
        [
            NewConversationItem(
                type="message",
                response_id="turn_old",
                data=MessageData(role="user", content=[{"type": "input_text", "text": "old"}]),
                created_by="alice",
            )
        ],
    )
    [item] = alice.get(f"/v1/sessions/{thread}/items").json()["data"]
    assert "attribution" not in item and item["created_by"] == "alice"
