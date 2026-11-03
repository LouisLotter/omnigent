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
from omnigent.runtime import get_turn_store, pending_inputs
from omnigent.server.auth import LEVEL_EDIT
from omnigent.server.routes._sessions import orchestration
from omnigent.server.schemas import SessionEventInput
from omnigent.server.turn_attribution import (
    TURN_ACTOR_LABEL,
    person_initiator,
    startup_initiator,
)
from omnigent.stores.turn_store import InputKind
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
    refused = bob.patch(f"/v1/sessions/{thread}", json={"labels": {TURN_ACTOR_LABEL: "alice"}})
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


def _register(
    runner: TestClient, session_id: str, *input_ids: str, turn_id: str | None = None
) -> Any:
    return runner.post(
        f"/v1/sessions/{session_id}/turns",
        json={"turn_id": turn_id or f"turn_{uuid.uuid4().hex}", "input_ids": list(input_ids)},
    )


def test_the_bound_runner_registers_each_turn_and_the_server_names_its_person(
    people: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R4: the turn's person is set when the runner registers it, never by a queued message."""
    monkeypatch.setattr(orchestration, "_runner_reports_turn_starts", lambda conv: True)
    _admin, alice, bob, stores = people
    thread = _session(alice)
    runner = _runner(alice.app, _on_computer(stores, thread, "alice", _READY))
    stores["permission_store"].grant("bob", thread, LEVEL_EDIT)

    assert _send(alice, thread, "start the migration").status_code == 202
    [alices] = _turns(stores, thread)
    assert TURN_ACTOR_LABEL not in _labels(stores, thread)
    registered = _register(runner, thread, alices.id, turn_id="turn_alice")
    assert registered.status_code == 200, registered.text
    assert registered.json() == {
        "turn_id": "turn_alice",
        "initiator": "alice",
        "unresolved": False,
    }
    assert _labels(stores, thread)[TURN_ACTOR_LABEL] == "alice"

    assert _send(bob, thread, "also update the docs").status_code == 202
    bobs = _turns(stores, thread)[-1]
    assert _labels(stores, thread)[TURN_ACTOR_LABEL] == "alice"
    # Bob's queued message may be injected into Alice's turn: he counts as taking part.
    assert get_turn_store().turn_authority(thread).people == {"alice", "bob"}

    assert _register(runner, thread, bobs.id, turn_id="turn_bob").json()["initiator"] == "bob"
    assert _labels(stores, thread)[TURN_ACTOR_LABEL] == "bob"
    # A retried registration is the same turn.
    assert _register(runner, thread, alices.id, turn_id="turn_bob").json()["initiator"] == "bob"


def test_only_the_bound_runner_registers_a_threads_turns(people: Any) -> None:
    _admin, alice, bob, stores = people
    thread, other = _session(alice), _session(alice)
    _runner(alice.app, _on_computer(stores, thread, "alice", _READY))
    other_runner = _runner(alice.app, _on_computer(stores, other, "alice", _READY, name="desk"))
    stores["permission_store"].grant("bob", thread, LEVEL_EDIT)

    assert _register(bob, thread, "item_x").status_code == 403
    assert _register(alice, thread, "item_x").status_code == 403
    assert _register(other_runner, thread, "item_x").status_code in (403, 404)
    assert get_turn_store().latest_turn(thread) is None


def test_a_turn_naming_input_the_server_never_recorded_is_unresolved(people: Any) -> None:
    _admin, alice, _bob, stores = people
    thread = _session(alice)
    runner = _runner(alice.app, _on_computer(stores, thread, "alice", _READY))
    stores["conversation_store"].set_labels(thread, {TURN_ACTOR_LABEL: "alice"})

    answer = _register(runner, thread, "item_never_recorded").json()

    assert (answer["initiator"], answer["unresolved"]) == (None, True)
    assert get_turn_store().turn_authority(thread).unresolved


def test_an_older_runner_keeps_the_forward_time_record(people: Any) -> None:
    """A runner that does not register turns: recorded at forward time, as before."""
    _admin, alice, bob, stores = people
    thread = _session(alice)
    _on_computer(stores, thread, "alice", _READY)
    stores["permission_store"].grant("bob", thread, LEVEL_EDIT)

    assert _send(alice, thread, "first").status_code == 202
    assert _send(bob, thread, "queued").status_code == 202

    assert _labels(stores, thread)[TURN_ACTOR_LABEL] == "bob"
    # Every accepted input is still recorded for approvals.
    assert get_turn_store().turn_authority(thread).people == {"alice", "bob"}


def _native_dispatch(
    monkeypatch: pytest.MonkeyPatch, stores: dict[str, Any], thread: str, text: str, sender: str
) -> str:
    """Send a terminal-harness message as the server does, without a client stable id."""
    delivered: dict[str, Any] = {}

    async def ready(*_args: Any, **_kwargs: Any) -> Any:
        return orchestration._NativeTerminalEnsureOutcome(error=None)

    async def forward(*_args: Any, **kwargs: Any) -> str:
        delivered.update(kwargs.get("input_delivery") or {})
        return "accepted"

    monkeypatch.setattr(orchestration, "_is_native_terminal_session", lambda _conv: True)
    monkeypatch.setattr(
        orchestration, "_build_native_terminal_message_event", lambda *_a, **_k: None
    )
    monkeypatch.setattr(orchestration, "_ensure_native_terminal_ready", ready)
    monkeypatch.setattr(orchestration, "_forward_native_terminal_message", forward)
    conversations = stores["conversation_store"]
    result = asyncio.run(
        orchestration._dispatch_session_event_to_runner_impl(
            thread,
            conversations.get_conversation(thread),
            SessionEventInput(
                type="message",
                data={"role": "user", "content": [{"type": "input_text", "text": text}]},
            ),
            conversations,
            object(),  # type: ignore[arg-type]
            agent_name=None,
            file_store=None,
            artifact_store=None,
            created_by=sender,
            initiator=person_initiator(sender),
        )
    )
    assert result.pending_id is not None
    assert delivered["pending_id"] == result.pending_id
    return result.pending_id


def _mirror(stores: dict[str, Any], thread: str, text: str, forwarder: str = "alice") -> Any:
    conversations = stores["conversation_store"]
    asyncio.run(
        orchestration._persist_external_conversation_item(
            thread,
            conversations.get_conversation(thread),
            _typed_prompt(thread, text),
            conversations,
            created_by=forwarder,
        )
    )
    return _turns(stores, thread)[-1]


@pytest.fixture
def quiet_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("omnigent.runtime.session_stream.publish", Mock())


@pytest.mark.parametrize("order", ["mirror_first", "registration_first"])
def test_a_native_message_without_a_client_id_keeps_its_sender(
    people: Any, monkeypatch: pytest.MonkeyPatch, quiet_stream: None, order: str
) -> None:
    """R2: every accepted terminal-harness input has a durable identity and sender."""
    _admin, alice, _bob, stores = people
    thread = _session(alice)
    runner = _runner(alice.app, _on_computer(stores, thread, "alice", _READY))
    stores["permission_store"].grant("bob", thread, LEVEL_EDIT)
    pending_id = _native_dispatch(monkeypatch, stores, thread, "rotate the keys", "bob")
    assert get_turn_store().get_input(thread, pending_id).person == "bob"

    if order == "mirror_first":
        mirrored = _mirror(stores, thread, "rotate the keys")
        registered = _register(runner, thread, pending_id).json()
    else:
        registered = _register(runner, thread, pending_id).json()
        mirrored = _mirror(stores, thread, "rotate the keys")

    assert registered["initiator"] == "bob"
    assert mirrored.attribution.initiator.user == "bob"
    assert get_turn_store().get_input(thread, pending_id).item_id == mirrored.id


def test_after_a_restart_or_on_another_replica_the_durable_record_names_the_sender(
    people: Any, monkeypatch: pytest.MonkeyPatch, quiet_stream: None
) -> None:
    """R2: the queued entry in memory is gone; the durable record still names Bob."""
    _admin, alice, _bob, stores = people
    thread = _session(alice)
    runner = _runner(alice.app, _on_computer(stores, thread, "alice", _READY))
    stores["permission_store"].grant("bob", thread, LEVEL_EDIT)
    pending_id = _native_dispatch(monkeypatch, stores, thread, "rotate the keys", "bob")
    # Another process: nothing of this server's memory survives.
    pending_inputs.reset_for_tests()

    assert _register(runner, thread, pending_id).json()["initiator"] == "bob"
    mirrored = _mirror(stores, thread, "rotate the keys")
    initiator = mirrored.attribution.initiator
    assert (initiator.user, initiator.source) == ("bob", "pending_input")
    assert mirrored.created_by == "bob"


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


def _attach_to(client: TestClient, thread: str, terminal: str) -> Any:
    return client.websocket_connect(
        f"/v1/sessions/{thread}/resources/terminals/{terminal}/attach?read_only=false"
    )


def _wait_for_typists(thread: str, expected: list[str]) -> None:
    for _ in range(100):
        if get_turn_store().typists_since(thread, None) == expected:
            return
        time.sleep(0.02)
    assert get_turn_store().typists_since(thread, None) == expected


@pytest.mark.timeout(30)
def test_terminal_typing_is_attributed_to_whoever_sent_input(
    people: Any, idle_terminal: _IdleRunnerSocket, quiet_stream: None
) -> None:
    """R3: two people attached for typing; only the administrator types; the turn names them."""
    admin, alice, _bob, stores = people
    thread = _session(alice)
    _on_computer(stores, thread, "alice", _READY)

    with (
        _attach_to(alice, thread, "terminal_s1"),
        _attach_to(admin, thread, "terminal_s1") as typed,
    ):
        typed.send_bytes(b"restart the service\r")
        _wait_for_typists(thread, ["admin"])
        prompt = _mirror(stores, thread, "restart the service")

    initiator = prompt.attribution.initiator
    assert (initiator.kind, initiator.user, initiator.source) == (
        "terminal",
        "admin",
        "terminal_input",
    )
    assert prompt.created_by == "admin"
    assert _labels(stores, thread)[TURN_ACTOR_LABEL] == "admin"


@pytest.mark.timeout(30)
def test_typing_on_two_terminals_makes_authorship_ambiguous_and_counts_both(
    people: Any, idle_terminal: _IdleRunnerSocket, quiet_stream: None
) -> None:
    """R3: overlapping typists (here on two terminals of one thread) are all possible."""
    admin, alice, _bob, stores = people
    thread = _session(alice)
    _on_computer(stores, thread, "alice", _READY)
    stores["conversation_store"].set_labels(thread, {TURN_ACTOR_LABEL: "alice"})

    with (
        _attach_to(alice, thread, "terminal_s1") as one,
        _attach_to(admin, thread, "terminal_s2") as two,
    ):
        one.send_bytes(b"l")
        two.send_bytes(b"s\r")
        _wait_for_typists(thread, ["admin", "alice"])
        prompt = _mirror(stores, thread, "ls")

    initiator = prompt.attribution.initiator
    assert (initiator.user, initiator.source) == (None, "terminal_ambiguous")
    assert initiator.possible == ["admin", "alice"]
    # The turn's named person is not guessed; both count as taking part.
    assert _labels(stores, thread)[TURN_ACTOR_LABEL] == "alice"
    assert get_turn_store().turn_authority(thread).people >= {"admin", "alice"}


def test_typing_no_server_saw_is_assumed_to_be_the_computer_owners(
    people: Any, quiet_stream: None
) -> None:
    _admin, alice, _bob, stores = people
    thread = _session(alice)
    _on_computer(stores, thread, "alice", _READY)

    prompt = _mirror(stores, thread, "check the status", forwarder="alice")

    initiator = prompt.attribution.initiator
    assert (initiator.user, initiator.source, initiator.possible) == (
        "alice",
        "owner_assumed",
        ["alice"],
    )
    stored = get_turn_store().get_input(thread, prompt.id)
    assert (stored.kind, stored.possible) == (InputKind.TERMINAL, ("alice",))


@pytest.mark.timeout(30)
def test_a_read_only_attachment_records_no_typing(
    people: Any, idle_terminal: _IdleRunnerSocket
) -> None:
    admin, alice, _bob, stores = people
    thread = _session(alice)
    _on_computer(stores, thread, "alice", _READY)
    with _attach(admin, thread, read_only=True) as viewer:
        viewer.send_bytes(b"x\r")
        time.sleep(0.1)
    assert get_turn_store().typists_since(thread, None) == []


class _RecordingRunner(_CaptureRunnerClient):
    """A stand-in runner that keeps every body forwarded to it."""

    def __init__(self) -> None:
        self.bodies: list[dict[str, Any]] = []

    async def post(self, path: str, *, json: dict[str, Any], **kwargs: Any) -> Any:
        self.bodies.append(json)
        return await super().post(path, json=json, **kwargs)


@pytest.fixture
def recording_runner(monkeypatch: pytest.MonkeyPatch) -> _RecordingRunner:
    from omnigent.server.routes import sessions as sessions_mod

    runner = _RecordingRunner()

    async def _client(*_: Any, **__: Any) -> _RecordingRunner:
        return runner

    monkeypatch.setattr(sessions_mod, "_get_runner_client", _client)
    return runner


def test_a_per_message_model_override_is_what_the_turn_records(
    people: Any, recording_runner: _RecordingRunner
) -> None:
    """R5: the record is built from the settings the turn is dispatched with."""
    _admin, alice, _bob, stores = people
    thread = _session(alice)
    _on_computer(stores, thread, "alice", _READY)
    stores["conversation_store"].update_conversation(thread, model_override="model-a")

    assert _send(alice, thread, "go", model_override="model-b").status_code == 202

    [body] = recording_runner.bodies
    [turn] = _turns(stores, thread)
    assert body["model_override"] == "model-b"
    assert turn.attribution.model.requested == "model-b"


def test_an_auto_routed_turn_records_the_harness_and_provider_it_was_routed_to(
    people: Any, recording_runner: _RecordingRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R5: an automatic route into another provider family is what the turn records."""
    from omnigent.server import smart_routing

    async def route(*_args: Any, **_kwargs: Any) -> Any:
        return "codex", "gpt-5.5", {"reason": "coding"}, None

    monkeypatch.setattr(smart_routing, "route_session_harness", route)
    _admin, alice, _bob, stores = people
    thread = _session(alice)
    _on_computer(stores, thread, "alice", _READY)
    stores["conversation_store"].update_conversation(thread, harness_override="auto")

    assert _send(alice, thread, "write the migration").status_code == 202

    [body] = recording_runner.bodies
    [turn] = _turns(stores, thread)
    assert (body["harness_override"], body["model_override"]) == ("codex", "gpt-5.5")
    attribution = turn.attribution
    assert (attribution.harness, attribution.model.requested) == ("codex", "gpt-5.5")
    assert attribution.subscription.provider == "openai"


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
