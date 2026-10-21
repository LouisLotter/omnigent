"""A runner answers a resent message as a duplicate instead of running it twice."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest

from omnigent.runner import create_runner_app
from omnigent.runner.message_acceptance import AcceptedMessages, message_keys
from tests.runner.conftest import (
    _FakeProcessManager,
    _runner_client,
    _ScriptedHarnessClient,
    _sse,
)
from tests.runner.helpers import NullServerClient


def test_message_keys_name_stable_and_persisted_ids() -> None:
    assert message_keys({"stable_id": "a" * 32, "persisted_item_id": "item_1"}) == (
        "stable:" + "a" * 32,
        "item:item_1",
    )
    assert message_keys({"input_stable_id": "b" * 32}) == ("stable:" + "b" * 32,)
    assert message_keys({"content": []}) == ()


def test_remembered_message_is_seen_by_any_of_its_ids() -> None:
    accepted = AcceptedMessages()
    accepted.remember("conv", {"stable_id": "a" * 32, "persisted_item_id": "item_1"})

    assert accepted.seen("conv", {"stable_id": "a" * 32})
    assert accepted.seen("conv", {"persisted_item_id": "item_1"})
    assert not accepted.seen("conv", {"stable_id": "c" * 32})
    assert not accepted.seen("other", {"stable_id": "a" * 32})
    assert not accepted.seen("conv", {"content": []}), "a message without ids is always new"


def test_forgetting_one_id_forgets_the_whole_message() -> None:
    accepted = AcceptedMessages()
    accepted.remember("conv", {"stable_id": "a" * 32, "persisted_item_id": "item_1"})

    accepted.forget("conv", {"input_stable_id": "a" * 32})

    assert not accepted.seen("conv", {"persisted_item_id": "item_1"})


def test_memory_is_bounded_per_session_and_released_with_it() -> None:
    accepted = AcceptedMessages(per_session_limit=2)
    for index in range(3):
        accepted.remember("conv", {"persisted_item_id": f"item_{index}"})

    assert not accepted.seen("conv", {"persisted_item_id": "item_0"})
    assert accepted.seen("conv", {"persisted_item_id": "item_2"})
    accepted.forget_session("conv")
    assert not accepted.seen("conv", {"persisted_item_id": "item_2"})


@pytest.mark.asyncio
@pytest.mark.parametrize("harness", ["claude-native", "openai-agents"])
async def test_resent_message_is_answered_as_a_duplicate(harness: str) -> None:
    """The same message forwarded twice starts one turn."""
    harness_client = _ScriptedHarnessClient(
        [
            _sse({"type": "response.created", "response": {"id": "resp_1"}}),
            _sse({"type": "response.completed", "response": {"id": "resp_1"}}),
        ]
    )
    app = create_runner_app(
        process_manager=_FakeProcessManager(harness_client),  # type: ignore[arg-type]
        server_client=NullServerClient(),  # type: ignore[arg-type]
    )
    body = {
        "type": "message",
        "role": "user",
        "model": "test-agent",
        "content": [{"type": "input_text", "text": "append one line"}],
        "harness": harness,
        "stable_id": "d" * 32,
        "persisted_item_id": "item_lost_answer",
    }
    events = "/v1/sessions/" + "e" * 32 + "/events"
    async with _runner_client(app) as client:
        first = await client.post(events, json=body)
        for _ in range(200):
            if harness_client.posted_bodies:
                break
            await asyncio.sleep(0.01)
        second = await client.post(events, json=body)
        await asyncio.sleep(0.1)

    assert first.status_code == 202
    assert first.json()["status"] != "duplicate"
    assert second.status_code == 202
    assert second.json()["status"] == "duplicate"
    assert len(harness_client.posted_bodies) == 1


@pytest.mark.asyncio
async def test_duplicate_after_the_turn_ended_leaves_the_session_idle() -> None:
    """A resend answered as a duplicate starts no turn bookkeeping."""
    from omnigent.runner.resource_registry import SessionResourceRegistry

    registry = SessionResourceRegistry()
    harness_client = _ScriptedHarnessClient(
        [
            _sse({"type": "response.created", "response": {"id": "resp_1"}}),
            _sse({"type": "response.completed", "response": {"id": "resp_1"}}),
        ]
    )
    app = create_runner_app(
        process_manager=_FakeProcessManager(harness_client),  # type: ignore[arg-type]
        server_client=NullServerClient(),  # type: ignore[arg-type]
        resource_registry=registry,
    )
    session_id = "c" * 32
    body = {
        "type": "message",
        "role": "user",
        "model": "test-agent",
        "content": [{"type": "input_text", "text": "once"}],
        "harness": "openai-agents",
        "stable_id": "e" * 32,
    }
    async with _runner_client(app) as client:
        await client.post(f"/v1/sessions/{session_id}/events", json=body)
        for _ in range(200):
            if session_id not in app.state.active_turns:
                break
            await asyncio.sleep(0.01)
        epoch = registry.session_activity_epoch(session_id)
        resent = await client.post(f"/v1/sessions/{session_id}/events", json=body)

    assert resent.json()["status"] == "duplicate"
    assert registry.session_activity_epoch(session_id) == epoch
    assert not registry.session_turn_is_active(session_id)


@pytest.mark.asyncio
async def test_message_the_harness_refused_runs_when_resent() -> None:
    """A message that never reached the harness is not remembered as accepted."""

    class RefusesFirstHarnessClient(_ScriptedHarnessClient):
        refusals = 1

        @contextlib.asynccontextmanager
        async def stream(
            self, method: str, url: str, *, json: dict[str, Any], timeout: Any
        ) -> AsyncIterator[httpx.Response]:
            if type(self).refusals:
                type(self).refusals -= 1
                yield httpx.Response(503)
                return
            async with super().stream(method, url, json=json, timeout=timeout) as response:
                yield response

    harness_client = RefusesFirstHarnessClient(
        [
            _sse({"type": "response.created", "response": {"id": "resp_1"}}),
            _sse({"type": "response.completed", "response": {"id": "resp_1"}}),
        ]
    )
    app = create_runner_app(
        process_manager=_FakeProcessManager(harness_client),  # type: ignore[arg-type]
        server_client=NullServerClient(),  # type: ignore[arg-type]
    )
    body = {
        "type": "message",
        "role": "user",
        "model": "test-agent",
        "content": [{"type": "input_text", "text": "try again"}],
        "harness": "openai-agents",
        "stable_id": "f" * 32,
    }
    events = "/v1/sessions/" + "9" * 32 + "/events"
    async with _runner_client(app) as client:
        first = await client.post(events, json=body)
        await asyncio.sleep(0.2)
        resent = await client.post(events, json=body)
        for _ in range(200):
            if harness_client.posted_bodies:
                break
            await asyncio.sleep(0.01)

    assert first.status_code == 202
    assert resent.status_code == 202
    assert resent.json()["status"] != "duplicate"
    assert len(harness_client.posted_bodies) == 1
