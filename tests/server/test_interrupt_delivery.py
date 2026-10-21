"""A Stop lost on a dropped runner tunnel is sent again once the runner is back."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Iterator
from typing import Any

import httpx
import pytest

from omnigent.runner.transports.ws_tunnel.frames import HelloFrame
from omnigent.runner.transports.ws_tunnel.registry import TunnelRegistry
from omnigent.runtime import pending_inputs
from omnigent.server.routes import sessions as sessions_module
from omnigent.server.routes._sessions import interrupt_delivery
from omnigent.server.routes._sessions.common import (
    _interrupt_fenced_sessions,
    _session_status_cache,
)

_SESSION = "conv_stop_after_reconnect"
_RUNNER = "runner_0123456789abcdef"


class _NoopWS:
    async def send_text(self, data: str) -> None:
        del data

    async def receive_text(self) -> str:
        return await asyncio.Future()


def _hello() -> HelloFrame:
    return HelloFrame(runner_version="0.1.0", frame_protocol_version=1, harnesses=[], envs=[])


@pytest.fixture
def interrupts(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[httpx.Request]]:
    """Record interrupt forwards reaching the session's runner."""
    posted: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        posted.append(request)
        return httpx.Response(204)

    runner = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://runner")

    async def get_runner_client(session_id: str, runner_router: Any, **_: Any) -> Any:
        del session_id, runner_router
        return runner

    monkeypatch.setattr(sessions_module, "_get_runner_client", get_runner_client)
    _session_status_cache[_SESSION] = "running"
    yield posted
    _session_status_cache.pop(_SESSION, None)
    _interrupt_fenced_sessions.discard(_SESSION)
    interrupt_delivery.cancel_pending_interrupt(_SESSION)


async def _schedule(registry: TunnelRegistry, after_generation: int | None) -> None:
    assert interrupt_delivery.deliver_interrupt_after_reconnect(
        _SESSION,
        runner_id=_RUNNER,
        after_generation=after_generation,
        tunnel_registry=registry,
        runner_router=None,
        queued_pending_ids=[],
        wait_s=2.0,
        stop_generation=interrupt_delivery.input_generation(_SESSION),
        pressed_at=time.monotonic(),
    )
    await asyncio.sleep(0)


async def _settled() -> None:
    for _ in range(200):
        if not interrupt_delivery.has_pending_interrupt(_SESSION):
            return
        await asyncio.sleep(0.01)
    raise AssertionError("the redelivery did not finish")


@pytest.mark.asyncio
async def test_stop_is_sent_once_the_half_open_tunnel_is_replaced(
    interrupts: list[httpx.Request],
) -> None:
    registry = TunnelRegistry()
    stale = registry.register(_RUNNER, _NoopWS(), _hello())
    await _schedule(registry, stale.generation)
    await asyncio.sleep(0.05)
    assert interrupts == [], "the half-open tunnel the Stop was lost on does not count"

    registry.register(_RUNNER, _NoopWS(), _hello())
    await _settled()

    assert len(interrupts) == 1
    assert interrupts[0].url.path == f"/v1/sessions/{_SESSION}/events"
    # The runner applies it only to a turn already running when Stop was pressed.
    body = json.loads(interrupts[0].content)
    assert body["type"] == "interrupt" and body["pressed_ago_s"] >= 0.05
    assert _SESSION in _interrupt_fenced_sessions


@pytest.mark.asyncio
async def test_stop_whose_turn_ended_on_the_runner_is_not_counted_as_delivered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The runner refuses a re-sent Stop for a turn that ended, so nothing is fenced or dropped."""
    posted: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        posted.append(request)
        return httpx.Response(409, json={"status": "turn_ended"})

    runner = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://runner")

    async def get_runner_client(session_id: str, runner_router: Any, **_: Any) -> Any:
        del session_id, runner_router
        return runner

    monkeypatch.setattr(sessions_module, "_get_runner_client", get_runner_client)
    pending_inputs.reset_for_tests()
    pending_id = pending_inputs.record(_SESSION, [{"type": "input_text", "text": "queued"}])
    _session_status_cache[_SESSION] = "running"
    registry = TunnelRegistry()
    try:
        interrupt_delivery.deliver_interrupt_after_reconnect(
            _SESSION,
            runner_id=_RUNNER,
            after_generation=None,
            tunnel_registry=registry,
            runner_router=None,
            queued_pending_ids=[pending_id],
            wait_s=2.0,
            stop_generation=interrupt_delivery.input_generation(_SESSION),
            pressed_at=time.monotonic(),
        )
        registry.register(_RUNNER, _NoopWS(), _hello())
        await _settled()

        assert len(posted) == 1
        assert _SESSION not in _interrupt_fenced_sessions
        assert pending_inputs.has_pending(_SESSION), "the queued message was not stopped"
    finally:
        _session_status_cache.pop(_SESSION, None)
        pending_inputs.reset_for_tests()


@pytest.mark.asyncio
async def test_stop_marks_the_messages_queued_when_it_was_pressed(
    interrupts: list[httpx.Request],
) -> None:
    pending_inputs.reset_for_tests()
    pending_id = pending_inputs.record(_SESSION, [{"type": "input_text", "text": "queued"}])
    registry = TunnelRegistry()
    interrupt_delivery.deliver_interrupt_after_reconnect(
        _SESSION,
        runner_id=_RUNNER,
        after_generation=None,
        tunnel_registry=registry,
        runner_router=None,
        queued_pending_ids=[pending_id],
        wait_s=2.0,
        stop_generation=interrupt_delivery.input_generation(_SESSION),
        pressed_at=time.monotonic(),
    )
    registry.register(_RUNNER, _NoopWS(), _hello())
    await _settled()

    assert len(interrupts) == 1
    assert not pending_inputs.has_pending(_SESSION), "the stopped message no longer counts"
    pending_inputs.reset_for_tests()


@pytest.mark.asyncio
async def test_turn_that_ended_meanwhile_is_not_interrupted(
    interrupts: list[httpx.Request],
) -> None:
    registry = TunnelRegistry()
    await _schedule(registry, None)
    _session_status_cache[_SESSION] = "idle"
    registry.register(_RUNNER, _NoopWS(), _hello())
    await _settled()

    assert interrupts == []


@pytest.mark.asyncio
async def test_new_message_cancels_a_pending_stop(interrupts: list[httpx.Request]) -> None:
    registry = TunnelRegistry()
    await _schedule(registry, None)
    interrupt_delivery.note_new_input(_SESSION)
    registry.register(_RUNNER, _NoopWS(), _hello())
    await asyncio.sleep(0.05)

    assert interrupts == []
    assert not interrupt_delivery.has_pending_interrupt(_SESSION)


@pytest.mark.asyncio
async def test_runner_that_never_returns_gets_no_stop(
    interrupts: list[httpx.Request],
) -> None:
    registry = TunnelRegistry()
    interrupt_delivery.deliver_interrupt_after_reconnect(
        _SESSION,
        runner_id=_RUNNER,
        after_generation=None,
        tunnel_registry=registry,
        runner_router=None,
        queued_pending_ids=[],
        wait_s=0.05,
        stop_generation=interrupt_delivery.input_generation(_SESSION),
        pressed_at=time.monotonic(),
    )
    await _settled()

    assert interrupts == []


@pytest.mark.asyncio
async def test_message_during_the_first_forward_keeps_the_stop_from_being_redelivered(
    interrupts: list[httpx.Request],
) -> None:
    """A Stop pressed before a new message must never interrupt that message's turn.

    The message arrives while the Stop's first forward is still waiting, so no
    redelivery exists yet for it to cancel.
    """
    stop_generation = interrupt_delivery.input_generation(_SESSION)
    interrupt_delivery.note_new_input(_SESSION)
    registry = TunnelRegistry()

    scheduled = interrupt_delivery.deliver_interrupt_after_reconnect(
        _SESSION,
        runner_id=_RUNNER,
        after_generation=None,
        tunnel_registry=registry,
        runner_router=None,
        queued_pending_ids=[],
        wait_s=2.0,
        stop_generation=stop_generation,
        pressed_at=time.monotonic(),
    )
    registry.register(_RUNNER, _NoopWS(), _hello())
    await asyncio.sleep(0.05)

    assert scheduled is False
    assert interrupts == []


@pytest.mark.asyncio
async def test_a_superseded_stop_leaves_a_newer_stops_redelivery_alone(
    interrupts: list[httpx.Request],
) -> None:
    """Stop A, then a message, then Stop B: A failing later must not cancel B."""
    registry = TunnelRegistry()
    stop_a = interrupt_delivery.input_generation(_SESSION)
    interrupt_delivery.note_new_input(_SESSION)
    await _schedule(registry, None)  # Stop B, at the newer generation

    superseded = interrupt_delivery.deliver_interrupt_after_reconnect(
        _SESSION,
        runner_id=_RUNNER,
        after_generation=None,
        tunnel_registry=registry,
        runner_router=None,
        queued_pending_ids=[],
        wait_s=2.0,
        stop_generation=stop_a,
        pressed_at=time.monotonic(),
    )

    assert superseded is False
    assert interrupt_delivery.has_pending_interrupt(_SESSION), "Stop B must stay pending"
    registry.register(_RUNNER, _NoopWS(), _hello())
    await _settled()
    assert len(interrupts) == 1
