"""A Stop goes to the runner once, and its answer says whether it got there.

The server's Stop route and tunnel route, and the runner's real tunnel client
and app, run in one process. Only the socket between them is simulated, so it
can be dropped, or hold frames back like a congested link.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

import httpx
import pytest
from asgiref.testing import ApplicationCommunicator
from fastapi import FastAPI
from websockets.exceptions import ConnectionClosedError

from omnigent.db.utils import generate_agent_id
from omnigent.entities import Agent
from omnigent.runner import create_runner_app
from omnigent.runner.transports.ws_tunnel import serve as serve_module
from omnigent.runtime import get_artifact_store
from omnigent.server.bundles import bundle_location
from omnigent.server.routes import runner_tunnel, sessions
from tests.budgets import budget
from tests.runner.conftest import _FakeProcessManager, _ScriptedHarnessClient
from tests.runner.helpers import NullServerClient
from tests.server.helpers import build_agent_bundle
from tests.server.integration.test_runner_tunnel_route import _websocket_scope

pytestmark = pytest.mark.asyncio

_RUNNER_ID = "runner-stop-over-tunnel"
_TUNNEL_PATH = f"/v1/runners/{_RUNNER_ID}/tunnel"

# Signature of a runner ASGI app.
_App = Callable[[dict[str, Any], Any, Any], Awaitable[None]]


class _Link:
    """The socket between the runner and the server's tunnel route.

    :param server_app: Server ASGI app owning the tunnel route.
    :param delay_s: How long each server-to-runner frame is held, e.g. ``1.0``.
    """

    def __init__(self, server_app: FastAPI, *, delay_s: float) -> None:
        self._communicator = ApplicationCommunicator(server_app, _websocket_scope(_TUNNEL_PATH))
        self._delay_s = delay_s
        self._cut = asyncio.Event()

    async def open(self) -> None:
        await self._communicator.send_input({"type": "websocket.connect"})
        accepted = await self._communicator.receive_output(timeout=budget(1.0))
        assert accepted["type"] == "websocket.accept", accepted

    async def send(self, data: str) -> None:
        if not self._cut.is_set():
            await self._communicator.send_input({"type": "websocket.receive", "text": data})

    async def recv(self) -> str:
        received = asyncio.ensure_future(self._communicator.receive_output(timeout=3600))
        cut = asyncio.ensure_future(self._cut.wait())
        done, _ = await asyncio.wait({received, cut}, return_when=asyncio.FIRST_COMPLETED)
        if cut in done:
            received.cancel()
            raise ConnectionClosedError(None, None)
        cut.cancel()
        message = received.result()
        if message["type"] != "websocket.send":
            raise ConnectionClosedError(None, None)
        await asyncio.sleep(self._delay_s)
        return message["text"]

    async def cut(self) -> None:
        """Drop the connection on both ends, as a network outage does."""
        if self._cut.is_set():
            return
        self._cut.set()
        await self._communicator.send_input({"type": "websocket.disconnect", "code": 1006})
        with contextlib.suppress(asyncio.TimeoutError):
            await self._communicator.wait(timeout=budget(1.0))


@asynccontextmanager
async def _runner_connection(
    runner_app: _App,
    server_app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
    *,
    delay_s: float = 0.0,
) -> AsyncIterator[_Link]:
    """Connect the runner's real tunnel client to the server through one :class:`_Link`."""
    import websockets

    link = _Link(server_app, delay_s=delay_s)

    @asynccontextmanager
    async def _connect(url: str, **kwargs: Any) -> AsyncIterator[_Link]:
        del url, kwargs
        await link.open()
        yield link

    monkeypatch.setattr(websockets, "connect", _connect)
    registry = server_app.state.tunnel_registry
    previous = registry.get(_RUNNER_ID)
    serving = asyncio.create_task(
        serve_module._serve_tunnel_once(
            runner_app,
            tunnel_url=f"ws://testserver{_TUNNEL_PATH}",
            server_url="http://testserver",
            runner_id=_RUNNER_ID,
            runner_version="0.1.0",
        )
    )
    try:
        for _ in range(int(budget(2.0) / 0.01)):
            current = registry.get(_RUNNER_ID)
            if current is not None and current is not previous:
                break
            await asyncio.sleep(0.01)
        else:
            pytest.fail("the runner's connection never registered")
        yield link
    finally:
        await link.cut()
        serving.cancel()
        with contextlib.suppress(asyncio.CancelledError, ConnectionClosedError):
            await serving


def _interrupts_reaching(app: _App, answers: list[int]) -> _App:
    """Wrap the runner app to record the status of each event it answers."""

    async def wrapped(scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http" or not scope["path"].endswith("/events"):
            await app(scope, receive, send)
            return

        async def recording_send(event: dict[str, Any]) -> None:
            if event["type"] == "http.response.start":
                answers.append(event["status"])
            await send(event)

        await app(scope, receive, recording_send)

    return wrapped


async def _no_recovery(hook: Any, connection: Any) -> None:
    """Skip reconnect recovery: these tests drive only the Stop."""
    del hook, connection


@pytest.fixture
def agent(app: FastAPI) -> Agent:
    agent_id = generate_agent_id()
    bundle = build_agent_bundle("test")
    location = bundle_location(agent_id, bundle)
    get_artifact_store().put(location, bundle)
    return app.state.agent_store.create(agent_id, "test", location)


@pytest.fixture
async def stop_session(
    app: FastAPI, agent: Agent, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[str]:
    """A running session bound to the runner, cleaned up afterwards."""
    monkeypatch.setattr(runner_tunnel, "_run_connect_hook", _no_recovery)
    monkeypatch.setattr("omnigent.cli_auth.load_databricks_org_id", lambda _server_url: None)
    store = app.state.runner_router._conversation_store
    session_id = store.create_conversation(agent_id=agent.id, runner_id=_RUNNER_ID).id
    sessions._session_status_cache[session_id] = "running"
    yield session_id
    sessions._session_status_cache.pop(session_id, None)
    sessions._interrupt_fenced_sessions.discard(session_id)
    for task in asyncio.all_tasks():
        if task.get_name() == f"runner-disconnect-grace-{_RUNNER_ID}":
            task.cancel()


def _runner_with_turn(session_id: str) -> tuple[Any, asyncio.Task[None]]:
    """A runner app with one turn running for *session_id*."""
    runner_app = create_runner_app(
        process_manager=_FakeProcessManager(_ScriptedHarnessClient([])),  # type: ignore[arg-type]
        server_client=NullServerClient(),  # type: ignore[arg-type]
    )
    turn = asyncio.create_task(asyncio.Event().wait())
    runner_app.state.begin_turn_slot(session_id)
    runner_app.state.active_turns[session_id] = turn
    return runner_app, turn


async def _until(predicate: Callable[[], bool], timeout_s: float) -> None:
    for _ in range(int(budget(timeout_s) / 0.01)):
        if predicate():
            return
        await asyncio.sleep(0.01)


async def _stop(client: httpx.AsyncClient, session_id: str) -> dict[str, Any]:
    response = await client.post(
        f"/v1/sessions/{session_id}/events", json={"type": "interrupt", "data": {}}
    )
    assert response.status_code == 202, response.text
    return response.json()


async def test_stop_lost_with_the_tunnel_never_reaches_a_later_turn(
    app: FastAPI, stop_session: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stop pressed while the runner is away; a later turn begins; the runner reconnects.

    Nothing is sent again, so no Stop reaches the later turn however slowly the
    reconnected link delivers; pressing Stop again stops it.
    """
    runner_app, turn_pressed_for = _runner_with_turn(stop_session)
    answers: list[int] = []
    later = asyncio.create_task(asyncio.Event().wait())
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            async with _runner_connection(
                _interrupts_reaching(runner_app, answers), app, monkeypatch
            ):
                pass  # The tunnel drops with the turn still running.

            lost = await _stop(client, stop_session)

            # The turn ends during the outage and another begins, e.g. a woken parent's.
            runner_app.state.active_turns.pop(stop_session)
            runner_app.state.begin_turn_slot(stop_session)
            runner_app.state.active_turns[stop_session] = later

            async with _runner_connection(
                _interrupts_reaching(runner_app, answers), app, monkeypatch, delay_s=1.0
            ):
                await _until(lambda: bool(answers), 2.5)
                assert answers == [], "no Stop is sent again after the reconnect"
                assert not later.done() and later.cancelling() == 0
                assert lost == {"queued": False, "delivery": "not_delivered"}

                again = await _stop(client, stop_session)

            assert again == {"queued": False, "delivery": "delivered"}
            assert answers == [204]
            assert later.cancelled() or later.cancelling()
            assert not turn_pressed_for.cancelling()
    finally:
        for turn in (turn_pressed_for, later):
            turn.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await turn


async def test_stop_whose_answer_is_lost_is_reported_unknown(
    app: FastAPI, stop_session: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The runner applies the Stop, then the tunnel drops before its answer gets back."""
    runner_app, turn = _runner_with_turn(stop_session)
    answers: list[int] = []
    links: list[_Link] = []

    async def drop_the_answer(scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http" or not scope["path"].endswith("/events"):
            await runner_app(scope, receive, send)
            return

        async def lost_send(event: dict[str, Any]) -> None:
            if event["type"] == "http.response.start":
                answers.append(event["status"])
                await links[0].cut()

        await runner_app(scope, receive, lost_send)

    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            async with _runner_connection(drop_the_answer, app, monkeypatch) as link:
                links.append(link)
                answer = await _stop(client, stop_session)

            assert answer == {"queued": False, "delivery": "unknown"}
            assert answers == [204], "the runner took the Stop and applied it"
            assert turn.cancelled() or turn.cancelling()
            assert stop_session not in sessions._interrupt_fenced_sessions

            async with _runner_connection(
                _interrupts_reaching(runner_app, answers), app, monkeypatch
            ):
                await asyncio.sleep(0.2)
            assert answers == [204], "the Stop is not sent again after the reconnect"
    finally:
        turn.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await turn
