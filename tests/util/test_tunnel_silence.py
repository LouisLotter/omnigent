"""The tunnel silence deadline against real WebSocket peers.

The peers answer protocol pings, as a half-open proxy path does, so these tests
show that pongs are not liveness. A blocked event loop stands in for a paused
process: on resume, the expired deadline and the socket data that queued during
the pause are handled together, which is the case the grace read exists for.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager, suppress
from types import SimpleNamespace
from typing import Any

import pytest
from websockets.asyncio.client import ClientConnection, connect
from websockets.asyncio.server import ServerConnection, serve
from websockets.exceptions import ConnectionClosed, ConnectionClosedError, ConnectionClosedOK
from websockets.frames import Close
from websockets.protocol import State
from websockets.sync.server import ServerConnection as ThreadPeerConnection
from websockets.sync.server import serve as serve_in_thread
from websockets.typing import Data

from omnigent.util import tunnel_silence
from omnigent.util.tunnel_silence import (
    TUNNEL_SILENCE_TIMEOUT_ENV_VAR,
    TUNNEL_SILENCE_TIMEOUT_S,
    TunnelSilenceTimeoutError,
    recv_within_silence_deadline,
    tunnel_silence_timeout_s,
)
from tests.budgets import budget

_DEADLINE_S = 0.2
_GRACE_S = 0.1


class _CountingSocket:
    """Delegate to a client connection, counting ``recv()`` calls."""

    def __init__(self, ws: ClientConnection) -> None:
        self._ws = ws
        self.transport = ws.transport
        self.reads = 0

    async def recv(self) -> Data:
        self.reads += 1
        return await self._ws.recv()


@asynccontextmanager
async def _client_for(
    handler: Callable[[ServerConnection], Any],
) -> AsyncIterator[ClientConnection]:
    """Connect to an in-loop peer, with client protocol pings flowing throughout."""
    async with (
        serve(handler, "127.0.0.1", 0) as server,
        connect(
            f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}",
            ping_interval=0.05,
            ping_timeout=budget(10),
        ) as ws,
    ):
        yield ws


@asynccontextmanager
async def _client_for_thread_peer(
    handler: Callable[[ThreadPeerConnection], None],
) -> AsyncIterator[ClientConnection]:
    """Connect to a peer on its own thread, which can send while this loop is blocked."""
    with serve_in_thread(handler, "127.0.0.1", 0) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            async with connect(
                f"ws://127.0.0.1:{server.socket.getsockname()[1]}", ping_interval=None
            ) as ws:
                yield ws
        finally:
            server.shutdown()
            thread.join(timeout=budget(5))


def _hold_open(ws: ThreadPeerConnection) -> None:
    """Keep a thread peer's connection open, echoing until the client goes away."""
    with suppress(ConnectionClosed):
        while True:
            ws.send(f"echo: {ws.recv()}")


async def _assert_still_open(ws: ClientConnection) -> None:
    """The tunnel survived: it still carries messages both ways."""
    await ws.send("ping")
    assert await recv_within_silence_deadline(ws, budget(5), grace_s=_GRACE_S) == "echo: ping"
    assert ws.state is State.OPEN


async def test_returns_messages_that_arrive_within_the_deadline() -> None:
    async def peer(ws: ServerConnection) -> None:
        await ws.send("first")
        await asyncio.sleep(_DEADLINE_S / 2)
        await ws.send("second")
        await ws.wait_closed()

    async with _client_for(peer) as ws:
        assert await recv_within_silence_deadline(ws, _DEADLINE_S, grace_s=_GRACE_S) == "first"
        assert await recv_within_silence_deadline(ws, _DEADLINE_S, grace_s=_GRACE_S) == "second"
        assert ws.state is State.OPEN


async def test_half_open_peer_is_aborted_after_the_deadline_and_grace() -> None:
    peer_closed = asyncio.Event()

    async def peer(ws: ServerConnection) -> None:
        # Never sends a message, while the library answers every ping.
        await ws.wait_closed()
        peer_closed.set()

    async with _client_for(peer) as ws:
        started = time.monotonic()
        with pytest.raises(TunnelSilenceTimeoutError) as raised:
            await recv_within_silence_deadline(ws, _DEADLINE_S, grace_s=_GRACE_S)
        elapsed = time.monotonic() - started
        assert ws.state is State.CLOSED
        await asyncio.wait_for(peer_closed.wait(), timeout=budget(5))

    assert ws.latency > 0, "the peer should have answered the client's pings"
    assert _DEADLINE_S + _GRACE_S <= elapsed < budget(5)
    assert raised.value.silent_s == pytest.approx(_DEADLINE_S + _GRACE_S)
    assert raised.value.rcvd is None
    assert raised.value.sent is None
    assert str(raised.value) == "no message from the server for 0.3s; tunnel presumed dead"


async def test_message_landing_as_the_grace_read_expires_is_too_late() -> None:
    """Once the tunnel is aborted, the read reports silence, not a stray message."""
    stalled = threading.Event()
    sent = threading.Event()

    def peer(ws: ThreadPeerConnection) -> None:
        assert stalled.wait(timeout=budget(5))
        ws.send("too late")
        sent.set()
        _hold_open(ws)

    def stall_through_the_grace_read() -> None:
        stalled.set()
        assert sent.wait(timeout=budget(5))
        time.sleep(_GRACE_S)

    async with _client_for_thread_peer(peer) as ws:
        asyncio.get_running_loop().call_later(
            _DEADLINE_S + _GRACE_S / 4, stall_through_the_grace_read
        )
        with pytest.raises(TunnelSilenceTimeoutError):
            await recv_within_silence_deadline(ws, _DEADLINE_S, grace_s=_GRACE_S)
        assert ws.state is State.CLOSED


async def test_message_queued_during_a_pause_is_read_by_the_grace_read() -> None:
    """Resume with the deadline expired and the message already in the socket."""
    paused = threading.Event()
    sent = threading.Event()

    def peer(ws: ThreadPeerConnection) -> None:
        assert paused.wait(timeout=budget(5))
        ws.send("queued during the pause")
        sent.set()
        _hold_open(ws)

    def pause() -> None:
        paused.set()
        assert sent.wait(timeout=budget(5))
        time.sleep(_DEADLINE_S)

    async with _client_for_thread_peer(peer) as ws:
        counting = _CountingSocket(ws)
        asyncio.get_running_loop().call_later(_DEADLINE_S / 4, pause)
        message = await recv_within_silence_deadline(counting, _DEADLINE_S, grace_s=budget(5))
        assert message == "queued during the pause"
        # The deadline fired before the message was read; the grace read found it.
        assert counting.reads == 2
        await _assert_still_open(ws)


async def test_message_arriving_just_after_resume_is_read_by_the_grace_read() -> None:
    """Resume with the deadline expired before the message reaches the socket."""
    resumed = threading.Event()

    def peer(ws: ThreadPeerConnection) -> None:
        assert resumed.wait(timeout=budget(5))
        ws.send("sent after the resume")
        _hold_open(ws)

    async with _client_for_thread_peer(peer) as ws:
        counting = _CountingSocket(ws)
        loop = asyncio.get_running_loop()

        def pause() -> None:
            # Runs on the first loop pass after the pause, with the expired deadline.
            loop.call_soon(resumed.set)
            time.sleep(_DEADLINE_S * 2)

        loop.call_later(_DEADLINE_S / 4, pause)
        message = await recv_within_silence_deadline(counting, _DEADLINE_S, grace_s=budget(5))
        assert message == "sent after the resume"
        assert counting.reads == 2
        await _assert_still_open(ws)


async def test_loss_during_a_pause_is_detected_one_grace_read_after_resume() -> None:
    deadline_s, grace_s = 0.5, 0.05
    resumed_at: list[float] = []

    def pause() -> None:
        time.sleep(deadline_s * 2)
        resumed_at.append(time.monotonic())

    # The peer never sends: the tunnel died while this process was paused.
    async with _client_for_thread_peer(_hold_open) as ws:
        asyncio.get_running_loop().call_later(deadline_s / 4, pause)
        with pytest.raises(TunnelSilenceTimeoutError):
            await recv_within_silence_deadline(ws, deadline_s, grace_s=grace_s)
        detected_after_resume_s = time.monotonic() - resumed_at[0]

    # One grace read, not a fresh deadline counted from the resume.
    assert grace_s <= detected_after_resume_s < deadline_s


async def test_message_queued_before_a_loss_restarts_the_deadline() -> None:
    """A server ping queued during the pause is read; the loss then costs a full deadline."""
    paused = threading.Event()
    sent = threading.Event()

    def peer(ws: ThreadPeerConnection) -> None:
        assert paused.wait(timeout=budget(5))
        ws.send("last ping before the loss")
        sent.set()
        _hold_open(ws)

    def pause() -> None:
        paused.set()
        assert sent.wait(timeout=budget(5))
        time.sleep(_DEADLINE_S * 2)

    async with _client_for_thread_peer(peer) as ws:
        asyncio.get_running_loop().call_later(_DEADLINE_S / 4, pause)
        message = await recv_within_silence_deadline(ws, _DEADLINE_S, grace_s=_GRACE_S)
        assert message == "last ping before the loss"
        read_at = time.monotonic()
        with pytest.raises(TunnelSilenceTimeoutError):
            await recv_within_silence_deadline(ws, _DEADLINE_S, grace_s=_GRACE_S)
        assert time.monotonic() - read_at >= _DEADLINE_S + _GRACE_S


async def test_silence_error_keeps_the_close_the_server_sent() -> None:
    """A close whose TCP shutdown stalls past the grace read keeps its code for the caller."""
    rejected = Close(4002, "unsupported frame protocol")

    class _ClosingSocket:
        def __init__(self) -> None:
            self.aborted = asyncio.Event()
            self.transport = SimpleNamespace(abort=self.aborted.set)

        async def recv(self) -> Data:
            await self.aborted.wait()
            raise ConnectionClosedError(rejected, None)

    with pytest.raises(TunnelSilenceTimeoutError) as raised:
        await recv_within_silence_deadline(_ClosingSocket(), _DEADLINE_S, grace_s=_GRACE_S)
    assert raised.value.rcvd == rejected


async def test_disabled_deadline_waits_like_recv() -> None:
    async def peer(ws: ServerConnection) -> None:
        await asyncio.sleep(_DEADLINE_S * 3)
        await ws.send("late")
        await ws.wait_closed()

    async with _client_for(peer) as ws:
        assert await recv_within_silence_deadline(ws, None, grace_s=_GRACE_S) == "late"


async def test_server_close_is_not_reported_as_silence() -> None:
    async def peer(ws: ServerConnection) -> None:
        await ws.close(code=1001, reason="going away")

    async with _client_for(peer) as ws:
        with pytest.raises(ConnectionClosedOK) as raised:
            await recv_within_silence_deadline(ws, _DEADLINE_S, grace_s=_GRACE_S)
    assert not isinstance(raised.value, TunnelSilenceTimeoutError)
    assert raised.value.rcvd is not None
    assert raised.value.rcvd.code == 1001


async def test_cancelled_read_leaves_the_tunnel_open_and_loses_nothing() -> None:
    release = asyncio.Event()

    async def peer(ws: ServerConnection) -> None:
        await release.wait()
        await ws.send("after the cancel")
        await ws.wait_closed()

    async with _client_for(peer) as ws:
        read = asyncio.create_task(recv_within_silence_deadline(ws, budget(5), grace_s=_GRACE_S))
        await asyncio.sleep(_DEADLINE_S / 4)
        read.cancel()
        with pytest.raises(asyncio.CancelledError):
            await read
        assert ws.state is State.OPEN
        release.set()
        message = await recv_within_silence_deadline(ws, budget(5), grace_s=_GRACE_S)
        assert message == "after the cancel"


@pytest.fixture
def set_silence_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[[str | None], None]]:
    """Set or clear the override, with a fresh resolver cache."""

    def _set(value: str | None) -> None:
        if value is None:
            monkeypatch.delenv(TUNNEL_SILENCE_TIMEOUT_ENV_VAR, raising=False)
        else:
            monkeypatch.setenv(TUNNEL_SILENCE_TIMEOUT_ENV_VAR, value)
        tunnel_silence_timeout_s.cache_clear()

    yield _set
    tunnel_silence_timeout_s.cache_clear()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        pytest.param(None, TUNNEL_SILENCE_TIMEOUT_S, id="unset"),
        pytest.param("  ", TUNNEL_SILENCE_TIMEOUT_S, id="blank"),
        pytest.param("60", 60.0, id="seconds"),
        pytest.param("30.5", 30.5, id="just-above-the-ping"),
        pytest.param("0", None, id="disabled"),
        pytest.param("0.0", None, id="disabled-float"),
    ],
)
def test_timeout_setting(
    set_silence_env: Callable[[str | None], None], raw: str | None, expected: float | None
) -> None:
    set_silence_env(raw)
    assert tunnel_silence_timeout_s() == expected


@pytest.mark.parametrize("raw", ["30", "10", "-5", "soon", "nan", "inf"])
def test_invalid_timeout_setting_falls_back_with_one_warning(
    set_silence_env: Callable[[str | None], None],
    raw: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    set_silence_env(raw)
    with caplog.at_level("WARNING", logger=tunnel_silence.__name__):
        assert tunnel_silence_timeout_s() == TUNNEL_SILENCE_TIMEOUT_S
        assert tunnel_silence_timeout_s() == TUNNEL_SILENCE_TIMEOUT_S
    [record] = caplog.records
    assert TUNNEL_SILENCE_TIMEOUT_ENV_VAR in record.getMessage()
    assert repr(raw) in record.getMessage()
