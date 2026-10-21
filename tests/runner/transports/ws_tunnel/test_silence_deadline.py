"""The runner tunnel gives up on a silent server and reconnects."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import pytest
from websockets.asyncio.server import ServerConnection, serve

from omnigent.runner.transports.ws_tunnel import serve as serve_module
from omnigent.runner.transports.ws_tunnel.frames import (
    HelloFrame,
    PingFrame,
    PongFrame,
    decode_frame,
    encode_frame,
)
from omnigent.runner.transports.ws_tunnel.serve import _serve_tunnel_once, serve_tunnel
from omnigent.util import tunnel_silence
from omnigent.util.tunnel_silence import TunnelSilenceTimeoutError
from tests.budgets import budget

_DEADLINE_S = 0.3
_GRACE_S = 0.05


@pytest.fixture(autouse=True)
def _short_silence_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(serve_module, "tunnel_silence_timeout_s", lambda: _DEADLINE_S)
    monkeypatch.setattr(tunnel_silence, "TUNNEL_SILENCE_GRACE_S", _GRACE_S)


async def _no_requests_app(_scope: Any, _receive: Any, _send: Any) -> None:
    raise AssertionError("No application request is expected")


async def _serve_once(port: int, *, shutdown_signal: bool) -> Any:
    return await _serve_tunnel_once(
        _no_requests_app,
        tunnel_url=f"ws://127.0.0.1:{port}/tunnel",
        server_url=f"http://127.0.0.1:{port}",
        runner_id="runner-silence",
        runner_version="0.1.0",
        shutdown_event=asyncio.Event() if shutdown_signal else None,
    )


@pytest.mark.parametrize("shutdown_signal", [False, True])
async def test_half_open_server_is_dropped_after_the_deadline(shutdown_signal: bool) -> None:
    async def peer(ws: ServerConnection) -> None:
        assert isinstance(decode_frame(await ws.recv()), HelloFrame)
        # Says nothing more; the library keeps answering protocol pings.
        await ws.wait_closed()

    async with serve(peer, "127.0.0.1", 0) as server:
        started = time.monotonic()
        with pytest.raises(TunnelSilenceTimeoutError):
            await asyncio.wait_for(
                _serve_once(server.sockets[0].getsockname()[1], shutdown_signal=shutdown_signal),
                timeout=budget(5),
            )
    assert time.monotonic() - started >= _DEADLINE_S + _GRACE_S


@pytest.mark.parametrize("shutdown_signal", [False, True])
async def test_server_pings_keep_the_tunnel_open(shutdown_signal: bool) -> None:
    pongs: list[Any] = []

    async def peer(ws: ServerConnection) -> None:
        assert isinstance(decode_frame(await ws.recv()), HelloFrame)
        # Three deadlines' worth of pings, each well inside the deadline.
        for ts in range(6):
            await asyncio.sleep(_DEADLINE_S / 2)
            await ws.send(encode_frame(PingFrame(ts=ts)))
            pongs.append(decode_frame(await ws.recv()))
        await ws.close()

    async with serve(peer, "127.0.0.1", 0) as server:
        await asyncio.wait_for(
            _serve_once(server.sockets[0].getsockname()[1], shutdown_signal=shutdown_signal),
            timeout=budget(10),
        )
    assert pongs == [PongFrame(ts=ts) for ts in range(6)]


async def test_runner_reconnects_after_a_silent_tunnel(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger=serve_module.__name__)
    monkeypatch.setattr("omnigent.cli_auth.load_databricks_org_id", lambda _url: None)
    hellos: list[HelloFrame] = []
    reconnected = asyncio.Event()

    async def peer(ws: ServerConnection) -> None:
        hello = decode_frame(await ws.recv())
        assert isinstance(hello, HelloFrame)
        hellos.append(hello)
        if len(hellos) > 1:
            reconnected.set()
        await ws.wait_closed()

    async with serve(peer, "127.0.0.1", 0) as server:
        runner = asyncio.create_task(
            serve_tunnel(
                _no_requests_app,
                server_url=f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}",
                runner_id="runner-silence-reconnect",
                runner_version="0.1.0",
            )
        )
        try:
            await asyncio.wait_for(reconnected.wait(), timeout=budget(10))
        finally:
            runner.cancel()
            with pytest.raises(asyncio.CancelledError):
                await runner

    assert hellos[0].connection_id != hellos[1].connection_id
    rows = [
        r.attributes
        for r in caplog.records
        if getattr(r, "event_name", None) == "runner_tunnel_disconnected"
    ]
    assert rows[0]["disconnect_reason"] == "silence_timeout"
    assert rows[0]["error_type"] == "TunnelSilenceTimeoutError"
    assert rows[0]["connected"] is True
