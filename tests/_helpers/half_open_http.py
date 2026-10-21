"""Local HTTP server whose chosen requests are never answered.

Models a keep-alive connection that went half-open on the way to the server:
the request is read, no response and no close ever arrive, and the socket
stays open. Requests after the stalled ones are answered normally, as a new
connection through the same path would be.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

JsonResponder = Callable[[str, str, bytes], tuple[int, object]]


@dataclass
class HalfOpenHttpServer:
    """
    Serve JSON answers on loopback, leaving the first *stalled* requests unanswered.

    :param respond: Builds ``(status, json_body)`` from ``(method, path, body)``.
    :param stalled: How many requests, in arrival order, get no answer.
    """

    respond: JsonResponder
    stalled: int = 0
    url: str = ""
    requests: list[tuple[int, str, str, bytes]] = field(default_factory=list)
    _connections: int = 0
    _writers: list[asyncio.StreamWriter] = field(default_factory=list)
    _closed: asyncio.Event = field(default_factory=asyncio.Event)

    @property
    def connections(self) -> int:
        """Number of connections accepted so far."""
        return self._connections

    async def _serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self._connections += 1
        connection = self._connections
        self._writers.append(writer)
        while True:
            try:
                head = await reader.readuntil(b"\r\n\r\n")
            except (asyncio.IncompleteReadError, ConnectionError):
                return
            request_line, *header_lines = head.decode("latin-1").split("\r\n")
            method, path, _version = request_line.split(" ", 2)
            length = 0
            for line in header_lines:
                name, _, value = line.partition(":")
                if name.strip().lower() == "content-length":
                    length = int(value.strip())
            body = await reader.readexactly(length) if length else b""
            self.requests.append((connection, method, path, body))
            if len(self.requests) <= self.stalled:
                # Half-open: never answer and never close this connection.
                await self._closed.wait()
                return
            status, payload = self.respond(method, path, body)
            raw = json.dumps(payload).encode()
            writer.write(
                f"HTTP/1.1 {status} OK\r\nContent-Type: application/json\r\n"
                f"Content-Length: {len(raw)}\r\n\r\n".encode()
                + raw
            )
            await writer.drain()


@asynccontextmanager
async def half_open_http_server(
    respond: JsonResponder, *, stalled: int = 0
) -> AsyncIterator[HalfOpenHttpServer]:
    """
    Run a :class:`HalfOpenHttpServer` on the current event loop.

    :param respond: Builds ``(status, json_body)`` from ``(method, path, body)``.
    :param stalled: How many requests, in arrival order, get no answer.
    :returns: The running server, with ``url`` set.
    """
    fake = HalfOpenHttpServer(respond=respond, stalled=stalled)
    server = await asyncio.start_server(fake._serve, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    fake.url = f"http://127.0.0.1:{port}"
    try:
        yield fake
    finally:
        fake._closed.set()
        server.close()
        for writer in fake._writers:
            writer.close()
        await server.wait_closed()
