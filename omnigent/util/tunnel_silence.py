"""Client-side silence deadline for the host and runner tunnels.

The server sends an application ping on both tunnels every
:data:`SERVER_APP_PING_INTERVAL_S` seconds, so a healthy tunnel is never silent
for long. A path that keeps answering WebSocket protocol pings while dropping
data (a proxy, NAT or egress gateway holding a half-open connection) defeats
the protocol keepalive, and the tunnel looks alive forever. The silence deadline
catches that: no message from the server within :func:`tunnel_silence_timeout_s`,
then one short grace read, then the socket is aborted and the caller's reconnect
loop takes over.

The grace read matters when this process was paused (SIGSTOP, a frozen cgroup,
a long event-loop stall). On resume the deadline can fire before the messages
that queued during the pause are read; the grace read gives them one more turn.

Liveness counts whole messages. A single server message that takes longer than
the deadline to arrive, such as a large payload on a very slow link, also trips
it; such deployments raise the deadline or disable it.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import math
import os
from typing import Protocol

from websockets.exceptions import ConnectionClosedError
from websockets.typing import Data

_logger = logging.getLogger(__name__)

TUNNEL_SILENCE_TIMEOUT_ENV_VAR = "OMNIGENT_TUNNEL_SILENCE_TIMEOUT_S"
# The server's application ping cadence on both tunnels (``PING_INTERVAL_S`` in
# ``server/routes/host_tunnel.py`` and ``runner_tunnel.py``).
SERVER_APP_PING_INTERVAL_S = 30.0
# One and a half ping intervals: never reached by a healthy tunnel, and well
# inside the server's own 90 s liveness budget.
TUNNEL_SILENCE_TIMEOUT_S = 45.0
TUNNEL_SILENCE_GRACE_S = 1.0


class _TunnelSocket(Protocol):
    """The part of a ``websockets`` client connection the deadline needs."""

    transport: asyncio.Transport

    async def recv(self) -> Data:
        """Return the next message; cancelling it must not lose one."""
        ...


class TunnelSilenceTimeoutError(ConnectionClosedError):
    """No message arrived within the silence deadline, so the socket was aborted.

    A :class:`~websockets.exceptions.ConnectionClosedError`, so reconnect paths
    treat it like any other abnormal drop.

    :param closed: The error the aborted connection raised.
    :param silent_s: Seconds without a message, deadline plus grace, e.g. ``46.0``.
    """

    def __init__(self, closed: ConnectionClosedError, silent_s: float) -> None:
        super().__init__(closed.rcvd, closed.sent, closed.rcvd_then_sent)
        self.silent_s = silent_s

    def __str__(self) -> str:
        return f"no message from the server for {self.silent_s:g}s; tunnel presumed dead"


@functools.cache
def tunnel_silence_timeout_s() -> float | None:
    """Return the tunnel silence deadline in seconds, or ``None`` when disabled.

    Read from :data:`TUNNEL_SILENCE_TIMEOUT_ENV_VAR`: ``0`` disables the deadline,
    and any other value must be a number of seconds above
    :data:`SERVER_APP_PING_INTERVAL_S`, or every healthy tunnel would trip it.
    An invalid value is ignored with a warning instead of raised, so a typo
    cannot stop a host or runner from connecting.

    Cached, so the warning is logged once; tests call ``cache_clear()``.

    :returns: Seconds, e.g. ``45.0``, or ``None`` when disabled.
    """
    raw = os.environ.get(TUNNEL_SILENCE_TIMEOUT_ENV_VAR)
    if raw is None or not raw.strip():
        return TUNNEL_SILENCE_TIMEOUT_S
    try:
        value = float(raw)
    except ValueError:
        value = math.nan
    if value == 0:
        return None
    # Also rejects the "nan" and "inf" that ``float()`` accepts.
    if not (math.isfinite(value) and value > SERVER_APP_PING_INTERVAL_S):
        _logger.warning(
            "%s must be 0 (disabled) or more than the server's %gs ping interval, "
            "got %r; using %gs",
            TUNNEL_SILENCE_TIMEOUT_ENV_VAR,
            SERVER_APP_PING_INTERVAL_S,
            raw,
            TUNNEL_SILENCE_TIMEOUT_S,
        )
        return TUNNEL_SILENCE_TIMEOUT_S
    return value


async def recv_within_silence_deadline(
    ws: _TunnelSocket, timeout_s: float | None, *, grace_s: float | None = None
) -> Data:
    """Return the next message, aborting the socket if the server falls silent.

    After *timeout_s* without a message, one more read of *grace_s* runs before
    the socket is aborted. Cancelling ``recv()`` loses no message in
    ``websockets``, so a message that arrives as either read times out is
    returned by the next one.

    :param ws: An open ``websockets`` client connection.
    :param timeout_s: Seconds to wait for a message, e.g. ``45.0``, or ``None``
        to wait indefinitely like ``ws.recv()``.
    :param grace_s: Length of the grace read; defaults to
        :data:`TUNNEL_SILENCE_GRACE_S`.
    :returns: The next message.
    :raises TunnelSilenceTimeoutError: When neither read returns a message.
    :raises websockets.exceptions.ConnectionClosed: When the connection closes
        for another reason.
    """
    if timeout_s is None:
        return await ws.recv()
    grace = TUNNEL_SILENCE_GRACE_S if grace_s is None else grace_s
    for budget_s in (timeout_s, grace):
        try:
            async with asyncio.timeout(budget_s):
                return await ws.recv()
        except TimeoutError:
            continue
    ws.transport.abort()
    try:
        # Raises once the aborted connection has closed.
        return await ws.recv()
    except ConnectionClosedError as exc:
        raise TunnelSilenceTimeoutError(exc, timeout_s + grace) from exc
