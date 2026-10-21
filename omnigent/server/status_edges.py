"""Apply each native status edge once, however often its sender retries it.

A native forwarder posts a session's status edges (``running``, ``idle``,
``failed``) and retries a POST whose answer it did not get. An edge applied
twice publishes a second edge and, for a sub-agent, can hand its result to
the parent a second time. A sender that names each edge with an
``Idempotency-Key`` header, reused across its retries, gets every repeat
answered from the first: a repeat that arrives while the first is still being
applied waits for it.
"""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, TypeVar

from omnigent.db.db_models import current_workspace_id
from omnigent.session_event_batch import IDEMPOTENCY_KEY_HEADER

_ResultT = TypeVar("_ResultT")

_MAX_KEY_LENGTH = 128

# Retries follow their original within minutes; edges are kept well beyond.
_EDGE_TTL_S = 900.0
_MAX_EDGES = 4096


def status_edge_id(headers: Mapping[str, str]) -> str | None:
    """
    Return the id a status edge's sender gave it, if it is usable.

    :param headers: The request's headers, e.g. ``{"Idempotency-Key": "4f0c..."}``.
    """
    value = headers.get(IDEMPOTENCY_KEY_HEADER)
    if value and len(value) <= _MAX_KEY_LENGTH:
        return value
    return None


class StatusEdges:
    """Applied status edges by session and id, kept for a bounded time.

    Used only on the server's event loop.

    :param ttl_s: How long an applied edge is remembered.
    :param max_edges: How many applied edges are remembered at most; edges
        still being applied are always kept.
    :param clock: Monotonic clock, injectable for tests.
    """

    def __init__(
        self,
        *,
        ttl_s: float = _EDGE_TTL_S,
        max_edges: int = _MAX_EDGES,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._ttl_s = ttl_s
        self._max_edges = max_edges
        self._clock = clock
        self._edges: OrderedDict[tuple[int, str, str], tuple[float, asyncio.Future[Any]]] = (
            OrderedDict()
        )

    async def apply_once(
        self, session_id: str, edge_id: str, apply: Callable[[], Awaitable[_ResultT]]
    ) -> _ResultT:
        """
        Apply an edge the first time its id arrives; answer a repeat with that result.

        An application that fails or is cancelled is forgotten, so a repeat
        waiting for it, or the sender's next retry, applies the edge instead.

        :param session_id: Session id, e.g. ``"conv_abc123"``.
        :param edge_id: The sender's id for the edge.
        :param apply: Applies the edge and returns its answer.
        :returns: The answer of the edge's one application.
        """
        key = (current_workspace_id(), session_id, edge_id)
        while True:
            self._expire()
            known = self._edges.get(key)
            if known is None:
                break
            try:
                return await asyncio.shield(known[1])
            except BaseException:
                if not known[1].done():
                    raise  # this repeat itself was cancelled
        outcome: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self._edges[key] = (self._clock(), outcome)
        self._evict_applied()
        try:
            result = await apply()
        except BaseException:
            if self._edges.get(key, (0.0, None))[1] is outcome:
                del self._edges[key]
            outcome.cancel()
            raise
        outcome.set_result(result)
        return result

    def _evict_applied(self) -> None:
        """Forget the oldest applied edges beyond the bound; ones still applying stay."""
        excess = len(self._edges) - self._max_edges
        for key in [key for key, (_at, outcome) in self._edges.items() if outcome.done()]:
            if excess <= 0:
                return
            del self._edges[key]
            excess -= 1

    def _expire(self) -> None:
        horizon = self._clock() - self._ttl_s
        while self._edges:
            key, (applied_at, outcome) = next(iter(self._edges.items()))
            if applied_at >= horizon or not outcome.done():
                return
            del self._edges[key]
