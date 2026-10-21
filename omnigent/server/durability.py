"""Acknowledge a recorded outcome only once its write is durable.

Some peers drop their own copy of an outcome as soon as the server
acknowledges it: a native transcript forwarder forgets an item once its
``event.ack`` frame, or its HTTP answer, arrives. A deployment whose database
replicates asynchronously, such as SQLite with Litestream, loses its latest
commits when the server is replaced, and with them the only copy of an
outcome acknowledged too early.

Such a deployment passes ``create_app(durability_barrier=...)``: an async
callable that returns ``True`` once every commit made before the call is
durable, or ``False`` when that is not confirmed in time. The server awaits it
before those acknowledgements, and otherwise asks the peer to send the outcome
again, which the source-keyed item writes absorb. Without a barrier a commit
counts as durable at once.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any, cast

_logger = logging.getLogger(__name__)

DurabilityBarrier = Callable[[], Awaitable[bool]]


def durability_barrier(app: Any) -> DurabilityBarrier | None:
    """
    Return the barrier the embedding application installed, if any.

    :param app: The FastAPI application, or anything with its ``state``.
    """
    barrier = getattr(getattr(app, "state", None), "durability_barrier", None)
    return cast("DurabilityBarrier", barrier) if callable(barrier) else None


async def commits_durable(barrier: DurabilityBarrier | None) -> bool:
    """
    Wait until every commit made so far is durable.

    :param barrier: The deployment's barrier, or ``None`` when commits are
        durable once committed.
    :returns: ``False`` when durability was not confirmed, including when the
        barrier failed; the caller must not acknowledge the outcome then.
    """
    if barrier is None:
        return True
    try:
        return bool(await barrier())
    except Exception:  # noqa: BLE001 — an unconfirmed write is retried, never acknowledged
        _logger.warning("Could not confirm that recorded writes are durable", exc_info=True)
        return False
