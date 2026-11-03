"""Tell the server which inputs each turn starts with, as best effort.

The runner names the inputs a turn starts with (only ids the server issued:
the persisted item id of a forwarded message, or a queued terminal-harness
message's pending id) so the server can record whose turn it is. The runner
never names a person; the server reads the senders from its own records.

Attribution is a display (D59): registration runs beside the turn, retries
briefly and logs a failure. It never delays or fails the turn.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Awaitable, Callable, Iterable, Mapping
from typing import Any

import httpx

_logger = logging.getLogger(__name__)

TURN_INPUTS_KEY = "_omnigent_turn_inputs"
"""Body key carrying a turn's input ids from where it starts to its registration."""

_ATTEMPTS = 3
_BACKOFF_S = (0.25, 1.0)


def turn_input_ids(bodies: Iterable[Mapping[str, object]]) -> list[str]:
    """The server-issued ids of the inputs a turn starts with, in order."""
    ids: list[str] = []
    for body in bodies:
        for key in ("persisted_item_id", "pending_id"):
            value = body.get(key)
            if isinstance(value, str) and value:
                ids.append(value)
                break
    return list(dict.fromkeys(ids))


async def register_turn(
    server_client: Any,
    conversation_id: str,
    input_ids: list[str],
    *,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> bool:
    """Record a turn on the server; ``True`` once it is acknowledged.

    A lost or failed request is retried with the same turn id, which the server
    treats as the same turn. A refusal, an older server without the route, or
    an unreachable server ends it with ``False`` (logged); never raises.
    """
    turn_id = f"turn_{uuid.uuid4().hex}"
    payload = {"turn_id": turn_id, "input_ids": input_ids}
    reason = "no answer"
    for attempt in range(_ATTEMPTS):
        try:
            response = await server_client.post(
                f"/v1/sessions/{conversation_id}/turns", json=payload, timeout=15.0
            )
        except (httpx.HTTPError, RuntimeError) as exc:
            reason = type(exc).__name__
        else:
            status = getattr(response, "status_code", 500)
            if 200 <= status < 300:
                return True
            reason = f"HTTP {status}"
            if status < 500:
                break
        if attempt < _ATTEMPTS - 1:
            await sleep(_BACKOFF_S[min(attempt, len(_BACKOFF_S) - 1)])
    _logger.info(
        "Turn not registered for %s (%s); its attribution may be incomplete",
        conversation_id,
        reason,
        extra={"session_id": conversation_id},
    )
    return False


__all__ = ["TURN_INPUTS_KEY", "register_turn", "turn_input_ids"]
