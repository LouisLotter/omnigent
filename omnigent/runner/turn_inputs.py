"""Register each turn with the server before the harness acts on it.

The runner names the inputs a turn starts with (only ids the server issued:
the persisted item id of a forwarded message, or a queued terminal-harness
message's pending id) and waits for the server to record the turn and
acknowledge it. Only then does the harness start, so every policy check and
approval in the turn sees whose turn it is. The runner never names a person;
the server reads the senders from its own records.
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

_ATTEMPTS = 4
_BACKOFF_S = (0.25, 0.5, 1.0)


class TurnRegistrationError(RuntimeError):
    """The server did not record the turn; the harness must not start it."""


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


def _unsupported(response: httpx.Response) -> bool:
    """A server without turn registration (an older build): the turn proceeds as before."""
    if response.status_code not in (404, 405):
        return False
    try:
        body = response.json()
    except ValueError:
        return True
    error = body.get("error") if isinstance(body, dict) else None
    code = error.get("code") if isinstance(error, dict) else None
    return code in (None, "turns_unsupported")


async def register_turn(
    server_client: Any,
    conversation_id: str,
    input_ids: list[str],
    *,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> None:
    """Record a turn on the server and wait for its acknowledgement.

    Retries a lost or failed request with the same turn id, which the server
    treats as the same turn. Raises :class:`TurnRegistrationError` when the
    server refuses it or cannot be reached, so the turn fails instead of
    running unregistered.
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
            if 200 <= status < 300 or _unsupported(response):
                return
            if status < 500:
                raise TurnRegistrationError(
                    f"the server refused to register this turn (HTTP {status})"
                )
            reason = f"HTTP {status}"
        if attempt < _ATTEMPTS - 1:
            await sleep(_BACKOFF_S[min(attempt, len(_BACKOFF_S) - 1)])
    _logger.warning(
        "Turn registration failed for %s: %s",
        conversation_id,
        reason,
        extra={"session_id": conversation_id},
    )
    raise TurnRegistrationError(f"the server could not register this turn ({reason})")


__all__ = [
    "TURN_INPUTS_KEY",
    "TurnRegistrationError",
    "register_turn",
    "turn_input_ids",
]
