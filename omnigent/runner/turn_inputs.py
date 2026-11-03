"""Name the inputs a turn starts with, so the server records whose turn it is.

The runner reports only ids the server issued: the persisted item id of a
forwarded message, or for a terminal harness the queued message's stable id.
It never names a person; the server reads the sender from its own records.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

TURN_STARTED_EVENT = "session.turn_started"


def turn_started_event(bodies: Iterable[Mapping[str, object]]) -> dict[str, Any] | None:
    """The ``session.turn_started`` event for a turn starting with *bodies*.

    :param bodies: The forwarded message bodies the turn starts with, e.g.
        ``[{"persisted_item_id": "item_1", ...}]``.
    :returns: The event, or ``None`` when no body carries a server-issued id
        (a recovery or continuation turn, which keeps the current person).
    """
    item_ids: list[str] = []
    stable_ids: list[str] = []
    for body in bodies:
        persisted = body.get("persisted_item_id")
        if isinstance(persisted, str) and persisted:
            item_ids.append(persisted)
            continue
        for key in ("stable_id", "input_stable_id"):
            stable = body.get(key)
            if isinstance(stable, str) and stable:
                stable_ids.append(stable)
                break
    if not item_ids and not stable_ids:
        return None
    return {"type": TURN_STARTED_EVENT, "input_item_ids": item_ids, "input_stable_ids": stable_ids}


__all__ = ["TURN_STARTED_EVENT", "turn_started_event"]
