"""Remember which messages a runner accepted, so a resent message runs once.

A server forwards a message, the runner accepts it, and the answer is lost on
the way back. The server may then send the same message again: a client
retry carrying the same stable id, or a re-forward of the same persisted
item. The runner answers those as duplicates instead of running the prompt
a second time. The memory lasts as long as the runner process; a new process
starts empty.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping

# Message ids remembered per session. Far above how many messages a retry can
# trail behind: a retry follows its original within seconds.
_ACCEPTED_KEYS_PER_SESSION = 256

DUPLICATE_MESSAGE_STATUS = "duplicate"


def message_keys(body: Mapping[str, object]) -> tuple[str, ...]:
    """
    Return the ids that identify one forwarded message.

    :param body: Forwarded message body, e.g.
        ``{"type": "message", "stable_id": "0f3a...", "persisted_item_id": "item_1"}``.
    :returns: Prefixed ids, e.g. ``("stable:0f3a...", "item:item_1")``; empty when
        the message carries none.
    """
    keys: list[str] = []
    for field, prefix in (
        ("stable_id", "stable"),
        ("input_stable_id", "stable"),
        ("persisted_item_id", "item"),
    ):
        value = body.get(field)
        if isinstance(value, str) and value:
            key = f"{prefix}:{value}"
            if key not in keys:
                keys.append(key)
    return tuple(keys)


class AcceptedMessages:
    """Bounded per-session memory of the message ids a runner accepted."""

    def __init__(self, per_session_limit: int = _ACCEPTED_KEYS_PER_SESSION) -> None:
        self._limit = per_session_limit
        # Each id maps to every id of its message, so forgetting one forgets all.
        self._keys: dict[str, OrderedDict[str, tuple[str, ...]]] = {}

    def seen(self, session_id: str, body: Mapping[str, object]) -> bool:
        """
        Return whether any id of *body* was already accepted for *session_id*.

        :param session_id: Session the message targets, e.g. ``"conv_abc123"``.
        :param body: Forwarded message body.
        :returns: ``True`` for a message this process already accepted.
        """
        accepted = self._keys.get(session_id)
        return accepted is not None and any(key in accepted for key in message_keys(body))

    def remember(self, session_id: str, body: Mapping[str, object]) -> None:
        """
        Record the ids of a message this process accepted.

        :param session_id: Session the message targets, e.g. ``"conv_abc123"``.
        :param body: Forwarded message body.
        """
        keys = message_keys(body)
        if not keys:
            return
        accepted = self._keys.setdefault(session_id, OrderedDict())
        for key in keys:
            accepted[key] = keys
            accepted.move_to_end(key)
        while len(accepted) > self._limit:
            accepted.popitem(last=False)

    def forget(self, session_id: str, body: Mapping[str, object]) -> None:
        """
        Drop a message that never reached the harness, so a resend runs it.

        :param session_id: Session the message targets, e.g. ``"conv_abc123"``.
        :param body: Any body naming one of the message's ids, e.g.
            ``{"input_stable_id": "0f3a..."}``.
        """
        accepted = self._keys.get(session_id)
        if accepted is None:
            return
        for key in message_keys(body):
            for sibling in accepted.get(key, (key,)):
                accepted.pop(sibling, None)

    def forget_session(self, session_id: str) -> None:
        """
        Drop everything remembered for a session the runner released.

        :param session_id: Session id, e.g. ``"conv_abc123"``.
        """
        self._keys.pop(session_id, None)
