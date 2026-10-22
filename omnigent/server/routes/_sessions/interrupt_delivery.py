"""Keep a Stop from interrupting a message sent after it was pressed.

A Stop looks up its runner before it is forwarded. A message that arrives
meanwhile starts a newer turn, which the older Stop must not interrupt, so the
Stop records the session's message count when pressed and is not sent once it
has changed.

A Stop is forwarded once. When its tunnel has dropped, the Stop is reported as
not delivered, or as of unknown delivery, and is not sent again after the
runner reconnects: by then another turn may be running, and the person can
press Interrupt again.
"""

from __future__ import annotations

from omnigent.db.workspace_cache import WorkspaceScopedCache

# Counts each session's new messages, the reference point of a Stop.
_input_generations: WorkspaceScopedCache[str, int] = WorkspaceScopedCache()


def input_generation(session_id: str) -> int:
    """
    Return how many new messages *session_id* has had, as a Stop's reference point.

    :param session_id: Session id, e.g. ``"conv_abc123"``.
    :returns: The current count.
    """
    return _input_generations.get(session_id, 0)


def note_new_input(session_id: str) -> None:
    """
    Record a new message: an earlier Stop must not interrupt the turn it starts.

    :param session_id: Session id, e.g. ``"conv_abc123"``.
    """
    _input_generations[session_id] = input_generation(session_id) + 1
