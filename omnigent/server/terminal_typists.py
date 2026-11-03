"""Who is attached to a thread's terminal with typing rights, in this process.

Keystrokes carry no identity, so a prompt typed into a terminal harness is
attributed to the person attached to that terminal with typing rights (the
owner, or an administrator under the human override). With several attached,
the most recent attachment is named. A typed prompt nobody here is attached
for (a direct local attach, or an attachment held by another server replica)
is assumed to be the thread owner's and recorded as assumed.
"""

from __future__ import annotations

import itertools
import threading

from omnigent.db.workspace_cache import WorkspaceScopedCache

_lock = threading.Lock()
_tokens = itertools.count(1)
# Per session: (attachment token, person), oldest first.
_attached: WorkspaceScopedCache[str, list[tuple[int, str]]] = WorkspaceScopedCache()


def attach(session_id: str, user_id: str) -> int:
    """Record a typing attachment; returns the token :func:`detach` takes."""
    with _lock:
        token = next(_tokens)
        _attached[session_id] = [*_attached.get(session_id, []), (token, user_id)]
        return token


def detach(session_id: str, token: int) -> None:
    """Forget a typing attachment; unknown tokens are ignored."""
    with _lock:
        remaining = [entry for entry in _attached.get(session_id, []) if entry[0] != token]
        if remaining:
            _attached[session_id] = remaining
        else:
            _attached.pop(session_id, None)


def typist(session_id: str) -> str | None:
    """The person most recently attached with typing rights, or ``None``."""
    with _lock:
        entries = _attached.get(session_id, [])
        return entries[-1][1] if entries else None


def reset_for_tests() -> None:
    """Forget the current workspace's attachments."""
    with _lock:
        sessions = _attached.keys()
        for session_id in sessions:
            del _attached[session_id]


__all__ = ["attach", "detach", "reset_for_tests", "typist"]
