"""Bounds for host-side long-polls to the Omnigent server.

Approval and policy long-polls park on the server until a person answers.
A keep-alive connection can go half-open on the way: the request never
returns and no close arrives, so a poll without a finite read timeout hangs
the turn. Each poll therefore re-posts with its stable id after
:data:`LONG_POLL_READ_TIMEOUT_S` without an answer. The server re-parks the
same request and hands back an answer given in the gap, which it keeps for
five minutes, so a period well inside that loses nothing.
"""

from __future__ import annotations

# Below the server's 300 s memory of a gap answer, and short enough that an
# answer given during a dead poll reaches the CLI within about a minute.
LONG_POLL_READ_TIMEOUT_S = 45.0
