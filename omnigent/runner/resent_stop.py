"""Keep a Stop sent again after a dropped tunnel from stopping a later turn.

When the tunnel to the runner drops, the server sends a lost Stop again once
the runner reconnects. By then the turn the person pressed Stop for may have
ended, and another may be running: a queued message's, a woken parent's, or
one typed into the terminal. A re-sent Stop says how long ago it was pressed,
and the runner applies it only to a turn that was already running then.

The two sides share no clock: the runner places the press by subtracting that
duration from its own time, which puts it late by the Stop's transit. A turn
must therefore have started at least the longest transit before that, the
server's timeout for the forward, to count.
"""

from __future__ import annotations

from collections.abc import Mapping

# The interrupt event's field naming how long ago the person pressed Stop.
PRESSED_AGO_FIELD = "pressed_ago_s"

# The runner's answer to a re-sent Stop whose turn has ended.
TURN_ENDED_STATUS = "turn_ended"

# The server gives up on a re-sent Stop's forward after this long, and so
# bounds its transit.
RESENT_STOP_TRANSIT_ALLOWANCE_S = 5.0


def resent_stop_outlived_its_turn(
    event: Mapping[str, object],
    *,
    turn_active: bool,
    turn_started_at: float | None,
    now: float,
) -> bool:
    """
    Return whether a re-sent Stop arrived after the turn it was pressed for.

    :param event: The interrupt event, e.g.
        ``{"type": "interrupt", "pressed_ago_s": 47.2}``. A Stop without
        :data:`PRESSED_AGO_FIELD` is a first delivery and always applies.
    :param turn_active: Whether the session has a turn running now.
    :param turn_started_at: When the latest turn started, on the runner's
        monotonic clock, or ``None`` when none is known.
    :param now: The runner's monotonic time.
    :returns: ``True`` when no turn runs now, or the running one may have
        started after the Stop was pressed.
    """
    pressed_ago_s = event.get(PRESSED_AGO_FIELD)
    if (
        not isinstance(pressed_ago_s, int | float)
        or isinstance(pressed_ago_s, bool)
        or pressed_ago_s < 0
    ):
        return False
    if not turn_active or turn_started_at is None:
        return True
    return turn_started_at > now - pressed_ago_s - RESENT_STOP_TRANSIT_ALLOWANCE_S
