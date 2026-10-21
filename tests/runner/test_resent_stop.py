"""Tests for applying a re-sent Stop only to the turn it was pressed for."""

from __future__ import annotations

import pytest

from omnigent.runner.resent_stop import resent_stop_outlived_its_turn


@pytest.mark.parametrize(
    ("event", "turn_active", "turn_started_at", "outlived"),
    [
        ({"type": "interrupt"}, False, None, False),
        ({"type": "interrupt", "pressed_ago_s": True}, True, 900.0, False),
        ({"type": "interrupt", "pressed_ago_s": -1}, True, 990.0, False),
        ({"type": "interrupt", "pressed_ago_s": 30}, True, 900.0, False),
        ({"type": "interrupt", "pressed_ago_s": 30.0}, True, 980.0, True),
        # Began just before the press as the runner places it, which is late
        # by the Stop's transit: it may have begun after the actual press.
        ({"type": "interrupt", "pressed_ago_s": 30.0}, True, 969.8, True),
        # Pressed two seconds into its turn, as people do.
        ({"type": "interrupt", "pressed_ago_s": 30.0}, True, 968.0, False),
        ({"type": "interrupt", "pressed_ago_s": 30.0}, False, 900.0, True),
        ({"type": "interrupt", "pressed_ago_s": 30.0}, True, None, True),
    ],
    ids=[
        "first-delivery",
        "malformed",
        "negative",
        "turn-running-since-before-the-press",
        "turn-began-after-the-press",
        "turn-began-within-the-transit-allowance",
        "pressed-soon-after-its-turn-began",
        "turn-ended",
        "turn-start-unknown",
    ],
)
def test_resent_stop_applies_only_to_a_turn_running_when_it_was_pressed(
    event: dict[str, object],
    turn_active: bool,
    turn_started_at: float | None,
    outlived: bool,
) -> None:
    """A first delivery always applies; a re-sent one only to a turn older than its press."""
    assert (
        resent_stop_outlived_its_turn(
            event, turn_active=turn_active, turn_started_at=turn_started_at, now=1000.0
        )
        is outlived
    )
