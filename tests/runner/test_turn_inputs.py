"""The runner names the inputs each turn starts with (``session.turn_started``)."""

from __future__ import annotations

from omnigent.runner.turn_inputs import turn_started_event


def test_a_forwarded_message_is_named_by_its_persisted_item() -> None:
    event = turn_started_event(
        [{"persisted_item_id": "item_a", "stable_id": "0" * 32}, {"persisted_item_id": "item_b"}]
    )
    assert event == {
        "type": "session.turn_started",
        "input_item_ids": ["item_a", "item_b"],
        "input_stable_ids": [],
    }


def test_a_terminal_harness_message_is_named_by_its_queued_stable_id() -> None:
    event = turn_started_event([{"stable_id": "a" * 32}, {"input_stable_id": "b" * 32}])
    assert event is not None
    assert event["input_stable_ids"] == ["a" * 32, "b" * 32]


def test_a_turn_without_server_issued_inputs_names_nobody() -> None:
    """A recovery or continuation keeps the server's current person; the runner names no one."""
    assert turn_started_event([{"content": [], "created_by": "mallory"}]) is None
    assert turn_started_event([]) is None
