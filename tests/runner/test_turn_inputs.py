"""Each turn is registered with the server beside the turn, as best effort (D59)."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from omnigent.runner.turn_inputs import register_turn, turn_input_ids


def test_a_turns_inputs_are_named_by_server_issued_ids() -> None:
    bodies = [
        {"persisted_item_id": "item_a", "stable_id": "0" * 32},
        {"pending_id": "pending_b", "stable_id": "1" * 32},
        {"content": [], "created_by": "mallory"},
        {"persisted_item_id": "item_a"},
    ]
    assert turn_input_ids(bodies) == ["item_a", "pending_b"]


class _Answer:
    def __init__(self, status: int) -> None:
        self.status_code = status


class _Server:
    def __init__(self, *answers: Any) -> None:
        self.answers = list(answers)
        self.payloads: list[dict[str, Any]] = []

    async def post(self, url: str, *, json: dict[str, Any], timeout: float) -> Any:
        assert url == "/v1/sessions/conv_1/turns"
        self.payloads.append(json)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


async def _no_sleep(_seconds: float) -> None:
    return None


@pytest.mark.asyncio
async def test_a_lost_or_failed_registration_is_retried_as_the_same_turn() -> None:
    server = _Server(httpx.ConnectError("down"), _Answer(503), _Answer(200))

    assert await register_turn(server, "conv_1", ["item_a"], sleep=_no_sleep) is True

    assert len({payload["turn_id"] for payload in server.payloads}) == 1
    assert [payload["input_ids"] for payload in server.payloads] == [["item_a"]] * 3


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "answers",
    [
        [_Answer(403)],
        [_Answer(404)],
        [httpx.ConnectError("down")] * 3,
        [_Answer(500)] * 3,
    ],
    ids=["refused", "older-server", "unreachable", "failing"],
)
async def test_a_registration_that_cannot_land_is_logged_and_never_raises(
    answers: list[Any],
) -> None:
    """D59: attribution is a display; nothing here may stop the turn."""
    assert await register_turn(_Server(*answers), "conv_1", ["item_a"], sleep=_no_sleep) is False
