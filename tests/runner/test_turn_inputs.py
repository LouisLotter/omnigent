"""Each turn is registered with the server before the harness acts (``turn_inputs``)."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from omnigent.runner.turn_inputs import TurnRegistrationError, register_turn, turn_input_ids


def test_a_turns_inputs_are_named_by_server_issued_ids() -> None:
    bodies = [
        {"persisted_item_id": "item_a", "stable_id": "0" * 32},
        {"pending_id": "pending_b", "stable_id": "1" * 32},
        {"content": [], "created_by": "mallory"},
        {"persisted_item_id": "item_a"},
    ]
    assert turn_input_ids(bodies) == ["item_a", "pending_b"]


class _Answer:
    def __init__(self, status: int, body: Any = None) -> None:
        self.status_code = status
        self._body = body

    def json(self) -> Any:
        if self._body is None:
            raise ValueError("no body")
        return self._body


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
    server = _Server(httpx.ConnectError("down"), _Answer(503), _Answer(200, {"turn_id": "t"}))

    await register_turn(server, "conv_1", ["item_a"], sleep=_no_sleep)

    assert len({payload["turn_id"] for payload in server.payloads}) == 1
    assert [payload["input_ids"] for payload in server.payloads] == [["item_a"]] * 3


@pytest.mark.asyncio
async def test_a_refused_or_unreachable_registration_stops_the_turn() -> None:
    with pytest.raises(TurnRegistrationError, match="refused"):
        await register_turn(
            _Server(_Answer(403, {"error": {"code": "forbidden"}})), "conv_1", ["item_a"]
        )
    with pytest.raises(TurnRegistrationError, match="could not register"):
        await register_turn(_Server(*[_Answer(500)] * 4), "conv_1", ["item_a"], sleep=_no_sleep)
    with pytest.raises(TurnRegistrationError):
        await register_turn(
            _Server(_Answer(404, {"error": {"code": "not_found"}})), "conv_1", ["item_a"]
        )


@pytest.mark.asyncio
async def test_a_server_without_turn_registration_lets_the_turn_proceed() -> None:
    """An older server has no route; the turn runs as it did before."""
    await register_turn(_Server(_Answer(404, {"detail": "Not Found"})), "conv_1", ["item_a"])
    await register_turn(_Server(_Answer(404)), "conv_1", ["item_a"])
