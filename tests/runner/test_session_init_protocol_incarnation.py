"""The runner tells a reconnect of itself from a replacement of a dead runner."""

from __future__ import annotations

import pytest

from omnigent.entities import Conversation
from omnigent.runner.session_init_protocol import (
    RUNNER_INCARNATION_LABEL_KEY,
    RunnerSessionInitEnvelope,
    automatic_continuation_allowed,
    build_runner_session_init_payload,
    earlier_runner_incarnation,
    parse_runner_session_init_envelope,
)


def _envelope(
    labels: dict[str, str], *, continuation_requested: bool = False
) -> RunnerSessionInitEnvelope | None:
    conv = Conversation(
        id="conv_x",
        agent_id="ag_x",
        runner_id="runner_x",
        created_at=0,
        updated_at=0,
        root_conversation_id="conv_x",
        labels=labels,
    )
    return parse_runner_session_init_envelope(
        build_runner_session_init_payload(
            conv, server_version="test", continuation_requested=continuation_requested
        )
    )


@pytest.mark.parametrize(
    ("labels", "requested", "ran_before", "earlier", "allowed"),
    [
        ({}, False, False, None, True),
        ({}, False, True, None, False),
        ({RUNNER_INCARNATION_LABEL_KEY: "me"}, False, True, None, True),
        ({RUNNER_INCARNATION_LABEL_KEY: "dead"}, False, True, "dead", False),
        ({RUNNER_INCARNATION_LABEL_KEY: "dead"}, True, True, "dead", True),
        ({}, True, True, None, True),
    ],
    ids=[
        "fresh-session",
        "provenance-lost",
        "reconnect",
        "replacement",
        "person-asked",
        "person-asked-without-provenance",
    ],
)
def test_incarnation_check(
    labels: dict[str, str], requested: bool, ran_before: bool, earlier: str | None, allowed: bool
) -> None:
    envelope = _envelope(labels, continuation_requested=requested)
    assert earlier_runner_incarnation(envelope, "me") == earlier
    assert automatic_continuation_allowed(envelope, "me", ran_before=ran_before) is allowed


def test_history_shows_agent_work() -> None:
    from omnigent.runner.session_init_protocol import history_shows_agent_work

    user = {"type": "message", "role": "user", "content": []}
    assert not history_shows_agent_work([user])
    assert history_shows_agent_work([user, {"type": "message", "role": "assistant"}])
    assert history_shows_agent_work([user, {"type": "function_call", "name": "Bash"}])


def test_legacy_body_keeps_its_behaviour() -> None:
    assert earlier_runner_incarnation(None, "me") is None
    assert automatic_continuation_allowed(None, "me") is True


def test_an_unknown_outcome_blocks_continuation_until_the_person_asks() -> None:
    """Even the runner now holding the session never continues work left unknown."""
    from omnigent.runner.session_init_protocol import (
        INTERRUPTED_OUTCOME_UNKNOWN_CODE,
        LAST_TASK_ERROR_CODE_LABEL_KEY,
    )

    labels = {
        RUNNER_INCARNATION_LABEL_KEY: "me",
        LAST_TASK_ERROR_CODE_LABEL_KEY: INTERRUPTED_OUTCOME_UNKNOWN_CODE,
    }
    assert automatic_continuation_allowed(_envelope(labels), "me") is False
    assert (
        automatic_continuation_allowed(_envelope(labels, continuation_requested=True), "me")
        is True
    )


def test_error_code_label_matches_the_servers() -> None:
    """The runner reads the label the server writes."""
    from omnigent.runner.session_init_protocol import LAST_TASK_ERROR_CODE_LABEL_KEY
    from omnigent.server.routes._sessions.common import _LAST_TASK_ERROR_CODE_LABEL_KEY

    assert LAST_TASK_ERROR_CODE_LABEL_KEY == _LAST_TASK_ERROR_CODE_LABEL_KEY
