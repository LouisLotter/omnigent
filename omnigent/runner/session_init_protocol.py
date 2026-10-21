"""Versioned server-to-runner session initialization payloads."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal, TypeAlias
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from omnigent.entities import Conversation

SessionInitProtocolVersion: TypeAlias = Literal[2]
SESSION_INIT_PROTOCOL_VERSION: SessionInitProtocolVersion = 2
SESSION_INIT_PAYLOAD_KEY = "session_init"
# Conversation label naming the runner process (incarnation) that last
# initialized the session, written by the server from the runner's answer.
RUNNER_INCARNATION_LABEL_KEY = "omnigent.runner_incarnation"
# The server's persisted failure code, and the code it records for work a dead
# runner left with an unknown outcome. Until the person acts, that work is never
# continued, by any runner.
LAST_TASK_ERROR_CODE_LABEL_KEY = "omnigent.last_task_error_code"
INTERRUPTED_OUTCOME_UNKNOWN_CODE = "interrupted_outcome_unknown"


class RunnerSessionInitSnapshot(BaseModel):  # type: ignore[explicit-any]  # Pydantic uses Any
    """Server-owned session state needed while starting a runner session."""

    model_config = ConfigDict(extra="ignore")

    created_at: int
    updated_at: int
    workspace: str | None = None
    labels: dict[str, str] = Field(default_factory=dict)
    reasoning_effort: str | None = None
    model_override: str | None = None
    harness_override: str | None = None
    cost_control_mode_override: str | None = None
    terminal_launch_args: list[str] | None = None
    external_session_id: str | None = None
    parent_session_id: str | None = None
    root_session_id: str | None = None
    inference_config: dict[str, object] | None = None


class RunnerSessionInitEnvelope(BaseModel):  # type: ignore[explicit-any]  # Pydantic uses Any
    """Metadata a current server can send instead of runner callback reads."""

    model_config = ConfigDict(extra="ignore")

    protocol_version: SessionInitProtocolVersion
    server_version: str
    session_id: str
    agent_id: str
    sub_agent_name: str | None = None
    snapshot: RunnerSessionInitSnapshot
    # When True the runner must skip crash-recovery turn detection on this
    # create_session call.  Set by the server whenever it calls session-init
    # immediately before forwarding a message — the forward carries the
    # message, so a recovery turn started from history would process it twice
    # (once from the recovery path, once from the buffered forward).
    suppress_recovery_turn: bool = False
    # Resume an interrupted child task in its existing native session.
    resume_interrupted_turn: bool = False
    recovery_id: str | None = None
    # The person asked to continue the session's interrupted work, so a runner
    # that is not the one that ran it may still continue it.
    continuation_requested: bool = False


def build_runner_session_init_payload(
    conversation: Conversation,
    *,
    server_version: str,
    suppress_recovery_turn: bool = False,
    resume_interrupted_turn: bool = False,
    recovery_id: str | None = None,
    continuation_requested: bool = False,
) -> dict[str, object]:
    """Build the versioned initialization fields appended to the legacy body."""
    from omnigent.inference_config import snapshot_runtime_config

    if conversation.agent_id is None:
        raise ValueError("runner session initialization requires an agent_id")
    envelope = RunnerSessionInitEnvelope(
        protocol_version=SESSION_INIT_PROTOCOL_VERSION,
        server_version=server_version,
        session_id=conversation.id,
        agent_id=conversation.agent_id,
        sub_agent_name=conversation.sub_agent_name,
        suppress_recovery_turn=suppress_recovery_turn,
        resume_interrupted_turn=resume_interrupted_turn,
        recovery_id=(recovery_id or uuid4().hex) if resume_interrupted_turn else None,
        continuation_requested=continuation_requested,
        snapshot=RunnerSessionInitSnapshot(
            created_at=conversation.created_at,
            updated_at=conversation.updated_at,
            workspace=conversation.workspace,
            labels=conversation.labels,
            reasoning_effort=conversation.reasoning_effort,
            model_override=conversation.model_override,
            harness_override=conversation.harness_override,
            cost_control_mode_override=conversation.cost_control_mode_override,
            terminal_launch_args=conversation.terminal_launch_args,
            external_session_id=conversation.external_session_id,
            parent_session_id=conversation.parent_conversation_id,
            root_session_id=conversation.root_conversation_id,
            inference_config=snapshot_runtime_config(conversation.inference_snapshot),
        ),
    )
    return {
        "session_id": conversation.id,
        "agent_id": conversation.agent_id,
        "sub_agent_name": conversation.sub_agent_name,
        SESSION_INIT_PAYLOAD_KEY: envelope.model_dump(mode="json"),
    }


def earlier_runner_incarnation(
    envelope: RunnerSessionInitEnvelope | None, incarnation_id: str
) -> str | None:
    """
    Return the other runner incarnation that last held this session, if any.

    A transport reconnect of the same runner process keeps its incarnation; a
    replacement process (after a crash, a host or computer restart, or a
    rebind) has a new one. Every path that re-attaches a session to a runner
    sends this envelope, so the runner can tell the two apart here.

    :param envelope: The initialization envelope, or ``None`` for a legacy body.
    :param incarnation_id: This runner process's incarnation id.
    :returns: The earlier incarnation's id, or ``None`` for a session this
        process already held, a fresh session, or a legacy body.
    """
    if envelope is None:
        return None
    earlier = envelope.snapshot.labels.get(RUNNER_INCARNATION_LABEL_KEY)
    return earlier if earlier and earlier != incarnation_id else None


def automatic_continuation_allowed(
    envelope: RunnerSessionInitEnvelope | None,
    incarnation_id: str,
    *,
    ran_before: bool = True,
) -> bool:
    """
    Return whether this runner may continue the session's earlier work on its own.

    Work interrupted in another runner process may already have had effects that
    nothing recorded, so a new incarnation never re-runs a turn
    (``history_resume``) or sends the recovery prompt unless the person asked.
    Missing provenance fails closed: a session that already ran but carries no
    incarnation (its record was lost, or predates it) is treated as another
    runner's work.

    :param envelope: The initialization envelope, or ``None`` for a legacy body.
    :param incarnation_id: This runner process's incarnation id.
    :param ran_before: Whether an agent already worked on the session, e.g. its
        history holds an assistant reply or a tool call. ``False`` only for work
        nothing has started yet, such as a new session's seeded first message.
    :returns: ``False`` when another or an unknown incarnation did the work, or
        the session still carries an unknown outcome, and the person did not
        ask to continue.
    """
    if envelope is None:
        return True
    if envelope.continuation_requested:
        return True
    labels = envelope.snapshot.labels
    if labels.get(LAST_TASK_ERROR_CODE_LABEL_KEY) == INTERRUPTED_OUTCOME_UNKNOWN_CODE:
        return False
    holder = labels.get(RUNNER_INCARNATION_LABEL_KEY)
    if not holder:
        return not ran_before
    return holder == incarnation_id


_AGENT_WORK_ITEM_TYPES = frozenset(
    {"function_call", "function_call_output", "reasoning", "compaction"}
)


def history_shows_agent_work(history: Sequence[Mapping[str, object]]) -> bool:
    """
    Return whether a session's history holds anything an agent produced.

    :param history: History items in the runner's input shape, e.g.
        ``[{"type": "message", "role": "user", "content": [...]}]``.
    :returns: ``True`` once an assistant reply or a tool call is present.
    """
    return any(
        item.get("type") in _AGENT_WORK_ITEM_TYPES
        or (item.get("type") == "message" and item.get("role") == "assistant")
        for item in history
    )


class RunnerInferenceConfigMismatch(ValueError):
    """A session cannot share a runner carrying another saved provider configuration."""


def validate_runner_inference_config(expected: dict[str, object] | None) -> None:
    """Require the session to use the provider map the runner loaded at launch."""
    from omnigent.inference_config import load_runtime_inference_config

    actual = load_runtime_inference_config({})
    expected = expected or {}
    if any(actual.get(key, {}) != expected.get(key, {}) for key in ("providers", "inference")):
        raise RunnerInferenceConfigMismatch(
            "This runner has a different saved provider configuration; launch a new runner."
        )


def parse_runner_session_init_envelope(
    body: dict[str, object],
) -> RunnerSessionInitEnvelope | None:
    """Return a supported envelope, or ``None`` for the removable legacy path."""
    raw = body.get(SESSION_INIT_PAYLOAD_KEY)
    if not isinstance(raw, dict):
        return None
    if raw.get("protocol_version") != SESSION_INIT_PROTOCOL_VERSION:
        return None
    try:
        return RunnerSessionInitEnvelope.model_validate(raw)
    except ValidationError as exc:
        raise ValueError("invalid runner session initialization envelope") from exc
