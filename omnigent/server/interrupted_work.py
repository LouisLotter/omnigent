"""Keep the outcome of work a dead runner was doing visible until the person acts.

A runner process (an incarnation) can die mid-turn: killed, its host restarted,
or its computer restarted. Its last tool call may have had effects that never
reached the transcript. A replacement runner never continues that work on its
own (see :func:`omnigent.runner.session_init_protocol.automatic_continuation_allowed`).
The server records which incarnation holds each session, names the tool call
left without a result, and keeps an "interrupted, outcome unknown" failure on
the session that only the person's next action clears.
"""

from __future__ import annotations

import asyncio
import logging

import httpx

from omnigent.entities import Conversation
from omnigent.runner.session_init_protocol import (
    INTERRUPTED_OUTCOME_UNKNOWN_CODE,
    LAST_TASK_ERROR_CODE_LABEL_KEY,
    RUNNER_INCARNATION_LABEL_KEY,
)
from omnigent.server.schemas import ErrorDetail
from omnigent.stores.conversation_store import ConversationStore

_logger = logging.getLogger(__name__)

# How far back the current turn's unanswered tool call is looked for.
_TOOL_CALL_LOOKBACK_ITEMS = 200


def _init_answer(response: httpx.Response) -> dict[str, object]:
    """Return a runner's session-initialization answer, or ``{}`` when unreadable."""
    try:
        payload = response.json()
    except (ValueError, AttributeError):
        return {}
    return payload if isinstance(payload, dict) else {}


async def record_runner_incarnation(
    conversation: Conversation,
    response: httpx.Response,
    conversation_store: ConversationStore,
) -> None:
    """
    Remember which runner incarnation now holds *conversation*.

    The next initialization carries this label, so a replacement process can
    tell that the session's earlier work was not its own.

    :param conversation: The session as sent in the initialization.
    :param response: The runner's answer.
    :param conversation_store: Store holding the label.
    """
    if not 200 <= response.status_code < 300:
        return
    incarnation = _init_answer(response).get("runner_incarnation_id")
    if not isinstance(incarnation, str) or not incarnation:
        return
    if conversation.labels.get(RUNNER_INCARNATION_LABEL_KEY) == incarnation:
        return
    try:
        await asyncio.to_thread(
            conversation_store.set_labels,
            conversation.id,
            {RUNNER_INCARNATION_LABEL_KEY: incarnation},
        )
    except Exception:  # noqa: BLE001 — a stale label only makes the next check stricter
        _logger.warning(
            "Could not record the runner incarnation of session %s",
            conversation.id,
            exc_info=True,
            extra={"session_id": conversation.id},
        )


async def unanswered_tool_call(
    conversation_store: ConversationStore, session_id: str
) -> tuple[str, str] | None:
    """
    Find the current turn's latest tool call that has no result.

    :param conversation_store: Store holding the transcript.
    :param session_id: Session id, e.g. ``"conv_abc123"``.
    :returns: ``(item_id, tool_name)``, or ``None`` when every call of the
        current turn has its result.
    """
    return await asyncio.to_thread(_unanswered_tool_call, conversation_store, session_id)


def _unanswered_tool_call(
    conversation_store: ConversationStore, session_id: str
) -> tuple[str, str] | None:
    """Blocking body of :func:`unanswered_tool_call`."""
    try:
        page = conversation_store.list_items(
            session_id, limit=_TOOL_CALL_LOOKBACK_ITEMS, order="desc"
        )
    except Exception:  # noqa: BLE001 — naming the tool is best effort
        _logger.warning(
            "Could not read the transcript of session %s",
            session_id,
            exc_info=True,
            extra={"session_id": session_id},
        )
        return None
    answered: set[str] = set()
    for item in page.data:
        data = item.data
        if item.type == "function_call_output":
            answered.add(getattr(data, "call_id", ""))
        elif item.type == "function_call":
            if getattr(data, "call_id", None) not in answered:
                return item.id, str(getattr(data, "name", "") or "a tool")
        elif item.type == "message" and getattr(data, "role", None) == "user":
            return None
    return None


def interrupted_outcome_error(tool: tuple[str, str] | None) -> ErrorDetail:
    """
    Build the "interrupted, outcome unknown" failure for a session.

    :param tool: ``(item_id, tool_name)`` of the call left without a result.
    :returns: The failure to publish and persist.
    """
    if tool is None:
        message = (
            "The runner stopped during this turn, so what it last did is unknown. "
            "Nothing was run again."
        )
    else:
        message = (
            f"The runner stopped while {tool[1]} was running, so its outcome is unknown. "
            "It was not run again."
        )
    return ErrorDetail(
        code=INTERRUPTED_OUTCOME_UNKNOWN_CODE,
        title="Interrupted, outcome unknown",
        message=message,
        remediation="Check whether it took effect before you continue or retry.",
    )


async def with_unanswered_tool(
    error: ErrorDetail, conversation_store: ConversationStore, session_id: str
) -> ErrorDetail:
    """
    Name the tool call a lost runner left without a result in *error*'s message.

    :param error: The disconnect failure, e.g. ``runner_disconnected``.
    :param conversation_store: Store holding the transcript.
    :param session_id: Session id, e.g. ``"conv_abc123"``.
    :returns: *error*, with the unanswered tool named when there is one.
    """
    return _naming_tool(error, await unanswered_tool_call(conversation_store, session_id))


def runner_lost_error(conversation_store: ConversationStore, session_id: str) -> ErrorDetail:
    """
    Build, blocking, the disconnect failure for a turn whose runner is gone.

    :param conversation_store: Store holding the transcript.
    :param session_id: Session id, e.g. ``"conv_abc123"``.
    :returns: A ``runner_disconnected`` failure naming the unanswered tool.
    """
    return _naming_tool(
        ErrorDetail(code="runner_disconnected", message="Runner disconnected unexpectedly."),
        _unanswered_tool_call(conversation_store, session_id),
    )


def _naming_tool(error: ErrorDetail, tool: tuple[str, str] | None) -> ErrorDetail:
    if tool is None:
        return error
    return error.model_copy(
        update={
            "message": (
                f"{error.message} {tool[1]} was running and its outcome is unknown."
            ).strip()
        }
    )


async def keep_interrupted_outcome(
    session_id: str,
    response: httpx.Response,
    conversation_store: ConversationStore,
    *,
    runner_id: str | None,
) -> bool:
    """
    Mark interrupted work a replacement runner declined to continue.

    Call after initializing a session on a runner, before clearing its failure.
    When the runner reports that another incarnation held the session and the
    session was interrupted mid-turn, the failure becomes the durable
    "interrupted, outcome unknown" state, which only the person's next action
    clears.

    :param session_id: Session id, e.g. ``"conv_abc123"``.
    :param response: The runner's answer to the initialization.
    :param conversation_store: Store for the transcript and failure labels.
    :param runner_id: The replacement runner, recorded as the failure's owner.
    :returns: ``True`` when the state is kept, including one recorded earlier,
        so the caller must not clear it.
    """
    from omnigent.server.child_session_recovery import _interrupted
    from omnigent.server.routes._sessions.helpers import (
        _persist_session_status_error_labels,
        _publish_status,
    )

    if not 200 <= response.status_code < 300:
        return False
    conversation = await asyncio.to_thread(conversation_store.get_conversation, session_id)
    if conversation is None:
        return False
    if conversation.labels.get(LAST_TASK_ERROR_CODE_LABEL_KEY) == (
        INTERRUPTED_OUTCOME_UNKNOWN_CODE
    ):
        # Already recorded: only the person's next message or retry clears it.
        return True
    if _init_answer(response).get("continuation_blocked") is not True:
        return False
    if not _interrupted(conversation):
        return False
    tool = await unanswered_tool_call(conversation_store, session_id)
    error = interrupted_outcome_error(tool)
    _logger.warning(
        "Session %s was interrupted in another runner process; its outcome is unknown",
        session_id,
        extra={"session_id": session_id},
    )
    _publish_status(session_id, "failed", error, failure_origin="runner_incarnation_changed")
    await _persist_session_status_error_labels(
        session_id,
        error,
        conversation_store,
        owner_runner_id=runner_id,
        item_id=tool[0] if tool is not None else None,
    )
    return True
