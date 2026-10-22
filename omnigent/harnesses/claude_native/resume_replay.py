"""Deliver a saved Claude transcript's undelivered items before a cold resume.

The forwarder posts transcript items in order and keeps its cursor in the
bridge directory, which a restored home does not have. A cold resume then
rebuilds Claude's transcript from server history, which would drop every item
Claude wrote but the server never received. These helpers find that
undelivered tail in the saved transcript and post it under the forwarder's own
``source_id`` keys, so a re-post of an item the server already holds is a
no-op.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import httpx

from omnigent.harnesses.claude_native.bridge import (
    CLAUDE_RESPONSE_ID_PREFIX,
    ClaudeTranscriptItem,
    read_transcript_items_from_offset,
)
from omnigent.harnesses.claude_native.forwarder import (
    _POST_TIMEOUT_S,
    _http_status_for_log,
    _maybe_sync_effort_from_slash_command,
    _post_external_conversation_item,
    _PostRetryTracker,
)

_logger = logging.getLogger(__name__)
_sleep = asyncio.sleep

#: How long one item may keep failing before the replay gives up.
_REPLAY_STALL_TIMEOUT_S = 30.0
#: Refusals of the request itself rather than of one item: retrying the rest is futile.
_REQUEST_REFUSED_STATUS_CODES = frozenset({401, 403, 404})


@dataclass(frozen=True)
class TranscriptReplayResult:
    """
    Outcome of replaying a saved transcript's undelivered items.

    :param delivered: Items the server accepted.
    :param rejected: Items the server refused permanently, e.g. as invalid;
        the live forwarder drops these too.
    :param undelivered: Items still not delivered when the replay stopped.
    """

    delivered: int = 0
    rejected: int = 0
    undelivered: int = 0


def external_item_id(session_id: str, source_id: str) -> str:
    """
    Return the id the server gives an item posted with *source_id*.

    Mirrors the server's derivation for ``external_conversation_item`` events.

    :param session_id: Omnigent conversation id, e.g. ``"conv_abc123"``.
    :param source_id: Transcript item source id, e.g. ``"747e:0:message"``.
    :returns: The 32-character hex item id.
    """
    return uuid.uuid5(
        uuid.NAMESPACE_URL, f"omnigent-external-item:{session_id}:{source_id.strip()}"
    ).hex


def undelivered_transcript_items(
    transcript: Path,
    *,
    session_id: str,
    server_items: Sequence[Mapping[str, object]],
    server_record_positions: Mapping[str, int],
    agent_name: str,
) -> list[ClaudeTranscriptItem]:
    """
    Return the saved transcript's items the server never received.

    The forwarder delivers in order, so the undelivered items are the tail
    after the last item the server holds. An item is held when the server has
    its source-derived id, or when its record was rebuilt from a server item by
    an earlier cold resume. Nothing is returned when the transcript cannot be
    placed in the server's history: no held item while the server has items
    (e.g. a fork's copied prefix), or Claude items from another transcript
    after the last held one.

    :param transcript: Saved Claude transcript, e.g.
        ``"~/.claude/projects/-work-repo/<sid>.jsonl"``.
    :param session_id: Omnigent conversation id, e.g. ``"conv_abc123"``.
    :param server_items: Committed session items in chronological order.
    :param server_record_positions: Maps each record uuid a rebuild from
        *server_items* writes to the index of the item it came from.
    :param agent_name: Agent name the forwarder stamps on items, e.g.
        ``"claude-native-ui"``.
    :returns: Undelivered items in transcript order; compaction summaries are
        left out, since the server records those as compaction boundaries.
    """
    read = read_transcript_items_from_offset(transcript, 0, start_line=0, agent_name=agent_name)
    local = [(item, record.record_uuid) for record in read.record_items for item in record.items]
    if not local:
        return []
    server_index = {
        item_id: index
        for index, item in enumerate(server_items)
        if isinstance(item_id := item.get("id"), str)
    }
    positions: list[int | None] = []
    for item, record_uuid in local:
        position = server_index.get(external_item_id(session_id, item.source_id))
        if position is None and record_uuid is not None:
            position = server_record_positions.get(record_uuid)
        positions.append(position)
    held = [index for index, position in enumerate(positions) if position is not None]
    last_held = held[-1] if held else -1
    tail = [item for item, _ in local[last_held + 1 :] if not item.is_compact_summary]
    if not tail:
        return []
    if not held:
        if server_items:
            _logger.warning(
                "Not replaying the saved Claude transcript for %s: none of its items "
                "is in the server's history",
                session_id,
            )
            return []
        return tail
    anchor_position = positions[last_held]
    assert anchor_position is not None
    if _history_continued_elsewhere(
        server_items[anchor_position + 1 :],
        local_items=[item for item, _ in local],
        session_id=session_id,
    ):
        _logger.warning(
            "Not replaying the saved Claude transcript for %s: the server's "
            "history continued from another transcript",
            session_id,
        )
        return []
    return tail


def _history_continued_elsewhere(
    later_server_items: Sequence[Mapping[str, object]],
    *,
    local_items: Sequence[ClaudeTranscriptItem],
    session_id: str,
) -> bool:
    """
    Whether Claude items the saved transcript never wrote follow its last held item.

    That happens when the session was resumed and continued on another
    machine; appending this transcript's tail would then misorder the history.

    :param later_server_items: Server items after the last held one.
    :param local_items: Every item parsed from the saved transcript.
    :param session_id: Omnigent conversation id.
    :returns: ``True`` when the history diverged.
    """
    source_keys = {item.source_id.rsplit(":", 2)[0] for item in local_items}
    # A scheduled-wake marker is derived from a local record but not parsed here.
    known_ids = {external_item_id(session_id, item.source_id) for item in local_items} | {
        external_item_id(session_id, f"{key}:0:scheduled_wake") for key in source_keys
    }
    for item in later_server_items:
        response_id = item.get("response_id")
        if (
            isinstance(response_id, str)
            and response_id.startswith(CLAUDE_RESPONSE_ID_PREFIX)
            and item.get("id") not in known_ids
        ):
            return True
    return False


async def replay_transcript_items(
    client: httpx.AsyncClient,
    *,
    session_id: str,
    items: Sequence[ClaudeTranscriptItem],
    stall_timeout_s: float | None = None,
) -> TranscriptReplayResult:
    """
    Post *items* in order, as the live forwarder would.

    Transient failures, including an unconfirmed-durability 503, are retried
    with the forwarder's backoff; a retry is safe because the server applies
    each ``source_id`` once. A permanent rejection of one item skips it, as the
    forwarder does. The replay stops when an item keeps failing for
    *stall_timeout_s*, or when the server refuses the request itself.

    :param client: HTTP client pointed at the Omnigent server.
    :param session_id: Omnigent conversation id, e.g. ``"conv_abc123"``.
    :param items: Undelivered transcript items in order.
    :param stall_timeout_s: Seconds one item may keep failing, e.g. ``30.0``;
        ``None`` uses the module default.
    :returns: Counts of delivered, rejected and still-undelivered items.
    """
    if stall_timeout_s is None:
        stall_timeout_s = _REPLAY_STALL_TIMEOUT_S
    retry_tracker = _PostRetryTracker()
    delivered = 0
    rejected = 0
    for index, item in enumerate(items):
        retry_key = f"item:{item.source_id}"
        give_up_at = time.monotonic() + stall_timeout_s
        while True:
            try:
                await _post_external_conversation_item(
                    client, session_id=session_id, item=item, timeout=_POST_TIMEOUT_S
                )
            except httpx.HTTPError as exc:
                status = _http_status_for_log(exc)
                decision = retry_tracker.record_failure(retry_key, exc, session_id=session_id)
                if decision.permanent and status in _REQUEST_REFUSED_STATUS_CODES:
                    _logger.warning(
                        "Server refused the transcript replay for %s (HTTP %s)",
                        session_id,
                        status,
                    )
                    return TranscriptReplayResult(delivered, rejected, len(items) - index)
                if decision.exhausted:
                    _logger.error(
                        "Server rejected a replayed Claude transcript item; "
                        "session=%s source_id=%s http_status=%s",
                        session_id,
                        item.source_id,
                        status,
                    )
                    rejected += 1
                    break
                if time.monotonic() + decision.delay_s > give_up_at:
                    _logger.warning(
                        "Stopped replaying the saved Claude transcript for %s; "
                        "%d item(s) still undelivered",
                        session_id,
                        len(items) - index,
                        exc_info=True,
                    )
                    return TranscriptReplayResult(delivered, rejected, len(items) - index)
                await _sleep(decision.delay_s)
                continue
            retry_tracker.clear(retry_key)
            delivered += 1
            await _maybe_sync_effort_from_slash_command(client, session_id=session_id, item=item)
            break
    if delivered or rejected:
        _logger.info(
            "Replayed the saved Claude transcript for %s: delivered=%d rejected=%d",
            session_id,
            delivered,
            rejected,
        )
    return TranscriptReplayResult(delivered, rejected, 0)
