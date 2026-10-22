"""Deliver a saved Claude transcript's undelivered items before a cold resume.

The forwarder posts transcript items in order and keeps its cursor in the
bridge directory, which a restored home does not have. A cold resume then
rebuilds Claude's transcript from server history, which would drop every item
Claude wrote but the server never received. These helpers find that
undelivered tail in the saved transcript and post it under the forwarder's own
``source_id`` keys, so a re-post of an item the server already holds is a
no-op. Whatever a bounded replay cannot deliver is left to the live forwarder,
which starts at the first undelivered record.
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

#: Wall-clock budget for the whole replay. It runs inside a terminal start,
#: which its caller bounds; the live forwarder delivers whatever is left.
_REPLAY_DEADLINE_S = 5.0
#: Refusals of the request itself rather than of one item: retrying the rest is futile.
_REQUEST_REFUSED_STATUS_CODES = frozenset({401, 403, 404})


@dataclass(frozen=True)
class UndeliveredItem:
    """
    A saved transcript item the server never received.

    :param item: The parsed transcript item.
    :param record_offset: Byte offset where its transcript record starts; a
        forwarder started there delivers it and everything after it.
    """

    item: ClaudeTranscriptItem
    record_offset: int


@dataclass(frozen=True)
class TranscriptReplayResult:
    """
    Outcome of reconciling a saved transcript with server history.

    :param delivered: Items the server accepted.
    :param rejected: Items the server refused permanently, e.g. as invalid;
        the live forwarder drops these too.
    :param undelivered: Items still not delivered when the replay stopped.
    :param resume_offset: Record offset of the first undelivered item, or
        ``None`` when every item was delivered or rejected.
    :param reconciled: ``False`` when the transcript could not be read, so
        whether it holds undelivered items is unknown.
    """

    delivered: int = 0
    rejected: int = 0
    undelivered: int = 0
    resume_offset: int | None = None
    reconciled: bool = True


def external_item_id(session_id: str, source_id: str) -> str:
    """
    Return the id the server gives an item posted with *source_id*.

    Mirrors the server's derivation for ``external_conversation_item`` events.

    :param session_id: Omnigent conversation id, e.g. ``"conv_abc123"``.
    :param source_id: Transcript item source id, e.g. ``"747e:0:message"``.
    :returns: The 32-character hex item id.
    :raises UnicodeEncodeError: If *source_id* holds a lone surrogate, which
        the server cannot key either.
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
) -> list[UndeliveredItem]:
    """
    Return the saved transcript's items the server never received.

    The forwarder delivers in order, so the undelivered items are the tail
    after the last item the server holds. An item is held when the server has
    its source-derived id, or when its record was rebuilt from a server item by
    an earlier cold resume. Without either, as in a fork whose copied items got
    fresh ids, an item whose response id the server holds counts as held.
    Nothing is returned when the transcript cannot be placed in the server's
    history: no held item while the server has items, or Claude items from
    another transcript after the last held one.

    :param transcript: Saved Claude transcript, e.g.
        ``"~/.claude/projects/-work-repo/<sid>.jsonl"``.
    :param session_id: Omnigent conversation id, e.g. ``"conv_abc123"``.
    :param server_items: Committed session items in chronological order.
    :param server_record_positions: Maps each record uuid a rebuild from
        *server_items* writes to the index of the last item it represents.
    :param agent_name: Agent name the forwarder stamps on items, e.g.
        ``"claude-native-ui"``.
    :returns: Undelivered items in transcript order; compaction summaries are
        left out, since the server records those as compaction boundaries.
    :raises OSError: If the transcript cannot be read.
    """
    local = _keyed_transcript_items(transcript, session_id=session_id, agent_name=agent_name)
    if not local:
        return []
    server_index: dict[str, int] = {}
    last_index_by_response: dict[str, int] = {}
    for index, server_item in enumerate(server_items):
        if isinstance(item_id := server_item.get("id"), str):
            server_index[item_id] = index
        if isinstance(response_id := server_item.get("response_id"), str):
            last_index_by_response[response_id] = index
    positions = [
        _first_known(server_index.get(item_id), server_record_positions.get(record_uuid or ""))
        for _, item_id, record_uuid, _ in local
    ]
    if all(position is None for position in positions):
        # A fork copies items under fresh ids but keeps their response ids.
        positions = [last_index_by_response.get(item.response_id) for item, *_ in local]
    held = [index for index, position in enumerate(positions) if position is not None]
    last_held = held[-1] if held else -1
    tail = [
        UndeliveredItem(item, record_offset)
        for item, _, _, record_offset in local[last_held + 1 :]
        if not item.is_compact_summary
    ]
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
        local_items=[item for item, *_ in local],
        local_ids={item_id for _, item_id, _, _ in local},
        session_id=session_id,
    ):
        _logger.warning(
            "Not replaying the saved Claude transcript for %s: the server's "
            "history continued from another transcript",
            session_id,
        )
        return []
    return tail


def _keyed_transcript_items(
    transcript: Path, *, session_id: str, agent_name: str
) -> list[tuple[ClaudeTranscriptItem, str, str | None, int]]:
    """
    Parse *transcript* with the forwarder's reader and key each item.

    :param transcript: Saved Claude transcript.
    :param session_id: Omnigent conversation id.
    :param agent_name: Agent name the forwarder stamps on items.
    :returns: ``(item, server item id, record uuid, record offset)`` per item,
        in order. Items the server could not key are left out.
    """
    read = read_transcript_items_from_offset(transcript, 0, start_line=0, agent_name=agent_name)
    keyed: list[tuple[ClaudeTranscriptItem, str, str | None, int]] = []
    unkeyable = 0
    record_offset = 0
    for record in read.record_items:
        for item in record.items:
            try:
                item_id = external_item_id(session_id, item.source_id)
            except UnicodeEncodeError:
                unkeyable += 1
                continue
            keyed.append((item, item_id, record.record_uuid, record_offset))
        record_offset = record.next_byte_offset
    if unkeyable:
        _logger.warning(
            "Skipping %d item(s) of the saved Claude transcript for %s with an "
            "undeliverable source id",
            unkeyable,
            session_id,
        )
    return keyed


def _first_known(*positions: int | None) -> int | None:
    """Return the first position that is not ``None``."""
    return next((position for position in positions if position is not None), None)


def _history_continued_elsewhere(
    later_server_items: Sequence[Mapping[str, object]],
    *,
    local_items: Sequence[ClaudeTranscriptItem],
    local_ids: set[str],
    session_id: str,
) -> bool:
    """
    Whether Claude items the saved transcript never wrote follow its last held item.

    That happens when the session was resumed and continued on another
    machine; appending this transcript's tail would then misorder the history.

    :param later_server_items: Server items after the last held one.
    :param local_items: Every item parsed from the saved transcript.
    :param local_ids: Server ids of *local_items*.
    :param session_id: Omnigent conversation id.
    :returns: ``True`` when the history diverged.
    """
    local_response_ids = {item.response_id for item in local_items}
    # A scheduled-wake marker is derived from a local record but not parsed here.
    known_ids = local_ids | {
        external_item_id(session_id, f"{item.source_id.rsplit(':', 2)[0]}:0:scheduled_wake")
        for item in local_items
    }
    for server_item in later_server_items:
        response_id = server_item.get("response_id")
        if (
            isinstance(response_id, str)
            and response_id.startswith(CLAUDE_RESPONSE_ID_PREFIX)
            and response_id not in local_response_ids
            and server_item.get("id") not in known_ids
        ):
            return True
    return False


async def replay_transcript_items(
    client: httpx.AsyncClient,
    *,
    session_id: str,
    items: Sequence[UndeliveredItem],
    deadline_s: float | None = None,
) -> TranscriptReplayResult:
    """
    Post *items* in order, as the live forwarder would, within a deadline.

    Transient failures, including an unconfirmed-durability 503, are retried
    with the forwarder's backoff; a retry is safe because the server applies
    each ``source_id`` once. A permanent rejection of one item skips it, as the
    forwarder does. The replay stops at the deadline, or when the server
    refuses the request itself; the result then names the first undelivered
    record so the live forwarder can continue from it.

    :param client: HTTP client pointed at the Omnigent server.
    :param session_id: Omnigent conversation id, e.g. ``"conv_abc123"``.
    :param items: Undelivered transcript items in order.
    :param deadline_s: Wall-clock budget in seconds, e.g. ``5.0``; ``None``
        uses the module default.
    :returns: Counts of delivered, rejected and still-undelivered items.
    """
    give_up_at = time.monotonic() + (_REPLAY_DEADLINE_S if deadline_s is None else deadline_s)
    retry_tracker = _PostRetryTracker()
    delivered = 0
    rejected = 0

    def stopped_at(index: int) -> TranscriptReplayResult:
        _logger.warning(
            "Stopped replaying the saved Claude transcript for %s; %d item(s) are "
            "left to the transcript forwarder",
            session_id,
            len(items) - index,
        )
        return TranscriptReplayResult(
            delivered, rejected, len(items) - index, items[index].record_offset
        )

    for index, pending in enumerate(items):
        item = pending.item
        retry_key = f"item:{item.source_id}"
        while True:
            remaining = give_up_at - time.monotonic()
            if remaining <= 0:
                return stopped_at(index)
            try:
                await _post_external_conversation_item(
                    client,
                    session_id=session_id,
                    item=item,
                    timeout=min(_POST_TIMEOUT_S, remaining),
                    replayed=True,
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
                    return stopped_at(index)
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
                if time.monotonic() + decision.delay_s >= give_up_at:
                    return stopped_at(index)
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
    return TranscriptReplayResult(delivered, rejected)
