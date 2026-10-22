"""Claude-native cold resume: replay of a saved transcript's undelivered items.

The real-server scenarios (exactly once, durability barrier, lost response,
queued web input, forwarder continuation) live in
``tests/server/integration/test_claude_native_resume_replay.py``. These cover
the placement rules and the failure paths against a fake server that keys
items by ``source_id`` as the real one does.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from omnigent.harnesses.claude_native import main as claude_native
from omnigent.harnesses.claude_native import resume_replay
from omnigent.harnesses.claude_native.bridge import (
    ClaudeTranscriptItem,
    read_transcript_items_from_offset,
)
from omnigent.harnesses.claude_native.resume_replay import external_item_id

_SESSION_ID = "conv_replay"
_CLAUDE_SESSION_ID = "4b0f2a52-2b5e-4d8e-9a39-6f0f1c1d2e3f"
_WORKSPACE = Path("/work/replay-repo")


def _user(uuid: str, text: str, **extra: Any) -> dict[str, Any]:
    return {"type": "user", "uuid": uuid, "message": {"role": "user", "content": text}, **extra}


def _assistant(uuid: str, text: str) -> dict[str, Any]:
    return {
        "type": "assistant",
        "uuid": uuid,
        "message": {"role": "assistant", "content": [{"type": "text", "text": text}]},
    }


def _line(entry: dict[str, Any]) -> bytes:
    return (json.dumps(entry) + "\n").encode("utf-8")


def _parsed(transcript: Path) -> dict[str, ClaudeTranscriptItem]:
    """Items the live forwarder would post from *transcript*, by source id."""
    result = read_transcript_items_from_offset(
        transcript, 0, start_line=0, agent_name="claude-native-ui"
    )
    return {item.source_id: item for item in result.items}


def _stored(item: ClaudeTranscriptItem, *, item_id: str | None = None) -> dict[str, Any]:
    """A committed item as the server lists it; a fork copy has a fresh id."""
    return {
        "id": item_id or external_item_id(_SESSION_ID, item.source_id),
        "type": item.item_type,
        "response_id": item.response_id,
        **item.data,
    }


class _FakeServer:
    """Session items keyed like the server's source-keyed append."""

    def __init__(
        self,
        items: list[dict[str, Any]] | None = None,
        *,
        fail_post: Callable[[str], httpx.Response | None] | None = None,
    ) -> None:
        self.items = list(items or [])
        self.posts: list[str] = []
        self.replayed_flags: list[object] = []
        self._fail_post = fail_post

    def handle(self, request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path.endswith("/items"):
            return httpx.Response(200, json={"data": self.items, "has_more": False})
        if request.method == "POST" and request.url.path.endswith("/events"):
            data = json.loads(request.content)["data"]
            self.posts.append(data["source_id"])
            self.replayed_flags.append(data.get("replayed"))
            failure = self._fail_post(data["source_id"]) if self._fail_post else None
            if failure is not None:
                return failure
            item_id = external_item_id(_SESSION_ID, data["source_id"])
            if all(item["id"] != item_id for item in self.items):
                self.items.append(
                    {
                        "id": item_id,
                        "type": data["item_type"],
                        "response_id": data["response_id"],
                        **data["item_data"],
                    }
                )
            return httpx.Response(202, json={"queued": False, "item_id": item_id})
        return httpx.Response(404)

    def texts(self) -> list[str]:
        return [
            block["text"]
            for item in self.items
            for block in item.get("content", [])
            if isinstance(block, dict) and "text" in block
        ]


@pytest.fixture
def saved_transcript(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Path of the transcript a restored home holds for the session."""
    projects = tmp_path / "projects"
    monkeypatch.setattr(claude_native, "_CLAUDE_PROJECTS_DIR", projects)
    target = (
        projects
        / claude_native._sanitize_claude_project_name(str(_WORKSPACE))
        / f"{_CLAUDE_SESSION_ID}.jsonl"
    )
    target.parent.mkdir(parents=True)
    return target


async def _cold_resume(server: _FakeServer) -> claude_native.ClaudeResumeTranscript | None:
    transport = httpx.MockTransport(server.handle)
    async with httpx.AsyncClient(transport=transport, base_url="https://example.com") as client:
        return await claude_native._prepare_claude_resume_transcript(
            client,
            session_id=_SESSION_ID,
            external_session_id=_CLAUDE_SESSION_ID,
            workspace=_WORKSPACE,
        )


async def _no_sleep(delay: float) -> None:
    del delay


@pytest.mark.asyncio
async def test_corrupt_and_partial_lines_are_skipped(saved_transcript: Path) -> None:
    """Undecodable lines and a torn final record are never posted; valid records are."""
    saved_transcript.write_bytes(
        _line(_user("u-1", "first question"))
        + b"not json at all\n"
        + b"\xff\xfe binary \x80\n"
        + _line(_assistant("a-1", "first answer"))
        + _line(_user("u-2", "second question"))[:20]
    )
    server = _FakeServer([_stored(_parsed(saved_transcript)["u-1:0:message"])])

    resumed = await _cold_resume(server)

    assert server.posts == ["a-1:0:message"]
    assert server.replayed_flags == [True]
    assert server.texts() == ["first question", "first answer"]
    assert resumed is not None
    assert resumed.forward_from is None
    assert "second question" not in resumed.path.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_unkeyable_record_is_isolated(saved_transcript: Path) -> None:
    """A record whose source id the server cannot key does not stop the others."""
    saved_transcript.write_bytes(
        _line(_user("u-1", "first question"))
        + _line(_assistant("a-1", "first answer"))
        + b'{"type":"assistant","uuid":"\\ud800","message":{"role":"assistant",'
        b'"content":[{"type":"text","text":"unkeyable"}]}}\n'
    )
    server = _FakeServer([_stored(_parsed(saved_transcript)["u-1:0:message"])])

    resumed = await _cold_resume(server)

    assert server.posts == ["a-1:0:message"]
    assert resumed is not None
    assert "first answer" in resumed.path.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_transcript_without_items_is_rebuilt_from_server_history(
    saved_transcript: Path,
) -> None:
    """A transcript with no parsable record holds nothing to lose and is rewritten."""
    saved_transcript.write_bytes(b"\x00\x01garbage\nnot json\n")
    server = _FakeServer(
        [
            {
                "id": "msg_1",
                "type": "message",
                "role": "user",
                "response_id": "resp_1",
                "content": [{"type": "input_text", "text": "first question"}],
            }
        ]
    )

    resumed = await _cold_resume(server)

    assert server.posts == []
    assert resumed is not None
    assert "first question" in resumed.path.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_transcript_that_cannot_be_read_is_kept(
    saved_transcript: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A read error leaves the transcript unknown, so it is not overwritten."""
    saved = _line(_user("u-1", "first question")) + _line(_assistant("a-1", "unread answer"))
    saved_transcript.write_bytes(saved)
    server = _FakeServer([_stored(_parsed(saved_transcript)["u-1:0:message"])])

    def unreadable(*args: object, **kwargs: object) -> None:
        raise PermissionError("transcript is not readable")

    monkeypatch.setattr(resume_replay, "read_transcript_items_from_offset", unreadable)

    resumed = await _cold_resume(server)

    assert server.posts == []
    assert resumed is not None
    assert resumed.path == saved_transcript
    assert resumed.forward_from is None
    assert saved_transcript.read_bytes() == saved


@pytest.mark.asyncio
async def test_fork_continuation_is_delivered_after_its_copied_prefix(
    saved_transcript: Path,
) -> None:
    """A fork's copies have fresh ids but their response ids place the new turn."""
    saved_transcript.write_bytes(
        _line(_user("copied-1", "copied question"))
        + _line(_assistant("copied-2", "copied answer"))
        + _line(_user("new-1", "first fork question"))
    )
    parsed = _parsed(saved_transcript)
    server = _FakeServer(
        [
            _stored(parsed["copied-1:0:message"], item_id="fork_copy_1"),
            _stored(parsed["copied-2:0:message"], item_id="fork_copy_2"),
        ]
    )

    await _cold_resume(server)

    assert server.posts == ["new-1:0:message"]


@pytest.mark.asyncio
async def test_transcript_absent_from_server_history_is_not_replayed(
    saved_transcript: Path,
) -> None:
    """With neither an id nor a response in common, nothing can be placed."""
    saved_transcript.write_bytes(
        _line(_user("other-1", "unrelated question")) + _line(_assistant("other-2", "answer"))
    )
    server = _FakeServer(
        [
            {
                "id": "msg_1",
                "type": "message",
                "role": "user",
                "response_id": "resp_claude_unrelated",
                "content": [{"type": "input_text", "text": "server question"}],
            }
        ]
    )

    await _cold_resume(server)

    assert server.posts == []


@pytest.mark.asyncio
async def test_history_continued_elsewhere_is_not_replayed(saved_transcript: Path) -> None:
    """Another machine's Claude items after the last held one block the replay."""
    saved_transcript.write_bytes(
        _line(_user("u-1", "first question")) + _line(_assistant("a-1", "stale answer"))
    )
    server = _FakeServer(
        [
            _stored(_parsed(saved_transcript)["u-1:0:message"]),
            {
                "id": external_item_id(_SESSION_ID, "elsewhere:0:message"),
                "type": "message",
                "role": "assistant",
                "response_id": "resp_claude_elsewhere",
                "content": [{"type": "output_text", "text": "answer from another machine"}],
            },
        ]
    )

    await _cold_resume(server)

    assert server.posts == []


@pytest.mark.asyncio
async def test_scheduled_wake_marker_is_not_mistaken_for_divergence(
    saved_transcript: Path,
) -> None:
    """The forwarder's wake marker for a local record is this transcript's own."""
    saved_transcript.write_bytes(
        _line(_user("u-1", "first question")) + _line(_assistant("a-1", "woken answer"))
    )
    server = _FakeServer(
        [
            _stored(_parsed(saved_transcript)["u-1:0:message"]),
            {
                "id": external_item_id(_SESSION_ID, "a-1:0:scheduled_wake"),
                "type": "message",
                "role": "user",
                "response_id": "resp_claude_fresh_wake_turn",
                "content": [{"type": "input_text", "text": "[System: scheduled wake]"}],
            },
        ]
    )

    await _cold_resume(server)

    assert server.posts == ["a-1:0:message"]


@pytest.mark.asyncio
async def test_empty_server_history_receives_the_whole_transcript(
    saved_transcript: Path,
) -> None:
    """A session whose first turn never reached the server gets it in order."""
    saved_transcript.write_bytes(
        _line(_user("u-1", "first question")) + _line(_assistant("a-1", "first answer"))
    )
    server = _FakeServer()

    resumed = await _cold_resume(server)

    assert server.posts == ["u-1:0:message", "a-1:0:message"]
    assert resumed is not None
    assert "first answer" in resumed.path.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_compaction_summary_is_not_replayed_as_a_message(saved_transcript: Path) -> None:
    """The summary record maps to a compaction boundary, not an item; later items still go."""
    saved_transcript.write_bytes(
        _line(_user("u-1", "first question"))
        + _line(_user("s-1", "Summary of the conversation", isCompactSummary=True))
        + _line(_assistant("a-2", "after compaction"))
    )
    server = _FakeServer([_stored(_parsed(saved_transcript)["u-1:0:message"])])

    await _cold_resume(server)

    assert server.posts == ["a-2:0:message"]


@pytest.mark.asyncio
async def test_new_turn_after_a_rebuilt_compaction_is_delivered(saved_transcript: Path) -> None:
    """Rows a compaction snapshot stands for are not taken for another transcript."""
    saved_transcript.write_bytes(
        _line(_user("u-1", "first question")) + _line(_assistant("a-1", "first answer"))
    )
    parsed = _parsed(saved_transcript)
    saved_transcript.unlink()
    server = _FakeServer(
        [
            _stored(parsed["u-1:0:message"]),
            {
                "id": "cmp_1",
                "type": "compaction",
                "response_id": "compact_1",
                "compacted_messages": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "summary so far"}],
                        "forwarded_source_ids": ["a-1:0:message"],
                    }
                ],
            },
            # Mirrored after the boundary; the rebuild leaves it to the snapshot.
            _stored(parsed["a-1:0:message"]),
        ]
    )
    rebuilt = await _cold_resume(server)
    assert rebuilt is not None
    with rebuilt.path.open("ab") as handle:
        handle.write(_line(_user("n-1", "question after compaction")))

    await _cold_resume(server)

    assert server.posts == ["n-1:0:message"]


@pytest.mark.asyncio
async def test_rejected_item_is_skipped_and_the_rest_delivered(
    saved_transcript: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A permanent 4xx for one item drops it, as the live forwarder does."""
    monkeypatch.setattr(resume_replay, "_sleep", _no_sleep)
    saved_transcript.write_bytes(
        _line(_user("u-1", "first question"))
        + _line(_assistant("a-1", "rejected answer"))
        + _line(_user("u-2", "second question"))
    )

    def reject_first_answer(source_id: str) -> httpx.Response | None:
        if source_id == "a-1:0:message":
            return httpx.Response(422, json={"error": {"code": "invalid_input"}})
        return None

    server = _FakeServer(
        [_stored(_parsed(saved_transcript)["u-1:0:message"])], fail_post=reject_first_answer
    )

    resumed = await _cold_resume(server)

    assert server.posts == ["a-1:0:message"] * 3 + ["u-2:0:message"]
    assert server.texts() == ["first question", "second question"]
    assert resumed is not None
    assert resumed.forward_from is None
    assert "rejected answer" not in resumed.path.read_text(encoding="utf-8")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status", [pytest.param(503, id="unavailable"), pytest.param(403, id="refused")]
)
async def test_undelivered_items_are_left_to_the_forwarder(
    saved_transcript: Path,
    monkeypatch: pytest.MonkeyPatch,
    status: int,
) -> None:
    """The transcript is kept, and the forwarder starts at the first undelivered record."""
    monkeypatch.setattr(resume_replay, "_sleep", _no_sleep)
    monkeypatch.setattr(resume_replay, "_REPLAY_DEADLINE_S", 0.5)
    first_record = _line(_user("u-1", "first question"))
    saved = first_record + _line(_assistant("a-1", "first answer"))
    saved_transcript.write_bytes(saved)
    server = _FakeServer(
        [_stored(_parsed(saved_transcript)["u-1:0:message"])],
        fail_post=lambda source_id: httpx.Response(status, json={"error": {"code": "refused"}}),
    )

    resumed = await _cold_resume(server)

    assert server.posts == ["a-1:0:message"]
    assert resumed is not None
    assert resumed.path == saved_transcript
    assert resumed.forward_from == len(first_record)
    assert saved_transcript.read_bytes() == saved


@pytest.mark.asyncio
async def test_transient_failures_are_retried_until_delivered(
    saved_transcript: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unconfirmed-durability 503 and a 502 are retried; the item lands once."""
    monkeypatch.setattr(resume_replay, "_sleep", _no_sleep)
    saved_transcript.write_bytes(
        _line(_user("u-1", "first question")) + _line(_assistant("a-1", "first answer"))
    )
    failures = [
        httpx.Response(503, json={"error": {"code": "durability_unconfirmed"}}),
        httpx.Response(502),
    ]

    def flaky(source_id: str) -> httpx.Response | None:
        del source_id
        return failures.pop(0) if failures else None

    server = _FakeServer([_stored(_parsed(saved_transcript)["u-1:0:message"])], fail_post=flaky)

    resumed = await _cold_resume(server)

    assert server.posts == ["a-1:0:message"] * 3
    assert server.texts() == ["first question", "first answer"]
    assert resumed is not None
    assert "first answer" in resumed.path.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_cold_resume_tells_the_cli_forwarder_where_to_start(
    saved_transcript: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The CLI launch carries a kept transcript's first undelivered offset."""

    async def kept(*args: object, **kwargs: object) -> claude_native.ClaudeResumeTranscript:
        return claude_native.ClaudeResumeTranscript(saved_transcript, forward_from=42)

    monkeypatch.setattr(claude_native, "_prepare_claude_resume_transcript", kept)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "labels": {claude_native._WRAPPER_LABEL_KEY: claude_native._WRAPPER_LABEL_VALUE},
                "external_session_id": _CLAUDE_SESSION_ID,
            },
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport, base_url="https://example.com") as client:
        cold_resume = await claude_native._resolve_cold_resume(client, _SESSION_ID)

    assert cold_resume == claude_native._ColdResume(("--resume", _CLAUDE_SESSION_ID), 42)


def test_external_item_id_matches_the_server() -> None:
    """The client's id derivation is the one the server stores items under."""
    from omnigent.server.routes._sessions.orchestration import _new_external_conversation_item
    from omnigent.server.schemas import SessionEventInput

    body = SessionEventInput(
        type="external_conversation_item",
        data={
            "item_type": "message",
            "item_data": {"role": "user", "content": [{"type": "input_text", "text": "x"}]},
            "response_id": "resp_claude_x",
            "source_id": "u-1:0:message",
        },
    )
    persisted = _new_external_conversation_item(_SESSION_ID, body)

    assert persisted.stable_id == external_item_id(_SESSION_ID, "u-1:0:message")


def test_record_positions_cover_rebuilt_and_compacted_records() -> None:
    """Every rebuilt record maps to the last server row it stands for."""
    overlap_id = external_item_id(_SESSION_ID, "a-1:0:message")
    items: list[dict[str, Any]] = [
        {"id": "msg_1", "type": "message", "role": "user", "response_id": "r", "content": "hi"},
        {
            "id": "cmp_1",
            "type": "compaction",
            "response_id": "compact_1",
            "compacted_messages": [
                {
                    "type": "message",
                    "role": "user",
                    "content": "summary",
                    "forwarded_source_ids": ["a-1:0:message"],
                },
            ],
        },
        {"id": overlap_id, "type": "message", "role": "user", "response_id": "r", "content": "x"},
    ]
    positions = claude_native._claude_resume_record_positions(
        items, session_id=_SESSION_ID, external_session_id=_CLAUDE_SESSION_ID
    )
    records = claude_native._claude_transcript_records_from_session_items(
        items,
        session_id=_SESSION_ID,
        external_session_id=_CLAUDE_SESSION_ID,
        cwd=_WORKSPACE,
        bridge_dir=Path("/nonexistent-bridge"),
    )

    assert {record["uuid"] for record in records} <= positions.keys()
    compacted = claude_native._synthetic_claude_transcript_uuid(
        session_id=_SESSION_ID,
        external_session_id=_CLAUDE_SESSION_ID,
        item=items[1]["compacted_messages"][0],
        index=0,
    )
    assert positions[compacted] == 2
