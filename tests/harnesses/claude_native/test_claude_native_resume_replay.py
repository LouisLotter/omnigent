"""Claude-native cold resume: replay of a saved transcript's undelivered items.

The real-server scenarios (exactly once, durability barrier, lost response)
live in ``tests/server/integration/test_claude_native_resume_replay.py``. These
cover the placement rules and the failure paths against a fake server that
keys items by ``source_id`` as the real one does.
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


def _server_item(source_id: str, text: str, *, role: str = "user") -> dict[str, Any]:
    """A committed item as the server lists it after a source-keyed post."""
    block_type = "input_text" if role == "user" else "output_text"
    return {
        "id": external_item_id(_SESSION_ID, source_id),
        "type": "message",
        "role": role,
        "response_id": "resp_claude_" + source_id.split(":", 1)[0],
        "content": [{"type": block_type, "text": text}],
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
        self._fail_post = fail_post

    def handle(self, request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path.endswith("/items"):
            return httpx.Response(200, json={"data": self.items, "has_more": False})
        if request.method == "POST" and request.url.path.endswith("/events"):
            data = json.loads(request.content)["data"]
            self.posts.append(data["source_id"])
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
        return [block["text"] for item in self.items for block in item.get("content", [])]


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


async def _cold_resume(server: _FakeServer) -> Path | None:
    transport = httpx.MockTransport(server.handle)
    async with httpx.AsyncClient(transport=transport, base_url="https://example.com") as client:
        return await claude_native._ensure_local_claude_resume_transcript(
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
    server = _FakeServer([_server_item("u-1:0:message", "first question")])

    written = await _cold_resume(server)

    assert server.posts == ["a-1:0:message"]
    assert server.texts() == ["first question", "first answer"]
    assert written is not None
    assert "second question" not in written.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_unreadable_transcript_is_rebuilt_from_server_history(
    saved_transcript: Path,
) -> None:
    """A transcript with no parsable record replays nothing and is rewritten."""
    saved_transcript.write_bytes(b"\x00\x01garbage\nnot json\n")
    server = _FakeServer([_server_item("u-1:0:message", "first question")])

    written = await _cold_resume(server)

    assert server.posts == []
    assert written == saved_transcript
    assert "first question" in written.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_transcript_that_cannot_be_read_is_rebuilt(
    saved_transcript: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A read error replays nothing and leaves the rewrite to proceed."""
    saved_transcript.write_bytes(
        _line(_user("u-1", "first question")) + _line(_assistant("a-1", "unread answer"))
    )

    def unreadable(*args: object, **kwargs: object) -> None:
        raise PermissionError("transcript is not readable")

    monkeypatch.setattr(resume_replay, "read_transcript_items_from_offset", unreadable)
    server = _FakeServer([_server_item("u-1:0:message", "first question")])

    written = await _cold_resume(server)

    assert server.posts == []
    assert written is not None
    assert "first question" in written.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_transcript_absent_from_server_history_is_not_replayed(
    saved_transcript: Path,
) -> None:
    """With no item in common, e.g. a fork's copied prefix, nothing can be placed."""
    saved_transcript.write_bytes(
        _line(_user("copied-1", "copied question")) + _line(_assistant("new-1", "new answer"))
    )
    server = _FakeServer([_server_item("other:0:message", "copied question")])

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
            _server_item("u-1:0:message", "first question"),
            _server_item("elsewhere:0:message", "answer from another machine", role="assistant"),
        ]
    )

    await _cold_resume(server)

    assert server.posts == []


@pytest.mark.asyncio
async def test_empty_server_history_receives_the_whole_transcript(
    saved_transcript: Path,
) -> None:
    """A session whose first turn never reached the server gets it in order."""
    saved_transcript.write_bytes(
        _line(_user("u-1", "first question")) + _line(_assistant("a-1", "first answer"))
    )
    server = _FakeServer()

    written = await _cold_resume(server)

    assert server.posts == ["u-1:0:message", "a-1:0:message"]
    assert written is not None
    assert "first answer" in written.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_compaction_summary_is_not_replayed_as_a_message(saved_transcript: Path) -> None:
    """The summary record maps to a compaction boundary, not an item; later items still go."""
    saved_transcript.write_bytes(
        _line(_user("u-1", "first question"))
        + _line(_user("s-1", "Summary of the conversation", isCompactSummary=True))
        + _line(_assistant("a-2", "after compaction"))
    )
    server = _FakeServer([_server_item("u-1:0:message", "first question")])

    await _cold_resume(server)

    assert server.posts == ["a-2:0:message"]


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
        [_server_item("u-1:0:message", "first question")], fail_post=reject_first_answer
    )

    written = await _cold_resume(server)

    assert server.posts == ["a-1:0:message"] * 3 + ["u-2:0:message"]
    assert server.texts() == ["first question", "second question"]
    assert written is not None
    assert "rejected answer" not in written.read_text(encoding="utf-8")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "attempts"),
    [pytest.param(503, 1, id="server-unavailable"), pytest.param(403, 1, id="refused")],
)
async def test_undeliverable_items_keep_the_local_transcript(
    saved_transcript: Path,
    monkeypatch: pytest.MonkeyPatch,
    status: int,
    attempts: int,
) -> None:
    """Items the server cannot take are kept in the transcript, which is not rewritten."""
    monkeypatch.setattr(resume_replay, "_sleep", _no_sleep)
    monkeypatch.setattr(resume_replay, "_REPLAY_STALL_TIMEOUT_S", 0.0)
    saved = _line(_user("u-1", "first question")) + _line(_assistant("a-1", "first answer"))
    saved_transcript.write_bytes(saved)
    server = _FakeServer(
        [_server_item("u-1:0:message", "first question")],
        fail_post=lambda source_id: httpx.Response(status, json={"error": {"code": "refused"}}),
    )

    written = await _cold_resume(server)

    assert server.posts == ["a-1:0:message"] * attempts
    assert written == saved_transcript
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

    server = _FakeServer([_server_item("u-1:0:message", "first question")], fail_post=flaky)

    written = await _cold_resume(server)

    assert server.posts == ["a-1:0:message"] * 3
    assert server.texts() == ["first question", "first answer"]
    assert written is not None
    assert "first answer" in written.read_text(encoding="utf-8")


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
    """Every record a rebuild writes, compacted messages included, maps to its item."""
    items: list[dict[str, Any]] = [
        _server_item("u-1:0:message", "first question"),
        {
            "id": "cmp_1",
            "type": "compaction",
            "response_id": "compact_1",
            "compacted_messages": [
                {"type": "message", "role": "user", "content": "summary"},
            ],
        },
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
    rebuilt_uuids = {record["uuid"] for record in records}

    assert rebuilt_uuids <= positions.keys()
    first = claude_native._synthetic_claude_transcript_uuid(
        session_id=_SESSION_ID, external_session_id=_CLAUDE_SESSION_ID, item=items[0], index=0
    )
    assert positions[first] == 0
