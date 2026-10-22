"""A Claude-native cold resume delivers the saved transcript's undelivered items.

The forwarder keeps its cursor in a bridge directory under ``$TMPDIR``, which a
restored home does not have, and a cold resume rebuilds Claude's transcript
from server history. Items Claude wrote that the server never received must be
posted from the saved transcript first, once, under the forwarder's own
``source_id`` keys. These scenarios run against the real server and store.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from omnigent.harnesses.claude_native import main as claude_native
from omnigent.harnesses.claude_native import resume_replay
from omnigent.harnesses.claude_native.bridge import read_transcript_items_from_offset
from omnigent.harnesses.claude_native.forwarder import _post_external_conversation_item
from tests.server.helpers import create_test_agent

pytestmark = pytest.mark.asyncio

_CLAUDE_SESSION_ID = "4b0f2a52-2b5e-4d8e-9a39-6f0f1c1d2e3f"
_AGENT_NAME = "claude-native-ui"


def _user(uuid: str, text: str, parent: str | None) -> dict[str, Any]:
    return {
        "type": "user",
        "uuid": uuid,
        "parentUuid": parent,
        "message": {"role": "user", "content": text},
    }


def _assistant(uuid: str, text: str, parent: str) -> dict[str, Any]:
    return {
        "type": "assistant",
        "uuid": uuid,
        "parentUuid": parent,
        "message": {"role": "assistant", "content": [{"type": "text", "text": text}]},
    }


_TRANSCRIPT = [
    _user("u-1", "first question", None),
    _assistant("a-1", "first answer", "u-1"),
    _user("u-2", "second question", "a-1"),
    _assistant("a-2", "second answer", "u-2"),
]


async def _no_sleep(delay: float) -> None:
    del delay


class _CountingTransport(httpx.AsyncBaseTransport):
    """Pass requests to the app and count transcript item posts."""

    def __init__(self, inner: httpx.AsyncBaseTransport) -> None:
        self._inner = inner
        self.item_posts: list[str] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path.endswith("/events"):
            body = json.loads(request.content)
            if body.get("type") == "external_conversation_item":
                self.item_posts.append(body["data"]["source_id"])
        return await self._inner.handle_async_request(request)


async def _create_session(client: httpx.AsyncClient) -> str:
    agent = await create_test_agent(client, name="claude-replay")
    resp = await client.post("/v1/sessions", json={"agent_id": agent["id"]})
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


def _write_saved_transcript(projects: Path, workspace: Path) -> Path:
    """Write the transcript a restored home holds; returns its path."""
    target_dir = projects / claude_native._sanitize_claude_project_name(str(workspace))
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{_CLAUDE_SESSION_ID}.jsonl"
    target.write_text("".join(json.dumps(entry) + "\n" for entry in _TRANSCRIPT), encoding="utf-8")
    return target


async def _deliver_records(
    client: httpx.AsyncClient, session_id: str, transcript: Path, record_uuids: set[str]
) -> None:
    """Post some records' items the way the live forwarder does."""
    result = read_transcript_items_from_offset(transcript, 0, start_line=0, agent_name=_AGENT_NAME)
    for item in result.items:
        if item.source_id.split(":", 1)[0] in record_uuids:
            await _post_external_conversation_item(client, session_id=session_id, item=item)


async def _message_texts(client: httpx.AsyncClient, session_id: str) -> list[str]:
    items = (await client.get(f"/v1/sessions/{session_id}/items")).json()["data"]
    texts: list[str] = []
    for item in items:
        if item.get("type") != "message":
            continue
        for block in item.get("content", []):
            texts.append(block.get("text", ""))
    return texts


async def test_cold_resume_delivers_an_undelivered_item_once(
    app: FastAPI,
    client: httpx.AsyncClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The saved transcript's undelivered turn reaches the server once, and only it."""
    projects = tmp_path / "projects"
    monkeypatch.setattr(claude_native, "_CLAUDE_PROJECTS_DIR", projects)
    workspace = Path("/work/replay-repo")
    session_id = await _create_session(client)
    saved = _write_saved_transcript(projects, workspace)
    saved_bytes = saved.read_bytes()
    # The forwarder delivered the first turn; the second never reached the server.
    await _deliver_records(client, session_id, saved, {"u-1", "a-1"})

    counting = _CountingTransport(httpx.ASGITransport(app=app))
    async with httpx.AsyncClient(transport=counting, base_url="http://test") as resume_client:
        written = await claude_native._ensure_local_claude_resume_transcript(
            resume_client,
            session_id=session_id,
            external_session_id=_CLAUDE_SESSION_ID,
            workspace=workspace,
        )
        assert counting.item_posts == ["u-2:0:message", "a-2:0:message"]
        expected = ["first question", "first answer", "second question", "second answer"]
        assert await _message_texts(client, session_id) == expected
        assert written is not None
        assert "second answer" in written.read_text(encoding="utf-8")

        # The same checkpoint restored again: everything is delivered, nothing is re-posted.
        written.write_bytes(saved_bytes)
        counting.item_posts.clear()
        await claude_native._ensure_local_claude_resume_transcript(
            resume_client,
            session_id=session_id,
            external_session_id=_CLAUDE_SESSION_ID,
            workspace=workspace,
        )
        assert counting.item_posts == []
        assert await _message_texts(client, session_id) == expected


async def test_records_rebuilt_by_an_earlier_resume_are_not_reposted(
    app: FastAPI,
    client: httpx.AsyncClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After a first cold resume, only Claude's own undelivered records are posted."""
    projects = tmp_path / "projects"
    monkeypatch.setattr(claude_native, "_CLAUDE_PROJECTS_DIR", projects)
    workspace = Path("/work/replay-repo")
    session_id = await _create_session(client)
    saved = _write_saved_transcript(projects, workspace)
    await _deliver_records(client, session_id, saved, {"u-1", "a-1"})
    saved.unlink()
    # A first cold resume (a new machine) rebuilds the transcript from server history.
    rebuilt = await claude_native._ensure_local_claude_resume_transcript(
        client,
        session_id=session_id,
        external_session_id=_CLAUDE_SESSION_ID,
        workspace=workspace,
    )
    assert rebuilt is not None
    last_uuid = json.loads(rebuilt.read_text(encoding="utf-8").splitlines()[-1])["uuid"]
    # Claude continues the rebuilt transcript; the forwarder delivers one record of two.
    with rebuilt.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_user("u-3", "third question", last_uuid)) + "\n")
        handle.write(json.dumps(_assistant("a-3", "third answer", "u-3")) + "\n")
    await _deliver_records(client, session_id, rebuilt, {"u-3"})

    counting = _CountingTransport(httpx.ASGITransport(app=app))
    async with httpx.AsyncClient(transport=counting, base_url="http://test") as resume_client:
        await claude_native._ensure_local_claude_resume_transcript(
            resume_client,
            session_id=session_id,
            external_session_id=_CLAUDE_SESSION_ID,
            workspace=workspace,
        )
    assert counting.item_posts == ["a-3:0:message"]
    assert await _message_texts(client, session_id) == [
        "first question",
        "first answer",
        "third question",
        "third answer",
    ]


async def test_replay_waits_for_durability_and_lands_once(
    app: FastAPI,
    client: httpx.AsyncClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unconfirmed write is re-sent; the source-keyed append keeps one copy."""
    projects = tmp_path / "projects"
    monkeypatch.setattr(claude_native, "_CLAUDE_PROJECTS_DIR", projects)
    monkeypatch.setattr(resume_replay, "_sleep", _no_sleep)
    workspace = Path("/work/replay-repo")
    session_id = await _create_session(client)
    saved = _write_saved_transcript(projects, workspace)
    await _deliver_records(client, session_id, saved, {"u-1", "a-1", "u-2"})
    answers = [False, True]

    async def barrier() -> bool:
        return answers.pop(0) if answers else True

    counting = _CountingTransport(httpx.ASGITransport(app=app))
    app.state.durability_barrier = barrier
    try:
        async with httpx.AsyncClient(transport=counting, base_url="http://test") as resume_client:
            written = await claude_native._ensure_local_claude_resume_transcript(
                resume_client,
                session_id=session_id,
                external_session_id=_CLAUDE_SESSION_ID,
                workspace=workspace,
            )
    finally:
        app.state.durability_barrier = None
    assert counting.item_posts == ["a-2:0:message", "a-2:0:message"]
    assert (await _message_texts(client, session_id))[-1] == "second answer"
    assert (await _message_texts(client, session_id)).count("second answer") == 1
    assert written is not None
    assert "second answer" in written.read_text(encoding="utf-8")


class _LostFirstResponseTransport(_CountingTransport):
    """Commit the first item post on the server, then lose its response."""

    def __init__(self, inner: httpx.AsyncBaseTransport) -> None:
        super().__init__(inner)
        self._lost = False

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        response = await super().handle_async_request(request)
        if self.item_posts and not self._lost:
            self._lost = True
            raise httpx.ReadTimeout("response lost", request=request)
        return response


async def test_replay_after_a_lost_response_lands_once(
    app: FastAPI,
    client: httpx.AsyncClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A post whose answer is lost is retried; the server keeps one copy."""
    projects = tmp_path / "projects"
    monkeypatch.setattr(claude_native, "_CLAUDE_PROJECTS_DIR", projects)
    monkeypatch.setattr(resume_replay, "_sleep", _no_sleep)
    workspace = Path("/work/replay-repo")
    session_id = await _create_session(client)
    saved = _write_saved_transcript(projects, workspace)
    await _deliver_records(client, session_id, saved, {"u-1", "a-1"})

    lossy = _LostFirstResponseTransport(httpx.ASGITransport(app=app))
    async with httpx.AsyncClient(transport=lossy, base_url="http://test") as resume_client:
        await claude_native._ensure_local_claude_resume_transcript(
            resume_client,
            session_id=session_id,
            external_session_id=_CLAUDE_SESSION_ID,
            workspace=workspace,
        )
    assert lossy.item_posts == ["u-2:0:message", "u-2:0:message", "a-2:0:message"]
    assert await _message_texts(client, session_id) == [
        "first question",
        "first answer",
        "second question",
        "second answer",
    ]
