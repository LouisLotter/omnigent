"""Who may publish through a session's GitHub account, and where a sandbox fork lands.

The team behind the proxy (``computer_team``): alice owns the session on her own
computer; bob and dave hold Edit and Read on it. Everything a person publishes
goes out under the computer's GitHub account, so GitHub names alice. The server
keeps who pressed the button, as a member id. A persistent provider never
provisions a one-session sandbox, so a fork cannot land on a cloud computer.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from omnigent.server import session_pull_requests as server_prs
from omnigent.server.routes import sessions
from omnigent.stores.session_pull_request_store import SessionPullRequestStore
from tests.server.test_account_revocation import create_session
from tests.server.test_support_access import _computer, _db_url, _member, _team
from tests.server.test_team_ingress import World, _join

pytestmark = [pytest.mark.timeout(90)]

_PR = "https://github.com/acme/widgets/pull/7"
_HEAD = "0123456789abcdef0123456789abcdef01234567"
_ANCHOR = {"path": "src/app.py", "line": 12, "side": "RIGHT"}
_ACTIONS = [
    ("resources/github/comments", "github_comment_create"),
    ("resources/github/review-threads", "github_review_thread_create"),
    ("resources/github/review-threads/thread_1/replies", "github_review_thread_reply"),
    ("resources/github/merge", "github_merge"),
    ("resources/github/disable-auto-merge", "github_disable_auto_merge"),
    ("resources/github/details", "github_details_update"),
]
# Everything a person can fill in; each action forwards only what it is for.
_FORM = {
    "pr_url": _PR,
    "body": "looks good",
    "merge_method": "squash",
    "expected_head": _HEAD,
    "anchor": _ANCHOR,
    "title": "A better title",
}
_LINKED = {"pr_url": _PR, "server_tracked": True}
_RUNNER_RECEIVES: dict[str, dict[str, Any]] = {
    "github_comment_create": {**_LINKED, "body": "looks good"},
    "github_review_thread_create": {**_LINKED, "body": "looks good", "anchor": _ANCHOR},
    "github_review_thread_reply": {**_LINKED, "thread_id": "thread_1", "body": "looks good"},
    "github_merge": {**_LINKED, "merge_method": "squash", "expected_head": _HEAD},
    "github_disable_auto_merge": _LINKED,
    "github_details_update": {**_LINKED, "title": "A better title", "body": "looks good"},
}
_PUBLISHED = {"object": "session.github.published", "url": f"{_PR}#issuecomment-1"}


class _RecordingRunner:
    """The session's runner: records each request and answers that it was carried out."""

    def __init__(self) -> None:
        self.received: list[tuple[str, str, dict[str, Any]]] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else {}
        self.received.append((request.method, request.url.path, body))
        return httpx.Response(200, json=_PUBLISHED)


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[World]:
    yield from _team(tmp_path, monkeypatch, support=True)


@pytest.fixture
def runner(monkeypatch: pytest.MonkeyPatch) -> _RecordingRunner:
    recording = _RecordingRunner()

    async def _client(*_args: Any, **_kwargs: Any) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            transport=httpx.MockTransport(recording.handle), base_url="http://runner.test"
        )

    monkeypatch.setattr(sessions, "_get_runner_client_for_resource_access", _client)
    return recording


@pytest.fixture
def link_pull_request(tmp_path: Path) -> Callable[[str], None]:
    """Have a session's host report *_PR*, as its first complete list."""
    store = SessionPullRequestStore(_db_url(tmp_path))
    reports = server_prs.parse_reports(
        [
            {
                "url": _PR,
                "provider": "github",
                "host": "github.com",
                "repository": "acme/widgets",
                "number": 7,
                "relationship": "created",
                "source": "native_hook",
                "first_seen_at": 100.0,
                "last_seen_at": 100.0,
            }
        ]
    )

    def _link(session_id: str) -> None:
        assert store.apply_report(session_id, reports, complete=True) is not None

    return _link


def _shared_session(world: World, grants: dict[str, int]) -> str:
    """A session of alice's, shared at each level in *grants*."""
    alice = world.people["alice"]
    created = create_session(alice)
    assert created.status_code == 201, created.text
    session_id = created.json()["session_id"]
    for person, level in grants.items():
        shared = alice.put(
            f"/v1/sessions/{session_id}/permissions",
            json={"user_id": person, "level": level},
        )
        assert shared.status_code in (200, 201, 204), shared.text
    return session_id


def _viewer(world: World) -> TestClient:
    """A team member who holds Read on the session, added for this test."""
    if "dave" not in world.people:
        _join(world, "dave")
    return world.people["dave"]


@pytest.mark.parametrize(("suffix", "op"), _ACTIONS, ids=[op for _, op in _ACTIONS])
def test_edit_publishes_and_the_log_names_the_member(
    world: World,
    runner: _RecordingRunner,
    link_pull_request: Callable[[str], None],
    caplog: pytest.LogCaptureFixture,
    suffix: str,
    op: str,
) -> None:
    """The action reaches the runner as asked, with the acting member on record."""
    session_id = _shared_session(world, {"bob": 2})
    link_pull_request(session_id)
    bob_id = _member(world, "bob").id
    logger = "omnigent.server.routes.sessions.routes_resources"

    with caplog.at_level(logging.INFO, logger=logger):
        answer = world.people["bob"].post(f"/v1/sessions/{session_id}/{suffix}", json=_FORM)

    assert answer.status_code == 200, answer.text
    assert answer.json() == _PUBLISHED
    assert runner.received == [
        ("POST", f"/v1/sessions/{session_id}/{suffix}", _RUNNER_RECEIVES[op])
    ]
    lines = [r.getMessage() for r in caplog.records if "github action" in r.getMessage()]
    assert lines == [f"github action op={op} session={session_id} actor_member={bob_id}"]
    assert "bob" not in lines[0] and "@" not in lines[0]


@pytest.mark.parametrize(("suffix", "op"), _ACTIONS, ids=[op for _, op in _ACTIONS])
def test_edit_cannot_publish_to_a_pull_request_the_session_does_not_link(
    world: World,
    runner: _RecordingRunner,
    link_pull_request: Callable[[str], None],
    suffix: str,
    op: str,
) -> None:
    """The computer's GitHub account answers only for the session's own PRs."""
    session_id = _shared_session(world, {"bob": 2})
    link_pull_request(session_id)

    refused = world.people["bob"].post(
        f"/v1/sessions/{session_id}/{suffix}",
        json={**_FORM, "pr_url": "https://github.com/acme/widgets/pull/8"},
    )

    assert refused.status_code == 400, (op, refused.text)
    assert runner.received == []


@pytest.mark.parametrize(("suffix", "op"), _ACTIONS, ids=[op for _, op in _ACTIONS])
def test_read_never_publishes(
    world: World,
    runner: _RecordingRunner,
    link_pull_request: Callable[[str], None],
    caplog: pytest.LogCaptureFixture,
    suffix: str,
    op: str,
) -> None:
    viewer = _viewer(world)
    session_id = _shared_session(world, {"dave": 1})
    link_pull_request(session_id)

    with caplog.at_level(logging.INFO):
        refused = viewer.post(f"/v1/sessions/{session_id}/{suffix}", json=_FORM)

    assert refused.status_code == 403, (op, refused.text)
    assert "github action" not in caplog.text
    assert runner.received == []


def test_read_sees_the_linked_pull_requests_and_edit_unlinks_them(
    world: World, link_pull_request: Callable[[str], None]
) -> None:
    viewer = _viewer(world)
    session_id = _shared_session(world, {"bob": 2, "dave": 1})
    link_pull_request(session_id)
    path = f"/v1/sessions/{session_id}/pull-requests"

    def _linked() -> list[str]:
        listed = viewer.get(path)
        assert listed.status_code == 200, listed.text
        return [entry["url"] for entry in listed.json()["data"]]

    assert _linked() == [_PR]
    assert viewer.delete(path, params={"url": _PR}).status_code == 403
    assert _linked() == [_PR]

    unlinked = world.people["bob"].delete(path, params={"url": _PR})

    assert unlinked.status_code == 200, unlinked.text
    assert unlinked.json()["data"] == []
    assert _linked() == []


def test_a_person_without_a_grant_reaches_none_of_it(
    world: World, runner: _RecordingRunner, link_pull_request: Callable[[str], None]
) -> None:
    session_id = _shared_session(world, {})
    link_pull_request(session_id)
    stranger = world.people["bob"]
    base = f"/v1/sessions/{session_id}"

    answers = [
        stranger.get(f"{base}/pull-requests"),
        stranger.get(f"{base}/resources/repository-branches"),
        stranger.get(f"{base}/resources/github/review-threads", params={"pr_url": _PR}),
        stranger.get(
            f"{base}/resources/github/review-threads/thread_1/comments",
            params={"pr_url": _PR, "after": "cursor"},
        ),
        stranger.delete(f"{base}/pull-requests", params={"url": _PR}),
    ]
    for answer in answers:
        assert answer.status_code in (403, 404), answer.text
    assert runner.received == []


def test_a_fork_onto_a_new_sandbox_is_refused_on_a_persistent_provider(world: World) -> None:
    """Nothing is provisioned: a computer is added as a host, never launched for a fork."""
    session_id = _shared_session(world, {"bob": 1})
    computer = _computer(world)
    calls_before = list(computer.owners)

    refused = world.people["bob"].post(
        f"/v1/sessions/{session_id}/fork",
        json={"host_type": "managed", "workspace": "https://github.com/acme/widgets"},
    )

    assert refused.status_code == 400, refused.text
    assert "creates persistent hosts only" in refused.text
    assert computer.owners == calls_before
