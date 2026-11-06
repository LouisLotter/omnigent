"""Who may publish through a session's GitHub account, and where a sandbox fork lands.

The team behind the proxy (``computer_team``): alice owns the session on her own
computer; bob and dave hold Edit and Read on it. Everything a person publishes
goes out under the computer's GitHub account, so GitHub names alice. The server
keeps who pressed the button, as a member id. A persistent provider never
provisions a one-session sandbox, so a fork cannot land on a cloud computer.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests.server.test_account_revocation import create_session
from tests.server.test_support_access import _computer, _member, _team
from tests.server.test_team_ingress import World, _join

pytestmark = [pytest.mark.timeout(90)]

_PR = "https://github.com/acme/widgets/pull/7"
_ACTIONS = [
    ("resources/github/comments", "github_comment_create"),
    ("resources/github/review-threads", "github_review_thread_create"),
    ("resources/github/review-threads/thread_1/replies", "github_review_thread_reply"),
    ("resources/github/merge", "github_merge"),
    ("resources/github/disable-auto-merge", "github_disable_auto_merge"),
    ("resources/github/details", "github_details_update"),
]


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[World]:
    yield from _team(tmp_path, monkeypatch, support=True)


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
    world: World, caplog: pytest.LogCaptureFixture, suffix: str, op: str
) -> None:
    """With no runner the action fails later; the record is written before it acts."""
    session_id = _shared_session(world, {"bob": 2})
    bob_id = _member(world, "bob").id
    logger = "omnigent.server.routes.sessions.routes_resources"

    with caplog.at_level(logging.INFO, logger=logger):
        answer = world.people["bob"].post(
            f"/v1/sessions/{session_id}/{suffix}",
            json={"pr_url": _PR, "body": "looks good", "merge_method": "squash"},
        )

    assert answer.status_code != 403, answer.text
    lines = [r.getMessage() for r in caplog.records if "github action" in r.getMessage()]
    assert lines == [f"github action op={op} session={session_id} actor_member={bob_id}"]
    assert "bob" not in lines[0] and "@" not in lines[0]


@pytest.mark.parametrize(("suffix", "op"), _ACTIONS, ids=[op for _, op in _ACTIONS])
def test_read_never_publishes(
    world: World, caplog: pytest.LogCaptureFixture, suffix: str, op: str
) -> None:
    viewer = _viewer(world)
    session_id = _shared_session(world, {"dave": 1})

    with caplog.at_level(logging.INFO):
        refused = viewer.post(
            f"/v1/sessions/{session_id}/{suffix}", json={"pr_url": _PR, "body": "looks good"}
        )

    assert refused.status_code == 403, (op, refused.text)
    assert "github action" not in caplog.text


def test_read_sees_the_linked_pull_requests_and_edit_unlinks_them(world: World) -> None:
    viewer = _viewer(world)
    session_id = _shared_session(world, {"bob": 2, "dave": 1})
    path = f"/v1/sessions/{session_id}/pull-requests"

    assert viewer.get(path).status_code == 200
    assert viewer.delete(path, params={"url": _PR}).status_code == 403
    assert world.people["bob"].delete(path, params={"url": _PR}).status_code == 200


def test_a_person_without_a_grant_reaches_none_of_it(world: World) -> None:
    session_id = _shared_session(world, {})
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
