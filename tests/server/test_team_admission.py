"""The admission push to a front proxy, and the wake for open connections."""

from __future__ import annotations

import asyncio
import time

import pytest

from omnigent.entities.team import AdmissionSnapshot
from omnigent.server import team_admission
from omnigent.server.authority_changes import (
    authority_change_count,
    authority_changed,
    sleep_until_authority_change,
)
from omnigent.server.team_admission import AdmissionSync


class _Store:
    def __init__(self, failures: int = 0) -> None:
        self.snapshot = AdmissionSnapshot(
            revision=1, emails=("owner@example.test",), owners=("owner@example.test",)
        )
        self.failures = failures
        self.sweeps = 0

    def expire_invitations(self) -> int:
        self.sweeps += 1
        return 0

    def admission_snapshot(self) -> AdmissionSnapshot:
        if self.failures:
            self.failures -= 1
            raise OSError("database unavailable")
        return self.snapshot


class _Proxy:
    def __init__(self, failures: int = 0) -> None:
        self.failures = failures
        self.applied: list[AdmissionSnapshot] = []

    def apply(self, snapshot: AdmissionSnapshot) -> None:
        if self.failures:
            self.failures -= 1
            raise ConnectionError("proxy unreachable")
        self.applied.append(snapshot)


@pytest.mark.asyncio
async def test_every_push_is_applied_so_the_proxy_can_compare_it() -> None:
    """An unchanged revision is sent again: the proxy compares it with what it holds."""
    store, proxy = _Store(), _Proxy()
    sync = AdmissionSync(store, proxy)  # type: ignore[arg-type]
    assert await sync.push_once()
    assert await sync.push_once()
    assert [s.revision for s in proxy.applied] == [1, 1]
    assert store.sweeps == 2
    store.snapshot = AdmissionSnapshot(revision=2, emails=("alice@example.test",))
    assert await sync.push_once()
    assert [s.revision for s in proxy.applied] == [1, 1, 2]


@pytest.mark.asyncio
async def test_a_failed_snapshot_read_is_retried_without_a_new_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A database blip neither kills the loop nor waits for another membership change."""
    monkeypatch.setattr(team_admission, "RETRY_INITIAL_S", 0.01)
    store, proxy = _Store(failures=2), _Proxy()
    sync = AdmissionSync(store, proxy)  # type: ignore[arg-type]
    assert await sync.push_once() is False
    sync.start()
    try:
        deadline = time.monotonic() + 5
        while not proxy.applied and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        assert sync._task is not None and not sync._task.done()
    finally:
        await sync.stop()
    assert [s.revision for s in proxy.applied] == [1]


@pytest.mark.asyncio
async def test_the_next_push_is_due_when_the_next_invitation_expires() -> None:
    store, proxy = _Store(), _Proxy()
    store.snapshot = AdmissionSnapshot(
        revision=3, emails=("owner@example.test",), next_change_at=int(time.time()) + 5
    )
    sync = AdmissionSync(store, proxy)  # type: ignore[arg-type]
    assert await sync.push_once()
    assert 1.0 <= sync._until_next_change() <= 6.0
    store.snapshot = AdmissionSnapshot(revision=3, emails=("owner@example.test",))
    assert await sync.push_once()
    assert sync._until_next_change() == team_admission.PERIODIC_S


@pytest.mark.asyncio
async def test_a_failed_push_is_retried_until_applied(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(team_admission, "RETRY_INITIAL_S", 0.01)
    store, proxy = _Store(), _Proxy(failures=2)
    sync = AdmissionSync(store, proxy)  # type: ignore[arg-type]
    sync.start()
    try:
        deadline = time.monotonic() + 5
        while not proxy.applied and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
    finally:
        await sync.stop()
    assert [s.revision for s in proxy.applied] == [1]
    assert sync.applied_revision == 1


@pytest.mark.asyncio
async def test_a_request_pushes_a_change_without_waiting_for_the_period() -> None:
    store, proxy = _Store(), _Proxy()
    sync = AdmissionSync(store, proxy)  # type: ignore[arg-type]
    sync.start()
    try:
        await asyncio.sleep(0.05)
        store.snapshot = AdmissionSnapshot(revision=5, emails=())
        sync.request()
        deadline = time.monotonic() + 5
        while sync.applied_revision != 5 and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
    finally:
        await sync.stop()
    assert sync.applied_revision == 5


@pytest.mark.asyncio
async def test_an_authority_change_ends_the_wait_and_the_period_still_bounds_it() -> None:
    before = authority_change_count()
    started = time.monotonic()
    waiter = asyncio.create_task(sleep_until_authority_change(30))
    await asyncio.sleep(0.05)
    authority_changed()
    await asyncio.wait_for(waiter, timeout=2)
    assert time.monotonic() - started < 2
    assert authority_change_count() == before + 1
    started = time.monotonic()
    await sleep_until_authority_change(0.05)
    assert time.monotonic() - started >= 0.04
