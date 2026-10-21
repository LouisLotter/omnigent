"""Tests for waiting on the deployment's durability barrier."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from omnigent.server.durability import commits_durable, durability_barrier


async def test_without_a_barrier_a_commit_is_durable() -> None:
    """A deployment without asynchronous replication never waits."""
    assert durability_barrier(SimpleNamespace(state=SimpleNamespace())) is None
    assert await commits_durable(None) is True


@pytest.mark.parametrize(("confirmed", "durable"), [(True, True), (False, False)])
async def test_the_barrier_decides(confirmed: bool, durable: bool) -> None:
    """The barrier's answer is the outcome's durability."""

    async def barrier() -> bool:
        return confirmed

    app = SimpleNamespace(state=SimpleNamespace(durability_barrier=barrier))
    assert durability_barrier(app) is barrier
    assert await commits_durable(barrier) is durable


async def test_a_failing_barrier_never_confirms() -> None:
    """A barrier that fails leaves the outcome unacknowledged, so its sender retries."""

    async def barrier() -> bool:
        raise OSError("replication socket gone")

    assert await commits_durable(barrier) is False
