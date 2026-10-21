"""Tests for applying each native status edge once."""

from __future__ import annotations

import asyncio

import pytest

from omnigent.server.status_edges import StatusEdges, status_edge_id


class _Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def test_status_edge_id_takes_only_a_usable_key() -> None:
    """An edge without a usable key is applied every time, as before."""
    assert status_edge_id({"Idempotency-Key": "4f0c"}) == "4f0c"
    assert status_edge_id({}) is None
    assert status_edge_id({"Idempotency-Key": ""}) is None
    assert status_edge_id({"Idempotency-Key": "x" * 129}) is None


async def test_a_retried_edge_is_answered_from_its_first_application() -> None:
    """A repeat, even one arriving while the first is applied, does not apply again."""
    edges = StatusEdges()
    applied: list[str] = []
    release = asyncio.Event()

    async def apply() -> dict[str, bool]:
        applied.append("idle")
        await release.wait()
        return {"queued": False}

    first = asyncio.create_task(edges.apply_once("conv_a", "edge-1", apply))
    await asyncio.sleep(0)
    repeat = asyncio.create_task(edges.apply_once("conv_a", "edge-1", apply))
    await asyncio.sleep(0)
    release.set()
    assert await first == await repeat == {"queued": False}
    assert await edges.apply_once("conv_a", "edge-1", apply) == {"queued": False}
    assert applied == ["idle"]
    # The same id on another session is another edge.
    await edges.apply_once("conv_b", "edge-1", apply)
    assert applied == ["idle", "idle"]


async def test_an_edge_whose_application_failed_is_applied_by_its_retry() -> None:
    """A failure is not remembered: the retry, and a repeat waiting on it, apply the edge."""
    edges = StatusEdges()
    attempts: list[int] = []
    release = asyncio.Event()

    async def fail_then_succeed() -> str:
        attempts.append(len(attempts))
        if len(attempts) == 1:
            await release.wait()
            raise RuntimeError("store unavailable")
        return "applied"

    first = asyncio.create_task(edges.apply_once("conv_a", "edge-1", fail_then_succeed))
    await asyncio.sleep(0)
    waiting = asyncio.create_task(edges.apply_once("conv_a", "edge-1", fail_then_succeed))
    await asyncio.sleep(0)
    release.set()
    with pytest.raises(RuntimeError):
        await first
    assert await waiting == "applied"
    assert attempts == [0, 1]


async def test_an_edge_is_remembered_for_its_ttl_and_within_the_bound() -> None:
    """Old and excess edges are forgotten, so memory stays bounded."""
    clock = _Clock()
    edges = StatusEdges(ttl_s=60.0, max_edges=2, clock=clock)
    applied: list[str] = []

    def applying(name: str):
        async def apply() -> str:
            applied.append(name)
            return name

        return apply

    await edges.apply_once("conv", "a", applying("a"))
    clock.now += 61.0
    await edges.apply_once("conv", "a", applying("a"))
    await edges.apply_once("conv", "b", applying("b"))
    await edges.apply_once("conv", "c", applying("c"))
    await edges.apply_once("conv", "a", applying("a"))
    assert applied == ["a", "a", "b", "c", "a"]
