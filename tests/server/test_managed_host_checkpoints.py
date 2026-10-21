"""Tests for turn-aligned managed sandbox checkpoints.

The scheduler runs against a virtual clock and an inline worker, so every
timing rule (debounce, long-turn interval, spacing, folding) is checked
deterministically. Stub stores stand in for the conversation and host stores:
the scheduler reads only a few attributes, so real stores would add setup
without adding cover. The deployment, policy and launchers are real types.
"""

from __future__ import annotations

import asyncio
import heapq
import itertools
import logging
from collections.abc import AsyncIterator, Callable
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest

from omnigent.db.db_models import current_workspace_id, workspace_scope
from omnigent.onboarding.sandboxes.base import (
    SandboxCapabilityError,
    SandboxGoneError,
    SandboxHostLauncher,
)
from omnigent.server import managed_host_checkpoints
from omnigent.server.managed_host_checkpoints import TurnCheckpointScheduler
from omnigent.server.managed_hosts import (
    ManagedSandboxConfig,
    ManagedSandboxDeployment,
    SandboxCheckpointPolicy,
)
from omnigent.server.routes._sessions import helpers

_DEBOUNCE_S = 10.0


async def _settle() -> None:
    """Let every runnable task step until each waits on fake time again."""
    for _ in range(50):
        await asyncio.sleep(0)


class _FakeTime:
    """A monotonic clock whose sleeps end only when the test advances it."""

    def __init__(self) -> None:
        self.now = 0.0
        self._sleepers: list[tuple[float, int, asyncio.Future[None]]] = []
        self._order = itertools.count()

    def clock(self) -> float:
        return self.now

    async def sleep(self, delay: float) -> None:
        future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        heapq.heappush(self._sleepers, (self.now + max(delay, 0.0), next(self._order), future))
        await future

    async def advance_to(self, moment: float) -> None:
        """Move to *moment*, waking each sleeper at its own deadline, in order."""
        await _settle()
        while self._sleepers and self._sleepers[0][0] <= moment:
            due, _, future = heapq.heappop(self._sleepers)
            if future.done():
                continue
            self.now = max(self.now, due)
            future.set_result(None)
            await _settle()
        self.now = moment
        await _settle()


class _Worker:
    """Runs blocking calls inline; an unset ``hold`` keeps checkpoint calls in flight."""

    def __init__(self) -> None:
        self.hold: asyncio.Event | None = None

    async def __call__(self, function: Callable[..., Any], /, *args: Any) -> Any:
        if self.hold is not None and function.__name__ == "_checkpoint":
            await self.hold.wait()
        return function(*args)


class _Stores:
    """Conversation and host store stand-in, recording the workspace of each read."""

    def __init__(self) -> None:
        self.conversations: dict[str, SimpleNamespace] = {}
        self.hosts: dict[str, SimpleNamespace] = {}
        self.read_workspaces: list[int] = []
        self.fail_reads = False

    def add_session(
        self,
        session_id: str,
        *,
        host_id: str | None = "host1",
        runner_id: str | None = None,
        provider: str | None = "acme",
        sandbox_id: str | None = "sbx-1",
    ) -> None:
        self.conversations[session_id] = SimpleNamespace(
            host_id=host_id, runner_id=runner_id, live_status="running"
        )
        if host_id is not None:
            self.hosts[host_id] = SimpleNamespace(sandbox_id=sandbox_id, sandbox_provider=provider)

    def get_conversation(self, session_id: str) -> SimpleNamespace | None:
        self.read_workspaces.append(current_workspace_id())
        if self.fail_reads:
            raise RuntimeError("database unavailable")
        return self.conversations.get(session_id)

    def list_conversations_by_runner_id(self, runner_id: str) -> list[SimpleNamespace]:
        return [conv for conv in self.conversations.values() if conv.runner_id == runner_id]

    def get_host(self, host_id: str) -> SimpleNamespace | None:
        self.read_workspaces.append(current_workspace_id())
        return self.hosts.get(host_id)


class _CheckpointLauncher(SandboxHostLauncher):
    """A managed launcher that implements ``checkpoint`` and records each call."""

    provider: ClassVar[str] = "acme"

    def __init__(self, clock: Callable[[], float]) -> None:
        self._clock = clock
        self.calls: list[tuple[str, str, float]] = []
        self.failures: list[BaseException] = []

    def prepare(self) -> None:
        """No local tooling to prepare."""

    def provision(self, name: str) -> str:
        return "sbx-1"

    def start_host(self, sandbox_id: str, **kwargs: Any) -> str:
        return "/workspace"

    def checkpoint(self, sandbox_id: str, *, reason: str) -> str | None:  # type: ignore[override]
        self.calls.append((sandbox_id, reason, self._clock()))
        if self.failures:
            raise self.failures.pop(0)
        return f"op-{len(self.calls)}"


class _PlainLauncher(_CheckpointLauncher):
    """A managed launcher that keeps the base ``checkpoint``: no capability."""

    checkpoint = SandboxHostLauncher.checkpoint  # type: ignore[assignment]


class _Harness:
    """One scheduler on a virtual clock, with stub stores and a recording launcher."""

    def __init__(
        self,
        *,
        policy: SandboxCheckpointPolicy | None = None,
        launcher_type: type[_CheckpointLauncher] = _CheckpointLauncher,
        run_sync: Callable[..., Any] | None = None,
    ) -> None:
        self.time = _FakeTime()
        self.worker = _Worker()
        self.stores = _Stores()
        self.launcher = launcher_type(self.time.clock)
        self.launchers_built = 0
        self.statuses: dict[str, str] = {}

        self.factory_failures = 0

        def build() -> SandboxHostLauncher:
            self.launchers_built += 1
            if self.factory_failures:
                self.factory_failures -= 1
                raise RuntimeError("provider SDK not ready")
            return self.launcher

        config = ManagedSandboxConfig(
            server_url="https://srv.example.com",
            launcher_factory=build,
            token_ttl_s=3600,
            provider="acme",
            checkpoint=policy or SandboxCheckpointPolicy(),
        )
        self.scheduler = TurnCheckpointScheduler(
            conversation_store=self.stores,  # type: ignore[arg-type]
            host_store=self.stores,  # type: ignore[arg-type]
            deployment=ManagedSandboxDeployment.single(config),
            turn_end_debounce_s=_DEBOUNCE_S,
            clock=self.time.clock,
            sleep=self.time.sleep,
            run_sync=run_sync or self.worker,
        )
        self.scheduler.start()

    def publish(self, session_id: str, *statuses: str) -> None:
        """Report each status as the funnel would, and persist it as live state does."""
        for status in statuses:
            previous = self.statuses.get(session_id)
            self.statuses[session_id] = status
            conversation = self.stores.conversations.get(session_id)
            if conversation is not None:
                conversation.live_status = status
            self.scheduler.note_status(session_id, previous, status)

    @property
    def reasons(self) -> list[str]:
        return [reason for _, reason, _ in self.launcher.calls]

    @property
    def call_times(self) -> list[float]:
        return [moment for _, _, moment in self.launcher.calls]


@pytest.fixture
async def make_harness() -> AsyncIterator[Callable[..., _Harness]]:
    harnesses: list[_Harness] = []

    def make(**kwargs: Any) -> _Harness:
        harness = _Harness(**kwargs)
        harnesses.append(harness)
        return harness

    yield make
    for harness in harnesses:
        if harness.worker.hold is not None:
            harness.worker.hold.set()
        await harness.scheduler.aclose()


# ── turn ends ────────────────────────────────────────────────────────────────


async def test_completed_turn_requests_one_checkpoint_once_its_end_holds(
    make_harness: Callable[..., _Harness],
) -> None:
    harness = make_harness()
    harness.stores.add_session("conv1")
    harness.publish("conv1", "running", "idle")

    await harness.time.advance_to(_DEBOUNCE_S - 0.1)
    assert harness.launcher.calls == []
    await harness.time.advance_to(_DEBOUNCE_S)
    assert harness.launcher.calls == [("sbx-1", "turn_completed", _DEBOUNCE_S)]

    # The ended turn's long-turn timer is gone with it.
    await harness.time.advance_to(5000)
    assert harness.reasons == ["turn_completed"]


@pytest.mark.parametrize(
    ("previous", "status", "requests"),
    [
        pytest.param("running", "idle", 1, id="completed"),
        pytest.param("waiting", "idle", 1, id="subagents-finished"),
        pytest.param("running", "failed", 1, id="failed"),
        pytest.param(None, "idle", 1, id="unknown-previous-after-restart"),
        pytest.param("failed", "idle", 1, id="runner-recovered"),
        pytest.param("idle", "failed", 0, id="failed-while-idle"),
        pytest.param("running", "waiting", 0, id="still-busy"),
        pytest.param(None, "running", 0, id="turn-started"),
    ],
)
async def test_which_status_changes_end_a_turn(
    make_harness: Callable[..., _Harness],
    previous: str | None,
    status: str,
    requests: int,
) -> None:
    harness = make_harness()
    harness.stores.add_session("conv1")
    harness.scheduler.note_status("conv1", previous, status)
    await harness.time.advance_to(_DEBOUNCE_S)
    assert harness.reasons == ["turn_completed"] * requests


async def test_spurious_idle_edges_from_a_terminal_watcher_are_debounced(
    make_harness: Callable[..., _Harness],
) -> None:
    """A pane that goes quiet mid-turn must not pause the computer (FORK-12)."""
    harness = make_harness()
    harness.stores.add_session("conv1")
    harness.publish("conv1", "running", "idle")
    await harness.time.advance_to(1)
    harness.publish("conv1", "running")
    await harness.time.advance_to(5)
    harness.publish("conv1", "idle")
    await harness.time.advance_to(9)
    harness.publish("conv1", "running")
    await harness.time.advance_to(60)
    assert harness.launcher.calls == []

    # A real end, then the trailing idle after a prompt redraw: still one save.
    harness.publish("conv1", "idle")
    await harness.time.advance_to(60.5)
    harness.publish("conv1", "running", "idle")
    await harness.time.advance_to(200)
    assert harness.launcher.calls == [("sbx-1", "turn_completed", 70.5)]


async def test_a_repeated_turn_end_does_not_restart_the_debounce(
    make_harness: Callable[..., _Harness],
) -> None:
    """The trailing idle a terminal watcher adds after a hook's idle changes nothing."""
    harness = make_harness()
    harness.stores.add_session("conv1")
    harness.publish("conv1", "running", "idle")
    await harness.time.advance_to(6)
    harness.publish("conv1", "idle")
    await harness.time.advance_to(100)
    assert harness.call_times == [_DEBOUNCE_S]


async def test_turn_edges_survive_other_writers_of_the_status_cache(
    make_harness: Callable[..., _Harness],
) -> None:
    """A runner status probe can write the cache between publishes."""
    harness = make_harness()
    harness.stores.add_session("conv1")
    # The probe cached "running" before the relay's first running publish.
    harness.scheduler.note_status("conv1", "running", "running")
    await harness.time.advance_to(600)
    assert harness.reasons == ["long_turn"]
    # A stale probe answer cached "idle" before the relay's real idle.
    harness.scheduler.note_status("conv1", "idle", "idle")
    await harness.time.advance_to(700)
    assert harness.reasons == ["long_turn", "turn_completed"]


async def test_turn_ends_during_an_in_flight_request_fold_into_one_follow_up(
    make_harness: Callable[..., _Harness],
) -> None:
    harness = make_harness()
    for session_id in ("conv1", "conv2", "conv3"):
        harness.stores.add_session(session_id)
        harness.publish(session_id, "running")
    await harness.time.advance_to(1)

    harness.worker.hold = asyncio.Event()  # keep the first call in flight
    harness.publish("conv1", "idle")
    await harness.time.advance_to(1 + _DEBOUNCE_S)
    harness.publish("conv2", "idle")
    harness.publish("conv3", "idle")
    await harness.time.advance_to(30)
    assert harness.launcher.calls == []

    harness.worker.hold.set()
    await _settle()
    assert harness.reasons == ["turn_completed", "turn_completed"]


async def test_folded_follow_up_keeps_the_turn_completed_reason(
    make_harness: Callable[..., _Harness],
) -> None:
    """A long-turn request folded after a turn end does not relabel the save."""
    harness = make_harness()
    for session_id in ("long", "first", "second"):
        harness.stores.add_session(session_id)
        harness.publish(session_id, "running")
    harness.worker.hold = asyncio.Event()
    harness.publish("first", "idle")
    await harness.time.advance_to(20)
    harness.publish("second", "idle")
    # "second" folds at 30; the long turn then comes due at 630 and folds too.
    await harness.time.advance_to(700)
    harness.worker.hold.set()
    await _settle()
    assert harness.reasons == ["turn_completed", "turn_completed"]


async def test_turn_end_on_one_session_ignores_another_sessions_running_edge(
    make_harness: Callable[..., _Harness],
) -> None:
    """The debounce is per session: a busy neighbour never cancels a save."""
    harness = make_harness()
    harness.stores.add_session("conv1")
    harness.stores.add_session("conv2")
    harness.publish("conv1", "running")
    harness.publish("conv2", "running", "idle")
    await harness.time.advance_to(5)
    harness.publish("conv2", "running")
    harness.publish("conv1", "idle")
    await harness.time.advance_to(15)
    assert harness.reasons == ["turn_completed"]


# ── long turns ───────────────────────────────────────────────────────────────


async def test_long_turn_is_checkpointed_every_max_interval(
    make_harness: Callable[..., _Harness],
) -> None:
    """A 25-minute turn is saved at 10 and 20 minutes, then at its end."""
    harness = make_harness()
    harness.stores.add_session("conv1")
    harness.publish("conv1", "running")

    await harness.time.advance_to(599)
    assert harness.launcher.calls == []
    await harness.time.advance_to(1500)
    assert harness.reasons == ["long_turn", "long_turn"]
    assert harness.call_times == [600, 1200]

    harness.publish("conv1", "idle")
    await harness.time.advance_to(3000)
    assert harness.reasons == ["long_turn", "long_turn", "turn_completed"]
    assert harness.call_times == [600, 1200, 1500 + _DEBOUNCE_S]


async def test_any_checkpoint_on_the_host_restarts_the_long_turn_interval(
    make_harness: Callable[..., _Harness],
) -> None:
    harness = make_harness(policy=SandboxCheckpointPolicy(max_interval_s=600))
    harness.stores.add_session("long")
    harness.stores.add_session("short")
    harness.publish("long", "running")
    harness.publish("short", "running")
    await harness.time.advance_to(300)
    harness.publish("short", "idle")

    await harness.time.advance_to(909)
    assert harness.call_times == [310]
    await harness.time.advance_to(910)
    assert harness.reasons == ["turn_completed", "long_turn"]
    assert harness.call_times == [310, 910]


async def test_long_turn_resumes_after_a_lull_seen_at_its_recheck(
    make_harness: Callable[..., _Harness],
) -> None:
    """An idle persisted just before the recheck must not end long-turn saves."""
    harness = make_harness()
    harness.stores.add_session("conv1")
    harness.publish("conv1", "running")
    await harness.time.advance_to(599)
    harness.publish("conv1", "idle")
    await harness.time.advance_to(605)
    harness.publish("conv1", "running")
    await harness.time.advance_to(1300)
    assert harness.reasons == ["long_turn", "long_turn"]
    assert harness.call_times == [605, 1205]


@pytest.mark.parametrize("how", ["deleted", "ended-elsewhere"])
async def test_long_turn_stops_for_a_session_that_is_gone_or_ended_elsewhere(
    make_harness: Callable[..., _Harness],
    how: str,
) -> None:
    harness = make_harness()
    harness.stores.add_session("conv1")
    harness.publish("conv1", "running")
    await harness.time.advance_to(100)
    if how == "deleted":
        del harness.stores.conversations["conv1"]
    else:
        harness.stores.conversations["conv1"].live_status = "idle"

    await harness.time.advance_to(5000)
    assert harness.launcher.calls == []


async def test_a_failed_first_lookup_is_retried_while_the_turn_runs_and_at_its_end(
    make_harness: Callable[..., _Harness],
) -> None:
    harness = make_harness()
    harness.stores.add_session("conv1")
    harness.stores.fail_reads = True
    harness.publish("conv1", "running")
    await harness.time.advance_to(1)
    harness.stores.fail_reads = False
    await harness.time.advance_to(1250)
    assert harness.call_times == [600, 1200]
    harness.publish("conv1", "idle")
    await harness.time.advance_to(1300)
    assert harness.reasons == ["long_turn", "long_turn", "turn_completed"]


async def test_a_failed_lookup_is_retried_when_a_short_turn_ends(
    make_harness: Callable[..., _Harness],
) -> None:
    harness = make_harness()
    harness.stores.add_session("conv1")
    harness.stores.fail_reads = True
    harness.publish("conv1", "running")
    await harness.time.advance_to(1)
    harness.stores.fail_reads = False
    harness.publish("conv1", "idle")
    await harness.time.advance_to(100)
    assert harness.call_times == [1 + _DEBOUNCE_S]


# ── spacing ──────────────────────────────────────────────────────────────────


async def test_min_interval_spaces_checkpoint_calls(
    make_harness: Callable[..., _Harness],
) -> None:
    harness = make_harness(policy=SandboxCheckpointPolicy(min_interval_s=60))
    for session_id in ("conv1", "conv2", "conv3"):
        harness.stores.add_session(session_id)
        harness.publish(session_id, "running")
    harness.publish("conv1", "idle")
    await harness.time.advance_to(20)
    harness.publish("conv2", "idle")
    await harness.time.advance_to(25)
    harness.publish("conv3", "idle")

    await harness.time.advance_to(69)
    assert harness.call_times == [10]
    await harness.time.advance_to(500)
    # Both later turn ends fold into the one spaced call.
    assert harness.call_times == [10, 70]


async def test_min_interval_is_measured_from_the_end_of_a_slow_call(
    make_harness: Callable[..., _Harness],
) -> None:
    harness = make_harness(policy=SandboxCheckpointPolicy(min_interval_s=60))
    for session_id in ("conv1", "conv2"):
        harness.stores.add_session(session_id)
        harness.publish(session_id, "running")
    harness.worker.hold = asyncio.Event()
    harness.publish("conv1", "idle")
    await harness.time.advance_to(20)
    harness.publish("conv2", "idle")
    await harness.time.advance_to(100)
    harness.worker.hold.set()
    await harness.time.advance_to(500)
    assert harness.call_times == [100, 160]


# ── providers and hosts that are skipped ─────────────────────────────────────


@pytest.mark.parametrize(
    ("launcher_type", "policy", "logged"),
    [
        pytest.param(_PlainLauncher, None, 1, id="provider-without-capability"),
        pytest.param(
            _CheckpointLauncher, SandboxCheckpointPolicy(enabled=False), 0, id="disabled"
        ),
    ],
)
async def test_skipped_providers_change_nothing_and_are_logged_once(
    make_harness: Callable[..., _Harness],
    caplog: pytest.LogCaptureFixture,
    launcher_type: type[_CheckpointLauncher],
    policy: SandboxCheckpointPolicy | None,
    logged: int,
) -> None:
    harness = make_harness(launcher_type=launcher_type, policy=policy)
    harness.stores.add_session("conv1")
    harness.stores.add_session("conv2")
    with caplog.at_level(logging.INFO, logger=managed_host_checkpoints.__name__):
        for _ in range(3):
            harness.publish("conv1", "running", "idle")
            harness.publish("conv2", "running", "idle")
            await harness.time.advance_to(harness.time.now + 1000)

    assert harness.launcher.calls == []
    # Found once, then the scheduler stops listening: no store reads at all,
    # and later status changes start no background work.
    assert harness.launchers_built == logged
    assert harness.stores.read_workspaces == []
    skipped = [r for r in caplog.records if "does not implement checkpoint" in r.getMessage()]
    assert len(skipped) == logged
    tasks_before = asyncio.all_tasks()
    harness.publish("conv1", "running", "idle")
    assert asyncio.all_tasks() == tasks_before


async def test_a_launcher_that_fails_to_build_is_retried(
    make_harness: Callable[..., _Harness], caplog: pytest.LogCaptureFixture
) -> None:
    harness = make_harness()
    harness.factory_failures = 2
    harness.stores.add_session("conv1")
    with caplog.at_level(logging.WARNING, logger=managed_host_checkpoints.__name__):
        for _ in range(3):
            harness.publish("conv1", "running", "idle")
            await harness.time.advance_to(harness.time.now + 100)
    assert harness.reasons == ["turn_completed"]
    assert harness.launchers_built >= 3
    failures = [r for r in caplog.records if "could not build" in r.getMessage()]
    assert len(failures) == 1


@pytest.mark.parametrize(
    "host",
    [
        pytest.param({"provider": None, "sandbox_id": None}, id="laptop-host"),
        pytest.param({"sandbox_id": None}, id="no-current-sandbox"),
        pytest.param({"provider": "other"}, id="another-provider"),
        pytest.param({"host_id": None}, id="no-host"),
    ],
)
async def test_sessions_without_a_capable_sandbox_are_ignored(
    make_harness: Callable[..., _Harness], host: dict[str, Any]
) -> None:
    harness = make_harness()
    harness.stores.add_session("conv1", **host)
    harness.publish("conv1", "running", "idle")
    await harness.time.advance_to(5000)
    assert harness.launcher.calls == []


async def test_host_relaunched_on_another_provider_is_not_checkpointed(
    make_harness: Callable[..., _Harness],
) -> None:
    harness = make_harness()
    harness.stores.add_session("conv1")
    harness.publish("conv1", "running", "idle")
    await _settle()
    harness.stores.hosts["host1"].sandbox_provider = "other"
    await harness.time.advance_to(_DEBOUNCE_S)
    assert harness.launcher.calls == []


async def test_mirrored_subagent_resolves_its_host_through_its_runner(
    make_harness: Callable[..., _Harness],
) -> None:
    harness = make_harness()
    harness.stores.add_session("parent", runner_id="runner1")
    harness.stores.add_session("child", host_id=None, runner_id="runner1")
    harness.publish("child", "running", "idle")
    await harness.time.advance_to(_DEBOUNCE_S)
    assert harness.reasons == ["turn_completed"]


# ── failures stay contained ──────────────────────────────────────────────────


async def test_failed_checkpoint_is_logged_and_later_turns_still_checkpoint(
    make_harness: Callable[..., _Harness], caplog: pytest.LogCaptureFixture
) -> None:
    harness = make_harness()
    harness.launcher.failures.append(RuntimeError("provider unavailable"))
    harness.stores.add_session("conv1")
    with caplog.at_level(logging.WARNING, logger=managed_host_checkpoints.__name__):
        harness.publish("conv1", "running", "idle")
        await harness.time.advance_to(100)
    assert any("failed" in r.getMessage() for r in caplog.records)

    harness.publish("conv1", "running", "idle")
    await harness.time.advance_to(200)
    assert harness.call_times == [10, 110]


async def test_gone_sandbox_is_skipped_quietly(
    make_harness: Callable[..., _Harness], caplog: pytest.LogCaptureFixture
) -> None:
    harness = make_harness()
    harness.launcher.failures.append(SandboxGoneError("gone"))
    harness.stores.add_session("conv1")
    with caplog.at_level(logging.INFO, logger=managed_host_checkpoints.__name__):
        harness.publish("conv1", "running", "idle")
        await harness.time.advance_to(100)
    assert harness.reasons == ["turn_completed"]
    records = [r for r in caplog.records if r.name == managed_host_checkpoints.__name__]
    assert [r.levelno for r in records] == [logging.INFO]
    assert "is gone" in records[0].getMessage()


async def test_capability_flag_without_a_working_method_drops_the_provider_once(
    make_harness: Callable[..., _Harness], caplog: pytest.LogCaptureFixture
) -> None:
    harness = make_harness()
    harness.launcher.failures.append(SandboxCapabilityError("not really"))
    harness.stores.add_session("conv1")
    with caplog.at_level(logging.WARNING, logger=managed_host_checkpoints.__name__):
        for _ in range(3):
            harness.publish("conv1", "running", "idle")
            await harness.time.advance_to(harness.time.now + 100)
    assert harness.reasons == ["turn_completed"]
    assert len([r for r in caplog.records if "refused the call" in r.getMessage()]) == 1


async def test_store_failure_never_reaches_the_status_path(
    make_harness: Callable[..., _Harness], caplog: pytest.LogCaptureFixture
) -> None:
    harness = make_harness()
    harness.stores.add_session("conv1")
    harness.stores.fail_reads = True
    with caplog.at_level(logging.WARNING, logger=managed_host_checkpoints.__name__):
        harness.publish("conv1", "running", "idle")
        await harness.time.advance_to(100)
    assert harness.launcher.calls == []
    assert any("could not resolve" in r.getMessage() for r in caplog.records)


async def test_long_turn_is_still_saved_when_its_recheck_cannot_read_the_store(
    make_harness: Callable[..., _Harness],
) -> None:
    harness = make_harness()
    harness.stores.add_session("conv1")
    harness.publish("conv1", "running")
    await harness.time.advance_to(100)
    harness.stores.fail_reads = True
    await harness.time.advance_to(1300)
    assert harness.call_times == [600, 1200]


async def test_aclose_cancels_pending_saves_and_ignores_later_edges(
    make_harness: Callable[..., _Harness],
) -> None:
    harness = make_harness()
    harness.stores.add_session("conv1")
    harness.publish("conv1", "running", "idle")
    await harness.scheduler.aclose()
    harness.publish("conv1", "running", "idle")
    await harness.time.advance_to(5000)
    assert harness.launcher.calls == []


# ── workspace scope and module wiring ────────────────────────────────────────


async def test_store_reads_and_calls_run_in_the_callers_workspace(
    make_harness: Callable[..., _Harness],
) -> None:
    """Every store query filters on the workspace, so worker threads must inherit it."""
    harness = make_harness(run_sync=asyncio.to_thread)
    harness.stores.add_session("conv1")
    with workspace_scope(4242):
        harness.publish("conv1", "running", "idle")
    await harness.time.advance_to(_DEBOUNCE_S)
    for _ in range(500):
        if harness.launcher.calls:
            break
        await asyncio.sleep(0.01)
    assert harness.reasons == ["turn_completed"]
    assert set(harness.stores.read_workspaces) == {4242}


def test_configure_disables_the_hook_without_managed_sandboxes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(managed_host_checkpoints, "_scheduler", None)
    stores = _Stores()
    assert managed_host_checkpoints.configure(stores, stores, None) is None  # type: ignore[arg-type]
    assert managed_host_checkpoints.configure(stores, None, object()) is None  # type: ignore[arg-type]
    # Outside an event loop, and without a scheduler, a status change is a no-op.
    managed_host_checkpoints.note_status("conv1", "running", "idle")


async def test_publish_status_drives_a_real_checkpoint(
    make_harness: Callable[..., _Harness], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The status funnel reaches the scheduler: one turn, one save."""
    harness = make_harness()
    harness.stores.add_session("conv_wired")
    monkeypatch.setattr(managed_host_checkpoints, "_scheduler", harness.scheduler)
    try:
        helpers._publish_status("conv_wired", "running")
        helpers._publish_status("conv_wired", "running")
        # A runner status probe writes the cache directly, outside the funnel.
        helpers._session_status_cache["conv_wired"] = "idle"
        helpers._publish_status("conv_wired", "idle")
        await harness.time.advance_to(100)
    finally:
        helpers._session_status_cache.pop("conv_wired", None)
        helpers._session_active_response_cache.pop("conv_wired", None)
    assert harness.reasons == ["turn_completed"]


def test_publish_status_reports_every_publish_past_the_sticky_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    published: list[tuple[str, str | None, str]] = []
    monkeypatch.setattr(
        managed_host_checkpoints, "note_status", lambda *status: published.append(status)
    )
    try:
        for status in ("running", "running", "idle", "failed", "idle", "running"):
            helpers._publish_status("conv_edges", status)
    finally:
        helpers._session_status_cache.pop("conv_edges", None)
        helpers._session_active_response_cache.pop("conv_edges", None)
    # The idle after failed is swallowed by the sticky-failed rule.
    assert published == [
        ("conv_edges", None, "running"),
        ("conv_edges", "running", "running"),
        ("conv_edges", "running", "idle"),
        ("conv_edges", "idle", "failed"),
        ("conv_edges", "failed", "running"),
    ]
