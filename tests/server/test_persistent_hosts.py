"""Persistent managed hosts: readiness, admission, start claims and the fenced sleep.

Each test drives the real lifecycle module against a real host store and host
registry. A fake computer stands in for the provider (typed status, start
claims, suspend) and a fake host process connects only with a launch token the
store accepts and answers ``host.quiesce``. Blocking points in the fakes park a
coroutine at a chosen await, so interleavings are deterministic. The assertions
check the physical outcome: what the provider was asked to do, which tokens
worked, and when work was admitted relative to that.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import dataclasses
import threading
import uuid
from collections.abc import AsyncIterator, Callable
from typing import Any

import pytest

from omnigent.errors import ErrorCode, OmnigentError
from omnigent.onboarding.sandboxes.types import AdmissionReport
from omnigent.server import persistent_hosts
from omnigent.server.host_registry import HostConnection, HostRegistry
from omnigent.server.managed_hosts import (
    MANAGED_HOST_ONLINE_TIMEOUT_S,
    ManagedSandboxConfig,
    ManagedSandboxDeployment,
    PersistentHostPolicy,
    resume_managed_host,
)
from omnigent.stores.host_store import HostStore
from tests.server.persistent_host_fakes import (
    FIRST_TOKEN,
    HOST_ID,
    OWNER,
    SANDBOX_ID,
    FakeComputer,
    FakeHost,
    FakeUntypedComputer,
    call_soon,
    register_persistent_row,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.timeout(60)]

# Set by a request under test; worker threads see it through to_thread's context copy.
_in_request: contextvars.ContextVar[bool] = contextvars.ContextVar("in_request", default=False)


@pytest.fixture(autouse=True)
def _bounded_waits(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep fences and redelivery waits from outliving a failing test.

    Each test runs on its own event loop, so the host's lifecycle lock (bound
    to the loop that first contended it) starts fresh too.
    """
    monkeypatch.setattr(persistent_hosts, "FENCE_DRAIN_S", 5.0)
    monkeypatch.setattr(persistent_hosts, "_STATUS_POLL_S", 0.05)
    monkeypatch.setattr(persistent_hosts, "_UNKNOWN_RETRY_S", 0.05)
    monkeypatch.setattr(persistent_hosts, "_resume_locks", {})


def _deployment(
    computer: FakeComputer, *, online_timeout_s: float | None = 10.0, **policy: Any
) -> ManagedSandboxDeployment:
    return ManagedSandboxDeployment.single(
        ManagedSandboxConfig(
            server_url="http://omnigent.test",
            launcher_factory=lambda: computer,
            token_ttl_s=3600,
            provider=computer.provider,
            persistent=PersistentHostPolicy(
                idle_sleep_s=policy.pop("idle_sleep_s", 0),
                admission_max_age_s=policy.pop("admission_max_age_s", 0),
                **policy,
            ),
            online_timeout_s=online_timeout_s,
        )
    )


@pytest.fixture
async def host(db_uri: str) -> AsyncIterator[FakeHost]:
    """An awake persistent host whose computer reports ``host_running``."""
    store = HostStore(db_uri)
    registry = HostRegistry()
    # Wired as create_app wires them.
    registry.admission_gate = persistent_hosts.admit_host_work
    registry.send_guard = persistent_hosts.guard_host_request
    computer = FakeComputer()
    fake = FakeHost(store, registry, computer)
    loop = asyncio.get_running_loop()
    computer.on_host_start = lambda token: call_soon(loop, fake.connect, token)
    persistent_hosts._states.clear()
    persistent_hosts.configure(store, registry, _deployment(computer))
    register_persistent_row(store)
    assert fake.connect(FIRST_TOKEN)
    yield fake
    fake.close()
    for state in persistent_hosts._states.values():
        if state.wake_task is not None:
            state.wake_task.cancel()
    persistent_hosts.configure(None, None, None)
    persistent_hosts._states.clear()


async def _admitted_after(host: FakeHost, ready: Callable[[], Any]) -> str:
    """Run *ready* to completion and note it in the provider's event order."""
    await ready()
    host.computer.events.append("admitted")
    return "admitted"


def _ready(operation: str, timeout_s: float = 30) -> Callable[[], Any]:
    return lambda: persistent_hosts.ensure_host_ready(
        HOST_ID, operation=operation, timeout_s=timeout_s
    )


async def _asleep(host: FakeHost) -> None:
    """Put the host to sleep and forget the provider calls that took."""
    assert await persistent_hosts.sleep_if_idle(HOST_ID) == "stopped"
    host.computer.events.clear()


def _token(host: FakeHost) -> str | None:
    return host.token


# ── Readiness and the fence (X6 scenarios) ─────────────────────────────


@pytest.mark.parametrize("park", ["host_read", "admission_report"])
async def test_sleep_closing_while_readiness_awaits_holds_the_request(
    host: FakeHost, park: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sleep that closes admission during readiness's await is never raced.

    The request is parked at each await between the admission check and the
    grant; the fence then stops the sandbox. The request must be admitted only
    after a fresh wake, never on the host being stopped.
    """
    gate = threading.Event()
    if park == "admission_report":
        host.computer.admission_gate = gate
    else:
        original = host.store.get_host
        reads = {"request": 0}

        def parked_get_host(host_id: str) -> Any:
            # Only the request's own reads count; to_thread carries its context.
            if _in_request.get():
                reads["request"] += 1
                if reads["request"] == 2:  # the re-read after the admission wait
                    host.computer.parked.set()
                    gate.wait(10)
            return original(host_id)

        monkeypatch.setattr(host.store, "get_host", parked_get_host)

    async def request_task() -> str:
        _in_request.set(True)
        return await _admitted_after(host, _ready("message"))

    request = asyncio.create_task(request_task())
    assert await asyncio.to_thread(host.computer.parked.wait, 10)
    assert await persistent_hosts.sleep_if_idle(HOST_ID) == "stopped"
    gate.set()
    assert await asyncio.wait_for(request, 30) == "admitted"
    events = host.computer.events
    assert events.index("stopped") < events.index("resume") < events.index("admitted")
    assert events.count("start_host") == 1


async def test_distinct_request_waits_while_another_owns_admission(
    host: FakeHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the owner of an admission passes a closed fence; another request waits."""
    monkeypatch.setattr(persistent_hosts, "FENCE_DRAIN_S", 30.0)
    owner_done = asyncio.Event()
    order: list[str] = []

    owner_admitted = asyncio.Event()

    async def owner() -> None:
        await persistent_hosts.ensure_host_ready(HOST_ID, operation="message", timeout_s=5)
        owner_admitted.set()
        await asyncio.sleep(0.3)  # the fence closes meanwhile
        await persistent_hosts.admit_host_work(HOST_ID, "runner_launch")
        order.append("owner launch admitted")
        host.runners.append("runner-1")  # the launch shows on the host
        await owner_done.wait()

    async def other() -> None:
        await persistent_hosts.admit_host_work(HOST_ID, "runner_launch")
        order.append("other launch admitted")

    owner_task = asyncio.create_task(owner())
    await asyncio.wait_for(owner_admitted.wait(), 5)
    fence = asyncio.create_task(persistent_hosts.sleep_if_idle(HOST_ID))
    await asyncio.sleep(0.1)
    other_task = asyncio.create_task(other())
    await asyncio.sleep(0.5)
    assert order == ["owner launch admitted"]
    owner_done.set()
    await owner_task
    assert await asyncio.wait_for(fence, 10) == "abandoned"
    await asyncio.wait_for(other_task, 10)
    assert order == ["owner launch admitted", "other launch admitted"]
    assert "suspend" not in host.computer.events


async def test_operation_longer_than_any_lease_keeps_the_host_awake(
    host: FakeHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An admitted operation blocks sleep for as long as it runs, not for a fixed lease."""
    admitted, release = asyncio.Event(), asyncio.Event()

    async def long_install() -> None:
        await persistent_hosts.admit_host_work(HOST_ID, "harness_install")
        admitted.set()
        await release.wait()

    install = asyncio.create_task(long_install())
    await asyncio.wait_for(admitted.wait(), 5)
    clock = persistent_hosts.time.monotonic
    monkeypatch.setattr(persistent_hosts.time, "monotonic", lambda: clock() + 3600)
    outcomes = await persistent_hosts.PersistentHostIdleMonitor().sweep_once()
    assert outcomes[HOST_ID] == "busy: admitted work ['harness_install']"
    assert "suspend" not in host.computer.events
    release.set()
    await install
    outcomes = await persistent_hosts.PersistentHostIdleMonitor().sweep_once()
    assert outcomes[HOST_ID] == "stopped"


async def test_cancelled_sleep_after_the_stop_was_accepted_stays_closed(host: FakeHost) -> None:
    """Cancelling the fence mid-suspend keeps admission closed until the outcome is known."""
    host.computer.suspend_gate = threading.Event()
    fence = asyncio.create_task(persistent_hosts.sleep_if_idle(HOST_ID))
    await asyncio.to_thread(host.computer.suspend_accepted.wait, 10)
    fence.cancel()
    with pytest.raises(asyncio.CancelledError):
        await fence
    state = persistent_hosts._states[HOST_ID]
    assert not state.admitting.is_set()
    recorded = host.store.get_host(HOST_ID)
    assert recorded is not None
    assert state.unknown_suspend_operation == recorded.sleep_operation_id is not None
    request = asyncio.create_task(_admitted_after(host, _ready("message")))
    await asyncio.sleep(0.5)
    assert not request.done()
    host.computer.suspend_gate.set()  # the stop the provider accepted completes
    assert await asyncio.wait_for(request, 30) == "admitted"
    assert host.computer.events == [
        "suspend",
        "stopped",
        "resume",
        "claim",
        "start_host",
        "admitted",
    ]
    current = host.store.get_host(HOST_ID)
    assert current is not None and current.sleep_operation_id is None


async def test_restart_before_the_stop_reply_recovers_the_sleep(host: FakeHost) -> None:
    """A new server process finds the recorded sleep and resolves it before admitting."""
    host.computer.suspend_gate = threading.Event()
    fence = asyncio.create_task(persistent_hosts.sleep_if_idle(HOST_ID))
    await asyncio.to_thread(host.computer.suspend_accepted.wait, 10)
    # The process dies: in-memory lifecycle state and the live tunnel are gone.
    fence.cancel()
    with pytest.raises(asyncio.CancelledError):
        await fence
    persistent_hosts._states.clear()
    host.registry.deregister(HOST_ID)
    assert await persistent_hosts.recover_pending_sleeps() == [HOST_ID]
    # The old token stopped working before the stop was requested.
    assert host.store.resolve_launch_token(HOST_ID, FIRST_TOKEN) is None
    request = asyncio.create_task(_admitted_after(host, _ready("message")))
    await asyncio.sleep(0.5)
    assert not request.done()
    host.computer.suspend_gate.set()
    assert await asyncio.wait_for(request, 30) == "admitted"
    assert host.computer.events[:2] == ["suspend", "stopped"]
    assert host.computer.events[-1] == "admitted"
    assert host.computer.events.count("start_host") == 1


async def test_refused_sleep_restores_the_same_token_exactly(host: FakeHost) -> None:
    """A refused sleep reopens admission and the host's token works again, unextended."""
    before = host.store.get_host(HOST_ID)
    assert before is not None
    host.computer.suspend_refuses = True
    assert await persistent_hosts.sleep_if_idle(HOST_ID) == "refused"
    after = host.store.get_host(HOST_ID)
    assert after is not None and after.token_expires_at == before.token_expires_at
    assert host.store.resolve_launch_token(HOST_ID, FIRST_TOKEN) is not None
    assert persistent_hosts._states[HOST_ID].admitting.is_set()
    await persistent_hosts.ensure_host_ready(HOST_ID, operation="message", timeout_s=5)
    assert host.computer.events == ["suspend"]


async def test_not_saved_refuses_new_work_with_the_reason(host: FakeHost) -> None:
    """A provider that reports "not saved" refuses new work, typed and readable."""
    host.computer.admitting = False
    with pytest.raises(OmnigentError) as raised:
        await persistent_hosts.ensure_host_ready(HOST_ID, operation="message", timeout_s=5)
    assert raised.value.code == ErrorCode.HOST_NOT_ADMITTING
    assert "the last save is 35 minutes old" in str(raised.value)
    status = persistent_hosts.persistent_status(host.store.get_host(HOST_ID), online=True)
    assert status is not None and status["admission"]["reason"] == "not_saved"


async def test_new_work_for_a_live_runner_is_refused_while_admitted_work_continues(
    host: FakeHost,
) -> None:
    """Admission closes for new work only: an admitted operation keeps its handle."""
    admitted = asyncio.Event()
    finish = asyncio.Event()

    async def running_turn() -> None:
        await persistent_hosts.ensure_host_ready(HOST_ID, operation="message", timeout_s=5)
        admitted.set()
        await finish.wait()
        # Still part of the admitted operation: no refusal for its own steps.
        await persistent_hosts.admit_host_work(HOST_ID, "runner_launch")

    turn = asyncio.create_task(running_turn())
    await admitted.wait()
    host.computer.admitting = False
    with pytest.raises(OmnigentError) as raised:
        await persistent_hosts.ensure_host_ready(HOST_ID, operation="message", timeout_s=5)
    assert raised.value.code == ErrorCode.HOST_NOT_ADMITTING
    finish.set()
    await asyncio.wait_for(turn, 5)


async def test_needs_sign_in_names_the_setup_page(db_uri: str) -> None:
    """A "needs sign-in" refusal says so and links the provider's setup page."""
    store, registry, computer = HostStore(db_uri), HostRegistry(), FakeComputer()
    computer.admitting = False
    computer.refusal = ("needs_sign_in", "Claude is signed out")
    persistent_hosts._states.clear()
    persistent_hosts.configure(
        store,
        registry,
        _deployment(computer, manage_url="https://computers.test/c/{sandbox_id}"),
    )
    try:
        register_persistent_row(store)
        fake = FakeHost(store, registry, computer)
        assert fake.connect(FIRST_TOKEN)
        with pytest.raises(OmnigentError) as raised:
            await persistent_hosts.ensure_host_ready(HOST_ID, operation="message", timeout_s=5)
        assert raised.value.code == ErrorCode.HOST_NOT_ADMITTING
        message = str(raised.value)
        assert "sign in" in message and f"https://computers.test/c/{SANDBOX_ID}" in message
        assert "Claude is signed out" in message
        fake.close()
    finally:
        persistent_hosts.configure(None, None, None)
        persistent_hosts._states.clear()


async def test_cancelled_waiter_does_not_stop_a_shared_wake(host: FakeHost) -> None:
    """Cancelling one request waiting on a wake leaves the wake for the others."""
    await _asleep(host)
    first = asyncio.create_task(_ready("message")())
    second = asyncio.create_task(_ready("message")())
    await asyncio.sleep(0.05)
    first.cancel()
    await asyncio.wait_for(second, 30)
    assert host.computer.events.count("resume") == 1
    assert host.registry.get(HOST_ID) is not None


async def test_concurrent_triggers_cause_one_physical_wake(host: FakeHost) -> None:
    """Two conversations and a scheduled run share one resume, one claim and one start."""
    await _asleep(host)
    requests = [
        asyncio.create_task(_ready(operation)())
        for operation in ("message", "message", "scheduled_fire")
    ]
    await asyncio.wait_for(asyncio.gather(*requests), 30)
    assert host.computer.events == ["resume", "claim", "start_host"]


async def test_expired_wait_admits_nothing(host: FakeHost) -> None:
    """A request whose deadline passes during a sleep is refused and never admitted."""
    host.computer.suspend_gate = threading.Event()
    fence = asyncio.create_task(persistent_hosts.sleep_if_idle(HOST_ID))
    await asyncio.to_thread(host.computer.suspend_accepted.wait, 10)
    with pytest.raises(OmnigentError) as raised:
        await persistent_hosts.ensure_host_ready(HOST_ID, operation="message", timeout_s=0.5)
    assert raised.value.code == ErrorCode.RUNNER_UNAVAILABLE
    assert persistent_hosts._states[HOST_ID].admitted() == []
    host.computer.suspend_gate.set()
    assert await fence == "stopped"


async def test_deleted_or_reowned_host_is_refused_after_the_wait(host: FakeHost) -> None:
    """Authorization is re-checked after a wait: no launch on a removed host."""
    host.computer.suspend_gate = threading.Event()
    fence = asyncio.create_task(persistent_hosts.sleep_if_idle(HOST_ID))
    await asyncio.to_thread(host.computer.suspend_accepted.wait, 10)
    request = asyncio.create_task(_ready("message")())
    await asyncio.sleep(0.2)
    host.store.delete_host(HOST_ID)
    host.computer.suspend_gate.set()
    await fence
    with pytest.raises(OmnigentError) as raised:
        await asyncio.wait_for(request, 10)
    assert raised.value.code == ErrorCode.NOT_FOUND
    assert "start_host" not in host.computer.events


async def test_wrong_owner_is_refused_before_any_wake(host: FakeHost) -> None:
    """A caller that does not own the host never wakes it."""
    await _asleep(host)
    with pytest.raises(OmnigentError) as raised:
        await persistent_hosts.ensure_host_ready(
            HOST_ID, operation="message", timeout_s=5, owner="mallory@example.com"
        )
    assert raised.value.code == ErrorCode.FORBIDDEN
    assert host.computer.events == []


@pytest.mark.parametrize("written", [True, False])
async def test_cancel_after_the_intent_commits_before_the_await_returns(
    host: FakeHost, monkeypatch: pytest.MonkeyPatch, written: bool
) -> None:
    """A fence cancelled around its intent write learns what was written and reconciles.

    Written: the sleep completes and the next request wakes the computer with a
    new token, without a server restart. Not written: admission reopens and the
    running host keeps its working token.
    """
    original = host.store.begin_host_sleep
    committed, release = threading.Event(), threading.Event()

    def commit_then_park(host_id: str, *, operation_id: str) -> bool:
        result = original(host_id, operation_id=operation_id) if written else False
        committed.set()
        release.wait(10)
        return result

    monkeypatch.setattr(host.store, "begin_host_sleep", commit_then_park)
    fence = asyncio.create_task(persistent_hosts.sleep_if_idle(HOST_ID))
    await asyncio.to_thread(committed.wait, 10)
    fence.cancel()
    with pytest.raises(asyncio.CancelledError):
        await fence
    state = persistent_hosts._states[HOST_ID]
    assert not state.admitting.is_set() and state.unknown_suspend_operation is not None
    request = asyncio.create_task(_admitted_after(host, _ready("message")))
    await asyncio.sleep(0.3)
    assert not request.done()
    release.set()
    assert await asyncio.wait_for(request, 30) == "admitted"
    if written:
        assert host.computer.events == [
            "suspend",
            "stopped",
            "resume",
            "claim",
            "start_host",
            "admitted",
        ]
        assert host.store.resolve_launch_token(HOST_ID, FIRST_TOKEN) is None
    else:
        assert host.computer.events == ["admitted"]
        assert host.store.resolve_launch_token(HOST_ID, FIRST_TOKEN) is not None
    current = host.store.get_host(HOST_ID)
    assert current is not None and current.sleep_operation_id is None
    assert host.registry.get(HOST_ID) is not None


@pytest.mark.parametrize("parent_end", ["completes", "cancelled"])
async def test_child_task_keeps_the_operation_admitted_after_its_parent(
    host: FakeHost, monkeypatch: pytest.MonkeyPatch, parent_end: str
) -> None:
    """A child of an admitted request holds its own handle; the parent ending does not free it."""
    monkeypatch.setattr(persistent_hosts, "FENCE_DRAIN_S", 0.5)
    child_release = asyncio.Event()
    child_started = asyncio.Event()

    async def child() -> None:
        await persistent_hosts.ensure_host_ready(HOST_ID, operation="child", timeout_s=5)
        child_started.set()
        await child_release.wait()

    children: list[asyncio.Task[None]] = []

    async def parent() -> None:
        await persistent_hosts.ensure_host_ready(HOST_ID, operation="message", timeout_s=5)
        children.append(asyncio.create_task(child()))
        await child_started.wait()
        if parent_end == "cancelled":
            await asyncio.Event().wait()

    parent_task = asyncio.create_task(parent())
    await child_started.wait()
    if parent_end == "cancelled":
        parent_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await parent_task
    assert persistent_hosts._states[HOST_ID].admitted() == ["child"]
    assert await persistent_hosts.sleep_if_idle(HOST_ID) == "abandoned"
    assert "suspend" not in host.computer.events
    child_release.set()
    await children[0]
    assert await persistent_hosts.sleep_if_idle(HOST_ID) == "stopped"


def _request_slots() -> list[str]:
    """Every per-connection map of server-to-host requests awaiting a reply."""
    return [
        item.name
        for item in dataclasses.fields(HostConnection)
        if item.name.startswith(("pending_", "inflight_")) and item.name != "pending_quiesce"
    ]


@pytest.mark.parametrize("slot", _request_slots())
async def test_every_direct_host_request_in_flight_holds_the_sleep(
    host: FakeHost, monkeypatch: pytest.MonkeyPatch, slot: str
) -> None:
    """A direct host request awaiting its reply, of any kind, keeps the sandbox running."""
    monkeypatch.setattr(persistent_hosts, "FENCE_DRAIN_S", 0.3)
    conn = host.registry.get(HOST_ID)
    assert conn is not None
    getattr(conn, slot)["request-1"] = asyncio.get_running_loop().create_future()
    assert await persistent_hosts.sleep_if_idle(HOST_ID) == "abandoned"
    assert "suspend" not in host.computer.events
    getattr(conn, slot).clear()
    assert await persistent_hosts.sleep_if_idle(HOST_ID) == "stopped"


async def test_requests_on_a_superseded_connection_hold_the_sleep(
    host: FakeHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A request still awaiting its reply on a replaced tunnel keeps the sandbox running."""
    monkeypatch.setattr(persistent_hosts, "FENCE_DRAIN_S", 0.3)
    old = host.registry.get(HOST_ID)
    assert old is not None
    host.registry.send_text(old, '{"kind": "host.stat", "request_id": "r1", "path": "~"}')
    old.pending_launches["launch-1"] = asyncio.get_running_loop().create_future()
    assert host.connect()  # the host reconnects; the old connection is superseded
    assert host.registry.get(HOST_ID) is not old
    assert await persistent_hosts.sleep_if_idle(HOST_ID) == "abandoned"
    # The launch's waiter was answered at once, but the host may still be
    # launching: it keeps counting until its hold expires.
    assert old.pending_launches == {}
    assert "launch-1" in old.inflight_unanswered
    old.inflight_unanswered.clear()
    assert await persistent_hosts.sleep_if_idle(HOST_ID) == "stopped"


@pytest.mark.parametrize(
    ("runners", "operations"),
    [(["runner-1"], []), ([], ["install_harness"]), ([], ["create_worktree", "store_secret"])],
)
async def test_quiesce_reporting_live_work_gives_up_the_sleep(
    host: FakeHost, runners: list[str], operations: list[str]
) -> None:
    """A live runner, or a host operation whose request already ended, holds the sleep."""
    host.runners.extend(runners)
    host.operations.extend(operations)
    assert await persistent_hosts.sleep_if_idle(HOST_ID) == "abandoned"
    assert "suspend" not in host.computer.events
    assert persistent_hosts._states[HOST_ID].admitting.is_set()


async def test_older_host_without_quiesce_never_sleeps(host: FakeHost) -> None:
    """A host that cannot answer ``host.quiesce`` makes every sleep give up."""
    host.quiesce = False
    assert host.connect()
    assert await persistent_hosts.sleep_if_idle(HOST_ID) == "abandoned"
    assert "suspend" not in host.computer.events


async def test_direct_host_requests_are_refused_while_the_fence_is_closed(
    host: FakeHost,
) -> None:
    """A new direct host request during a sleep never reaches the host being stopped."""
    host.computer.suspend_gate = threading.Event()
    fence = asyncio.create_task(persistent_hosts.sleep_if_idle(HOST_ID))
    await asyncio.to_thread(host.computer.suspend_accepted.wait, 10)
    conn = host.registry.get(HOST_ID)
    assert conn is not None
    queued = conn.outbound_queue.qsize()
    with pytest.raises(ConnectionError, match="going to sleep"):
        host.registry.send_text(conn, '{"kind": "host.stat", "request_id": "r1", "path": "~"}')
    assert conn.outbound_queue.qsize() == queued
    host.computer.suspend_gate.set()
    assert await fence == "stopped"


async def test_lost_suspend_reply_keeps_admission_closed_until_the_same_operation_reports(
    host: FakeHost,
) -> None:
    """An unknown sleep is repeated with its operation id; work waits, then wakes the host."""
    host.computer.lose_suspend_replies = 2
    assert await persistent_hosts.sleep_if_idle(HOST_ID) == "unknown"
    state = persistent_hosts._states[HOST_ID]
    assert not state.admitting.is_set()
    request = asyncio.create_task(_admitted_after(host, _ready("message")))
    assert await asyncio.wait_for(request, 30) == "admitted"
    operations = host.computer.suspend_operations
    assert len(operations) >= 3 and set(operations) == {operations[0]}
    assert host.computer.events.count("suspend") == 1
    assert host.computer.events[-1] == "admitted"


class _FlakyStore:
    """Wraps one store method so it fails, blocks or succeeds on demand."""

    def __init__(self, store: HostStore, name: str, monkeypatch: pytest.MonkeyPatch) -> None:
        self.real = getattr(store, name)
        self.failures = 0
        self.gate: threading.Event | None = None
        self.entered = threading.Event()
        monkeypatch.setattr(store, name, self)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.entered.set()
        if self.gate is not None:
            self.gate.wait(10)
        if self.failures > 0:
            self.failures -= 1
            raise RuntimeError("database unavailable")
        return self.real(*args, **kwargs)


async def _unknown_sleep(host: FakeHost) -> str:
    """A sleep whose stop was accepted but whose reply was lost."""
    host.computer.lose_suspend_replies = 1
    assert await persistent_hosts.sleep_if_idle(HOST_ID) == "unknown"
    operation = persistent_hosts._states[HOST_ID].unknown_suspend_operation
    assert operation is not None
    return operation


def _still_unresolved(operation: str) -> None:
    state = persistent_hosts._states[HOST_ID]
    assert not state.admitting.is_set()
    assert state.unknown_suspend_operation == operation


@pytest.mark.parametrize("fault", ["failed", "blocked", "cancelled"])
async def test_unknown_sleep_stays_closed_while_its_intent_cannot_be_read(
    host: FakeHost, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    """A failed, blocked or cancelled read of the sleep intent keeps admission closed.

    Nothing is admitted, the operation is kept, the provider is never asked to
    stop twice, and once the database answers again the sleep resolves and the
    next request wakes the computer.
    """
    operation = await _unknown_sleep(host)
    reads = _FlakyStore(host.store, "get_host", monkeypatch)
    if fault == "failed":
        reads.failures = 1
        with pytest.raises(RuntimeError):
            await persistent_hosts.sleep_if_idle(HOST_ID, only_operation=operation)
    else:
        reads.gate = threading.Event()
        recovery = asyncio.create_task(
            persistent_hosts.sleep_if_idle(HOST_ID, only_operation=operation)
        )
        assert await asyncio.to_thread(reads.entered.wait, 10)
        if fault == "cancelled":
            recovery.cancel()
            with pytest.raises(asyncio.CancelledError):
                await recovery
            reads.gate.set()
        else:
            # Work waiting meanwhile is refused at its deadline, never admitted.
            with pytest.raises(OmnigentError):
                await asyncio.create_task(_ready("message", timeout_s=0.3)())
            reads.gate.set()
            assert await asyncio.wait_for(recovery, 10) == "stopped"
    if fault != "blocked":
        _still_unresolved(operation)
        reads.gate = None
        assert await persistent_hosts.sleep_if_idle(HOST_ID, only_operation=operation) == "stopped"
    assert persistent_hosts._states[HOST_ID].admitting.is_set()
    assert host.computer.events.count("suspend") == 1
    assert set(host.computer.suspend_operations) == {operation}
    await asyncio.create_task(_admitted_after(host, _ready("message")))
    assert host.computer.events[-4:] == ["resume", "claim", "start_host", "admitted"]


@pytest.mark.parametrize("acknowledged", ["committed", "not_committed"])
async def test_ambiguous_intent_write_is_settled_from_the_store(
    host: FakeHost, monkeypatch: pytest.MonkeyPatch, acknowledged: str
) -> None:
    """An intent write that raised is reconciled by reading what the store holds.

    Committed: the sleep goes ahead once, with that operation. Not committed:
    nothing stopped, admission reopens and the token still works.
    """
    real = host.store.begin_host_sleep

    def write_then_fail(host_id: str, *, operation_id: str) -> bool:
        if acknowledged == "committed":
            real(host_id, operation_id=operation_id)
        raise RuntimeError("connection reset after the write")

    monkeypatch.setattr(host.store, "begin_host_sleep", write_then_fail)
    outcome = await persistent_hosts.sleep_if_idle(HOST_ID)
    state = persistent_hosts._states[HOST_ID]
    if acknowledged == "committed":
        assert outcome == "stopped"
        assert host.computer.events == ["suspend", "stopped"]
    else:
        assert outcome == "abandoned"
        assert host.computer.events == []
        assert host.store.resolve_launch_token(HOST_ID, FIRST_TOKEN) is not None
    assert state.admitting.is_set() and state.unknown_suspend_operation is None


async def test_restart_with_an_unreadable_store_keeps_the_sleep_closed(
    host: FakeHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After a restart, recovery that cannot read the store yet is retried, not abandoned."""
    operation = await _unknown_sleep(host)
    persistent_hosts._states.clear()
    host.registry.deregister(HOST_ID)
    assert await persistent_hosts.recover_pending_sleeps() == [HOST_ID]
    reads = _FlakyStore(host.store, "get_host", monkeypatch)
    reads.failures = 1
    outcomes = await persistent_hosts.PersistentHostIdleMonitor().sweep_once()
    assert outcomes[HOST_ID] == "unknown"
    _still_unresolved(operation)
    outcomes = await persistent_hosts.PersistentHostIdleMonitor().sweep_once()
    assert outcomes[HOST_ID] == "stopped"
    assert persistent_hosts._states[HOST_ID].admitting.is_set()
    assert host.computer.events.count("suspend") == 1


async def test_confirmed_sleep_revokes_the_token_and_a_stale_token_stays_refused(
    host: FakeHost,
) -> None:
    """After a confirmed stop the old token never registers, before or after the next wake."""
    await _asleep(host)
    assert host.store.resolve_launch_token(HOST_ID, FIRST_TOKEN) is None
    assert not host.connect(FIRST_TOKEN)
    await persistent_hosts.ensure_host_ready(HOST_ID, operation="message", timeout_s=10)
    assert host.token is not None and host.token != FIRST_TOKEN
    assert not host.connect(FIRST_TOKEN)


# ── Restarts fenced by start claims (R-02, G1BR-005, RC-54) ────────────


async def test_host_only_drop_never_rearms_a_live_host_process(host: FakeHost) -> None:
    """A dropped tunnel with a live host process reconnects with its current token.

    Its runners, their progress and pending approvals live in that process, so
    a wake request must never claim, re-arm or restart it.
    """
    host.runners.append("runner-1")
    host.drop()
    request = asyncio.create_task(_admitted_after(host, _ready("message")))
    await asyncio.sleep(0.3)
    assert not request.done()
    assert host.connect()  # the host process reconnects on its own
    assert await asyncio.wait_for(request, 10) == "admitted"
    assert host.computer.events == ["admitted"]
    assert host.token == FIRST_TOKEN
    assert host.store.resolve_launch_token(HOST_ID, FIRST_TOKEN) is not None


@pytest.mark.parametrize("parked_status", ["started_no_host", "host_refused"])
async def test_parked_computer_is_restarted_through_a_granted_claim(
    host: FakeHost, parked_status: str
) -> None:
    """A server killed between resume and start, or a host that exited "token refused".

    The next request claims a start attempt, re-arms and starts the host
    without a manual restart (RC-54).
    """
    host.drop()
    host.computer.status = parked_status  # type: ignore[assignment]
    await asyncio.wait_for(_admitted_after(host, _ready("message")), 10)
    assert host.computer.events == ["claim", "start_host", "admitted"]
    assert host.computer.accepted_starts == [(1, host.token)]
    assert host.token != FIRST_TOKEN
    assert host.store.resolve_launch_token(HOST_ID, FIRST_TOKEN) is None


async def test_database_restored_to_before_a_wake_recovers_through_a_claim(
    host: FakeHost,
) -> None:
    """A restored database holds an older token than the live host.

    The host's reconnect is refused; after its refusals last long enough it
    exits "token refused" and its supervisor parks it (``host_refused``). The
    next request then claims, re-arms and starts it with a new token.
    """
    live_token = host.token
    current = host.store.get_host(HOST_ID)
    assert current is not None
    # The restore brings back the row as it was before the last wake.
    host.store.rearm_managed_host(
        HOST_ID,
        sandbox_id=SANDBOX_ID,
        expected_updated_at=current.updated_at,
        token="token-before-the-wake",
        token_expires_at=current.token_expires_at or 0,
    )
    host.drop()
    assert live_token is not None and not host.connect(live_token)
    host.computer.status = "host_refused"  # the host exited; its supervisor reports it
    await asyncio.wait_for(_admitted_after(host, _ready("message")), 10)
    assert host.computer.events == ["claim", "start_host", "admitted"]
    assert host.token not in (live_token, "token-before-the-wake")


async def test_drop_around_an_admitted_launch_neither_rearms_nor_sleeps(
    host: FakeHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A host-only drop while admitted work is in flight keeps the host's process and token.

    The work's own follow-up step waits for the host to come back instead of
    waking a second one, and no sleep runs while the operation is admitted.
    """
    monkeypatch.setattr(persistent_hosts, "FENCE_DRAIN_S", 0.3)
    admitted, continue_launch = asyncio.Event(), asyncio.Event()

    async def message_then_launch() -> None:
        await persistent_hosts.ensure_host_ready(HOST_ID, operation="message", timeout_s=5)
        admitted.set()
        await continue_launch.wait()
        await persistent_hosts.admit_host_work(HOST_ID, "runner_launch")

    turn = asyncio.create_task(message_then_launch())
    await asyncio.wait_for(admitted.wait(), 5)
    host.drop()
    assert await persistent_hosts.sleep_if_idle(HOST_ID) == "abandoned"
    continue_launch.set()
    await asyncio.wait_for(turn, 5)  # still part of the admitted operation
    assert host.connect()
    assert host.computer.events == []
    assert host.token == FIRST_TOKEN


async def test_status_read_then_supervisor_restart_never_rearms(host: FakeHost) -> None:
    """A status read, then the supervisor taking over, then the continued wake.

    The status said ``started_no_host``; by the time Omnigent claims, the
    supervisor has restarted the host process. The claim is refused, nothing is
    re-armed, and the host reconnects with its own token.
    """
    host.drop()
    host.computer.status = "started_no_host"

    def supervisor_restarts_host() -> None:
        host.computer.status = "host_running"
        call_soon(loop, host.connect, FIRST_TOKEN)

    loop = asyncio.get_running_loop()
    host.computer.before_claim = supervisor_restarts_host
    await asyncio.wait_for(_admitted_after(host, _ready("message")), 10)
    assert host.computer.events == ["claim", "admitted"]
    assert host.computer.accepted_starts == []
    assert host.token == FIRST_TOKEN
    assert host.store.resolve_launch_token(HOST_ID, FIRST_TOKEN) is not None


@pytest.mark.parametrize("old_start", ["before_new_claim", "after_new_claim"])
async def test_old_start_delayed_across_a_server_restart_and_a_new_claim(
    host: FakeHost, old_start: str
) -> None:
    """A dead server's start for attempt 1 arrives before or after the new server's claim.

    Before: the platform accepts it, the host runs, and the new server only
    waits for it. After: attempt 1 is no longer current and is refused. Either
    way exactly one start runs, with the token the database holds.
    """
    host.drop()
    # The old server claimed attempt 1, armed a token, then died before starting.
    host.computer.status = "started_no_host"
    old_claim = host.computer.claim_start(SANDBOX_ID, operation_id="old-server")
    current = host.store.get_host(HOST_ID)
    assert current is not None and old_claim.attempt == 1
    host.store.rearm_managed_host(
        HOST_ID,
        sandbox_id=SANDBOX_ID,
        expected_updated_at=current.updated_at,
        token="old-server-token",
        token_expires_at=current.token_expires_at or 0,
    )
    host.computer.events.clear()
    if old_start == "before_new_claim":
        assert host.computer.deliver_start(1, "old-server-token")
        await asyncio.wait_for(_ready("message")(), 10)
        assert host.computer.events == ["start_host"]
    else:
        await asyncio.wait_for(_ready("message")(), 10)
        assert not host.computer.deliver_start(1, "old-server-token")
        assert host.computer.refused_starts == [1]
    assert len(host.computer.accepted_starts) == 1
    (_, started_token) = host.computer.accepted_starts[0]
    assert host.store.resolve_launch_token(HOST_ID, started_token) is not None
    assert host.token == started_token


async def test_cached_status_never_authorizes_a_rearm(
    host: FakeHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stale report of ``started_no_host`` leads only to a claim, which is refused."""
    host.drop()
    stale = AdmissionReport(admitting=True, status="started_no_host")
    monkeypatch.setattr(
        persistent_hosts, "_fresh_report", _first_then_real(stale, persistent_hosts._fresh_report)
    )
    request = asyncio.create_task(_ready("message")())
    await asyncio.sleep(0.2)
    assert host.connect()  # the live host reconnects meanwhile
    await asyncio.wait_for(request, 10)
    assert "start_host" not in host.computer.events
    assert host.token == FIRST_TOKEN


def _first_then_real(first: AdmissionReport, real: Callable[..., Any]) -> Callable[..., Any]:
    calls = {"n": 0}

    async def report(*args: Any, **kwargs: Any) -> AdmissionReport | None:
        calls["n"] += 1
        return first if calls["n"] == 1 else await real(*args, **kwargs)

    return report


async def test_lost_claim_reply_is_repeated_with_the_same_operation(host: FakeHost) -> None:
    """A claim whose reply was lost returns its recorded grant; one start follows."""
    host.drop()
    host.computer.status = "started_no_host"
    host.computer.lose_claim_replies = 2
    await asyncio.wait_for(_ready("message")(), 10)
    assert host.computer.events == ["claim", "start_host"]
    assert len(host.computer.claims) == 1


async def test_lost_start_reply_after_delivery_never_starts_twice(host: FakeHost) -> None:
    """A start that reached the supervisor but whose reply was lost: wait, never re-claim."""
    host.drop()
    host.computer.status = "started_no_host"
    host.computer.lose_start_reply = True
    await asyncio.wait_for(_ready("message")(), 10)
    assert host.computer.events == ["claim", "start_host"]


async def test_undelivered_start_is_superseded_by_the_next_claim(
    host: FakeHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A start that never arrived leaves ``start_claimed``; the next claim supersedes it."""
    monkeypatch.setattr(persistent_hosts, "START_REDELIVERY_S", 0.2)
    host.drop()
    host.computer.status = "started_no_host"
    host.computer.drop_start = True
    await asyncio.wait_for(_ready("message")(), 10)
    assert host.computer.events == ["claim", "claim", "start_host"]
    assert host.computer.accepted_starts == [(2, host.token)]


# ── Admission reports (R-13) ───────────────────────────────────────────


async def test_failed_admission_refresh_is_retried_and_never_refuses(host: FakeHost) -> None:
    """An unanswered report is "unknown": the work waits and is admitted once it answers."""
    host.computer.unanswered_reports = 3
    await asyncio.wait_for(_ready("message")(), 10)
    assert host.computer.unanswered_reports == 0
    assert persistent_hosts._states[HOST_ID].report is not None


async def test_outage_fails_visibly_as_status_unknown(host: FakeHost) -> None:
    """A platform that stays unreachable fails the work at its deadline, not as a refusal."""
    host.computer.unanswered_reports = 10_000
    with pytest.raises(OmnigentError) as raised:
        await persistent_hosts.ensure_host_ready(HOST_ID, operation="message", timeout_s=0.5)
    assert raised.value.code == ErrorCode.RUNNER_UNAVAILABLE
    assert "computer status unknown" in str(raised.value)
    assert persistent_hosts._states[HOST_ID].report is None


async def test_report_age_counts_from_the_provider_reply(
    host: FakeHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One cache of at most the policy's age, measured from the provider's own reply."""
    policy = dataclasses.replace(
        persistent_hosts.policy_for(host.store.get_host(HOST_ID)), admission_max_age_s=60
    )
    monkeypatch.setattr(persistent_hosts, "policy_for", lambda _host: policy)
    calls = {"n": 0}
    real = host.computer.admission

    def old_reply(sandbox_id: str) -> AdmissionReport | None:
        calls["n"] += 1
        report = real(sandbox_id)
        assert report is not None
        # Answered from a platform reply that is already 59 s old.
        return dataclasses.replace(report, observed_at=persistent_hosts.time.time() - 59)

    monkeypatch.setattr(host.computer, "admission", old_reply)
    # Separate requests: a task already holding an admission is not re-checked.
    await asyncio.create_task(_ready("message")())
    await asyncio.create_task(_ready("message")())
    assert calls["n"] == 1  # 59 s old: still within the age
    await asyncio.sleep(1.1)
    await asyncio.create_task(_ready("message")())
    assert calls["n"] == 2


async def test_admission_answer_after_the_deadline_admits_nothing(
    host: FakeHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A slow provider answer that arrives after the request's deadline admits nothing."""
    real = host.computer.admission

    def slow_admission(sandbox_id: str) -> AdmissionReport | None:
        threading.Event().wait(0.5)
        return real(sandbox_id)

    monkeypatch.setattr(host.computer, "admission", slow_admission)
    with pytest.raises(OmnigentError) as raised:
        await asyncio.create_task(_ready("message", timeout_s=0.1)())
    assert raised.value.code == ErrorCode.RUNNER_UNAVAILABLE
    assert persistent_hosts._states[HOST_ID].admitted() == []


@pytest.mark.parametrize("admitting", [True, False])
async def test_report_already_too_old_on_arrival_is_unknown(
    host: FakeHost, monkeypatch: pytest.MonkeyPatch, admitting: bool
) -> None:
    """A launcher answering from a stale cached reply neither admits nor refuses."""
    policy = dataclasses.replace(
        persistent_hosts.policy_for(host.store.get_host(HOST_ID)), admission_max_age_s=60
    )
    monkeypatch.setattr(persistent_hosts, "policy_for", lambda _host: policy)
    host.computer.admitting = admitting
    real = host.computer.admission

    def stale_reply(sandbox_id: str) -> AdmissionReport | None:
        report = real(sandbox_id)
        assert report is not None
        return dataclasses.replace(report, observed_at=persistent_hosts.time.time() - 600)

    monkeypatch.setattr(host.computer, "admission", stale_reply)
    with pytest.raises(OmnigentError) as raised:
        await asyncio.create_task(_ready("message", timeout_s=0.3)())
    assert raised.value.code == ErrorCode.RUNNER_UNAVAILABLE
    assert "computer status unknown" in str(raised.value)
    assert persistent_hosts._states[HOST_ID].report is None


async def test_reply_from_before_a_sleep_never_admits_into_the_next_generation(
    host: FakeHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A positive report asked for before a sleep and wake cannot admit work afterwards."""
    asked, release = threading.Event(), threading.Event()
    real = host.computer.admission
    calls = {"n": 0}

    def delayed_first_reply(sandbox_id: str) -> AdmissionReport | None:
        calls["n"] += 1
        report = real(sandbox_id)
        if calls["n"] == 1:
            asked.set()
            release.wait(10)
        return report

    monkeypatch.setattr(host.computer, "admission", delayed_first_reply)
    request = asyncio.create_task(_ready("message", timeout_s=10)())
    assert await asyncio.to_thread(asked.wait, 10)
    assert await persistent_hosts.sleep_if_idle(HOST_ID) == "stopped"
    await persistent_hosts.start_host_now(HOST_ID)
    host.computer.admitting = False  # the new generation refuses new work
    release.set()
    with pytest.raises(OmnigentError) as raised:
        await asyncio.wait_for(request, 10)
    assert raised.value.code == ErrorCode.HOST_NOT_ADMITTING


async def test_host_deleted_while_admission_is_asked_admits_nothing(
    host: FakeHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Deleting the host while its admission report is pending refuses the waiting work."""
    gate = host.computer.admission_gate = threading.Event()
    request = asyncio.create_task(_ready("message", timeout_s=10)())
    assert await asyncio.to_thread(host.computer.parked.wait, 10)
    current = host.store.get_host(HOST_ID)
    assert current is not None
    await persistent_hosts.delete_host(current)
    gate.set()
    with pytest.raises(OmnigentError) as raised:
        await asyncio.wait_for(request, 10)
    assert raised.value.code == ErrorCode.NOT_FOUND


async def test_waiting_on_an_unknown_sleep_respects_the_deadline(
    host: FakeHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A slow repeat of an unknown sleep never holds work past its own deadline."""
    host.computer.suspend_gate = threading.Event()
    fence = asyncio.create_task(persistent_hosts.sleep_if_idle(HOST_ID))
    await asyncio.to_thread(host.computer.suspend_accepted.wait, 10)
    fence.cancel()
    with pytest.raises(asyncio.CancelledError):
        await fence
    started = persistent_hosts.time.monotonic()
    with pytest.raises(OmnigentError) as raised:
        await persistent_hosts.ensure_host_ready(HOST_ID, operation="message", timeout_s=0.3)
    assert raised.value.code == ErrorCode.RUNNER_UNAVAILABLE
    assert persistent_hosts.time.monotonic() - started < 2.0
    state = persistent_hosts._states[HOST_ID]
    operation = state.unknown_suspend_operation
    assert operation is not None and state.recovery_task is not None
    host.computer.suspend_gate.set()
    outcome = "unknown"
    for _ in range(50):
        outcome = await persistent_hosts.sleep_if_idle(HOST_ID, only_operation=operation)
        if outcome != "unknown":
            break
        await asyncio.sleep(0.05)
    assert outcome == "stopped"


async def test_start_again_waits_for_an_unknown_sleep_to_resolve(host: FakeHost) -> None:
    """An explicit start never resumes or re-arms before a pending sleep reports."""
    host.computer.suspend_gate = threading.Event()
    fence = asyncio.create_task(persistent_hosts.sleep_if_idle(HOST_ID))
    await asyncio.to_thread(host.computer.suspend_accepted.wait, 10)
    fence.cancel()
    with pytest.raises(asyncio.CancelledError):
        await fence
    start = persistent_hosts.start_host_now(HOST_ID)
    await asyncio.sleep(0.3)
    assert host.computer.events == ["suspend"]
    host.computer.suspend_gate.set()
    await asyncio.wait_for(start, 10)
    assert host.computer.events == ["suspend", "stopped", "resume", "claim", "start_host"]
    assert host.registry.get(HOST_ID) is not None


# ── Providers without a typed status ───────────────────────────────────


@pytest.fixture
async def untyped(db_uri: str) -> AsyncIterator[tuple[FakeHost, FakeUntypedComputer]]:
    """A persistent host on a provider that only answers ``is_running``."""
    store, registry = HostStore(db_uri), HostRegistry()
    computer = FakeUntypedComputer()
    fake = FakeHost(store, registry, FakeComputer())
    loop = asyncio.get_running_loop()
    computer.on_host_start = lambda token: call_soon(loop, fake.connect, token)
    persistent_hosts._states.clear()
    persistent_hosts.configure(
        store,
        registry,
        ManagedSandboxDeployment.single(
            ManagedSandboxConfig(
                server_url="http://omnigent.test",
                launcher_factory=lambda: computer,
                token_ttl_s=3600,
                provider=computer.provider,
                persistent=PersistentHostPolicy(),
                online_timeout_s=2,
            )
        ),
    )
    yield fake, computer
    fake.close()
    persistent_hosts.configure(None, None, None)
    persistent_hosts._states.clear()


async def test_untyped_unknown_status_is_never_taken_as_stopped(
    untyped: tuple[FakeHost, FakeUntypedComputer],
) -> None:
    """``is_running() is None`` holds the wake; nothing is resumed or re-armed."""
    fake, computer = untyped
    register_persistent_row(fake.store)
    computer.running = None
    with pytest.raises(OmnigentError) as raised:
        await persistent_hosts.ensure_host_ready(HOST_ID, operation="message", timeout_s=5)
    assert "computer status unknown" in str(raised.value)
    assert computer.events == []
    assert fake.store.resolve_launch_token(HOST_ID, FIRST_TOKEN) is not None


async def test_untyped_running_host_is_repaired_and_a_first_start_is_started(
    untyped: tuple[FakeHost, FakeUntypedComputer],
) -> None:
    """A running sandbox with an armed host is only waited for; a fresh one is started."""
    fake, computer = untyped
    created = await persistent_hosts.create_host(owner=OWNER, name="Computer", provider=None)
    fake.host_id = created.host_id
    await asyncio.wait_for(persistent_hosts.start_host_now(created.host_id), 5)
    assert computer.events == ["provision", "start_host"]
    computer.events.clear()
    first_token = fake.token
    fake.drop()
    request = asyncio.create_task(
        persistent_hosts.ensure_host_ready(created.host_id, operation="message", timeout_s=5)
    )
    await asyncio.sleep(0.2)
    assert fake.connect()  # the live host reconnects on its own
    await asyncio.wait_for(request, 5)
    assert computer.events == []
    assert fake.token == first_token


# ── Lifecycle: create, failure, removal ────────────────────────────────


@pytest.fixture
async def fresh(db_uri: str) -> AsyncIterator[FakeHost]:
    """A configured persistent provider with no host yet."""
    store = HostStore(db_uri)
    registry = HostRegistry()
    registry.admission_gate = persistent_hosts.admit_host_work
    registry.send_guard = persistent_hosts.guard_host_request
    computer = FakeComputer(status="stopped")
    fake = FakeHost(store, registry, computer)
    loop = asyncio.get_running_loop()
    computer.on_host_start = lambda token: call_soon(loop, fake.connect, token)
    persistent_hosts._states.clear()
    persistent_hosts.configure(store, registry, _deployment(computer))
    yield fake
    fake.close()
    persistent_hosts.configure(None, None, None)
    persistent_hosts._states.clear()


async def _created(fake: FakeHost, name: str = "Work computer") -> str:
    created = await persistent_hosts.create_host(owner=OWNER, name=name, provider=None)
    fake.host_id = created.host_id
    task = persistent_hosts._states[created.host_id].wake_task
    assert task is not None
    with contextlib.suppress(OmnigentError):
        await asyncio.wait_for(task, 10)
    return created.host_id


async def test_create_writes_the_pending_row_before_provision(fresh: FakeHost) -> None:
    """The host row exists, persistent and without a sandbox, before the provider acts."""
    seen: list[Any] = []

    def provision_sees_row() -> None:
        seen.extend(fresh.store.list_hosts(OWNER))

    fresh.computer.on_provision = provision_sees_row
    host_id = await _created(fresh)
    assert [(row.host_id, row.lifecycle, row.sandbox_id) for row in seen] == [
        (host_id, "persistent", None)
    ]
    created = fresh.store.get_host(host_id)
    assert created is not None and created.sandbox_id == SANDBOX_ID
    assert fresh.computer.events == ["provision", "resume", "claim", "start_host"]
    status = persistent_hosts.persistent_status(created, online=True)
    assert status is not None and status["state"] == "awake"


async def test_duplicate_name_is_a_clean_conflict(fresh: FakeHost) -> None:
    """A second host with the same display name is refused before anything is provisioned."""
    await _created(fresh)
    fresh.computer.events.clear()
    with pytest.raises(OmnigentError) as raised:
        await persistent_hosts.create_host(owner=OWNER, name="Work computer", provider=None)
    assert raised.value.code == ErrorCode.CONFLICT
    assert fresh.computer.events == []


async def test_failed_provision_leaves_a_failed_host_that_starts_again(fresh: FakeHost) -> None:
    """A refused provision keeps the row as "failed" with the reason; Start again recovers."""
    fresh.computer.provision_error = "all computers are busy"
    host_id = await _created(fresh)
    failed = fresh.store.get_host(host_id)
    assert failed is not None and failed.sandbox_id is None
    status = persistent_hosts.persistent_status(failed, online=False)
    assert status is not None
    assert status["state"] == "failed" and status["failure"] == "all computers are busy"
    fresh.computer.provision_error = None
    await asyncio.wait_for(persistent_hosts.start_host_now(host_id), 10)
    recovered = fresh.store.get_host(host_id)
    assert recovered is not None and recovered.start_failure is None
    assert "terminate" not in fresh.computer.events


async def test_failed_first_start_never_terminates_the_computer(
    fresh: FakeHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A first start whose host never comes online leaves the computer, marked failed."""
    fresh.computer.on_host_start = None  # the host never connects
    deployment = _deployment(fresh.computer, online_timeout_s=0.5)
    persistent_hosts.configure(fresh.store, fresh.registry, deployment)
    host_id = await _created(fresh)
    failed = fresh.store.get_host(host_id)
    assert failed is not None and failed.sandbox_id == SANDBOX_ID
    assert failed.start_failure is not None and "did not come online" in failed.start_failure
    assert "terminate" not in fresh.computer.events
    status = persistent_hosts.persistent_status(failed, online=False)
    assert status is not None and status["state"] == "failed"


async def test_host_deleted_during_provisioning_removes_the_new_sandbox(
    fresh: FakeHost,
) -> None:
    """Deleting a pending host while it provisions turns the new sandbox into a removal."""

    def delete_meanwhile() -> None:
        for row in fresh.store.list_hosts(OWNER):
            fresh.store.delete_host(row.host_id)

    fresh.computer.on_provision = delete_meanwhile
    await _created(fresh)
    assert fresh.computer.events == ["provision", "terminate"]


@pytest.mark.parametrize("removal_fails", [False, True])
async def test_deleting_while_provisioning_still_removes_the_new_sandbox(
    fresh: FakeHost, removal_fails: bool
) -> None:
    """The public delete path during a paused provision never loses the sandbox.

    The provision finishes, finds the host gone and removes its sandbox; when
    that removal fails, a tombstone keeps the sandbox for the reaper to retry.
    """
    provisioning, release = threading.Event(), threading.Event()

    def pause() -> None:
        provisioning.set()
        release.wait(10)

    fresh.computer.on_provision = pause
    fresh.computer.terminate_error = "provider unreachable" if removal_fails else None
    created = await persistent_hosts.create_host(owner=OWNER, name="Doomed", provider=None)
    task = persistent_hosts._states[created.host_id].wake_task
    assert task is not None
    assert await asyncio.to_thread(provisioning.wait, 10)
    await persistent_hosts.delete_host(created)
    # A new computer takes the deleted one's name before the provision returns.
    fresh.computer.on_provision = None
    reused = fresh.store.create_persistent_host(
        host_id=uuid.uuid4().hex, name="Doomed", user_id=OWNER, provider="fakepersist"
    )
    assert reused.name == "Doomed"
    release.set()
    with contextlib.suppress(OmnigentError):
        await asyncio.wait_for(task, 10)
    assert fresh.computer.events == ["provision", "terminate"]
    leftovers = fresh.store.list_deleted_persistent_hosts(OWNER)
    assert [row.sandbox_id for row in leftovers] == ([SANDBOX_ID] if removal_fails else [])


async def test_removed_persistent_policy_never_wakes_the_host_as_disposable(
    host: FakeHost,
) -> None:
    """A persistent row whose provider lost its persistent policy is refused, not re-armed."""
    await _asleep(host)
    plain = ManagedSandboxDeployment.single(
        dataclasses.replace(persistent_hosts._deployment.configs[0], persistent=None)
    )
    persistent_hosts.configure(host.store, host.registry, plain)
    with pytest.raises(OmnigentError) as raised:
        await resume_managed_host(HOST_ID, host.store, plain, force=True)
    assert "no longer configured" in str(raised.value)
    assert host.computer.events == []


async def test_sandbox_gone_marks_the_host_failed_and_never_provisions(host: FakeHost) -> None:
    """A computer deleted on the platform is reported, never replaced by a blank one."""
    await _asleep(host)
    host.computer.gone = True
    with pytest.raises(OmnigentError) as raised:
        await persistent_hosts.ensure_host_ready(HOST_ID, operation="message", timeout_s=5)
    assert "no longer exists" in str(raised.value)
    failed = host.store.get_host(HOST_ID)
    assert failed is not None and failed.sandbox_id == SANDBOX_ID
    assert failed.start_failure is not None
    assert "provision" not in host.computer.events
    assert "terminate" not in host.computer.events


async def test_managed_wake_of_a_gone_persistent_host_never_relaunches(host: FakeHost) -> None:
    """The generic managed wake raises no ``SandboxGoneError``, so nothing relaunches."""
    await _asleep(host)
    host.computer.gone = True
    deployment = persistent_hosts._deployment
    with pytest.raises(OmnigentError):
        await resume_managed_host(HOST_ID, host.store, deployment, force=True)
    assert "provision" not in host.computer.events
    assert "terminate" not in host.computer.events


async def test_relaunch_never_replaces_a_persistent_sandbox(host: FakeHost) -> None:
    """``relaunch_managed_host`` refuses a persistent host without terminating it."""
    from fastapi import HTTPException

    from omnigent.server.managed_hosts import relaunch_managed_host

    current = host.store.get_host(HOST_ID)
    assert current is not None
    with pytest.raises(HTTPException) as raised:
        await relaunch_managed_host(
            config=persistent_hosts._deployment, host=current, host_store=host.store
        )
    assert raised.value.status_code == 409
    assert host.computer.events == []


async def test_changed_row_while_waking_is_not_terminated(
    host: FakeHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A re-arm that loses its race fails the wake; the computer is never terminated."""
    await _asleep(host)
    monkeypatch.setattr(host.store, "rearm_managed_host", lambda *args, **kwargs: None)
    with pytest.raises(OmnigentError):
        await persistent_hosts.ensure_host_ready(HOST_ID, operation="message", timeout_s=5)
    assert "terminate" not in host.computer.events


async def test_delete_terminates_once_as_a_removal(host: FakeHost) -> None:
    """Deleting a persistent host tombstones it, then asks the provider to remove it."""
    current = host.store.get_host(HOST_ID)
    assert current is not None
    await persistent_hosts.delete_host(current)
    assert host.store.get_host(HOST_ID) is None
    assert host.computer.events == ["terminate"]
    assert HOST_ID not in persistent_hosts._states


async def test_deleted_account_hosts_are_removed_after_the_commit(host: FakeHost) -> None:
    """Account deletion's tombstones are terminated; live hosts of others are not."""
    host.store.delete_host(HOST_ID)  # what the account deletion's transaction does
    await persistent_hosts.remove_deleted_account_hosts(OWNER)
    assert host.computer.events == ["terminate"]
    await persistent_hosts.remove_deleted_account_hosts(OWNER)
    assert host.computer.events == ["terminate"]  # the tombstone was cleared


# ── Provider settings and token lifetime ───────────────────────────────


async def test_online_wait_is_a_provider_setting(
    host: FakeHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The wake gives up after the provider's online wait, not Omnigent's default."""
    await _asleep(host)
    host.computer.on_host_start = None
    deployment = _deployment(host.computer, online_timeout_s=0.4)
    persistent_hosts.configure(host.store, host.registry, deployment)
    with pytest.raises(OmnigentError) as raised:
        await persistent_hosts.ensure_host_ready(HOST_ID, operation="message", timeout_s=5)
    assert "within 0.4s" in str(raised.value)
    assert deployment.configs[0].host_online_timeout_s == 0.4
    default = dataclasses.replace(deployment.configs[0], online_timeout_s=None)
    assert default.host_online_timeout_s == MANAGED_HOST_ONLINE_TIMEOUT_S
    assert dataclasses.replace(default, online_timeout_s=240).host_online_timeout_s == 240


async def test_token_near_expiry_sleeps_at_the_next_idle_moment(
    host: FakeHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A host whose token expires within the margin sleeps when idle, not after the window."""
    deployment = _deployment(host.computer, idle_sleep_s=3600, token_refresh_margin_s=7200)
    persistent_hosts.configure(host.store, host.registry, deployment)
    host.runners.append("runner-1")
    outcomes = await persistent_hosts.PersistentHostIdleMonitor().sweep_once()
    assert outcomes[HOST_ID].startswith("busy")
    host.runners.clear()
    outcomes = await persistent_hosts.PersistentHostIdleMonitor().sweep_once()
    assert outcomes[HOST_ID] == "stopped"


async def test_token_far_from_expiry_waits_for_the_idle_window(host: FakeHost) -> None:
    """Without a near expiry, an idle host sleeps only after the policy's idle window."""
    deployment = _deployment(host.computer, idle_sleep_s=3600, token_refresh_margin_s=60)
    persistent_hosts.configure(host.store, host.registry, deployment)
    outcomes = await persistent_hosts.PersistentHostIdleMonitor().sweep_once()
    assert outcomes[HOST_ID].startswith("idle for")
    assert "suspend" not in host.computer.events


async def test_reads_never_keep_a_host_awake(host: FakeHost) -> None:
    """A read admitted after the idle clock started does not restart it."""
    deployment = _deployment(host.computer, idle_sleep_s=3600, token_refresh_margin_s=60)
    persistent_hosts.configure(host.store, host.registry, deployment)
    await persistent_hosts.PersistentHostIdleMonitor().sweep_once()
    idle_since = persistent_hosts._states[HOST_ID].idle_since
    assert idle_since is not None
    await persistent_hosts.ensure_host_ready(
        HOST_ID, operation="list_dir", timeout_s=5, renews_idle=False
    )
    assert persistent_hosts._states[HOST_ID].idle_since == idle_since
    await persistent_hosts.ensure_host_ready(HOST_ID, operation="message", timeout_s=5)
    assert persistent_hosts._states[HOST_ID].idle_since is None


async def test_wake_progress_reaches_the_waiting_session(host: FakeHost) -> None:
    """A request that waits on a wake hears its stages, then "ready"."""
    await _asleep(host)
    stages: list[tuple[str, str | None]] = []
    await persistent_hosts.ensure_host_ready(
        HOST_ID,
        operation="message",
        timeout_s=10,
        progress=lambda stage, error: stages.append((stage, error)),
    )
    assert [stage for stage, _ in stages] == ["provisioning", "starting", "connecting", "ready"]


async def test_non_persistent_hosts_are_untouched(db_uri: str) -> None:
    """Readiness is a no-op for laptops and per-session sandboxes."""
    store, registry = HostStore(db_uri), HostRegistry()
    persistent_hosts.configure(store, registry, _deployment(FakeComputer()))
    try:
        store.upsert_on_connect(host_id=HOST_ID, name="laptop", user_id=OWNER)
        await persistent_hosts.ensure_host_ready(HOST_ID, operation="message", timeout_s=1)
        assert HOST_ID not in persistent_hosts._states
        laptop = store.get_host(HOST_ID)
        assert laptop is not None
        assert persistent_hosts.persistent_status(laptop, online=True) is None
    finally:
        persistent_hosts.configure(None, None, None)
