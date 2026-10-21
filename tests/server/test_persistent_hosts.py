"""Persistent managed hosts: admission handles, the fenced sleep and its recovery.

Each test drives the real lifecycle module against a real host store and host
registry. A fake provider stands in for the sandbox and a fake host answers
``host.list_runners`` from the registry's outbound queue. Blocking points in the
fakes park a coroutine at a chosen await, so interleavings are deterministic.
The assertions check the physical outcome: whether the provider stopped or
resumed the sandbox, and when work was admitted relative to that.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import dataclasses
import threading
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

import pytest

from omnigent.db.utils import now_epoch
from omnigent.errors import ErrorCode, OmnigentError
from omnigent.host.frames import (
    CAP_LIST_RUNNERS,
    HostHelloFrame,
    HostListRunnersFrame,
    decode_host_frame,
)
from omnigent.onboarding.sandboxes.base import SandboxHostLauncher
from omnigent.onboarding.sandboxes.types import AdmissionReport, RepoWorkspace, SuspendResult
from omnigent.server import persistent_hosts
from omnigent.server.host_registry import HostRegistry
from omnigent.server.managed_hosts import (
    ManagedSandboxConfig,
    ManagedSandboxDeployment,
    PersistentHostPolicy,
)
from omnigent.stores.host_store import HostStore

pytestmark = [pytest.mark.asyncio, pytest.mark.timeout(60)]

# Set by a request under test; worker threads see it through to_thread's context copy.
_in_request: contextvars.ContextVar[bool] = contextvars.ContextVar("in_request", default=False)


@pytest.fixture(autouse=True)
def _bounded_fence(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep a fence that waits for admitted work from outliving a failing test."""
    monkeypatch.setattr(persistent_hosts, "FENCE_DRAIN_S", 5.0)


_HOST_ID = "5b0e7c1d9a2f4e6b8c3d1a0f9e8d7c6b"
_OWNER = "local"
_SANDBOX = "sb-persistent-1"


class _FakeWs:
    """Accepts frames without a peer; the fake host reads the outbound queue."""

    async def send_text(self, data: str) -> None:
        del data

    async def close(self, code: int = 1000, reason: str = "") -> None:
        del code, reason


@dataclass
class _Provider(SandboxHostLauncher):
    """In-memory persistent sandbox with blocking points for interleaving tests."""

    provider: ClassVar[str] = "fakepersist"
    can_resume: ClassVar[bool] = True

    events: list[str] = field(default_factory=list)
    running: bool = True
    admitting: bool = True
    admission_gate: threading.Event | None = None
    suspend_gate: threading.Event | None = None
    suspend_accepted: threading.Event = field(default_factory=threading.Event)
    parked: threading.Event = field(default_factory=threading.Event)
    suspend_refuses: bool = False
    outcomes: dict[str, SuspendResult] = field(default_factory=dict)
    in_flight: set[str] = field(default_factory=set)
    on_start: Callable[[str], None] | None = None

    def prepare(self) -> None:
        return None

    def provision(self, name: str) -> str:
        del name
        return _SANDBOX

    def start_host(
        self,
        sandbox_id: str,
        *,
        token: str,
        host_id: str,
        host_name: str,
        server_url: str,
        repos: Sequence[RepoWorkspace] = (),
        host_config: dict[str, object] | None = None,
        on_stage: Callable[[str], None] | None = None,
    ) -> str:
        del sandbox_id, host_id, host_name, server_url, repos, host_config, on_stage
        self.events.append("start_host")
        if self.on_start is not None:
            self.on_start(token)
        return "/root/workspace"

    def resume(self, sandbox_id: str) -> None:
        del sandbox_id
        self.events.append("resume")
        self.running = True

    def is_running(self, sandbox_id: str) -> bool | None:
        del sandbox_id
        return self.running

    def admission(self, sandbox_id: str) -> AdmissionReport:
        del sandbox_id
        if self.admission_gate is not None:
            gate, self.admission_gate = self.admission_gate, None
            self.parked.set()
            gate.wait(10)
        return AdmissionReport(
            admitting=self.admitting, reason=None if self.admitting else "not saved"
        )

    def suspend(self, sandbox_id: str, *, operation_id: str) -> SuspendResult:
        del sandbox_id
        if operation_id in self.outcomes:
            return self.outcomes[operation_id]
        if operation_id in self.in_flight:
            return SuspendResult(outcome="unknown")  # the first call is still stopping
        self.in_flight.add(operation_id)
        self.events.append("suspend")
        self.suspend_accepted.set()
        if self.suspend_gate is not None:
            self.suspend_gate.wait(10)
        if self.suspend_refuses:
            result = SuspendResult(outcome="refused", reason="saving")
        else:
            self.running = False
            self.events.append("stopped")
            result = SuspendResult(outcome="stopped")
        self.outcomes[operation_id] = result
        return result

    def terminate(self, sandbox_id: str) -> None:
        del sandbox_id


@dataclass
class _Host:
    """One persistent host wired to the lifecycle module."""

    store: HostStore
    registry: HostRegistry
    provider: _Provider
    runners: list[str] = field(default_factory=list)
    _answerers: list[asyncio.Task[None]] = field(default_factory=list)

    def connect(self, token: str) -> None:
        """The host dials in with its launch token and answers runner queries."""
        self.store.upsert_on_connect(
            host_id=_HOST_ID, name="computer", user_id=_OWNER, managed_token=token
        )
        hello = HostHelloFrame(
            version="test",
            frame_protocol_version=1,
            name="computer",
            capabilities=[CAP_LIST_RUNNERS],
        )
        conn = self.registry.register(_HOST_ID, _FakeWs(), hello, owner=_OWNER)
        self._answerers.append(asyncio.get_running_loop().create_task(self._answer(conn)))

    async def _answer(self, conn: Any) -> None:
        while True:
            outbound = await conn.outbound_queue.get()
            if outbound is None:
                return
            frame = decode_host_frame(outbound.data)
            if isinstance(frame, HostListRunnersFrame):
                future = conn.pending_runner_lists.get(frame.request_id)
                if future is not None and not future.done():
                    future.set_result({"runners": list(self.runners), "launching": 0})

    def close(self) -> None:
        for task in self._answerers:
            task.cancel()


@pytest.fixture
async def host(db_uri: str) -> AsyncIterator[_Host]:
    store = HostStore(db_uri)
    registry = HostRegistry()
    # Wired as create_app wires them.
    registry.admission_gate = persistent_hosts.admit_host_work
    registry.send_guard = persistent_hosts.guard_host_request
    provider = _Provider()
    persistent = _Host(store, registry, provider)
    loop = asyncio.get_running_loop()
    provider.on_start = lambda token: loop.call_soon_threadsafe(persistent.connect, token)
    deployment = ManagedSandboxDeployment.single(
        ManagedSandboxConfig(
            server_url="http://omnigent.test",
            launcher_factory=lambda: provider,
            token_ttl_s=3600,
            provider=provider.provider,
            persistent=PersistentHostPolicy(idle_sleep_s=0, admission_max_age_s=0),
        )
    )
    persistent_hosts._states.clear()
    persistent_hosts.configure(store, registry, deployment)
    store.register_managed_host(
        host_id=_HOST_ID,
        name="computer",
        user_id=_OWNER,
        token="token-generation-1",
        provider=provider.provider,
        sandbox_id=_SANDBOX,
        token_expires_at=now_epoch() + 3600,
    )
    persistent.connect("token-generation-1")
    yield persistent
    persistent.close()
    persistent_hosts.configure(None, None, None)
    persistent_hosts._states.clear()


async def _admitted_after(host: _Host, ready: Callable[[], Any]) -> str:
    """Run *ready* to completion and note it in the provider's event order."""
    await ready()
    host.provider.events.append("admitted")
    return "admitted"


def _ready(operation: str) -> Callable[[], Any]:
    return lambda: persistent_hosts.ensure_host_ready(_HOST_ID, operation=operation, timeout_s=30)


@pytest.mark.parametrize("park", ["host_read", "admission_report"])
async def test_sleep_closing_while_readiness_awaits_holds_the_request(
    host: _Host, park: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sleep that closes admission during readiness's await is never raced.

    The request is parked at each await between the admission check and the
    grant; the fence then stops the sandbox. The request must be admitted only
    after a fresh wake, never on the host being stopped.
    """
    gate = threading.Event()
    if park == "admission_report":
        host.provider.admission_gate = gate
    else:
        original = host.store.get_host
        reads = {"request": 0}

        def parked_get_host(host_id: str) -> Any:
            # Only the request's own reads count; to_thread carries its context.
            if _in_request.get():
                reads["request"] += 1
                if reads["request"] == 2:  # the re-read after the admission wait
                    host.provider.parked.set()
                    gate.wait(10)
            return original(host_id)

        monkeypatch.setattr(host.store, "get_host", parked_get_host)

    async def request_task() -> str:
        _in_request.set(True)
        return await _admitted_after(host, _ready("message"))

    request = asyncio.create_task(request_task())
    assert await asyncio.to_thread(host.provider.parked.wait, 10)
    assert await persistent_hosts.sleep_if_idle(_HOST_ID) == "stopped"
    gate.set()
    assert await asyncio.wait_for(request, 30) == "admitted"
    assert host.provider.events == ["suspend", "stopped", "resume", "start_host", "admitted"]


async def test_distinct_request_waits_while_another_owns_admission(
    host: _Host, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the owner of an admission passes a closed fence; another request waits."""
    monkeypatch.setattr(persistent_hosts, "FENCE_DRAIN_S", 30.0)
    owner_done = asyncio.Event()
    order: list[str] = []

    async def owner() -> None:
        await persistent_hosts.ensure_host_ready(_HOST_ID, operation="message", timeout_s=5)
        await asyncio.sleep(0.3)  # the fence closes meanwhile
        await persistent_hosts.admit_host_work(_HOST_ID, "runner_launch")
        order.append("owner launch admitted")
        host.runners.append("runner-1")  # the launch shows on the host
        await owner_done.wait()

    async def other() -> None:
        await persistent_hosts.admit_host_work(_HOST_ID, "runner_launch")
        order.append("other launch admitted")

    owner_task = asyncio.create_task(owner())
    await asyncio.sleep(0.1)
    fence = asyncio.create_task(persistent_hosts.sleep_if_idle(_HOST_ID))
    await asyncio.sleep(0.1)
    other_task = asyncio.create_task(other())
    await asyncio.sleep(0.5)
    assert order == ["owner launch admitted"]
    owner_done.set()
    await owner_task
    assert await asyncio.wait_for(fence, 10) == "abandoned"
    await asyncio.wait_for(other_task, 10)
    assert order == ["owner launch admitted", "other launch admitted"]
    assert "suspend" not in host.provider.events


async def test_operation_longer_than_any_lease_keeps_the_host_awake(
    host: _Host, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An admitted operation blocks sleep for as long as it runs, not for a fixed lease."""
    release = asyncio.Event()

    async def long_install() -> None:
        await persistent_hosts.admit_host_work(_HOST_ID, "harness_install")
        await release.wait()

    install = asyncio.create_task(long_install())
    await asyncio.sleep(0.1)
    clock = persistent_hosts.time.monotonic
    monkeypatch.setattr(persistent_hosts.time, "monotonic", lambda: clock() + 3600)
    outcomes = await persistent_hosts.PersistentHostIdleMonitor().sweep_once()
    assert outcomes[_HOST_ID] == "busy: admitted work ['harness_install']"
    assert "suspend" not in host.provider.events
    release.set()
    await install
    outcomes = await persistent_hosts.PersistentHostIdleMonitor().sweep_once()
    assert outcomes[_HOST_ID] == "stopped"


async def test_cancelled_sleep_after_the_stop_was_accepted_stays_closed(host: _Host) -> None:
    """Cancelling the fence mid-suspend keeps admission closed until the outcome is known."""
    host.provider.suspend_gate = threading.Event()
    fence = asyncio.create_task(persistent_hosts.sleep_if_idle(_HOST_ID))
    await asyncio.to_thread(host.provider.suspend_accepted.wait, 10)
    fence.cancel()
    with pytest.raises(asyncio.CancelledError):
        await fence
    state = persistent_hosts._states[_HOST_ID]
    assert not state.admitting.is_set()
    recorded = host.store.get_host(_HOST_ID)
    assert recorded is not None
    assert state.unknown_suspend_operation == recorded.sleep_operation_id is not None
    request = asyncio.create_task(_admitted_after(host, _ready("message")))
    await asyncio.sleep(0.5)
    assert not request.done()
    host.provider.suspend_gate.set()  # the stop the provider accepted completes
    assert await asyncio.wait_for(request, 30) == "admitted"
    assert host.provider.events == ["suspend", "stopped", "resume", "start_host", "admitted"]
    assert host.store.get_host(_HOST_ID).sleep_operation_id is None


async def test_restart_before_the_stop_reply_recovers_the_sleep(host: _Host) -> None:
    """A new server process finds the recorded sleep and resolves it before admitting."""
    host.provider.suspend_gate = threading.Event()
    fence = asyncio.create_task(persistent_hosts.sleep_if_idle(_HOST_ID))
    await asyncio.to_thread(host.provider.suspend_accepted.wait, 10)
    # The process dies: in-memory lifecycle state and the live tunnel are gone.
    fence.cancel()
    with pytest.raises(asyncio.CancelledError):
        await fence
    persistent_hosts._states.clear()
    host.registry.deregister(_HOST_ID)
    assert await persistent_hosts.recover_pending_sleeps() == [_HOST_ID]
    # The old token stopped working before the stop was requested.
    assert host.store.resolve_launch_token(_HOST_ID, "token-generation-1") is None
    request = asyncio.create_task(_admitted_after(host, _ready("message")))
    await asyncio.sleep(0.5)
    assert not request.done()
    host.provider.suspend_gate.set()
    assert await asyncio.wait_for(request, 30) == "admitted"
    assert host.provider.events == ["suspend", "stopped", "resume", "start_host", "admitted"]


async def test_refused_sleep_restores_the_same_token(host: _Host) -> None:
    """A refused sleep reopens admission and the host's existing token works again."""
    host.provider.suspend_refuses = True
    assert await persistent_hosts.sleep_if_idle(_HOST_ID) == "refused"
    assert host.store.resolve_launch_token(_HOST_ID, "token-generation-1") is not None
    assert persistent_hosts._states[_HOST_ID].admitting.is_set()
    await persistent_hosts.ensure_host_ready(_HOST_ID, operation="message", timeout_s=5)
    assert host.provider.events == ["suspend"]


async def test_not_admitting_refuses_with_the_reason(host: _Host) -> None:
    """A provider that reports "not saved" refuses new work with that reason."""
    host.provider.admitting = False
    with pytest.raises(OmnigentError) as raised:
        await persistent_hosts.ensure_host_ready(_HOST_ID, operation="message", timeout_s=5)
    assert raised.value.code == ErrorCode.HOST_NOT_ADMITTING
    assert "not saved" in str(raised.value)


async def test_cancelled_waiter_does_not_stop_a_shared_wake(host: _Host) -> None:
    """Cancelling one request waiting on a wake leaves the wake for the others."""
    assert await persistent_hosts.sleep_if_idle(_HOST_ID) == "stopped"
    first = asyncio.create_task(_ready("message")())
    second = asyncio.create_task(_ready("message")())
    await asyncio.sleep(0.05)
    first.cancel()
    await asyncio.wait_for(second, 30)
    assert host.provider.events.count("resume") == 1
    assert host.registry.get(_HOST_ID) is not None


async def test_expired_wait_admits_nothing(host: _Host) -> None:
    """A request whose deadline passes during a sleep is refused and never admitted."""
    host.provider.suspend_gate = threading.Event()
    fence = asyncio.create_task(persistent_hosts.sleep_if_idle(_HOST_ID))
    await asyncio.to_thread(host.provider.suspend_accepted.wait, 10)
    with pytest.raises(OmnigentError) as raised:
        await persistent_hosts.ensure_host_ready(_HOST_ID, operation="message", timeout_s=0.5)
    assert raised.value.code == ErrorCode.RUNNER_UNAVAILABLE
    assert persistent_hosts._states[_HOST_ID].admitted() == []
    host.provider.suspend_gate.set()
    assert await fence == "stopped"


@pytest.mark.parametrize("written", [True, False])
async def test_cancel_after_the_intent_commits_before_the_await_returns(
    host: _Host, monkeypatch: pytest.MonkeyPatch, written: bool
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
    fence = asyncio.create_task(persistent_hosts.sleep_if_idle(_HOST_ID))
    await asyncio.to_thread(committed.wait, 10)
    fence.cancel()
    with pytest.raises(asyncio.CancelledError):
        await fence
    state = persistent_hosts._states[_HOST_ID]
    assert not state.admitting.is_set() and state.unknown_suspend_operation is not None
    request = asyncio.create_task(_admitted_after(host, _ready("message")))
    await asyncio.sleep(0.3)
    assert not request.done()
    release.set()
    assert await asyncio.wait_for(request, 30) == "admitted"
    if written:
        assert host.provider.events == ["suspend", "stopped", "resume", "start_host", "admitted"]
        assert host.store.resolve_launch_token(_HOST_ID, "token-generation-1") is None
    else:
        assert host.provider.events == ["admitted"]
        assert host.store.resolve_launch_token(_HOST_ID, "token-generation-1") is not None
    current = host.store.get_host(_HOST_ID)
    assert current is not None and current.sleep_operation_id is None
    assert host.registry.get(_HOST_ID) is not None


@pytest.mark.parametrize("parent_end", ["completes", "cancelled"])
async def test_child_task_keeps_the_operation_admitted_after_its_parent(
    host: _Host, monkeypatch: pytest.MonkeyPatch, parent_end: str
) -> None:
    """A child of an admitted request holds its own handle; the parent ending does not free it."""
    monkeypatch.setattr(persistent_hosts, "FENCE_DRAIN_S", 0.5)
    child_release = asyncio.Event()
    child_started = asyncio.Event()

    async def child() -> None:
        await persistent_hosts.ensure_host_ready(_HOST_ID, operation="child", timeout_s=5)
        child_started.set()
        await child_release.wait()

    children: list[asyncio.Task[None]] = []

    async def parent() -> None:
        await persistent_hosts.ensure_host_ready(_HOST_ID, operation="message", timeout_s=5)
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
    assert persistent_hosts._states[_HOST_ID].admitted() == ["child"]
    assert await persistent_hosts.sleep_if_idle(_HOST_ID) == "abandoned"
    assert "suspend" not in host.provider.events
    child_release.set()
    await children[0]
    assert await persistent_hosts.sleep_if_idle(_HOST_ID) == "stopped"


def _request_slots() -> list[str]:
    """Every per-connection map of server-to-host requests awaiting a reply."""
    from omnigent.server.host_registry import HostConnection

    return [
        item.name
        for item in dataclasses.fields(HostConnection)
        if item.name.startswith(("pending_", "inflight_")) and item.name != "pending_runner_lists"
    ]


@pytest.mark.parametrize("slot", _request_slots())
async def test_every_direct_host_request_in_flight_holds_the_sleep(
    host: _Host, monkeypatch: pytest.MonkeyPatch, slot: str
) -> None:
    """A direct host request awaiting its reply, of any kind, keeps the sandbox running."""
    monkeypatch.setattr(persistent_hosts, "FENCE_DRAIN_S", 0.3)
    conn = host.registry.get(_HOST_ID)
    assert conn is not None
    getattr(conn, slot)["request-1"] = asyncio.get_running_loop().create_future()
    assert await persistent_hosts.sleep_if_idle(_HOST_ID) == "abandoned"
    assert "suspend" not in host.provider.events
    getattr(conn, slot).clear()
    assert await persistent_hosts.sleep_if_idle(_HOST_ID) == "stopped"


async def test_direct_host_requests_are_refused_while_the_fence_is_closed(host: _Host) -> None:
    """A new direct host request during a sleep never reaches the host being stopped."""
    host.provider.suspend_gate = threading.Event()
    fence = asyncio.create_task(persistent_hosts.sleep_if_idle(_HOST_ID))
    await asyncio.to_thread(host.provider.suspend_accepted.wait, 10)
    conn = host.registry.get(_HOST_ID)
    assert conn is not None
    queued = conn.outbound_queue.qsize()
    with pytest.raises(ConnectionError, match="going to sleep"):
        host.registry.send_text(conn, '{"kind": "host.stat", "request_id": "r1", "path": "~"}')
    assert conn.outbound_queue.qsize() == queued
    host.provider.suspend_gate.set()
    assert await fence == "stopped"
