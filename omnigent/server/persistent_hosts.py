"""Persistent managed hosts: sandboxes that outlive sessions and sleep when idle.

A managed host normally belongs to one session. A provider entry whose
:class:`~omnigent.server.managed_hosts.ManagedSandboxConfig` carries a
:class:`~omnigent.server.managed_hosts.PersistentHostPolicy` creates hosts that
behave like a laptop instead: they are created without a session, survive
session deletion, appear in host pickers (asleep or awake), accept pinned
automations, and are exempt from the offline reaper. Their compute sleeps when
idle and wakes when work needs it.

This module owns that lifecycle on one replica:

* **Readiness.** Every path that starts work on a host calls
  :func:`ensure_host_ready`. It waits out a sleep in progress, wakes an asleep
  host once per host (a wake shared by every waiter, so a cancelled waiter never
  stops a wake another needs), re-checks authorization after the wait, and
  checks the provider's admission report.
* **Admission.** New work is admitted only while the host admits it. Admitted
  work holds a short lease; the sleep fence waits for leases to lapse.
* **The fenced sleep.** :func:`sleep_if_idle` closes admission under the host's
  lifecycle lock (the lock its wakes take), lets admitted work finish, asks the
  host for live runners, and only then calls the provider's ``suspend``. Host
  shutdown ends every runner, so nothing admitted is cut off by a sleep. An
  unknown outcome keeps admission closed until the same operation reports it.
  A confirmed sleep revokes the launch token.
* **Repair, not re-arm.** A host whose sandbox still runs is never re-armed or
  restarted: its process reconnects with its current token.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import secrets
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from omnigent.errors import ErrorCode, OmnigentError
from omnigent.onboarding.sandboxes.types import AdmissionReport, SuspendResult
from omnigent.server.managed_hosts import (
    MANAGED_HOST_ONLINE_TIMEOUT_S,
    _launcher_for_teardown,
    _resume_locks,
    _start_sandbox_host,
    _wait_for_host_online,
)

if TYPE_CHECKING:
    from omnigent.onboarding.sandboxes import SandboxHostLauncher
    from omnigent.server.host_registry import HostConnection, HostRegistry
    from omnigent.server.managed_hosts import ManagedSandboxDeployment, PersistentHostPolicy
    from omnigent.stores.host_store import Host, HostStore

_logger = logging.getLogger(__name__)

# How long admitted work holds off a sleep: it covers the gap between admission
# and the work showing on the host as a pending launch or a live runner.
ADMISSION_LEASE_S = 30.0
# How long a runner launch waits for a sleep in progress before giving up.
ADMISSION_WAIT_S = 120.0
_LIST_RUNNERS_TIMEOUT_S = 10.0
_POLL_S = 0.2

_host_store: HostStore | None = None
_host_registry: HostRegistry | None = None
_deployment: ManagedSandboxDeployment | None = None


@dataclass
class _HostLifecycle:
    """Per-host lifecycle state on this replica."""

    admitting: asyncio.Event = field(default_factory=asyncio.Event)
    closed_reason: str | None = None
    unknown_suspend_operation: str | None = None
    leases: dict[str, tuple[float, str]] = field(default_factory=dict)
    report: AdmissionReport | None = None
    report_at: float = 0.0
    wake_task: asyncio.Task[None] | None = None
    idle_since: float | None = None

    def __post_init__(self) -> None:
        self.admitting.set()

    def active_leases(self) -> list[str]:
        now = time.monotonic()
        for lease_id, (expires_at, _operation) in list(self.leases.items()):
            if expires_at <= now:
                self.leases.pop(lease_id, None)
        return [operation for _expires_at, operation in self.leases.values()]


# custom-lint: disable-next=workspace-scoped-cache -- host_id state; one per persistent host
_states: dict[str, _HostLifecycle] = {}


def configure(
    host_store: HostStore | None,
    host_registry: HostRegistry | None,
    deployment: ManagedSandboxDeployment | None,
) -> None:
    """Wire the stores and provider set at app construction.

    :param host_store: Persistent host registrations.
    :param host_registry: Live host tunnels on this replica.
    :param deployment: The deployment's providers; ``None`` disables the lifecycle.
    """
    global _host_store, _host_registry, _deployment
    _host_store, _host_registry, _deployment = host_store, host_registry, deployment


def enabled() -> bool:
    """Whether any configured provider creates persistent hosts."""
    return (
        _host_store is not None
        and _host_registry is not None
        and _deployment is not None
        and any(config.persistent is not None for config in _deployment.offered())
    )


def policy_for(host: Host) -> PersistentHostPolicy | None:
    """Return the persistent policy governing *host*, or ``None`` for other hosts."""
    if _deployment is None or host.sandbox_provider is None:
        return None
    config = _deployment.for_provider(host.sandbox_provider)
    return config.persistent if config is not None else None


def is_persistent_host(host: Host) -> bool:
    """Whether *host* is a persistent managed host."""
    return policy_for(host) is not None


def _state(host_id: str) -> _HostLifecycle:
    return _states.setdefault(host_id, _HostLifecycle())


def forget(host_id: str) -> None:
    """Drop lifecycle state for a deleted host."""
    _states.pop(host_id, None)


def lifecycle_status(host: Host, *, online: bool) -> str | None:
    """Picker state for a persistent host: ``awake``, ``asleep`` or ``sleeping``."""
    if not is_persistent_host(host):
        return None
    state = _states.get(host.host_id)
    if state is not None and not state.admitting.is_set():
        return "sleeping"
    return "awake" if online else "asleep"


async def _persistent_host(host_id: str) -> Host | None:
    if not enabled():
        return None
    assert _host_store is not None
    host = await asyncio.to_thread(_host_store.get_host, host_id)
    return host if host is not None and is_persistent_host(host) else None


def _unavailable(message: str) -> OmnigentError:
    return OmnigentError(message, code=ErrorCode.RUNNER_UNAVAILABLE)


async def ensure_host_ready(
    host_id: str,
    *,
    operation: str,
    timeout_s: float,
    owner: str | None = None,
) -> None:
    """Make a persistent host ready for one new operation, within a deadline.

    No-op for hosts that are not persistent. Waits while a sleep is in
    progress, wakes an asleep host (one physical wake per host, shared by all
    waiters), re-checks authorization after any wait, and admits the operation
    only while the provider reports the host admitting.

    :param host_id: Host the operation targets.
    :param operation: What the work is, e.g. ``"message"``, for logs and leases.
    :param timeout_s: Deadline for the whole wait, in seconds.
    :param owner: Caller the host must still belong to, or ``None`` to skip.
    :raises OmnigentError: ``HOST_NOT_ADMITTING`` with the provider's reason;
        ``RUNNER_UNAVAILABLE`` when the host cannot be made ready in time;
        ``NOT_FOUND`` when the host was deleted during the wait.
    """
    host = await _persistent_host(host_id)
    if host is None:
        return
    assert _host_store is not None and _host_registry is not None
    deadline = time.monotonic() + timeout_s
    state = _state(host_id)
    while True:
        await _wait_admitting(state, host, deadline)
        # Re-check authorization after every wait: the host may have been
        # deleted, re-owned or its owner's account revoked meanwhile.
        current = await asyncio.to_thread(_host_store.get_host, host_id)
        if current is None or current.deleted_at is not None:
            raise OmnigentError(f"host {host.name!r} was removed", code=ErrorCode.NOT_FOUND)
        if owner is not None and current.user_id != owner:
            raise OmnigentError(f"host {host.name!r} is not yours", code=ErrorCode.FORBIDDEN)
        host = current
        if _host_registry.get(host_id) is not None:
            report = await _admission(host, state)
            if not report.admitting:
                raise OmnigentError(
                    f"{host.name} is not accepting new work: {report.reason or 'no reason given'}",
                    code=ErrorCode.HOST_NOT_ADMITTING,
                )
            _grant_lease(state, operation)
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise _unavailable(f"{host.name} did not become ready within {timeout_s:.0f}s")
        if await asyncio.to_thread(_host_store.is_online, host_id):
            # Registered in the database a moment before this replica's registry.
            await asyncio.sleep(_POLL_S)
            continue
        try:
            await asyncio.wait_for(asyncio.shield(_wake_task(host_id, state)), remaining)
        except TimeoutError:
            raise _unavailable(
                f"{host.name} did not wake within {timeout_s:.0f}s; it may still be starting"
            ) from None
        except OmnigentError:
            raise
        except Exception as exc:
            detail = getattr(exc, "detail", None) or str(exc)
            raise _unavailable(f"{host.name} could not wake: {detail}") from exc


async def admit_host_work(host_id: str, operation: str) -> None:
    """Admission gate for host operations that bypass readiness, e.g. runner launches.

    Work that already holds a lease (admitted before a sleep closed admission)
    passes; anything else waits for the sleep to resolve, then must find the
    host still online and admitting.

    :raises OmnigentError: When the host went to sleep or does not admit work.
    """
    host = await _persistent_host(host_id)
    if host is None:
        return
    assert _host_registry is not None
    state = _state(host_id)
    if not state.admitting.is_set() and not state.active_leases():
        await _wait_admitting(state, host, time.monotonic() + ADMISSION_WAIT_S)
        if _host_registry.get(host_id) is None:
            raise _unavailable(f"{host.name} went to sleep; send the request again to wake it")
    report = await _admission(host, state)
    if not report.admitting:
        raise OmnigentError(
            f"{host.name} is not accepting new work: {report.reason or 'no reason given'}",
            code=ErrorCode.HOST_NOT_ADMITTING,
        )
    _grant_lease(state, operation)


async def wake_if_persistent(host_id: str) -> bool:
    """Wake *host_id* through its lifecycle if it is persistent.

    Used by the generic managed-host wake so every wake of a persistent host
    takes the same single-flight, repair-not-re-arm path.

    :returns: ``False`` when the host is not persistent (nothing was done).
    """
    host = await _persistent_host(host_id)
    if host is None:
        return False
    assert _host_registry is not None
    state = _state(host_id)
    await _wait_admitting(state, host, time.monotonic() + MANAGED_HOST_ONLINE_TIMEOUT_S)
    if _host_registry.get(host_id) is None:
        await asyncio.shield(_wake_task(host_id, state))
    return True


def _grant_lease(state: _HostLifecycle, operation: str) -> None:
    state.leases[secrets.token_hex(6)] = (time.monotonic() + ADMISSION_LEASE_S, operation)
    state.idle_since = None


async def _wait_admitting(state: _HostLifecycle, host: Host, deadline: float) -> None:
    """Hold work while a sleep is in progress, resolving an unknown outcome."""
    while not state.admitting.is_set():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise _unavailable(
                f"{host.name} is {state.closed_reason or 'not admitting'}; try again shortly"
            )
        if state.unknown_suspend_operation is not None:
            # A wake waits for the earlier sleep's outcome; ask for it again.
            await sleep_if_idle(host.host_id)
            if not state.admitting.is_set():
                await asyncio.sleep(min(1.0, max(remaining, 0)))
            continue
        try:
            await asyncio.wait_for(state.admitting.wait(), remaining)
        except TimeoutError:
            continue


async def _admission(host: Host, state: _HostLifecycle) -> AdmissionReport:
    """The provider's admission report, at most the policy's maximum age old."""
    policy = policy_for(host)
    launcher = _launcher_for_teardown(host, _deployment)
    if policy is None or launcher is None or not launcher.capabilities.admission:
        return AdmissionReport(admitting=True)
    now = time.monotonic()
    if state.report is not None and now - state.report_at <= policy.admission_max_age_s:
        return state.report
    assert host.sandbox_id is not None
    try:
        report = await asyncio.to_thread(launcher.admission, host.sandbox_id)
    except Exception:  # noqa: BLE001 — an unrefreshable report counts as not admitting.
        _logger.warning("admission report for host %s failed", host.host_id, exc_info=True)
        report = AdmissionReport(admitting=False, reason="computer status unknown")
    state.report, state.report_at = report, now
    return report


def _wake_task(host_id: str, state: _HostLifecycle) -> asyncio.Task[None]:
    """The host's in-flight wake, started if none is running."""
    if state.wake_task is None or state.wake_task.done():
        state.wake_task = asyncio.create_task(_wake(host_id, state), name=f"wake-host-{host_id}")
    return state.wake_task


async def _wake(host_id: str, state: _HostLifecycle) -> None:
    """Wake one persistent host under its lifecycle lock."""
    assert _host_store is not None and _host_registry is not None
    async with _resume_locks.setdefault(host_id, asyncio.Lock()):
        if _host_registry.get(host_id) is not None:
            return
        host = await asyncio.to_thread(_host_store.get_host, host_id)
        if host is None or host.sandbox_id is None:
            raise _unavailable(f"host {host_id} no longer exists")
        launcher = _launcher_for_teardown(host, _deployment)
        policy = policy_for(host)
        if launcher is None or policy is None:
            raise _unavailable(f"{host.name}'s provider is no longer configured")
        sandbox_id = host.sandbox_id
        if await asyncio.to_thread(launcher.is_running, sandbox_id):
            # Repair, not re-arm: stopping or re-arming a live host ends its
            # runners, so wait for it to reconnect with its current token.
            _logger.info("Host %s sandbox is running; waiting for it to reconnect", host_id)
            await _wait_for_host_online(_host_store, host_id)
            return
        _logger.info("Waking persistent host %s (sandbox %s)", host_id, sandbox_id)
        await asyncio.to_thread(launcher.prepare_for_launch, agent_name=None)
        await asyncio.to_thread(launcher.resume, sandbox_id)
        entry = _deployment.recorded(host.sandbox_provider) if _deployment else None
        assert entry is not None
        token = secrets.token_urlsafe(32)
        from omnigent.db.utils import now_epoch

        armed = await asyncio.to_thread(
            _host_store.rearm_managed_host,
            host_id,
            sandbox_id=sandbox_id,
            expected_updated_at=host.updated_at,
            token=token,
            token_expires_at=now_epoch() + entry.token_ttl_s,
        )
        if armed is None:
            raise _unavailable(f"{host.name} changed while waking; try again")
        # A new generation starts with a fresh admission report.
        state.report = None
        await _start_sandbox_host(
            launcher,
            sandbox_id,
            token=token,
            host_id=host_id,
            host_name=host.name,
            server_url=entry.server_url,
            repos=(),
            host_config=entry.host_config,
        )
        await _wait_for_host_online(_host_store, host_id)


async def list_host_runners(conn: HostConnection) -> dict[str, Any] | None:
    """Ask a connected host for its live and launching runners.

    :returns: ``{"runners": [...], "launching": n}``, or ``None`` when the host
        cannot answer (older host, timeout or dropped connection).
    """
    from omnigent.host.frames import CAP_LIST_RUNNERS, HostListRunnersFrame, encode_host_frame

    assert _host_registry is not None
    if CAP_LIST_RUNNERS not in conn.hello.capabilities:
        return None
    request_id = secrets.token_hex(8)
    future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
    conn.pending_runner_lists[request_id] = future
    try:
        _host_registry.send_text(conn, encode_host_frame(HostListRunnersFrame(request_id)))
        return await asyncio.wait_for(future, _LIST_RUNNERS_TIMEOUT_S)
    except (ConnectionError, TimeoutError):
        return None
    finally:
        conn.pending_runner_lists.pop(request_id, None)


def _busy_reason(
    state: _HostLifecycle, conn: HostConnection, runners: dict[str, Any] | None
) -> str | None:
    """Why the host must not sleep now, or ``None`` when it is quiescent."""
    if runners is None:
        return "host cannot list its runners"
    if runners["runners"] or runners["launching"]:
        return f"runners {runners['runners']}, launching {runners['launching']}"
    if conn.pending_launches or conn.inflight_installs:
        return "runner launch or harness install in flight"
    leases = state.active_leases()
    return f"admitted work {leases}" if leases else None


async def _call_suspend(
    launcher: SandboxHostLauncher, sandbox_id: str, operation_id: str
) -> SuspendResult:
    try:
        return await asyncio.to_thread(launcher.suspend, sandbox_id, operation_id=operation_id)
    except Exception:  # noqa: BLE001 — a lost reply means the stop may have begun.
        _logger.warning("suspend %s of sandbox %s failed", operation_id, sandbox_id, exc_info=True)
        return SuspendResult(outcome="unknown")


async def sleep_if_idle(host_id: str) -> str:
    """Run the fenced sleep for one persistent host.

    :returns: ``"abandoned"`` (work exists or the host cannot answer),
        ``"refused"``, ``"stopped"`` or ``"unknown"``.
    """
    host = await _persistent_host(host_id)
    if host is None or host.sandbox_id is None:
        return "abandoned"
    assert _host_store is not None and _host_registry is not None
    launcher = _launcher_for_teardown(host, _deployment)
    if launcher is None or not launcher.capabilities.suspend:
        return "abandoned"
    state = _state(host_id)
    outcome = "abandoned"
    async with _resume_locks.setdefault(host_id, asyncio.Lock()):
        operation_id = state.unknown_suspend_operation
        conn = _host_registry.get(host_id)
        if operation_id is None and conn is None:
            return outcome
        state.admitting.clear()
        state.closed_reason = "going to sleep"
        try:
            if operation_id is None:
                assert conn is not None
                # Admitted work, launches and installs finish before the host is asked.
                deadline = time.monotonic() + ADMISSION_LEASE_S
                while state.active_leases() or conn.pending_launches or conn.inflight_installs:
                    if time.monotonic() >= deadline:
                        return outcome
                    await asyncio.sleep(_POLL_S)
                if _busy_reason(state, conn, await list_host_runners(conn)) is not None:
                    return outcome
                operation_id = uuid.uuid4().hex
            result = await _call_suspend(launcher, host.sandbox_id, operation_id)
            outcome = result.outcome
            _logger.info(
                "Sleep %s of host %s: %s%s",
                operation_id,
                host_id,
                outcome,
                f" ({result.reason})" if result.reason else "",
            )
            if outcome == "stopped":
                await asyncio.to_thread(_host_store.revoke_launch_token, host_id)
                await asyncio.to_thread(_host_store.set_offline, host_id)
                _host_registry.deregister(host_id)
                state.report = None
            elif outcome == "unknown":
                state.unknown_suspend_operation = operation_id
                state.closed_reason = "going to sleep (outcome unknown)"
            return outcome
        finally:
            if outcome != "unknown":
                state.unknown_suspend_operation = None
                state.closed_reason = None
                state.idle_since = None
                state.admitting.set()


class PersistentHostIdleMonitor:
    """Put idle persistent hosts on this replica to sleep."""

    def __init__(self) -> None:
        self._task: asyncio.Task[None] | None = None

    def interval_s(self) -> float:
        assert _deployment is not None
        return min(
            config.persistent.sweep_interval_s
            for config in _deployment.offered()
            if config.persistent is not None
        )

    async def start(self) -> None:
        if enabled() and self._task is None:
            self._task = asyncio.create_task(self._run(), name="persistent-host-idle-monitor")

    async def shutdown(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def _run(self) -> None:
        while True:
            await asyncio.sleep(self.interval_s())
            try:
                await self.sweep_once()
            except Exception:
                _logger.exception("persistent host idle sweep failed")

    async def sweep_once(self) -> dict[str, str]:
        """Check every connected persistent host once; returns each host's state."""
        assert _host_store is not None and _host_registry is not None
        outcomes: dict[str, str] = {}
        for host_id, state in list(_states.items()):
            if state.unknown_suspend_operation is not None:
                outcomes[host_id] = await sleep_if_idle(host_id)
        for host_id in _host_registry.online_host_ids():
            host = await _persistent_host(host_id)
            conn = _host_registry.get(host_id)
            policy = policy_for(host) if host is not None else None
            if host is None or conn is None or policy is None or host_id in outcomes:
                continue
            state = _state(host_id)
            busy = _busy_reason(state, conn, await list_host_runners(conn))
            if busy is not None:
                state.idle_since = None
                outcomes[host_id] = f"busy: {busy}"
                continue
            now = time.monotonic()
            if state.idle_since is None:
                state.idle_since = now
            if now - state.idle_since >= policy.idle_sleep_s:
                outcomes[host_id] = await sleep_if_idle(host_id)
            else:
                outcomes[host_id] = f"idle for {now - state.idle_since:.0f}s"
        return outcomes
