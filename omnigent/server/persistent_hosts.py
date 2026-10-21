"""Persistent managed hosts: sandboxes that outlive sessions and sleep when idle.

A managed host normally belongs to one session. A provider entry whose
:class:`~omnigent.server.managed_hosts.ManagedSandboxConfig` carries a
:class:`~omnigent.server.managed_hosts.PersistentHostPolicy` creates hosts that
behave like a laptop instead: they are created without a session
(``POST /v1/hosts``), survive session deletion, appear in host pickers, accept
pinned automations, and are exempt from the offline reaper. Their compute
sleeps when idle and wakes when work needs it. The host row's ``lifecycle``
column records the policy, so a configuration change never turns a persistent
host back into a disposable one.

This module owns that lifecycle on one replica:

* **Readiness.** Every path that starts work on a host calls
  :func:`ensure_host_ready`. It waits out a sleep in progress, wakes an asleep
  host once per host (a wake shared by every waiter, so a cancelled waiter never
  stops a wake another needs), re-checks authorization after the wait, and
  admits the work only while the provider reports the host admitting. A
  provider report that cannot be refreshed is never cached and never treated
  as a refusal; readiness asks again until its deadline.
* **Admission handles.** Admitting an operation gives the current task a handle
  for that host, released when the task finishes. The check and the grant run
  with no await between them, so they are atomic with the sleep fence.
* **The fenced sleep.** :func:`sleep_if_idle` closes admission under the host's
  lifecycle lock (the lock its wakes take), waits for admitted operations and
  host requests to finish, asks the host through ``host.quiesce`` whether any
  runner or host operation is live, records the operation (which revokes the
  launch token), and only then calls the provider's ``suspend``. An unknown
  outcome keeps admission closed until the same operation reports it.
* **Repair, not re-arm.** A live host process is never re-armed or restarted,
  because host shutdown ends its runners: it reconnects with its current token.
  When the provider reports a typed status, Omnigent re-arms and starts a host
  only after the provider grants a start claim, never on a status report alone.

Fence and admission state live in this process, which suits one server replica.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import dataclasses
import logging
import secrets
import time
import uuid
import weakref
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import click

from omnigent.db.utils import now_epoch
from omnigent.errors import ErrorCode, OmnigentError
from omnigent.onboarding.sandboxes.base import SandboxGoneError
from omnigent.onboarding.sandboxes.types import (
    ADMISSION_NEEDS_SIGN_IN,
    AdmissionReport,
    StartClaim,
    SuspendResult,
)
from omnigent.server.managed_hosts import (
    _launcher_for_teardown,
    _resume_locks,
    _start_sandbox_host,
    _terminate_sandbox_best_effort,
    managed_host_name,
    terminate_managed_host,
)

if TYPE_CHECKING:
    from omnigent.onboarding.sandboxes import SandboxHostLauncher
    from omnigent.server.host_registry import HostConnection, HostRegistry
    from omnigent.server.managed_hosts import (
        ManagedSandboxConfig,
        ManagedSandboxDeployment,
        PersistentHostPolicy,
    )
    from omnigent.stores.host_store import Host, HostStore

_logger = logging.getLogger(__name__)

PERSISTENT = "persistent"

# How long a host operation outside readiness (e.g. a runner launch) waits for
# a sleep in progress before giving up.
ADMISSION_WAIT_S = 120.0
# How long the fence waits for admitted operations to finish before it abandons
# the sleep. Never a reason to proceed: only finished operations are.
FENCE_DRAIN_S = 60.0
# A start whose reply was lost is presumed undelivered after this long in
# ``start_claimed``, and the next claim supersedes it.
START_REDELIVERY_S = 15.0
_QUIESCE_TIMEOUT_S = 10.0
_STATUS_POLL_S = 2.0
_UNKNOWN_RETRY_S = 1.0
_POLL_S = 0.2

WakeProgress = Callable[[str, "str | None"], None]
"""Observer of a wake's progress: ``(stage, error)``, using the session launch stages."""

_host_store: HostStore | None = None
_host_registry: HostRegistry | None = None
_deployment: ManagedSandboxDeployment | None = None

# Admission handles held by the current task (and tasks it starts), by host id.
_held: contextvars.ContextVar[dict[str, str] | None] = contextvars.ContextVar(
    "persistent_host_admissions", default=None
)


@dataclass
class _HostLifecycle:
    """Per-host lifecycle state on this replica."""

    admitting: asyncio.Event = field(default_factory=asyncio.Event)
    closed_reason: str | None = None
    unknown_suspend_operation: str | None = None
    handles: dict[str, str] = field(default_factory=dict)
    report: AdmissionReport | None = None
    # Unix time of the provider reply behind ``report``.
    report_at: float = 0.0
    wake_task: asyncio.Task[None] | None = None
    wake_stage: str | None = None
    observers: set[WakeProgress] = field(default_factory=set)
    idle_since: float | None = None
    # The sleep-intent write in flight, kept so a cancelled fence can learn its result.
    pending_intent: asyncio.Future[bool] | None = None
    # The shared resolution of an unknown sleep that waiting work waits on.
    recovery_task: asyncio.Task[str] | None = None
    # Bumped whenever the sandbox stops or a new one comes up, so a provider
    # reply requested before that never counts for the new generation.
    epoch: int = 0
    # Set when the host is deleted, so work still waiting on it is refused.
    removed: bool = False

    def __post_init__(self) -> None:
        self.admitting.set()

    def admitted(self) -> list[str]:
        """Operations currently admitted on the host."""
        return list(self.handles.values())

    def waking(self) -> bool:
        """Whether a wake or first start is in flight on this replica."""
        return self.wake_task is not None and not self.wake_task.done()

    def new_generation(self) -> None:
        """Forget the last admission report and every reply still in flight."""
        self.report = None
        self.epoch += 1

    def publish(self, stage: str, error: str | None = None) -> None:
        """Report a wake stage to every observer waiting on this host."""
        self.wake_stage = stage
        for observer in list(self.observers):
            try:
                observer(stage, error)
            except Exception:  # noqa: BLE001 — progress reporting never breaks a wake.
                _logger.warning("wake progress observer failed", exc_info=True)


# custom-lint: disable-next=workspace-scoped-cache -- host_id state; one per persistent host
_states: dict[str, _HostLifecycle] = {}
# Every connection a request was sent on, superseded ones included, by host and identity.
# custom-lint: disable-next=workspace-scoped-cache -- host_id key; weak values die with the tunnel
_connections: dict[str, weakref.WeakValueDictionary[int, HostConnection]] = {}


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
        and bool(_deployment.persistent_providers())
    )


def is_persistent_host(host: Host) -> bool:
    """Whether *host* is a persistent managed host."""
    return host.lifecycle == PERSISTENT


def _entry_for(host: Host) -> ManagedSandboxConfig | None:
    """The provider entry governing a persistent host, if it is still configured."""
    if _deployment is None or host.sandbox_provider is None:
        return None
    entry = _deployment.for_provider(host.sandbox_provider)
    return entry if entry is not None and entry.persistent is not None else None


def policy_for(host: Host) -> PersistentHostPolicy | None:
    """Return the persistent policy governing *host*, or ``None`` for other hosts."""
    if not is_persistent_host(host):
        return None
    entry = _entry_for(host)
    return entry.persistent if entry is not None else None


def _state(host_id: str) -> _HostLifecycle:
    return _states.setdefault(host_id, _HostLifecycle())


def _lifecycle_lock(host_id: str) -> asyncio.Lock:
    """The per-host lock every wake and sleep of *host_id* takes."""
    return _resume_locks.setdefault(host_id, asyncio.Lock())


def forget(host_id: str) -> None:
    """Drop lifecycle state for a deleted host.

    A wake in flight is left to finish: it fails on the deleted row, and a
    sandbox it was provisioning is removed rather than lost.
    """
    _connections.pop(host_id, None)
    state = _states.pop(host_id, None)
    if state is not None:
        state.removed = True


def manage_url(host: Host) -> str | None:
    """The provider's setup page for *host*, from the policy's ``manage_url`` template."""
    policy = policy_for(host)
    template = policy.manage_url if policy is not None else None
    if template is None:
        return None
    if "{sandbox_id}" in template and host.sandbox_id is None:
        return None
    return template.replace("{host_id}", host.host_id).replace(
        "{sandbox_id}", host.sandbox_id or ""
    )


def persistent_status(host: Host, *, online: bool) -> dict[str, Any] | None:
    """Picker state for a persistent host, or ``None`` for other hosts.

    :param host: The host row.
    :param online: Whether the host is live (online with a fresh heartbeat).
    :returns: ``state`` (``awake``, ``asleep``, ``waking``, ``sleeping`` or
        ``failed``), ``wake_stage``, ``failure``, the last ``admission`` report,
        ``manage_url`` and ``provider``.
    """
    if not is_persistent_host(host):
        return None
    state = _states.get(host.host_id)
    failure: str | None = None
    if state is not None and state.waking():
        status = "waking"
    elif state is not None and not state.admitting.is_set():
        status = "sleeping"
    elif online:
        status = "awake"
    elif host.start_failure is not None or host.sandbox_id is None:
        status = "failed"
        failure = host.start_failure or "Setup did not finish. Start it again."
    else:
        status = "asleep"
    report = state.report if state is not None else None
    return {
        "state": status,
        "wake_stage": state.wake_stage if state is not None and status == "waking" else None,
        "failure": failure,
        "admission": (
            {"admitting": report.admitting, "reason": report.reason, "message": report.message}
            if report is not None
            else None
        ),
        "manage_url": manage_url(host),
        "provider": host.sandbox_provider,
    }


async def _persistent_host(host_id: str, host: Host | None = None) -> Host | None:
    """The persistent host *host_id*, reusing *host* when the caller already loaded it."""
    if not enabled():
        return None
    if host is None:
        assert _host_store is not None
        host = await asyncio.to_thread(_host_store.get_host, host_id)
    return host if host is not None and is_persistent_host(host) else None


def _unavailable(message: str) -> OmnigentError:
    return OmnigentError(message, code=ErrorCode.RUNNER_UNAVAILABLE)


def _not_admitting(host: Host, report: AdmissionReport) -> OmnigentError:
    detail = report.message or report.reason or "no reason given"
    if report.reason == ADMISSION_NEEDS_SIGN_IN:
        link = manage_url(host)
        where = f" at {link}" if link else ""
        message = f"{host.name} needs you to sign in{where} before it can take new work"
        if report.message:
            message = f"{message}: {report.message}"
    else:
        message = f"{host.name} is not accepting new work: {detail}"
    return OmnigentError(message, code=ErrorCode.HOST_NOT_ADMITTING)


def _owns(host_id: str, state: _HostLifecycle) -> bool:
    """Whether the current task holds a live admission handle for *host_id*."""
    handle = (_held.get() or {}).get(host_id)
    return handle is not None and handle in state.handles


def _grant(host_id: str, state: _HostLifecycle, operation: str, *, renews_idle: bool) -> None:
    """Admit *operation* for the current task until the task finishes.

    Callers check admission with no await between that check and this call,
    which makes the pair atomic with the sleep fence's close.
    """
    handle = secrets.token_hex(8)
    state.handles[handle] = operation
    _held.set({**(_held.get() or {}), host_id: handle})
    if renews_idle:
        state.idle_since = None
    task = asyncio.current_task()
    if task is not None:
        task.add_done_callback(lambda _task: state.handles.pop(handle, None))


async def ensure_host_ready(
    host_id: str,
    *,
    operation: str,
    timeout_s: float,
    owner: str | None = None,
    wake: bool = True,
    host: Host | None = None,
    progress: WakeProgress | None = None,
    renews_idle: bool = True,
) -> None:
    """Make a persistent host ready for one new operation, within a deadline.

    No-op for hosts that are not persistent, and for a task already holding an
    admission for the host. Waits while a sleep is in progress, wakes an asleep
    host (one physical wake per host, shared by all waiters), re-checks
    authorization after any wait, and admits the operation only while the
    provider reports the host admitting.

    :param host_id: Host the operation targets.
    :param operation: What the work is, e.g. ``"message"``, for logs and the fence.
    :param timeout_s: Deadline for the whole wait, in seconds.
    :param owner: Caller the host must still belong to, or ``None`` to skip.
    :param wake: ``False`` refuses instead of waking an asleep host.
    :param host: The host row when the caller already loaded it.
    :param progress: Observer told the wake's stages, then ``"ready"`` or
        ``"failed"`` when this call waited on a wake.
    :param renews_idle: ``False`` for reads, which never keep a host awake.
    :raises OmnigentError: ``HOST_NOT_ADMITTING`` with the provider's reason;
        ``RUNNER_UNAVAILABLE`` when the host cannot be made ready in time,
        including "computer status unknown"; ``NOT_FOUND`` when the host was
        deleted during the wait; ``FORBIDDEN`` when it changed owner.
    """
    found = await _persistent_host(host_id, host)
    if found is None:
        return
    assert _host_store is not None and _host_registry is not None
    state = _state(host_id)
    if _owns(host_id, state):
        # Part of an admitted operation; this task takes its own handle, so the
        # operation stays admitted until this task finishes too.
        _grant(host_id, state, operation, renews_idle=renews_idle)
        return
    deadline = time.monotonic() + timeout_s
    observed: list[str] = []

    def observe(stage: str, error: str | None) -> None:
        observed.append(stage)
        if progress is not None:
            progress(stage, error)

    state.observers.add(observe)
    try:
        await _until_admitted(
            found, state, operation, deadline, timeout_s, wake, owner, renews_idle
        )
    except OmnigentError as exc:
        if observed and progress is not None:
            progress("failed", str(exc))
        raise
    finally:
        state.observers.discard(observe)
    if observed and progress is not None:
        progress("ready", None)


async def _until_admitted(
    host: Host,
    state: _HostLifecycle,
    operation: str,
    deadline: float,
    timeout_s: float,
    wake: bool,
    owner: str | None,
    renews_idle: bool,
) -> None:
    """The readiness loop: wait, wake, re-authorize and admit."""
    assert _host_store is not None and _host_registry is not None
    host_id = host.host_id
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
            try:
                report = await asyncio.wait_for(
                    _admission(host, state), max(deadline - time.monotonic(), 0)
                )
            except TimeoutError:
                report = None
            if report is None:
                # Unknown is never a refusal: ask again while the work waits.
                if time.monotonic() >= deadline:
                    raise _unavailable(f"{host.name}: computer status unknown; try again shortly")
                await asyncio.sleep(min(_UNKNOWN_RETRY_S, max(deadline - time.monotonic(), 0)))
                continue
            if not report.admitting:
                raise _not_admitting(host, report)
            if time.monotonic() >= deadline:
                raise _unavailable(f"{host.name} did not become ready within {timeout_s:.0f}s")
            if state.removed:
                raise OmnigentError(f"host {host.name!r} was removed", code=ErrorCode.NOT_FOUND)
            # No await from here to the grant: a sleep that closed admission
            # while this task awaited is seen, and the loop waits for it.
            if state.admitting.is_set() and _host_registry.get(host_id) is not None:
                _grant(host_id, state, operation, renews_idle=renews_idle)
                return
            continue
        if not wake:
            raise _unavailable(f"{host.name} is asleep; send the request again to wake it")
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


def session_progress(session_id: str) -> WakeProgress:
    """Report a wake to a session's existing launch-progress surface.

    :param session_id: The session whose request waits on the wake.
    :returns: An observer for :func:`ensure_host_ready`'s ``progress``.
    """
    from omnigent.server.routes._sessions.helpers import _publish_sandbox_status

    def publish(stage: str, error: str | None) -> None:
        _publish_sandbox_status(session_id, stage, error)

    return publish


async def admit_host_work(host_id: str, operation: str) -> None:
    """Admission gate for host operations such as runner launches and installs.

    Part of an operation the current task already holds an admission for: the
    handle extends to this task. Otherwise the operation is admitted on its
    own, waiting out a sleep in progress but never waking the host.

    :raises OmnigentError: When the host went to sleep or does not admit work.
    """
    host = await _persistent_host(host_id)
    if host is None:
        return
    await ensure_host_ready(
        host_id, operation=operation, timeout_s=ADMISSION_WAIT_S, wake=False, host=host
    )


async def wake_if_persistent(host_id: str, host_store: HostStore) -> bool:
    """Wake *host_id* through its lifecycle if its row is persistent.

    Used by the generic managed-host wake so every wake of a persistent host
    takes the same single-flight, repair-not-re-arm path. The row decides,
    not the configuration: a persistent host whose provider is no longer
    configured as persistent is refused, never woken as a disposable sandbox.

    :param host_id: The host to wake.
    :param host_store: The caller's host store.
    :returns: ``False`` when the host is not persistent (nothing was done).
    :raises OmnigentError: When the wake fails; never ``SandboxGoneError``, so
        no caller can answer it by provisioning a blank sandbox.
    """
    host = await asyncio.to_thread(host_store.get_host, host_id)
    if host is None or not is_persistent_host(host):
        return False
    entry = _entry_for(host)
    if entry is None or not enabled():
        raise _unavailable(
            f"{host.name} is a persistent host, but its provider is no longer configured "
            "for persistent hosts"
        )
    await ensure_host_ready(
        host_id, operation="wake", timeout_s=entry.host_online_timeout_s, host=host
    )
    return True


async def _wait_admitting(state: _HostLifecycle, host: Host, deadline: float) -> None:
    """Hold work while a sleep is in progress, resolving an unknown outcome."""
    while not state.admitting.is_set():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise _unavailable(
                f"{host.name} is {state.closed_reason or 'not admitting'}; try again shortly"
            )
        if state.unknown_suspend_operation is not None:
            # A wake waits for the earlier sleep's outcome; ask for it again,
            # once for every waiter, and only as long as this waiter may wait.
            recovery = _recovery_task(host.host_id, state, state.unknown_suspend_operation)
            try:
                await asyncio.wait_for(asyncio.shield(recovery), remaining)
            except TimeoutError:
                pass
            except Exception:  # noqa: BLE001 — admission stays closed; it is asked again.
                _logger.warning(
                    "sleep outcome for host %s unresolved", host.host_id, exc_info=True
                )
            if not state.admitting.is_set():
                await asyncio.sleep(min(1.0, max(deadline - time.monotonic(), 0)))
            continue
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(state.admitting.wait(), remaining)


def _recovery_task(host_id: str, state: _HostLifecycle, operation_id: str) -> asyncio.Task[str]:
    """The shared repeat of an unknown sleep, started if none is running."""
    if state.recovery_task is None or state.recovery_task.done():
        state.recovery_task = asyncio.create_task(
            sleep_if_idle(host_id, only_operation=operation_id), name=f"sleep-outcome-{host_id}"
        )
        state.recovery_task.add_done_callback(_log_background_failure)
    return state.recovery_task


def _launcher(host: Host) -> SandboxHostLauncher | None:
    return _launcher_for_teardown(host, _deployment)


async def _admission(host: Host, state: _HostLifecycle) -> AdmissionReport | None:
    """The provider's admission report, at most the policy's maximum age old.

    :returns: The report, or ``None`` when the provider could not answer. An
        unanswered refresh is never cached, so the next call asks again.
    """
    policy = policy_for(host)
    launcher = _launcher(host)
    if policy is None or launcher is None or not launcher.capabilities.admission:
        return AdmissionReport(admitting=True)
    if state.report is not None and time.time() - state.report_at <= policy.admission_max_age_s:
        return state.report
    try:
        return await _fresh_report(host, launcher, state, max_age_s=policy.admission_max_age_s)
    except SandboxGoneError:
        raise _unavailable(f"{host.name} no longer exists on {host.sandbox_provider}") from None


async def _fresh_report(
    host: Host, launcher: SandboxHostLauncher, state: _HostLifecycle, *, max_age_s: float
) -> AdmissionReport | None:
    """Ask the provider for a new report.

    :param max_age_s: Oldest provider reply accepted; an older one is unknown.
    :returns: The report, or ``None`` when the provider could not answer or
        answered from a reply that is already too old. Neither is cached.
    """
    if host.sandbox_id is None:
        return None
    epoch = state.epoch
    try:
        report = await asyncio.to_thread(launcher.admission, host.sandbox_id)
    except SandboxGoneError:
        raise
    except Exception:  # noqa: BLE001 — an unanswered refresh is "unknown", never a refusal.
        _logger.warning("admission report for host %s failed", host.host_id, exc_info=True)
        return None
    if report is None or state.epoch != epoch:
        # Unknown, or asked before the sandbox stopped or came up again.
        return None
    observed_at = report.observed_at if report.observed_at is not None else time.time()
    if time.time() - observed_at > max_age_s:
        _logger.warning("admission report for host %s is too old to use", host.host_id)
        return None
    state.report, state.report_at = report, observed_at
    return report


def _wake_task(
    host_id: str, state: _HostLifecycle, *, provision: bool = False
) -> asyncio.Task[None]:
    """The host's in-flight wake, started if none is running."""
    if not state.waking():
        state.wake_stage = None
        state.wake_task = asyncio.create_task(
            _wake(host_id, state, provision=provision), name=f"wake-host-{host_id}"
        )
        state.wake_task.add_done_callback(_log_background_failure)
    assert state.wake_task is not None
    return state.wake_task


def _log_background_failure(task: asyncio.Task[Any]) -> None:
    """Retrieve a shared task's failure, which its waiters report and a first start records."""
    if not task.cancelled() and task.exception() is not None:
        _logger.info("%s failed: %s", task.get_name(), task.exception())


def start_host_now(host_id: str) -> asyncio.Task[None]:
    """Start or retry a persistent host's start in the background ("Start again").

    Provisions its sandbox first when it has none.

    :returns: The shared wake task, which callers may await.
    """
    return _wake_task(host_id, _state(host_id), provision=True)


async def _wake(host_id: str, state: _HostLifecycle, *, provision: bool) -> None:
    """Wake one persistent host under its lifecycle lock.

    :param provision: Provision a sandbox for a host that has none (its first
        start, or "Start again" after a failed one). Failures of such a start
        are recorded on the host, so it shows as failed with the reason.
    """
    assert _host_store is not None and _host_registry is not None
    pending = await asyncio.to_thread(_host_store.get_host, host_id)
    if pending is None:
        raise _unavailable(f"host {host_id} no longer exists")
    pending_entry = _entry_for(pending)
    online_wait_s = (
        pending_entry.host_online_timeout_s if pending_entry is not None else ADMISSION_WAIT_S
    )
    # A sleep whose outcome is unknown resolves before anything is resumed or re-armed.
    await _wait_admitting(state, pending, time.monotonic() + online_wait_s)
    async with _lifecycle_lock(host_id):
        if _host_registry.get(host_id) is not None:
            return
        host = await asyncio.to_thread(_host_store.get_host, host_id)
        if host is None:
            raise _unavailable(f"host {host_id} no longer exists")
        entry = _entry_for(host)
        launcher = _launcher(host)
        if entry is None or launcher is None:
            raise _unavailable(f"{host.name}'s provider is no longer configured")
        if host.sandbox_id is None and not provision:
            raise _unavailable(
                f"{host.name} is not set up: {host.start_failure or 'its setup did not finish'}. "
                "Start it again."
            )
        try:
            if host.sandbox_id is None:
                attached = await _provision(host, launcher, state)
                if attached is None:
                    return
                host = attached
            await _bring_up(host, launcher, entry, state)
        except SandboxGoneError:
            failure = (
                f"{host.name} no longer exists on {host.sandbox_provider}. "
                "Delete it and add a new computer."
            )
            await asyncio.to_thread(_host_store.record_start_failure, host_id, failure)
            raise _unavailable(failure) from None
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if not provision:
                raise
            failure = _failure_message(exc)
            await asyncio.to_thread(_host_store.record_start_failure, host_id, failure)
            raise _unavailable(f"{host.name} could not start: {failure}") from exc
        if host.start_failure is not None:
            await asyncio.to_thread(_host_store.record_start_failure, host_id, None)


def _failure_message(exc: BaseException) -> str:
    if isinstance(exc, click.ClickException):
        return exc.message
    detail = getattr(exc, "detail", None)
    return str(detail) if detail else str(exc) or type(exc).__name__


async def _provision(
    host: Host, launcher: SandboxHostLauncher, state: _HostLifecycle
) -> Host | None:
    """Provision the sandbox for a pending persistent host.

    The provider is asked under the host's stable name
    (:func:`~omnigent.server.managed_hosts.managed_host_name`), so a provider
    that provisions idempotently by name returns the same sandbox when a start
    is retried after a crash between its reply and the row update.

    :returns: The host with its sandbox recorded, or ``None`` when the host
        was deleted meanwhile; its new sandbox is then removed, and kept as a
        tombstone the reaper retries if that removal fails.
    """
    assert _host_store is not None
    state.publish("provisioning")
    await asyncio.to_thread(launcher.prepare_for_launch, agent_name=None)
    await asyncio.to_thread(launcher.prepare)
    sandbox_id = await asyncio.to_thread(launcher.provision, managed_host_name(host.host_id))
    attached = await asyncio.to_thread(
        _host_store.attach_persistent_sandbox, host.host_id, sandbox_id=sandbox_id
    )
    if attached is None and not await _terminate_sandbox_best_effort(
        launcher, sandbox_id, host_id=host.host_id, provider=launcher.provider
    ):
        await asyncio.to_thread(
            _host_store.retain_orphan_sandbox,
            host_id=host.host_id,
            user_id=host.user_id,
            provider=launcher.provider,
            sandbox_id=sandbox_id,
        )
    return attached


async def _online(host_id: str) -> bool:
    assert _host_store is not None and _host_registry is not None
    if _host_registry.get(host_id) is not None:
        return True
    return await asyncio.to_thread(_host_store.is_online, host_id)


async def _await_online(host_id: str, deadline: float, *, at_most_s: float | None = None) -> bool:
    """Poll until the host is online, the deadline passes or *at_most_s* elapses."""
    end = deadline if at_most_s is None else min(deadline, time.monotonic() + at_most_s)
    while True:
        if await _online(host_id):
            return True
        if time.monotonic() >= end:
            return False
        await asyncio.sleep(min(_POLL_S * 2.5, max(end - time.monotonic(), 0)))


async def _bring_up(
    host: Host,
    launcher: SandboxHostLauncher,
    entry: ManagedSandboxConfig,
    state: _HostLifecycle,
) -> None:
    """Bring a persistent host online, within the provider's online wait."""
    timeout_s = entry.host_online_timeout_s
    deadline = time.monotonic() + timeout_s
    if launcher.capabilities.admission:
        await _bring_up_typed(host, launcher, entry, state, deadline)
    else:
        await _bring_up_untyped(host, launcher, entry, state, deadline)
    # A new generation starts with a fresh admission report.
    state.new_generation()


async def _bring_up_untyped(
    host: Host,
    launcher: SandboxHostLauncher,
    entry: ManagedSandboxConfig,
    state: _HostLifecycle,
    deadline: float,
) -> None:
    """Wake through ``is_running`` for a provider without a typed status.

    A running sandbox whose host was started before is repaired, never
    re-armed: its host reconnects with its current token. A stopped sandbox
    is resumed and started anew, as is a running one whose host never started
    (its first start). An unknown status (``None``) is asked again within the
    deadline, never taken as stopped.
    """
    assert host.sandbox_id is not None
    while (running := await asyncio.to_thread(launcher.is_running, host.sandbox_id)) is None:
        if await _await_online(host.host_id, deadline, at_most_s=_STATUS_POLL_S):
            return
        if time.monotonic() >= deadline:
            raise _unavailable(f"{host.name}: computer status unknown; try again shortly")
    if running and host.token_expires_at is not None:
        _logger.info("Host %s sandbox is running; waiting for it to reconnect", host.host_id)
        state.publish("connecting")
        if not await _await_online(host.host_id, deadline):
            raise _unavailable(f"{host.name} is running but did not reconnect")
        return
    if not running:
        state.publish("provisioning")
        await _call_launcher(host, launcher.resume, host.sandbox_id)
    claim = await _claim_start(host, launcher, uuid.uuid4().hex, deadline)
    if claim.granted:
        current = await asyncio.to_thread(_require_host, host.host_id)
        await _arm_and_start(current, launcher, entry, state, attempt=claim.attempt)
    if not await _await_online(host.host_id, deadline):
        raise _unavailable(
            f"{host.name} did not come online within {entry.host_online_timeout_s:g}s"
        )


async def _bring_up_typed(
    host: Host,
    launcher: SandboxHostLauncher,
    entry: ManagedSandboxConfig,
    state: _HostLifecycle,
    deadline: float,
) -> None:
    """Wake by the provider's typed status, re-arming only through a granted claim.

    ``host_running`` and ``stopping`` are waited out; ``stopped`` is resumed;
    ``started_no_host``, ``start_claimed`` and ``host_refused`` are claimed,
    re-armed and started. A claim refused because a host process took over
    means its host reconnects on its own.
    """
    assert host.sandbox_id is not None
    # Statuses in which our own recent start is still on its way to the host.
    # Without claims a superseded start cannot be told apart, so both count.
    pending: tuple[str, ...] = (
        ("start_claimed",)
        if launcher.capabilities.start_claims
        else ("start_claimed", "started_no_host")
    )
    assert entry.persistent is not None
    max_age_s = entry.persistent.admission_max_age_s
    claim_operation = uuid.uuid4().hex
    started_at: float | None = None
    start_delivered = False
    while True:
        if await _online(host.host_id):
            return
        if time.monotonic() >= deadline:
            raise _unavailable(
                f"{host.name} did not come online within {entry.host_online_timeout_s:g}s"
            )
        report = await _fresh_report(host, launcher, state, max_age_s=max_age_s)
        status = report.status if report is not None else None
        if status is None or status in ("host_running", "stopping"):
            # Unknown, a live host reconnecting, or a stop in progress: wait.
            await _await_online(host.host_id, deadline, at_most_s=_STATUS_POLL_S)
            continue
        if status == "stopped":
            state.publish("provisioning")
            await _call_launcher(host, launcher.resume, host.sandbox_id)
            await _await_online(host.host_id, deadline, at_most_s=_STATUS_POLL_S)
            continue
        if status in pending and started_at is not None:
            if start_delivered or time.monotonic() - started_at < START_REDELIVERY_S:
                # Our attempt holds the sandbox; its host takes the parameters soon.
                await _await_online(host.host_id, deadline, at_most_s=_STATUS_POLL_S)
                continue
        claim = await _claim_start(host, launcher, claim_operation, deadline)
        # The answer is recorded under this operation; the next claim is a new one.
        claim_operation = uuid.uuid4().hex
        if not claim.granted:
            _logger.info(
                "Start claim for host %s refused (status %s); not re-arming",
                host.host_id,
                claim.status,
            )
            await _await_online(host.host_id, deadline, at_most_s=_STATUS_POLL_S)
            continue
        current = await asyncio.to_thread(_require_host, host.host_id)
        start_delivered = await _arm_and_start(
            current, launcher, entry, state, attempt=claim.attempt
        )
        started_at = time.monotonic()


def _require_host(host_id: str) -> Host:
    assert _host_store is not None
    host = _host_store.get_host(host_id)
    if host is None:
        raise _unavailable(f"host {host_id} was removed while waking")
    return host


async def _call_launcher(host: Host, method: Callable[..., Any], *args: Any) -> None:
    """Call a launcher step whose refusal is final and whose lost reply is retried.

    :raises OmnigentError: With the provider's readable message on a refusal.
    """
    try:
        await asyncio.to_thread(method, *args)
    except click.ClickException as exc:
        raise _unavailable(f"{host.name} could not wake: {exc.message}") from exc
    except SandboxGoneError:
        raise
    except Exception:  # noqa: BLE001 — a lost reply: the status loop finds out.
        _logger.warning("launcher call for host %s failed", host.host_id, exc_info=True)


async def _claim_start(
    host: Host, launcher: SandboxHostLauncher, operation_id: str, deadline: float
) -> StartClaim:
    """Claim a start attempt, repeating a lost reply with the same operation id."""
    assert host.sandbox_id is not None
    if not launcher.capabilities.start_claims:
        return StartClaim(granted=True)
    while True:
        try:
            return await asyncio.to_thread(
                launcher.claim_start, host.sandbox_id, operation_id=operation_id
            )
        except click.ClickException as exc:
            raise _unavailable(f"{host.name} could not start: {exc.message}") from exc
        except SandboxGoneError:
            raise
        except Exception:  # noqa: BLE001 — a lost reply; the same operation returns its outcome.
            _logger.warning("start claim for host %s failed", host.host_id, exc_info=True)
        if time.monotonic() >= deadline:
            raise _unavailable(f"{host.name}: computer status unknown; try again shortly")
        await asyncio.sleep(min(_UNKNOWN_RETRY_S, max(deadline - time.monotonic(), 0)))


async def _arm_and_start(
    host: Host,
    launcher: SandboxHostLauncher,
    entry: ManagedSandboxConfig,
    state: _HostLifecycle,
    *,
    attempt: int | None,
) -> bool:
    """Arm a fresh launch token and start the host for one attempt.

    :returns: ``True`` when the provider confirmed the start, ``False`` when
        its reply was lost.
    :raises OmnigentError: When the host changed meanwhile or the provider
        refused the start.
    """
    assert _host_store is not None and host.sandbox_id is not None
    token = secrets.token_urlsafe(32)
    armed = await asyncio.to_thread(
        _host_store.rearm_managed_host,
        host.host_id,
        sandbox_id=host.sandbox_id,
        expected_updated_at=host.updated_at,
        token=token,
        token_expires_at=now_epoch() + entry.token_ttl_s,
    )
    if armed is None:
        raise _unavailable(f"{host.name} changed while waking; try again")
    state.publish("starting")
    try:
        await _start_sandbox_host(
            launcher,
            host.sandbox_id,
            token=token,
            host_id=host.host_id,
            host_name=host.name,
            server_url=entry.server_url,
            repos=(),
            host_config=entry.host_config,
            start_attempt=attempt,
        )
        delivered = True
    except click.ClickException as exc:
        raise _unavailable(f"{host.name} could not start: {exc.message}") from exc
    except SandboxGoneError:
        raise
    except Exception:  # noqa: BLE001 — a lost reply; the status shows whether it arrived.
        _logger.warning("start_host for host %s failed", host.host_id, exc_info=True)
        delivered = False
    state.publish("connecting")
    return delivered


def guard_host_request(conn: HostConnection) -> None:
    """Refuse a server-to-host request while the host's sleep fence is closed.

    Installed as the host registry's send guard, so it covers every direct
    host operation. Requests from an admitted operation still pass: the fence
    waits for them.

    :raises ConnectionError: When the host is going to sleep.
    """
    _connections.setdefault(conn.host_id, weakref.WeakValueDictionary())[id(conn)] = conn
    state = _states.get(conn.host_id)
    if state is not None and not state.admitting.is_set() and not _owns(conn.host_id, state):
        raise ConnectionError(f"host {conn.host_id!r} is going to sleep")


def _pending_requests(conn: HostConnection) -> int:
    """Server-to-host requests awaiting a reply, on this and superseded connections."""
    total = 0
    known = _connections.get(conn.host_id, {})
    for connection in {**known, id(conn): conn}.values():
        total += sum(
            len(getattr(connection, item.name))
            for item in dataclasses.fields(connection)
            if item.name.startswith(("pending_", "inflight_"))
            and item.name != "pending_quiesce"
            and isinstance(getattr(connection, item.name), dict)
        )
    return total


@dataclass(frozen=True)
class QuiesceReport:
    """What a host reports as live: runners, runner launches and host operations."""

    runners: list[str]
    launching: int
    operations: list[str]


async def quiesce_host(conn: HostConnection) -> QuiesceReport | None:
    """Ask a connected host whether any runner or host operation is live.

    :returns: The host's report, or ``None`` when it cannot answer (an older
        host, a timeout or a dropped connection).
    """
    from omnigent.host.frames import (
        CAP_QUIESCE,
        HostQuiesceFrame,
        HostQuiesceResultFrame,
        encode_host_frame,
    )

    assert _host_registry is not None
    if CAP_QUIESCE not in conn.hello.capabilities:
        return None
    request_id = secrets.token_hex(8)
    future: asyncio.Future[HostQuiesceResultFrame] = asyncio.get_running_loop().create_future()
    conn.pending_quiesce[request_id] = future
    try:
        # The fence's own question bypasses the send guard it has just closed.
        if _host_registry.get(conn.host_id) is not conn:
            return None
        conn.enqueue(encode_host_frame(HostQuiesceFrame(request_id)))
        result = await asyncio.wait_for(future, _QUIESCE_TIMEOUT_S)
    except (ConnectionError, TimeoutError):
        return None
    finally:
        conn.pending_quiesce.pop(request_id, None)
    return QuiesceReport(
        runners=list(result.runners),
        launching=result.launching,
        operations=list(result.operations),
    )


def _busy_reason(
    state: _HostLifecycle, conn: HostConnection, report: QuiesceReport | None
) -> str | None:
    """Why the host must not sleep now, or ``None`` when it is quiescent."""
    if report is None:
        return "host cannot report quiescence"
    if report.runners or report.launching or report.operations:
        return (
            f"runners {report.runners}, launching {report.launching}, "
            f"host operations {report.operations}"
        )
    pending = _pending_requests(conn)
    if pending:
        return f"{pending} host request(s) in flight"
    admitted = state.admitted()
    return f"admitted work {admitted}" if admitted else None


async def _call_suspend(
    launcher: SandboxHostLauncher, sandbox_id: str, operation_id: str
) -> SuspendResult:
    try:
        return await asyncio.to_thread(launcher.suspend, sandbox_id, operation_id=operation_id)
    except Exception:  # noqa: BLE001 — a lost reply means the stop may have begun.
        _logger.warning("suspend %s of sandbox %s failed", operation_id, sandbox_id, exc_info=True)
        return SuspendResult(outcome="unknown")


async def sleep_if_idle(host_id: str, *, only_operation: str | None = None) -> str:
    """Run the fenced sleep for one persistent host, or resolve an unknown one.

    :param only_operation: Resolve this unknown operation only; never start a
        new sleep. Used by work waiting on a sleep's outcome.
    :returns: ``"abandoned"`` (work exists, the host cannot answer, or there is
        nothing to resolve), ``"refused"``, ``"stopped"`` or ``"unknown"``.
    """
    host = await _persistent_host(host_id)
    if host is None or host.sandbox_id is None:
        return "abandoned"
    assert _host_store is not None and _host_registry is not None
    launcher = _launcher(host)
    if launcher is None or not launcher.capabilities.suspend:
        return "abandoned"
    state = _state(host_id)
    outcome = "abandoned"
    async with _lifecycle_lock(host_id):
        operation_id = state.unknown_suspend_operation
        if only_operation is not None and operation_id != only_operation:
            return outcome
        conn = _host_registry.get(host_id)
        if operation_id is None and conn is None:
            return outcome
        if operation_id is not None:
            # Resolving an earlier sleep: until its outcome is learned and
            # recorded, any failure here keeps admission closed.
            outcome = "unknown"
        state.admitting.clear()
        state.closed_reason = "going to sleep"
        try:
            if operation_id is None:
                assert conn is not None
                # Admitted operations and host requests finish first; if they do not, abandon.
                deadline = time.monotonic() + FENCE_DRAIN_S
                while state.handles or _pending_requests(conn):
                    if time.monotonic() >= deadline:
                        return outcome
                    await asyncio.sleep(_POLL_S)
                if _busy_reason(state, conn, await quiesce_host(conn)) is not None:
                    return outcome
                operation_id = uuid.uuid4().hex
                # From the intent write on, the outcome is unknown until learned,
                # even if this task is cancelled or the server restarts.
                outcome = "unknown"
                state.unknown_suspend_operation = operation_id
                state.closed_reason = "going to sleep (outcome unknown)"
                # Persist before the effect: the operation, which revokes the token.
                state.pending_intent = asyncio.ensure_future(
                    asyncio.to_thread(
                        _host_store.begin_host_sleep, host_id, operation_id=operation_id
                    )
                )
            recorded = await _sleep_intent_recorded(state, host_id, operation_id)
            if not recorded:
                # No intent is recorded: none was written, so nothing stopped and
                # the token is unchanged, or its outcome is already recorded.
                outcome = "abandoned"
                return outcome  # noqa: RET504 — ``finally`` reads the outcome.
            outcome = "unknown"
            state.unknown_suspend_operation = operation_id
            result = await _call_suspend(launcher, host.sandbox_id, operation_id)
            _logger.info(
                "Sleep %s of host %s: %s%s",
                operation_id,
                host_id,
                result.outcome,
                f" ({result.reason})" if result.reason else "",
            )
            if result.outcome != "unknown":
                await asyncio.to_thread(
                    _host_store.end_host_sleep,
                    host_id,
                    operation_id=operation_id,
                    stopped=result.outcome == "stopped",
                )
                if result.outcome == "stopped":
                    _host_registry.deregister(host_id)
                    state.new_generation()
            outcome = result.outcome
            return outcome  # noqa: RET504 — ``finally`` reads the outcome.
        finally:
            if outcome != "unknown":
                state.unknown_suspend_operation = None
                state.closed_reason = None
                state.idle_since = None
                state.admitting.set()


async def _sleep_intent_recorded(state: _HostLifecycle, host_id: str, operation_id: str) -> bool:
    """Whether the sleep intent *operation_id* is durably recorded for the host.

    Learns it from the intent write still in flight when there is one;
    otherwise, or when that write failed ambiguously, from the store. A
    failed or cancelled read raises, leaving the outcome unknown.
    """
    assert _host_store is not None
    if state.pending_intent is not None:
        try:
            recorded = await asyncio.shield(state.pending_intent)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — the write may have committed; the store knows.
            _logger.warning("sleep intent write for host %s failed", host_id, exc_info=True)
        else:
            state.pending_intent = None
            return recorded
        state.pending_intent = None
    current = await asyncio.to_thread(_host_store.get_host, host_id)
    return current is not None and current.sleep_operation_id == operation_id


async def recover_pending_sleeps() -> list[str]:
    """Close admission for hosts whose sleep outcome a previous process never learned.

    :returns: Host ids recovered; each resolves by repeating its operation.
    """
    if not enabled():
        return []
    assert _host_store is not None
    recovered = []
    for host_id, operation_id in await asyncio.to_thread(_host_store.list_pending_host_sleeps):
        state = _state(host_id)
        state.admitting.clear()
        state.unknown_suspend_operation = operation_id
        state.closed_reason = "going to sleep (outcome unknown)"
        recovered.append(host_id)
    return recovered


async def create_host(*, owner: str, name: str, provider: str | None) -> Host:
    """Create a persistent host and start it in the background.

    The pending row is written before the provider provisions anything; the
    first start then provisions, arms and starts the host. A failure leaves
    the host ``failed`` with its reason, never deleted.

    :param owner: The caller, e.g. ``"alice@example.com"``.
    :param name: Caller-chosen display name, e.g. ``"Work computer"``.
    :param provider: A persistent provider, or ``None`` for the first one.
    :returns: The pending host.
    :raises OmnigentError: ``INVALID_INPUT`` when no such provider exists;
        ``CONFLICT`` when the name is taken.
    """
    assert _host_store is not None
    offered = _deployment.persistent_providers() if _deployment is not None else ()
    if not offered or (provider is not None and provider not in offered):
        raise OmnigentError(
            f"no persistent host provider {provider!r} is configured"
            if provider is not None
            else "no persistent host provider is configured",
            code=ErrorCode.INVALID_INPUT,
        )
    try:
        host = await asyncio.to_thread(
            _host_store.create_persistent_host,
            host_id=uuid.uuid4().hex,
            name=name,
            user_id=owner,
            provider=provider or offered[0],
        )
    except ValueError as exc:
        raise OmnigentError(str(exc), code=ErrorCode.CONFLICT) from exc
    start_host_now(host.host_id)
    return host


async def delete_host(host: Host) -> None:
    """Delete a persistent host: tombstone it, then ask the provider to remove it.

    ``terminate`` means removal for a persistent host and is called only here,
    after account deletion, and by the reaper's retry of a tombstone.
    """
    assert _host_store is not None
    forget(host.host_id)
    await terminate_managed_host(host, _host_store, _deployment)


async def remove_deleted_account_hosts(user_id: str) -> None:
    """Remove the sandboxes of a deleted account's persistent hosts.

    Called after the account deletion commits; a failure is retried by the reaper.
    """
    if not enabled():
        return
    assert _host_store is not None
    for host in await asyncio.to_thread(_host_store.list_deleted_persistent_hosts, user_id):
        forget(host.host_id)
        await terminate_managed_host(host, _host_store, _deployment)


def _token_expiring(host: Host, policy: PersistentHostPolicy) -> bool:
    return (
        host.token_expires_at is not None
        and host.token_expires_at - now_epoch() <= policy.token_refresh_margin_s
    )


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
            recovered = await recover_pending_sleeps()
            if recovered:
                _logger.warning("Sleep outcome unknown after restart for hosts %s", recovered)
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
        """Check every connected persistent host once; returns each host's state.

        A host sleeps once it has been quiescent for the policy's idle window,
        or at its first quiescent moment when its launch token is near expiry.
        """
        assert _host_store is not None and _host_registry is not None
        outcomes: dict[str, str] = {}
        for host_id, state in list(_states.items()):
            if state.unknown_suspend_operation is not None:
                try:
                    outcomes[host_id] = await sleep_if_idle(
                        host_id, only_operation=state.unknown_suspend_operation
                    )
                except Exception:  # noqa: BLE001 — admission stays closed until the next sweep.
                    _logger.warning("sleep outcome for host %s unresolved", host_id, exc_info=True)
                    outcomes[host_id] = "unknown"
        for host_id in _host_registry.online_host_ids():
            host = await _persistent_host(host_id)
            conn = _host_registry.get(host_id)
            policy = policy_for(host) if host is not None else None
            if host is None or conn is None or policy is None or host_id in outcomes:
                continue
            state = _state(host_id)
            busy = _busy_reason(state, conn, await quiesce_host(conn))
            if busy is not None:
                state.idle_since = None
                outcomes[host_id] = f"busy: {busy}"
                continue
            now = time.monotonic()
            if state.idle_since is None:
                state.idle_since = now
            if _token_expiring(host, policy) or now - state.idle_since >= policy.idle_sleep_s:
                outcomes[host_id] = await sleep_if_idle(host_id)
            else:
                outcomes[host_id] = f"idle for {now - state.idle_since:.0f}s"
        return outcomes
