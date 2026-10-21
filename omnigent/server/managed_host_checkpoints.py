"""Ask managed sandboxes to checkpoint at turn boundaries.

A provider whose launcher implements
:meth:`~omnigent.onboarding.sandboxes.base.SandboxLifecycle.checkpoint` is asked
to save a managed sandbox when a turn on that host ends (``turn_completed``)
and, while a turn runs, whenever the provider's
:class:`~omnigent.server.managed_hosts.SandboxCheckpointPolicy` interval passes
without a request (``long_turn``). Losing a sandbox then costs the turn in
progress, or the last interval of a long one, plus the debounce below and the
provider's time to make a save durable, rather than everything since it was
last suspended.

Statuses reach :func:`note_status` from the server's status funnel
(``_publish_status``) and from the runner-recovered path that bypasses it, so
every harness is covered. The scheduler judges turn edges from its own record
of each session's turn, because other paths also write the funnel's status
cache. Terminal watchers publish a false ``idle`` when a pane goes quiet
mid-turn, so a turn end counts only once it has held for
:data:`TURN_END_DEBOUNCE_S` without a new ``running`` on that session.

Per host, one ``checkpoint`` call runs at a time from each server process:
requests made meanwhile fold into a single follow-up, and the policy's
``min_interval_s`` spaces calls further apart. Providers without the capability
are skipped and logged once, so a deployment where none implements it behaves
exactly as before. Everything is best effort and never disturbs the status path
that feeds it.

Scheduling state lives on the event loop and is per server replica. Store reads
and launcher calls run in worker threads inside the caller's ``contextvars``,
which carry the workspace scope every store query filters on.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import threading
import time
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from omnigent.db.db_models import current_workspace_id
from omnigent.onboarding.sandboxes.base import SandboxCapabilityError, SandboxGoneError
from omnigent.server.managed_hosts import _launcher_for_teardown

if TYPE_CHECKING:
    from omnigent.entities import Conversation
    from omnigent.onboarding.sandboxes.types import CheckpointReason
    from omnigent.server.managed_hosts import ManagedSandboxDeployment, SandboxCheckpointPolicy
    from omnigent.stores.conversation_store import ConversationStore
    from omnigent.stores.host_store import HostStore

_logger = logging.getLogger(__name__)

# Well above the terminal watchers' one-second quiet threshold. A longer
# mid-turn lull costs one extra save, never a missed one.
TURN_END_DEBOUNCE_S: float = 10.0

# How often a running turn retries finding its host after a store failure.
_RESOLVE_RETRY_S = 60.0

_BUSY_STATUSES = frozenset({"running", "waiting"})
_TURN_END_STATUSES = frozenset({"idle", "failed"})

# Turns and hosts with nothing pending are forgotten past this many, since
# managed hosts churn over a long-lived server's life.
_MAX_TRACKED = 4096

# (workspace_id, session_id or host_id): session ids collide across workspaces.
_Key = tuple[int, str]


@dataclass(frozen=True)
class _Target:
    """A checkpoint-capable managed host that a session's turn runs on."""

    host_id: str
    provider: str
    policy: SandboxCheckpointPolicy


@dataclass
class _Turn:
    """One session's turn, from its first edge until its end has held.

    ``targets`` resolves to ``None`` when the hosts could not be determined.
    ``activity`` counts busy publishes, so a stale read can be recognised.
    """

    context: contextvars.Context
    targets: asyncio.Task[list[_Target] | None]
    busy_since: float | None = None
    activity: int = 0
    ending: asyncio.Task[None] | None = None
    long_turn: asyncio.Task[None] | None = None


@dataclass
class _HostSaves:
    """Checkpoint requests for one host: folding, spacing and their timestamps."""

    target: _Target
    context: contextvars.Context
    last_requested_at: float | None = None
    last_finished_at: float | None = None
    pending: CheckpointReason | None = None
    in_flight: bool = False
    spacing: asyncio.Task[None] | None = None


def _active(task: asyncio.Task[Any] | None) -> bool:
    return task is not None and not task.done()


def _found_no_targets(targets: asyncio.Task[list[_Target] | None]) -> bool:
    return targets.done() and not targets.cancelled() and targets.result() == []


class TurnCheckpointScheduler:
    """Turns session statuses into rate-limited ``checkpoint`` calls.

    :param conversation_store: Resolves a session to its host.
    :param host_store: Reads the host's provider and current sandbox.
    :param deployment: The deployment's providers and their checkpoint policies.
    :param turn_end_debounce_s: How long a turn end must hold before it counts.
    :param clock: Monotonic seconds; injectable for tests.
    :param sleep: Awaitable delay; injectable for tests.
    :param run_sync: Runs a blocking call off the event loop, carrying the
        caller's ``contextvars``; injectable for tests.
    """

    def __init__(
        self,
        *,
        conversation_store: ConversationStore,
        host_store: HostStore,
        deployment: ManagedSandboxDeployment,
        turn_end_debounce_s: float = TURN_END_DEBOUNCE_S,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        run_sync: Callable[..., Awaitable[Any]] = asyncio.to_thread,
    ) -> None:
        self._conversation_store = conversation_store
        self._host_store = host_store
        self._deployment = deployment
        self._turn_end_debounce_s = turn_end_debounce_s
        self._clock = clock
        self._sleep = sleep
        self._run_sync = run_sync
        self._turns: dict[_Key, _Turn] = {}
        self._hosts: dict[_Key, _HostSaves] = {}
        self._tasks: set[asyncio.Task[Any]] = set()
        # Which offered entries (by index) implement checkpoint: the provider
        # name, or None when the launcher lacks it. Found lazily in a worker
        # thread, because it builds launchers; an entry whose launcher fails to
        # build is retried on the next resolution.
        self._entry_providers: dict[int, str | None] = {}
        self._unbuildable_entries: set[int] = set()
        self._dropped_providers: set[str] = set()
        self._none_capable = False
        self._providers_lock = threading.Lock()
        # Statuses count only between start() and aclose(), i.e. while the server runs.
        self._listening = False

    def note_status(self, session_id: str, previous_status: str | None, status: str) -> None:
        """Track one published session status. Call on the event loop.

        Repeats are fine: the scheduler keeps its own record of each turn.

        :param session_id: Session whose status was published, e.g. ``"conv_abc123"``.
        :param previous_status: The status cached before this publish, or
            ``None`` when unknown. Read only for a session with no turn in
            progress here.
        :param status: The published status, e.g. ``"idle"``.
        """
        if not self._listening:
            return
        key = (current_workspace_id(), session_id)
        turn = self._turns.get(key)
        if status in _BUSY_STATUSES:
            self._turn_running(key, turn)
        elif status in _TURN_END_STATUSES:
            if turn is not None:
                # A repeated end while one is pending must not restart its debounce.
                if turn.ending is None:
                    self._turn_ending(key, turn)
            elif previous_status not in ("idle", status):
                # An unknown previous status (a server restart mid-turn) or
                # failed -> idle from the runner-recovered path.
                self._turn_ending(key, self._start_turn(key))

    def start(self) -> None:
        """Start acting on statuses; called when the server starts."""
        self._listening = True

    async def aclose(self) -> None:
        """Cancel every timer and pending call; later statuses are ignored.

        A ``checkpoint`` call already running in a worker thread still finishes.
        """
        self._listening = False
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._turns.clear()
        self._hosts.clear()

    # ── turns ────────────────────────────────────────────────────────────────

    def _turn_running(self, key: _Key, turn: _Turn | None) -> None:
        if turn is None:
            turn = self._start_turn(key)
        turn.activity += 1
        if turn.ending is not None:
            # Running again before the end held: it was a lull, not the end.
            turn.ending.cancel()
            turn.ending = None
        if turn.busy_since is None:
            turn.busy_since = self._clock()
        # Also restarts a watcher that stopped on a stale "ended" read.
        if not _active(turn.long_turn) and not _found_no_targets(turn.targets):
            turn.long_turn = self._spawn(
                self._watch_long_turn(key, turn, turn.busy_since), turn.context
            )

    def _turn_ending(self, key: _Key, turn: _Turn) -> None:
        turn.ending = self._spawn(self._finish_turn(key, turn), turn.context)

    def _start_turn(self, key: _Key) -> _Turn:
        self._prune()
        context = contextvars.copy_context()
        turn = _Turn(context=context, targets=self._spawn(self._resolve(key[1]), context))
        self._turns[key] = turn
        return turn

    async def _finish_turn(self, key: _Key, turn: _Turn) -> None:
        """Request a ``turn_completed`` save once the turn end has held."""
        await self._sleep(self._turn_end_debounce_s)
        # The end held, so the next status starts a new turn.
        if self._turns.get(key) is turn:
            del self._turns[key]
        if turn.long_turn is not None:
            turn.long_turn.cancel()
        targets = await asyncio.shield(turn.targets)
        if targets is None:
            targets = await self._resolve(key[1]) or []
        for target in targets:
            self._request(key[0], target, "turn_completed")

    async def _watch_long_turn(self, key: _Key, turn: _Turn, busy_since: float) -> None:
        """Request a ``long_turn`` save whenever the turn goes an interval without one."""
        targets = await asyncio.shield(turn.targets)
        while targets is None:
            await self._sleep(_RESOLVE_RETRY_S)
            targets = await self._resolve(key[1])
        while targets:
            now = self._clock()
            due = min(self._long_turn_due(key[0], target, busy_since) for target in targets)
            if due > now:
                await self._sleep(due - now)
                continue
            # Re-read first: the session may have been deleted, or its turn may
            # have ended where this replica could not see it. Keep watching
            # through a failed read, or an "ended" read that newer local
            # activity contradicts.
            activity = turn.activity
            rechecked = await self._resolve(key[1], require_busy=True)
            if rechecked is not None and (rechecked or turn.activity == activity):
                targets = rechecked
            if not targets and self._turns.get(key) is turn and not _active(turn.ending):
                del self._turns[key]
            for target in targets:
                if self._long_turn_due(key[0], target, busy_since) <= self._clock():
                    self._request(key[0], target, "long_turn")

    def _long_turn_due(self, workspace_id: int, target: _Target, busy_since: float) -> float:
        host = self._hosts.get((workspace_id, target.host_id))
        last = host.last_requested_at if host is not None else None
        start = busy_since if last is None else max(busy_since, last)
        return start + target.policy.max_interval_s

    # ── per-host requests ────────────────────────────────────────────────────

    def _request(self, workspace_id: int, target: _Target, reason: CheckpointReason) -> None:
        key = (workspace_id, target.host_id)
        host = self._hosts.get(key)
        if host is None:
            self._prune()
            host = _HostSaves(target=target, context=contextvars.copy_context())
            self._hosts[key] = host
        host.target = target
        host.last_requested_at = self._clock()
        # One follow-up covers every request folded into it; a turn end is
        # the more specific reason, so it wins.
        if host.pending != "turn_completed":
            host.pending = reason
        self._pump(key, host)

    def _pump(self, key: _Key, host: _HostSaves) -> None:
        """Start the host's pending ``checkpoint`` call when nothing holds it back."""
        if not self._listening or host.in_flight or host.pending is None:
            return
        if _active(host.spacing):
            return
        if host.last_finished_at is not None:
            # Spaced from the last call's end, so a slow worker cannot bunch calls.
            wait = host.last_finished_at + host.target.policy.min_interval_s - self._clock()
            if wait > 0:
                host.spacing = self._spawn(self._pump_after(key, host, wait), host.context)
                return
        reason, host.pending = host.pending, None
        host.in_flight = True
        self._spawn(self._call(key, host, reason), host.context)

    async def _pump_after(self, key: _Key, host: _HostSaves, delay: float) -> None:
        await self._sleep(delay)
        host.spacing = None
        self._pump(key, host)

    async def _call(self, key: _Key, host: _HostSaves, reason: CheckpointReason) -> None:
        try:
            await self._run_sync(self._checkpoint, host.target, reason)
        except Exception:  # noqa: BLE001 - a failed save must never reach the status path
            _logger.warning(
                "checkpoint request for managed host %s failed",
                host.target.host_id,
                exc_info=True,
            )
        finally:
            host.in_flight = False
            host.last_finished_at = self._clock()
            self._pump(key, host)

    def _prune(self) -> None:
        """Forget idle turns and quiet hosts once there are many of them.

        Safe at any time: a forgotten turn only loses its start time, which
        matters while its long-turn watcher runs, and a forgotten host only
        loses timestamps older than its longest interval.
        """
        if len(self._turns) > _MAX_TRACKED:
            for key, turn in list(self._turns.items()):
                if not _active(turn.ending) and not _active(turn.long_turn):
                    del self._turns[key]
        if len(self._hosts) > _MAX_TRACKED:
            now = self._clock()
            for key, host in list(self._hosts.items()):
                stamps = (host.last_requested_at, host.last_finished_at)
                last = max((stamp for stamp in stamps if stamp is not None), default=now)
                quiet = not host.in_flight and host.pending is None
                if quiet and now - last >= host.target.policy.max_interval_s:
                    del self._hosts[key]

    def _spawn(
        self, coroutine: Coroutine[Any, Any, Any], context: contextvars.Context
    ) -> asyncio.Task[Any]:
        task = asyncio.get_running_loop().create_task(coroutine, context=context.copy())
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    # ── worker-thread side ───────────────────────────────────────────────────

    async def _resolve(
        self, session_id: str, *, require_busy: bool = False
    ) -> list[_Target] | None:
        """Find the checkpoint-capable hosts a session's turn runs on.

        :param session_id: The session, e.g. ``"conv_abc123"``.
        :param require_busy: Return nothing once the session's persisted
            status says its turn is over.
        :returns: The targets, possibly empty, or ``None`` when they could not
            be determined: a store failed, or a launcher could not be built.
        """
        try:
            targets = await self._run_sync(self._resolve_targets, session_id, require_busy)
        except Exception:  # noqa: BLE001 - best effort; retried by the caller or the next turn
            _logger.warning(
                "could not resolve the managed host of session %s for checkpoints",
                session_id,
                exc_info=True,
            )
            return None
        if self._none_capable:
            # Every provider is known to lack checkpoint: stop listening for good.
            self._listening = False
            self._turns.clear()
        return targets

    def _resolve_targets(self, session_id: str, require_busy: bool) -> list[_Target] | None:
        providers, settled = self._checkpoint_providers()
        if not providers:
            # Unknown, not empty, while a launcher that failed to build is retried.
            return [] if settled else None
        conversation = self._conversation_store.get_conversation(session_id)
        if conversation is None:
            return []
        if require_busy and conversation.live_status in _TURN_END_STATUSES:
            return []
        targets: list[_Target] = []
        unknown = False
        for host_id in sorted(self._host_ids(conversation)):
            host = self._host_store.get_host(host_id)
            if host is None or not host.sandbox_id or host.sandbox_provider is None:
                continue
            provider = host.sandbox_provider
            if provider not in providers:
                unknown = unknown or not settled
                continue
            policy = self._deployment.recorded(provider).checkpoint
            targets.append(_Target(host_id, provider, policy))
        return None if unknown and not targets else targets

    def _host_ids(self, conversation: Conversation) -> set[str]:
        if conversation.host_id:
            return {conversation.host_id}
        # A mirrored sub-agent has no host of its own; its runner's sessions do.
        if conversation.runner_id:
            return {
                sibling.host_id
                for sibling in self._conversation_store.list_conversations_by_runner_id(
                    conversation.runner_id
                )
                if sibling.host_id
            }
        return set()

    def _checkpoint_providers(self) -> tuple[frozenset[str], bool]:
        """Offered providers that implement checkpoint with an enabled policy.

        :returns: Those providers, and whether every eligible entry was checked
            (``False`` while a launcher that failed to build awaits a retry).
        """
        with self._providers_lock:
            settled = True
            for index, entry in enumerate(self._deployment.offered()):
                if (
                    index in self._entry_providers
                    or not entry.managed_launch_supported
                    or not entry.checkpoint.enabled
                ):
                    continue
                try:
                    launcher = entry.launcher_factory()
                except Exception:  # noqa: BLE001 - one broken provider must not disable the rest
                    settled = False
                    if index not in self._unbuildable_entries:
                        self._unbuildable_entries.add(index)
                        _logger.warning(
                            "could not build the %s launcher to check checkpoint support; "
                            "will retry",
                            entry.provider,
                            exc_info=True,
                        )
                    continue
                if launcher.capabilities.checkpoint:
                    self._entry_providers[index] = launcher.provider
                else:
                    self._entry_providers[index] = None
                    _logger.info(
                        "sandbox provider %s does not implement checkpoint; "
                        "its managed hosts get no turn-aligned checkpoints",
                        launcher.provider,
                    )
            providers = {name for name in self._entry_providers.values() if name is not None}
            capable = frozenset(providers - self._dropped_providers)
            self._none_capable = settled and not capable
            return capable, settled

    def _checkpoint(self, target: _Target, reason: CheckpointReason) -> None:
        host = self._host_store.get_host(target.host_id)
        # Re-read: the host may be gone, or relaunched on another sandbox.
        if host is None or not host.sandbox_id or host.sandbox_provider != target.provider:
            return
        launcher = _launcher_for_teardown(host, self._deployment)
        if launcher is None:
            return
        try:
            operation_id = launcher.checkpoint(host.sandbox_id, reason=reason)
        except SandboxCapabilityError:
            self._drop_provider(target.provider)
            return
        except SandboxGoneError:
            _logger.info(
                "managed sandbox %s is gone; skipped its %s checkpoint", host.sandbox_id, reason
            )
            return
        _logger.info(
            "requested %s checkpoint of managed sandbox %s (provider %s, operation %s)",
            reason,
            host.sandbox_id,
            target.provider,
            operation_id or "none",
        )

    def _drop_provider(self, provider: str) -> None:
        """Stop asking a provider whose capability flag outlived its method."""
        with self._providers_lock:
            if provider in self._dropped_providers:
                return
            self._dropped_providers.add(provider)
        _logger.warning(
            "sandbox provider %s declares checkpoint but refused the call; "
            "its managed hosts get no turn-aligned checkpoints",
            provider,
        )


_scheduler: TurnCheckpointScheduler | None = None


def configure(
    conversation_store: ConversationStore,
    host_store: HostStore | None,
    sandbox_config: ManagedSandboxDeployment | None,
) -> TurnCheckpointScheduler | None:
    """Wire the stores and provider set; called once at app construction.

    Without a sandbox config or host store :func:`note_status` is a no-op, and
    otherwise it acts only once the server has started the returned scheduler.

    :param conversation_store: Resolves a session to its host.
    :param host_store: Reads the host's sandbox, or ``None`` to disable.
    :param sandbox_config: The deployment's providers, or ``None`` to disable.
    :returns: The scheduler now receiving edges, for the server to start and
        close with its lifespan, or ``None`` when disabled.
    """
    global _scheduler
    _scheduler = None
    if sandbox_config is not None and host_store is not None:
        _scheduler = TurnCheckpointScheduler(
            conversation_store=conversation_store,
            host_store=host_store,
            deployment=sandbox_config,
        )
    return _scheduler


def note_status(session_id: str, previous_status: str | None, status: str) -> None:
    """Report a published session status to the turn-aligned checkpoint policy.

    Non-blocking and fail-safe; a no-op unless :func:`configure` wired a sandbox
    deployment, the server has started it, and the caller runs on the event loop.

    :param session_id: Session whose status was published, e.g. ``"conv_abc123"``.
    :param previous_status: The status cached before this publish, or ``None``.
    :param status: The published status, e.g. ``"idle"``.
    """
    scheduler = _scheduler
    if scheduler is None:
        return
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return
    try:
        scheduler.note_status(session_id, previous_status, status)
    except Exception:  # noqa: BLE001 - must never break the status publish
        _logger.warning("checkpoint scheduling failed for session %s", session_id, exc_info=True)
