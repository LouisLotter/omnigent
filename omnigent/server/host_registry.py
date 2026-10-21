"""In-memory registry of live host WebSocket connections.

Each server replica maintains one :class:`HostRegistry` tracking
hosts with active WebSocket tunnels on this replica. The persistent
``hosts`` DB table (queried by ``HostStore``) is the cross-replica
source of truth for which hosts exist; this registry only tracks
which hosts are live *here*.

Simpler than :class:`TunnelRegistry` because the host tunnel
carries only control frames (launch/stop runner), not HTTP
request/response traffic. No per-request reassembly queues needed.

The registry also holds what connected hosts *report* about themselves and
nothing persists — today the per-family gateway-inference map (see
:mod:`omnigent.gateway_inference`) and interactive-shell inventory. They are
delivered on the connect handshake, so a replica that has never seen a host
simply knows nothing about them until the host reconnects and re-reports.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import Awaitable, Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar

from cachetools import TTLCache

from omnigent._platform import normalize_interactive_shells
from omnigent.db.account_authority import (
    AccountAuthority,
    account_generation,
    current_account_user,
)
from omnigent.db.db_models import InvalidUuidError, current_workspace_id, uuid_to_bytes
from omnigent.host.frames import (
    CAP_CODEX_SIDE_CHAT,
    CAP_OPERATION_RECORDS,
    OPERATION_RECORD_RETENTION_S,
    HostHelloFrame,
    HostMcpServersResultFrame,
    HostMcpToolsResultFrame,
    HostPluginsResultFrame,
    HostQuiesceResultFrame,
    HostSkillContentResultFrame,
    HostSkillsResultFrame,
)
from omnigent.host.harness_startup import HarnessStartup
from omnigent.runner.transports.ws_tunnel.diagnostics import OutboundFrame, TunnelDiagnostics
from omnigent.stores.host_store import LAUNCH_TOKEN_ONLY_REFUSAL, HostRegistrationRefusedError

_logger = logging.getLogger(__name__)

_AnswerT = TypeVar("_AnswerT")

# How often a request whose tunnel closed looks for the host's reconnect.
_RECONNECT_POLL_S = 0.1

_expected_host_owner: ContextVar[AccountAuthority | None] = ContextVar(
    "expected_host_owner", default=None
)


@contextmanager
def host_owner_scope(user_id: str | None, generation: str | None) -> Iterator[None]:
    """Restrict scheduled host RPCs to their saved owner's registration."""
    owner = AccountAuthority(user_id, generation, current_workspace_id()) if user_id else None
    token = _expected_host_owner.set(owner)
    try:
        yield
    finally:
        _expected_host_owner.reset(token)


class ConcurrentRegistrationError(RuntimeError):
    """Another connection registered the host after the caller read its registration epoch."""


def _canonical_host_id(host_id: str) -> str:
    """Reduce a host id to the canonical bare-hex form used as the key.

    Host ids reach the registry in every spelling ``uuid_to_bytes``
    accepts: the bare 32-char hex the tunnel route registers under,
    the legacy ``host_<hex>`` form that pre-migration clients still
    send in REST paths, and the dashed uuid form. The DB layer
    normalizes all of them (``Uuid16``), so the registry must key on
    the same canonical form — otherwise a legacy-form lookup misses a
    live tunnel and runner launches 409 "host is offline" while
    ``GET /v1/hosts`` reports the host online. Ids that aren't
    uuid-shaped at all are keyed verbatim so they simply miss.

    :param host_id: A host id in any accepted spelling, e.g.
        ``"host_a1b2..."``, ``"a1b2..."``, or the dashed uuid.
    :returns: The bare-hex form, or *host_id* unchanged when it is
        not uuid-shaped.
    """
    try:
        return uuid_to_bytes(host_id).hex()
    except InvalidUuidError:
        return host_id


def _fail_pending_harness_startup(conn: HostConnection) -> None:
    """Settle launch settings requests when their tunnel can no longer reply."""
    while conn.pending_harness_startup:
        _, future = conn.pending_harness_startup.popitem()
        if not future.done():
            future.set_result(None)


def _fail_pending_imports(conn: HostConnection) -> None:
    """Fail the connection's in-flight import streams immediately.

    A dead tunnel can never deliver another session frame; without this
    signal the import request only learns of the drop by waiting out its
    per-frame timeout (60s of "Importing…" in the UI).
    """
    while conn.pending_import_local:
        _request_id, queue = conn.pending_import_local.popitem()
        queue.put_nowait(
            (
                "done",
                {
                    "status": "failed",
                    "error": f"host '{conn.host_id}' disconnected mid-import",
                },
            )
        )


def _fail_pending_plugins(conn: HostConnection) -> None:
    """Fail plugin requests as soon as their tunnel disconnects or is replaced."""
    while conn.pending_plugins:
        _request_id, future = conn.pending_plugins.popitem()
        if not future.done():
            future.set_exception(ConnectionError(f"host '{conn.host_id}' disconnected"))


def _fail_pending_skill_content(conn: HostConnection) -> None:
    """Settle lookups immediately when their host connection disappears."""
    while conn.pending_skill_content:
        _request_id, future = conn.pending_skill_content.popitem()
        if not future.done():
            future.set_exception(ConnectionError("host disconnected"))


def _fail_pending_mcp_tools(conn: HostConnection) -> None:
    """Settle probes immediately when their host connection disappears."""
    while conn.pending_mcp_tools:
        _request_id, future = conn.pending_mcp_tools.popitem()
        if not future.done():
            future.set_exception(ConnectionError("host disconnected"))


def _fail_pending_inventory(pending: dict[str, asyncio.Future[Any]], host_id: str) -> None:
    """Fail inventory waiters on their owner loops and clear the retired map."""
    _fail_pending_futures(pending, lambda: ConnectionError(f"host '{host_id}' disconnected"))


class HostOperationOutcomeUnknownError(ConnectionError):
    """The host may have carried out a request, but its tunnel closed before it answered."""


# Requests that change something on the host: a lost answer leaves their
# outcome unknown rather than failed.
_MUTATING_PENDING_MAPS = (
    "pending_launches",
    "pending_stops",
    "pending_create_worktrees",
    "pending_remove_worktrees",
    "pending_create_dirs",
    "pending_installs",
    "pending_secret_writes",
    "pending_fs_requests",
)
# Requests that only read: a lost answer is a plain failure.
_READ_PENDING_MAPS = (
    "pending_runner_status",
    "pending_quiesce",
    "pending_stats",
    "pending_list_dirs",
    "pending_list_worktrees",
    "pending_credential_detects",
    "pending_model_options",
)


# How long an operation whose answer was lost keeps counting as in flight: the
# longest host operation budget (a harness install, 420 s).
UNANSWERED_OPERATION_HOLD_S = 420.0


def _fail_pending_requests(conn: HostConnection) -> None:
    """
    Answer every request still waiting on a closed connection at once.

    Without this, each waited out its own timeout (5 s to 420 s) for an
    answer that can no longer arrive. An operation that changes the host may
    still be running there, so it stays in ``inflight_unanswered`` for
    :data:`UNANSWERED_OPERATION_HOLD_S`.

    :param conn: The connection that was deregistered or replaced.
    """
    _hold_unanswered_operations(conn)
    for name in _MUTATING_PENDING_MAPS:
        _fail_pending_futures(
            getattr(conn, name),
            lambda: HostOperationOutcomeUnknownError(
                f"host '{conn.host_id}' disconnected before answering; "
                "the operation may have taken effect"
            ),
        )
    for name in _READ_PENDING_MAPS:
        _fail_pending_futures(
            getattr(conn, name), lambda: ConnectionError(f"host '{conn.host_id}' disconnected")
        )


def _hold_unanswered_operations(conn: HostConnection) -> None:
    """Keep the closed connection's unanswered host-changing requests counted as in flight."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    now = time.monotonic()
    for name in _MUTATING_PENDING_MAPS:
        for request_id, future in getattr(conn, name).items():
            if future.done() or request_id in conn.inflight_unanswered:
                continue
            conn.inflight_unanswered[request_id] = now
            # The callback holds the connection itself, so a fence that tracks
            # connections weakly keeps seeing the hold until it expires.
            loop.call_later(UNANSWERED_OPERATION_HOLD_S, _release_unanswered, conn, request_id)


def _release_unanswered(conn: HostConnection, request_id: str) -> None:
    """End one unanswered operation's hold."""
    conn.inflight_unanswered.pop(request_id, None)


def _fail_pending_futures(
    pending: dict[str, asyncio.Future[Any]], error: Callable[[], BaseException]
) -> None:
    """Fail waiters on their owner loops and clear the retired map."""
    try:
        current_loop = asyncio.get_running_loop()
    except RuntimeError:
        current_loop = None
    while True:
        try:
            _request_id, future = pending.popitem()
        except KeyError:
            break
        if future.done():
            continue
        try:
            owner_loop = future.get_loop()
            if owner_loop.is_closed():
                continue

            def settle(future: asyncio.Future[Any] = future) -> None:
                if not future.done():
                    future.set_exception(error())

            if owner_loop is current_loop:
                settle()
            else:
                owner_loop.call_soon_threadsafe(settle)
        except RuntimeError:
            # The owner loop can close between the check and scheduling.
            continue


def _fail_pending_skills(conn: HostConnection) -> None:
    """Fail skill-discovery requests as soon as their host disappears."""
    _fail_pending_inventory(conn.pending_skills, conn.host_id)


def _fail_pending_mcp_servers(conn: HostConnection) -> None:
    """Fail MCP inventory requests as soon as their host disappears."""
    _fail_pending_inventory(conn.pending_mcp_servers, conn.host_id)


# How long a runner exit report stays answerable, and how many are kept.
# Reports only matter while a client is still waiting for the runner to
# come online (a 60s window today); 10 minutes covers slow retries with
# margin. Runner ids are unique per launch, so entries never need
# invalidation — the TTL is purely a memory bound.
_EXIT_REPORT_TTL_S = 600.0
_EXIT_REPORT_MAX_ENTRIES = 1024


@dataclass
class RunnerExitReport:
    """A host daemon's report that a spawned runner died unexpectedly.

    :param error: Human-readable cause composed by the daemon (exit
        code, host-side log path, log tail), e.g.
        ``"runner process exited with code 1 (log on host: ~/...)"``.
    :param owner: User who owns the host tunnel the report arrived on,
        e.g. ``"alice@example.com"``. ``None`` when auth is disabled.
        Gates visibility: only the owner may read the report (the log
        tail can contain agent output).
    :param had_connected: Host evidence that the runner connected at least once,
        or ``None`` when unavailable.
    """

    error: str
    owner: str | None
    had_connected: bool | None = None


class RunnerExitReports:
    """Thread-safe, TTL-bounded store of runner exit reports.

    Written by the host tunnel when a ``host.runner_exited`` frame
    arrives; read by the runner status endpoint so a client polling a
    never-connecting runner learns *why* instead of timing out.
    In-memory and per-replica, same posture as :class:`HostRegistry` —
    the report and the status poll meet on the replica holding the
    host tunnel.
    """

    def __init__(self) -> None:
        """Initialize an empty report store."""
        self._lock = threading.Lock()
        self._reports: TTLCache[str, RunnerExitReport] = TTLCache(
            maxsize=_EXIT_REPORT_MAX_ENTRIES,
            ttl=_EXIT_REPORT_TTL_S,
        )

    def record(
        self, runner_id: str, error: str, owner: str | None, *, had_connected: bool | None = None
    ) -> None:
        """Store a runner exit report.

        :param runner_id: The dead runner, e.g. ``"runner_abc123"``.
        :param error: Human-readable cause from the host daemon.
        :param owner: Owner of the reporting host tunnel, or ``None``
            when auth is disabled.
        :param had_connected: First-connect evidence supplied by the host.
        """
        with self._lock:
            self._reports[runner_id] = RunnerExitReport(
                error=error, owner=owner, had_connected=had_connected
            )

    def was_connected(self, runner_id: str) -> bool:
        """Return whether the host's exit report proves this runner had connected.

        :param runner_id: Runner whose lifetime evidence is needed.
        :returns: ``True`` only for a report with affirmative connection evidence.
        """
        with self._lock:
            report = self._reports.get(runner_id)
        return report is not None and report.had_connected is True

    def get(self, runner_id: str) -> str | None:
        """Look up a report's error without owner scoping.

        For callers that have already authorized access by another
        means (e.g. the session snapshot, gated on session permission):
        the report pertains to that session's own runner, so no
        separate owner check is needed. The runner status endpoint —
        keyed only by ``runner_id`` with no session-level auth — must
        use :meth:`get_visible` instead.

        :param runner_id: Runner id, e.g. ``"runner_abc123"``.
        :returns: The error message, or ``None`` when no report exists.
        """
        with self._lock:
            report: RunnerExitReport | None = self._reports.get(runner_id)
        return report.error if report is not None else None

    def get_visible(self, runner_id: str, user_id: str | None) -> str | None:
        """Look up a report, scoped to its owner.

        :param runner_id: Runner id, e.g. ``"runner_abc123"``.
        :param user_id: The requesting user, or ``None`` when auth is
            disabled.
        :returns: The error message, or ``None`` when no report exists
            or the caller doesn't own it (W6-2 posture: other users'
            runners reveal nothing).
        """
        with self._lock:
            report: RunnerExitReport | None = self._reports.get(runner_id)
        if report is None:
            return None
        if user_id is not None and report.owner is not None and report.owner != user_id:
            return None
        return report.error


class WebSocketLike(Protocol):
    """Minimal WebSocket protocol for the host tunnel.

    Both Starlette's ``WebSocket`` and test fakes implement this.
    """

    async def send_text(self, data: str) -> None:
        """Send a text frame."""
        ...

    async def receive_text(self) -> str:
        """Receive a text frame."""
        ...


@dataclass
class HostConnection:
    """Per-host state while the tunnel is open.

    :param workspace_id: Tenant partition the tunnel belongs to,
        mirroring the ``hosts`` table's ``(workspace_id, host_id)`` PK.
        Captured at register time so ``send_text``'s replaced-connection
        guard keys on the full ``(workspace_id, host_id)`` without
        reading request context from the long-lived sender loop.
    :param host_id: Stable host identifier, e.g.
        ``"host_a1b2c3d4..."``.
    :param ws: The live WebSocket to this host.
    :param hello: The hello frame the host sent on connect.
    :param owner: Authenticated user who established the tunnel,
        e.g. ``"alice@example.com"``. ``None`` when auth is
        disabled (single-user mode).
    :param outbound_queue: Queue consumed by the WebSocket route's
        sender task. Control frames are enqueued here rather than
        calling ``ws.send_text`` directly, since the caller may
        be on a different thread.
    :param connected_at: Unix epoch float of connect time.
    :param last_frame_at: Unix epoch float of the most recent
        frame from this host.
    :param last_frame_monotonic: Monotonic timestamp of the last frame,
        used for watchdog inactivity checks across wall-clock corrections.
    :param pending_launches: Per-``request_id`` futures for
        in-flight ``host.launch_runner`` requests. Resolved when
        the host sends ``host.launch_runner_result``.
    :param pending_stops: Per-``request_id`` futures for
        in-flight ``host.stop_runner`` requests. Resolved when
        the host sends ``host.stop_runner_result``.
    :param pending_runner_status: Per-``request_id`` futures for
        in-flight ``host.runner_status`` queries. Resolved when the
        host sends ``host.runner_status_result``. Values carry the
        single ``status`` field (``"alive"`` / ``"dead"`` /
        ``"unknown"``).
    :param pending_stats: Per-``request_id`` futures for in-flight
        ``host.stat`` requests. Resolved when the host sends
        ``host.stat_result``. The dict values carry the full
        stat-result fields (``status``, ``exists``, ``type``,
        ``canonical_path``, ``error``); typed as ``Any`` because
        Python ``dict`` parametric types here would force every
        callsite to cast.
    :param pending_list_dirs: Per-``request_id`` futures for
        in-flight ``host.list_dir`` requests. Resolved when the
        host sends ``host.list_dir_result``. Values carry the
        listing fields (``status``, ``entries`` as list of
        dicts, ``has_more``, ``error``). Same ``Any`` typing
        rationale as ``pending_stats``.
    :param pending_create_worktrees: Per-``request_id`` futures for
        in-flight ``host.create_worktree`` requests. Resolved when
        the host sends ``host.create_worktree_result``. Values
        carry the result fields (``status``, ``worktree_path``,
        ``branch``, ``error``). Same ``Any`` typing rationale as
        ``pending_stats``.
    :param pending_remove_worktrees: Per-``request_id`` futures for
        in-flight ``host.remove_worktree`` requests. Resolved when
        the host sends ``host.remove_worktree_result``. Values
        carry ``status`` and ``error``.
    :param pending_create_dirs: Per-``request_id`` futures for
        in-flight ``host.create_dir`` requests. Resolved when the
        host sends ``host.create_dir_result``. Values carry the
        result fields (``status``, ``path``, ``error``). Same
        ``Any`` typing rationale as ``pending_stats``.
    :param pending_installs: Per-``request_id`` futures for in-flight
        ``host.install_harness`` requests. Resolved when the host sends
        ``host.install_harness_result``. Values carry the result fields
        (``status``, ``configured_harnesses``, ``gateway_inference``,
        ``error``). Same ``Any`` typing rationale as ``pending_stats``.
    :param inflight_installs: Install tasks used to coalesce concurrent
        install requests for the same harness family (a double-click, or
        two spellings of one npm package) onto one in-flight install, so
        npm's non-race-safe global writes never run twice at once. Keyed by
        the resolved install key (not ``request_id``) and cleared when the
        install completes.
    :param inflight_unanswered: Request ids of operations that changed the host
        and lost their answer when this connection closed, mapped to when they
        were given up. The host may still be carrying them out, so they keep
        counting as in flight (a sleep fence waits for them) until the longest
        operation budget has passed.
    :param pending_secret_writes: Per-``request_id`` futures for in-flight
        ``host.store_secret`` requests (a UI-driven harness credential write).
        Resolved when the host sends ``host.store_secret_result``. Values carry
        the result fields (``status``, ``configured_harnesses``,
        ``gateway_inference``, ``error``) — never the secret. Same ``Any``
        typing rationale as ``pending_stats``.
    :param credential_write_lock: Serializes credential writes to this host so
        two overlapping requests (a double-click, or key + gateway in quick
        succession) can't interleave the daemon's non-atomic
        load→merge→save of ``config.yaml`` and clobber a sibling ``providers:``
        entry. Held around the whole store-secret round-trip.
    :param pending_fs_requests: Per-``request_id`` futures for
        in-flight ``host.fs_request`` reads (the workspace file
        panel served from the host while the runner is offline).
        Resolved when the host sends ``host.fs_result``. Values
        carry ``status``, ``payload``, ``error_status``,
        ``error_code``, and ``error``.
    :param pending_model_options: Per-``request_id`` futures for pre-launch
        model catalogs resolved by the selected host.
    :param pending_skills: Per-``request_id`` futures for sessionless skill discovery.
    :param pending_mcp_servers: Per-``request_id`` futures for MCP inventory requests.
    :param pending_skill_content: Per-request futures for transient SKILL.md reads.
    :param pending_mcp_tools: Per-request futures for lazy MCP discovery.
    """

    workspace_id: int
    host_id: str
    ws: WebSocketLike
    hello: HostHelloFrame
    owner: str | None
    outbound_queue: asyncio.Queue[OutboundFrame | None]
    connected_at: float
    last_frame_at: float
    last_frame_monotonic: float = field(default_factory=time.monotonic, kw_only=True)
    diagnostics: TunnelDiagnostics = field(default_factory=TunnelDiagnostics)
    retirement_reason: str | None = None
    sender_end_reason: str | None = None
    close_requested_code: int | None = None
    close_requested_reason: str | None = None
    account_generation: str | None = None
    # True when this tunnel authenticated with a valid managed-sandbox launch
    # token, i.e. a server-provisioned sandbox proving itself on connect — not a
    # user machine reusing a managed host's id under ordinary login. Read by the
    # default-public policy so only genuine sandboxes count as managed.
    registered_with_managed_token: bool = False
    pending_launches: dict[str, asyncio.Future[dict[str, str | None]]] = field(
        default_factory=dict,
    )
    pending_stops: dict[str, asyncio.Future[dict[str, str | None]]] = field(
        default_factory=dict,
    )
    pending_runner_status: dict[str, asyncio.Future[dict[str, str | None]]] = field(
        default_factory=dict,
    )
    pending_quiesce: dict[str, asyncio.Future[HostQuiesceResultFrame]] = field(
        default_factory=dict,
    )
    pending_stats: dict[str, asyncio.Future[dict[str, Any]]] = field(
        default_factory=dict,
    )
    pending_list_dirs: dict[str, asyncio.Future[dict[str, Any]]] = field(
        default_factory=dict,
    )
    pending_create_worktrees: dict[str, asyncio.Future[dict[str, Any]]] = field(
        default_factory=dict,
    )
    pending_remove_worktrees: dict[str, asyncio.Future[dict[str, Any]]] = field(
        default_factory=dict,
    )
    pending_list_worktrees: dict[str, asyncio.Future[dict[str, Any]]] = field(
        default_factory=dict,
    )
    pending_create_dirs: dict[str, asyncio.Future[dict[str, Any]]] = field(
        default_factory=dict,
    )
    pending_installs: dict[str, asyncio.Future[dict[str, Any]]] = field(
        default_factory=dict,
    )
    inflight_installs: dict[str, asyncio.Task[dict[str, Any]]] = field(
        default_factory=dict,
    )
    inflight_unanswered: dict[str, float] = field(default_factory=dict)
    pending_secret_writes: dict[str, asyncio.Future[dict[str, Any]]] = field(
        default_factory=dict,
    )
    pending_credential_detects: dict[str, asyncio.Future[dict[str, Any]]] = field(
        default_factory=dict,
    )
    credential_write_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    pending_fs_requests: dict[str, asyncio.Future[dict[str, Any]]] = field(
        default_factory=dict,
    )
    pending_model_options: dict[str, asyncio.Future[dict[str, Any]]] = field(
        default_factory=dict,
    )
    pending_skills: dict[str, asyncio.Future[HostSkillsResultFrame]] = field(
        default_factory=dict,
    )
    pending_harness_startup: dict[str, asyncio.Future[HarnessStartup | None]] = field(
        default_factory=dict
    )
    pending_plugins: dict[str, asyncio.Future[HostPluginsResultFrame]] = field(
        default_factory=dict
    )
    pending_skill_content: dict[str, asyncio.Future[HostSkillContentResultFrame]] = field(
        default_factory=dict
    )
    pending_mcp_tools: dict[str, asyncio.Future[HostMcpToolsResultFrame]] = field(
        default_factory=dict
    )
    pending_mcp_servers: dict[str, asyncio.Future[HostMcpServersResultFrame]] = field(
        default_factory=dict,
    )
    # Import streams one session per frame, so the tunnel pushes each onto a
    # per-request queue the /imports/local handler drains (vs a single future).
    # Each item is a ("session", dict) or ("done", dict) tuple.
    pending_import_local: dict[str, asyncio.Queue[tuple[str, dict[str, Any]]]] = field(
        default_factory=dict,
    )

    def enqueue(self, data: str, *, app_ping_ts: int | None = None) -> None:
        """Queue a frame on the socket loop, retaining local timing metadata."""
        queued_at = self.diagnostics.timestamp()
        frame = OutboundFrame(data, queued_at=queued_at, app_ping_ts=app_ping_ts)
        self.outbound_queue.put_nowait(frame)
        try:
            self.diagnostics.enqueued(frame, self.outbound_queue.qsize(), queued_at)
        except Exception:
            _logger.exception("Host %s outbound queue diagnostics failed", self.host_id)


class HostRegistry:
    """Thread-safe registry of live host WebSocket connections.

    All public methods acquire ``_lock`` so callers on different
    threads (e.g. REST route handlers vs. WebSocket event loops)
    don't race.
    """

    def __init__(self) -> None:
        """Initialize an empty host registry."""
        self._lock = threading.RLock()
        # Keyed by (workspace_id, host_id) to mirror the hosts-table PK:
        # one stable host_id can be live in more than one workspace.
        self._hosts: dict[tuple[int, str], HostConnection] = {}
        # Registrations per key, kept after a disconnect; see registration_epoch().
        self._registration_epochs: dict[tuple[int, str], int] = {}
        # Last gateway-inference map each host reported, keyed by canonical
        # host_id alone: the map describes the machine's local config, so the
        # same machine connected to two workspaces reports the same answer.
        # Kept across a host disconnect (a tunnel flap shouldn't blank a known
        # answer) and lost with the process, which is the point — a restarted
        # server re-learns it from the reconnect handshake.
        self._gateway_inference: dict[str, dict[str, bool]] = {}
        self._interactive_shells: dict[str, list[str]] = {}
        self.launch_authorizer: (
            Callable[[str, str, str | None, str | None, bool, str | None], None] | None
        ) = None
        # Optional host admission check (persistent managed hosts); may wait or refuse.
        self.admission_gate: Callable[[str, str], Awaitable[None]] | None = None
        # Optional check on every server-to-host request; raises ConnectionError to refuse.
        self.send_guard: Callable[[HostConnection], None] | None = None

    def register(
        self,
        host_id: str,
        ws: WebSocketLike,
        hello: HostHelloFrame,
        owner: str | None,
        workspace_id: int | None = None,
        registered_with_managed_token: bool = False,
        unless_registered_since: int | None = None,
    ) -> HostConnection:
        """Register a host connection (newest wins).

        If ``(workspace_id, host_id)`` is already registered (stale
        connection), the old connection is replaced and its outbound
        queue is poisoned with ``None`` so the sender loop exits. A
        connection without a launch token never replaces one that
        registered with it.

        Scoping the key by workspace means the same stable ``host_id``
        (a laptop's config id) connecting to two workspaces is tracked
        as two independent connections rather than one evicting the
        other — matching the ``hosts`` table's ``(workspace_id,
        host_id)`` PK.

        :param host_id: Stable host identifier, e.g.
            ``"host_a1b2c3d4..."``.
        :param ws: The live WebSocket.
        :param hello: The hello frame from the host.
        :param owner: Authenticated user ID, or ``None``.
        :param workspace_id: Tenant partition the connection belongs to.
            Defaults to the request-bound :func:`current_workspace_id`
            (``0`` in single-tenant deployments); captured into the
            connection so ``send_text`` need not read request context
            from the sender loop.
        :param registered_with_managed_token: ``True`` when the tunnel
            authenticated with a valid managed-sandbox launch token, so
            the default-public policy can treat it as a genuine sandbox.
        :param unless_registered_since: A :meth:`registration_epoch` read
            before the connection's credential was checked. When given, the
            connection is not registered if another one registered after
            that read, because the other may hold a newer credential.
        :returns: The new :class:`HostConnection`. Its ``host_id`` is
            the canonical form (see :func:`_canonical_host_id`).
        :raises HostRegistrationRefusedError: When a connection without a
            launch token would replace one registered with it.
        :raises ConcurrentRegistrationError: When another connection
            registered after *unless_registered_since*.
        """
        ws_id = current_workspace_id() if workspace_id is None else workspace_id
        host_id = _canonical_host_id(host_id)
        now = time.time()
        conn = HostConnection(
            workspace_id=ws_id,
            host_id=host_id,
            ws=ws,
            hello=hello,
            owner=owner,
            account_generation=account_generation(owner) if owner else None,
            outbound_queue=asyncio.Queue(),
            connected_at=now,
            last_frame_at=now,
            last_frame_monotonic=time.monotonic(),
            registered_with_managed_token=registered_with_managed_token,
        )
        with self._lock:
            key = (ws_id, host_id)
            epoch = self._registration_epochs.get(key, 0)
            if unless_registered_since is not None and epoch != unless_registered_since:
                raise ConcurrentRegistrationError(f"host {host_id!r} registered meanwhile")
            old = self._hosts.get(key)
            # A bearer registration that raced the host's arming must not displace it.
            displaces_launch_token = old is not None and old.registered_with_managed_token
            if displaces_launch_token and not registered_with_managed_token:
                raise HostRegistrationRefusedError(LAUNCH_TOKEN_ONLY_REFUSAL)
            if old is not None:
                _logger.info(
                    "replacing stale host connection: ws=%s host=%s",
                    ws_id,
                    host_id,
                )
                old.retirement_reason = "replaced"
                old.outbound_queue.put_nowait(None)
                _fail_pending_imports(old)
                _fail_pending_harness_startup(old)
                _fail_pending_plugins(old)
                _fail_pending_skill_content(old)
                _fail_pending_mcp_tools(old)
                _fail_pending_skills(old)
                _fail_pending_mcp_servers(old)
                _fail_pending_requests(old)
            self._hosts[key] = conn
            self._registration_epochs[key] = epoch + 1
            if hello.interactive_shells is not None:
                self._interactive_shells[host_id] = normalize_interactive_shells(
                    hello.interactive_shells
                )
            else:
                self._interactive_shells.pop(host_id, None)
        return conn

    def deregister(
        self,
        host_id: str,
        workspace_id: int | None = None,
        conn: HostConnection | None = None,
    ) -> bool:
        """Remove a host connection and end its sender loop.

        No-op if ``(workspace_id, host_id)`` is not registered.

        :param host_id: Host identifier to remove, in any accepted
            spelling (see :func:`_canonical_host_id`).
        :param workspace_id: Tenant partition; defaults to
            :func:`current_workspace_id`.
        :param conn: Optional generation guard, as on
            :meth:`TunnelRegistry.deregister`. When given, the entry is
            removed only if it is still this exact connection.
        :returns: ``True`` when an entry was removed. ``False`` means
            nothing was registered or the guard did not match, so the
            caller is superseded and must not flip the host's durable
            row offline — that row describes the live reconnect.
        """
        ws_id = current_workspace_id() if workspace_id is None else workspace_id
        with self._lock:
            key = (ws_id, _canonical_host_id(host_id))
            current = self._hosts.get(key)
            if current is None or (conn is not None and current is not conn):
                return False
            removed = self._hosts.pop(key)
        # Without this the route handler's loops keep running and its ping loop
        # keeps the host row online, even though the host is now unreachable.
        removed.retirement_reason = "deregistered"
        removed.outbound_queue.put_nowait(None)
        _fail_pending_imports(removed)
        _fail_pending_harness_startup(removed)
        _fail_pending_plugins(removed)
        _fail_pending_skill_content(removed)
        _fail_pending_mcp_tools(removed)
        _fail_pending_skills(removed)
        _fail_pending_mcp_servers(removed)
        _fail_pending_requests(removed)
        return True

    def mark_frame_seen(self, conn: HostConnection) -> bool:
        """Record that a frame arrived for ``conn``.

        :param conn: Connection that received the frame.
        :returns: ``True`` if the connection is still current,
            ``False`` if it has been replaced or deregistered.
        """
        with self._lock:
            if self._hosts.get((conn.workspace_id, conn.host_id)) is not conn:
                return False
            conn.last_frame_at = time.time()
            conn.last_frame_monotonic = time.monotonic()
            return True

    def get(self, host_id: str, workspace_id: int | None = None) -> HostConnection | None:
        """Look up a live host connection.

        :param host_id: Host identifier, in any accepted spelling
            (see :func:`_canonical_host_id`).
        :param workspace_id: Tenant partition; defaults to
            :func:`current_workspace_id`.
        :returns: The :class:`HostConnection` if online,
            otherwise ``None``.
        """
        ws_id = current_workspace_id() if workspace_id is None else workspace_id
        with self._lock:
            return self._hosts.get((ws_id, _canonical_host_id(host_id)))

    def registration_epoch(self, host_id: str, workspace_id: int | None = None) -> int:
        """Count the connections ever registered for a host on this replica.

        A tunnel reads it before checking its credential and passes it to
        :meth:`register` as ``unless_registered_since``. Disconnects leave the
        count unchanged, so a connection that registered and left meanwhile
        still counts.

        :param host_id: Host identifier, in any accepted spelling.
        :param workspace_id: Tenant partition; defaults to
            :func:`current_workspace_id`.
        :returns: The number of registrations so far, ``0`` for a new host.
        """
        ws_id = current_workspace_id() if workspace_id is None else workspace_id
        with self._lock:
            return self._registration_epochs.get((ws_id, _canonical_host_id(host_id)), 0)

    def online_host_ids(self, workspace_id: int | None = None) -> list[str]:
        """Return IDs of all hosts connected in one workspace.

        :param workspace_id: Tenant partition; defaults to
            :func:`current_workspace_id`.
        :returns: List of host_id strings live in the workspace.
        """
        ws_id = current_workspace_id() if workspace_id is None else workspace_id
        with self._lock:
            return [hid for (ws, hid) in self._hosts if ws == ws_id]

    def is_host_telemetry_opted_out(self, host_id: str, workspace_id: int | None = None) -> bool:
        """Return whether the host has opted out of telemetry.

        :param host_id: Host identifier, e.g. ``"host_a1b2c3d4..."``.
        :param workspace_id: Tenant partition; defaults to
            :func:`current_workspace_id`.
        :returns: ``True`` when the host sent ``telemetry_opt_out=True``
            in its hello frame.  Defaults to ``False`` when the host is
            offline or unknown.
        """
        conn = self.get(host_id, workspace_id)
        if conn is None:
            return False
        return conn.hello.telemetry_opt_out

    def get_host_installation_id(
        self, host_id: str, workspace_id: int | None = None
    ) -> str | None:
        """Return the installation ID the host advertised in its hello frame.

        :param host_id: Host identifier, e.g. ``"host_a1b2c3d4..."``.
        :param workspace_id: Tenant partition; defaults to
            :func:`current_workspace_id`.
        :returns: The host's installation ID, or ``None`` when offline or
            not set.
        """
        conn = self.get(host_id, workspace_id)
        if conn is None:
            return None
        return conn.hello.installation_id

    def host_supports_codex_side_chat(self, host_id: str, workspace_id: int | None = None) -> bool:
        """Whether the connected host's build can fork a codex `/side` chat.

        Reads the capability the host advertised in its hello frame. An older
        host that predates the feature sends no such token, so it reads as
        unsupported with no version check. Fails OPEN (``True``) only when the
        host is offline/unknown — we then can't prove it's too old, and the
        forward will surface any real connection failure on its own.

        :param host_id: Host identifier, e.g. ``"host_a1b2c3d4..."``.
        :param workspace_id: Tenant partition; defaults to
            :func:`current_workspace_id`.
        :returns: ``False`` only for a connected host that did not advertise the
            codex side-chat capability.
        """
        conn = self.get(host_id, workspace_id)
        if conn is None:
            return True
        return CAP_CODEX_SIDE_CHAT in conn.hello.capabilities

    def record_gateway_inference(
        self,
        host_id: str,
        gateway_inference: Mapping[str, bool] | None,
    ) -> None:
        """Store the gateway-inference map a host just reported.

        Called for every frame that carries the map — the connect handshake and
        each readiness refresh — so the server's view is delivered rather than
        persisted. ``None`` (a host that cannot evaluate the map at all) clears
        the entry back to unknown instead of recording "nothing is backed".

        :param host_id: Host identifier, in any accepted spelling (see
            :func:`_canonical_host_id`).
        :param gateway_inference: Harness spelling → gateway-backed flag, e.g.
            ``{"claude-native": True, "codex": False}``, or ``None``.
        """
        key = _canonical_host_id(host_id)
        with self._lock:
            if gateway_inference is None:
                self._gateway_inference.pop(key, None)
            else:
                self._gateway_inference[key] = dict(gateway_inference)

    def gateway_inference(self, host_id: str) -> dict[str, bool] | None:
        """Return the gateway-inference map *host_id* last reported here.

        :param host_id: Host identifier, in any accepted spelling.
        :returns: A copy of the reported map, or ``None`` when this replica has
            never had a report from the host — unknown, which readers treat as
            gateway-backed rather than unavailable.
        """
        with self._lock:
            reported = self._gateway_inference.get(_canonical_host_id(host_id))
        return dict(reported) if reported is not None else None

    def interactive_shells(self, host_id: str) -> list[str] | None:
        """Return the ordered shell inventory last reported by *host_id*."""
        with self._lock:
            reported = self._interactive_shells.get(_canonical_host_id(host_id))
        return list(reported) if reported is not None else None

    async def admit_launch(
        self,
        conn: HostConnection,
        session_id: str,
        *,
        allow_unbound: bool = False,
        transfer_from_host_id: str | None = None,
    ) -> None:
        """Reauthorize immediately before a new runner binding is created."""
        if self.admission_gate is not None:
            await self.admission_gate(conn.host_id, "runner_launch")
        if self.launch_authorizer is not None:
            await asyncio.to_thread(
                self.launch_authorizer,
                conn.host_id,
                session_id,
                conn.owner,
                conn.account_generation,
                allow_unbound,
                transfer_from_host_id,
            )

    async def await_answer(
        self,
        conn: HostConnection,
        pending: str,
        request_id: str,
        frame: str,
        answer: asyncio.Future[_AnswerT],
        *,
        timeout_s: float,
    ) -> _AnswerT:
        """
        Await a host request's answer, sending it again if its tunnel closes first.

        A host that keeps operation records answers a request it already
        received with the recorded result instead of acting again (see
        :mod:`omnigent.host.operation_records`). So when the tunnel closes
        before the answer, the request is sent again under the same
        ``request_id`` once the same host process reconnects, within the
        caller's budget. A restarted host or an older one leaves the outcome
        unknown, as before.

        :param conn: The connection the request was sent on.
        :param pending: Name of its map of pending answers, e.g.
            ``"pending_installs"``, where *answer* is registered.
        :param request_id: The request's id, e.g. ``"9f2c41d07ab35e18"``.
        :param frame: The encoded request frame, as sent.
        :param answer: The future registered for the answer.
        :param timeout_s: The caller's whole budget, e.g. ``30.0``.
        :returns: The host's answer.
        :raises asyncio.TimeoutError: When no answer arrived within the budget.
        :raises HostOperationOutcomeUnknownError: When the tunnel closed and the
            request could not be sent again to the same host process in time.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        sent_at = time.monotonic()
        current, waiting = conn, answer
        repeated_from: list[HostConnection] = []
        try:
            while True:
                try:
                    result = await asyncio.wait_for(waiting, max(0.0, deadline - loop.time()))
                except HostOperationOutcomeUnknownError:
                    successor = await self._same_process_reconnect(current, deadline, sent_at)
                    if successor is None:
                        raise
                else:
                    for earlier in repeated_from:
                        _release_unanswered(earlier, request_id)
                    return result
                if current is not conn:
                    getattr(current, pending).pop(request_id, None)
                repeated_from.append(current)
                current, waiting = successor, loop.create_future()
                getattr(current, pending)[request_id] = waiting
                try:
                    self.send_text(current, frame)
                except ConnectionError as exc:
                    raise HostOperationOutcomeUnknownError(
                        f"host '{conn.host_id}' could not be asked again: {exc}"
                    ) from exc
                _logger.info(
                    "Sent host request %s again after its tunnel closed",
                    request_id,
                    extra={"host_id": conn.host_id},
                )
        finally:
            if current is not conn:
                getattr(current, pending).pop(request_id, None)

    async def _same_process_reconnect(
        self, closed: HostConnection, deadline: float, sent_at: float
    ) -> HostConnection | None:
        """Wait for the host process behind *closed* to reconnect, until *deadline*."""
        incarnation = closed.hello.incarnation
        if incarnation is None or CAP_OPERATION_RECORDS not in closed.hello.capabilities:
            return None
        loop = asyncio.get_running_loop()
        # Past the retention the host may have forgotten the request and would run it again.
        while time.monotonic() - sent_at < OPERATION_RECORD_RETENTION_S:
            current = self.get(closed.host_id, closed.workspace_id)
            if current is not None and current is not closed:
                same_process = (
                    current.hello.incarnation == incarnation
                    and CAP_OPERATION_RECORDS in current.hello.capabilities
                )
                # Owned as when the request was sent: an owner change ends the request.
                same_owner = (current.owner, current.account_generation) == (
                    closed.owner,
                    closed.account_generation,
                )
                return current if same_process and same_owner else None
            remaining = deadline - loop.time()
            if remaining <= 0:
                return None
            await asyncio.sleep(min(_RECONNECT_POLL_S, remaining))
        return None

    def send_text(self, conn: HostConnection, data: str) -> None:
        """Enqueue a text frame for sending to the host.

        Must be called on the host WebSocket's owning event loop.
        ``asyncio.Queue`` is coroutine-safe within a single loop, NOT
        thread-safe — ``put_nowait`` mutates the underlying deque
        without a lock. Every current caller (REST handlers, the WS
        receive loop, the ping loop) runs on the uvicorn event loop,
        so the call below is safe. A caller on another thread must use
        ``loop.call_soon_threadsafe(conn.enqueue, data)`` instead.

        :param conn: The target host connection.
        :param data: JSON-encoded frame text.
        :raises ConnectionError: If the connection has been
            replaced or its owner does not match captured authority.
        """
        expected = _expected_host_owner.get()
        if (
            expected is not None
            and expected.workspace_id == current_workspace_id()
            and (conn.workspace_id, conn.owner, conn.account_generation)
            != (expected.workspace_id, expected.user_id, expected.generation)
        ):
            raise ConnectionError(f"host {conn.host_id!r} account ownership changed")
        actor = current_account_user()
        if (
            actor is not None
            and actor == conn.owner
            and account_generation(actor) != conn.account_generation
        ):
            raise ConnectionError(f"host {conn.host_id!r} account registration changed")
        with self._lock:
            current = self._hosts.get((conn.workspace_id, conn.host_id))
            if current is not conn:
                raise ConnectionError(f"host {conn.host_id!r} connection was replaced")
        if self.send_guard is not None:
            self.send_guard(conn)

        conn.enqueue(data)
