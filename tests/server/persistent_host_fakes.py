"""An in-memory persistent computer and host for the persistent-host lifecycle tests.

:class:`FakeComputer` stands in for a provider whose sandbox reports a typed
status and fences host starts with start claims, the contract a persistent
provider implements (see ``SandboxLifecycle.claim_start``). Its methods record
what Omnigent asked for, and its knobs inject the faults the lifecycle must
survive: lost replies, unanswered status reports, slow or refused stops.

:class:`FakeHost` is the host process on that computer: it connects to the
registry only with a launch token the store accepts, answers ``host.quiesce``
from its runner and operation lists (and directory listings with an empty
directory), and reconnects with its current token.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

import click

from omnigent.host.frames import (
    CAP_QUIESCE,
    HostHelloFrame,
    HostListDirFrame,
    HostQuiesceFrame,
    HostQuiesceResultFrame,
    decode_host_frame,
)
from omnigent.onboarding.sandboxes.base import SandboxGoneError, SandboxHostLauncher
from omnigent.onboarding.sandboxes.types import (
    AdmissionReport,
    HostSandboxStatus,
    RepoWorkspace,
    StartClaim,
    SuspendResult,
)
from omnigent.server.host_registry import HostConnection, HostRegistry
from omnigent.stores.host_store import HostStore

HOST_ID = "5b0e7c1d9a2f4e6b8c3d1a0f9e8d7c6b"
OWNER = "local"
SANDBOX_ID = "sb-persistent-1"
FIRST_TOKEN = "token-generation-1"


class LostReplyError(RuntimeError):
    """The provider acted, but its reply never reached Omnigent."""


class _FakeWs:
    """Accepts frames without a peer; the fake host reads the outbound queue."""

    async def send_text(self, data: str) -> None:
        del data

    async def close(self, code: int = 1000, reason: str = "") -> None:
        del code, reason


@dataclass
class FakeComputer(SandboxHostLauncher):
    """A persistent sandbox with a typed status and fenced host starts.

    :ivar status: The computer's typed status, as the provider tracks it.
    :ivar events: Calls in order: ``resume``, ``claim``, ``start_host``,
        ``suspend``, ``stopped``, ``terminate``, ``provision``.
    """

    provider: ClassVar[str] = "fakepersist"
    can_resume: ClassVar[bool] = True

    status: HostSandboxStatus = "host_running"
    attempt: int = 0
    generation: int = 1
    events: list[str] = field(default_factory=list)
    admitting: bool = True
    refusal: tuple[str, str] = ("not_saved", "the last save is 35 minutes old")
    # Admission calls that fail before one answers (an unreachable platform).
    unanswered_reports: int = 0
    admission_gate: threading.Event | None = None
    parked: threading.Event = field(default_factory=threading.Event)
    gone: bool = False
    claims: dict[str, StartClaim] = field(default_factory=dict)
    lose_claim_replies: int = 0
    # Runs inside claim_start before the claim is decided (e.g. a supervisor restart).
    before_claim: Callable[[], None] | None = None
    accepted_starts: list[tuple[int | None, str]] = field(default_factory=list)
    refused_starts: list[int | None] = field(default_factory=list)
    lose_start_reply: bool = False
    # A lost start that never reached the supervisor.
    drop_start: bool = False
    suspend_gate: threading.Event | None = None
    suspend_accepted: threading.Event = field(default_factory=threading.Event)
    suspend_refuses: bool = False
    lose_suspend_replies: int = 0
    suspend_operations: list[str] = field(default_factory=list)
    outcomes: dict[str, SuspendResult] = field(default_factory=dict)
    in_flight: set[str] = field(default_factory=set)
    provision_error: str | None = None
    on_provision: Callable[[], None] | None = None
    # The supervisor taking a start's parameters: the host process connects.
    on_host_start: Callable[[str], None] | None = None

    def prepare(self) -> None:
        return None

    def provision(self, name: str) -> str:
        del name
        self.events.append("provision")
        if self.on_provision is not None:
            self.on_provision()
        if self.provision_error is not None:
            raise click.ClickException(self.provision_error)
        self.status = "stopped"
        return SANDBOX_ID

    def admission(self, sandbox_id: str) -> AdmissionReport | None:
        del sandbox_id
        if self.gone:
            raise SandboxGoneError("the computer was deleted")
        if self.admission_gate is not None:
            gate, self.admission_gate = self.admission_gate, None
            self.parked.set()
            gate.wait(10)
        if self.unanswered_reports > 0:
            self.unanswered_reports -= 1
            raise LostReplyError("platform unreachable")
        reason, message = self.refusal
        return AdmissionReport(
            admitting=self.admitting,
            reason=None if self.admitting else reason,
            message=None if self.admitting else message,
            generation=str(self.generation),
            status=self.status,
        )

    def resume(self, sandbox_id: str) -> None:
        del sandbox_id
        if self.gone:
            raise SandboxGoneError("the computer was deleted")
        self.events.append("resume")
        if self.status == "stopped":
            self.generation += 1
            self.status = "started_no_host"

    def claim_start(self, sandbox_id: str, *, operation_id: str) -> StartClaim:
        del sandbox_id
        if operation_id in self.claims:
            return self.claims[operation_id]
        if self.before_claim is not None:
            hook, self.before_claim = self.before_claim, None
            hook()
        self.events.append("claim")
        if self.status in ("started_no_host", "start_claimed", "host_refused"):
            self.attempt += 1
            self.status = "start_claimed"
            claim = StartClaim(granted=True, attempt=self.attempt)
        else:
            claim = StartClaim(granted=False, status=self.status)
        self.claims[operation_id] = claim
        if self.lose_claim_replies > 0:
            self.lose_claim_replies -= 1
            raise LostReplyError("claim reply lost")
        return claim

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
        start_attempt: int | None = None,
    ) -> str:
        del sandbox_id, host_id, host_name, server_url, repos, host_config, on_stage
        if self.drop_start:
            self.drop_start = False
            raise LostReplyError("start never arrived")
        self.deliver_start(start_attempt, token)
        if self.lose_start_reply:
            self.lose_start_reply = False
            raise LostReplyError("start reply lost")
        return "/root/workspace"

    def deliver_start(self, attempt: int | None, token: str) -> bool:
        """A ``start_host`` reaching the platform, from this or an older server."""
        if attempt != self.attempt or self.status != "start_claimed":
            self.refused_starts.append(attempt)
            return False
        self.events.append("start_host")
        self.accepted_starts.append((attempt, token))
        self.status = "host_running"
        if self.on_host_start is not None:
            self.on_host_start(token)
        return True

    def is_running(self, sandbox_id: str) -> bool | None:
        del sandbox_id
        return self.status not in ("stopped",)

    def suspend(self, sandbox_id: str, *, operation_id: str) -> SuspendResult:
        del sandbox_id
        self.suspend_operations.append(operation_id)
        if operation_id in self.outcomes:
            result = self.outcomes[operation_id]
        elif operation_id in self.in_flight:
            return SuspendResult(outcome="unknown")  # the first call is still stopping
        else:
            self.in_flight.add(operation_id)
            self.events.append("suspend")
            self.suspend_accepted.set()
            if self.suspend_gate is not None:
                self.suspend_gate.wait(10)
            if self.suspend_refuses:
                result = SuspendResult(outcome="refused", reason="saving")
            else:
                self.status = "stopped"
                self.events.append("stopped")
                result = SuspendResult(outcome="stopped")
            self.outcomes[operation_id] = result
        if self.lose_suspend_replies > 0:
            self.lose_suspend_replies -= 1
            raise LostReplyError("suspend reply lost")
        return result

    def terminate(self, sandbox_id: str) -> None:
        del sandbox_id
        self.events.append("terminate")


@dataclass
class FakeHost:
    """The host process on a :class:`FakeComputer`, wired to a real store and registry."""

    store: HostStore
    registry: HostRegistry
    computer: FakeComputer
    runners: list[str] = field(default_factory=list)
    operations: list[str] = field(default_factory=list)
    quiesce: bool = True
    host_id: str = HOST_ID
    token: str | None = None
    refused_tokens: list[str] = field(default_factory=list)
    _answerers: list[asyncio.Task[None]] = field(default_factory=list)

    def connect(self, token: str | None = None) -> bool:
        """Dial in with *token* (default: the current one); refused unless the store accepts it."""
        token = token if token is not None else self.token
        assert token is not None
        row = self.store.resolve_launch_token(self.host_id, token)
        if row is None:
            self.refused_tokens.append(token)
            return False
        self.token = token
        self.store.upsert_on_connect(
            host_id=self.host_id, name=row.name, user_id=OWNER, managed_token=token
        )
        hello = HostHelloFrame(
            version="test",
            frame_protocol_version=1,
            name=row.name,
            capabilities=[CAP_QUIESCE] if self.quiesce else [],
        )
        conn = self.registry.register(
            self.host_id, _FakeWs(), hello, owner=OWNER, registered_with_managed_token=True
        )
        self._answerers.append(asyncio.get_running_loop().create_task(self._answer(conn)))
        return True

    def drop(self) -> None:
        """A host-only drop: the tunnel goes, the process and its runners stay."""
        self.registry.deregister(self.host_id)
        self.store.set_offline(self.host_id)

    async def _answer(self, conn: HostConnection) -> None:
        while True:
            outbound = await conn.outbound_queue.get()
            if outbound is None:
                return
            frame = decode_host_frame(outbound.data)
            if isinstance(frame, HostListDirFrame):
                listing = conn.pending_list_dirs.pop(frame.request_id, None)
                if listing is not None and not listing.done():
                    listing.set_result({"status": "ok", "entries": [], "has_more": False})
            if isinstance(frame, HostQuiesceFrame):
                future = conn.pending_quiesce.get(frame.request_id)
                if future is not None and not future.done():
                    future.set_result(
                        HostQuiesceResultFrame(
                            request_id=frame.request_id,
                            runners=list(self.runners),
                            operations=list(self.operations),
                        )
                    )

    def close(self) -> None:
        for task in self._answerers:
            task.cancel()


def register_persistent_row(store: HostStore, *, token: str = FIRST_TOKEN) -> None:
    """Write a persistent host row with its sandbox and an armed launch token."""
    from omnigent.db.utils import now_epoch

    store.create_persistent_host(
        host_id=HOST_ID, name="computer", user_id=OWNER, provider=FakeComputer.provider
    )
    attached = store.attach_persistent_sandbox(HOST_ID, sandbox_id=SANDBOX_ID)
    assert attached is not None
    armed = store.rearm_managed_host(
        HOST_ID,
        sandbox_id=SANDBOX_ID,
        expected_updated_at=attached.updated_at,
        token=token,
        token_expires_at=now_epoch() + 3600,
    )
    assert armed is not None


def call_soon(loop: asyncio.AbstractEventLoop, callback: Callable[..., Any], *args: Any) -> None:
    """Run *callback* on *loop* from a provider thread, as the supervisor's host would."""
    loop.call_soon_threadsafe(callback, *args)
