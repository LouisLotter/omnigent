"""Keep a front proxy's admission list in step with the team.

An identity-aware proxy in front of Omnigent (for example a Cloudflare Access
group) admits people before Omnigent sees them. This module pushes the team's
:class:`~omnigent.entities.team.AdmissionSnapshot` (active members' verified
addresses and pending invitations) to a deployment-provided
:class:`MembershipAdmission` after every change, and again periodically so an
expired invitation drops out. Omnigent's own membership check never depends on
it: a person the proxy still admits but Omnigent has suspended reaches nothing.

Each change is committed with a higher admission revision before it is pushed
(persist before effect); the adapter refuses a snapshot older than the last
one it applied, so a late, older push cannot undo a newer one, and compares an
equal one with what the proxy holds, so drift is repaired. Expired invitations
are swept (a revision bump) before every push, and a push is due again when
the next invitation expires. A failed read or push is retried with backoff.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from omnigent.entities.team import AdmissionSnapshot
    from omnigent.stores.team_store import SqlAlchemyTeamStore

_logger = logging.getLogger(__name__)

RETRY_INITIAL_S = 5.0
RETRY_MAX_S = 300.0
PERIODIC_S = 3600.0


class MembershipAdmission(Protocol):
    """A deployment's adapter that applies who its front proxy admits."""

    def apply(self, snapshot: AdmissionSnapshot) -> None:
        """Apply *snapshot*; raise when the outcome is not confirmed."""


class AdmissionSync:
    """Push the team's admission list after changes, retrying until applied.

    :param team_store: Reads the current snapshot.
    :param admission: The deployment's adapter.
    """

    def __init__(self, team_store: SqlAlchemyTeamStore, admission: MembershipAdmission) -> None:
        self._team_store = team_store
        self._admission = admission
        self._wake = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self.applied_revision: int | None = None
        self.next_change_at: int | None = None

    def start(self) -> None:
        """Start the background loop on the running event loop; idempotent."""
        if self._task is None or self._task.done():
            self._task = asyncio.get_running_loop().create_task(self._run())

    async def stop(self) -> None:
        """Stop the background loop."""
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None

    def request(self) -> None:
        """Ask for a push soon; called after a committed membership change."""
        self._wake.set()
        # Without a running loop (a synchronous caller) the next start covers it.
        with contextlib.suppress(RuntimeError):
            self.start()

    async def push_once(self) -> bool:
        """Sweep expiries, read and apply the current snapshot; ``True`` when confirmed.

        Every push is applied, an unchanged revision included, so the proxy
        compares it with what it holds. Nothing here raises but cancellation.
        """
        try:
            await asyncio.to_thread(self._team_store.expire_invitations)
            snapshot = await asyncio.to_thread(self._team_store.admission_snapshot)
        except Exception as exc:  # noqa: BLE001 — a database blip is retried
            _logger.warning("team admission: snapshot not read (%s)", type(exc).__name__)
            return False
        self.next_change_at = snapshot.next_change_at
        try:
            await asyncio.to_thread(self._admission.apply, snapshot)
        except Exception as exc:  # noqa: BLE001 — retried; never fatal to the server
            _logger.warning(
                "team admission: revision %s not applied (%s)",
                snapshot.revision,
                type(exc).__name__,
            )
            return False
        self.applied_revision = snapshot.revision
        _logger.info(
            "team admission: revision %s applied (%d addresses)",
            snapshot.revision,
            len(snapshot.emails),
        )
        return True

    def _until_next_change(self) -> float:
        if self.next_change_at is None:
            return PERIODIC_S
        return min(PERIODIC_S, max(1.0, self.next_change_at - time.time() + 1))

    async def _run(self) -> None:
        delay = RETRY_INITIAL_S
        while True:
            self._wake.clear()
            if await self.push_once():
                delay = RETRY_INITIAL_S
                timeout = self._until_next_change()
            else:
                timeout = delay
                delay = min(delay * 2, RETRY_MAX_S)
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._wake.wait(), timeout=timeout)
