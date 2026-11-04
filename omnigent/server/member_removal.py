"""Removing a team member, and where each removal stands.

A removal is one database transaction
(:meth:`~omnigent.server.accounts_store.SqlAlchemyAccountStore.remove_member`):
the account is tombstoned, every credential and grant it holds is revoked, the
membership is retired, the invitations they could still use are revoked, and a
record of the removal is written. Once it commits, every request with one of
their credentials is refused.

Then, outside the database, the same steps a suspension takes: cached sessions
are dropped, open streams and tunnels re-check their caller at once, and a
front proxy is told who to admit. Last, their cloud computers are handed to
their provider for removal. Each of these is safe to repeat, and each is
retried on its own anyway: the admission sync pushes at start and hourly, and
the managed-sandbox reaper retries a removal the provider did not confirm.
Asking to remove the member again runs them all once more.

The team page reads the record back: access blocked, the proxy's list
updated, computers handed over, and device registrations, which a deployment
with per-member devices reports through :class:`DeviceRegistrations`.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import TYPE_CHECKING, Literal, Protocol

from omnigent.server.authority_changes import authority_changed

if TYPE_CHECKING:
    from starlette.requests import HTTPConnection

    from omnigent.entities.team import MemberRemoval
    from omnigent.server.accounts_store import SqlAlchemyAccountStore
    from omnigent.server.auth import AuthProvider
    from omnigent.server.team_admission import AdmissionSync
    from omnigent.stores.conversation_store import ConversationStore
    from omnigent.stores.scheduled_task_store import ScheduledTaskStore
    from omnigent.stores.team_store import SqlAlchemyTeamStore

_logger = logging.getLogger(__name__)

StepState = Literal["done", "pending", "not_used"]
#: The account a deployment names as its installation's own owner, e.g. the one its own
#: owner pages are built around. The team never removes it; unset, only the last owner is kept.
INSTALLATION_OWNER_ENV = "OMNIGENT_INSTALLATION_OWNER"
# Threads listed on one removal's page; the count covers the rest.
MAX_LISTED_THREADS = 500


def installation_owner() -> str | None:
    """The account named in :data:`INSTALLATION_OWNER_ENV`, or ``None``."""
    return os.environ.get(INSTALLATION_OWNER_ENV, "").strip().lower() or None


class DeviceRegistrations(Protocol):
    """A deployment's per-member device registrations, revoked when a member is removed."""

    def revoke(self, member_id: str) -> None:
        """Revoke every registration of *member_id*; raise when that is not confirmed."""

    def remaining(self, member_id: str) -> int:
        """How many registrations of *member_id* are not revoked yet."""


class MemberRemovals:
    """Remove team members and report each removal's steps.

    :param account_store: Runs the removal's transaction.
    :param team_store: Reads removal records back.
    :param auth_provider: Drops the removed person's cached sessions.
    :param admission_sync: Tells a front proxy who to admit, when one is wired.
    :param scheduled_task_store: Finds their schedules' in-memory timers.
    :param conversation_store: Titles for the threads a removal lists.
    :param devices: Per-member device registrations, when the deployment has them.
    """

    def __init__(
        self,
        account_store: SqlAlchemyAccountStore,
        team_store: SqlAlchemyTeamStore,
        *,
        auth_provider: AuthProvider | None,
        admission_sync: AdmissionSync | None = None,
        scheduled_task_store: ScheduledTaskStore | None = None,
        conversation_store: ConversationStore | None = None,
        devices: DeviceRegistrations | None = None,
    ) -> None:
        self._accounts = account_store
        self._team = team_store
        self._auth_provider = auth_provider
        self._admission_sync = admission_sync
        self._schedules = scheduled_task_store
        self._conversations = conversation_store
        self._devices = devices

    async def remove(
        self, connection: HTTPConnection, member_id: str, *, actor_member_id: str
    ) -> MemberRemoval:
        """Remove *member_id*, or repeat an earlier removal's outside steps.

        :raises OmnigentError: ``NOT_FOUND``, ``FORBIDDEN`` or
            :class:`~omnigent.stores.team_store.LastActiveOwnerError`.
        """
        member = await asyncio.to_thread(self._team.get_member, member_id)
        timers = (
            await asyncio.to_thread(self._schedule_timers, connection, member.user_id)
            if member is not None and member.state != "removed"
            else []
        )
        removal = await asyncio.to_thread(
            self._accounts.remove_member,
            member_id,
            actor_member_id=actor_member_id,
            protected_user_id=installation_owner(),
        )
        _logger.info("team: %s removed %s", actor_member_id, removal.member_id)
        await self.finish(connection, removal, timers)
        # Read back: the computers handed over just now count as done.
        return await asyncio.to_thread(self._team.get_removal, member_id) or removal

    def _schedule_timers(self, connection: HTTPConnection, user_id: str) -> list[str]:
        """The schedules whose timers this process arms, read before the removal disables them."""
        persistent = getattr(connection.app.state, "scheduled_task_dispatch_enabled", False)
        if self._schedules is None or persistent:
            return []
        return [task.id for task in self._schedules.list(owner_user_id=user_id)]

    async def finish(
        self, connection: HTTPConnection, removal: MemberRemoval, timers: list[str]
    ) -> None:
        """The steps outside the database; each is safe to repeat."""
        user_id = removal.user_id
        revoke = getattr(self._auth_provider, "revoke_user_sessions", None)
        if callable(revoke):
            revoke(user_id)
        authority_changed()
        if self._admission_sync is not None:
            self._admission_sync.request()
        scheduler = getattr(connection.app.state, "scheduled_task_scheduler", None)
        if scheduler is not None:
            for task_id in timers:
                scheduler.remove(task_id)
        if self._devices is not None:
            try:
                await asyncio.to_thread(self._devices.revoke, removal.member_id)
            except Exception:
                _logger.exception("Revoking removed member %s's devices failed", removal.member_id)
        from omnigent.server import persistent_hosts

        try:
            await persistent_hosts.remove_deleted_account_hosts(user_id)
        except Exception:
            # The tombstones stay; the reaper and the next request retry them.
            _logger.exception("Removing removed member %s's computers failed", removal.member_id)

    def member_group_state(self, removal: MemberRemoval) -> StepState:
        """Whether a front proxy has applied a list without them (it reads it back first)."""
        if self._admission_sync is None:
            return "not_used"
        applied = self._admission_sync.applied_revision
        return (
            "done" if applied is not None and applied >= removal.admission_revision else "pending"
        )

    def status(self, removal: MemberRemoval) -> dict[str, object]:
        """The removal's steps for the team page (reads the store: call it in a thread)."""
        remover = (
            self._team.get_member(removal.removed_by_member_id)
            if removal.removed_by_member_id is not None
            else None
        )
        computers_state: StepState = (
            "not_used"
            if removal.computers == 0
            else "pending"
            if removal.computers_pending
            else "done"
        )
        devices_state: StepState = "not_used"
        devices_remaining = 0
        if self._devices is not None:
            devices_remaining = self._devices.remaining(removal.member_id)
            devices_state = "pending" if devices_remaining else "done"
        return {
            "member_id": removal.member_id,
            "user_id": removal.user_id,
            "removed_at": removal.removed_at,
            "removed_by": remover.user_id if remover is not None else None,
            "thread_count": removal.thread_count,
            "steps": {
                # The removal's own transaction refused every credential at once.
                "access": "done",
                "member_group": self.member_group_state(removal),
                "computers": {
                    "state": computers_state,
                    "total": removal.computers,
                    "handed_over": removal.computers - removal.computers_pending,
                },
                "devices": {"state": devices_state, "remaining": devices_remaining},
            },
        }

    def threads(self, member_id: str) -> list[dict[str, object]]:
        """The threads a removed member owned, with titles where they can be read."""
        listed = []
        for thread_id in self._team.removal_threads(member_id, limit=MAX_LISTED_THREADS):
            title = None
            if self._conversations is not None:
                try:
                    conversation = self._conversations.get_conversation(thread_id)
                except Exception:  # noqa: BLE001 — a title is a convenience
                    conversation = None
                title = getattr(conversation, "title", None) or None
            listed.append({"session_id": thread_id, "title": title})
        return listed
