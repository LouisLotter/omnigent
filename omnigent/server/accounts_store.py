"""Accounts-mode credentials, durable revocation, and token issuance.

Accounts and permission stores share the users table. Accounts mode owns
passwords, generation IDs, deletion tombstones, and invite/magic-login tokens.
Header/OIDC identities leave the accounts-specific columns unset.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar, cast

from sqlalchemy import and_, delete, exists, or_, select, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from omnigent.db.account_authority import (
    account_authority_scope,
    current_account_user,
    lock_account,
    require_active_account,
)
from omnigent.db.db_models import (
    SqlAccountToken,
    SqlConnection,
    SqlConversationMetadata,
    SqlDeviceGrant,
    SqlHost,
    SqlPreference,
    SqlProject,
    SqlScheduledTask,
    SqlSessionPermission,
    SqlTeamMember,
    SqlUser,
    SqlUserDailyCost,
    current_workspace_id,
)
from omnigent.db.enum_codecs import (
    decode_account_token_kind,
    decode_team_member_role,
    encode_account_token_kind,
    encode_device_grant_status,
    encode_scheduled_task_state,
    encode_team_member_role,
    encode_team_member_state,
)
from omnigent.db.utils import (
    get_or_create_engine,
    make_named_managed_session_maker,
    run_write_transaction,
)
from omnigent.entities import Account, AccountToken
from omnigent.entities.team import InvitationRole, MemberRemoval, MemberState, TeamMember
from omnigent.server.auth import RESERVED_USER_LOCAL, RESERVED_USER_PUBLIC
from omnigent.server.principals import MachineBinding
from omnigent.stores.approval_store import remove_account_approval_authority
from omnigent.stores.host_store import RETIRED_RUNNER_INCARNATION, resolve_runner_binding
from omnigent.stores.team_store import (
    InvitationEmailMismatchError,
    LastActiveOwnerError,
    admin_flag_for_account,
    apply_member_state,
    bind_identity_in_session,
    carry_membership_registration,
    claim_founding_owner,
    close_redeemed_invitation,
    insert_active_membership,
    lock_pending_invitation,
    lock_team,
    member_account,
    member_addresses,
    member_state_row,
    owned_before_removal,
    record_member_removal,
    removals_in_session,
    require_actor,
    retire_deleted_account_membership,
    revoke_invitations_for,
    to_member,
)

if TYPE_CHECKING:
    from omnigent.server.external_identity import ExternalIdentity

_HIDDEN_LIST_USERS = frozenset({RESERVED_USER_PUBLIC, RESERVED_USER_LOCAL})
_SUSPENDED = encode_team_member_state("suspended")


def _to_account(row: SqlUser) -> Account:
    """Convert a :class:`SqlUser` ORM row to an :class:`Account` entity.

    Strips ``password_hash`` — it never leaves the store via this
    conversion. Callers that need the hash use
    :meth:`SqlAlchemyAccountStore.get_password_hash` explicitly.
    """
    return Account(
        id=row.id,
        is_admin=row.is_admin,
        created_at=row.created_at,
        last_login_at=row.last_login_at,
        has_password=row.password_hash is not None,
        account_generation=row.account_generation,
    )


def _revoke_durable_authority(session: Session, user_id: str, *, now: int) -> None:
    """Disable everything *user_id* owns that could act without them present.

    Runs inside the caller's ``delete_user`` transaction. Hosts mirror
    :meth:`HostStore.delete_host`: bound sessions are detached, launch
    credentials cleared, and rows with pending sandbox cleanup are kept
    as tombstones instead of being dropped.
    """
    workspace_id = current_workspace_id()
    session.execute(
        delete(SqlUserDailyCost).where(
            SqlUserDailyCost.workspace_id == workspace_id,
            SqlUserDailyCost.user_id == user_id,
        )
    )
    session.execute(
        delete(SqlConnection).where(
            SqlConnection.workspace_id == workspace_id,
            SqlConnection.user_id == user_id,
        )
    )
    project_ids = select(SqlProject.id).where(
        SqlProject.workspace_id == workspace_id,
        SqlProject.user_id == user_id,
    )
    session.execute(
        update(SqlConversationMetadata)
        .where(
            SqlConversationMetadata.workspace_id == workspace_id,
            SqlConversationMetadata.project_id.in_(project_ids),
        )
        .values(project_id=None)
    )
    session.execute(
        delete(SqlProject).where(
            SqlProject.workspace_id == workspace_id,
            SqlProject.user_id == user_id,
        )
    )
    session.execute(
        update(SqlScheduledTask)
        .where(
            SqlScheduledTask.workspace_id == workspace_id,
            SqlScheduledTask.user_id == user_id,
            SqlScheduledTask.state != encode_scheduled_task_state("deleted"),
        )
        .values(state=encode_scheduled_task_state("deleted"), next_run_at=None, updated_at=now)
    )
    session.execute(
        update(SqlDeviceGrant)
        .where(
            SqlDeviceGrant.workspace_id == workspace_id,
            SqlDeviceGrant.user_id == user_id,
            SqlDeviceGrant.status != encode_device_grant_status("revoked"),
        )
        .values(
            status=encode_device_grant_status("revoked"),
            refresh_token_hash=None,
            prev_refresh_token_hash=None,
        )
    )
    from omnigent.stores.host_store import delete_host_in_session

    hosts = (
        session.execute(
            select(SqlHost)
            .where(
                SqlHost.workspace_id == workspace_id,
                SqlHost.user_id == user_id,
            )
            .order_by(SqlHost.host_id)
            .with_for_update()
        )
        .scalars()
        .all()
    )
    for host in hosts:
        delete_host_in_session(session, host)


def _tombstone_account(session: Session, target: SqlUser, admin_ids: list[str]) -> bool:
    """Tombstone the locked account *target* and revoke the authority it owns.

    Runs after the team lock and the deletion's account lock set. Shares,
    schedules, device and refresh grants, hosts, preferences, approval
    authority and tokens go in the same transaction, and the current team
    membership is retired with its identity bindings.

    :returns: ``False``, changing nothing, when *target* is the last admin or
        the team's last active owner; ``True`` once tombstoned.
    """
    user_id = target.id
    if target.is_admin and not [uid for uid in admin_ids if uid != user_id]:
        return False
    if not retire_deleted_account_membership(session, user_id):
        return False
    # A username registered again must not inherit approval authority.
    remove_account_approval_authority(session, user_id)
    session.execute(
        delete(SqlSessionPermission).where(
            SqlSessionPermission.workspace_id == current_workspace_id(),
            SqlSessionPermission.user_id == user_id,
        )
    )
    _revoke_durable_authority(session, user_id, now=int(time.time()))
    session.execute(
        delete(SqlPreference).where(
            SqlPreference.workspace_id == current_workspace_id(),
            SqlPreference.user_id == user_id,
        )
    )
    target.deleted_at = int(time.time())
    target.password_hash = None
    target.is_admin = False
    session.execute(
        delete(SqlAccountToken).where(
            SqlAccountToken.workspace_id == current_workspace_id(),
            or_(SqlAccountToken.user_id == user_id, SqlAccountToken.created_by == user_id),
        )
    )
    return True


def _carry_registration_work(session: Session, user_id: str, *, old: str, new: str) -> None:
    """Move *user_id*'s scheduled tasks and hosts from generation *old* to *new*.

    Runs under the account lock when the generation rotates. Tasks and hosts belong
    to the registration, so they keep working; grants and magic links stay revoked.
    Like :func:`_revoke_durable_authority`, this updates every row the user owns in
    one transaction, which only accounts mode does, and only on a password change.
    Team memberships move too, so a removal keeps applying to the registration.
    """
    workspace_id = current_workspace_id()
    carry_membership_registration(session, user_id, old=old, new=new)
    session.execute(
        update(SqlScheduledTask)
        .where(
            SqlScheduledTask.workspace_id == workspace_id,
            SqlScheduledTask.user_id == user_id,
            SqlScheduledTask.account_generation == old,
        )
        .values(account_generation=new)
    )
    session.execute(
        update(SqlHost)
        .where(
            SqlHost.workspace_id == workspace_id,
            SqlHost.user_id == user_id,
            SqlHost.account_generation == old,
        )
        .values(account_generation=new)
    )


def _retire_machine_credentials(session: Session, user_id: str) -> None:
    """End every launch token and runner binding *user_id* holds; keep hosts and schedules.

    Runs in a suspension's transaction. A reinstated member's computers get
    fresh launch tokens when they next start, and their threads fresh runners.
    """
    from omnigent.server.auth import LEVEL_OWNER

    workspace_id = current_workspace_id()
    session.execute(
        update(SqlHost)
        .where(
            SqlHost.workspace_id == workspace_id,
            SqlHost.user_id == user_id,
            SqlHost.token_hash.is_not(None),
        )
        .values(token_hash=None, token_expires_at=None)
    )
    owned = select(SqlSessionPermission.conversation_id).where(
        SqlSessionPermission.workspace_id == workspace_id,
        SqlSessionPermission.user_id == user_id,
        SqlSessionPermission.level >= LEVEL_OWNER,
    )
    session.execute(
        update(SqlConversationMetadata)
        .where(
            SqlConversationMetadata.workspace_id == workspace_id,
            SqlConversationMetadata.id.in_(owned),
            SqlConversationMetadata.runner_id.is_not(None),
        )
        .values(runner_host_incarnation=RETIRED_RUNNER_INCARNATION)
    )


def _to_account_token(row: SqlAccountToken) -> AccountToken:
    """Convert a :class:`SqlAccountToken` row to a domain entity."""
    return AccountToken(
        id=row.id,
        kind=decode_account_token_kind(row.kind),
        user_id=row.user_id,
        created_by=row.created_by,
        created_at=row.created_at,
        expires_at=row.expires_at,
        invited_is_admin=row.invited_is_admin,
        account_generation=row.account_generation,
    )


_T = TypeVar("_T")


class SqlAlchemyAccountStore:
    """SQLAlchemy-backed persistence for accounts-mode credentials and tokens.

    Concrete class (no ABC) — accounts persistence has exactly one
    backend today and a Protocol can be extracted later if a second
    appears. Constructor matches PermissionStore so the wiring in
    ``create_app`` is mechanical.

    :param storage_location: SQLAlchemy database URI, e.g.
        ``"sqlite:///omnigent.db"``. Shares the connection pool
        with PermissionStore via :func:`get_or_create_engine`.
    """

    def __init__(self, storage_location: str) -> None:
        self.storage_location = storage_location
        self._engine = get_or_create_engine(storage_location)
        self._session = make_named_managed_session_maker(
            self._engine,
            query_name_prefix="omnigent.account_store",
        )
        # Immediate session: for the last-admin invariant in delete_user,
        # which must lock the current admin set before counting it. On
        # SQLite, ``BEGIN IMMEDIATE`` acquires the write lock before the
        # first read, so a second concurrent delete/demote blocks instead
        # of reading the same stale admin count. On other dialects this is
        # a no-op — those paths lock explicitly with ``SELECT ... FOR
        # UPDATE`` instead (see ``_supports_for_update``).
        self._session_immediate = make_named_managed_session_maker(
            self._engine,
            query_name_prefix="omnigent.account_store",
            immediate=True,
        )
        self._supports_for_update = self._engine.dialect.name != "sqlite"

    # ── User credentials (extends rows in the `users` table) ──────

    def create_user_with_password(
        self,
        user_id: str,
        password_hash: str,
        *,
        is_admin: bool = False,
        team_owner: bool = False,
    ) -> Account:
        """Insert a user row with a password hash.

        Used by ``/auth/register`` (invite redemption), by the
        first-boot admin bootstrap, and by admin "create user"
        flows. Re-registration replaces a tombstone with a fresh generation.
        Raises if an active user already exists — registration
        UX should check uniqueness first to give a clean error.

        :param user_id: Chosen username, e.g. ``"alice"``.
        :param password_hash: Pre-hashed password (see
            :mod:`omnigent.server.passwords`). Plaintext never
            crosses this boundary.
        :param is_admin: Admin flag at creation. Defaults False;
            the first-boot admin bootstrap passes True.
        :param team_owner: Make the account the team's active owner when the
            team has no members yet; the first-admin claim passes True.
        :returns: The created :class:`Account`.
        :raises ValueError: If a user with this id already exists.
        """
        now = int(time.time())

        def write(session: Session) -> Account:
            if team_owner:
                lock_team(session)
            existing = lock_account(session, user_id)
            if existing is not None and existing.deleted_at is None:
                raise ValueError(f"user {user_id!r} already exists")
            row = existing or SqlUser(id=user_id)
            row.is_admin = is_admin
            row.password_hash = password_hash
            row.created_at = now
            row.last_login_at = None
            row.deleted_at = None
            row.account_generation = uuid.uuid4().hex
            session.add(row)
            try:
                session.flush()
            except IntegrityError as exc:
                # TOCTOU: another worker / request inserted the same
                # user_id between our SELECT and our INSERT. Surface
                # as the same ValueError the SELECT path raises so
                # callers handle uniqueness violation in one place.
                raise ValueError(f"user {user_id!r} already exists") from exc
            if team_owner:
                claim_founding_owner(session, user_id)
            return _to_account(row)

        return run_write_transaction(self._session_immediate, "create_user_with_password", write)

    def get_user(self, user_id: str) -> Account | None:
        """Look up a user by id. Returns ``None`` if missing."""
        with self._session("select_user_by_id") as session:
            row = session.get(SqlUser, (current_workspace_id(), user_id))
            return _to_account(row) if row is not None and row.deleted_at is None else None

    def with_runner_authority(
        self, runner_id: str, issue: Callable[[MachineBinding], _T | None]
    ) -> _T | None:
        """Issue for a runner's binding while holding its owner's account lock.

        The binding comes from the server's records only: the bound thread's
        single owner, who must also own the thread's live host under the same
        registration. Locking the owner's account serializes issuance with
        account deletion; the binding is read again under that lock.

        :param runner_id: Token-bound runner id.
        :param issue: Called with the binding under the lock, e.g. to mint.
        :returns: What *issue* returns, or ``None`` when the runner has no
            binding, an ambiguous one, or one held by someone else.
        """

        def write(session: Session) -> _T | None:
            binding = resolve_runner_binding(session, runner_id)
            if binding is None:
                return None
            require_active_account(session, binding.user_id, generation=binding.account_generation)
            # Deletion can remove ownership while the first read waits for the lock;
            # a locking read sees the latest committed binding on every backend.
            if resolve_runner_binding(session, runner_id, for_update=True) != binding:
                return None
            with account_authority_scope(binding.user_id, binding.account_generation):
                return issue(binding)

        return run_write_transaction(self._session_immediate, "authorize_runner", write)

    def is_admin(self, user_id: str) -> bool:
        """Whether ``user_id`` has the admin flag set.

        Duplicates :meth:`PermissionStore.is_admin` reading the
        same column on ``users`` — kept here so the accounts
        routes don't have to wire in a PermissionStore reference
        just to gate admin endpoints. The two stores agree by
        construction (single source of truth: the column).
        """
        with self._session("select_user_admin_status") as session:
            row = session.get(SqlUser, (current_workspace_id(), user_id))
            return row is not None and row.deleted_at is None and row.is_admin

    def set_admin(self, user_id: str, is_admin: bool) -> None:
        """Set the admin flag on an existing user row.

        The accounts-mode counterpart to
        :meth:`PermissionStore.set_admin` — both write the same
        ``users.is_admin`` column (single source of truth). Used by
        the file-backed admin-list promotion at login
        (:func:`omnigent.server.admin_list.promote_if_listed`), which
        only ever promotes (passes ``True``). No-op if the row is
        missing (the login path ensures it first).

        A team member's role follows the flag; see
        :func:`omnigent.stores.team_store.admin_flag_for_account`.

        :param user_id: The username to update, e.g. ``"alice"``.
        :param is_admin: The flag value to set.
        """

        def write(session: Session) -> None:
            lock_team(session)
            require_active_account(session, user_id)
            flag = admin_flag_for_account(session, user_id, is_admin)
            session.execute(
                update(SqlUser)
                .where(
                    SqlUser.workspace_id == current_workspace_id(),
                    SqlUser.id == user_id,
                )
                .values(is_admin=flag)
            )

        run_write_transaction(self._session_immediate, "set_user_admin_status", write)

    def list_users(self) -> list[Account]:
        """Return all users for the admin members page.

        Excludes two sentinel rows that aren't actionable in
        accounts mode:

        - ``"__public__"`` — anonymous-grant sentinel, never a
          real user.
        - ``"local"`` — backfilled by the original session-permissions
          migration so pre-accounts deploys had a default owner row
          for existing conversations. In accounts mode the name is
          reserved (can't authenticate, can't be reset, can't be
          promoted), so showing it as an "External" member on the
          Members page is dead weight. The row stays in the DB so
          a deploy that ever flipped back to header single-user
          mode would still find its legacy permission grants.

        Result is unordered; UI sorts.
        """
        with self._session("list_users") as session:
            rows = (
                session.execute(
                    select(SqlUser).where(SqlUser.workspace_id == current_workspace_id())
                )
                .scalars()
                .all()
            )
            return [
                _to_account(r)
                for r in rows
                if r.id not in _HIDDEN_LIST_USERS and r.deleted_at is None
            ]

    def _locked_admin_ids(self, session: Session, *, include: tuple[str, ...] = ()) -> list[str]:
        """Return every admin's user id, locked against concurrent change.

        Must be called on a session opened via ``self._session_immediate``
        (SQLite) or ``self._session`` with ``_supports_for_update`` True
        (other dialects) — see callers. On Postgres this issues
        ``SELECT ... FOR UPDATE`` on the admin rows, so a second
        transaction doing the same read blocks until this one commits
        instead of observing the same stale count. On SQLite the
        immediate session already holds the write lock, so no per-row
        clause is needed.

        Excludes ``_HIDDEN_LIST_USERS`` (``"local"``, ``"__public__"``)
        the same way :meth:`list_users` does — the legacy ``"local"``
        row can carry ``is_admin=True`` from the pre-accounts backfill,
        but it's reserved and can't authenticate in accounts mode, so
        counting it as a real admin would let the actual last admin
        get deleted believing a usable admin remains.
        """
        query = (
            select(SqlUser)
            .where(
                SqlUser.workspace_id == current_workspace_id(),
                or_(SqlUser.is_admin.is_(True), SqlUser.id.in_(include)),
            )
            .order_by(SqlUser.id)
        )
        if self._supports_for_update:
            query = query.with_for_update()
        rows = session.execute(query.execution_options(populate_existing=True)).scalars().all()
        return [
            row.id
            for row in rows
            if row.is_admin and row.deleted_at is None and row.id not in _HIDDEN_LIST_USERS
        ]

    def delete_user(self, user_id: str) -> bool | None:
        """Tombstone an account and revoke the durable authority it owns,
        refusing to remove the last admin.

        In the same transaction as the tombstone this also removes
        the user's ``session_permissions`` rows, marks their scheduled
        tasks ``deleted`` (so the scheduler never fires them again),
        revokes their device/refresh grants (so no further access tokens
        can be minted), and deletes or tombstones their hosts (so an
        unattended run has nowhere to land). Because everything shares
        one transaction, a partial failure rolls back the whole delete
        rather than leaving authority behind.

        The admin-invariant check and the delete run in the same locked
        transaction (see :meth:`_locked_admin_ids`), so this is atomic
        against a concurrent delete of a different admin — unlike a
        plain read-then-delete, the two can't both observe "an admin
        remains" and both apply.

        JWT authentication checks the stored generation and tombstone on every
        request, including requests reaching another server replica.

        The account's team membership is removed in the same transaction, so
        the username can later register again only as a new member.

        :returns: ``True`` if deleted, ``False`` if refused because
            ``user_id`` is the last remaining admin or the team's last active
            owner, ``None`` if no such user exists.
        """

        def write(session: Session) -> bool | None:
            # The team lock comes first, then the complete account lock set in
            # username order, before checking the last-admin and last-owner rules.
            lock_team(session)
            admin_ids = self._lock_for_deletion(session, user_id)
            target = lock_account(session, user_id)
            if target is None or target.deleted_at is not None:
                return None
            return _tombstone_account(session, target, admin_ids)

        return run_write_transaction(self._session_immediate, "delete_user", write)

    def _lock_for_deletion(self, session: Session, user_id: str) -> list[str]:
        """Lock the admins, the actor and *user_id* in username order; return the admins."""
        actor = current_account_user()
        include = (user_id, actor) if actor is not None else (user_id,)
        admin_ids = self._locked_admin_ids(session, include=include)
        require_active_account(session, None)
        return admin_ids

    def remove_member(
        self,
        member_id: str,
        *,
        actor_member_id: str | None,
        protected_user_id: str | None = None,
    ) -> MemberRemoval:
        """Remove a team member: their account, everything it can still use, and their access.

        One transaction tombstones the account as :meth:`delete_user` does and
        retires the membership with its identity bindings. It also revokes the
        pending invitations they could still use and records the removal: who
        removed them, the admission revision it committed, the threads they
        owned (still readable by owners and admins) and the cloud computers
        their provider must remove. A crash before the commit changes nothing;
        after it every credential of theirs is refused.

        Removing an already removed member changes nothing and returns its
        record, so the steps outside the database can be repeated.

        :param actor_member_id: The owner or admin removing them, authorized
            under the team lock: nobody removes themselves, and an admin
            removes members only. ``None`` for the system.
        :param protected_user_id: The installation's own owner account, which
            is never removed, e.g. the operator-named first admin.
        :raises OmnigentError: ``NOT_FOUND``, ``FORBIDDEN``, or
            :class:`LastActiveOwnerError`.
        """
        from omnigent.errors import ErrorCode, OmnigentError

        def write(session: Session) -> MemberRemoval:
            lock_team(session)
            found = member_account(session, member_id)
            if found is None:
                raise OmnigentError("no such team member", code=ErrorCode.NOT_FOUND)
            user_id, state = found
            if state == "removed":
                existing = removals_in_session(session, [member_id])
                if not existing:
                    raise OmnigentError("no such team member", code=ErrorCode.NOT_FOUND)
                return existing[0]
            admin_ids = self._lock_for_deletion(session, user_id)
            target = lock_account(session, user_id)
            row = member_state_row(session, user_id)
            if (
                target is None
                or target.deleted_at is not None
                or row is None
                or row.id != member_id
            ):
                raise OmnigentError("no such team member", code=ErrorCode.NOT_FOUND)
            if actor_member_id is not None:
                actor = require_actor(session, actor_member_id, owner=False)
                if actor.id == row.id:
                    raise OmnigentError("you cannot remove yourself", code=ErrorCode.FORBIDDEN)
                if actor.role != encode_team_member_role("owner") and row.role != (
                    encode_team_member_role("member")
                ):
                    raise OmnigentError(
                        "only an owner removes owners and admins", code=ErrorCode.FORBIDDEN
                    )
            if protected_user_id is not None and user_id == protected_user_id:
                raise OmnigentError(
                    "the installation's owner cannot be removed", code=ErrorCode.FORBIDDEN
                )
            threads, computers = owned_before_removal(session, user_id)
            emails = member_addresses(session, member_id)
            if not _tombstone_account(session, target, admin_ids):
                raise LastActiveOwnerError()
            now = int(time.time())
            revoke_invitations_for(session, emails=emails, user_id=user_id, now=now)
            return record_member_removal(
                session,
                member_id=member_id,
                user_id=user_id,
                removed_by_member_id=actor_member_id,
                threads=threads,
                computers=computers,
                now=now,
            )

        return run_write_transaction(self._session_immediate, "remove_team_member", write)

    def login_snapshot(self, user_id: str) -> tuple[str | None, str | None]:
        """Read the password and generation together before password verification."""
        with self._session("authenticate_account") as session:
            row = session.get(SqlUser, (current_workspace_id(), user_id))
            if row is None or row.deleted_at is not None:
                return None, None
            return row.password_hash, row.account_generation

    def accepts_generation(self, user_id: str, generation: str) -> bool:
        """Read revocation state on every accounts authentication attempt.

        A suspended team member's credentials are refused too, whatever their
        kind: sessions, grants, runner and host tokens.
        """
        with self._session("validate_account_authority") as session:
            row = session.execute(
                select(SqlUser.deleted_at, SqlUser.account_generation, SqlTeamMember.state)
                .outerjoin(
                    SqlTeamMember,
                    (SqlTeamMember.workspace_id == SqlUser.workspace_id)
                    & (SqlTeamMember.current_user_id == SqlUser.id),
                )
                .where(SqlUser.workspace_id == current_workspace_id(), SqlUser.id == user_id)
            ).one_or_none()
            return (
                row is not None
                and row.deleted_at is None
                and row.account_generation == generation
                and row.state != _SUSPENDED
            )

    # ── Team membership through accounts ──────────────────────────

    def redeem_team_invitation(
        self,
        token: str,
        *,
        username: str,
        password_hash: str,
        identity: ExternalIdentity | None,
    ) -> tuple[Account, TeamMember]:
        """Create the invited account, its membership and its identity binding at once.

        The invitation is consumed in the same transaction, so a concurrent
        second redemption fails. With a proxy-verified *identity*, its address
        must be the invited one and the identity is bound (provenance
        ``invitation``); without one (no proxy in front) no binding is made.

        :raises InvitationUnavailableError: Unknown, used, revoked or expired.
        :raises InvitationEmailMismatchError: The identity carries another address.
        :raises IdentityBindingConflictError: The identity is already bound.
        :raises ValueError: The username is taken.
        """
        now = int(time.time())

        def write(session: Session) -> tuple[Account, TeamMember]:
            lock_team(session)
            invitation = lock_pending_invitation(session, token, now)
            if invitation.target_user_id is not None:
                # A link invitation is redeemed by proving the existing account.
                raise InvitationEmailMismatchError()
            if identity is not None and identity.email != invitation.email:
                raise InvitationEmailMismatchError()
            existing = lock_account(session, username)
            if existing is not None and existing.deleted_at is None:
                raise ValueError(f"user {username!r} already exists")
            row = existing or SqlUser(id=username)
            row.is_admin = False
            row.password_hash = password_hash
            row.created_at = now
            row.last_login_at = None
            row.deleted_at = None
            row.account_generation = uuid.uuid4().hex
            session.add(row)
            try:
                session.flush()
            except IntegrityError as exc:
                raise ValueError(f"user {username!r} already exists") from exc
            member = insert_active_membership(
                session,
                username,
                cast(InvitationRole, decode_team_member_role(invitation.role)),
                row.account_generation,
            )
            if identity is not None:
                bind_identity_in_session(
                    session,
                    member.id,
                    issuer=identity.issuer,
                    subject=identity.subject,
                    audience=identity.audience,
                    provenance="invitation",
                    bound_by_member_id=invitation.created_by_member_id,
                    email=identity.email,
                )
            close_redeemed_invitation(session, invitation, member.id, now)
            session.refresh(row)
            return _to_account(row), to_member(member)

        return run_write_transaction(self._session_immediate, "redeem_team_invitation", write)

    def link_team_invitation(
        self, token: str, *, user_id: str, identity: ExternalIdentity | None
    ) -> TeamMember:
        """Give an existing account the membership a link invitation offers.

        The caller has verified the account's password. With a proxy-verified
        *identity*, its address must be the invited one and it is bound
        (provenance ``account_proof``).

        :raises InvitationUnavailableError: Unknown, used, revoked, expired, or
            for another account.
        :raises InvitationEmailMismatchError: The identity carries another address.
        """
        now = int(time.time())

        def write(session: Session) -> TeamMember:
            from omnigent.stores.team_store import InvitationUnavailableError

            lock_team(session)
            invitation = lock_pending_invitation(session, token, now)
            if invitation.target_user_id != user_id:
                raise InvitationUnavailableError()
            if identity is not None and identity.email != invitation.email:
                raise InvitationEmailMismatchError()
            require_active_account(session, user_id)
            account = lock_account(session, user_id)
            if account is None or account.deleted_at is not None:
                raise InvitationUnavailableError()
            member = insert_active_membership(
                session,
                user_id,
                cast(InvitationRole, decode_team_member_role(invitation.role)),
                account.account_generation,
            )
            if identity is not None:
                bind_identity_in_session(
                    session,
                    member.id,
                    issuer=identity.issuer,
                    subject=identity.subject,
                    audience=identity.audience,
                    provenance="account_proof",
                    bound_by_member_id=invitation.created_by_member_id,
                    email=identity.email,
                )
            close_redeemed_invitation(session, invitation, member.id, now)
            return to_member(member)

        return run_write_transaction(self._session_immediate, "link_team_invitation", write)

    def change_member_state(
        self, member_id: str, state: MemberState, *, actor_member_id: str | None = None
    ) -> TeamMember:
        """Suspend or reinstate a member, or move them on in the team lifecycle.

        Suspending also rotates the account's registration through the path a
        password reset uses: every session, refresh and device grant, magic
        link and runner or host token issued before dies, while hosts,
        schedules and the membership move to the new registration. Managed
        hosts' launch tokens are cleared and runner bindings retired, so no
        machine credential from before the suspension works after
        reinstatement: computers get fresh launch tokens at their next start,
        threads fresh runners, and people sign in again everywhere.

        :param actor_member_id: The owner or admin making the change, authorized
            inside the write against the roles as they are then: nobody changes
            their own state, and an admin changes members only. ``None`` for the
            system.
        :raises OmnigentError: ``NOT_FOUND``, ``CONFLICT`` for a transition the
            lifecycle does not allow, ``FORBIDDEN`` for an actor who may not, or
            :class:`LastActiveOwnerError`.
        """
        from omnigent.errors import ErrorCode, OmnigentError

        def write(session: Session) -> TeamMember:
            lock_team(session)
            user_id = session.execute(
                select(SqlTeamMember.current_user_id).where(
                    SqlTeamMember.workspace_id == current_workspace_id(),
                    SqlTeamMember.id == member_id,
                )
            ).scalar_one_or_none()
            if user_id is None:
                raise OmnigentError("no such team member", code=ErrorCode.NOT_FOUND)
            require_active_account(session, user_id)
            account = lock_account(session, user_id)
            row = member_state_row(session, user_id)
            if row is None or row.id != member_id:
                raise OmnigentError("no such team member", code=ErrorCode.NOT_FOUND)
            if actor_member_id is not None:
                actor = require_actor(session, actor_member_id, owner=False)
                if actor.id == row.id:
                    raise OmnigentError("you cannot suspend yourself", code=ErrorCode.FORBIDDEN)
                if actor.role != encode_team_member_role("owner") and row.role != (
                    encode_team_member_role("member")
                ):
                    raise OmnigentError(
                        "only an owner suspends or reinstates owners and admins",
                        code=ErrorCode.FORBIDDEN,
                    )
            changed = apply_member_state(session, row, state)
            generation = account.account_generation if account is not None else None
            if changed and state == "suspended":
                if generation is not None:
                    new_generation = uuid.uuid4().hex
                    _carry_registration_work(session, user_id, old=generation, new=new_generation)
                    session.execute(
                        update(SqlUser)
                        .where(
                            SqlUser.workspace_id == current_workspace_id(), SqlUser.id == user_id
                        )
                        .values(account_generation=new_generation)
                    )
                _retire_machine_credentials(session, user_id)
            return to_member(row)

        return run_write_transaction(self._session_immediate, "change_team_member_state", write)

    def get_password_hash(self, user_id: str) -> str | None:
        """Fetch a user's password hash for verification.

        ONLY method that surfaces the hash. Routes that call this
        must pass the result straight into
        :func:`omnigent.server.passwords.verify_password` — never
        log, return, or store the value elsewhere.
        """
        with self._session("select_password_hash") as session:
            row = session.get(SqlUser, (current_workspace_id(), user_id))
            return row.password_hash if row is not None and row.deleted_at is None else None

    def update_password(
        self, user_id: str, password_hash: str, *, revoke_sessions: bool = False
    ) -> str | None:
        """Replace a user's stored password hash.

        Used by self-serve ``/auth/users/me/password``, admin-initiated
        reset, and the login-time rehash. Provision a generation when a
        passwordless sharing identity first receives credentials. With
        *revoke_sessions*, an existing generation is rotated: session and
        runner JWTs, refresh and device grants, and magic links carry the
        generation they were issued under, so the per-request check rejects
        them afterwards on every replica, while the user's scheduled tasks
        and hosts move to the new generation. Otherwise the generation stays
        unchanged. No-op if the user does not exist (the route should 404
        first).

        :returns: The account generation after the write, or ``None`` when
            the user does not exist.
        """

        def write(session: Session) -> str | None:
            generation = require_active_account(session, user_id)
            new_generation = generation
            if generation is None or revoke_sessions:
                new_generation = uuid.uuid4().hex
                if generation is not None:
                    _carry_registration_work(session, user_id, old=generation, new=new_generation)
                else:
                    # A passwordless identity first receiving credentials stays the same
                    # registration for its team membership.
                    carry_membership_registration(session, user_id, old=None, new=new_generation)
            result = cast(
                CursorResult[tuple[object]],
                session.execute(
                    update(SqlUser)
                    .where(
                        SqlUser.workspace_id == current_workspace_id(),
                        SqlUser.id == user_id,
                    )
                    .values(password_hash=password_hash, account_generation=new_generation)
                ),
            )
            return new_generation if result.rowcount == 1 else None

        return run_write_transaction(self._session_immediate, "update_password", write)

    def mark_logged_in(self, user_id: str, when_epoch_seconds: int) -> None:
        """Bump ``last_login_at`` on every successful login.

        :param when_epoch_seconds: Login timestamp. Tests pass a
            fixed value for determinism.
        """

        def write(session: Session) -> None:
            require_active_account(session, user_id)
            session.execute(
                update(SqlUser)
                .where(
                    SqlUser.workspace_id == current_workspace_id(),
                    SqlUser.id == user_id,
                )
                .values(last_login_at=when_epoch_seconds)
            )

        run_write_transaction(self._session_immediate, "mark_user_logged_in", write)

    # ── Account tokens (invite + magic-link) ──────────────────────

    def create_token(
        self,
        token_id: str,
        *,
        kind: str,
        user_id: str | None,
        created_by: str | None,
        created_at: int,
        expires_at: int,
        invited_is_admin: bool = False,
    ) -> AccountToken:
        """Persist a new invite or magic token.

        The token id (the secret) is generated by the caller —
        see :func:`secrets.token_urlsafe`. The store does not
        validate entropy. Bounds:

        - ``kind`` must be ``"invite"`` or ``"magic"``
          (enforced by a DB check constraint).
        - For ``"invite"``, ``user_id`` is ``None`` and
          ``created_by`` is the admin's id.
        - For ``"magic"``, ``user_id`` is the user being signed
          in and ``created_by`` is ``None`` (self-issued).

        :raises ValueError: On an unknown kind. Fail fast so the
            DB-level error never has to surface to the route.
        """
        if kind not in ("invite", "magic"):
            raise ValueError(f"unknown token kind {kind!r}")

        def write(session: Session) -> AccountToken:
            generation = require_active_account(session, user_id)
            row = SqlAccountToken(
                account_generation=generation,
                id=token_id,
                kind=encode_account_token_kind(kind),
                user_id=user_id,
                created_by=created_by,
                created_at=created_at,
                expires_at=expires_at,
                invited_is_admin=invited_is_admin,
            )
            session.add(row)
            session.flush()
            return _to_account_token(row)

        return run_write_transaction(self._session_immediate, "create_account_token", write)

    def redeem_token(
        self, token_id: str, *, kind: str, now_epoch_seconds: int
    ) -> AccountToken | None:
        """Atomically mark a token as redeemed.

        A naive "SELECT then UPDATE" race would let two concurrent
        requests both succeed. A single
        ``UPDATE … WHERE redeemed_at IS NULL`` + rowcount check
        makes the redeem step itself atomic — at most one caller
        sees ``rowcount == 1`` even under concurrent redeem
        attempts.

        Returns ``None`` for missing / wrong-kind / already-redeemed
        / expired tokens. Caller can't distinguish (intentional —
        opaque-to-bruteforce-guessing).
        """

        def write(session: Session) -> AccountToken | None:
            result = cast(
                CursorResult[tuple[object]],
                session.execute(
                    update(SqlAccountToken)
                    .where(
                        and_(
                            SqlAccountToken.workspace_id == current_workspace_id(),
                            SqlAccountToken.id == token_id,
                            SqlAccountToken.kind == encode_account_token_kind(kind),
                            SqlAccountToken.redeemed_at.is_(None),
                            SqlAccountToken.expires_at > now_epoch_seconds,
                        )
                    )
                    .values(redeemed_at=now_epoch_seconds)
                ),
            )
            if result.rowcount == 0:
                return None
            row = session.get(SqlAccountToken, (current_workspace_id(), token_id))
            return _to_account_token(row) if row is not None else None

        return run_write_transaction(self._session_immediate, "redeem_account_token", write)

    def purge_expired_tokens(self, now_epoch_seconds: int) -> int:
        """Delete tokens whose ``expires_at`` is in the past.

        Called periodically (e.g. on app startup) so the table
        doesn't accumulate stale rows. Single-use enforcement is
        via ``redeemed_at`` regardless of expiry, so purging is
        purely housekeeping.

        :returns: The number of rows deleted.
        """

        def write(session: Session) -> int:
            result = cast(
                CursorResult[tuple[object]],
                session.execute(
                    delete(SqlAccountToken).where(
                        SqlAccountToken.workspace_id == current_workspace_id(),
                        SqlAccountToken.expires_at <= now_epoch_seconds,
                    )
                ),
            )
            return result.rowcount

        return run_write_transaction(
            self._session_immediate,
            "purge_expired_account_tokens",
            write,
        )

    # ── OIDC invited emails (opt-in pre-authorization) ────────────
    #
    # No dedicated table: the OIDC invite reuses the existing
    # ``account_tokens`` rows (kind="invite"). The
    # single-use token is minted with ``user_id=NULL``; at the OIDC
    # callback we atomically redeem it AND stamp the redeeming email
    # into ``user_id``. That stamped, redeemed row IS the durable
    # pre-authorization — ``is_email_invited`` just looks for one. This
    # keeps the OSS-only invite feature from adding a table that would
    # ship (empty, unused) into the hosted / Databricks-Apps schema.

    def redeem_oidc_invite(self, token_id: str, email: str, *, now_epoch_seconds: int) -> bool:
        """Atomically redeem an OIDC invite token and bind it to ``email``.

        A single ``UPDATE … WHERE redeemed_at IS NULL`` makes redemption
        single-use even under concurrent callbacks, and stamps
        ``user_id=email`` so the redeemed row doubles as the durable
        pre-authorization that :meth:`is_email_invited` later finds.

        :param token_id: The invite token secret from the invite URL.
        :param email: The IdP-returned email, lowercased by the caller,
            e.g. ``"contractor@gmail.com"``.
        :param now_epoch_seconds: Current time; the token must not be
            expired or already redeemed.
        :returns: ``True`` if this call redeemed the token, ``False`` if
            it was missing / wrong-kind / already-redeemed / expired.
        """

        def write(session: Session) -> bool:
            result = cast(
                CursorResult[tuple[object]],
                session.execute(
                    update(SqlAccountToken)
                    .where(
                        and_(
                            SqlAccountToken.workspace_id == current_workspace_id(),
                            SqlAccountToken.id == token_id,
                            SqlAccountToken.kind == encode_account_token_kind("invite"),
                            SqlAccountToken.redeemed_at.is_(None),
                            SqlAccountToken.expires_at > now_epoch_seconds,
                        )
                    )
                    .values(redeemed_at=now_epoch_seconds, user_id=email)
                ),
            )
            return result.rowcount == 1

        return run_write_transaction(self._session_immediate, "redeem_oidc_invite", write)

    def is_email_invited(self, email: str) -> bool:
        """Whether ``email`` redeemed an OIDC invite (durable pre-auth).

        Looks for a redeemed invite token stamped with this email by
        :meth:`redeem_oidc_invite`. Persists across logins, so an invited
        off-domain user stays admitted. Accounts-mode invites leave
        ``user_id`` NULL, so they never match here.

        :param email: The email to check, lowercased, e.g.
            ``"contractor@gmail.com"``.
        :returns: ``True`` if a redeemed invite token is bound to it.
        """
        with self._session("select_email_invitation_status") as session:
            return session.execute(
                select(
                    exists().where(
                        and_(
                            SqlAccountToken.workspace_id == current_workspace_id(),
                            SqlAccountToken.kind == encode_account_token_kind("invite"),
                            SqlAccountToken.user_id == email,
                            SqlAccountToken.redeemed_at.is_not(None),
                        )
                    )
                )
            ).scalar_one()
