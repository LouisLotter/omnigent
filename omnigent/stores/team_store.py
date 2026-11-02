"""Team membership, external identity bindings and the ownership inventory.

Each workspace has one team. A membership maps an existing account to a
stable member id, a role and a state; the account admin flag follows from
the role and state (see :func:`omnigent.entities.team.account_admin_flag`).

Membership writes lock the team row first, then account rows, then member
rows. That order serializes role and state changes per team, so two
concurrent changes cannot both remove the last active owner; SQLite sessions
start ``BEGIN IMMEDIATE`` for the same effect. Like other admin-flag writes,
a change reaches cached permission decisions within their short TTL.
"""

from __future__ import annotations

import time
import uuid
from collections import Counter
from collections.abc import Callable, Iterator
from typing import TypeVar, cast

from sqlalchemy import ColumnElement, Insert, delete, func, select, update
from sqlalchemy.dialects.mysql import insert as mysql_insert
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import InstrumentedAttribute, Session

from omnigent.db.account_authority import AccountAuthority, lock_account, require_active_account
from omnigent.db.db_models import (
    SqlConnection,
    SqlConversationMetadata,
    SqlHost,
    SqlScheduledTask,
    SqlSessionPermission,
    SqlTeam,
    SqlTeamIdentityBinding,
    SqlTeamMember,
    SqlUser,
    current_workspace_id,
    external_identity_cksum,
)
from omnigent.db.enum_codecs import (
    decode_team_identity_provenance,
    decode_team_member_role,
    decode_team_member_state,
    encode_conversation_kind,
    encode_scheduled_task_state,
    encode_team_identity_provenance,
    encode_team_member_role,
    encode_team_member_state,
)
from omnigent.db.utils import (
    get_or_create_engine,
    make_named_managed_session_maker,
    run_write_transaction,
)
from omnigent.entities.team import (
    ExternalIdentityBinding,
    IdentityProvenance,
    InventoryAccount,
    MemberRole,
    MemberState,
    OwnershipFinding,
    Team,
    TeamInventory,
    TeamMember,
    account_admin_flag,
)
from omnigent.errors import ErrorCode, OmnigentError
from omnigent.server.auth import LEVEL_OWNER, RESERVED_USER_LOCAL, RESERVED_USER_PUBLIC

_RESERVED_USERS = frozenset({RESERVED_USER_LOCAL, RESERVED_USER_PUBLIC})
_ROLES: frozenset[str] = frozenset({"owner", "admin", "member"})
_STATE_TRANSITIONS: dict[str, frozenset[str]] = {
    "invited": frozenset({"active", "removed"}),
    "active": frozenset({"suspended", "removed"}),
    "suspended": frozenset({"active", "removed"}),
    "removed": frozenset(),
}
_REMOVED = encode_team_member_state("removed")
_ACTIVE = encode_team_member_state("active")
_OWNER = encode_team_member_role("owner")
# Bounds the bindings deleted with a membership.
MAX_IDENTITIES_PER_MEMBER = 8
_ISSUER_MAX_CHARS = 512
_SUBJECT_MAX_CHARS = 255
_AUDIENCE_MAX_CHARS = 255
_INVENTORY_PAGE_SIZE = 500
_Row = TypeVar("_Row")


class LastActiveOwnerError(OmnigentError):
    """The change would leave the team without an active owner."""

    def __init__(self) -> None:
        super().__init__("the team must keep at least one active owner", code=ErrorCode.CONFLICT)


class IdentityBindingConflictError(OmnigentError):
    """The external identity is already bound differently."""

    def __init__(self) -> None:
        super().__init__(
            "this external identity is already bound to a different member or audience",
            code=ErrorCode.CONFLICT,
        )


def _now() -> int:
    return int(time.time())


def _to_team(row: SqlTeam) -> Team:
    return Team(id=row.id, name=row.name, created_at=row.created_at)


def _to_member(row: SqlTeamMember) -> TeamMember:
    return TeamMember(
        id=row.id,
        user_id=row.user_id,
        role=cast(MemberRole, decode_team_member_role(row.role)),
        state=cast(MemberState, decode_team_member_state(row.state)),
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _to_binding(row: SqlTeamIdentityBinding) -> ExternalIdentityBinding:
    return ExternalIdentityBinding(
        issuer=row.issuer,
        subject=row.subject,
        audience=row.audience,
        member_id=row.member_id,
        provenance=cast(IdentityProvenance, decode_team_identity_provenance(row.provenance)),
        bound_by_member_id=row.bound_by_member_id,
        created_at=row.created_at,
    )


def _insert_team_if_missing(session: Session, now: int) -> None:
    """Insert the workspace's team unless a concurrent writer already did."""
    values = {
        "workspace_id": current_workspace_id(),
        "id": uuid.uuid4().hex,
        "name": None,
        "created_at": now,
    }
    dialect = session.get_bind().dialect.name
    statement: Insert
    if dialect == "sqlite":
        statement = (
            sqlite_insert(SqlTeam)
            .values(**values)
            .on_conflict_do_nothing(index_elements=["workspace_id"])
        )
    elif dialect == "mysql":
        mysql_statement = mysql_insert(SqlTeam).values(**values)
        statement = mysql_statement.on_duplicate_key_update(id=SqlTeam.__table__.c.id)
    else:
        statement = (
            pg_insert(SqlTeam)
            .values(**values)
            .on_conflict_do_nothing(index_elements=["workspace_id"])
        )
    session.execute(statement)


def lock_team(session: Session) -> SqlTeam:
    """Lock the workspace's team row, creating it on first use.

    Every membership write calls this before touching account or member rows.
    """
    query = (
        select(SqlTeam)
        .where(SqlTeam.workspace_id == current_workspace_id())
        .with_for_update()
        .execution_options(populate_existing=True)
    )
    team = session.execute(query).scalar_one_or_none()
    if team is None:
        _insert_team_if_missing(session, _now())
        team = session.execute(query).scalar_one()
    return team


def _lock_member(session: Session, member_id: str) -> SqlTeamMember | None:
    return session.execute(
        select(SqlTeamMember)
        .where(
            SqlTeamMember.workspace_id == current_workspace_id(),
            SqlTeamMember.id == member_id,
        )
        .with_for_update()
        .execution_options(populate_existing=True)
    ).scalar_one_or_none()


def _lock_current_member(session: Session, user_id: str) -> SqlTeamMember | None:
    return session.execute(
        select(SqlTeamMember)
        .where(
            SqlTeamMember.workspace_id == current_workspace_id(),
            SqlTeamMember.current_user_id == user_id,
        )
        .with_for_update()
        .execution_options(populate_existing=True)
    ).scalar_one_or_none()


def _same_registration(
    column: InstrumentedAttribute[str | None], generation: str | None
) -> ColumnElement[bool]:
    """Match a stored generation to *generation*, treating ``NULL`` as a value."""
    return column.is_(None) if generation is None else column == generation


def _live_generation(session: Session, user_id: str) -> tuple[bool, str | None]:
    """Return whether *user_id* is a live account, and its generation."""
    row = session.execute(
        select(SqlUser.deleted_at, SqlUser.account_generation).where(
            SqlUser.workspace_id == current_workspace_id(), SqlUser.id == user_id
        )
    ).one_or_none()
    if row is None or row.deleted_at is not None:
        return False, None
    return True, row.account_generation


def _removed_from_registration(session: Session, user_id: str, generation: str | None) -> bool:
    """Whether this registration of *user_id* has a removed membership; reads one row."""
    return (
        session.execute(
            select(SqlTeamMember.id)
            .where(
                SqlTeamMember.workspace_id == current_workspace_id(),
                SqlTeamMember.user_id == user_id,
                _same_registration(SqlTeamMember.account_generation, generation),
                SqlTeamMember.state == _REMOVED,
            )
            .limit(1)
        ).first()
        is not None
    )


def carry_membership_registration(
    session: Session, user_id: str, *, old: str | None, new: str
) -> None:
    """Move *user_id*'s memberships from generation *old* to *new*.

    Runs under the account lock when an account's generation rotates or is
    first provisioned, as hosts and schedules move with it; a removal stays
    in force for the same registration.
    """
    session.execute(
        update(SqlTeamMember)
        .where(
            SqlTeamMember.workspace_id == current_workspace_id(),
            SqlTeamMember.user_id == user_id,
            _same_registration(SqlTeamMember.account_generation, old),
        )
        .values(account_generation=new)
    )


def _has_other_active_owner(session: Session, member_id: str) -> bool:
    """Whether an active owner other than *member_id* exists; reads at most one row."""
    return (
        session.execute(
            select(SqlTeamMember.id)
            .where(
                SqlTeamMember.workspace_id == current_workspace_id(),
                SqlTeamMember.role == _OWNER,
                SqlTeamMember.state == _ACTIVE,
                SqlTeamMember.id != member_id,
            )
            .limit(1)
        ).first()
        is not None
    )


def _require_owner_remains(session: Session, row: SqlTeamMember, role: int, state: int) -> None:
    """Refuse a change that takes the team's last active owner out of that role."""
    was_active_owner = row.role == _OWNER and row.state == _ACTIVE
    stays_active_owner = role == _OWNER and state == _ACTIVE
    if (
        was_active_owner
        and not stays_active_owner
        and not _has_other_active_owner(session, row.id)
    ):
        raise LastActiveOwnerError()


def _store_admin_flag(session: Session, row: SqlTeamMember) -> None:
    """Write the admin flag the membership implies onto its live account."""
    flag = account_admin_flag(
        cast(MemberRole, decode_team_member_role(row.role)),
        cast(MemberState, decode_team_member_state(row.state)),
    )
    session.execute(
        update(SqlUser)
        .where(
            SqlUser.workspace_id == current_workspace_id(),
            SqlUser.id == row.user_id,
            SqlUser.deleted_at.is_(None),
        )
        .values(is_admin=flag)
    )


def _retire(session: Session, row: SqlTeamMember, now: int) -> None:
    """Mark *row* removed and drop its identity bindings."""
    row.state = _REMOVED
    row.current_user_id = None
    row.updated_at = now
    session.execute(
        delete(SqlTeamIdentityBinding).where(
            SqlTeamIdentityBinding.workspace_id == current_workspace_id(),
            SqlTeamIdentityBinding.member_id == row.id,
        )
    )
    session.flush()


def retire_deleted_account_membership(session: Session, user_id: str) -> bool:
    """Remove the current membership of an account that is being deleted.

    Runs inside the account deletion's transaction, after :func:`lock_team`.

    :param session: The deletion's session.
    :param user_id: The account being deleted, e.g. ``"alice"``.
    :returns: ``False`` when the account is the team's last active owner and
        must not be deleted; ``True`` otherwise.
    """
    row = _lock_current_member(session, user_id)
    if row is None:
        return True
    try:
        _require_owner_remains(session, row, row.role, _REMOVED)
    except LastActiveOwnerError:
        return False
    _retire(session, row, _now())
    return True


def admin_flag_for_account(session: Session, user_id: str, is_admin: bool) -> bool:
    """Align a member's role with a direct admin-flag write; return the flag to store.

    Promotion makes a member an admin and demotion makes an admin a member; an
    owner's role changes only through the team. A registration whose
    membership was removed is not promoted until it is a member again, so a
    login-time roster cannot restore it. Accounts that never joined, including
    a later registration of the same username, keep the requested flag. Runs
    after :func:`lock_team` and the account lock.

    :param session: The admin-flag write's session.
    :param user_id: The account being changed, e.g. ``"alice"``.
    :param is_admin: The requested flag.
    :returns: The flag the account should carry.
    :raises OmnigentError: ``CONFLICT`` when asked to demote an owner.
    """
    row = _lock_current_member(session, user_id)
    if row is None:
        if not is_admin:
            return False
        live, generation = _live_generation(session, user_id)
        return not (live and _removed_from_registration(session, user_id, generation))
    role = cast(MemberRole, decode_team_member_role(row.role))
    if role == "owner" and not is_admin:
        raise OmnigentError(
            "an owner's role changes through the team, not the admin flag",
            code=ErrorCode.CONFLICT,
        )
    new_role: MemberRole = role if role == "owner" else ("admin" if is_admin else "member")
    if new_role != role:
        row.role = encode_team_member_role(new_role)
        row.updated_at = _now()
        session.flush()
    return account_admin_flag(new_role, cast(MemberState, decode_team_member_state(row.state)))


def claim_founding_owner(session: Session, user_id: str) -> TeamMember | None:
    """Make *user_id* the active owner of a team that has no current members.

    Used when the installation's first administrator is created, inside that
    account's transaction. A team that already has members is left alone.

    :param session: The account creation's session.
    :param user_id: The new first administrator, e.g. ``"alice"``.
    :returns: The owner membership, or ``None`` when the team had members.
    """
    lock_team(session)
    has_member = session.execute(
        select(SqlTeamMember.id)
        .where(
            SqlTeamMember.workspace_id == current_workspace_id(),
            SqlTeamMember.state != _REMOVED,
        )
        .limit(1)
    ).first()
    if has_member is not None:
        return None
    now = _now()
    row = SqlTeamMember(
        id=uuid.uuid4().hex,
        user_id=user_id,
        account_generation=_live_generation(session, user_id)[1],
        current_user_id=user_id,
        role=_OWNER,
        state=_ACTIVE,
        created_at=now,
        updated_at=now,
    )
    session.add(row)
    session.flush()
    _store_admin_flag(session, row)
    return _to_member(row)


def _validated_identity(issuer: str, subject: str) -> tuple[str, str]:
    if not issuer or len(issuer) > _ISSUER_MAX_CHARS:
        raise OmnigentError("issuer must be 1-512 characters", code=ErrorCode.INVALID_INPUT)
    if not subject or len(subject) > _SUBJECT_MAX_CHARS:
        raise OmnigentError("subject must be 1-255 characters", code=ErrorCode.INVALID_INPUT)
    return issuer, subject


class SqlAlchemyTeamStore:
    """SQLAlchemy persistence for the team, its members and identity bindings.

    :param storage_location: SQLAlchemy database URI, e.g.
        ``"sqlite:///omnigent.db"``; shares the engine of the other stores.
    """

    def __init__(self, storage_location: str) -> None:
        self.storage_location = storage_location
        self._engine = get_or_create_engine(storage_location)
        self._session = make_named_managed_session_maker(
            self._engine, query_name_prefix="omnigent.team_store"
        )
        self._session_immediate = make_named_managed_session_maker(
            self._engine, query_name_prefix="omnigent.team_store", immediate=True
        )

    # ── Team and members ───────────────────────────────────────────

    def get_team(self) -> Team | None:
        """Return the workspace's team, or ``None`` before it exists."""
        with self._session("select_team") as session:
            row = session.get(SqlTeam, current_workspace_id())
            return _to_team(row) if row is not None else None

    def add_member(
        self, user_id: str, *, role: MemberRole, state: MemberState = "invited"
    ) -> TeamMember:
        """Create a membership for an existing live account.

        :param user_id: The account, e.g. ``"alice"``.
        :param role: The initial role.
        :param state: ``"invited"`` (the default) or ``"active"``.
        :returns: The new membership with a fresh member id.
        :raises OmnigentError: ``INVALID_INPUT`` for a reserved or unknown
            account or another state; ``CONFLICT`` when the account already
            has a current membership.
        """
        if role not in _ROLES or state not in ("invited", "active"):
            raise OmnigentError("invalid role or initial state", code=ErrorCode.INVALID_INPUT)
        if user_id in _RESERVED_USERS:
            raise OmnigentError(
                f"{user_id!r} is reserved and cannot join a team",
                code=ErrorCode.INVALID_INPUT,
            )

        def write(session: Session) -> TeamMember:
            lock_team(session)
            require_active_account(session, user_id)
            account = lock_account(session, user_id)
            if account is None or account.deleted_at is not None:
                raise OmnigentError(f"no account {user_id!r}", code=ErrorCode.INVALID_INPUT)
            if _lock_current_member(session, user_id) is not None:
                raise OmnigentError(
                    f"{user_id!r} already has a team membership", code=ErrorCode.CONFLICT
                )
            now = _now()
            row = SqlTeamMember(
                id=uuid.uuid4().hex,
                user_id=user_id,
                account_generation=account.account_generation,
                current_user_id=user_id,
                role=encode_team_member_role(role),
                state=encode_team_member_state(state),
                created_at=now,
                updated_at=now,
            )
            session.add(row)
            try:
                session.flush()
            except IntegrityError as exc:
                raise OmnigentError(
                    f"{user_id!r} already has a team membership", code=ErrorCode.CONFLICT
                ) from exc
            _store_admin_flag(session, row)
            return _to_member(row)

        return run_write_transaction(self._session_immediate, "add_team_member", write)

    def get_member(self, member_id: str) -> TeamMember | None:
        """Return a membership by member id, including a removed one."""
        with self._session("select_team_member") as session:
            row = session.get(SqlTeamMember, (current_workspace_id(), member_id))
            return _to_member(row) if row is not None else None

    def get_member_for_account(self, user_id: str) -> TeamMember | None:
        """Return the live account's current membership, or ``None``."""
        return self.member_for_owner(AccountAuthority(user_id, None, current_workspace_id()))

    def member_for_owner(self, owner: AccountAuthority) -> TeamMember | None:
        """Map a resource's actual owner to that registration's current membership.

        A generation that no longer matches the account means the resource
        belonged to an earlier registration of the same username.

        :param owner: An owner from an actual-ownership lookup, e.g.
            :meth:`ConversationStore.get_actual_session_owner`.
        :returns: The membership, or ``None`` when the owner is not a member.
        """
        with self._session("select_team_member_for_owner") as session:
            row = session.execute(
                select(SqlTeamMember, SqlUser.account_generation)
                .join(
                    SqlUser,
                    (SqlUser.workspace_id == SqlTeamMember.workspace_id)
                    & (SqlUser.id == SqlTeamMember.current_user_id),
                )
                .where(
                    SqlTeamMember.workspace_id == owner.workspace_id,
                    SqlTeamMember.current_user_id == owner.user_id,
                    SqlUser.deleted_at.is_(None),
                )
            ).one_or_none()
            if row is None:
                return None
            member, generation = row
            if member.account_generation != generation or (
                owner.generation is not None and owner.generation != generation
            ):
                return None
            return _to_member(member)

    def list_members(
        self, *, limit: int = 100, after_user_id: str | None = None
    ) -> tuple[list[TeamMember], str | None]:
        """List current memberships ordered by username.

        :param limit: Page size, 1-1000.
        :param after_user_id: Continue after this username.
        :returns: The page and the cursor for the next one, or ``None``.
        """
        if not 1 <= limit <= 1000:
            raise OmnigentError("limit must be 1-1000", code=ErrorCode.INVALID_INPUT)
        with self._session("list_team_members") as session:
            rows = (
                session.execute(
                    select(SqlTeamMember)
                    .where(
                        SqlTeamMember.workspace_id == current_workspace_id(),
                        SqlTeamMember.current_user_id > (after_user_id or ""),
                    )
                    .order_by(SqlTeamMember.current_user_id)
                    .limit(limit + 1)
                )
                .scalars()
                .all()
            )
            page = [_to_member(row) for row in rows[:limit]]
            return page, page[-1].user_id if len(rows) > limit else None

    def _update_member(
        self,
        operation: str,
        member_id: str,
        change: Callable[[SqlTeamMember], tuple[int, int]],
    ) -> TeamMember:
        """Apply a role/state change under the team lock and keep the invariants."""

        def write(session: Session) -> TeamMember:
            lock_team(session)
            account = session.execute(
                select(SqlTeamMember.current_user_id).where(
                    SqlTeamMember.workspace_id == current_workspace_id(),
                    SqlTeamMember.id == member_id,
                )
            ).one_or_none()
            if account is None:
                raise OmnigentError("no such team member", code=ErrorCode.NOT_FOUND)
            # A removed membership's account may be gone; only the actor is checked then.
            require_active_account(session, account.current_user_id)
            row = _lock_member(session, member_id)
            if row is None:
                raise OmnigentError("no such team member", code=ErrorCode.NOT_FOUND)
            role, state = change(row)
            if (role, state) == (row.role, row.state):
                return _to_member(row)
            _require_owner_remains(session, row, role, state)
            if state == _REMOVED:
                _retire(session, row, _now())
            else:
                row.role, row.state, row.updated_at = role, state, _now()
                session.flush()
            _store_admin_flag(session, row)
            return _to_member(row)

        return run_write_transaction(self._session_immediate, operation, write)

    def change_role(self, member_id: str, role: MemberRole) -> TeamMember:
        """Change a current member's role, keeping an active owner.

        :raises OmnigentError: ``NOT_FOUND`` for an unknown member,
            ``CONFLICT`` for a removed one, :class:`LastActiveOwnerError` when
            the last active owner would lose the role.
        """
        if role not in _ROLES:
            raise OmnigentError(f"unknown role {role!r}", code=ErrorCode.INVALID_INPUT)

        def change(row: SqlTeamMember) -> tuple[int, int]:
            if row.state == _REMOVED:
                raise OmnigentError("a removed member has no role", code=ErrorCode.CONFLICT)
            return encode_team_member_role(role), row.state

        return self._update_member("change_team_member_role", member_id, change)

    def change_state(self, member_id: str, state: MemberState) -> TeamMember:
        """Move a membership through invited → active ⇄ suspended → removed.

        Removal is final and drops the member's identity bindings; the account
        itself is not deleted. The same state again is a no-op.

        :raises OmnigentError: ``NOT_FOUND`` for an unknown member,
            ``CONFLICT`` for a transition the lifecycle does not allow,
            :class:`LastActiveOwnerError` when the last active owner would
            stop being active.
        """
        if state not in _STATE_TRANSITIONS:
            raise OmnigentError(f"unknown state {state!r}", code=ErrorCode.INVALID_INPUT)

        def change(row: SqlTeamMember) -> tuple[int, int]:
            current = decode_team_member_state(row.state)
            if state != current and state not in _STATE_TRANSITIONS[current]:
                raise OmnigentError(
                    f"a {current} member cannot become {state}", code=ErrorCode.CONFLICT
                )
            return row.role, encode_team_member_state(state)

        return self._update_member("change_team_member_state", member_id, change)

    # ── External identities ────────────────────────────────────────

    def bind_external_identity(
        self,
        member_id: str,
        *,
        issuer: str,
        subject: str,
        audience: str,
        provenance: IdentityProvenance,
        bound_by_member_id: str | None = None,
    ) -> ExternalIdentityBinding:
        """Bind a verified ``(issuer, subject)`` to a current member.

        The caller has verified the provider's assertion and the person's
        right to this member. Binding the same identity to the same member and
        audience again returns the existing binding, so concurrent identical
        binds converge; any other existing binding is a conflict.

        :param member_id: The member to bind.
        :param issuer: The provider's issuer, at most 512 characters.
        :param subject: The provider's subject, at most 255 characters.
        :param audience: The audience its assertions must carry.
        :param provenance: How the identity was verified.
        :param bound_by_member_id: The member completing the binding, if any.
        :returns: The binding.
        :raises IdentityBindingConflictError: When bound differently.
        :raises OmnigentError: ``NOT_FOUND`` for an unknown or removed member,
            ``CONFLICT`` when the member already has the maximum bindings,
            ``INVALID_INPUT`` for malformed values.
        """
        issuer, subject = _validated_identity(issuer, subject)
        if not audience or len(audience) > _AUDIENCE_MAX_CHARS:
            raise OmnigentError("audience must be 1-255 characters", code=ErrorCode.INVALID_INPUT)
        provenance_code = encode_team_identity_provenance(provenance)
        cksum = external_identity_cksum(issuer, subject)
        existing_query = (
            select(SqlTeamIdentityBinding)
            .where(
                SqlTeamIdentityBinding.workspace_id == current_workspace_id(),
                SqlTeamIdentityBinding.identity_cksum == cksum,
            )
            .with_for_update()
            .execution_options(populate_existing=True)
        )

        def converge(existing: SqlTeamIdentityBinding) -> ExternalIdentityBinding:
            if (existing.issuer, existing.subject, existing.member_id, existing.audience) != (
                issuer,
                subject,
                member_id,
                audience,
            ):
                raise IdentityBindingConflictError()
            return _to_binding(existing)

        def write(session: Session) -> ExternalIdentityBinding:
            member = _lock_member(session, member_id)
            if member is None or member.state == _REMOVED:
                raise OmnigentError("no such team member", code=ErrorCode.NOT_FOUND)
            existing = session.execute(existing_query).scalar_one_or_none()
            if existing is not None:
                return converge(existing)
            bound = session.execute(
                select(func.count())
                .select_from(SqlTeamIdentityBinding)
                .where(
                    SqlTeamIdentityBinding.workspace_id == current_workspace_id(),
                    SqlTeamIdentityBinding.member_id == member_id,
                )
            ).scalar_one()
            if bound >= MAX_IDENTITIES_PER_MEMBER:
                raise OmnigentError(
                    "this member already has the maximum number of external identities",
                    code=ErrorCode.CONFLICT,
                )
            row = SqlTeamIdentityBinding(
                identity_cksum=cksum,
                issuer=issuer,
                subject=subject,
                audience=audience,
                member_id=member_id,
                provenance=provenance_code,
                bound_by_member_id=bound_by_member_id,
                created_at=_now(),
            )
            session.add(row)
            session.flush()
            return _to_binding(row)

        try:
            return run_write_transaction(self._session_immediate, "bind_external_identity", write)
        except IntegrityError:
            # A concurrent bind committed the same identity first; the retry reads it.
            return run_write_transaction(self._session_immediate, "bind_external_identity", write)

    def get_external_identity(self, issuer: str, subject: str) -> ExternalIdentityBinding | None:
        """Return the binding for ``(issuer, subject)``, or ``None``."""
        issuer, subject = _validated_identity(issuer, subject)
        with self._session("select_external_identity") as session:
            row = session.get(
                SqlTeamIdentityBinding,
                (current_workspace_id(), external_identity_cksum(issuer, subject)),
            )
            if row is None or (row.issuer, row.subject) != (issuer, subject):
                return None
            return _to_binding(row)

    def member_for_external_identity(
        self, issuer: str, subject: str, *, audience: str
    ) -> TeamMember | None:
        """Resolve a verified identity to its current member.

        Matches only the exact ``(issuer, subject)`` and the audience recorded
        at binding; email plays no part. The caller decides what the member's
        state allows.

        :returns: The member, or ``None`` when unbound, bound for another
            audience, or no longer a current member of a live account.
        """
        issuer, subject = _validated_identity(issuer, subject)
        with self._session("select_member_for_external_identity") as session:
            row = session.execute(
                select(SqlTeamIdentityBinding, SqlTeamMember)
                .join(
                    SqlTeamMember,
                    (SqlTeamMember.workspace_id == SqlTeamIdentityBinding.workspace_id)
                    & (SqlTeamMember.id == SqlTeamIdentityBinding.member_id),
                )
                .join(
                    SqlUser,
                    (SqlUser.workspace_id == SqlTeamMember.workspace_id)
                    & (SqlUser.id == SqlTeamMember.current_user_id),
                )
                .where(
                    SqlTeamIdentityBinding.workspace_id == current_workspace_id(),
                    SqlTeamIdentityBinding.identity_cksum
                    == external_identity_cksum(issuer, subject),
                    SqlUser.deleted_at.is_(None),
                )
            ).one_or_none()
            if row is None:
                return None
            binding, member = row
            if (binding.issuer, binding.subject, binding.audience) != (issuer, subject, audience):
                return None
            return _to_member(member)

    def list_external_identities(self, member_id: str) -> list[ExternalIdentityBinding]:
        """Return a member's bindings; at most :data:`MAX_IDENTITIES_PER_MEMBER`."""
        with self._session("list_member_external_identities") as session:
            rows = (
                session.execute(
                    select(SqlTeamIdentityBinding)
                    .where(
                        SqlTeamIdentityBinding.workspace_id == current_workspace_id(),
                        SqlTeamIdentityBinding.member_id == member_id,
                    )
                    .order_by(SqlTeamIdentityBinding.identity_cksum)
                    .limit(MAX_IDENTITIES_PER_MEMBER)
                )
                .scalars()
                .all()
            )
            return [_to_binding(row) for row in rows]

    def unbind_external_identity(self, issuer: str, subject: str) -> bool:
        """Remove the binding for ``(issuer, subject)``.

        :returns: ``True`` when a binding was removed.
        """
        issuer, subject = _validated_identity(issuer, subject)

        def write(session: Session) -> bool:
            row = session.execute(
                select(SqlTeamIdentityBinding)
                .where(
                    SqlTeamIdentityBinding.workspace_id == current_workspace_id(),
                    SqlTeamIdentityBinding.identity_cksum
                    == external_identity_cksum(issuer, subject),
                )
                .with_for_update()
            ).scalar_one_or_none()
            if row is None or (row.issuer, row.subject) != (issuer, subject):
                return False
            session.delete(row)
            return True

        return run_write_transaction(self._session_immediate, "unbind_external_identity", write)

    # ── Inventory ──────────────────────────────────────────────────

    def inventory(self, *, page_size: int = _INVENTORY_PAGE_SIZE) -> TeamInventory:
        """Report who owns what, without credentials, for a team upgrade.

        Reads in bounded pages. Ownership that cannot be assigned to exactly
        one live account is reported as a finding, never assigned. Threads
        count by their explicit owner grant; sub-agent threads follow their
        root and are not listed.

        :param page_size: Rows per read.
        :returns: The inventory.
        """
        with self._session("team_inventory") as session:
            return _Inventory(session, page_size).run()


class _Inventory:
    """One pass over the workspace's accounts and owned resources."""

    def __init__(self, session: Session, page_size: int) -> None:
        self._session = session
        self._page_size = page_size
        self._workspace = current_workspace_id()
        self._findings: list[OwnershipFinding] = []
        self._generations: dict[str, str | None] = {}
        self._members: dict[str, SqlTeamMember] = {}
        self._live_hosts: dict[str, str] = {}

    def run(self) -> TeamInventory:
        team = self._session.get(SqlTeam, self._workspace)
        accounts = self._accounts()
        self._check_memberships()
        thread_owners = self._threads()
        host_owners = self._hosts()
        connection_owners = self._connections()
        # Schedules pinned to a host are checked against the hosts read above.
        schedule_owners = self._schedules()
        return TeamInventory(
            team_id=team.id if team is not None else None,
            accounts=tuple(accounts),
            thread_owners=thread_owners,
            host_owners=host_owners,
            connection_owners=connection_owners,
            schedule_owners=schedule_owners,
            findings=tuple(self._findings),
        )

    def _find(self, resource: str, resource_id: str, reason: str, *owners: str) -> None:
        self._findings.append(
            OwnershipFinding(resource, resource_id, reason, tuple(sorted(owners)))
        )

    def _owner_problem(self, user_id: str | None, generation: str | None = None) -> str | None:
        """Classify a recorded owner; ``None`` means a live registration."""
        if user_id is None:
            return "no_owner"
        if user_id == RESERVED_USER_LOCAL:
            return "legacy_local_owner"
        if user_id not in self._generations:
            return "unknown_owner"
        current = self._generations[user_id]
        if generation is not None and current is not None and generation != current:
            return "earlier_registration"
        return None

    def _accounts(self) -> list[InventoryAccount]:
        for row in self._pages(SqlTeamMember, SqlTeamMember.id, SqlTeamMember.workspace_id):
            if row.state != _REMOVED and row.current_user_id is not None:
                self._members[row.current_user_id] = row
        accounts: list[InventoryAccount] = []
        admins: list[str] = []
        for user in self._pages(SqlUser, SqlUser.id, SqlUser.workspace_id):
            if user.id in _RESERVED_USERS or user.deleted_at is not None:
                continue
            self._generations[user.id] = user.account_generation
            member = self._members.get(user.id)
            role = cast(MemberRole, decode_team_member_role(member.role)) if member else None
            state = cast(MemberState, decode_team_member_state(member.state)) if member else None
            accounts.append(
                InventoryAccount(
                    user_id=user.id,
                    is_admin=user.is_admin,
                    has_password=user.password_hash is not None,
                    member_id=member.id if member else None,
                    role=role,
                    state=state,
                )
            )
            if user.is_admin:
                admins.append(user.id)
            if member is None:
                self._find("account", user.id, "not_member", user.id)
            elif role is not None and state is not None:
                if account_admin_flag(role, state) != user.is_admin:
                    self._find("account", user.id, "admin_flag_mismatch", user.id)
        if not any(
            m.role == _OWNER and m.state == _ACTIVE
            for user_id, m in self._members.items()
            if user_id in self._generations
        ):
            self._find("team", str(self._workspace), "no_active_owner", *admins)
        return accounts

    def _check_memberships(self) -> None:
        for user_id, member in self._members.items():
            if user_id not in self._generations:
                self._find("member", member.id, "member_without_account", user_id)

    def _threads(self) -> dict[str, int]:
        owners: Counter[str] = Counter()
        default_kind = encode_conversation_kind("default")
        for page in self._id_pages(
            SqlConversationMetadata.id,
            SqlConversationMetadata.workspace_id,
            SqlConversationMetadata.kind == default_kind,
        ):
            grants: dict[str, list[str]] = {conversation_id: [] for conversation_id in page}
            for conversation_id, user_id in self._session.execute(
                select(SqlSessionPermission.conversation_id, SqlSessionPermission.user_id).where(
                    SqlSessionPermission.workspace_id == self._workspace,
                    SqlSessionPermission.conversation_id.in_(page),
                    SqlSessionPermission.level >= LEVEL_OWNER,
                    SqlSessionPermission.user_id != RESERVED_USER_PUBLIC,
                )
            ):
                grants[conversation_id].append(user_id)
            for conversation_id, holders in grants.items():
                if len(holders) > 1:
                    self._find("thread", conversation_id, "multiple_owners", *holders)
                    continue
                owner = holders[0] if holders else None
                problem = self._owner_problem(owner)
                if problem is not None:
                    self._find("thread", conversation_id, problem, *([owner] if owner else []))
                elif owner is not None:
                    owners[owner] += 1
        return dict(owners)

    def _hosts(self) -> dict[str, int]:
        owners: Counter[str] = Counter()
        for host in self._pages(SqlHost, SqlHost.host_id, SqlHost.workspace_id):
            if host.deleted_at is not None:
                continue
            problem = self._owner_problem(host.user_id, host.account_generation)
            if problem is not None:
                self._find("host", host.host_id, problem, host.user_id)
                continue
            self._live_hosts[host.host_id] = host.user_id
            owners[host.user_id] += 1
        return dict(owners)

    def _connections(self) -> dict[str, int]:
        owners: Counter[str] = Counter()
        after: tuple[str, str, str] = ("", "", "")
        while True:
            rows = self._session.execute(
                select(SqlConnection.user_id, SqlConnection.provider, SqlConnection.account_id)
                .where(
                    SqlConnection.workspace_id == self._workspace,
                    (SqlConnection.user_id > after[0])
                    | (
                        (SqlConnection.user_id == after[0])
                        & (
                            (SqlConnection.provider > after[1])
                            | (
                                (SqlConnection.provider == after[1])
                                & (SqlConnection.account_id > after[2])
                            )
                        )
                    ),
                )
                .order_by(SqlConnection.user_id, SqlConnection.provider, SqlConnection.account_id)
                .limit(self._page_size)
            ).all()
            for user_id, provider, account_id in rows:
                problem = self._owner_problem(user_id)
                if problem is not None:
                    self._find(
                        "connection", f"{user_id}/{provider}/{account_id}", problem, user_id
                    )
                else:
                    owners[user_id] += 1
            if len(rows) < self._page_size:
                return dict(owners)
            after = cast(tuple[str, str, str], tuple(rows[-1]))

    def _schedules(self) -> dict[str, int]:
        owners: Counter[str] = Counter()
        retired = (
            encode_scheduled_task_state("deleted"),
            encode_scheduled_task_state("completed"),
        )
        for task in self._pages(
            SqlScheduledTask, SqlScheduledTask.id, SqlScheduledTask.workspace_id
        ):
            if task.state in retired:
                continue
            problem = self._owner_problem(task.user_id, task.account_generation)
            if problem is not None:
                self._find("schedule", task.id, problem, *([task.user_id] if task.user_id else []))
                continue
            assert task.user_id is not None
            if task.host_id is not None and self._live_hosts.get(task.host_id) != task.user_id:
                self._find("schedule", task.id, "host_not_owned", task.user_id)
                continue
            owners[task.user_id] += 1
        return dict(owners)

    def _pages(
        self,
        model: type[_Row],
        key: InstrumentedAttribute[str],
        workspace_column: InstrumentedAttribute[int],
    ) -> Iterator[_Row]:
        """Yield every row of *model* in this workspace, keyset-paged by *key*."""
        after: str | None = None
        while True:
            query = select(model).where(workspace_column == self._workspace)
            if after is not None:
                query = query.where(key > after)
            rows = (
                self._session.execute(query.order_by(key).limit(self._page_size)).scalars().all()
            )
            yield from rows
            if len(rows) < self._page_size:
                return
            after = cast(str, getattr(rows[-1], key.key))

    def _id_pages(
        self,
        key: InstrumentedAttribute[str],
        workspace_column: InstrumentedAttribute[int],
        *conditions: ColumnElement[bool],
    ) -> Iterator[list[str]]:
        """Yield non-empty pages of ids from a workspace-scoped id column."""
        after: str | None = None
        while True:
            query = select(key).where(workspace_column == self._workspace, *conditions)
            if after is not None:
                query = query.where(key > after)
            ids = list(self._session.execute(query.order_by(key).limit(self._page_size)).scalars())
            if ids:
                yield ids
            if len(ids) < self._page_size:
                return
            after = ids[-1]
