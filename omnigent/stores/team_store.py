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

import hashlib
import re
import secrets
import time
import uuid
from collections import Counter
from collections.abc import Callable, Iterator
from enum import Enum
from typing import TYPE_CHECKING, TypeVar, cast

from sqlalchemy import ColumnElement, Insert, delete, func, insert, or_, select, update
from sqlalchemy.dialects.mysql import insert as mysql_insert
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import CursorResult
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
    SqlTeamInvitation,
    SqlTeamMember,
    SqlTeamMemberRemoval,
    SqlTeamMemberRemovalItem,
    SqlUser,
    current_workspace_id,
    external_identity_cksum,
)
from omnigent.db.enum_codecs import (
    decode_team_identity_provenance,
    decode_team_invitation_state,
    decode_team_member_role,
    decode_team_member_state,
    encode_conversation_kind,
    encode_host_lifecycle,
    encode_scheduled_task_state,
    encode_team_identity_provenance,
    encode_team_invitation_state,
    encode_team_member_role,
    encode_team_member_state,
    encode_team_removal_item_kind,
)
from omnigent.db.utils import (
    get_or_create_engine,
    make_named_managed_session_maker,
    run_write_transaction,
)
from omnigent.entities.team import (
    AdmissionSnapshot,
    ExternalIdentityBinding,
    IdentityProvenance,
    InventoryAccount,
    InvitationRole,
    InvitationState,
    MemberRemoval,
    MemberRole,
    MemberState,
    OwnershipFinding,
    Team,
    TeamInventory,
    TeamInvitation,
    TeamMember,
    account_admin_flag,
)
from omnigent.errors import ErrorCode, OmnigentError
from omnigent.server.auth import LEVEL_OWNER, RESERVED_USER_LOCAL, RESERVED_USER_PUBLIC

if TYPE_CHECKING:
    from omnigent.server.external_identity import ExternalIdentity

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
_EMAIL_MAX_CHARS = 320
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_INVITATION_ROLES: frozenset[str] = frozenset({"admin", "member"})
_PENDING = encode_team_invitation_state("pending")
_REDEEMED = encode_team_invitation_state("redeemed")
_REVOKED = encode_team_invitation_state("revoked")
_EXPIRED = encode_team_invitation_state("expired")
_ADMIN = encode_team_member_role("admin")
_MEMBER_ROLE = encode_team_member_role("member")
_THREAD_ITEM = encode_team_removal_item_kind("thread")
_COMPUTER_ITEM = encode_team_removal_item_kind("computer")
_PERSISTENT_HOST = encode_host_lifecycle("persistent")
# Rows per insert when a removal records what the member owned.
_REMOVAL_ITEM_BATCH = 500
# Bounds one admission list and one page of invitations.
MAX_ADMITTED = 1000
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


class InvitationUnavailableError(OmnigentError):
    """The invitation is unknown, expired, revoked or already used."""

    def __init__(self) -> None:
        super().__init__("this invitation is invalid or has expired", code=ErrorCode.INVALID_INPUT)


class InvitationEmailMismatchError(OmnigentError):
    """The verified sign-in carries another address than the invitation's."""

    def __init__(self) -> None:
        super().__init__(
            "this invitation is for another email address; sign in with the invited address",
            code=ErrorCode.FORBIDDEN,
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
        email=row.email,
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
    # Their address leaves the admission list, so a front proxy must hear of it now.
    bump_admission_revision(session)
    return True


# ── A member's removal, inside the account deletion's transaction ──────


def member_account(session: Session, member_id: str) -> tuple[str, MemberState] | None:
    """The account a membership names and its state, read under the team lock.

    The row is not locked here: the caller locks the account first, then the
    membership, as every account writer does.
    """
    row = session.execute(
        select(SqlTeamMember.user_id, SqlTeamMember.state).where(
            SqlTeamMember.workspace_id == current_workspace_id(),
            SqlTeamMember.id == member_id,
        )
    ).one_or_none()
    if row is None:
        return None
    return row.user_id, cast(MemberState, decode_team_member_state(row.state))


def owned_before_removal(session: Session, user_id: str) -> tuple[list[str], list[str]]:
    """The threads *user_id* owns and their cloud computers, before the deletion drops them.

    Cloud computers are persistent hosts that are live or still have a
    sandbox for their provider to remove.

    :returns: ``(thread_ids, host_ids)``, each sorted.
    """
    workspace_id = current_workspace_id()
    threads = session.execute(
        select(SqlSessionPermission.conversation_id)
        .where(
            SqlSessionPermission.workspace_id == workspace_id,
            SqlSessionPermission.user_id == user_id,
            SqlSessionPermission.level >= LEVEL_OWNER,
        )
        .order_by(SqlSessionPermission.conversation_id)
    ).scalars()
    computers = session.execute(
        select(SqlHost.host_id)
        .where(
            SqlHost.workspace_id == workspace_id,
            SqlHost.user_id == user_id,
            SqlHost.lifecycle == _PERSISTENT_HOST,
            or_(
                SqlHost.deleted_at.is_(None),
                SqlHost.sandbox_id.is_not(None),
                SqlHost.terminating_sandbox_id.is_not(None),
            ),
        )
        .order_by(SqlHost.host_id)
    ).scalars()
    return list(threads), list(computers)


def member_identity_emails(session: Session, member_id: str) -> list[str]:
    """The addresses *member_id*'s identities were verified with, before they are dropped."""
    rows = session.execute(
        select(SqlTeamIdentityBinding.email)
        .where(
            SqlTeamIdentityBinding.workspace_id == current_workspace_id(),
            SqlTeamIdentityBinding.member_id == member_id,
            SqlTeamIdentityBinding.email.is_not(None),
        )
        .limit(MAX_IDENTITIES_PER_MEMBER)
    ).scalars()
    return sorted({email for email in rows if email})


def revoke_invitations_for(session: Session, *, emails: list[str], user_id: str, now: int) -> int:
    """Revoke the pending invitations a removed person could still use.

    Those to one of their addresses, and those linking their account. An
    invitation they issued to someone else is that person's and stays.

    :returns: How many were revoked.
    """
    for_them = [SqlTeamInvitation.target_user_id == user_id]
    if emails:
        for_them.append(SqlTeamInvitation.email.in_(emails))
    result = session.execute(
        update(SqlTeamInvitation)
        .where(
            SqlTeamInvitation.workspace_id == current_workspace_id(),
            SqlTeamInvitation.state == _PENDING,
            or_(*for_them),
        )
        .values(state=_REVOKED, closed_at=now)
    )
    revoked = cast(CursorResult, result).rowcount or 0
    if revoked:
        bump_admission_revision(session)
    return revoked


def record_member_removal(
    session: Session,
    *,
    member_id: str,
    user_id: str,
    removed_by_member_id: str | None,
    threads: list[str],
    computers: list[str],
    now: int,
) -> MemberRemoval:
    """Write the removal's record, after every change it makes, in the same transaction."""
    team = lock_team(session)
    session.add(
        SqlTeamMemberRemoval(
            member_id=member_id,
            user_id=user_id,
            removed_by_member_id=removed_by_member_id,
            removed_at=now,
            admission_revision=team.admission_revision or 0,
        )
    )
    session.flush()
    items = [(_THREAD_ITEM, thread_id) for thread_id in threads]
    items += [(_COMPUTER_ITEM, host_id) for host_id in computers]
    workspace_id = current_workspace_id()
    for start in range(0, len(items), _REMOVAL_ITEM_BATCH):
        session.execute(
            insert(SqlTeamMemberRemovalItem),
            [
                {
                    "workspace_id": workspace_id,
                    "member_id": member_id,
                    "kind": kind,
                    "item_id": item,
                }
                for kind, item in items[start : start + _REMOVAL_ITEM_BATCH]
            ],
        )
    removal = removals_in_session(session, [member_id])
    return removal[0]


def removals_in_session(session: Session, member_ids: list[str]) -> list[MemberRemoval]:
    """The removal records for *member_ids*, newest first, with their counts.

    A computer is pending while its host's tombstone still records a sandbox:
    its provider has not confirmed the removal, and the reaper retries it.
    """
    if not member_ids:
        return []
    workspace_id = current_workspace_id()
    rows = (
        session.execute(
            select(SqlTeamMemberRemoval)
            .where(
                SqlTeamMemberRemoval.workspace_id == workspace_id,
                SqlTeamMemberRemoval.member_id.in_(member_ids),
            )
            .order_by(SqlTeamMemberRemoval.removed_at.desc(), SqlTeamMemberRemoval.member_id)
        )
        .scalars()
        .all()
    )
    counts: Counter[tuple[str, int]] = Counter()
    for member_id, kind, count in session.execute(
        select(
            SqlTeamMemberRemovalItem.member_id,
            SqlTeamMemberRemovalItem.kind,
            func.count(),
        )
        .where(
            SqlTeamMemberRemovalItem.workspace_id == workspace_id,
            SqlTeamMemberRemovalItem.member_id.in_(member_ids),
        )
        .group_by(SqlTeamMemberRemovalItem.member_id, SqlTeamMemberRemovalItem.kind)
    ).all():
        counts[(member_id, kind)] = count
    pending: dict[str, int] = dict(
        session.execute(
            select(SqlTeamMemberRemovalItem.member_id, func.count())
            .join(
                SqlHost,
                (SqlHost.workspace_id == SqlTeamMemberRemovalItem.workspace_id)
                & (SqlHost.host_id == SqlTeamMemberRemovalItem.item_id),
            )
            .where(
                SqlTeamMemberRemovalItem.workspace_id == workspace_id,
                SqlTeamMemberRemovalItem.member_id.in_(member_ids),
                SqlTeamMemberRemovalItem.kind == _COMPUTER_ITEM,
                SqlHost.deleted_at.is_not(None),
                or_(SqlHost.sandbox_id.is_not(None), SqlHost.terminating_sandbox_id.is_not(None)),
            )
            .group_by(SqlTeamMemberRemovalItem.member_id)
        )
        .tuples()
        .all()
    )
    return [
        MemberRemoval(
            member_id=row.member_id,
            user_id=row.user_id,
            removed_by_member_id=row.removed_by_member_id,
            removed_at=row.removed_at,
            admission_revision=row.admission_revision,
            thread_count=counts[(row.member_id, _THREAD_ITEM)],
            computers=counts[(row.member_id, _COMPUTER_ITEM)],
            computers_pending=pending.get(row.member_id, 0),
        )
        for row in rows
    ]


def admin_flag_for_account(session: Session, user_id: str, is_admin: bool) -> bool:
    """The admin flag a direct admin-flag write may store; team roles win.

    A current membership's role is authoritative: the flag stays what its role
    and state imply, so a login-time roster or another flag write cannot undo
    an owner's demotion or promote anyone (roles change only through the
    team). Accounts without a membership keep the requested flag, except a
    registration whose membership was removed, which is not promoted until it
    is a member again. Runs after :func:`lock_team` and the account lock.

    :param session: The admin-flag write's session.
    :param user_id: The account being changed, e.g. ``"alice"``.
    :param is_admin: The requested flag.
    :returns: The flag the account should carry.
    """
    row = _lock_current_member(session, user_id)
    if row is not None:
        return account_admin_flag(
            cast(MemberRole, decode_team_member_role(row.role)),
            cast(MemberState, decode_team_member_state(row.state)),
        )
    if not is_admin:
        return False
    live, generation = _live_generation(session, user_id)
    return not (live and _removed_from_registration(session, user_id, generation))


def claim_founding_owner(session: Session, user_id: str) -> TeamMember | None:
    """Make *user_id* the active owner of a team that has no current members.

    Used when the installation's first administrator is created, inside that
    account's transaction. A team that already has members is left alone.

    :param session: The account creation's session.
    :param user_id: The new first administrator, e.g. ``"alice"``.
    :returns: The owner membership, or ``None`` when the team had members.
    """
    team = lock_team(session)
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
    # The proxy's sign-in of the operator may bind to this membership, once.
    team.founding_member_id = row.id
    session.flush()
    return _to_member(row)


def require_actor(session: Session, actor_member_id: str, *, owner: bool) -> SqlTeamMember:
    """Lock and return the acting membership, re-read under the team lock.

    Team administration is authorized here, inside the write that applies it,
    so a role or state change that commits first is seen.

    :param owner: Require an active owner; otherwise an active owner or admin.
    :raises OmnigentError: ``FORBIDDEN`` when the actor may not administer.
    """
    actor = _lock_member(session, actor_member_id)
    allowed = (_OWNER,) if owner else (_OWNER, _ADMIN)
    if actor is None or actor.state != _ACTIVE or actor.role not in allowed:
        raise OmnigentError(
            "only an owner may" if owner else "only the team's owners and admins manage it",
            code=ErrorCode.FORBIDDEN,
        )
    return actor


def _validated_identity(issuer: str, subject: str) -> tuple[str, str]:
    if not issuer or len(issuer) > _ISSUER_MAX_CHARS:
        raise OmnigentError("issuer must be 1-512 characters", code=ErrorCode.INVALID_INPUT)
    if not subject or len(subject) > _SUBJECT_MAX_CHARS:
        raise OmnigentError("subject must be 1-255 characters", code=ErrorCode.INVALID_INPUT)
    return issuer, subject


def normalize_invitation_email(email: str) -> str:
    """Return *email* trimmed and lowercased, or raise for a malformed address."""
    normalized = email.strip().lower()
    if len(normalized) > _EMAIL_MAX_CHARS or not _EMAIL_RE.match(normalized):
        raise OmnigentError("enter a valid email address", code=ErrorCode.INVALID_INPUT)
    return normalized


def _token_digest(token: str) -> bytes:
    return hashlib.sha256(token.encode()).digest()


def _invitation_state(row: SqlTeamInvitation, now: int) -> InvitationState:
    state = cast(InvitationState, decode_team_invitation_state(row.state))
    return "expired" if state == "pending" and row.expires_at <= now else state


def _to_invitation(row: SqlTeamInvitation, now: int) -> TeamInvitation:
    return TeamInvitation(
        id=row.id,
        email=row.email,
        role=cast(InvitationRole, decode_team_member_role(row.role)),
        target_user_id=row.target_user_id,
        created_by_member_id=row.created_by_member_id,
        created_at=row.created_at,
        expires_at=row.expires_at,
        state=_invitation_state(row, now),
        redeemed_member_id=row.redeemed_member_id,
    )


def bump_admission_revision(session: Session) -> int:
    """Record that who a front proxy should admit changed; returns the new revision.

    Runs in the changing transaction, after :func:`lock_team`.
    """
    team = lock_team(session)
    team.admission_revision = (team.admission_revision or 0) + 1
    session.flush()
    return team.admission_revision


def bind_identity_in_session(
    session: Session,
    member_id: str,
    *,
    issuer: str,
    subject: str,
    audience: str,
    provenance: IdentityProvenance,
    bound_by_member_id: str | None,
    email: str | None,
) -> ExternalIdentityBinding:
    """Bind a verified identity to a current member inside the caller's transaction.

    The same identity bound again to the same member and audience converges;
    any other existing binding is a conflict.

    :raises IdentityBindingConflictError: When bound differently.
    :raises OmnigentError: ``NOT_FOUND`` for an unknown or removed member,
        ``CONFLICT`` at the binding limit, ``INVALID_INPUT`` for bad values.
    """
    issuer, subject = _validated_identity(issuer, subject)
    if not audience or len(audience) > _AUDIENCE_MAX_CHARS:
        raise OmnigentError("audience must be 1-255 characters", code=ErrorCode.INVALID_INPUT)
    if email is not None and len(email) > _EMAIL_MAX_CHARS:
        raise OmnigentError("email must be at most 320 characters", code=ErrorCode.INVALID_INPUT)
    # Team row first, as every membership write does.
    lock_team(session)
    member = _lock_member(session, member_id)
    if member is None or member.state == _REMOVED:
        raise OmnigentError("no such team member", code=ErrorCode.NOT_FOUND)
    existing = session.execute(
        select(SqlTeamIdentityBinding)
        .where(
            SqlTeamIdentityBinding.workspace_id == current_workspace_id(),
            SqlTeamIdentityBinding.identity_cksum == external_identity_cksum(issuer, subject),
        )
        .with_for_update()
        .execution_options(populate_existing=True)
    ).scalar_one_or_none()
    if existing is not None:
        if (existing.issuer, existing.subject, existing.member_id, existing.audience) != (
            issuer,
            subject,
            member_id,
            audience,
        ):
            raise IdentityBindingConflictError()
        return _to_binding(existing)
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
        identity_cksum=external_identity_cksum(issuer, subject),
        issuer=issuer,
        subject=subject,
        audience=audience,
        member_id=member_id,
        provenance=encode_team_identity_provenance(provenance),
        bound_by_member_id=bound_by_member_id,
        created_at=_now(),
        email=email,
    )
    session.add(row)
    session.flush()
    if email is not None:
        bump_admission_revision(session)
    return _to_binding(row)


def lock_pending_invitation(session: Session, token: str, now: int) -> SqlTeamInvitation:
    """Lock the pending, unexpired invitation for *token*.

    :raises InvitationUnavailableError: For any other token, without saying why.
    """
    row = session.execute(
        select(SqlTeamInvitation)
        .where(
            SqlTeamInvitation.workspace_id == current_workspace_id(),
            SqlTeamInvitation.token_digest == _token_digest(token),
        )
        .with_for_update()
        .execution_options(populate_existing=True)
    ).scalar_one_or_none()
    if row is None or row.state != _PENDING or row.expires_at <= now:
        raise InvitationUnavailableError()
    return row


def close_redeemed_invitation(
    session: Session, row: SqlTeamInvitation, member_id: str, now: int
) -> None:
    """Mark a locked invitation used by *member_id*; it can never be redeemed again."""
    row.state = _REDEEMED
    row.closed_at = now
    row.redeemed_member_id = member_id
    session.flush()


def insert_active_membership(
    session: Session, user_id: str, role: InvitationRole, generation: str | None
) -> SqlTeamMember:
    """Create an active membership for a live account that has none.

    Runs after :func:`lock_team` and the account lock.

    :raises OmnigentError: ``CONFLICT`` when the account already has a current
        membership.
    """
    if _lock_current_member(session, user_id) is not None:
        raise OmnigentError(f"{user_id!r} already has a team membership", code=ErrorCode.CONFLICT)
    now = _now()
    row = SqlTeamMember(
        id=uuid.uuid4().hex,
        user_id=user_id,
        account_generation=generation,
        current_user_id=user_id,
        role=encode_team_member_role(role),
        state=_ACTIVE,
        created_at=now,
        updated_at=now,
    )
    session.add(row)
    session.flush()
    _store_admin_flag(session, row)
    return row


def member_state_row(session: Session, user_id: str) -> SqlTeamMember | None:
    """Lock and return the account's current membership, if any."""
    return _lock_current_member(session, user_id)


def apply_member_state(session: Session, row: SqlTeamMember, state: MemberState) -> bool:
    """Move a locked membership to *state*, keeping the lifecycle and an active owner.

    Runs after :func:`lock_team`. Removal is final and drops the identity
    bindings; any state change alters admission, so it bumps the revision.

    :returns: Whether the state changed.
    :raises OmnigentError: ``CONFLICT`` for a transition the lifecycle does
        not allow; :class:`LastActiveOwnerError` when the last active owner
        would stop being active.
    """
    current = decode_team_member_state(row.state)
    if state == current:
        return False
    if state not in _STATE_TRANSITIONS[current]:
        raise OmnigentError(f"a {current} member cannot become {state}", code=ErrorCode.CONFLICT)
    encoded = encode_team_member_state(state)
    _require_owner_remains(session, row, row.role, encoded)
    if state == "removed":
        _retire(session, row, _now())
    else:
        row.state, row.updated_at = encoded, _now()
        session.flush()
    _store_admin_flag(session, row)
    bump_admission_revision(session)
    return True


def to_member(row: SqlTeamMember) -> TeamMember:
    """The entity for a membership row."""
    return _to_member(row)


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
        *,
        actor_member_id: str | None = None,
    ) -> TeamMember:
        """Apply a role/state change under the team lock and keep the invariants.

        With *actor_member_id*, the actor must be an active owner, re-read under
        the same lock.
        """

        def write(session: Session) -> TeamMember:
            lock_team(session)
            if actor_member_id is not None:
                require_actor(session, actor_member_id, owner=True)
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
            state_changed = state != row.state
            if state == _REMOVED:
                _retire(session, row, _now())
            else:
                row.role, row.state, row.updated_at = role, state, _now()
                session.flush()
            _store_admin_flag(session, row)
            if state_changed:
                bump_admission_revision(session)
            return _to_member(row)

        return run_write_transaction(self._session_immediate, operation, write)

    def change_role(
        self, member_id: str, role: MemberRole, *, actor_member_id: str | None = None
    ) -> TeamMember:
        """Change a current member's role, keeping an active owner.

        :param actor_member_id: The owner making the change, authorized inside
            the write; ``None`` for the system.
        :raises OmnigentError: ``NOT_FOUND`` for an unknown member,
            ``CONFLICT`` for a removed one, :class:`LastActiveOwnerError` when
            the last active owner would lose the role, ``FORBIDDEN`` when the
            actor is not an active owner.
        """
        if role not in _ROLES:
            raise OmnigentError(f"unknown role {role!r}", code=ErrorCode.INVALID_INPUT)

        def change(row: SqlTeamMember) -> tuple[int, int]:
            if row.state == _REMOVED:
                raise OmnigentError("a removed member has no role", code=ErrorCode.CONFLICT)
            return encode_team_member_role(role), row.state

        return self._update_member(
            "change_team_member_role", member_id, change, actor_member_id=actor_member_id
        )

    def change_state(self, member_id: str, state: MemberState) -> TeamMember:
        """Move a membership through invited → active ⇄ suspended → removed.

        Removal is final and drops the member's identity bindings; the account
        itself is not deleted. The same state again is a no-op. This is the
        bare state change: suspending a person through the team, which also
        rotates their registration, is
        :meth:`SqlAlchemyAccountStore.change_member_state`.

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
        email: str | None = None,
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
        :param email: The address the identity was verified with, lowercase,
            kept for the front proxy's admission list.
        :returns: The binding.
        :raises IdentityBindingConflictError: When bound differently.
        :raises OmnigentError: ``NOT_FOUND`` for an unknown or removed member,
            ``CONFLICT`` when the member already has the maximum bindings,
            ``INVALID_INPUT`` for malformed values.
        """

        def write(session: Session) -> ExternalIdentityBinding:
            return bind_identity_in_session(
                session,
                member_id,
                issuer=issuer,
                subject=subject,
                audience=audience,
                provenance=provenance,
                bound_by_member_id=bound_by_member_id,
                email=email,
            )

        try:
            return run_write_transaction(self._session_immediate, "bind_external_identity", write)
        except IntegrityError:
            # A concurrent bind committed the same identity first; the retry reads it.
            return run_write_transaction(self._session_immediate, "bind_external_identity", write)

    def bind_installation_owner(
        self, user_id: str, identity: ExternalIdentity
    ) -> TeamMember | None:
        """Bind the proxy's identity of the installation's owner to their own membership.

        The single-owner upgrade, once: the proxy asserts the operator (its owner
        pin matched) and the request carries that account's own sign-in, so both
        the address and the account are proven. Only the team's designated
        founding membership is bound, as an active owner without any binding
        yet, with provenance ``migration``; the designation is then cleared in
        the same transaction, so every later link goes through an invitation
        proved by the account's password.

        :param user_id: The account the request's own credential speaks for.
        :param identity: The proxy-verified identity.
        :returns: The member, or ``None`` when *user_id* is not the unbound
            founding owner, or the identity is bound elsewhere.
        """
        if not identity.installation_owner:
            return None

        def write(session: Session) -> TeamMember | None:
            team = lock_team(session)
            row = _lock_current_member(session, user_id)
            if (
                row is None
                or team.founding_member_id is None
                or row.id != team.founding_member_id
                or row.role != _OWNER
                or row.state != _ACTIVE
            ):
                return None
            bound = session.execute(
                select(func.count())
                .select_from(SqlTeamIdentityBinding)
                .where(
                    SqlTeamIdentityBinding.workspace_id == current_workspace_id(),
                    SqlTeamIdentityBinding.member_id == row.id,
                )
            ).scalar_one()
            if bound:
                return None
            team.founding_member_id = None
            session.flush()
            bind_identity_in_session(
                session,
                row.id,
                issuer=identity.issuer,
                subject=identity.subject,
                audience=identity.audience,
                provenance="migration",
                bound_by_member_id=None,
                email=identity.email,
            )
            return _to_member(row)

        try:
            return run_write_transaction(self._session_immediate, "bind_installation_owner", write)
        except (IdentityBindingConflictError, IntegrityError):
            return None
        except OmnigentError as exc:
            if exc.code is ErrorCode.CONFLICT:
                return None
            raise

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

    # ── Invitations and admission ──────────────────────────────────

    def create_invitation(
        self,
        email: str,
        role: InvitationRole,
        *,
        created_by_member_id: str | None,
        ttl_seconds: int,
        target_user_id: str | None = None,
    ) -> tuple[TeamInvitation, str]:
        """Issue a single-use invitation for *email*; returns it and its token.

        An earlier pending invitation for the same address is revoked, so one
        address has at most one usable invitation. Only the token's digest is
        stored: the token is returned once, for the administrator to hand over.

        :param email: The invited address.
        :param role: ``"admin"`` or ``"member"``.
        :param created_by_member_id: The issuing member.
        :param ttl_seconds: Lifetime, 60 seconds to 30 days.
        :param target_user_id: Link this existing account instead of creating one.
        :raises OmnigentError: ``INVALID_INPUT`` for a bad address, role, lifetime
            or target; ``CONFLICT`` when the target already has a membership.
        """
        normalized = normalize_invitation_email(email)
        if role not in _INVITATION_ROLES:
            raise OmnigentError(
                "an invitation offers admin or member", code=ErrorCode.INVALID_INPUT
            )
        if not 60 <= ttl_seconds <= 30 * 24 * 3600:
            raise OmnigentError("invalid invitation lifetime", code=ErrorCode.INVALID_INPUT)
        if target_user_id is not None and target_user_id in _RESERVED_USERS:
            raise OmnigentError("that account cannot join a team", code=ErrorCode.INVALID_INPUT)
        token = secrets.token_urlsafe(32)

        def write(session: Session) -> TeamInvitation:
            lock_team(session)
            if created_by_member_id is not None:
                actor = require_actor(session, created_by_member_id, owner=False)
                if actor.role != _OWNER and (role != "member" or target_user_id is not None):
                    raise OmnigentError(
                        "only an owner invites admins or links existing accounts",
                        code=ErrorCode.FORBIDDEN,
                    )
            now = _now()
            if target_user_id is not None:
                live, _ = _live_generation(session, target_user_id)
                if not live:
                    raise OmnigentError("no such account", code=ErrorCode.INVALID_INPUT)
                if _lock_current_member(session, target_user_id) is not None:
                    raise OmnigentError(
                        "that account already has a team membership", code=ErrorCode.CONFLICT
                    )
            session.execute(
                update(SqlTeamInvitation)
                .where(
                    SqlTeamInvitation.workspace_id == current_workspace_id(),
                    SqlTeamInvitation.email == normalized,
                    SqlTeamInvitation.state == _PENDING,
                )
                .values(state=_REVOKED, closed_at=now)
            )
            row = SqlTeamInvitation(
                id=uuid.uuid4().hex,
                token_digest=_token_digest(token),
                email=normalized,
                role=encode_team_member_role(role),
                target_user_id=target_user_id,
                created_by_member_id=created_by_member_id,
                created_at=now,
                expires_at=now + ttl_seconds,
                state=_PENDING,
            )
            session.add(row)
            session.flush()
            bump_admission_revision(session)
            return _to_invitation(row, now)

        return run_write_transaction(
            self._session_immediate, "create_team_invitation", write
        ), token

    def invitation_for_token(self, token: str) -> TeamInvitation | None:
        """Return the invitation a token names, without using it."""
        if not token or len(token) > 128:
            return None
        with self._session("select_team_invitation_by_token") as session:
            row = session.execute(
                select(SqlTeamInvitation).where(
                    SqlTeamInvitation.workspace_id == current_workspace_id(),
                    SqlTeamInvitation.token_digest == _token_digest(token),
                )
            ).scalar_one_or_none()
            return _to_invitation(row, _now()) if row is not None else None

    def get_invitation(self, invitation_id: str) -> TeamInvitation | None:
        """Return an invitation by id, in any state."""
        with self._session("select_team_invitation") as session:
            row = session.get(SqlTeamInvitation, (current_workspace_id(), invitation_id))
            return _to_invitation(row, _now()) if row is not None else None

    def list_invitations(self, *, limit: int = 100) -> list[TeamInvitation]:
        """List pending invitations, newest first, expired ones included."""
        if not 1 <= limit <= MAX_ADMITTED:
            raise OmnigentError("limit must be 1-1000", code=ErrorCode.INVALID_INPUT)
        with self._session("list_team_invitations") as session:
            now = _now()
            rows = (
                session.execute(
                    select(SqlTeamInvitation)
                    .where(
                        SqlTeamInvitation.workspace_id == current_workspace_id(),
                        SqlTeamInvitation.state == _PENDING,
                    )
                    .order_by(SqlTeamInvitation.created_at.desc(), SqlTeamInvitation.id)
                    .limit(limit)
                )
                .scalars()
                .all()
            )
            return [_to_invitation(row, now) for row in rows]

    def revoke_invitation(
        self, invitation_id: str, *, actor_member_id: str | None = None
    ) -> TeamInvitation:
        """Revoke a pending invitation.

        :param actor_member_id: The owner or admin revoking it, authorized
            inside the write (an admin revokes member invitations only).
        :raises OmnigentError: ``NOT_FOUND`` for an unknown one, ``CONFLICT``
            for one already used or revoked, ``FORBIDDEN`` for an actor who may
            not.
        """

        def write(session: Session) -> TeamInvitation:
            lock_team(session)
            actor = (
                require_actor(session, actor_member_id, owner=False)
                if actor_member_id is not None
                else None
            )
            row = session.execute(
                select(SqlTeamInvitation)
                .where(
                    SqlTeamInvitation.workspace_id == current_workspace_id(),
                    SqlTeamInvitation.id == invitation_id,
                )
                .with_for_update()
                .execution_options(populate_existing=True)
            ).scalar_one_or_none()
            if row is None:
                raise OmnigentError("no such invitation", code=ErrorCode.NOT_FOUND)
            if actor is not None and actor.role != _OWNER and row.role != _MEMBER_ROLE:
                raise OmnigentError(
                    "only an owner revokes an admin's invitation", code=ErrorCode.FORBIDDEN
                )
            if row.state != _PENDING:
                raise OmnigentError(
                    "this invitation is no longer pending", code=ErrorCode.CONFLICT
                )
            now = _now()
            row.state, row.closed_at = _REVOKED, now
            session.flush()
            bump_admission_revision(session)
            return _to_invitation(row, now)

        return run_write_transaction(self._session_immediate, "revoke_team_invitation", write)

    def expire_invitations(self) -> int:
        """Store ``expired`` on pending invitations past their expiry; bump admission once.

        Time alone changes who a proxy should admit when an invitation expires;
        this records it as a change, so the admission list's revision moves.

        :returns: How many invitations expired.
        """

        def write(session: Session) -> int:
            lock_team(session)
            now = _now()
            result = cast(
                CursorResult[tuple[object]],
                session.execute(
                    update(SqlTeamInvitation)
                    .where(
                        SqlTeamInvitation.workspace_id == current_workspace_id(),
                        SqlTeamInvitation.state == _PENDING,
                        SqlTeamInvitation.expires_at <= now,
                    )
                    .values(state=_EXPIRED, closed_at=now)
                ),
            )
            if result.rowcount:
                bump_admission_revision(session)
            return result.rowcount

        return run_write_transaction(self._session_immediate, "expire_team_invitations", write)

    def admission_snapshot(self) -> AdmissionSnapshot:
        """Who a front proxy should admit now, and the revision it reflects.

        Active members' verified addresses and pending, unexpired invitations.
        Suspended and removed members are left out.
        """
        with self._session("select_team_admission") as session:
            now = _now()
            team = session.get(SqlTeam, current_workspace_id())
            members = (
                select(SqlTeamIdentityBinding.email)
                .join(
                    SqlTeamMember,
                    (SqlTeamMember.workspace_id == SqlTeamIdentityBinding.workspace_id)
                    & (SqlTeamMember.id == SqlTeamIdentityBinding.member_id),
                )
                .where(
                    SqlTeamIdentityBinding.workspace_id == current_workspace_id(),
                    SqlTeamIdentityBinding.email.is_not(None),
                    SqlTeamMember.state == _ACTIVE,
                )
            )
            invited = select(SqlTeamInvitation.email).where(
                SqlTeamInvitation.workspace_id == current_workspace_id(),
                SqlTeamInvitation.state == _PENDING,
                SqlTeamInvitation.expires_at > now,
            )
            emails: set[str] = set()
            for statement in (members, invited):
                emails.update(
                    email
                    for email in session.execute(statement.limit(MAX_ADMITTED)).scalars()
                    if email
                )
            owners = {
                email
                for email in session.execute(
                    members.where(SqlTeamMember.role == _OWNER).limit(MAX_ADMITTED)
                ).scalars()
                if email
            }
            next_expiry = session.execute(
                select(func.min(SqlTeamInvitation.expires_at)).where(
                    SqlTeamInvitation.workspace_id == current_workspace_id(),
                    SqlTeamInvitation.state == _PENDING,
                    SqlTeamInvitation.expires_at > now,
                )
            ).scalar_one_or_none()
            listed = tuple(sorted(emails)[:MAX_ADMITTED])
            return AdmissionSnapshot(
                revision=team.admission_revision if team is not None else 0,
                emails=listed,
                owners=tuple(sorted(owners & set(listed))),
                next_change_at=next_expiry,
            )

    # ── Removals ───────────────────────────────────────────────────

    def get_removal(self, member_id: str) -> MemberRemoval | None:
        """Return a removed member's removal record, or ``None``."""
        with self._session("select_team_member_removal") as session:
            found = removals_in_session(session, [member_id])
            return found[0] if found else None

    def list_removals(self, *, limit: int = 50) -> list[MemberRemoval]:
        """Return the latest removals, newest first."""
        if not 1 <= limit <= MAX_ADMITTED:
            raise OmnigentError("limit must be 1-1000", code=ErrorCode.INVALID_INPUT)
        with self._session("list_team_member_removals") as session:
            member_ids = list(
                session.execute(
                    select(SqlTeamMemberRemoval.member_id)
                    .where(SqlTeamMemberRemoval.workspace_id == current_workspace_id())
                    .order_by(
                        SqlTeamMemberRemoval.removed_at.desc(), SqlTeamMemberRemoval.member_id
                    )
                    .limit(limit)
                ).scalars()
            )
            return removals_in_session(session, member_ids)

    def removal_threads(self, member_id: str, *, limit: int = 500) -> list[str]:
        """Return the ids of the threads a removed member owned, in id order."""
        if not 1 <= limit <= MAX_ADMITTED:
            raise OmnigentError("limit must be 1-1000", code=ErrorCode.INVALID_INPUT)
        with self._session("list_team_member_removal_threads") as session:
            return list(
                session.execute(
                    select(SqlTeamMemberRemovalItem.item_id)
                    .where(
                        SqlTeamMemberRemovalItem.workspace_id == current_workspace_id(),
                        SqlTeamMemberRemovalItem.member_id == member_id,
                        SqlTeamMemberRemovalItem.kind == _THREAD_ITEM,
                    )
                    .order_by(SqlTeamMemberRemovalItem.item_id)
                    .limit(limit)
                ).scalars()
            )

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


class _NoGeneration(Enum):
    """Marks a resource type that records no account generation."""

    NONE = "none"


_NO_GENERATION = _NoGeneration.NONE


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

    def _owner_problem(
        self, user_id: str | None, generation: str | _NoGeneration | None = _NO_GENERATION
    ) -> str | None:
        """Classify a recorded owner; ``None`` means a live registration.

        *generation* is the registration a host or schedule recorded, possibly
        ``None``; other resources record none and pass nothing. Once the
        account has a generation, a missing or different one belongs to
        another registration of the same username.
        """
        if user_id is None:
            return "no_owner"
        if user_id == RESERVED_USER_LOCAL:
            return "legacy_local_owner"
        if user_id not in self._generations:
            return "unknown_owner"
        current = self._generations[user_id]
        if isinstance(generation, _NoGeneration) or current is None:
            return None
        if generation is None:
            return "missing_registration"
        return "earlier_registration" if generation != current else None

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
