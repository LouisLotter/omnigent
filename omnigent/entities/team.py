"""Team, membership and external-identity entities.

One team belongs to each installation (workspace). A member is an existing
Omnigent account plus a stable member id, a role and a lifecycle state; it is
never a separate person record. External identities from a verified sign-in
provider bind to a member by ``(issuer, subject)``, never by email.
"""

from __future__ import annotations

import dataclasses
from typing import Literal

MemberRole = Literal["owner", "admin", "member"]
MemberState = Literal["invited", "active", "suspended", "removed"]
IdentityProvenance = Literal["invitation", "account_proof", "migration"]
InvitationState = Literal["pending", "redeemed", "revoked", "expired"]
# Roles an invitation may offer; an owner is appointed by changing a member's role.
InvitationRole = Literal["admin", "member"]

# Roles whose active members hold the account admin flag.
ADMIN_ROLES: frozenset[MemberRole] = frozenset({"owner", "admin"})


def account_admin_flag(role: MemberRole, state: MemberState) -> bool:
    """Return the ``users.is_admin`` value a membership implies.

    Only an active owner or admin is an account administrator; invited,
    suspended and removed members are not.

    :param role: The member's role, e.g. ``"admin"``.
    :param state: The member's state, e.g. ``"active"``.
    :returns: Whether the account should carry the admin flag.
    """
    return state == "active" and role in ADMIN_ROLES


@dataclasses.dataclass(frozen=True)
class Team:
    """The installation's team.

    :param id: Stable team id, a bare 32-char hex UUID.
    :param name: Display name, or ``None`` when unnamed.
    :param created_at: Unix epoch seconds when the team was created.
    """

    id: str
    name: str | None
    created_at: int


@dataclasses.dataclass(frozen=True)
class TeamMember:
    """One person's membership, mapped to their Omnigent account.

    :param id: Stable member id, a bare 32-char hex UUID. Ownership
        references use it rather than the reusable username.
    :param user_id: The account (``users.id``) this membership maps to,
        e.g. ``"alice"``. Kept after removal for history.
    :param role: ``"owner"``, ``"admin"`` or ``"member"``.
    :param state: ``"invited"``, ``"active"``, ``"suspended"`` or
        ``"removed"``.
    :param created_at: Unix epoch seconds when the membership was created.
    :param updated_at: Unix epoch seconds of the last role or state change.
    """

    id: str
    user_id: str
    role: MemberRole
    state: MemberState
    created_at: int
    updated_at: int

    @property
    def is_current(self) -> bool:
        """Whether the membership still maps to its account (not removed)."""
        return self.state != "removed"


@dataclasses.dataclass(frozen=True)
class ExternalIdentityBinding:
    """A verified external identity tied to one member.

    :param issuer: The identity provider's issuer, e.g.
        ``"https://example.cloudflareaccess.com"``.
    :param subject: The provider's stable subject for the person.
    :param audience: The audience the provider's assertions must carry for
        this binding, e.g. an Access application's audience tag.
    :param member_id: The bound member's id.
    :param provenance: How the identity was verified when bound.
    :param bound_by_member_id: The member who completed the binding, or
        ``None`` when the system bound it (for example the upgrade).
    :param created_at: Unix epoch seconds when the binding was created.
    :param email: The address the identity was verified with, if any; for the
        front proxy's admission list, never an authorization key.
    """

    issuer: str
    subject: str
    audience: str
    member_id: str
    provenance: IdentityProvenance
    bound_by_member_id: str | None
    created_at: int
    email: str | None = None


@dataclasses.dataclass(frozen=True)
class TeamInvitation:
    """A single-use invitation to join the team, or to link an existing account.

    :param id: Stable invitation id, a bare 32-char hex UUID.
    :param email: The invited address, lowercase. A proxy-verified sign-in
        must carry it to redeem the invitation.
    :param role: The role the member gets, ``"admin"`` or ``"member"``.
    :param target_user_id: For a link invitation, the existing account the
        invitee proves with its password; ``None`` creates a new account.
    :param created_by_member_id: The member who issued it.
    :param created_at: Unix epoch seconds when it was issued.
    :param expires_at: Unix epoch seconds after which it cannot be redeemed.
    :param state: ``"pending"``, ``"redeemed"``, ``"revoked"`` or ``"expired"``.
    :param redeemed_member_id: The membership it created or linked, once redeemed.
    """

    id: str
    email: str
    role: InvitationRole
    target_user_id: str | None
    created_by_member_id: str | None
    created_at: int
    expires_at: int
    state: InvitationState
    redeemed_member_id: str | None = None


@dataclasses.dataclass(frozen=True)
class MemberRemoval:
    """The record of one member's removal, from its own transaction.

    :param member_id: The removed membership.
    :param user_id: Their username, kept for display (e.g. ``"alice"``).
    :param removed_by_member_id: The owner or admin who removed them, or
        ``None`` for the system.
    :param removed_at: Unix epoch seconds of the removal.
    :param admission_revision: The admission revision the removal committed.
    :param thread_count: Threads they owned; owners and admins can still open them.
    :param computers: Cloud computers handed to their provider for removal.
    :param computers_pending: Of those, the ones whose provider has not yet
        confirmed the removal; they are retried.
    """

    member_id: str
    user_id: str
    removed_by_member_id: str | None
    removed_at: int
    admission_revision: int
    thread_count: int
    computers: int
    computers_pending: int


@dataclasses.dataclass(frozen=True)
class AdmissionSnapshot:
    """Who a front proxy should admit, for one membership revision.

    :param revision: Increases with every change that alters the list; a
        proxy ignores a snapshot older than the last one it applied.
    :param emails: Sorted lowercase addresses: active members' verified
        sign-ins and pending, unexpired invitations.
    :param owners: The active owners' addresses among *emails*; a proxy
        refuses a list without one, so no list locks every owner out.
    :param next_change_at: When the next pending invitation expires, in Unix
        epoch seconds, so a syncer pushes again then; ``None`` without one.
    """

    revision: int
    emails: tuple[str, ...]
    owners: tuple[str, ...] = ()
    next_change_at: int | None = None


@dataclasses.dataclass(frozen=True)
class OwnershipFinding:
    """An ownership the team inventory cannot assign on its own.

    :param resource: ``"account"``, ``"thread"``, ``"host"``,
        ``"connection"`` or ``"schedule"``.
    :param resource_id: The resource's id; for a connection,
        ``"<user_id>/<provider>/<account_id>"``.
    :param reason: A stable code, e.g. ``"no_owner"`` or
        ``"multiple_owners"``.
    :param owners: The usernames involved, sorted.
    """

    resource: str
    resource_id: str
    reason: str
    owners: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class InventoryAccount:
    """A live account as the inventory sees it, without credentials.

    :param user_id: The username, e.g. ``"alice"``.
    :param is_admin: The account admin flag.
    :param has_password: Whether the account signs in with a password.
    :param member_id: The current membership's id, or ``None``.
    :param role: The current membership's role, or ``None``.
    :param state: The current membership's state, or ``None``.
    """

    user_id: str
    is_admin: bool
    has_password: bool
    member_id: str | None
    role: MemberRole | None
    state: MemberState | None


@dataclasses.dataclass(frozen=True)
class TeamInventory:
    """Ownership report for moving an installation to a team.

    Holds identifiers and counts only: no password hashes, tokens,
    connection secrets or provider metadata.

    :param team_id: The team's id, or ``None`` before the team exists.
    :param accounts: Live accounts, sorted by username.
    :param thread_owners: Top-level threads per actual owner.
    :param host_owners: Live hosts per owner.
    :param connection_owners: Connections per owner.
    :param schedule_owners: Schedules that can still fire, per owner.
    :param findings: Ownerships left unassigned, in scan order per resource.
    """

    team_id: str | None
    accounts: tuple[InventoryAccount, ...]
    thread_owners: dict[str, int]
    host_owners: dict[str, int]
    connection_owners: dict[str, int]
    schedule_owners: dict[str, int]
    findings: tuple[OwnershipFinding, ...]

    @property
    def clean(self) -> bool:
        """Whether every resource has exactly one assignable owner."""
        return not self.findings
