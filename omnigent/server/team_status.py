"""Who the team's sign-ins are bound to, for a deployment moving behind an identity-aware proxy.

An operator turning the proxy on (``OMNIGENT_EXTERNAL_IDENTITY_MODE=bind``, then ``required``)
needs to know, before requiring it, that the people and machines they use are bound: the
active owners each have a proxy sign-in bound, and each personal host connected through the
proxy presents a sign-in bound to its own owner. :func:`team_status` reports both, plus the
admission revision. It names no address and no Access subject: hosts by ID, owners by count.

A host's identity is resolved when asked, from the sign-in its tunnel's handshake carried, so a
host that connected before its owner's browser bound shows as bound once that binding exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from omnigent.entities.team import TeamMember
from omnigent.server.host_registry import HostRegistry
from omnigent.stores.team_store import SqlAlchemyTeamStore

HostIdentity = Literal["owner", "unbound", "other", "none"]


@dataclass(frozen=True)
class HostIdentityStatus:
    """One connected host and the proxy sign-in its tunnel presented.

    :param identity: ``owner``: bound to the host owner's active membership; ``unbound``: a
        sign-in no member is bound to (a device whose sign-in differs from its owner's
        browser); ``other``: bound to someone else, or to an inactive membership; ``none``:
        no sign-in (a cloud computer's private route, or the proxy is off).
    """

    host_id: str
    owner: str | None
    launch_token: bool
    identity: HostIdentity


@dataclass(frozen=True)
class TeamStatus:
    """The team's binding state.

    :param owners: Active owners.
    :param owners_bound: Active owners with at least one proxy sign-in bound.
    :param revision: The team's admission revision.
    :param hosts: Every connected host, by ID.
    """

    owners: int
    owners_bound: int
    revision: int
    hosts: tuple[HostIdentityStatus, ...]


def team_status(team_store: SqlAlchemyTeamStore, host_registry: HostRegistry) -> TeamStatus:
    """Report the active owners' bindings and each connected host's proxy sign-in."""
    owners = [
        member
        for member in _all_members(team_store)
        if member.role == "owner" and member.state == "active"
    ]
    bound = sum(1 for member in owners if team_store.list_external_identities(member.id))
    hosts: list[HostIdentityStatus] = []
    for host_id in sorted(host_registry.online_host_ids()):
        conn = host_registry.get(host_id)
        if conn is None:
            continue
        hosts.append(
            HostIdentityStatus(
                host_id=conn.host_id,
                owner=conn.owner,
                launch_token=conn.registered_with_managed_token,
                identity=_host_identity(team_store, conn.owner, conn.external_identity),
            )
        )
    return TeamStatus(
        owners=len(owners),
        owners_bound=bound,
        revision=team_store.admission_snapshot().revision,
        hosts=tuple(hosts),
    )


def _all_members(team_store: SqlAlchemyTeamStore) -> list[TeamMember]:
    members: list[TeamMember] = []
    cursor: str | None = None
    while True:
        page, cursor = team_store.list_members(limit=1000, after_user_id=cursor)
        members.extend(page)
        if cursor is None:
            return members


def _host_identity(
    team_store: SqlAlchemyTeamStore, owner: str | None, identity: object
) -> HostIdentity:
    from omnigent.server.external_identity import ExternalIdentity

    if not isinstance(identity, ExternalIdentity):
        return "none"
    member = team_store.member_for_external_identity(
        identity.issuer, identity.subject, audience=identity.audience
    )
    if member is None:
        return "unbound"
    if member.user_id == owner and member.state == "active":
        return "owner"
    return "other"
