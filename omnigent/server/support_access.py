"""Support access: a team's owners and admins act on other members' computers and connections.

Turned on with ``create_app(admin_support_access=True)`` on a server with a team.
An owner or admin acts *on* a member's things, never *as* the member:

- Ownership never moves. A computer created for a member is the member's, and its
  provider is called with the member's owner context, read from the server's records.
- Logins stay the person's own. No owner or admin starts, finishes or enters a
  subscription, MCP, GitHub or harness login for someone else: a provider's terms
  tie a login to the person who holds it. An admin sees whether the member is
  signed in or connected, and may remove a connection.
- Only from a browser sign-in. A CLI token, device grant, runner or host never
  gets support access.

Every support action logs one line naming the acting member and the member acted
for by member id, never by username or address.

A request checks the actor and the member when it starts. Each write it makes,
and each host command it sends after a wait, checks them again under the team
lock (:func:`omnigent.stores.team_store.authorize_support_action`), so a role
change, suspension, password reset or removal that lands in between is seen.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from omnigent.entities.team import ADMIN_ROLES, MemberState, SupportAction, TeamMember

if TYPE_CHECKING:
    from omnigent.server.accounts_store import SqlAlchemyAccountStore
    from omnigent.server.principals import Principal
    from omnigent.stores.host_store import Host
    from omnigent.stores.team_store import SqlAlchemyTeamStore

_logger = logging.getLogger(__name__)

# Members whose things an owner or admin may act on: a suspended member's
# computers can still be looked after or deleted.
SUPPORTED_STATES: frozenset[MemberState] = frozenset({"active", "suspended"})
# A computer is added only for an active member.
ACTIVE_ONLY: frozenset[MemberState] = frozenset({"active"})


@dataclass(frozen=True)
class SupportedMember:
    """A member an owner or admin acts for, with the registration read with the membership."""

    member: TeamMember
    generation: str | None

    def action(
        self, actor: TeamMember, states: frozenset[MemberState] = SUPPORTED_STATES
    ) -> SupportAction:
        """The action *actor* takes for this member, for its write to check again."""
        return SupportAction(
            actor_member_id=actor.id,
            member_id=self.member.id,
            member_generation=self.generation,
            states=states,
        )


@dataclass(frozen=True)
class SupportAccess:
    """Who may act on other members' things, and for whom.

    The methods read the stores: call them from a worker thread in async code.

    :param team_store: The team's memberships.
    :param account_store: The accounts, whose writes apply support actions.
    """

    team_store: SqlAlchemyTeamStore
    account_store: SqlAlchemyAccountStore

    def supporter(self, principal: Principal | None, user_id: str | None) -> TeamMember | None:
        """The caller as an active owner or admin signed in through a browser, else ``None``.

        :param principal: The request's principal.
        :param user_id: The authenticated caller.
        """
        if user_id is None or principal is None or not principal.is_interactive_human:
            return None
        if principal.user_id != user_id:
            return None
        member = self.team_store.get_member_for_account(user_id)
        if member is None or member.state != "active" or member.role not in ADMIN_ROLES:
            return None
        return member

    def member(
        self,
        *,
        member_id: str | None = None,
        user_id: str | None = None,
        states: frozenset[MemberState] = SUPPORTED_STATES,
    ) -> SupportedMember | None:
        """The current member *member_id* (or *user_id*'s), when in one of *states*."""
        found = self.team_store.member_registration(member_id=member_id, user_id=user_id)
        if found is None or found[0].state not in states:
            return None
        return SupportedMember(*found)

    def member_for_host(self, host: Host) -> SupportedMember | None:
        """The membership of *host*'s owner, when it is active or suspended.

        ``None`` for a host of an earlier registration of the same username.
        """
        found = self.member(user_id=host.user_id)
        if found is None:
            return None
        if host.account_generation is not None and host.account_generation != found.generation:
            return None
        return found


def log_support_action(
    actor: TeamMember, action: str, member: TeamMember, *, subject: str | None = None
) -> None:
    """Record that *actor* did *action* for *member*: member ids only.

    :param action: What was done, e.g. ``"start_host"``.
    :param subject: What it was done to, e.g. a host id or a service id.
    """
    _logger.info(
        "support action=%s actor_member=%s member=%s%s",
        action,
        actor.id,
        member.id,
        f" subject={subject}" if subject is not None else "",
    )
