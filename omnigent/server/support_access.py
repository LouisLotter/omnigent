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
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from omnigent.db.account_authority import AccountAuthority
from omnigent.db.db_models import current_workspace_id
from omnigent.entities.team import ADMIN_ROLES, TeamMember

if TYPE_CHECKING:
    from omnigent.server.accounts_store import SqlAlchemyAccountStore
    from omnigent.server.principals import Principal
    from omnigent.stores.host_store import Host
    from omnigent.stores.team_store import SqlAlchemyTeamStore

_logger = logging.getLogger(__name__)

# Members whose things an owner or admin may act on: a suspended member's
# computers can still be looked after or deleted.
SUPPORTED_STATES = frozenset({"active", "suspended"})


@dataclass(frozen=True)
class SupportAccess:
    """Who may act on other members' things, and for whom.

    The methods read the stores: call them from a worker thread in async code.

    :param team_store: The team's memberships.
    :param account_store: The accounts, for a member's current registration.
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

    def member_for_host(self, host: Host) -> TeamMember | None:
        """The membership of *host*'s owner, when it is active or suspended.

        ``None`` for a host of an earlier registration of the same username.
        """
        member = self.team_store.member_for_owner(
            AccountAuthority(host.user_id, host.account_generation, current_workspace_id())
        )
        return member if member is not None and member.state in SUPPORTED_STATES else None

    def registration(self, member: TeamMember) -> str | None:
        """The member's current account registration, to pin a write made for them.

        :returns: The account generation, or ``None`` when the account has none.
        :raises LookupError: When the member's account is gone.
        """
        account = self.account_store.get_user(member.user_id)
        if account is None:
            raise LookupError(member.id)
        return account.account_generation


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
