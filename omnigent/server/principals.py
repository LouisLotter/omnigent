"""The kind of caller behind an authenticated request.

Authentication resolves the account a credential speaks for. A runner's
token, a host connection and a person's browser can all resolve to the same
account, so decisions that need the person themselves (approvals, team
administration) must also look at the principal kind recorded here.
"""

from __future__ import annotations

import dataclasses
from enum import Enum

from starlette.requests import HTTPConnection

_PRINCIPAL_SCOPE_KEY = "omnigent.principal"


class PrincipalKind(str, Enum):
    """What holds the credential.

    - ``HUMAN``: the person's own sign-in, such as a browser session, a CLI
      login or a trusted proxy's identity header.
    - ``DELEGATED_CLIENT``: an application the person approved to act for
      them, such as a device-authorization client.
    - ``RUNNER``: a runner acting for its thread's owner.
    - ``HOST``: a host connection acting for the host's owner.
    - ``SERVICE``: a machine client with no person behind it.
    """

    HUMAN = "human"
    DELEGATED_CLIENT = "delegated_client"
    RUNNER = "runner"
    HOST = "host"
    SERVICE = "service"


@dataclasses.dataclass(frozen=True)
class Principal:
    """An authenticated caller.

    :param kind: What holds the credential.
    :param user_id: The account it acts for, or ``None`` for a service.
    :param runner_id: The runner, for a runner principal.
    :param host_id: The host, for a host principal.
    """

    kind: PrincipalKind
    user_id: str | None
    runner_id: str | None = None
    host_id: str | None = None

    @property
    def is_human(self) -> bool:
        """Whether the person is acting in person rather than through software."""
        return self.kind is PrincipalKind.HUMAN


def record_principal(connection: HTTPConnection, principal: Principal) -> None:
    """Attach the authenticated principal to a request or WebSocket scope."""
    connection.scope[_PRINCIPAL_SCOPE_KEY] = principal


def clear_principal(connection: HTTPConnection) -> None:
    """Drop any principal before authenticating the connection again."""
    connection.scope.pop(_PRINCIPAL_SCOPE_KEY, None)


def recorded_principal(connection: HTTPConnection) -> Principal | None:
    """Return the principal recorded for this request, if any."""
    principal = connection.scope.get(_PRINCIPAL_SCOPE_KEY)
    return principal if isinstance(principal, Principal) else None
