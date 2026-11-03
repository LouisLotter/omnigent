"""The owner a persistent host's sandbox is launched for, for providers that ask.

A sandbox provider declaring
:attr:`~omnigent.onboarding.sandboxes.types.SandboxCapabilities.owner_context`
receives ``owner=``, a :class:`~omnigent.onboarding.sandboxes.types.SandboxOwner`,
on every lifecycle call the server makes for a persistent host. The owner comes
from the server's own records for the host: its owner and host id, and that
account's current team membership. No request field chooses it, so a provider
can bind each sandbox to one person and refuse every call for anyone else's.
Providers without the capability are called exactly as before.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from omnigent.onboarding.sandboxes.types import SandboxOwner

if TYPE_CHECKING:
    from omnigent.onboarding.sandboxes.base import SandboxLifecycle
    from omnigent.stores.host_store import Host, HostStore


def sandbox_owner(host: Host, host_store: HostStore | None) -> SandboxOwner:
    """Return who *host*'s sandbox belongs to.

    Reads the store (call it from a worker thread in async code).

    :param host: The host row, live or tombstoned (a removal still names its
        owner).
    :param host_store: The store for the membership lookup; ``None`` names the
        account alone.
    :returns: The owner; ``member_id`` and ``team_id`` are ``None`` when the
        account has no current team membership.
    """
    membership = (
        None
        if host_store is None
        else host_store.owner_membership(host.user_id, host.account_generation)
    )
    member_id, team_id = membership if membership is not None else (None, None)
    return SandboxOwner(
        user_id=host.user_id, host_id=host.host_id, member_id=member_id, team_id=team_id
    )


def owner_for(
    launcher: SandboxLifecycle, host: Host | None, host_store: HostStore | None
) -> SandboxOwner | None:
    """Return *host*'s owner when *launcher* takes one, else ``None``.

    :param launcher: The provider being called.
    :param host: The host the call is for, or ``None`` when there is none.
    :param host_store: The store for the membership lookup.
    """
    if host is None or not takes_owner(launcher):
        return None
    return sandbox_owner(host, host_store)


def takes_owner(launcher: object) -> bool:
    """Whether *launcher* declares ``owner_context``; a launcher without capabilities does not."""
    capabilities = getattr(launcher, "capabilities", None)
    return getattr(capabilities, "owner_context", False) is True


def launcher_call(
    launcher: SandboxLifecycle,
    method: str,
    host: Host | None,
    host_store: HostStore | None,
    /,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Call ``launcher.<method>(*args, **kwargs)``, naming *host*'s owner if it asks.

    Synchronous: async callers run it in a worker thread
    (``asyncio.to_thread(launcher_call, ...)``), the owner lookup included.

    :param launcher: The provider.
    :param method: The lifecycle method, e.g. ``"resume"``.
    :param host: The host the call is for.
    :param host_store: The store for the membership lookup.
    :returns: What the method returns.
    """
    owner = owner_for(launcher, host, host_store)
    if owner is not None:
        kwargs["owner"] = owner
    return getattr(launcher, method)(*args, **kwargs)
