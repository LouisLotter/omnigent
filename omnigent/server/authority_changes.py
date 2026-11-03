"""Wake long-lived connections when someone's authority changes.

Open sockets and streams re-check their caller every few seconds. A suspension
or other membership change calls :func:`authority_changed`, which ends every
waiting interval at once so each connection re-checks now rather than at its
next tick. The re-check itself decides; this only shortens the wait. Within one
server process: with several processes the regular interval still bounds it.
"""

from __future__ import annotations

import asyncio
import contextlib

_count = 0
# Future objects, not tenant data: a wake makes each connection re-check its own caller.
# custom-lint: disable-next=workspace-scoped-cache -- holds future objects
_waiters: set[asyncio.Future[None]] = set()


def authority_change_count() -> int:
    """How many changes have been signalled; a loop compares it to re-check early."""
    return _count


def authority_changed() -> None:
    """Signal a change: every waiting re-check runs now. Safe from any thread."""
    global _count
    _count += 1
    for waiter in list(_waiters):
        loop = waiter.get_loop()
        if not loop.is_closed():
            loop.call_soon_threadsafe(_resolve, waiter)


def _resolve(waiter: asyncio.Future[None]) -> None:
    if not waiter.done():
        waiter.set_result(None)


async def sleep_until_authority_change(seconds: float) -> None:
    """Sleep *seconds*, or less when :func:`authority_changed` is called meanwhile."""
    waiter: asyncio.Future[None] = asyncio.get_running_loop().create_future()
    _waiters.add(waiter)
    try:
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(asyncio.shield(waiter), timeout=seconds)
    finally:
        _waiters.discard(waiter)
        if not waiter.done():
            waiter.cancel()
