"""Deliver a Stop that a dropped runner tunnel lost, once the runner is back.

A Stop is forwarded to the runner over its tunnel. When the tunnel has
dropped, or went half-open, the forward fails or goes unanswered while the
turn keeps running. Instead of answering as if the Stop had landed, the
route reports it as pending and this module sends it again once the runner
reconnects, provided the turn is still running and nobody has sent a new
message since.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from omnigent.db.workspace_cache import WorkspaceScopedCache
from omnigent.debug_logging import debug_event
from omnigent.runner.transports.ws_tunnel.registry import TunnelRegistry
from omnigent.runtime import pending_inputs

_logger = logging.getLogger(__name__)

# Bound on the redelivered interrupt's own forward.
_INTERRUPT_FORWARD_TIMEOUT_S = 5.0

# One pending redelivery per session; a newer Stop or message replaces it.
_pending_interrupts: WorkspaceScopedCache[str, asyncio.Task[None]] = WorkspaceScopedCache()
# Counts each session's new messages, so a Stop pressed before one is never
# redelivered after it, even when its first forward was still in flight.
_input_generations: WorkspaceScopedCache[str, int] = WorkspaceScopedCache()


def input_generation(session_id: str) -> int:
    """
    Return how many new messages *session_id* has had, as a Stop's reference point.

    :param session_id: Session id, e.g. ``"conv_abc123"``.
    :returns: The current count.
    """
    return _input_generations.get(session_id, 0)


def note_new_input(session_id: str) -> None:
    """
    Record a new message: an earlier Stop must not interrupt the turn it starts.

    :param session_id: Session id, e.g. ``"conv_abc123"``.
    """
    _input_generations[session_id] = input_generation(session_id) + 1
    cancel_pending_interrupt(session_id)


def cancel_pending_interrupt(session_id: str) -> None:
    """
    Drop a Stop still waiting for the runner.

    :param session_id: Session id, e.g. ``"conv_abc123"``.
    """
    task = _pending_interrupts.pop(session_id, None)
    if task is not None and not task.done():
        task.cancel()


def has_pending_interrupt(session_id: str) -> bool:
    """
    Return whether a Stop for *session_id* waits for its runner to reconnect.

    :param session_id: Session id, e.g. ``"conv_abc123"``.
    :returns: ``True`` while a redelivery is scheduled.
    """
    task = _pending_interrupts.get(session_id)
    return task is not None and not task.done()


def deliver_interrupt_after_reconnect(
    session_id: str,
    *,
    runner_id: str,
    after_generation: int | None,
    tunnel_registry: TunnelRegistry,
    runner_router: Any,
    queued_pending_ids: list[str],
    wait_s: float,
    stop_generation: int,
) -> bool:
    """
    Send a lost Stop again once the runner's tunnel is back.

    :param session_id: Session to stop, e.g. ``"conv_abc123"``.
    :param runner_id: The session's runner, e.g. ``"runner_0123456789abcdef"``.
    :param after_generation: Generation of the tunnel the lost Stop used, so
        only a reconnect counts; ``None`` when the runner had no tunnel.
    :param tunnel_registry: Registry the runner reconnects to.
    :param runner_router: Router that resolves the session's runner client.
    :param queued_pending_ids: Pending inputs queued when the person pressed
        Stop, marked interrupted once the Stop lands.
    :param wait_s: How long to wait for the reconnect, e.g. the server's grace
        for a dropped runner.
    :param stop_generation: :func:`input_generation` when the person pressed
        Stop. A newer message means the Stop is not redelivered.
    :returns: ``False`` when a newer message already superseded the Stop.
    """
    if input_generation(session_id) != stop_generation:
        # Superseded: it must not replace a newer Stop's redelivery either.
        return False
    cancel_pending_interrupt(session_id)
    task = asyncio.create_task(
        _deliver_after_reconnect(
            session_id,
            runner_id=runner_id,
            after_generation=after_generation,
            tunnel_registry=tunnel_registry,
            runner_router=runner_router,
            queued_pending_ids=queued_pending_ids,
            wait_s=wait_s,
            stop_generation=stop_generation,
        ),
        name=f"interrupt-after-reconnect-{session_id}",
    )
    _pending_interrupts[session_id] = task

    def _forget(done: asyncio.Task[None]) -> None:
        if _pending_interrupts.get(session_id) is done:
            _pending_interrupts.pop(session_id, None)

    task.add_done_callback(_forget)
    return True


async def _deliver_after_reconnect(
    session_id: str,
    *,
    runner_id: str,
    after_generation: int | None,
    tunnel_registry: TunnelRegistry,
    runner_router: Any,
    queued_pending_ids: list[str],
    wait_s: float,
    stop_generation: int,
) -> None:
    from omnigent.server.routes._sessions.common import (
        _interrupt_fenced_sessions,
        _session_status_cache,
    )
    from omnigent.server.routes._sessions.helpers import _get_runner_client

    reconnected = await tunnel_registry.wait_for_runner(
        runner_id, timeout_s=wait_s, newer_than_generation=after_generation
    )
    outcome = "runner_not_back"
    if reconnected is not None:
        outcome = "turn_already_ended"
        if input_generation(session_id) != stop_generation:
            outcome = "superseded_by_new_message"
        elif _session_status_cache.get(session_id) in ("running", "waiting"):
            outcome = await _send_interrupt(session_id, runner_router, _get_runner_client)
            if outcome == "delivered":
                _interrupt_fenced_sessions.add(session_id)
                pending_inputs.mark_interrupted(session_id, queued_pending_ids)
    _logger.info(
        "Stop redelivery after runner reconnect finished for session=%s: %s",
        session_id,
        outcome,
        extra=debug_event(
            "interrupt_redelivery_finished",
            session_id=session_id,
            runner_id=runner_id,
            outcome=outcome,
        ),
    )


async def _send_interrupt(session_id: str, runner_router: Any, get_runner_client: Any) -> str:
    """
    Forward one interrupt to the session's current runner.

    :returns: ``"delivered"``, ``"refused"`` or ``"lost"``.
    """
    client: httpx.AsyncClient | None = await get_runner_client(session_id, runner_router)
    if client is None:
        return "lost"
    try:
        response = await client.post(
            f"/v1/sessions/{session_id}/events",
            json={"type": "interrupt"},
            timeout=_INTERRUPT_FORWARD_TIMEOUT_S,
        )
    except (httpx.HTTPError, ConnectionError):
        return "lost"
    return "delivered" if response.status_code < 400 else "refused"
