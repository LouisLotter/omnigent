"""In-process registries of pending harness-side elicitations.

The sessions route uses these to bridge a client-issued elicitation
verdict (PATCH on the session) to the in-flight Future an upstream
caller is awaiting. Keyed by ``elicitation_id`` — but harness-supplied
ids are deterministic (derived from the session and request), and
imported sessions share a ``conversation_id`` across workspaces, so the
key is namespaced by workspace to keep one tenant's verdict from
resolving another's parked Future.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from omnigent.db.workspace_cache import WorkspaceScopedCache
from omnigent.server.schemas import ElicitationResult

_harness_elicitation_registry: WorkspaceScopedCache[str, asyncio.Future[ElicitationResult]] = (
    WorkspaceScopedCache()
)

# Maps ``elicitation_id`` to the conversation id that issued it, so
# the PATCH handler can verify the caller owns the elicitation
# before resolving the Future.
_harness_elicitation_owners: WorkspaceScopedCache[str, str] = WorkspaceScopedCache()


@dataclass
class _ParkedHarnessElicitation:
    """
    Per-elicitation state used to resolve a prompt that was answered in
    the native terminal rather than the web UI.

    Lives alongside the legacy ``_harness_elicitation_registry`` /
    ``_harness_elicitation_owners`` entries (same ``elicitation_id``);
    those keep backing the web-verdict path and the ownership check
    untouched, while this carries the extra state the terminal-resolved
    fast path needs.

    :param session_id: Conversation id that issued the prompt, e.g.
        ``"conv_abc123"`` (mirrors the ``_harness_elicitation_owners``
        entry; kept here so the correlation helper is self-contained).
    :param tool_name: Gated tool name from the harness hook, e.g.
        ``"Bash"``. ``None`` when the harness supplied none (e.g. a
        Codex elicitation), which disables tool-correlated resolution.
    :param tool_input: Gated tool input the prompt was raised for, e.g.
        ``{"command": "rm -rf x"}``. ``None`` when not supplied.
    :param resolved_elsewhere: Set when a mirrored tool result for the
        SAME gated tool proves the prompt was already answered in the
        native terminal, so the parked hook long-poll returns promptly
        instead of blocking until its timeout. This positive,
        application-level signal is needed because
        ``request.is_disconnected()`` does not fire promptly behind the
        Databricks Apps proxy's idle-connection heartbeats.
    :param request_fingerprint: Digest of the elicitation's request
        params, e.g. a sha256 hex string. Copied onto a verdict
        tombstone written while this waiter may be a zombie, so a
        re-park only adopts the verdict for the SAME logical question
        (harness ids can be reused across distinct requests). ``None``
        when the caller supplied no params to fingerprint.
    """

    session_id: str
    tool_name: str | None
    tool_input: dict[str, Any] | None
    resolved_elsewhere: asyncio.Event
    request_fingerprint: str | None = None


@dataclass(frozen=True)
class _PreResolvedHarnessElicitation:
    """
    Tombstone for a harness elicitation resolved before hook registration.

    ``result`` distinguishes the two producers: a web verdict that
    arrived while no wait was parked (a re-park returns it), vs ``None``
    from ``external_elicitation_resolved`` — the terminal already
    answered, so a re-park fail-asks.

    :param session_id: Omnigent session id that issued the resolution, e.g.
        ``"conv_abc123"``.
    :param created_at: Wall-clock timestamp from ``time.time()``, e.g.
        ``1710000000.0``.
    :param result: Web verdict to hand a re-parking hook, e.g.
        ``ElicitationResult(action="accept")``, or ``None`` for a
        terminal-side resolution.
    :param request_fingerprint: Digest of the request params the verdict
        answered, copied from the parked waiter when the resolve found
        one still registered (or digested from the pending prompt on the
        nothing-parked gap path). A re-park adopts a verdict-carrying
        tombstone only when its own params produce the same digest, so a
        LATER, different question that reuses the same harness id
        (harness request ids recur) can never inherit a stale approval.
        ``None`` when the producer had no params to fingerprint; a
        verdict-carrying tombstone without a fingerprint fails closed at
        consume time (dropped, prompt re-published) rather than falling
        back to adopt-by-id.
    """

    session_id: str
    created_at: float
    result: ElicitationResult | None = None
    request_fingerprint: str | None = None


# Maps ``elicitation_id`` to its parked-elicitation state. Populated
# while a harness hook long-poll is parked; popped when it returns.
_harness_parked_elicitations: WorkspaceScopedCache[str, _ParkedHarnessElicitation] = (
    WorkspaceScopedCache()
)

# Maps deterministic ``elicitation_id`` values to the session that
# resolved them before the harness hook registered its parked wait. The
# hook consumes the tombstone at registration time, which closes the
# race between a native client answering instantly and the Omnigent hook
# request reaching this process.
_harness_pre_resolved_elicitations: WorkspaceScopedCache[str, _PreResolvedHarnessElicitation] = (
    WorkspaceScopedCache()
)


@dataclass(frozen=True)
class _HarnessElicitationDeadline:
    """
    When one parked harness elicitation stops waiting, across re-parks.

    A host re-posts a long-poll that went unanswered with the same id, and
    each re-post re-parks the elicitation. The wait still ends when the first
    park's timeout runs out, instead of restarting with every re-post.

    :param session_id: Session that parked the elicitation, e.g. ``"conv_abc123"``.
    :param request_fingerprint: Digest of the question, so a different question
        that reuses the id starts its own deadline.
    :param deadline: ``time.monotonic()`` value at which the wait ends.
    """

    session_id: str
    request_fingerprint: str
    deadline: float


# Maps ``elicitation_id`` to its wait deadline while it may still be re-parked.
# Popped when the wait settles or times out; pruned once long past.
_harness_elicitation_deadlines: WorkspaceScopedCache[str, _HarnessElicitationDeadline] = (
    WorkspaceScopedCache()
)


@dataclass(frozen=True)
class _AnsweredHarnessElicitation:
    """
    The outcome an ASK gate reached for one question, kept for re-posts.

    A host re-posts a poll whose answer was lost. The re-post must get the same
    outcome without applying the approval's writes a second time.

    :param session_id: Session that asked, e.g. ``"conv_abc123"``.
    :param request_fingerprint: Digest of the question.
    :param approved: Whether the person approved.
    :param declined_reason: The decline reason when the person declined, else ``None``.
    :param answered_at: ``time.monotonic()`` of the answer.
    """

    session_id: str
    request_fingerprint: str
    approved: bool
    declined_reason: str | None
    answered_at: float


# Maps ``elicitation_id`` to its ASK gate's outcome for the retention period.
_harness_elicitation_outcomes: WorkspaceScopedCache[str, _AnsweredHarnessElicitation] = (
    WorkspaceScopedCache()
)


def reset_for_tests() -> None:
    """
    Clear every elicitation registry. For test isolation only.

    The registries are module-global and keyed by
    ``elicitation_id``, so entries left behind by one test (a severed
    long-poll, an unresolved prompt, a pre-resolved tombstone) are
    visible to every later test in the same worker process. Mirrors
    :func:`omnigent.runtime.pending_elicitations.reset_for_tests`.
    Not for production callers.
    """
    _harness_elicitation_registry.clear()
    _harness_elicitation_owners.clear()
    _harness_parked_elicitations.clear()
    _harness_pre_resolved_elicitations.clear()
    _harness_elicitation_deadlines.clear()
    _harness_elicitation_outcomes.clear()


__all__ = [
    "_AnsweredHarnessElicitation",
    "_HarnessElicitationDeadline",
    "_ParkedHarnessElicitation",
    "_PreResolvedHarnessElicitation",
    "_harness_elicitation_deadlines",
    "_harness_elicitation_outcomes",
    "_harness_elicitation_owners",
    "_harness_elicitation_registry",
    "_harness_parked_elicitations",
    "_harness_pre_resolved_elicitations",
]
