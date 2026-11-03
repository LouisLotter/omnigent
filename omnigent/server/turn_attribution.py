"""Build the attribution of a turn from the server's own records.

Who started a turn comes from the request's sign-in, a scheduled task's record,
the turn actor the server recorded on the thread an agent works for, or the
sender recorded when a message was queued for a terminal harness. Never from a
runner's claim. Whose thread, computer and harness login it used come from the
actual-ownership lookups and the host row. The only value a computer reports
is its harness login state, and it is labelled as reported.

Building never fails a turn: a section that cannot be read is recorded as
unknown.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from omnigent.entities.attribution import (
    HostAttribution,
    InitiatorAttribution,
    InitiatorKind,
    InitiatorSource,
    LoginState,
    OwnerAttribution,
    RequestedReported,
    SubscriptionAttribution,
    TurnAttribution,
)

if TYPE_CHECKING:
    from omnigent.db.account_authority import AccountAuthority
    from omnigent.entities.conversation import Conversation
    from omnigent.stores import ConversationStore
    from omnigent.stores.host_store import Host, HostStore

_logger = logging.getLogger(__name__)

TURN_ACTOR_LABEL = "omnigent.turn_actor"
"""The person answerable for the thread's current turn; server-written only."""

TURN_STARTED_AT_LABEL = "omnigent.turn_started_at"
"""Epoch seconds the current turn started; server-written only. Everyone whose
input joined the thread since then counts as an initiator for approvals."""

_host_store: HostStore | None = None


def set_attribution_host_store(store: HostStore | None) -> None:
    """Wire the host store the builder reads computers from (set at server start)."""
    global _host_store
    _host_store = store


@dataclass(frozen=True)
class TurnInitiator:
    """Who started a turn, before the rest of its attribution is read.

    :param kind: ``person``, ``schedule``, ``agent`` or ``terminal``.
    :param user: The person, or a schedule's owner.
    :param source: Where the value came from; see
        :data:`omnigent.entities.attribution.InitiatorSource`.
    :param registration: The person's account registration, when known.
    :param schedule_id: The scheduled task, for ``schedule``.
    :param from_session_id: The thread whose agent sent it, for ``agent``.
    :param on_behalf_of: The person an agent acted for, for ``agent``.
    """

    kind: InitiatorKind
    user: str | None
    source: InitiatorSource
    registration: str | None = None
    schedule_id: str | None = None
    from_session_id: str | None = None
    on_behalf_of: str | None = None

    @property
    def responsible_person(self) -> str | None:
        """The person answerable for the turn: an agent's principal, else its initiator."""
        return self.on_behalf_of if self.kind == "agent" else self.user

    def as_attribution(self) -> InitiatorAttribution:
        return InitiatorAttribution(
            kind=self.kind,
            user=self.user,
            registration=self.registration,
            schedule_id=self.schedule_id,
            from_session_id=self.from_session_id,
            on_behalf_of=self.on_behalf_of,
            source=self.source,
        )


def person_initiator(user: str | None, registration: str | None = None) -> TurnInitiator:
    """A signed-in person sent the turn."""
    return TurnInitiator("person", user, "signed_in", registration=registration)


def schedule_initiator(
    owner: str | None, task_id: str, registration: str | None = None
) -> TurnInitiator:
    """A scheduled task fired the turn for its owner."""
    return TurnInitiator("schedule", owner, "schedule", registration, schedule_id=task_id)


def startup_initiator(sender: str | None, scheduled_task_id: str | None) -> TurnInitiator:
    """Who starts a saved startup input's turn: its schedule, or the person who saved it."""
    if scheduled_task_id is not None:
        return schedule_initiator(sender, scheduled_task_id)
    return person_initiator(sender)


def actual_owner_or_none(
    conversation_store: ConversationStore, session_id: str
) -> AccountAuthority | None:
    """The thread's actual owner; ``None`` when unknown or unreadable (never raises)."""
    try:
        return conversation_store.get_actual_session_owner(session_id)
    except Exception:
        _logger.warning("turn attribution: could not read the owner", exc_info=True)
        return None


def derive_agent_initiator(
    conversation_store: ConversationStore,
    conversation: Conversation,
    *,
    from_session_id: str | None = None,
) -> TurnInitiator:
    """An agent's message, for the person the server recorded where the agent works.

    A sub-agent's prompt works for its parent thread's turn actor; a result an
    agent returns into its own thread, for that thread's; a message from another
    thread, for the sending thread's. With no recorded turn actor, the agent is
    acting for the thread's actual owner.
    """
    source_id = from_session_id or conversation.parent_conversation_id or conversation.id
    source = (
        conversation
        if source_id == conversation.id
        else conversation_store.get_conversation(source_id)
    )
    person = (source.labels or {}).get(TURN_ACTOR_LABEL) if source is not None else None
    origin: InitiatorSource = "thread_record"
    if not person:
        owner = actual_owner_or_none(conversation_store, conversation.id)
        person = owner.user_id if owner is not None else None
        origin = "owner_assumed"
    return TurnInitiator(
        "agent",
        None,
        origin,
        from_session_id=source_id if source_id != conversation.id else None,
        on_behalf_of=person,
    )


def _host_id(conversation_store: ConversationStore, conversation: Conversation) -> str | None:
    """The computer the thread runs on; a sub-agent without its own runs on its top thread's."""
    if conversation.host_id or conversation.root_conversation_id in (None, conversation.id):
        return conversation.host_id
    root = conversation_store.get_conversation(conversation.root_conversation_id)
    return root.host_id if root is not None else None


def _host_kind(host: Host) -> str:
    if host.sandbox_provider is None:
        return "personal"
    return "cloud_computer" if host.lifecycle == "persistent" else "sandbox"


def _login_state(harness: str | None, host: Host) -> tuple[LoginState, bool]:
    """The computer's own report of its login for *harness*; ``True`` if it reported."""
    from omnigent.harness_availability import HARNESS_NEEDS_AUTH, reported_harness_availability

    if host.configured_harnesses is None:
        return "unknown", False
    available, reason = reported_harness_availability(harness, host.configured_harnesses)
    if available is True:
        return "ready", True
    if available is False:
        return ("needs_auth" if reason == HARNESS_NEEDS_AUTH else "unavailable"), True
    return "unknown", False


def _owner(conversation_store: ConversationStore, session_id: str) -> OwnerAttribution:
    owner = actual_owner_or_none(conversation_store, session_id)
    if owner is None:
        return OwnerAttribution()
    return OwnerAttribution(user=owner.user_id, registration=owner.generation)


def _harness_and_model(conversation: Conversation) -> tuple[str | None, str | None]:
    from omnigent.server.routes._sessions.helpers import _resolve_harness_impl, _resolve_llm_model

    harness = _resolve_harness_impl(conversation)
    model = conversation.model_override or _resolve_llm_model(conversation)
    return harness, model


def _subscription(
    conversation: Conversation, harness: str | None, host: Host | None
) -> SubscriptionAttribution:
    from omnigent.onboarding.provider_config import provider_family_for_harness
    from omnigent.server.routes.sandbox_inference import configured_snapshot

    provider = provider_family_for_harness(harness)
    if configured_snapshot(conversation.inference_snapshot):
        # A provider the installation configured for the session, not a person's login.
        return SubscriptionAttribution(provider=provider, source="configured_provider")
    if host is None:
        return SubscriptionAttribution(provider=provider)
    state, reported = _login_state(harness, host)
    return SubscriptionAttribution(
        subject=host.user_id,
        provider=provider,
        source="computer_login",
        login_state=state,
        login_state_reported_by="computer" if reported else None,
    )


def _guarded(label: str, read: Any, default: Any) -> Any:
    try:
        return read()
    except Exception:
        _logger.warning("turn attribution: could not read %s", label, exc_info=True)
        return default


def build_turn_attribution(
    conversation_store: ConversationStore,
    conversation: Conversation,
    initiator: TurnInitiator,
    *,
    host_store: HostStore | None = None,
) -> TurnAttribution:
    """Read the attribution of a turn starting now on *conversation*.

    Blocking (store reads); call it in a worker thread. Each section that cannot
    be read is recorded as unknown, so this never raises for a turn.
    """
    hosts = host_store if host_store is not None else _host_store
    owner = _guarded(
        "owner", lambda: _owner(conversation_store, conversation.id), OwnerAttribution()
    )
    host_id = _guarded("host", lambda: _host_id(conversation_store, conversation), None)
    host = (
        _guarded("host", lambda: hosts.get_host(host_id), None)
        if hosts is not None and host_id
        else None
    )
    harness, model = _guarded("harness", lambda: _harness_and_model(conversation), (None, None))
    subscription = _guarded(
        "subscription",
        lambda: _subscription(conversation, harness, host),
        SubscriptionAttribution(),
    )
    return TurnAttribution(
        initiator=initiator.as_attribution(),
        owner=owner,
        host=(
            HostAttribution(
                id=host.host_id,
                name=host.name,
                owner=host.user_id,
                kind=_host_kind(host),  # type: ignore[arg-type]
                runner_id=conversation.runner_id,
            )
            if host is not None
            else None
        ),
        subscription=subscription,
        harness=harness,
        model=RequestedReported(requested=model, reported=conversation.reported_model),
        effort=RequestedReported(
            requested=conversation.reasoning_effort,
            reported=conversation.reported_reasoning_effort,
        ),
        recorded_at=int(time.time()),
    )


def turn_actor_labels(person: str | None, started_at: int) -> dict[str, str]:
    """The labels that name a turn's answerable person and when it started."""
    labels = {TURN_STARTED_AT_LABEL: str(started_at)}
    if person:
        labels[TURN_ACTOR_LABEL] = person
    return labels


def record_turn_start(
    conversation_store: ConversationStore, session_id: str, person: str | None, started_at: int
) -> None:
    """Name the person answerable for the turn starting now; blocking.

    With nobody known, the previous turn's person is cleared, so approvals fall
    back to the thread's owner instead of naming someone from an earlier turn.
    """
    conversation_store.set_labels(session_id, turn_actor_labels(person, started_at))
    if not person:
        conversation_store.delete_label(session_id, TURN_ACTOR_LABEL)


def with_initiator(attribution: TurnAttribution, initiator: TurnInitiator) -> TurnAttribution:
    """The same attribution with another initiator (used when a queued sender is matched)."""
    return attribution.model_copy(update={"initiator": initiator.as_attribution()})


__all__ = [
    "TURN_ACTOR_LABEL",
    "TURN_STARTED_AT_LABEL",
    "TurnInitiator",
    "actual_owner_or_none",
    "build_turn_attribution",
    "derive_agent_initiator",
    "person_initiator",
    "record_turn_start",
    "schedule_initiator",
    "set_attribution_host_store",
    "startup_initiator",
    "turn_actor_labels",
    "with_initiator",
]
