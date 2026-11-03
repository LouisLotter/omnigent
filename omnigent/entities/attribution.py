"""Who did what on whose behalf: the attribution recorded with each turn.

A turn records who started it, whose thread it is, whose computer ran it and
whose configured harness login it used, plus the harness, model and effort.
Every value is read from the server's own records when the turn is saved;
``None`` means unknown. Nothing here is a credential, a sign-in code, an
access assertion or prompt content, and no cost is ever recorded.
"""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError

ATTRIBUTION_VERSION = 1

InitiatorKind = Literal["person", "schedule", "agent", "terminal"]
"""``person``: a signed-in person sent it. ``schedule``: a scheduled task fired
it for its owner. ``agent``: an agent sent it (a sub-agent's prompt, a result
returned to its parent, a message between threads) for the person named in
``on_behalf_of``. ``terminal``: typed into the computer's terminal."""

InitiatorSource = Literal[
    "signed_in",
    "schedule",
    "thread_record",
    "pending_input",
    "terminal_input",
    "terminal_ambiguous",
    "owner_assumed",
]
"""Where the initiator came from. ``signed_in``: the request's own sign-in.
``schedule``: the scheduled task's record. ``thread_record``: the turn actor the
server recorded on the thread the agent works for. ``pending_input``: the
sender the server recorded when a message was queued for a terminal harness.
``terminal_input``: the one person who sent input to the thread's terminals
since the turn started. ``terminal_ambiguous``: several did; all are listed in
``possible``. ``owner_assumed``: typing the server saw nobody send (a direct
local attach), assumed to be the computer owner's."""

HostKind = Literal["personal", "cloud_computer", "sandbox"]
SubscriptionSource = Literal["computer_login", "configured_provider", "unknown"]
LoginState = Literal["ready", "needs_auth", "unavailable", "unknown"]


class _Part(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)


class InitiatorAttribution(_Part):
    """Who started a turn.

    :param kind: See :data:`InitiatorKind`.
    :param user: The person (``person``/``terminal``) or the schedule's owner.
    :param registration: That person's account registration, when known; a
        username registered again gets a new one, so the two never merge.
    :param schedule_id: The scheduled task, for ``schedule``.
    :param from_session_id: The thread whose agent sent it, for ``agent``;
        ``None`` when an agent returned a result into its own thread.
    :param on_behalf_of: The person an agent acted for, for ``agent``.
    :param possible: Everyone who may have typed it, when authorship of
        terminal typing is not certain; each counts as having taken part.
    :param source: See :data:`InitiatorSource`.
    """

    kind: InitiatorKind
    user: str | None = None
    registration: str | None = None
    schedule_id: str | None = None
    from_session_id: str | None = None
    on_behalf_of: str | None = None
    possible: list[str] | None = None
    source: InitiatorSource


class OwnerAttribution(_Part):
    """The thread's actual owner (a sub-agent's is its top-level thread's)."""

    user: str | None = None
    registration: str | None = None


class HostAttribution(_Part):
    """The computer that ran the turn and the person it belongs to."""

    id: str
    name: str | None = None
    owner: str | None = None
    kind: HostKind
    runner_id: str | None = None


class SubscriptionAttribution(_Part):
    """Whose configured harness login the turn used.

    :param subject: The person whose login it is: the computer's owner.
    :param provider: The harness's provider family, e.g. ``"anthropic"``.
    :param source: ``computer_login`` for the login configured on that
        computer; ``configured_provider`` for a provider the installation
        configured for the session; ``unknown`` without a computer.
    :param login_state: The computer's own report of its harness login.
    :param login_state_reported_by: ``"computer"`` when ``login_state`` came
        from the computer's report rather than the server.
    :param account: A non-secret provider account id. Omnigent does not verify
        provider accounts, so this is ``None``.
    :param verified: Always ``False``: the provider account is not verified.
    """

    subject: str | None = None
    provider: str | None = None
    source: SubscriptionSource = "unknown"
    login_state: LoginState = "unknown"
    login_state_reported_by: Literal["computer"] | None = None
    account: str | None = None
    verified: bool = False


class RequestedReported(_Part):
    """A setting as requested for the session and as the harness last reported it."""

    requested: str | None = None
    reported: str | None = None


class TurnAttribution(_Part):
    """The attribution of one turn, recorded with the item that starts it."""

    v: int = ATTRIBUTION_VERSION
    initiator: InitiatorAttribution
    owner: OwnerAttribution
    host: HostAttribution | None = None
    subscription: SubscriptionAttribution = SubscriptionAttribution()
    harness: str | None = None
    model: RequestedReported = RequestedReported()
    effort: RequestedReported = RequestedReported()
    recorded_at: int

    def responsible_person(self) -> str | None:
        """The person answerable for the turn: an agent's principal, else its initiator."""
        if self.initiator.kind == "agent":
            return self.initiator.on_behalf_of
        return self.initiator.user

    def to_json(self) -> str:
        """Compact JSON for storage; unknown (``None``) fields are omitted."""
        return json.dumps(
            self.model_dump(exclude_none=True), sort_keys=True, separators=(",", ":")
        )

    @classmethod
    def from_json(cls, text: str | None) -> TurnAttribution | None:
        """Parse stored JSON; unreadable or missing attribution reads as unknown."""
        if not text:
            return None
        try:
            return cls.model_validate_json(text)
        except (ValidationError, ValueError):
            return None


__all__ = [
    "ATTRIBUTION_VERSION",
    "HostAttribution",
    "InitiatorAttribution",
    "OwnerAttribution",
    "RequestedReported",
    "SubscriptionAttribution",
    "TurnAttribution",
]
