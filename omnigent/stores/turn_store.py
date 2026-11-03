"""Who took part in a turn: every accepted input and every registered turn.

Approvals decide who may answer from these records, so they are durable and
complete: an input is recorded, with the person answerable for it, before it
reaches a runner; a runner registers each turn, naming the inputs it starts
with, and waits for the server's acknowledgement before its harness acts; and
typing into a terminal is recorded per terminal and person. Every server
replica reads the same rows, and nothing is scanned with a cap.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import time
from collections.abc import Callable, Iterable, Sequence
from enum import IntEnum

from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from omnigent.db.db_models import SqlSessionInput, SqlSessionTurn, current_workspace_id
from omnigent.db.utils import (
    get_or_create_engine,
    make_named_managed_session_maker,
    run_write_transaction,
)


class InputKind(IntEnum):
    """What sent an input."""

    PERSON = 1
    SCHEDULE = 2
    AGENT = 3
    TERMINAL = 4
    TERMINAL_ACTIVITY = 5


INPUT_KINDS: dict[str, InputKind] = {
    "person": InputKind.PERSON,
    "schedule": InputKind.SCHEDULE,
    "agent": InputKind.AGENT,
    "terminal": InputKind.TERMINAL,
}


@dataclasses.dataclass(frozen=True)
class InputRecord:
    """One accepted input.

    :param person: The person answerable for it, or ``None`` when uncertain.
    :param possible: Everyone who may have sent it when authorship is ambiguous.
    """

    input_id: str
    kind: InputKind
    person: str | None
    possible: tuple[str, ...]
    item_id: str | None
    accepted_at: int

    @property
    def people(self) -> frozenset[str]:
        """Everyone this input may be answerable to."""
        return frozenset({*self.possible, *([self.person] if self.person else [])})


@dataclasses.dataclass(frozen=True)
class TurnRecord:
    """One registered turn."""

    turn_id: str
    runner_id: str | None
    initiator: str | None
    inputs: tuple[str, ...]
    unresolved: bool
    started_at: int


@dataclasses.dataclass(frozen=True)
class TurnAuthority:
    """Who took part in a thread's current turn, read completely.

    :param initiator: The registered turn's initiator, or ``None`` when no turn
        is registered or its input names no person.
    :param people: Everyone who may have taken part: the turn's inputs and
        every input accepted (or terminal typing) since it started; with no
        registered turn, every input the thread ever accepted.
    :param turn_id: The current turn, or ``None``.
    :param unresolved: The turn named an input the server never recorded.
    """

    initiator: str | None
    people: frozenset[str]
    turn_id: str | None
    unresolved: bool


def now_ms() -> int:
    return int(time.time() * 1000)


def text_digest(text: str) -> str:
    """Digest of an input's text, whitespace-normalized, to match its transcript record."""
    return hashlib.sha256(" ".join(text.split()).encode("utf-8")).hexdigest()


def terminal_input_id(terminal_id: str, user_id: str) -> str:
    return f"terminal:{terminal_id}:{user_id}"[:160]


def _possible(text: str | None) -> tuple[str, ...]:
    if not text:
        return ()
    value = json.loads(text)
    return (
        tuple(person for person in value if isinstance(person, str))
        if isinstance(value, list)
        else ()
    )


def _to_input(row: SqlSessionInput) -> InputRecord:
    return InputRecord(
        input_id=row.input_id,
        kind=InputKind(row.kind),
        person=row.person,
        possible=_possible(row.possible),
        item_id=row.item_id,
        accepted_at=row.accepted_at,
    )


def _to_turn(row: SqlSessionTurn) -> TurnRecord:
    inputs = json.loads(row.inputs)
    return TurnRecord(
        turn_id=row.turn_id,
        runner_id=row.runner_id,
        initiator=row.initiator,
        inputs=tuple(value for value in inputs if isinstance(value, str)),
        unresolved=bool(row.unresolved),
        started_at=row.started_at,
    )


def delete_session_records(session: Session, session_ids: Sequence[str]) -> None:
    """Delete the inputs and turns of threads being deleted, inside their deletion."""
    if not session_ids:
        return
    for model in (SqlSessionInput, SqlSessionTurn):
        session.execute(
            delete(model).where(
                model.workspace_id == current_workspace_id(),
                model.session_id.in_(list(session_ids)),
            )
        )


InputResolver = Callable[[str], tuple[bool, str | None]]
"""For an input id with no recorded input: ``(found, person)`` from other server records."""


class SqlAlchemyTurnStore:
    """SQLAlchemy persistence for accepted inputs and registered turns."""

    def __init__(self, storage_location: str) -> None:
        self.storage_location = storage_location
        self._engine = get_or_create_engine(storage_location)
        self._session = make_named_managed_session_maker(
            self._engine, query_name_prefix="omnigent.turn_store"
        )
        self._session_immediate = make_named_managed_session_maker(
            self._engine, query_name_prefix="omnigent.turn_store", immediate=True
        )

    # ── Inputs ─────────────────────────────────────────────────────

    def record_input(
        self,
        session_id: str,
        input_id: str,
        *,
        kind: InputKind,
        person: str | None,
        possible: Iterable[str] = (),
        digest: str | None = None,
        item_id: str | None = None,
        accepted_at: int | None = None,
    ) -> InputRecord:
        """Record an accepted input once; a repeat (a retried send) returns the first record."""
        at = now_ms() if accepted_at is None else accepted_at
        people = sorted({value for value in possible if value})

        def write(session: Session) -> InputRecord:
            existing = session.get(SqlSessionInput, (current_workspace_id(), session_id, input_id))
            if existing is not None:
                return _to_input(existing)
            row = SqlSessionInput(
                session_id=session_id,
                input_id=input_id,
                kind=int(kind),
                person=person,
                possible=json.dumps(people) if people else None,
                item_id=item_id,
                text_digest=digest,
                accepted_at=at,
            )
            session.add(row)
            session.flush()
            return _to_input(row)

        try:
            return run_write_transaction(self._session_immediate, "insert_session_input", write)
        except IntegrityError:
            # A concurrent replica recorded the same input first.
            recorded = self.get_input(session_id, input_id)
            if recorded is None:
                raise
            return recorded

    def get_input(self, session_id: str, input_id: str) -> InputRecord | None:
        with self._session("select_session_input") as session:
            row = session.get(SqlSessionInput, (current_workspace_id(), session_id, input_id))
            return _to_input(row) if row is not None else None

    def map_input(self, session_id: str, input_id: str, item_id: str) -> InputRecord | None:
        """Note the transcript item an input became; returns the input, if recorded."""

        def write(session: Session) -> InputRecord | None:
            row = session.get(SqlSessionInput, (current_workspace_id(), session_id, input_id))
            if row is None:
                return None
            if row.item_id is None:
                row.item_id = item_id
            return _to_input(row)

        return run_write_transaction(self._session_immediate, "map_session_input", write)

    def find_unmapped_input(self, session_id: str, digest: str) -> InputRecord | None:
        """The oldest queued input with this text not yet matched to a transcript item.

        Used when a queued message's in-memory entry is gone: after a restart,
        or when the transcript record reaches another server replica.
        """
        with self._session("select_unmapped_session_input") as session:
            row = session.execute(
                select(SqlSessionInput)
                .where(
                    SqlSessionInput.workspace_id == current_workspace_id(),
                    SqlSessionInput.session_id == session_id,
                    SqlSessionInput.text_digest == digest,
                    SqlSessionInput.item_id.is_(None),
                )
                .order_by(SqlSessionInput.accepted_at, SqlSessionInput.input_id)
                .limit(1)
            ).scalar_one_or_none()
            return _to_input(row) if row is not None else None

    def record_terminal_activity(
        self, session_id: str, terminal_id: str, user_id: str, at: int | None = None
    ) -> None:
        """Record that *user_id* sent input to a terminal of the thread, now."""
        moment = now_ms() if at is None else at
        input_id = terminal_input_id(terminal_id, user_id)

        def write(session: Session) -> None:
            row = session.get(SqlSessionInput, (current_workspace_id(), session_id, input_id))
            if row is None:
                session.add(
                    SqlSessionInput(
                        session_id=session_id,
                        input_id=input_id,
                        kind=int(InputKind.TERMINAL_ACTIVITY),
                        person=user_id,
                        accepted_at=moment,
                    )
                )
            elif row.accepted_at < moment:
                row.accepted_at = moment

        try:
            run_write_transaction(self._session_immediate, "upsert_terminal_activity", write)
        except IntegrityError:
            # Another replica inserted it first; the next frame updates the time.
            run_write_transaction(self._session_immediate, "upsert_terminal_activity", write)

    def typists_for_prompt(self, session_id: str) -> list[str]:
        """Who may have typed a prompt just recorded in the thread's terminal.

        Everyone who sent terminal input since the later of the current turn's
        start and the previous typed prompt.
        """
        with self._session("select_prompt_typists") as session:
            turn_start = session.execute(
                select(func.max(SqlSessionTurn.started_at)).where(
                    SqlSessionTurn.workspace_id == current_workspace_id(),
                    SqlSessionTurn.session_id == session_id,
                )
            ).scalar_one_or_none()
            previous_prompt = session.execute(
                select(func.max(SqlSessionInput.accepted_at)).where(
                    SqlSessionInput.workspace_id == current_workspace_id(),
                    SqlSessionInput.session_id == session_id,
                    SqlSessionInput.kind == int(InputKind.TERMINAL),
                )
            ).scalar_one_or_none()
        bounds = [value for value in (turn_start, previous_prompt) if value is not None]
        return self.typists_since(session_id, max(bounds) if bounds else None)

    def typists_since(self, session_id: str, since: int | None) -> list[str]:
        """Everyone who typed into one of the thread's terminals since *since* (ms)."""
        query = select(SqlSessionInput.person).where(
            SqlSessionInput.workspace_id == current_workspace_id(),
            SqlSessionInput.session_id == session_id,
            SqlSessionInput.kind == int(InputKind.TERMINAL_ACTIVITY),
        )
        if since is not None:
            query = query.where(SqlSessionInput.accepted_at >= since)
        with self._session("select_terminal_typists") as session:
            return sorted({person for person in session.scalars(query) if person})

    # ── Turns ──────────────────────────────────────────────────────

    def latest_turn(self, session_id: str) -> TurnRecord | None:
        with self._session("select_latest_session_turn") as session:
            row = session.execute(
                select(SqlSessionTurn)
                .where(
                    SqlSessionTurn.workspace_id == current_workspace_id(),
                    SqlSessionTurn.session_id == session_id,
                )
                .order_by(SqlSessionTurn.started_at.desc(), SqlSessionTurn.turn_id.desc())
                .limit(1)
            ).scalar_one_or_none()
            return _to_turn(row) if row is not None else None

    def register_turn(
        self,
        session_id: str,
        turn_id: str,
        *,
        runner_id: str | None,
        input_ids: Sequence[str],
        resolve: InputResolver | None = None,
    ) -> TurnRecord:
        """Register a turn starting with *input_ids*; a retry returns the stored turn.

        Each input is looked up in the recorded inputs, then through *resolve*.
        An input found nowhere marks the turn unresolved.
        """
        ids = list(dict.fromkeys(value for value in input_ids if value))
        existing = self._get_turn(session_id, turn_id)
        if existing is not None:
            return existing
        recorded = {value: self.get_input(session_id, value) for value in ids}
        people: list[str] = []
        unresolved = False
        for value in ids:
            record = recorded[value]
            if record is not None:
                person = record.person or next(iter(sorted(record.possible)), None)
            elif resolve is not None and (found := resolve(value))[0]:
                person = found[1]
            else:
                unresolved = True
                continue
            if person:
                people.append(person)
        started_at = now_ms()

        def write(session: Session) -> TurnRecord:
            current = session.get(SqlSessionTurn, (current_workspace_id(), session_id, turn_id))
            if current is not None:
                return _to_turn(current)
            row = SqlSessionTurn(
                session_id=session_id,
                turn_id=turn_id,
                runner_id=runner_id,
                initiator=people[0] if people else None,
                inputs=json.dumps(ids),
                unresolved=unresolved,
                started_at=started_at,
            )
            session.add(row)
            session.flush()
            return _to_turn(row)

        try:
            return run_write_transaction(self._session_immediate, "insert_session_turn", write)
        except IntegrityError:
            stored = self._get_turn(session_id, turn_id)
            if stored is None:
                raise
            return stored

    def _get_turn(self, session_id: str, turn_id: str) -> TurnRecord | None:
        with self._session("select_session_turn") as session:
            row = session.get(SqlSessionTurn, (current_workspace_id(), session_id, turn_id))
            return _to_turn(row) if row is not None else None

    def turn_authority(self, session_id: str) -> TurnAuthority:
        """Everyone who may have taken part in the thread's current turn.

        Reads every row it needs, with no cap; a failed read raises, so callers
        never mistake an unreadable record for "nobody else took part".
        """
        with self._session("select_turn_authority") as session:
            turn_row = session.execute(
                select(SqlSessionTurn)
                .where(
                    SqlSessionTurn.workspace_id == current_workspace_id(),
                    SqlSessionTurn.session_id == session_id,
                )
                .order_by(SqlSessionTurn.started_at.desc(), SqlSessionTurn.turn_id.desc())
                .limit(1)
            ).scalar_one_or_none()
            turn = _to_turn(turn_row) if turn_row is not None else None
            query = select(SqlSessionInput).where(
                SqlSessionInput.workspace_id == current_workspace_id(),
                SqlSessionInput.session_id == session_id,
            )
            if turn is not None:
                since_or_own = SqlSessionInput.accepted_at >= turn.started_at
                if turn.inputs:
                    since_or_own = since_or_own | SqlSessionInput.input_id.in_(list(turn.inputs))
                query = query.where(since_or_own)
            people: set[str] = set()
            for row in session.scalars(query):
                people |= _to_input(row).people
        if turn is not None and turn.initiator:
            people.add(turn.initiator)
        return TurnAuthority(
            initiator=turn.initiator if turn is not None else None,
            people=frozenset(people),
            turn_id=turn.turn_id if turn is not None else None,
            unresolved=turn.unresolved if turn is not None else False,
        )


__all__ = [
    "INPUT_KINDS",
    "InputKind",
    "InputRecord",
    "InputResolver",
    "SqlAlchemyTurnStore",
    "TurnAuthority",
    "TurnRecord",
    "delete_session_records",
    "now_ms",
    "terminal_input_id",
    "text_digest",
]
