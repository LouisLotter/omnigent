"""Results of host-changing requests, kept so that a repeated request is never run twice.

A server whose host tunnel closes before a request is answered cannot tell
whether the host carried it out. Once the host reconnects, the server sends
the same request again, under the same ``request_id``. The host records each
host-changing request when it arrives and answers a repeat with the recorded
result, waiting for one still running, instead of acting again.

Records are kept in memory, so they belong to one host process: its
incarnation, which the hello frame reports. A finished record is kept for
:data:`~omnigent.host.frames.OPERATION_RECORD_RETENTION_S`, longer than any
server waits for an answer, so within that time a request with no record was
never received by this process.
"""

from __future__ import annotations

import asyncio
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from omnigent.host.frames import OPERATION_RECORD_RETENTION_S

OperationState = Literal["running", "done", "abandoned"]


@dataclass
class _OperationRecord:
    """One received host-changing request.

    :param outcome: Resolves with the encoded result frame once the request
        is answered, or with ``None`` when its handler ended without one.
    :param runner_id: The runner a launch request was for, else ``None``.
    :param finished_at: Monotonic time the outcome was set, or ``None``.
    """

    outcome: asyncio.Future[str | None]
    runner_id: str | None = None
    finished_at: float | None = field(default=None)


class HostOperationRecords:
    """The outcomes of the host-changing requests this host process received.

    Used only on the host's event loop.

    :param retention_s: How long a finished record is kept.
    :param clock: Monotonic clock, injectable for tests.
    """

    def __init__(
        self,
        *,
        retention_s: float = OPERATION_RECORD_RETENTION_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.incarnation = secrets.token_hex(8)
        self._retention_s = retention_s
        self._clock = clock
        self._records: dict[str, _OperationRecord] = {}

    def claim(
        self, request_id: str, *, runner_id: str | None = None
    ) -> asyncio.Future[str | None] | None:
        """
        Record a request as it arrives, or find the earlier arrival it repeats.

        Call it before the handler's first await, so that a repeat arriving
        later, on this tunnel or the next, always finds the record.

        :param request_id: The request's id, which names the operation.
        :param runner_id: For a launch, the runner it starts.
        :returns: ``None`` for a first arrival, which the caller carries out.
            For a repeat, the earlier arrival's outcome, to answer with.
        """
        self._expire()
        record = self._records.get(request_id)
        if record is not None:
            return record.outcome
        loop = asyncio.get_running_loop()
        self._records[request_id] = _OperationRecord(
            outcome=loop.create_future(), runner_id=runner_id
        )
        return None

    def finish(self, request_id: str, result: str | None) -> None:
        """
        Record a request's answer before it is sent.

        :param request_id: The request's id.
        :param result: The encoded result frame, or ``None`` when the handler
            ended without an answer; a repeat then gets none either, because
            the operation may have partly taken effect.
        """
        record = self._records.get(request_id)
        if record is None or record.outcome.done():
            return
        record.outcome.set_result(result)
        record.finished_at = self._clock()

    def state(self, request_id: str) -> OperationState | None:
        """
        Return what became of a request, or ``None`` if it has no record.

        :param request_id: The request's id.
        """
        record = self._records.get(request_id)
        if record is None:
            return None
        if not record.outcome.done():
            return "running"
        return "done" if record.outcome.result() is not None else "abandoned"

    def received_launch(self, runner_id: str) -> bool:
        """
        Return whether this process received a launch of *runner_id*.

        :param runner_id: The runner, e.g. ``"runner_abc123..."``.
        """
        self._expire()
        return any(record.runner_id == runner_id for record in self._records.values())

    def _expire(self) -> None:
        """Forget records finished longer ago than the retention."""
        horizon = self._clock() - self._retention_s
        expired = [
            request_id
            for request_id, record in self._records.items()
            if record.finished_at is not None and record.finished_at < horizon
        ]
        for request_id in expired:
            del self._records[request_id]
