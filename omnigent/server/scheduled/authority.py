"""The refusal shared by a scheduled fire and the delivery of its saved prompt."""

from __future__ import annotations

from dataclasses import dataclass


class ScheduledFireRefused(RuntimeError):
    """A scheduled fire cannot start, or lost the authority it started with.

    :param message: Why, for the run record and the transcript.
    :param error_code: A stable code, e.g. ``"task_authority_revoked"``.
    """

    def __init__(self, message: str, *, error_code: str) -> None:
        super().__init__(message)
        self.error_code = error_code


@dataclass(frozen=True)
class ScheduledOrigin:
    """The automation whose fire saved a startup input, and its owner's registration.

    :param task_id: The scheduled task, e.g. ``"3f2c…"``.
    :param generation: The owner registration the fire ran under, or ``None``
        outside accounts mode.
    :param registration_known: ``False`` for an input saved before inputs
        carried their registration; the check then takes it from the task row
        and refuses when that row is gone.
    """

    task_id: str
    generation: str | None
    registration_known: bool = True
