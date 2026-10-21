"""Tests for the host's records of host-changing requests."""

from __future__ import annotations

from omnigent.host.operation_records import HostOperationRecords


class _Clock:
    """A monotonic clock the test advances by hand."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


async def test_a_repeat_gets_the_first_arrivals_outcome() -> None:
    """The first arrival is carried out; a repeat waits for and shares its result."""
    records = HostOperationRecords()
    assert records.state("req-1") is None
    assert records.claim("req-1") is None
    assert records.state("req-1") == "running"

    repeat = records.claim("req-1")
    assert repeat is not None
    assert not repeat.done()

    records.finish("req-1", '{"kind": "host.create_dir_result"}')
    assert await repeat == '{"kind": "host.create_dir_result"}'
    assert records.state("req-1") == "done"
    # Only the first answer counts.
    records.finish("req-1", '{"kind": "other"}')
    later = records.claim("req-1")
    assert later is not None
    assert await later == '{"kind": "host.create_dir_result"}'


async def test_a_request_left_unanswered_gives_its_repeat_no_answer() -> None:
    """A handler that ended without a result may have acted: a repeat neither runs nor answers."""
    records = HostOperationRecords()
    assert records.claim("req-1") is None
    records.finish("req-1", None)
    assert records.state("req-1") == "abandoned"
    repeat = records.claim("req-1")
    assert repeat is not None
    assert await repeat is None


async def test_a_finished_record_is_kept_for_the_retention_only() -> None:
    """Finished records expire after the retention; running ones never do."""
    clock = _Clock()
    records = HostOperationRecords(retention_s=60.0, clock=clock)
    assert records.claim("finished") is None
    assert records.claim("running") is None
    records.finish("finished", "result")

    clock.now += 59.0
    assert records.claim("finished") is not None
    clock.now += 2.0
    assert records.claim("running") is not None
    assert records.state("finished") is None
    assert records.state("running") == "running"
    # After the retention the request is new again.
    assert records.claim("finished") is None


async def test_received_launch_names_the_launched_runner() -> None:
    """A launch record answers for its runner whatever the launch's result."""
    records = HostOperationRecords()
    assert records.claim("launch-1", runner_id="runner_a") is None
    assert records.received_launch("runner_a")
    assert not records.received_launch("runner_b")
    records.finish("launch-1", None)
    assert records.received_launch("runner_a")


def test_each_host_process_has_its_own_incarnation() -> None:
    """Records belong to one process, which the incarnation names."""
    first, second = HostOperationRecords(), HostOperationRecords()
    assert first.incarnation and second.incarnation
    assert first.incarnation != second.incarnation
