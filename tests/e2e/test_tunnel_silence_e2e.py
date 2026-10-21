"""E2E: host and runner tunnels across half-open paths and capture pauses.

A real server, host daemon and runner, with the host and runner dialing the
server through the resilience lab's FaultProxy. Its ``ping_only`` fault leaves a
tunnel half-open with false liveness: the proxy answers the client's protocol
pings and forwards nothing. A capture pause is a SIGSTOP of the host's and the
runner's process trees for 40 s, the hard budget a sandbox checkpoint may take.

Everything runs at production timing: the server's 30 s application ping and
the 45 s silence deadline. Each test takes one to two minutes.

Run::

    uv run --no-sync pytest tests/e2e/test_tunnel_silence_e2e.py -v -n 4
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import httpx
import psutil
import pytest

from omnigent.server.routes._sessions.common import RUNNER_DISCONNECT_GRACE_ENV_VAR
from omnigent.util.tunnel_silence import TUNNEL_SILENCE_GRACE_S, TUNNEL_SILENCE_TIMEOUT_S
from tests._helpers.host import Host, isolated_host
from tests._helpers.server_runner import ServerRunner, server_runner
from tests.e2e.resilience.lab.events import EventLog
from tests.e2e.resilience.lab.processes import frozen, process_tree
from tests.e2e.resilience.lab.proxy import ConnectionInfo, FaultProxy, LoopThread, host_link

pytestmark = pytest.mark.timeout(300)

_PAUSE_S = 40.0
_SILENT_FOR_S = TUNNEL_SILENCE_TIMEOUT_S + TUNNEL_SILENCE_GRACE_S
_RUNNER_GRACE_S = 150.0
_TUNNELS = ("host.tunnel", "runner.tunnel")
# Time for an aborted client to redial and register again.
_RECONNECT_S = 15.0
_POLL_S = 0.1


@dataclass
class _Topology:
    stack: ServerRunner
    host: Host
    proxy: FaultProxy
    events: EventLog
    client: httpx.Client

    def tunnel(self, tag: str) -> ConnectionInfo:
        [conn] = self.proxy.connections({tag})
        return conn

    def processes(self) -> list[psutil.Process]:
        assert self.stack.runner is not None
        return process_tree([self.host.proc.pid, self.stack.runner.pid])

    def wait_for_server_messages(
        self,
        tags: tuple[str, ...] = _TUNNELS,
        *,
        older_than_s: float = 0.0,
        newer_than_s: float | None = None,
    ) -> dict[str, ConnectionInfo]:
        """Wait until each tunnel's last server message is within the given ages."""
        deadline = time.monotonic() + 2 * TUNNEL_SILENCE_TIMEOUT_S
        ages: list[float | None] = []
        while time.monotonic() < deadline:
            tunnels = {tag: self.tunnel(tag) for tag in tags}
            ages = [conn.last_s2c_message_age_s for conn in tunnels.values()]
            if all(
                age is not None
                and age >= older_than_s
                and (newer_than_s is None or age <= newer_than_s)
                for age in ages
            ):
                return tunnels
            time.sleep(_POLL_S)
        raise AssertionError(f"tunnel message ages never fit: {ages}")

    def wait_gave_up(self, conn: ConnectionInfo, *, timeout: float) -> float:
        """Return when the client abandoned a caught connection (monotonic time)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for event in self.events.events(source="proxy:host", kind="disconnect"):
                if event.fields["conn"] == conn.id:
                    assert event.fields["reason"] == "client_gave_up", event.fields
                    return event.mono
            time.sleep(_POLL_S)
        raise AssertionError(f"{conn.tag} connection {conn.id} was never abandoned")

    def wait_reconnected(self, tag: str, *, after_wall: float) -> float:
        """Wait for a new *tag* tunnel to register; return when it opened (monotonic time)."""
        conn = self.proxy.wait_for_connection({tag}, opened_after=after_wall, timeout=_RECONNECT_S)
        if tag == "host.tunnel":
            self.host.wait_online(self.client, timeout=_RECONNECT_S, poll_interval=_POLL_S)
        else:
            self.wait_runner_online()
        return time.monotonic() - conn.age_s

    def wait_runner_online(self) -> None:
        deadline = time.monotonic() + _RECONNECT_S
        while time.monotonic() < deadline:
            status = self.client.get(f"/v1/runners/{self.stack.runner_id}/status")
            if status.status_code == 200 and status.json().get("online") is True:
                return
            time.sleep(_POLL_S)
        raise AssertionError(f"runner {self.stack.runner_id} is not online again")

    def server_log(self) -> str:
        paths = [
            self.stack.log_path("server"),
            *(self.stack.server_home / ".omnigent" / "logs").rglob("*.log"),
        ]
        return "\n".join(path.read_text(errors="replace") for path in paths if path.exists())


@pytest.fixture
def topology(tmp_path: Path) -> Iterator[_Topology]:
    events = EventLog(tmp_path / "events.jsonl")
    with ExitStack() as resources:
        stack = resources.enter_context(
            server_runner(
                tmp_path / "stack",
                server_env={RUNNER_DISCONNECT_GRACE_ENV_VAR: str(_RUNNER_GRACE_S)},
            )
        )
        loop = LoopThread("tunnel-silence-e2e")
        loop.start()
        resources.callback(loop.stop)
        proxy = FaultProxy(
            "host", ("127.0.0.1", stack.port), loop=loop, events=events, classify=host_link
        )
        proxy.start()
        resources.callback(proxy.stop)
        stack.start_runner(env={"RUNNER_SERVER_URL": proxy.url})
        client = resources.enter_context(
            httpx.Client(base_url=stack.base_url, timeout=10, trust_env=False)
        )
        host = resources.enter_context(isolated_host(tmp_path / "host", proxy.url))
        host.wait_online(client, timeout=60, poll_interval=_POLL_S)
        yield _Topology(stack, host, proxy, events, client)


def test_half_open_tunnels_are_redialed_after_the_deadline(topology: _Topology) -> None:
    topology.wait_for_server_messages()
    caught_wall = time.time()
    assert topology.proxy.ping_only(set(_TUNNELS)) == 2
    caught = {tag: topology.tunnel(tag) for tag in _TUNNELS}
    snapshot_at = time.monotonic()
    for tag, conn in caught.items():
        assert conn.ping_only == "abort"
        assert conn.last_s2c_message_age_s is not None
        last_message_at = snapshot_at - conn.last_s2c_message_age_s
        gave_up_at = topology.wait_gave_up(conn, timeout=_SILENT_FOR_S + 5)
        # Measured from the last message the client received, as the deadline is.
        silent_s = gave_up_at - last_message_at
        topology.events.emit("test", "detected", tag=tag, silent_s=silent_s)
        assert _SILENT_FOR_S <= silent_s < _SILENT_FOR_S + 3, tag
        topology.wait_reconnected(tag, after_wall=caught_wall)


def test_a_40s_capture_pause_trips_no_deadline(topology: _Topology) -> None:
    # Paused at least 10 s after the server's last message, each tunnel resumes
    # past its deadline with the server's pings queued, so the grace read decides.
    before = topology.wait_for_server_messages(older_than_s=10)
    paused_at = time.monotonic()
    with frozen(topology.processes(), topology.events):
        time.sleep(_PAUSE_S)
    for conn in before.values():
        assert conn.last_s2c_message_age_s is not None
        silent_at_resume_s = conn.last_s2c_message_age_s + time.monotonic() - paused_at
        topology.events.emit("test", "resumed", tag=conn.tag, silent_s=silent_at_resume_s)
        assert silent_at_resume_s > _SILENT_FOR_S, conn.tag

    # A false detection would show within one deadline of the resume.
    time.sleep(_SILENT_FOR_S + 4)
    for tag, conn in before.items():
        after = topology.tunnel(tag)
        assert after.id == conn.id, f"{tag} reconnected after the pause"
        assert after.last_s2c_message_age_s is not None
        assert after.last_s2c_message_age_s < TUNNEL_SILENCE_TIMEOUT_S
    assert not [
        event
        for event in topology.events.events(source="proxy:host", kind="disconnect")
        if event.fields["tag"] in _TUNNELS
    ]
    topology.host.wait_online(topology.client, timeout=5, poll_interval=_POLL_S)
    topology.wait_runner_online()


def test_a_loss_during_a_pause_is_detected_right_after_resume(topology: _Topology) -> None:
    before = topology.wait_for_server_messages(older_than_s=10)
    with frozen(topology.processes(), topology.events):
        lost_wall = time.time()
        assert topology.proxy.ping_only(set(_TUNNELS)) == 2
        time.sleep(_PAUSE_S)
    resumed_at = time.monotonic()
    for tag, conn in before.items():
        gave_up_at = topology.wait_gave_up(conn, timeout=_RECONNECT_S)
        # The deadline expired during the pause, so only the grace read is left.
        topology.events.emit("test", "detected", tag=tag, after_resume_s=gave_up_at - resumed_at)
        assert gave_up_at - resumed_at < TUNNEL_SILENCE_GRACE_S + 3, tag
        topology.wait_reconnected(tag, after_wall=lost_wall)


def test_a_drop_overlapping_a_pause_is_recovered_within_the_runner_grace(
    topology: _Topology,
) -> None:
    """The longest outage a pause can add: the drop, then the pause at the deadline."""
    [runner] = topology.wait_for_server_messages(("runner.tunnel",), newer_than_s=1).values()
    dropped_at, dropped_wall = time.monotonic(), time.time()
    assert topology.proxy.ping_only({"runner.tunnel"}) == 1
    time.sleep(TUNNEL_SILENCE_TIMEOUT_S - 1)
    with frozen(topology.processes(), topology.events):
        time.sleep(_PAUSE_S)
    topology.wait_gave_up(runner, timeout=_RECONNECT_S)
    recovered_at = topology.wait_reconnected("runner.tunnel", after_wall=dropped_wall)

    outage_s = recovered_at - dropped_at
    topology.events.emit("test", "recovered", tag="runner.tunnel", outage_s=outage_s)
    # Longer than upstream's 90 s grace leaves comfortable room for; inside 150 s.
    assert TUNNEL_SILENCE_TIMEOUT_S - 1 + _PAUSE_S < outage_s < _RUNNER_GRACE_S
    log = topology.server_log()
    runner_id = topology.stack.runner_id
    assert f"Runner {runner_id} reconnected within the liveness lease" in log
    assert f"Runner {runner_id} disconnected; reconciling" not in log
