"""Long-lived connections, sub-agents, files and runner binding (WP8.2b).

Accounts-mode app, real routes. Alice and Bob are members without the
administrator flag; ``admin`` is an administrator. An open socket or stream is
not lasting authority: each one re-checks its caller and closes once the
binding, share or sign-in behind it is gone. Sub-agents belong to their
thread tree's owner, file IDs to their thread, and a runner binds only to its
owner's threads.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
import uuid
from collections.abc import AsyncIterator, Callable
from types import SimpleNamespace
from typing import Any

import pytest
from asgiref.testing import ApplicationCommunicator
from fastapi.testclient import TestClient

from omnigent.errors import OmnigentError
from omnigent.host.frames import HostRunnerExitedFrame, encode_host_frame
from omnigent.runner.identity import token_bound_runner_id
from omnigent.runner.transports.ws_tunnel.frames import (
    EVENT_INGEST_CAPABILITY,
    EventAckFrame,
    EventBatchFrame,
    HelloFrame,
    encode_frame,
)
from omnigent.runtime import session_stream
from omnigent.runtime.content_resolver import _resolve_file_id_block, file_belongs_to
from omnigent.server.auth import LEVEL_EDIT, LEVEL_OWNER, LEVEL_READ
from omnigent.server.permissions import check_session_access
from omnigent.server.routes import host_tunnel, runner_tunnel, terminal_attach
from omnigent.server.routes._sessions import helpers as session_helpers
from omnigent.server.routes._sessions.helpers import MAX_SUBAGENT_DEPTH
from omnigent.server.routes.sessions import routes_core
from omnigent.stores.host_store import REGISTRATION_REVOKED_REFUSAL
from tests.server.integration.test_host_session_binding import _runner_exited_events
from tests.server.integration.test_host_tunnel_route import (
    _make_hello,
    _wait_registered,
    _websocket_scope,
)
from tests.server.test_account_revocation import create_session
from tests.server.test_account_revocation import setup_app as setup_app
from tests.server.test_runner_principal import _bind_runner, _runner, _session
from tests.server.test_runner_principal import people as people

_FAST = 0.05


# ── Runner tunnel ────────────────────────────────────────────────────────────


def _runner_tunnel(runner: Any) -> Any:
    """Open the runner's own tunnel with its runner token and binding token."""
    return runner.client.websocket_connect(f"/v1/runners/{runner.runner_id}/tunnel")


def _hello(*capabilities: str) -> str:
    return encode_frame(
        HelloFrame(
            runner_version="0.0.0-test",
            frame_protocol_version=runner_tunnel.SUPPORTED_FRAME_PROTOCOL_MAJOR,
            capabilities=list(capabilities),
        )
    )


def _close_code(ws: Any) -> int:
    """Read until the server closes the socket; return its close code."""
    while True:
        message = ws.receive()
        if message["type"] == "websocket.close":
            return int(message["code"])


def _stays_registered(app: Any, runner_id: str, seconds: float) -> bool:
    """Wait for the tunnel to register, then check it stays for *seconds*."""
    registered_by = time.monotonic() + 5
    while app.state.tunnel_registry.get(runner_id) is None:
        if time.monotonic() > registered_by:
            return False
        time.sleep(0.01)
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if app.state.tunnel_registry.get(runner_id) is None:
            return False
        time.sleep(0.02)
    return True


@pytest.mark.timeout(30)
def test_runner_tunnel_closes_once_its_binding_is_replaced(
    people: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T7: the socket opened by the old incarnation ends; it is not lasting authority."""
    monkeypatch.setattr(runner_tunnel, "RUNNER_AUTHORITY_RECHECK_S", _FAST)
    _admin, alice, _bob, stores = people
    thread = _session(alice)
    runner = _runner(stores, alice, thread, "alice")

    with _runner_tunnel(runner) as ws:
        ws.send_text(_hello())
        assert _stays_registered(alice.app, runner.runner_id, 0.3)
        stores["conversation_store"].replace_runner_id(thread, token_bound_runner_id("next"))
        assert _close_code(ws) == runner_tunnel.RUNNER_AUTHORITY_REVOKED_CLOSE_CODE
    assert alice.app.state.tunnel_registry.get(runner.runner_id) is None


@pytest.mark.timeout(30)
def test_runner_tunnel_closes_once_its_host_is_removed(
    people: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T7: removing the computer ends the runner tunnels it carried."""
    monkeypatch.setattr(runner_tunnel, "RUNNER_AUTHORITY_RECHECK_S", _FAST)
    _admin, alice, _bob, stores = people
    runner = _runner(stores, alice, _session(alice), "alice")

    with _runner_tunnel(runner) as ws:
        ws.send_text(_hello())
        assert _stays_registered(alice.app, runner.runner_id, 0.2)
        stores["host_store"].delete_host(runner.host_id)
        assert _close_code(ws) == runner_tunnel.RUNNER_AUTHORITY_REVOKED_CLOSE_CODE


@pytest.mark.timeout(30)
@pytest.mark.parametrize("rebound", [False, True], ids=["current", "replaced"])
def test_runner_tunnel_applies_no_event_after_its_binding_is_replaced(
    people: Any, monkeypatch: pytest.MonkeyPatch, rebound: bool
) -> None:
    """Each event batch re-checks the binding before it reaches the thread."""
    monkeypatch.setattr(runner_tunnel, "RUNNER_AUTHORITY_RECHECK_S", 3600.0)
    _admin, alice, _bob, stores = people
    thread = _session(alice)
    runner = _runner(stores, alice, thread, "alice")
    ingested: list[str] = []

    async def ingest(*, batch: EventBatchFrame, **_kwargs: Any) -> EventAckFrame:
        ingested.append(batch.id)
        return EventAckFrame(batch.id, len(batch.events))

    monkeypatch.setattr(alice.app.state, "runner_event_ingest", ingest, raising=False)
    batch = EventBatchFrame(id="batch-1", session_id=thread, events=[{"type": "noop"}])

    with _runner_tunnel(runner) as ws:
        ws.send_text(_hello(EVENT_INGEST_CAPABILITY))
        assert _stays_registered(alice.app, runner.runner_id, 0.1)
        if rebound:
            stores["conversation_store"].replace_runner_id(thread, token_bound_runner_id("next"))
        ws.send_text(encode_frame(batch))
        if rebound:
            assert _close_code(ws) == runner_tunnel.RUNNER_AUTHORITY_REVOKED_CLOSE_CODE
        else:
            deadline = time.monotonic() + 5
            while not ingested and time.monotonic() < deadline:
                time.sleep(0.02)
    assert ingested == ([] if rebound else ["batch-1"])


# ── Host tunnel ──────────────────────────────────────────────────────────────


async def _host_tunnel(app: Any, person: TestClient, host_id: str) -> ApplicationCommunicator:
    """Register *host_id* over the host tunnel with *person*'s sign-in."""
    scope = _websocket_scope(f"/v1/hosts/{host_id}/tunnel")
    cookie = "; ".join(f"{key}={value}" for key, value in person.cookies.items())
    scope["headers"] = [(b"cookie", cookie.encode())]
    connection = ApplicationCommunicator(app, scope)
    await connection.send_input({"type": "websocket.connect"})
    accepted = await connection.receive_output(timeout=5)
    assert accepted["type"] == "websocket.accept", accepted
    await connection.send_input(
        {"type": "websocket.receive", "text": _make_hello(name=f"host-{host_id[:8]}")}
    )
    await asyncio.wait_for(_wait_registered(app.state.host_registry, host_id), timeout=5)
    return connection


async def _until_closed(connection: ApplicationCommunicator) -> tuple[int, list[str]]:
    """Collect text frames until the server closes the host tunnel."""
    texts: list[str] = []
    while True:
        message = await connection.receive_output(timeout=5)
        if message["type"] == "websocket.close":
            return int(message["code"]), texts
        if message.get("text"):
            texts.append(message["text"])


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_host_tunnel_ends_once_its_host_is_removed(
    people: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T7: a removed computer's open tunnel ends at its next heartbeat."""
    monkeypatch.setattr(host_tunnel, "PING_INTERVAL_S", _FAST)
    monkeypatch.setattr(host_tunnel, "PING_MISS_THRESHOLD", 100_000)
    admin, alice, _bob, stores = people
    host_id = uuid.uuid4().hex
    connection = await _host_tunnel(alice.app, alice, host_id)
    try:
        # A password change carries the host to the new registration: it stays.
        reset = await asyncio.to_thread(admin.post, "/auth/users/alice/reset")
        assert reset.status_code == 200, reset.text
        await asyncio.sleep(0.3)
        assert alice.app.state.host_registry.get(host_id) is not None

        await asyncio.to_thread(stores["host_store"].delete_host, host_id)
        code, texts = await _until_closed(connection)
    finally:
        with contextlib.suppress(Exception):
            await connection.send_input({"type": "websocket.disconnect", "code": 1000})
    assert code == 4009
    assert any(REGISTRATION_REVOKED_REFUSAL in text for text in texts), texts


def test_heartbeat_holds_only_the_owners_current_registration(db_uri: str) -> None:
    """Store level: removal, another owner or a new registration end the tunnel."""
    from omnigent.server.accounts_store import SqlAlchemyAccountStore
    from omnigent.stores.host_store import HostStore

    accounts = SqlAlchemyAccountStore(db_uri)
    accounts.create_user_with_password("alice", "hash")
    hosts = HostStore(db_uri)
    host = uuid.uuid4().hex
    hosts.upsert_on_connect(host, "laptop", "alice")

    assert hosts.heartbeat(host, by_bearer=True, owner="alice")
    assert not hosts.heartbeat(host, by_bearer=True, owner="bob")
    # A password change carries hosts to the new registration.
    accounts.update_password("alice", "new-hash", revoke_sessions=True)
    assert hosts.heartbeat(host, by_bearer=True, owner="alice")
    # A row left on an earlier registration is not the owner's any more.
    with hosts._session("test_stale_host_generation") as session:
        from sqlalchemy import update

        from omnigent.db.db_models import SqlHost

        session.execute(
            update(SqlHost).where(SqlHost.host_id == host).values(account_generation="stale")
        )
    assert not hosts.heartbeat(host, by_bearer=True, owner="alice")
    hosts.upsert_on_connect(host, "laptop", "alice")
    assert hosts.heartbeat(host, by_bearer=True, owner="alice")
    hosts.delete_host(host)
    assert not hosts.heartbeat(host, by_bearer=True, owner="alice")
    # Without an owner the old contract holds: a missing host is a no-op.
    assert hosts.heartbeat(host, by_bearer=True)


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_another_members_host_cannot_report_a_runner_exit(
    people: Any, caplog: pytest.LogCaptureFixture
) -> None:
    """T3: Bob's computer can't fail or annotate Alice's thread with a crash report."""
    _admin, alice, bob, stores = people
    caplog.set_level("INFO")
    thread = (await asyncio.to_thread(create_session, alice)).json()["session_id"]
    binding = await asyncio.to_thread(_bind_runner, stores, alice.app, thread, "alice")
    runner_id = token_bound_runner_id(binding)
    alice_host = stores["conversation_store"].get_conversation(thread).host_id
    report = encode_host_frame(
        HostRunnerExitedFrame(runner_id=runner_id, error="boom", had_connected=False)
    )

    async def reported_for(thread_id: str) -> bool:
        await asyncio.sleep(0.3)
        return any(
            getattr(record, "session_id", None) == thread_id
            for record in _runner_exited_events(caplog)
        )

    bobs = await _host_tunnel(bob.app, bob, uuid.uuid4().hex)
    try:
        await bobs.send_input({"type": "websocket.receive", "text": report})
        assert not await reported_for(thread)
        assert any("runs elsewhere" in r.getMessage() for r in caplog.records)
    finally:
        await bobs.send_input({"type": "websocket.disconnect", "code": 1000})

    alices = await _host_tunnel(alice.app, alice, alice_host)
    try:
        await alices.send_input({"type": "websocket.receive", "text": report})
        assert await reported_for(thread)
    finally:
        await alices.send_input({"type": "websocket.disconnect", "code": 1000})


# ── Session stream ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_stream_stops_without_done_once_the_caller_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unit: no event after a refused re-check, and no ``[DONE]`` (a drop, not a close)."""

    async def heartbeats(*_args: Any, **_kwargs: Any) -> AsyncIterator[dict[str, Any]]:
        for _ in range(1000):
            yield {"type": "session.heartbeat"}

    class Connected:
        async def is_disconnected(self) -> bool:
            return False

    verdicts = iter([True, True, False])

    async def still_authorized() -> bool:
        return next(verdicts)

    monkeypatch.setattr(session_stream, "subscribe", heartbeats)
    frames = [
        frame
        async for frame in session_helpers._stream_live_events(
            Connected(),  # type: ignore[arg-type]
            "conv_shared",
            still_authorized=still_authorized,
            authority_recheck_s=0.0,
        )
    ]
    assert len(frames) == 2
    assert all("[DONE]" not in frame for frame in frames)

    verdicts = iter([True] * 1000)
    frames = [
        frame
        async for frame in session_helpers._stream_live_events(
            Connected(),  # type: ignore[arg-type]
            "conv_shared",
            still_authorized=still_authorized,
            authority_recheck_s=0.0,
        )
    ]
    assert len(frames) == 1001 and "[DONE]" in frames[-1]


@pytest.fixture
def stream_checks(monkeypatch: pytest.MonkeyPatch) -> dict[str, Callable[[], Any]]:
    """Capture the re-check each opened stream carries, keyed by caller."""
    from omnigent.server.routes.sessions import routes_events

    captured: dict[str, Callable[[], Any]] = {}

    async def finite_stream(
        request: Any, session_id: str, *_args: Any, still_authorized: Any = None, **_kwargs: Any
    ) -> AsyncIterator[str]:
        assert still_authorized is not None, "the stream route must re-check its caller"
        captured[request.headers.get("x-caller", "")] = still_authorized
        yield "data: [DONE]\n\n"

    monkeypatch.setattr(routes_events, "_stream_live_events", finite_stream)
    return captured


def _open_stream(client: TestClient, thread: str, caller: str) -> None:
    response = client.get(f"/v1/sessions/{thread}/stream", headers={"x-caller": caller})
    assert response.status_code == 200, response.text


def _still(checks: dict[str, Callable[[], Any]], caller: str) -> bool:
    return bool(asyncio.run(checks[caller]()))


def test_session_stream_ends_once_the_share_is_removed(people: Any, stream_checks: Any) -> None:
    """T3: Bob's open stream of Alice's thread is refused once she unshares it."""
    _admin, alice, bob, stores = people
    thread = _session(alice)
    stores["permission_store"].grant("bob", thread, LEVEL_READ)
    _open_stream(bob, thread, "bob")

    assert _still(stream_checks, "bob")
    stores["permission_store"].revoke("bob", thread)
    assert not _still(stream_checks, "bob")


def test_session_stream_ends_after_a_password_reset(people: Any, stream_checks: Any) -> None:
    """T7: a stream opened under a revoked sign-in is refused."""
    admin, _alice, bob, _stores = people
    thread = _session(bob)
    _open_stream(bob, thread, "bob")

    assert _still(stream_checks, "bob")
    assert admin.post("/auth/users/bob/reset").status_code == 200
    assert not _still(stream_checks, "bob")


def test_session_stream_ends_once_its_runner_binding_is_replaced(
    people: Any, stream_checks: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T7: a runner's stream of its own thread ends with its binding."""
    from omnigent.server.routes.sessions import routes_events

    async def connected_here(*_args: Any, **_kwargs: Any) -> None:
        return None

    # The runner's tunnel is not open in this test; the stream is served here regardless.
    monkeypatch.setattr(routes_events, "_raise_if_runner_on_another_replica", connected_here)
    monkeypatch.setattr(routes_events, "_ensure_runner_relay_ready", connected_here)
    _admin, alice, _bob, stores = people
    thread = _session(alice)
    runner = _runner(stores, alice, thread, "alice")
    _open_stream(runner.client, thread, "runner")

    assert _still(stream_checks, "runner")
    stores["conversation_store"].replace_runner_id(thread, token_bound_runner_id("next"))
    assert not _still(stream_checks, "runner")


def test_session_stream_keeps_the_owner_the_editor_and_the_admin(
    people: Any, stream_checks: Any
) -> None:
    """Same-person, D51 and D52 positives: re-checks never end an authorized stream."""
    admin, alice, bob, stores = people
    thread = _session(alice)
    stores["permission_store"].grant("bob", thread, LEVEL_EDIT)
    for caller, client in (("alice", alice), ("bob", bob), ("admin", admin)):
        _open_stream(client, thread, caller)
        assert _still(stream_checks, caller), caller


# ── Terminals ────────────────────────────────────────────────────────────────


class _IdleRunnerSocket:
    """A runner terminal that never writes and records what it is sent."""

    def __init__(self) -> None:
        self.sent: list[str | bytes] = []

    async def recv(self) -> str:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    async def send(self, data: str | bytes) -> None:
        self.sent.append(data)


@pytest.fixture
def idle_terminal(monkeypatch: pytest.MonkeyPatch) -> _IdleRunnerSocket:
    runner_socket = _IdleRunnerSocket()

    @contextlib.asynccontextmanager
    async def connect(_path: str) -> AsyncIterator[_IdleRunnerSocket]:
        yield runner_socket

    monkeypatch.setattr(terminal_attach, "get_runner_ws_factory", lambda: connect)
    monkeypatch.setattr(terminal_attach, "CONNECTION_AUTHORITY_RECHECK_S", _FAST)
    return runner_socket


def _attach(client: TestClient, thread: str, *, read_only: bool) -> Any:
    flag = "true" if read_only else "false"
    return client.websocket_connect(
        f"/v1/sessions/{thread}/resources/terminals/terminal_s1/attach?read_only={flag}"
    )


@pytest.mark.timeout(30)
@pytest.mark.parametrize("revocation", ["share_removed", "signed_out"])
def test_terminal_attach_closes_once_the_caller_loses_access(
    people: Any, idle_terminal: _IdleRunnerSocket, revocation: str
) -> None:
    """T3/T7: a viewer's open terminal closes with 1008 after unsharing or a reset."""
    admin, alice, bob, stores = people
    thread = _session(alice)
    stores["permission_store"].grant("bob", thread, LEVEL_READ)

    with _attach(bob, thread, read_only=True) as ws:
        time.sleep(0.2)
        if revocation == "share_removed":
            stores["permission_store"].revoke("bob", thread)
        else:
            assert admin.post("/auth/users/bob/reset").status_code == 200
        assert _close_code(ws) == 1008


@pytest.mark.timeout(30)
def test_owner_terminal_attach_keeps_typing_across_rechecks(
    people: Any, idle_terminal: _IdleRunnerSocket
) -> None:
    _admin, alice, _bob, _stores = people
    thread = _session(alice)

    with _attach(alice, thread, read_only=False) as ws:
        time.sleep(0.4)
        ws.send_text("ls\r")
        deadline = time.monotonic() + 5
        while not idle_terminal.sent and time.monotonic() < deadline:
            time.sleep(0.02)
    assert idle_terminal.sent == ["ls\r"]


@pytest.mark.asyncio
async def test_revoked_terminal_stops_its_pump_and_closes() -> None:
    """Unit: the pump is cancelled and the browser socket gets 1008."""
    closed: list[tuple[int, str]] = []
    pumping = asyncio.Event()
    cancelled = asyncio.Event()

    class Browser:
        async def close(self, code: int, reason: str = "") -> None:
            closed.append((code, reason))

    async def pump() -> None:
        pumping.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    checks = iter([True, True, False])

    async def still_authorized() -> bool:
        return next(checks)

    await terminal_attach._serve_while_authorized(
        Browser(),  # type: ignore[arg-type]
        pump(),
        still_authorized,
        recheck_s=0.01,
    )
    assert pumping.is_set() and cancelled.is_set()
    assert closed == [(1008, "terminal access revoked")]


# ── Session-list updates socket ──────────────────────────────────────────────


@pytest.mark.timeout(30)
def test_session_updates_socket_closes_after_a_password_reset(
    people: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(routes_core, "CONNECTION_AUTHORITY_RECHECK_S", _FAST)
    admin, _alice, bob, _stores = people
    thread = _session(bob)

    with bob.websocket_connect("/v1/sessions/updates") as ws:
        ws.send_text(json.dumps({"type": "watch", "session_ids": [thread]}))
        assert json.loads(ws.receive_text())["type"] == "snapshot"
        assert admin.post("/auth/users/bob/reset").status_code == 200
        assert _close_code(ws) == 1008


# ── Sub-agents ───────────────────────────────────────────────────────────────


def _child(client: TestClient, parent: str, *, bundle: bool) -> Any:
    """Create a sub-agent below *parent* through the JSON or the bundle path."""
    if bundle:
        from tests.server.helpers import build_agent_bundle

        return client.post(
            "/v1/sessions",
            data={"metadata": json.dumps({"parent_session_id": parent})},
            files={
                "bundle": ("agent.tar.gz", build_agent_bundle(name="child"), "application/gzip")
            },
        )
    agent_id = client.get(f"/v1/sessions/{parent}").json()["agent_id"]
    return client.post("/v1/sessions", json={"agent_id": agent_id, "parent_session_id": parent})


def _child_id(response: Any) -> str:
    assert response.status_code == 201, response.text
    body = response.json()
    return str(body.get("session_id") or body["id"])


@pytest.mark.parametrize("bundle", [False, True], ids=["json", "bundle"])
@pytest.mark.parametrize("creator", ["bob_reader", "bob_editor", "admin"])
def test_a_child_made_by_someone_else_belongs_to_the_owner_and_gets_no_runner(
    people: Any, bundle: bool, creator: str
) -> None:
    """T8 + D52: a share or the admin flag never yields the owner's offline runner."""
    admin, alice, bob, stores = people
    thread = _session(alice)
    binding = _bind_runner(stores, alice.app, thread, "alice")  # offline: no tunnel
    client = admin if creator == "admin" else bob
    if creator != "admin":
        level = LEVEL_READ if creator == "bob_reader" else LEVEL_EDIT
        stores["permission_store"].grant("bob", thread, level)

    child = _child_id(_child(client, thread, bundle=bundle))

    conversations = stores["conversation_store"]
    assert conversations.get_conversation(child).runner_id is None
    assert conversations.get_actual_session_owner(child).user_id == "alice"
    caller = "admin" if creator == "admin" else "bob"
    assert stores["permission_store"].get(caller, child) is None
    assert conversations.get_conversation(thread).runner_id == token_bound_runner_id(binding)


@pytest.mark.parametrize("bundle", [False, True], ids=["json", "bundle"])
def test_the_owners_runner_adds_a_child_on_its_own_runner(people: Any, bundle: bool) -> None:
    """Valid owner children still run: the child shares the parent's runner."""
    _admin, alice, _bob, stores = people
    thread = _session(alice)
    runner = _runner(stores, alice, thread, "alice")

    child = _child_id(_child(runner.client, thread, bundle=bundle))

    conversations = stores["conversation_store"]
    assert conversations.get_conversation(child).runner_id == runner.runner_id
    assert conversations.get_actual_session_owner(child).user_id == "alice"
    assert runner.client.get(f"/v1/sessions/{child}").status_code == 200


def test_sub_agents_nest_at_most_sixteen_levels(people: Any) -> None:
    _admin, alice, _bob, stores = people
    conversations = stores["conversation_store"]
    parent = _session(alice)
    agent_id = alice.get(f"/v1/sessions/{parent}").json()["agent_id"]
    for _ in range(MAX_SUBAGENT_DEPTH - 1):
        parent = conversations.create_conversation(
            agent_id=agent_id, parent_conversation_id=parent, kind="sub_agent"
        ).id

    def child(below: str) -> Any:
        return alice.post("/v1/sessions", json={"agent_id": agent_id, "parent_session_id": below})

    deepest = _child_id(child(parent))
    refused = child(deepest)
    assert refused.status_code == 400, refused.text
    assert "nest at most" in refused.text


def test_launch_admission_takes_a_childs_owner_from_its_root(people: Any) -> None:
    """T8: an old child owned by a colleague still runs only on the tree owner's host."""
    _admin, alice, _bob, stores = people
    thread = _session(alice)
    child = stores["conversation_store"].create_conversation(
        parent_conversation_id=thread, kind="sub_agent"
    )
    # As a reader's child was granted before: its creator owns it directly.
    stores["permission_store"].grant("bob", child.id, LEVEL_OWNER)
    authorize = alice.app.state.host_registry.launch_authorizer
    generation = {
        user: stores["account_store"].get_user(user).account_generation
        for user in ("alice", "bob")
    }

    for owner in ("bob", "alice"):
        host = uuid.uuid4().hex
        stores["host_store"].upsert_on_connect(host, f"{owner}-{host[:6]}", owner)
        stores["conversation_store"].set_host_id(child.id, host, workspace="/tmp/workspace")
        if owner == "bob":
            with pytest.raises(OmnigentError, match="someone other than the thread's owner"):
                authorize(host, child.id, owner, generation[owner], False, None)
        else:
            authorize(host, child.id, owner, generation[owner], False, None)


class _Chain:
    """Conversation store stub serving a parent chain."""

    def __init__(self, parents: dict[str, str | None]) -> None:
        self.parents = parents

    def get_conversation(self, conversation_id: str) -> Any:
        if conversation_id not in self.parents:
            return None
        return SimpleNamespace(
            id=conversation_id, parent_conversation_id=self.parents[conversation_id]
        )


class _Grants:
    def __init__(self) -> None:
        self.checked: list[str] = []

    def is_admin(self, _user_id: str) -> bool:
        return False

    def check_access(self, _user_id: str | None, conversation_id: str, _level: int) -> bool:
        self.checked.append(conversation_id)
        return True


def test_access_follows_parents_to_the_root_and_refuses_cycles_and_runaway_chains() -> None:
    grants = _Grants()
    linear = _Chain({"root": None, "a": "root", "b": "a"})
    assert check_session_access("alice", "b", LEVEL_READ, grants, linear)  # type: ignore[arg-type]
    assert grants.checked == ["root"]

    cyclic = _Chain({"a": "b", "b": "a"})
    assert not check_session_access("alice", "a", LEVEL_READ, grants, cyclic)  # type: ignore[arg-type]

    runaway = _Chain({"c0": None, **{f"c{i}": f"c{i - 1}" for i in range(1, 100)}})
    assert not check_session_access("alice", "c99", LEVEL_READ, grants, runaway)  # type: ignore[arg-type]
    assert grants.checked == ["root"]


# ── Files ────────────────────────────────────────────────────────────────────


def _upload(client: TestClient, thread: str) -> str:
    uploaded = client.post(
        f"/v1/sessions/{thread}/resources/files",
        files={"file": ("notes.txt", b"private notes", "text/plain")},
    )
    assert uploaded.status_code == 201, uploaded.text
    return str(uploaded.json()["id"])


def test_a_file_id_from_another_members_thread_is_refused(people: Any) -> None:
    """T3: Alice can't attach Bob's file to her own thread by its ID."""
    _admin, alice, bob, _stores = people
    alices, bobs = _session(alice), _session(bob)
    bobs_file = _upload(bob, bobs)
    own_file = _upload(alice, alices)

    def send(file_id: str) -> Any:
        content = [
            {"type": "input_text", "text": "summarize"},
            {"type": "input_file", "file_id": file_id},
        ]
        return alice.post(
            f"/v1/sessions/{alices}/events",
            json={"type": "message", "data": {"role": "user", "content": content}},
        )

    refused = send(bobs_file)
    assert refused.status_code == 400, refused.text
    assert "not uploaded to this session" in refused.text
    assert send(own_file).status_code != 400


def test_a_file_resolves_only_in_its_own_thread() -> None:
    from omnigent.entities.file import StoredFile

    def stored(session_id: str | None) -> StoredFile:
        return StoredFile(id="f1", created_at=0, filename="a.txt", bytes=1, session_id=session_id)

    own = uuid.uuid4().hex
    other = uuid.uuid4().hex
    assert file_belongs_to(stored(own), own)
    assert not file_belongs_to(stored(other), own)
    assert not file_belongs_to(stored(None), own)
    assert file_belongs_to(stored(None), None)
    assert not file_belongs_to(stored(own), None)

    class Files:
        def get(self, _file_id: str) -> StoredFile:
            return stored(other)

    # The block's own session claim never overrides the thread being resolved.
    with pytest.raises(ValueError, match="no longer exists"):
        _resolve_file_id_block(
            {"type": "input_file", "file_id": "f1", "session_id": other},
            Files(),  # type: ignore[arg-type]
            object(),  # type: ignore[arg-type]
            session_id=own,
        )


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_workspace_reads_never_reach_a_host_that_is_not_the_thread_owners(
    people: Any,
) -> None:
    """T3: a thread bound to Bob's computer can't read Bob's disk for Alice."""
    _admin, alice, bob, stores = people
    thread = (await asyncio.to_thread(create_session, alice)).json()["session_id"]
    bobs_host = uuid.uuid4().hex
    connection = await _host_tunnel(bob.app, bob, bobs_host)
    try:
        stores["conversation_store"].set_host_id(thread, bobs_host, workspace="/home/bob")
        # An offline runner: reads fall back to the thread's host.
        stores["conversation_store"].replace_runner_id(
            thread, token_bound_runner_id(f"binding-{uuid.uuid4().hex}")
        )
        response = await asyncio.to_thread(
            alice.get, f"/v1/sessions/{thread}/resources/environments/default/filesystem"
        )
        # The runner is offline and the only connected host is not Alice's: 503.
        assert response.status_code == 503, response.text
        await asyncio.sleep(0.2)
        sent: list[str] = []
        while not await connection.receive_nothing(timeout=0.05):
            message = await connection.receive_output(timeout=1)
            if message.get("text"):
                sent.append(message["text"])
        assert not any("host.fs_" in text for text in sent), sent
    finally:
        await connection.send_input({"type": "websocket.disconnect", "code": 1000})


# ── Runner rebinding ─────────────────────────────────────────────────────────


def test_only_the_actual_owner_rebinds_a_threads_runner(people: Any) -> None:
    """T4: the admin flag or an edit share never attaches a runner to someone's thread."""
    admin, alice, bob, stores = people
    thread = _session(alice)
    stores["permission_store"].grant("bob", thread, LEVEL_OWNER - 1)
    runner_id = token_bound_runner_id(f"binding-{uuid.uuid4().hex}")

    for client in (admin, bob):
        refused = client.patch(f"/v1/sessions/{thread}", json={"runner_id": runner_id})
        assert refused.status_code == 403, refused.text
        assert "Only the session owner" in refused.text
    passed_owner_gate = alice.patch(f"/v1/sessions/{thread}", json={"runner_id": runner_id})
    assert "Only the session owner" not in passed_owner_gate.text
