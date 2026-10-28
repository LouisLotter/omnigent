"""Persistent managed hosts through the real server app.

The production ``create_app`` is wired with a persistent provider backed by an
in-memory computer (see ``tests/server/persistent_host_fakes.py``); its host
process registers on the app's own registry with the launch token the store
accepts. The tests drive the public routes: creating, starting and deleting
computers, listing their states, session deletion, one-session launches, and a
host-scoped request waking an asleep computer.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from omnigent.runtime.agent_cache import AgentCache
from omnigent.server import persistent_hosts
from omnigent.server.app import create_app
from omnigent.server.feature_flags import resolve_feature_flags
from omnigent.server.managed_hosts import (
    ManagedSandboxConfig,
    ManagedSandboxDeployment,
    PersistentHostPolicy,
)
from omnigent.stores.agent_store.sqlalchemy_store import SqlAlchemyAgentStore
from omnigent.stores.artifact_store.local import LocalArtifactStore
from omnigent.stores.comment_store.sqlalchemy_store import SqlAlchemyCommentStore
from omnigent.stores.conversation_store.sqlalchemy_store import SqlAlchemyConversationStore
from omnigent.stores.file_store.sqlalchemy_store import SqlAlchemyFileStore
from omnigent.stores.host_store import HostStore
from tests.server.helpers import create_test_agent
from tests.server.persistent_host_fakes import (
    FIRST_TOKEN,
    HOST_ID,
    SANDBOX_ID,
    FakeComputer,
    FakeHost,
    call_soon,
    register_persistent_row,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.timeout(60)]


@dataclass
class _Env:
    app: FastAPI
    client: AsyncClient
    host: FakeHost
    conversations: SqlAlchemyConversationStore


@pytest.fixture
async def env(runtime_init: None, db_uri: str, tmp_path: Path) -> AsyncIterator[_Env]:
    """The full app with one persistent provider and no computer yet."""
    computer = FakeComputer(status="stopped")
    artifact_store = LocalArtifactStore(str(tmp_path / "artifacts"))
    host_store = HostStore(db_uri)
    conversations = SqlAlchemyConversationStore(db_uri)
    app = create_app(
        feature_flags=resolve_feature_flags({"OMNIGENT_FEATURES": "harness_install"}),
        agent_store=SqlAlchemyAgentStore(db_uri),
        file_store=SqlAlchemyFileStore(db_uri),
        conversation_store=conversations,
        artifact_store=artifact_store,
        agent_cache=AgentCache(artifact_store=artifact_store, cache_dir=tmp_path / "cache"),
        comment_store=SqlAlchemyCommentStore(db_uri),
        host_store=host_store,
        sandbox_config=ManagedSandboxDeployment.single(
            ManagedSandboxConfig(
                server_url="https://omnigent.test",
                launcher_factory=lambda: computer,
                token_ttl_s=3600,
                provider=computer.provider,
                persistent=PersistentHostPolicy(
                    manage_url="https://computers.test/c/{sandbox_id}"
                ),
                online_timeout_s=5,
            )
        ),
    )
    fake = FakeHost(host_store, app.state.host_registry, computer)
    loop = asyncio.get_running_loop()
    computer.on_host_start = lambda token: call_soon(loop, fake.connect, token)
    persistent_hosts._states.clear()
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            yield _Env(app=app, client=client, host=fake, conversations=conversations)
    finally:
        fake.close()
        for state in persistent_hosts._states.values():
            if state.wake_task is not None:
                state.wake_task.cancel()
        persistent_hosts.configure(None, None, None)
        persistent_hosts._states.clear()


async def _settled(host_id: str) -> None:
    state = persistent_hosts._states.get(host_id)
    if state is not None and state.wake_task is not None:
        await asyncio.wait({state.wake_task}, timeout=10)


async def _listed(client: AsyncClient, host_id: str) -> dict[str, object]:
    hosts = (await client.get("/v1/hosts")).json()["hosts"]
    return next(host for host in hosts if host["host_id"] == host_id)


async def test_add_a_computer_then_list_it_awake_with_its_setup_link(env: _Env) -> None:
    """``POST /v1/hosts`` writes the pending host and starts it in the background."""
    response = await env.client.post("/v1/hosts", json={"name": "Work computer"})
    assert response.status_code == 201, response.text
    created = response.json()
    assert created["name"] == "Work computer"
    assert created["lifecycle"] == "persistent"
    assert created["persistent_status"]["state"] == "waking"
    env.host.host_id = created["host_id"]
    await _settled(created["host_id"])

    listed = await _listed(env.client, created["host_id"])
    status = listed["persistent_status"]
    assert isinstance(status, dict)
    assert status["state"] == "awake"
    assert status["manage_url"] == f"https://computers.test/c/{SANDBOX_ID}"
    assert status["provider"] == "fakepersist"
    assert env.host.computer.events == ["provision", "resume", "claim", "start_host"]


@pytest.mark.parametrize(
    ("body", "status"),
    [
        ({"name": ""}, 422),
        ({"name": "x" * 65}, 422),
        ({"name": "bad\nname"}, 422),
        ({"name": "Work computer", "provider": "modal"}, 400),
    ],
)
async def test_add_a_computer_rejects_bad_input(
    env: _Env, body: dict[str, str], status: int
) -> None:
    """Names are bounded printable text and the provider must be a persistent one."""
    response = await env.client.post("/v1/hosts", json=body)
    assert response.status_code == status, response.text
    assert env.host.computer.events == []


async def test_add_a_computer_with_a_taken_name_is_a_conflict(env: _Env) -> None:
    """A second computer with the same name gets a clean 409 and provisions nothing."""
    first = await env.client.post("/v1/hosts", json={"name": "Work computer"})
    env.host.host_id = first.json()["host_id"]
    await _settled(env.host.host_id)
    env.host.computer.events.clear()
    second = await env.client.post("/v1/hosts", json={"name": "Work computer"})
    assert second.status_code == 409, second.text
    assert env.host.computer.events == []


async def test_failed_first_start_is_listed_as_failed_and_starts_again(env: _Env) -> None:
    """A failed first start shows "failed" with the reason; ``/start`` retries it."""
    env.host.computer.provision_error = "all computers are busy"
    created = (await env.client.post("/v1/hosts", json={"name": "Work computer"})).json()
    env.host.host_id = created["host_id"]
    await _settled(created["host_id"])
    failed = (await _listed(env.client, created["host_id"]))["persistent_status"]
    assert isinstance(failed, dict)
    assert (failed["state"], failed["failure"]) == ("failed", "all computers are busy")

    env.host.computer.provision_error = None
    response = await env.client.post(f"/v1/hosts/{created['host_id']}/start")
    assert response.status_code == 202, response.text
    await _settled(created["host_id"])
    recovered = (await _listed(env.client, created["host_id"]))["persistent_status"]
    assert isinstance(recovered, dict) and recovered["state"] == "awake"
    assert "terminate" not in env.host.computer.events


async def test_delete_a_computer_removes_it_through_the_provider(env: _Env) -> None:
    """``DELETE /v1/hosts/{id}`` hides the host and asks the provider to remove it once."""
    register_persistent_row(env.app.state.host_store)
    response = await env.client.delete(f"/v1/hosts/{HOST_ID}")
    assert response.status_code == 200, response.text
    assert response.json() == {"host_id": HOST_ID, "deleted": True}
    assert env.host.computer.events == ["terminate"]
    hosts = (await env.client.get("/v1/hosts")).json()["hosts"]
    assert HOST_ID not in {host["host_id"] for host in hosts}
    assert (await env.client.delete(f"/v1/hosts/{HOST_ID}")).status_code == 404


async def test_delete_refuses_hosts_that_are_not_persistent(env: _Env) -> None:
    """Laptops are never deleted through the computers route."""
    env.app.state.host_store.upsert_on_connect(host_id=HOST_ID, name="laptop", user_id="local")
    response = await env.client.delete(f"/v1/hosts/{HOST_ID}")
    assert response.status_code == 400, response.text
    assert env.app.state.host_store.get_host(HOST_ID) is not None


async def test_deleting_a_conversation_keeps_its_computer(env: _Env) -> None:
    """A computer outlives its conversations: no tombstone, no provider call."""
    register_persistent_row(env.app.state.host_store)
    agent = await create_test_agent(env.client, name="persistent-delete-agent")
    conversation = env.conversations.create_conversation(
        agent_id=agent["id"], host_id=HOST_ID, workspace="/root/workspace"
    )
    response = await env.client.delete(f"/v1/sessions/{conversation.id}")
    assert response.status_code == 200, response.text
    assert env.app.state.host_store.get_host(HOST_ID) is not None
    assert env.host.computer.events == []


async def test_new_sandbox_session_on_a_persistent_provider_is_refused(env: _Env) -> None:
    """A one-conversation launch never provisions a computer; the POST fails synchronously."""
    agent = await create_test_agent(env.client, name="persistent-managed-agent")
    response = await env.client.post(
        "/v1/sessions", json={"agent_id": agent["id"], "host_type": "managed"}
    )
    assert response.status_code == 400, response.text
    assert "persistent hosts only" in response.text
    assert env.host.computer.events == []


async def test_host_request_wakes_an_asleep_computer(env: _Env) -> None:
    """Browsing an asleep computer's files wakes it, then reaches its host."""
    register_persistent_row(env.app.state.host_store)
    env.host.computer.status = "stopped"
    response = await env.client.get(f"/v1/hosts/{HOST_ID}/filesystem", params={"path": "~"})
    assert response.status_code == 200, response.text
    assert env.host.computer.events == ["resume", "claim", "start_host"]


_MESSAGE = {
    "type": "message",
    "data": {"role": "user", "content": [{"type": "input_text", "text": "hello"}]},
}


@pytest.mark.parametrize(
    "entry",
    [
        "message",
        "new_conversation",
        "file_write",
        "file_edit",
        "shell",
        "host_files",
        "harness_refresh",
        "runner_launch",
    ],
)
async def test_every_entry_path_refuses_new_work_with_the_reason(env: _Env, entry: str) -> None:
    """With the computer online but "not saved", each path that starts work is refused.

    The refusal carries the provider's reason, so it proves the readiness step
    runs on that path (RC-35); nothing is woken or launched. Opening a terminal
    and changing the model share the file write's readiness step
    (``ensure_runner_connected``).
    """
    register_persistent_row(env.app.state.host_store)
    env.host.computer.status = "host_running"
    assert env.host.connect(FIRST_TOKEN)
    env.host.computer.admitting = False
    agent = await create_test_agent(env.client, name=f"not-saved-{entry}")
    session = env.conversations.create_conversation(
        agent_id=agent["id"], host_id=HOST_ID, workspace="/root/workspace"
    )
    files = f"/v1/sessions/{session.id}/resources/environments/default/filesystem/notes.txt"
    requests = {
        "message": ("POST", f"/v1/sessions/{session.id}/events", _MESSAGE),
        "new_conversation": (
            "POST",
            "/v1/sessions",
            {"agent_id": agent["id"], "host_id": HOST_ID, "workspace": "/root/workspace"},
        ),
        "file_write": ("PUT", files, {"content": "x"}),
        "file_edit": ("PATCH", files, {"old_text": "x", "new_text": "y"}),
        "shell": (
            "POST",
            f"/v1/sessions/{session.id}/resources/environments/default/shell",
            {"command": "true"},
        ),
        "host_files": ("GET", f"/v1/hosts/{HOST_ID}/filesystem?path=~", None),
        "harness_refresh": ("POST", f"/v1/hosts/{HOST_ID}/harnesses/refresh", None),
        "runner_launch": (
            "POST",
            f"/v1/hosts/{HOST_ID}/runners",
            {"session_id": session.id, "workspace": "/root/workspace"},
        ),
    }
    method, path, body = requests[entry]
    response = await env.client.request(method, path, json=body)
    assert response.status_code == 503, response.text
    error = response.json()["error"]
    assert error["code"] == "host_not_admitting"
    assert "the last save is 35 minutes old" in error["message"]
    assert env.host.computer.events == []


async def test_inventory_request_wakes_an_asleep_computer(env: _Env) -> None:
    """Host inventory routes (here, plugins) wake an asleep computer like any host request."""
    register_persistent_row(env.app.state.host_store)
    env.host.computer.status = "stopped"
    response = await env.client.get(f"/v1/hosts/{HOST_ID}/plugins")
    assert response.status_code == 200, response.text
    assert response.json() == {"plugins": []}
    assert env.host.computer.events == ["resume", "claim", "start_host"]
