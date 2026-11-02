"""A schedule's fire re-checks its owner's authority where it acts (WP8.2b, T8).

Authority checked at the start of a fire does not last: waking a computer or
starting a runner can take minutes. After the wake, before dispatch and again
before the prompt reaches the runner, the task must still exist under the same
owner registration, the owner must be active, and the host must still be the
owner's. A fire never falls back to another member's host.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from omnigent.server.scheduled import fire as fire_mod
from omnigent.server.scheduled.fire import build_on_fire
from tests.server.scheduled.test_fire import (
    FakeConversationStore,
    FakeHostRegistry,
    FakeHostStore,
    FakePermissionStore,
    FakeScheduledTaskStore,
    _deps,
    _drain,
    _FakeConversation,
    _FakeHost,
    _task,
)

_OWNER = "alice@example.com"


def _woken_fire(
    monkeypatch: pytest.MonkeyPatch, during_wake: Any
) -> tuple[FakeScheduledTaskStore, list[Any], AsyncMock, Any]:
    """A fire pinned to the owner's persistent computer; *during_wake* runs while it wakes."""
    from omnigent.server import persistent_hosts

    host = _FakeHost("host_1", _OWNER, sandbox_provider="cloud", lifecycle="persistent")
    store = FakeScheduledTaskStore(rows={"task_1": _task(user_id=_OWNER, workspace=None)})
    permissions = FakePermissionStore(users={_OWNER})
    hosts = FakeHostStore({host.host_id: host})

    async def ensure_host_ready(_host_id: str, **_kwargs: Any) -> None:
        during_wake(store=store, host=host, permissions=permissions)

    stat_home = AsyncMock(return_value="/home/alice")
    monkeypatch.setattr(persistent_hosts, "ensure_host_ready", ensure_host_ready)
    monkeypatch.setattr(fire_mod, "_resolve_default_workspace", stat_home)
    dispatched: list[Any] = []

    async def launch(conv: Any, task: Any) -> None:
        dispatched.append((conv, task))

    deps = _deps(
        store,
        permission_store=permissions,
        host_store=hosts,
        host_registry=FakeHostRegistry(online={host.host_id}),
    )
    return store, dispatched, stat_home, build_on_fire(deps, launch_dispatch=launch)


def _given_to_bob(*, host: _FakeHost, **_: Any) -> None:
    host.user_id = "bob@example.com"


def _re_registered(*, host: _FakeHost, **_: Any) -> None:
    host.account_generation = "a-later-registration"


def _task_deleted(*, store: FakeScheduledTaskStore, **_: Any) -> None:
    store._rows["task_1"] = replace(store._rows["task_1"], state="deleted")


def _owner_removed(*, permissions: FakePermissionStore, **_: Any) -> None:
    assert permissions.users is not None
    permissions.users.discard(_OWNER)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("during_wake", "error_code"),
    [
        (_given_to_bob, "host_authority_revoked"),
        (_re_registered, "host_authority_revoked"),
        (_task_deleted, "task_authority_revoked"),
        (_owner_removed, "owner_deleted"),
    ],
    ids=["host-given-to-bob", "host-re-registered", "task-deleted", "owner-removed"],
)
async def test_a_wake_that_loses_authority_dispatches_nothing(
    monkeypatch: pytest.MonkeyPatch, during_wake: Any, error_code: str
) -> None:
    store, dispatched, stat_home, on_fire = _woken_fire(monkeypatch, during_wake)

    await on_fire(0, "task_1")
    await _drain()

    assert dispatched == []
    # Not even a home-directory stat reaches the computer after the wake.
    stat_home.assert_not_awaited()
    assert store.runs[0]["status"] == "failed"
    assert store.runs[0]["error_code"] == error_code


@pytest.mark.asyncio
async def test_a_wake_that_keeps_authority_dispatches(monkeypatch: pytest.MonkeyPatch) -> None:
    store, dispatched, stat_home, on_fire = _woken_fire(monkeypatch, lambda **_: None)

    await on_fire(0, "task_1")
    await _drain()

    assert len(dispatched) == 1
    stat_home.assert_awaited_once()
    assert store.runs[0]["status"] == "running"


@pytest.mark.asyncio
async def test_host_lost_between_preflight_and_dispatch_dispatches_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The pre-dispatch re-check runs after the session exists, on the real preflight path."""
    host = _FakeHost("host_1", _OWNER)
    store = FakeScheduledTaskStore(rows={"task_1": _task(user_id=_OWNER)})
    conversations = FakeConversationStore()
    deps = _deps(
        store,
        conversation_store=conversations,
        permission_store=FakePermissionStore(users={_OWNER}),
        host_store=FakeHostStore({host.host_id: host}),
        host_registry=FakeHostRegistry(online={host.host_id}),
    )

    async def valid_but_host_lost(*_args: Any, **_kwargs: Any) -> None:
        # The workspace checks pass; meanwhile the computer passes to Bob.
        host.user_id = "bob@example.com"

    monkeypatch.setattr(fire_mod, "_validate_fire_session_inputs", valid_but_host_lost)
    from omnigent.server import persistent_hosts

    monkeypatch.setattr(persistent_hosts, "ensure_host_ready", AsyncMock())
    dispatched: list[Any] = []

    async def launch(conv: Any, task: Any) -> None:
        dispatched.append(conv)

    fire_dispatch = fire_mod._FireDispatch(
        connected=launch,
        connected_preflight=fire_mod._make_connected_host_preflight(deps),
        managed=launch,
    )
    await fire_mod._run_fire(deps, 0, "task_1", fire_dispatch)

    assert dispatched == []
    assert len(conversations.created) == 1
    assert store.runs[0]["status"] == "failed"
    assert store.runs[0]["error_code"] == "host_authority_revoked"


@pytest.mark.asyncio
async def test_unpinned_fire_never_falls_back_to_another_members_host() -> None:
    store = FakeScheduledTaskStore(rows={"task_1": _task(user_id=_OWNER, host_id=None)})
    dispatched: list[Any] = []

    async def launch(conv: Any, task: Any) -> None:
        dispatched.append(conv)

    deps = _deps(
        store,
        host_store=FakeHostStore({"bobs": _FakeHost("bobs", "bob@example.com")}),
        host_registry=FakeHostRegistry(online={"bobs"}),
    )
    await build_on_fire(deps, launch_dispatch=launch)(0, "task_1")
    await _drain()

    assert dispatched == []
    assert store.runs[0]["error_code"] == "no_online_host"


def _connected_dispatch(
    monkeypatch: pytest.MonkeyPatch, while_runner_starts: Any
) -> tuple[Any, list[dict[str, Any]], _FakeHost]:
    """The real connected-host dispatch with its launch helpers faked."""
    import omnigent.server.routes._host_launch as host_launch
    import omnigent.server.routes.sessions as sessions_routes

    host = _FakeHost("host_1", _OWNER)
    prompts: list[dict[str, Any]] = []

    def resolve(**kwargs: Any) -> Any:
        conv = kwargs["conversation_store"].get_conversation(kwargs["session_id"])
        return type("Target", (), {"conv": conv, "conn": object()})()

    async def launch(*_args: Any, **_kwargs: Any) -> Any:
        return type("Attempt", (), {"error": None, "runner_id": "runner_1"})()

    async def wait_for_runner(*_args: Any, **_kwargs: Any) -> object:
        while_runner_starts(host)
        return object()

    async def initialized(*_args: Any, **_kwargs: Any) -> None:
        return None

    async def dispatch_prompt(*_args: Any, **kwargs: Any) -> None:
        prompts.append(kwargs)

    monkeypatch.setattr(host_launch, "resolve_host_launch", resolve)
    monkeypatch.setattr(sessions_routes, "_launch_runner_on_host", launch)
    monkeypatch.setattr(sessions_routes, "_wait_for_runner_client", wait_for_runner)
    monkeypatch.setattr(sessions_routes, "_ensure_runner_session_initialized", initialized)
    monkeypatch.setattr(sessions_routes, "_dispatch_session_event_to_runner", dispatch_prompt)
    permissions = FakePermissionStore(users={_OWNER})
    permissions.grant(_OWNER, "conv_1", 4)
    deps = _deps(
        FakeScheduledTaskStore(rows={"task_1": _task(user_id=_OWNER)}),
        permission_store=permissions,
        host_store=FakeHostStore({host.host_id: host}),
        host_registry=FakeHostRegistry(online={host.host_id}),
    )
    return fire_mod._make_connected_host_dispatch(deps), prompts, host


@pytest.mark.asyncio
async def test_prompt_is_not_sent_when_the_host_changed_while_the_runner_started(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def give_to_bob(host: _FakeHost) -> None:
        host.user_id = "bob@example.com"

    dispatch, prompts, _host = _connected_dispatch(monkeypatch, give_to_bob)

    with pytest.raises(fire_mod._CannotLaunchScheduledFire) as refused:
        await dispatch(_FakeConversation(id="conv_1", agent_id="ag_1"), _task(user_id=_OWNER))
    assert refused.value.error_code == "host_authority_revoked"
    assert prompts == []


@pytest.mark.asyncio
async def test_prompt_is_sent_as_the_owner_when_nothing_changed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dispatch, prompts, _host = _connected_dispatch(monkeypatch, lambda _host: None)

    await dispatch(_FakeConversation(id="conv_1", agent_id="ag_1"), _task(user_id=_OWNER))
    assert [prompt["created_by"] for prompt in prompts] == [_OWNER]
