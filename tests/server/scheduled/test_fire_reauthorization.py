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


# ── Authority lost while the runner initializes (connected host) ─────────────


def _paused_connected_fire(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Any, FakeScheduledTaskStore, _FakeHost, FakePermissionStore, Any, list[Any]]:
    """A real connected-host fire whose session initialization waits for the test."""
    import asyncio

    import omnigent.server.routes._host_launch as host_launch
    import omnigent.server.routes.sessions as sessions_routes
    from omnigent.server import persistent_hosts

    host = _FakeHost("host_1", _OWNER)
    store = FakeScheduledTaskStore(rows={"task_1": _task(user_id=_OWNER)})
    permissions = FakePermissionStore(users={_OWNER})
    deps = _deps(
        store,
        permission_store=permissions,
        host_store=FakeHostStore({host.host_id: host}),
        host_registry=FakeHostRegistry(online={host.host_id}),
    )
    initializing, release = asyncio.Event(), asyncio.Event()
    submissions: list[Any] = []

    def resolve(**kwargs: Any) -> Any:
        conv = kwargs["conversation_store"].get_conversation(kwargs["session_id"])
        return type("Target", (), {"conv": conv, "conn": object()})()

    async def launch(*_args: Any, **_kwargs: Any) -> Any:
        return type("Attempt", (), {"error": None, "runner_id": "runner_1"})()

    async def connected(*_args: Any, **_kwargs: Any) -> object:
        return object()

    async def initialize(*_args: Any, **_kwargs: Any) -> None:
        initializing.set()
        await release.wait()

    async def submit(*args: Any, **kwargs: Any) -> None:
        # The real helper calls this right before it forwards.
        await kwargs["authorize_submission"]()
        submissions.append(args[2])

    monkeypatch.setattr(host_launch, "resolve_host_launch", resolve)
    monkeypatch.setattr(sessions_routes, "_launch_runner_on_host", launch)
    monkeypatch.setattr(sessions_routes, "_wait_for_runner_client", connected)
    monkeypatch.setattr(sessions_routes, "_ensure_runner_session_initialized", initialize)
    monkeypatch.setattr(sessions_routes, "_dispatch_session_event_to_runner", submit)
    monkeypatch.setattr(persistent_hosts, "ensure_host_ready", AsyncMock())
    monkeypatch.setattr(fire_mod, "_validate_fire_session_inputs", AsyncMock(return_value=None))
    fire_dispatch = fire_mod._FireDispatch(
        connected=fire_mod._make_connected_host_dispatch(deps),
        connected_preflight=fire_mod._make_connected_host_preflight(deps),
        managed=AsyncMock(),
    )
    firing = asyncio.create_task(fire_mod._run_fire(deps, 0, "task_1", fire_dispatch))
    return firing, store, host, permissions, (initializing, release), submissions


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("lose", "error_code"),
    [
        (_task_deleted, "task_authority_revoked"),
        (_given_to_bob, "host_authority_revoked"),
        (_owner_removed, "owner_deleted"),
    ],
    ids=["task-deleted", "host-handed-over", "owner-removed"],
)
async def test_authority_lost_during_initialization_sends_no_prompt(
    monkeypatch: pytest.MonkeyPatch, lose: Any, error_code: str
) -> None:
    """T8: pause initialization, lose authority, release: no prompt, a failed run."""
    import asyncio

    firing, store, host, permissions, (initializing, release), submissions = (
        _paused_connected_fire(monkeypatch)
    )
    await asyncio.wait_for(initializing.wait(), timeout=5)
    lose(store=store, host=host, permissions=permissions)
    release.set()
    await asyncio.wait_for(firing, timeout=5)

    assert submissions == []
    assert store.runs[0]["status"] == "failed"
    assert store.runs[0]["error_code"] == error_code


@pytest.mark.asyncio
async def test_initialization_that_keeps_authority_sends_the_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio

    firing, store, _host, _permissions, (initializing, release), submissions = (
        _paused_connected_fire(monkeypatch)
    )
    await asyncio.wait_for(initializing.wait(), timeout=5)
    release.set()
    await asyncio.wait_for(firing, timeout=5)

    assert len(submissions) == 1
    assert store.runs[0]["status"] == "running"


@pytest.mark.asyncio
@pytest.mark.parametrize("via", ["callback", "startup_input"])
@pytest.mark.parametrize("native", [False, True], ids=["chat", "native-terminal"])
async def test_dispatch_helper_checks_authority_right_before_forwarding(
    monkeypatch: pytest.MonkeyPatch, native: bool, via: str
) -> None:
    """The helper's own waits come first; the final check sits at the forward.

    A connected fire passes the check as a callback; a managed fire's saved
    prompt carries it on its startup input.
    """
    from starlette.datastructures import Headers

    from omnigent.entities import Conversation
    from omnigent.server.routes._sessions import orchestration
    from omnigent.server.schemas import SessionEventInput
    from omnigent.server.startup_inputs import StartupInputContext

    class Claimable(FakeConversationStore):
        def transition_pending_input(self, *_args: Any, **_kwargs: Any) -> bool:
            return True

    order: list[str] = []
    allowed = True

    async def authorize() -> None:
        order.append("authorize")
        if not allowed:
            raise fire_mod.ScheduledFireRefused(
                "task deleted", error_code="task_authority_revoked"
            )

    async def forward(*_args: Any, **_kwargs: Any) -> Any:
        order.append("forward")
        return "accepted" if native else "item_1"

    async def ready(*_args: Any, **_kwargs: Any) -> Any:
        order.append("terminal ready")
        return orchestration._NativeTerminalEnsureOutcome(error=None)

    monkeypatch.setattr(orchestration, "_is_native_terminal_session", lambda _conv: native)
    monkeypatch.setattr(orchestration, "_build_native_terminal_message_event", lambda *_a: None)
    monkeypatch.setattr(orchestration, "_ensure_native_terminal_ready", ready)
    monkeypatch.setattr(orchestration, "_forward_native_terminal_message", forward)
    monkeypatch.setattr(orchestration, "_forward_event_to_runner", forward)
    conv = Conversation(id="conv_1", created_at=0, updated_at=0, root_conversation_id="conv_1")
    event = SessionEventInput(
        type="message", data={"role": "user", "content": [{"type": "input_text", "text": "go"}]}
    )

    async def dispatch() -> Any:
        startup = (
            StartupInputContext(object(), Headers({}), _OWNER, "input_1", authorize=authorize)
            if via == "startup_input"
            else None
        )
        return await orchestration._dispatch_session_event_to_runner_impl(
            "conv_1",
            conv,
            event,
            Claimable(),
            object(),  # type: ignore[arg-type]
            agent_name=None,
            file_store=None,
            artifact_store=None,
            created_by=_OWNER,
            authorize_submission=authorize if via == "callback" else None,
            startup_input=startup,
        )

    await dispatch()
    assert order[-2:] == ["authorize", "forward"]
    if native:
        assert order.index("terminal ready") < order.index("authorize")

    order.clear()
    allowed = False
    with pytest.raises(fire_mod.ScheduledFireRefused):
        await dispatch()
    assert "forward" not in order


# ── A managed fire's saved prompt (delivered after the sandbox boots) ────────


class _ManagedFire:
    """Real stores and startup queue for one managed fire of Alice's task."""

    def __init__(self, db_uri: str) -> None:
        from omnigent.server.accounts_store import SqlAlchemyAccountStore
        from omnigent.server.auth import LEVEL_OWNER
        from omnigent.server.host_registry import HostRegistry
        from omnigent.server.managed_hosts import ManagedLaunchTracker
        from omnigent.stores.conversation_store.sqlalchemy_store import (
            SqlAlchemyConversationStore,
        )
        from omnigent.stores.permission_store.sqlalchemy_store import SqlAlchemyPermissionStore
        from omnigent.stores.scheduled_task_store.sqlalchemy_store import (
            SqlAlchemyScheduledTaskStore,
        )
        from tests.server.scheduled.test_fire import _managed_sandbox_config

        SqlAlchemyAccountStore(db_uri).create_user_with_password("alice", "hash")
        self.db_uri = db_uri
        self.conversations = SqlAlchemyConversationStore(db_uri)
        self.permissions = SqlAlchemyPermissionStore(db_uri)
        self.schedules = SqlAlchemyScheduledTaskStore(db_uri)
        self.task = self.schedules.create(
            "a" * 32,
            "nightly",
            "summarize the night",
            "FREQ=DAILY",
            "alice",
            "b" * 32,
            "UTC",
            execution_target="managed_sandbox",
        )
        self.conv = self.conversations.create_conversation()
        self.permissions.grant("alice", self.conv.id, LEVEL_OWNER)
        self.delivered: list[str] = []
        self.startup = self.queue()
        self.deps = _deps(
            self.schedules,
            conversation_store=self.conversations,
            permission_store=self.permissions,
            host_store=FakeHostStore(),
            host_registry=HostRegistry(),
            sandbox_config=_managed_sandbox_config(),
            managed_launches=ManagedLaunchTracker(),
            startup_inputs=self.startup,
        )
        self.startup.scheduled_authority = fire_mod.build_startup_authority(self.deps)

    def queue(self) -> Any:
        """A startup queue, as a (restarted) replica builds it."""
        from omnigent.server.host_registry import HostRegistry
        from omnigent.server.startup_inputs import ManagedStartupInputs

        startup = ManagedStartupInputs(self.conversations, HostRegistry())

        async def dispatch(context: Any, session_id: str, event: Any) -> dict[str, Any]:
            await context.authorize_submission()
            assert await context.claim(self.conversations, session_id)
            self.delivered.append(session_id)
            return {"item_id": "item_1"}

        startup.dispatch = dispatch
        return startup

    async def deliver(self, startup: Any) -> None:
        """Deliver every saved prompt, as the sandbox host's replica would."""
        for item in self.conversations.list_pending_inputs(self.conv.id):
            await startup._deliver(object(), self.conv.id, item)

    def pending(self) -> list[str | None]:
        return [item.status for item in self.conversations.list_pending_inputs(self.conv.id)]


@pytest.mark.asyncio
@pytest.mark.parametrize("queue", ["live", "recovered"])
async def test_a_deleted_tasks_saved_prompt_is_never_delivered(
    db_uri: str, monkeypatch: pytest.MonkeyPatch, queue: str
) -> None:
    """T8: enqueue, block provisioning, delete the task, release: nothing is delivered."""
    import asyncio

    from omnigent.server.routes import sessions

    fire = _ManagedFire(db_uri)
    provisioning, release = asyncio.Event(), asyncio.Event()

    async def provision(**_kwargs: Any) -> None:
        provisioning.set()
        await release.wait()

    monkeypatch.setattr(sessions, "_run_managed_launch", provision)
    await fire_mod._make_managed_sandbox_dispatch(fire.deps)(fire.conv, fire.task)
    fire_mod._make_managed_sandbox_start(fire.deps)(fire.conv, fire.task)
    await asyncio.wait_for(provisioning.wait(), timeout=5)
    [saved] = fire.conversations.list_pending_inputs(fire.conv.id)
    assert saved.data.startup_options.scheduled_task_id == fire.task.id

    assert fire.schedules.delete(fire.task.id)
    release.set()
    await asyncio.gather(*list(sessions._managed_launch_tasks), return_exceptions=True)
    if queue == "recovered":
        restarted = fire.queue()
        restarted.scheduled_authority = fire_mod.build_startup_authority(fire.deps)
        await fire.deliver(restarted)
    else:
        await fire.deliver(fire.startup)

    assert fire.delivered == []
    assert fire.pending() in ([], ["failed"])
    labels = fire.conversations.get_conversation(fire.conv.id).labels
    assert labels.get("omnigent.startup") == "cancelled"


@pytest.mark.asyncio
async def test_a_task_deleted_before_provisioning_starts_provisions_nothing(
    db_uri: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import asyncio

    from omnigent.server.routes import sessions

    fire = _ManagedFire(db_uri)
    provision = AsyncMock()
    monkeypatch.setattr(sessions, "_run_managed_launch", provision)
    await fire_mod._make_managed_sandbox_dispatch(fire.deps)(fire.conv, fire.task)
    assert fire.schedules.delete(fire.task.id)

    fire_mod._make_managed_sandbox_start(fire.deps)(fire.conv, fire.task)
    await asyncio.gather(*list(sessions._managed_launch_tasks), return_exceptions=True)

    provision.assert_not_awaited()
    assert fire.pending() == []
    await fire.deliver(fire.startup)
    assert fire.delivered == []


@pytest.mark.asyncio
@pytest.mark.parametrize("unchanged", [True, False], ids=["kept", "owner-re-registered"])
async def test_a_saved_prompt_is_delivered_only_while_its_task_holds(
    db_uri: str, unchanged: bool
) -> None:
    fire = _ManagedFire(db_uri)
    await fire_mod._make_managed_sandbox_dispatch(fire.deps)(fire.conv, fire.task)
    if not unchanged:
        # Alice's account removed and registered again: a new registration.
        from omnigent.server.accounts_store import SqlAlchemyAccountStore

        accounts = SqlAlchemyAccountStore(fire.db_uri)
        accounts.delete_user("alice")
        accounts.create_user_with_password("alice", "hash")

    await fire.deliver(fire.startup)

    assert fire.delivered == ([fire.conv.id] if unchanged else [])


@pytest.mark.asyncio
async def test_a_scheduled_prompt_is_refused_without_a_task_check(db_uri: str) -> None:
    """Fail closed: a queue that cannot verify the automation sends nothing."""
    fire = _ManagedFire(db_uri)
    await fire_mod._make_managed_sandbox_dispatch(fire.deps)(fire.conv, fire.task)
    unverified = fire.queue()

    await fire.deliver(unverified)

    assert fire.delivered == []


@pytest.mark.asyncio
async def test_native_dispatch_rechecks_after_its_own_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Authority lost between claiming and forwarding (routing waits there) sends nothing."""
    from omnigent.entities import Conversation
    from omnigent.server.routes._sessions import orchestration
    from omnigent.server.schemas import SessionEventInput

    checks: list[str] = []
    forwarded: list[str] = []

    async def authorize() -> None:
        checks.append("check")
        if len(checks) > 1:
            raise fire_mod.ScheduledFireRefused(
                "task deleted", error_code="task_authority_revoked"
            )

    async def forward(*_args: Any, **_kwargs: Any) -> Any:
        forwarded.append("forward")
        return "accepted"

    async def ready(*_args: Any, **_kwargs: Any) -> Any:
        return orchestration._NativeTerminalEnsureOutcome(error=None)

    monkeypatch.setattr(orchestration, "_is_native_terminal_session", lambda _conv: True)
    monkeypatch.setattr(orchestration, "_build_native_terminal_message_event", lambda *_a: None)
    monkeypatch.setattr(orchestration, "_ensure_native_terminal_ready", ready)
    monkeypatch.setattr(orchestration, "_forward_native_terminal_message", forward)
    conv = Conversation(id="conv_1", created_at=0, updated_at=0, root_conversation_id="conv_1")
    event = SessionEventInput(
        type="message", data={"role": "user", "content": [{"type": "input_text", "text": "go"}]}
    )

    with pytest.raises(fire_mod.ScheduledFireRefused):
        await orchestration._dispatch_session_event_to_runner_impl(
            "conv_1",
            conv,
            event,
            FakeConversationStore(),
            object(),  # type: ignore[arg-type]
            agent_name=None,
            file_store=None,
            artifact_store=None,
            created_by=_OWNER,
            authorize_submission=authorize,
        )
    assert checks == ["check", "check"]
    assert forwarded == []
