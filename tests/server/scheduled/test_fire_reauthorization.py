"""A schedule's fire re-checks its owner's authority where it acts (WP8.2b, T8).

Authority checked at the start of a fire does not last: waking a computer or
starting a runner can take minutes. After the wake, before dispatch and again
before the prompt reaches the runner, the task must still exist under the same
owner registration, the owner must be active, and the host must still be the
owner's. A fire never falls back to another member's host.
"""

from __future__ import annotations

import dataclasses
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
    _continued_session,
    _continuing_task,
    _deps,
    _drain,
    _ExistingConversationStore,
    _FakeConversation,
    _FakeHost,
    _FakeTeamStore,
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
    fire_mod._make_managed_sandbox_start(fire.deps)(
        fire.conv, fire.task, None, fire_mod._LEGACY_LAUNCH_ERRORS
    )
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
    assert fire.pending() == ["failed"]


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

    fire_mod._make_managed_sandbox_start(fire.deps)(
        fire.conv, fire.task, None, fire_mod._LEGACY_LAUNCH_ERRORS
    )
    await asyncio.gather(*list(sessions._managed_launch_tasks), return_exceptions=True)

    provision.assert_not_awaited()
    assert fire.pending() == ["failed"]
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


# ── r2: the real forward helper, paused mid-preparation ──────────────────────


class _World:
    """Real stores for Alice's task, thread and computer, and a recording runner."""

    def __init__(self, db_uri: str, *, harness_override: str | None = None) -> None:
        import threading

        import httpx

        from omnigent.server.accounts_store import SqlAlchemyAccountStore
        from omnigent.server.auth import LEVEL_OWNER
        from omnigent.server.routes._sessions.common import _TURN_ACTOR_LABEL
        from omnigent.stores.conversation_store.sqlalchemy_store import (
            SqlAlchemyConversationStore,
        )
        from omnigent.stores.host_store import HostStore
        from omnigent.stores.permission_store.sqlalchemy_store import SqlAlchemyPermissionStore
        from omnigent.stores.scheduled_task_store.sqlalchemy_store import (
            SqlAlchemyScheduledTaskStore,
        )

        paused, release = threading.Event(), threading.Event()
        self.pause_actor_label = False

        world = self

        class PausingConversations(SqlAlchemyConversationStore):
            """Holds the forward's turn-actor label write until the test releases it."""

            def set_labels(self, conversation_id: str, labels: dict[str, str]) -> Any:
                if world.pause_actor_label and _TURN_ACTOR_LABEL in labels:
                    paused.set()
                    assert release.wait(10)
                return super().set_labels(conversation_id, labels)

        self.paused, self.release = paused, release
        self.db_uri = db_uri
        self.accounts = SqlAlchemyAccountStore(db_uri)
        self.accounts.create_user_with_password("alice", "hash")
        self.conversations = PausingConversations(db_uri)
        self.permissions = SqlAlchemyPermissionStore(db_uri)
        self.schedules = SqlAlchemyScheduledTaskStore(db_uri)
        self.hosts = HostStore(db_uri)
        self.host_id = "c" * 32
        self.hosts.upsert_on_connect(self.host_id, "laptop", "alice")
        self.task = self.schedules.create(
            "a" * 32,
            "nightly",
            "summarize the night",
            "FREQ=DAILY",
            "alice",
            "b" * 32,
            "UTC",
            host_id=self.host_id,
            workspace="/tmp",
        )
        self.conv = self.conversations.create_conversation(
            labels={"omnigent.scheduled_task_origin": self.task.id},
            harness_override=harness_override,
        )
        self.permissions.grant("alice", self.conv.id, LEVEL_OWNER)
        self.conversations.set_host_id(self.conv.id, self.host_id, workspace="/tmp")
        self.conv = self.conversations.get_conversation(self.conv.id)
        self.posts: list[str] = []

        def handle(request: httpx.Request) -> httpx.Response:
            if request.method == "POST" and request.url.path.endswith("/events"):
                self.posts.append(request.url.path)
            return httpx.Response(202, json={"queued": True})

        self.runner = httpx.AsyncClient(
            transport=httpx.MockTransport(handle), base_url="http://runner"
        )
        self.deps = _deps(
            self.schedules,
            conversation_store=self.conversations,
            permission_store=self.permissions,
            host_store=self.hosts,
            host_registry=FakeHostRegistry(online={self.host_id}),
        )

    def items(self) -> list[Any]:
        return self.conversations.list_items(self.conv.id, limit=100, order="asc").data

    def last_item(self) -> Any:
        items = self.items()
        return items[-1] if items else None


def _hand_host_to_bob(world: _World) -> None:
    from sqlalchemy import update

    from omnigent.db.db_models import SqlHost
    from omnigent.db.utils import get_or_create_engine

    world.accounts.create_user_with_password("bob", "hash")
    bob = world.accounts.get_user("bob")
    with get_or_create_engine(world.db_uri).begin() as connection:
        connection.execute(
            update(SqlHost)
            .where(SqlHost.host_id == world.host_id)
            .values(user_id="bob", account_generation=bob.account_generation)
        )


_LOSSES = {
    "task-deleted": lambda world: world.schedules.delete(world.task.id),
    "host-handed-over": _hand_host_to_bob,
    "owner-removed": lambda world: world.accounts.delete_user("alice"),
}


async def _paused_routing(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Any]:
    """Hold the auto-harness routing call until the test releases it."""
    import asyncio

    from omnigent.server import smart_routing

    routing, release = asyncio.Event(), asyncio.Event()

    async def route(*_args: Any, **_kwargs: Any) -> tuple[None, None, None, str]:
        routing.set()
        await release.wait()
        return None, None, None, "unavailable"

    monkeypatch.setattr(smart_routing, "route_session_harness", route)
    return routing, release


def _real_connected_dispatch(monkeypatch: pytest.MonkeyPatch, world: _World) -> Any:
    """The real connected-host dispatch; only launching the runner is faked."""
    import omnigent.server.routes._host_launch as host_launch
    import omnigent.server.routes.sessions as sessions_routes

    def resolve(**kwargs: Any) -> Any:
        conv = kwargs["conversation_store"].get_conversation(kwargs["session_id"])
        return type("Target", (), {"conv": conv, "conn": object()})()

    async def launch(*_args: Any, **_kwargs: Any) -> Any:
        return type("Attempt", (), {"error": None, "runner_id": "runner_1"})()

    async def runner_client(*_args: Any, **_kwargs: Any) -> Any:
        return world.runner

    async def initialized(*_args: Any, **_kwargs: Any) -> None:
        return None

    monkeypatch.setattr(host_launch, "resolve_host_launch", resolve)
    monkeypatch.setattr(sessions_routes, "_launch_runner_on_host", launch)
    monkeypatch.setattr(sessions_routes, "_wait_for_runner_client", runner_client)
    monkeypatch.setattr(sessions_routes, "_ensure_runner_session_initialized", initialized)
    return fire_mod._make_connected_host_dispatch(world.deps)


@pytest.mark.asyncio
@pytest.mark.parametrize("pause", ["persistence", "routing"])
@pytest.mark.parametrize("loss", [*_LOSSES, None])
async def test_connected_prompt_is_checked_inside_the_forward(
    db_uri: str, monkeypatch: pytest.MonkeyPatch, pause: str, loss: str | None
) -> None:
    """r2: authority lost while the real forward prepares: nothing sent, nothing to replay."""
    import asyncio

    world = _World(db_uri, harness_override="auto" if pause == "routing" else None)
    dispatch = _real_connected_dispatch(monkeypatch, world)
    if pause == "routing":
        reached, release = await _paused_routing(monkeypatch)
    else:
        world.pause_actor_label = True
        reached = None

    sending = asyncio.create_task(dispatch(world.conv, world.task))
    if reached is not None:
        await asyncio.wait_for(reached.wait(), timeout=5)
    else:
        assert await asyncio.to_thread(world.paused.wait, 5)
    if loss is not None:
        _LOSSES[loss](world)
    if pause == "routing":
        release.set()
    else:
        world.release.set()

    if loss is None:
        await asyncio.wait_for(sending, timeout=10)
        assert len(world.posts) == 1
        assert world.items()[0].type == "message"
        return
    with pytest.raises(fire_mod.ScheduledFireRefused):
        await asyncio.wait_for(sending, timeout=10)
    assert world.posts == []
    last = world.last_item()
    # Nothing was persisted, or the persisted prompt is followed by why it wasn't sent.
    assert last is None or (last.type == "error" and last.data.code == "prompt_not_sent")


def _managed_queue(world: _World) -> Any:
    """A startup queue whose dispatch is the real event dispatch with the real forward."""
    from omnigent.server.host_registry import HostRegistry
    from omnigent.server.routes.sessions import _dispatch_session_event_to_runner
    from omnigent.server.startup_inputs import ManagedStartupInputs

    startup = ManagedStartupInputs(world.conversations, HostRegistry())

    async def dispatch(context: Any, session_id: str, event: Any) -> dict[str, Any]:
        conv = world.conversations.get_conversation(session_id)
        result = await _dispatch_session_event_to_runner(
            session_id,
            conv,
            event,
            world.conversations,
            world.runner,
            agent_name=None,
            file_store=None,
            artifact_store=None,
            created_by=context.user_id,
            startup_input=context,
        )
        return {"item_id": result.item_id} if result.item_id else {}

    startup.dispatch = dispatch
    startup.scheduled_authority = fire_mod.build_startup_authority(world.deps)
    return startup


async def _recover(startup: Any, world: _World) -> None:
    """Run the real recovery loop for the session until nothing is left to deliver."""
    import asyncio
    from types import SimpleNamespace

    connection = SimpleNamespace(host_id=world.host_id)
    for _ in range(5):
        if not any(
            item.status == "in_progress"
            for item in world.conversations.list_pending_inputs(world.conv.id)
        ):
            return
        await startup._recover_session(object(), world.conv.id, connection)
        await asyncio.gather(*list(startup._deliveries.values()), return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("loss", [*_LOSSES, None])
async def test_managed_prompt_is_checked_inside_the_forward(
    db_uri: str, monkeypatch: pytest.MonkeyPatch, loss: str | None
) -> None:
    """r2: a saved prompt whose task is lost while the real forward prepares is never sent."""
    import asyncio

    world = _World(db_uri)
    startup = _managed_queue(world)
    await fire_mod._make_managed_sandbox_dispatch(
        dataclasses.replace(world.deps, startup_inputs=startup)
    )(world.conv, world.task)
    [saved] = world.conversations.list_pending_inputs(world.conv.id)
    assert saved.data.startup_options.scheduled_task_id == world.task.id
    world.pause_actor_label = True

    delivering = asyncio.create_task(_recover(startup, world))
    assert await asyncio.to_thread(world.paused.wait, 5)
    if loss is not None:
        _LOSSES[loss](world)
    world.release.set()
    await asyncio.wait_for(delivering, timeout=10)

    if loss is None:
        assert len(world.posts) == 1
        return
    assert world.posts == []
    last = world.last_item()
    assert last is not None and last.type == "error"
    assert all(
        item.status == "failed" for item in world.conversations.list_pending_inputs(world.conv.id)
    )


# ── r2: inputs saved before they carried their origin ────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("task_state", ["deleted", "kept", "no-task-check"])
async def test_an_upgrade_format_scheduled_prompt_is_reconciled_on_recovery(
    db_uri: str, task_state: str
) -> None:
    """r2: a fired session's untagged first prompt is scheduled work; later messages aren't."""
    from starlette.datastructures import Headers

    from omnigent.server.schemas import SessionEventInput

    world = _World(db_uri)
    queue_before_upgrade = _managed_queue(world)

    def message(text: str) -> SessionEventInput:
        return SessionEventInput(
            type="message",
            data={"role": "user", "content": [{"type": "input_text", "text": text}]},
        )

    # As the previous release saved them: the fire's prompt carried no origin.
    scheduled = await queue_before_upgrade.enqueue(
        world.conv.id, message("summarize the night"), user_id="alice", headers=Headers({})
    )
    human = await queue_before_upgrade.enqueue(
        world.conv.id,
        message("and list my calendar"),
        user_id="alice",
        headers=Headers({}),
        first=False,
    )
    assert scheduled.data.startup_options.scheduled_task_id is None
    if task_state == "deleted":
        world.schedules.delete(world.task.id)

    restarted = _managed_queue(world)
    if task_state == "no-task-check":
        # Fail closed: an origin that can't be checked is not sent.
        restarted.scheduled_authority = None
    await _recover(restarted, world)

    statuses = {
        item.id: item.status for item in world.conversations.list_pending_inputs(world.conv.id)
    }
    if task_state == "kept":
        # Its registration is the task row's; both messages go out, in order.
        assert len(world.posts) == 2
        assert statuses == {}
    else:
        assert statuses.get(scheduled.id) == "failed"
        # The person's own message is still delivered as theirs.
        assert human.id not in statuses
        assert len(world.posts) == 1


# ── A continued session runs on its task owner's authority, at every firing ──


@dataclasses.dataclass
class _ContinuedFire:
    """A real firing into a continued session, with the runner's helpers faked."""

    store: FakeScheduledTaskStore
    permissions: FakePermissionStore
    host: _FakeHost
    conversations: _ExistingConversationStore
    deps: Any
    prompts: list[dict[str, Any]]
    # What happens while the runner is reached, and while it initializes.
    during_wake: Any = None
    during_initialization: Any = None

    async def fire(self) -> None:
        fire_dispatch = fire_mod._FireDispatch(
            connected=AsyncMock(),
            connected_preflight=None,
            managed=AsyncMock(),
            reuse=fire_mod._make_reuse_dispatch(self.deps),
        )
        await fire_mod._run_fire(self.deps, 0, "task_1", fire_dispatch)

    @property
    def run(self) -> tuple[str, str | None]:
        (run,) = self.store.runs
        return run["status"], run["error_code"]


@pytest.fixture
def continued(monkeypatch: pytest.MonkeyPatch) -> _ContinuedFire:
    import omnigent.server.routes.sessions as sessions_routes

    host = _FakeHost("host_1", _OWNER)
    permissions = FakePermissionStore(users={_OWNER})
    permissions.grant(_OWNER, "conv_prev", 4)
    conversations = _ExistingConversationStore(_continued_session())
    store = FakeScheduledTaskStore(rows={"task_1": _continuing_task(user_id=_OWNER)})
    deps = _deps(
        store,
        conversation_store=conversations,
        permission_store=permissions,
        host_store=FakeHostStore({host.host_id: host}),
        host_registry=FakeHostRegistry(online={host.host_id}),
    )
    world = _ContinuedFire(store, permissions, host, conversations, deps, [])

    async def reach_runner(*, conv: Any, **_kwargs: Any) -> tuple[object, Any]:
        if world.during_wake is not None:
            world.during_wake(world)
        return object(), conv

    async def initialize(*_args: Any, **_kwargs: Any) -> None:
        if world.during_initialization is not None:
            world.during_initialization(world)

    async def submit(*_args: Any, **kwargs: Any) -> None:
        # The real helper calls this right before it forwards.
        await kwargs["authorize_submission"]()
        world.prompts.append(kwargs)

    monkeypatch.setattr(sessions_routes, "ensure_runner_connected", reach_runner)
    monkeypatch.setattr(sessions_routes, "_ensure_runner_session_initialized", initialize)
    monkeypatch.setattr(sessions_routes, "_dispatch_session_event_to_runner", submit)
    return world


def _suspended(world: _ContinuedFire) -> None:
    world.deps.team_store = _FakeTeamStore({_OWNER: "suspended"})


def _removed(world: _ContinuedFire) -> None:
    assert world.permissions.users is not None
    world.permissions.users.discard(_OWNER)


def _session_passed_to_bob(world: _ContinuedFire) -> None:
    world.permissions.grants = [("bob@example.com", "conv_prev", 4)]


def _second_owner(world: _ContinuedFire) -> None:
    world.permissions.grants.append(("bob@example.com", "conv_prev", 4))


def _computer_passed_to_bob(world: _ContinuedFire) -> None:
    world.host.user_id = "bob@example.com"


def _computer_re_registered(world: _ContinuedFire) -> None:
    world.host.account_generation = "a-later-registration"


def _continued_task_deleted(world: _ContinuedFire) -> None:
    world.store._rows["task_1"] = replace(world.store._rows["task_1"], state="deleted")


@pytest.mark.asyncio
async def test_a_continued_firing_sends_the_prompt_as_the_task_owner(
    continued: _ContinuedFire,
) -> None:
    await continued.fire()

    assert continued.run == ("succeeded", None)
    (prompt,) = continued.prompts
    assert prompt["created_by"] == _OWNER
    assert prompt["initiator"].kind == "schedule"
    assert prompt["initiator"].user == _OWNER
    assert continued.conversations.created == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("loss", "expected"),
    [
        (_session_passed_to_bob, ("failed", "session_authority_revoked")),
        (_second_owner, ("failed", "session_authority_revoked")),
        (_computer_passed_to_bob, ("failed", "host_authority_revoked")),
        (_computer_re_registered, ("failed", "host_authority_revoked")),
    ],
    ids=["session-to-bob", "second-owner", "computer-to-bob", "computer-re-registered"],
)
async def test_a_continued_session_never_crosses_to_another_member(
    continued: _ContinuedFire, loss: Any, expected: tuple[str, str]
) -> None:
    """The owner grant alone is not enough: the session and its computer must be the owner's."""
    loss(continued)

    await continued.fire()

    assert continued.run == expected
    assert continued.prompts == []
    # Nothing falls back to a new session either.
    assert continued.conversations.created == []


@pytest.mark.asyncio
async def test_a_continued_session_the_owner_cannot_reach_tells_nothing_more(
    continued: _ContinuedFire,
) -> None:
    continued.permissions.unowned.add("conv_prev")

    await continued.fire()

    assert continued.run == ("failed", "continued_session_forbidden")
    assert continued.prompts == []


@pytest.mark.asyncio
async def test_a_suspended_owners_continued_task_skips_and_stays(
    continued: _ContinuedFire,
) -> None:
    _suspended(continued)

    await continued.fire()

    assert continued.run == ("skipped", "owner_suspended")
    assert continued.prompts == []
    assert {"id": "task_1", "state": "deleted"} not in continued.store.updates


@pytest.mark.asyncio
async def test_a_removed_owners_continued_task_is_disabled(continued: _ContinuedFire) -> None:
    _removed(continued)

    await continued.fire()

    assert continued.run == ("failed", "owner_deleted")
    assert continued.prompts == []
    assert {"id": "task_1", "state": "deleted"} in continued.store.updates


_LOST_WHILE_FIRING = [
    (_suspended, "owner_suspended"),
    (_removed, "owner_deleted"),
    (_session_passed_to_bob, "session_authority_revoked"),
    (_computer_passed_to_bob, "host_authority_revoked"),
    (_continued_task_deleted, "task_authority_revoked"),
]
_LOST_IDS = ["suspended", "removed", "session-to-bob", "computer-to-bob", "task-deleted"]


@pytest.mark.asyncio
@pytest.mark.parametrize(("loss", "error_code"), _LOST_WHILE_FIRING, ids=_LOST_IDS)
async def test_authority_lost_during_a_wake_sends_no_prompt_to_a_continued_session(
    continued: _ContinuedFire, loss: Any, error_code: str
) -> None:
    """Waking the computer can take minutes; nothing held before it carries over."""
    continued.during_wake = loss

    await continued.fire()

    assert continued.run == ("failed", error_code)
    assert continued.prompts == []


@pytest.mark.asyncio
@pytest.mark.parametrize(("loss", "error_code"), _LOST_WHILE_FIRING, ids=_LOST_IDS)
async def test_authority_lost_during_initialization_sends_no_prompt_to_a_continued_session(
    continued: _ContinuedFire, loss: Any, error_code: str
) -> None:
    continued.during_initialization = loss

    await continued.fire()

    assert continued.run == ("failed", error_code)
    assert continued.prompts == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("lifecycle", "problem"),
    [("persistent", None), (None, "continued_session_incompatible")],
    ids=["computer", "one-session-sandbox"],
)
async def test_a_task_continues_a_session_on_a_computer_but_not_in_a_sandbox(
    lifecycle: str | None, problem: str | None
) -> None:
    """A computer is pinned like a connected host, though a provider runs it (D38)."""
    host = _FakeHost("host_1", _OWNER, sandbox_provider="cloud", lifecycle=lifecycle)

    found = await fire_mod.continued_session_problem(
        _continuing_task(user_id=_OWNER),
        _continued_session(),
        host_store=FakeHostStore({host.host_id: host}),
        permission_store=FakePermissionStore(users={_OWNER}),
    )

    assert (found[1] if found is not None else None) == problem
