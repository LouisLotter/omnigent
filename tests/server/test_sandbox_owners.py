"""Owner metadata for sandbox providers that ask (``owner_context``).

A provider declaring the capability receives ``owner=``, the host's owner from
the server's own records, on every lifecycle call for a persistent host: the
account, the host and the account's current team membership. A provider
without it is called exactly as before.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import uuid
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

import pytest

from omnigent.db.utils import now_epoch
from omnigent.errors import OmnigentError
from omnigent.onboarding.sandboxes.base import SandboxHostLauncher
from omnigent.onboarding.sandboxes.types import (
    AdmissionReport,
    RepoWorkspace,
    SandboxCapabilities,
    SandboxOwner,
    StartClaim,
    SuspendResult,
)
from omnigent.server import persistent_hosts
from omnigent.server.accounts_store import SqlAlchemyAccountStore
from omnigent.server.host_registry import HostRegistry
from omnigent.server.managed_hosts import (
    ManagedSandboxConfig,
    ManagedSandboxDeployment,
    ManagedSandboxReaperConfig,
    PersistentHostPolicy,
)
from omnigent.server.managed_sandbox_reaper import ManagedSandboxReaper
from omnigent.server.sandbox_owners import launcher_call, owner_for, sandbox_owner, takes_owner
from omnigent.stores.host_store import Host, HostStore
from omnigent.stores.team_store import SqlAlchemyTeamStore
from tests.server import persistent_host_fakes
from tests.server.persistent_host_fakes import (
    FakeComputer,
    FakeHost,
    FakeUntypedComputer,
    call_soon,
)

pytestmark = [pytest.mark.timeout(60)]


@pytest.fixture(autouse=True)
def _bounded_waits(monkeypatch: pytest.MonkeyPatch) -> None:
    """The lifecycle's waits shortened as ``test_persistent_hosts`` shortens them.

    Without them each wake idles through the production status poll (2 s), and
    under a parallel run a wake can outlast the tests' own waits. Each test runs
    on its own event loop, so the host's lifecycle lock starts fresh too.
    """
    monkeypatch.setattr(persistent_hosts, "FENCE_DRAIN_S", 5.0)
    monkeypatch.setattr(persistent_hosts, "_STATUS_POLL_S", 0.05)
    monkeypatch.setattr(persistent_hosts, "_UNKNOWN_RETRY_S", 0.05)
    monkeypatch.setattr(persistent_hosts, "_resume_locks", {})


@dataclass
class OwnedComputer(FakeComputer):
    """:class:`FakeComputer` declaring ``owner_context``: records each call's owner."""

    provider: ClassVar[str] = "fakepersist"
    supports_owner_context: ClassVar[bool] = True
    owners: list[tuple[str, SandboxOwner | None]] = field(default_factory=list)

    def provision(self, name: str, *, owner: SandboxOwner | None = None) -> str:
        self.owners.append(("provision", owner))
        return super().provision(name)

    def admission(
        self, sandbox_id: str, *, owner: SandboxOwner | None = None
    ) -> AdmissionReport | None:
        self.owners.append(("admission", owner))
        return super().admission(sandbox_id)

    def resume(self, sandbox_id: str, *, owner: SandboxOwner | None = None) -> None:
        self.owners.append(("resume", owner))
        super().resume(sandbox_id)

    def claim_start(
        self, sandbox_id: str, *, operation_id: str, owner: SandboxOwner | None = None
    ) -> StartClaim:
        self.owners.append(("claim_start", owner))
        return super().claim_start(sandbox_id, operation_id=operation_id)

    def start_host(
        self,
        sandbox_id: str,
        *,
        token: str,
        host_id: str,
        host_name: str,
        server_url: str,
        repos: Sequence[RepoWorkspace] = (),
        host_config: dict[str, object] | None = None,
        on_stage: Any = None,
        start_attempt: int | None = None,
        owner: SandboxOwner | None = None,
    ) -> str:
        self.owners.append(("start_host", owner))
        return super().start_host(
            sandbox_id,
            token=token,
            host_id=host_id,
            host_name=host_name,
            server_url=server_url,
            repos=repos,
            host_config=host_config,
            on_stage=on_stage,
            start_attempt=start_attempt,
        )

    def is_running(self, sandbox_id: str, *, owner: SandboxOwner | None = None) -> bool | None:
        self.owners.append(("is_running", owner))
        return super().is_running(sandbox_id)

    def suspend(
        self, sandbox_id: str, *, operation_id: str, owner: SandboxOwner | None = None
    ) -> SuspendResult:
        self.owners.append(("suspend", owner))
        return super().suspend(sandbox_id, operation_id=operation_id)

    def terminate(self, sandbox_id: str, *, owner: SandboxOwner | None = None) -> None:
        self.owners.append(("terminate", owner))
        super().terminate(sandbox_id)


def test_the_capability_is_off_unless_a_provider_declares_it() -> None:
    assert FakeComputer().capabilities.owner_context is False
    assert OwnedComputer().capabilities.owner_context is True
    assert takes_owner(OwnedComputer()) and not takes_owner(FakeComputer())
    # A duck-typed launcher without capabilities never takes one.
    assert not takes_owner(object())


@pytest.fixture
def stores(db_uri: str) -> dict[str, Any]:
    return {
        "accounts": SqlAlchemyAccountStore(db_uri),
        "teams": SqlAlchemyTeamStore(db_uri),
        "hosts": HostStore(db_uri),
    }


def _host(store: HostStore, user_id: str) -> Host:
    return store.create_persistent_host(
        host_id=uuid.uuid4().hex, name=f"{user_id}-computer", user_id=user_id, provider="x"
    )


def test_owner_membership_names_the_current_registration_only(stores: dict[str, Any]) -> None:
    accounts, teams, hosts = stores["accounts"], stores["teams"], stores["hosts"]
    accounts.create_user_with_password("alice", "hash", is_admin=True, team_owner=True)
    accounts.create_user_with_password("bob", "hash")
    accounts.create_user_with_password("carol", "hash")
    team = teams.get_team()
    assert team is not None
    bob = teams.add_member("bob", role="member", state="active")
    alices, bobs, carols = _host(hosts, "alice"), _host(hosts, "bob"), _host(hosts, "carol")

    alice_member = teams.get_member_for_account("alice")
    assert alice_member is not None
    assert hosts.owner_membership("alice", alices.account_generation) == (
        alice_member.id,
        team.id,
    )
    assert hosts.owner_membership("bob", bobs.account_generation) == (bob.id, team.id)
    # No membership: the account alone.
    assert hosts.owner_membership("carol", carols.account_generation) is None
    assert sandbox_owner(carols, hosts) == SandboxOwner(user_id="carol", host_id=carols.host_id)
    # A host recorded under an earlier registration of the account.
    assert hosts.owner_membership("bob", "0" * 32) is None
    # A suspended member is still that member; a removed one is no member.
    teams.change_state(bob.id, "suspended")
    assert hosts.owner_membership("bob", bobs.account_generation) == (bob.id, team.id)
    teams.change_state(bob.id, "removed")
    assert hosts.owner_membership("bob", bobs.account_generation) is None
    # Without a store the account alone is named.
    assert sandbox_owner(alices, None) == SandboxOwner(user_id="alice", host_id=alices.host_id)


def test_launcher_call_names_the_owner_only_to_a_provider_that_asks(
    stores: dict[str, Any],
) -> None:
    accounts, teams, hosts = stores["accounts"], stores["teams"], stores["hosts"]
    accounts.create_user_with_password("alice", "hash", is_admin=True, team_owner=True)
    host = _host(hosts, "alice")
    member = teams.get_member_for_account("alice")
    team = teams.get_team()
    assert member is not None and team is not None
    owned, plain = OwnedComputer(), FakeComputer()
    launcher_call(owned, "resume", host, hosts, "sb-1")
    launcher_call(plain, "resume", host, hosts, "sb-1")  # would raise on an unexpected keyword
    assert owned.owners == [
        (
            "resume",
            SandboxOwner(
                user_id="alice", host_id=host.host_id, member_id=member.id, team_id=team.id
            ),
        )
    ]
    assert plain.events == ["resume"]
    assert owner_for(owned, None, hosts) is None
    assert owner_for(plain, host, hosts) is None


# ── The persistent-host lifecycle ──────────────────────────────────────


@pytest.fixture
async def owned(
    db_uri: str, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[tuple[FakeHost, OwnedComputer, dict[str, str]]]:
    """A configured persistent provider that takes owners, and Alice, a team member."""
    # The fake host process connects as Alice, whose computer it is.
    monkeypatch.setattr(persistent_host_fakes, "OWNER", "alice")
    accounts, teams = SqlAlchemyAccountStore(db_uri), SqlAlchemyTeamStore(db_uri)
    accounts.create_user_with_password("founder", "hash", is_admin=True, team_owner=True)
    accounts.create_user_with_password("alice", "hash")
    member = teams.add_member("alice", role="member", state="active")
    team = teams.get_team()
    assert team is not None
    store, registry = HostStore(db_uri), HostRegistry()
    registry.admission_gate = persistent_hosts.admit_host_work
    registry.send_guard = persistent_hosts.guard_host_request
    computer = OwnedComputer(status="stopped")
    fake = FakeHost(store, registry, computer)
    loop = asyncio.get_running_loop()
    computer.on_host_start = lambda token: call_soon(loop, fake.connect, token)
    persistent_hosts._states.clear()
    persistent_hosts.configure(
        store,
        registry,
        ManagedSandboxDeployment.single(
            ManagedSandboxConfig(
                server_url="http://omnigent.test",
                launcher_factory=lambda: computer,
                token_ttl_s=3600,
                provider=computer.provider,
                persistent=PersistentHostPolicy(idle_sleep_s=0, admission_max_age_s=0),
                online_timeout_s=10,
            )
        ),
    )
    yield fake, computer, {"member": member.id, "team": team.id}
    fake.close()
    for state in persistent_hosts._states.values():
        if state.wake_task is not None:
            state.wake_task.cancel()
    persistent_hosts.configure(None, None, None)
    persistent_hosts._states.clear()


@pytest.mark.asyncio
async def test_every_lifecycle_call_names_the_hosts_owner(
    owned: tuple[FakeHost, OwnedComputer, dict[str, str]],
) -> None:
    fake, computer, ids = owned
    created = await persistent_hosts.create_host(owner="alice", name="Alice's", provider=None)
    fake.host_id = created.host_id
    task = persistent_hosts._states[created.host_id].wake_task
    assert task is not None
    with contextlib.suppress(OmnigentError):
        await asyncio.wait_for(task, 10)
    assert computer.events == ["provision", "resume", "claim", "start_host"]
    alice = SandboxOwner(
        user_id="alice", host_id=created.host_id, member_id=ids["member"], team_id=ids["team"]
    )
    # Asleep, then removed.
    assert await persistent_hosts.sleep_if_idle(created.host_id) == "stopped"
    current = fake.store.get_host(created.host_id)
    assert current is not None
    await persistent_hosts.delete_host(current)
    called = [name for name, _ in computer.owners]
    for method in (
        "provision",
        "admission",
        "resume",
        "claim_start",
        "start_host",
        "suspend",
        "terminate",
    ):
        assert method in called, method
    assert {owner for _, owner in computer.owners} == {alice}


@pytest.mark.asyncio
async def test_a_provider_without_the_capability_is_called_as_before(
    owned: tuple[FakeHost, OwnedComputer, dict[str, str]],
) -> None:
    """The same lifecycle on a provider that never declared it: no keyword, no failure."""
    fake, computer, _ = owned
    plain = FakeComputer(status="stopped")
    plain.on_host_start = computer.on_host_start
    persistent_hosts.configure(
        fake.store,
        fake.registry,
        ManagedSandboxDeployment.single(
            ManagedSandboxConfig(
                server_url="http://omnigent.test",
                launcher_factory=lambda: plain,
                token_ttl_s=3600,
                provider=plain.provider,
                persistent=PersistentHostPolicy(idle_sleep_s=0, admission_max_age_s=0),
                online_timeout_s=10,
            )
        ),
    )
    created = await persistent_hosts.create_host(owner="alice", name="Plain", provider=None)
    fake.host_id = created.host_id
    task = persistent_hosts._states[created.host_id].wake_task
    assert task is not None
    await asyncio.wait_for(task, 10)
    assert plain.events == ["provision", "resume", "claim", "start_host"]
    assert computer.owners == []


# ── The reaper's removals ──────────────────────────────────────────────


class _OwnedRecordingLauncher(OwnedComputer):
    @contextlib.contextmanager
    def reaper_identity(self, workspace_id: int) -> Any:
        del workspace_id
        yield


@pytest.mark.asyncio
async def test_the_reaper_names_the_owner_of_a_removed_hosts_sandbox(
    stores: dict[str, Any],
) -> None:
    accounts, teams, hosts = stores["accounts"], stores["teams"], stores["hosts"]
    accounts.create_user_with_password("alice", "hash", is_admin=True, team_owner=True)
    member, team = teams.get_member_for_account("alice"), teams.get_team()
    assert member is not None and team is not None
    host = hosts.create_persistent_host(
        host_id=uuid.uuid4().hex, name="alice-computer", user_id="alice", provider="fakepersist"
    )
    attached = hosts.attach_persistent_sandbox(
        host.host_id, provider="fakepersist", sandbox_id="sb-a"
    )
    assert attached is not None
    hosts.delete_host(host.host_id)
    launcher = _OwnedRecordingLauncher()
    reaper = ManagedSandboxReaper(
        host_store=hosts,
        sandbox_config=ManagedSandboxDeployment(
            configs=(
                ManagedSandboxConfig(
                    server_url="http://omnigent.test",
                    launcher_factory=lambda: launcher,
                    token_ttl_s=3600,
                    provider="fakepersist",
                    persistent=PersistentHostPolicy(),
                ),
            ),
            reaper=ManagedSandboxReaperConfig(
                enabled=True, terminate_after_offline_days=1, sweep_interval_s=86400
            ),
        ),
    )
    assert await reaper.sweep_once(now=now_epoch()) == 1
    assert launcher.owners == [
        (
            "terminate",
            SandboxOwner(
                user_id="alice", host_id=host.host_id, member_id=member.id, team_id=team.id
            ),
        )
    ]


# ── Starts without start claims (Codex WP8.4 r1-3) ─────────────────────


class _NoClaimsOwned(SandboxHostLauncher):
    """``owner_context`` without start claims: its start requires the owner keyword.

    :ivar starts: Each start's keyword arguments, as received.
    """

    provider: ClassVar[str] = "fakeowned"
    supports_owner_context: ClassVar[bool] = True
    classifies: ClassVar[bool] = False

    def __init__(self) -> None:
        self.starts: list[dict[str, object]] = []

    @property
    def capabilities(self) -> SandboxCapabilities:
        return dataclasses.replace(
            super().capabilities, classifies_runner_by_agent=type(self).classifies
        )

    def prepare(self) -> None:
        return None

    def provision(self, name: str, *, owner: SandboxOwner | None = None) -> str:
        del name, owner
        return "sb-owned"

    def start_host(  # type: ignore[override]
        self, sandbox_id: str, *, owner: SandboxOwner, **kwargs: object
    ) -> str:
        del sandbox_id
        self.starts.append({"owner": owner, **kwargs})
        return "/root/workspace"


class _NoClaimsClassifying(_NoClaimsOwned):
    classifies: ClassVar[bool] = True


class _NoClaimsPlain(SandboxHostLauncher):
    """No ``owner_context``: a start that names an owner fails here."""

    provider: ClassVar[str] = "fakeplain"

    def __init__(self) -> None:
        self.starts: list[dict[str, object]] = []

    def prepare(self) -> None:
        return None

    def provision(self, name: str) -> str:
        del name
        return "sb-plain"

    def start_host(self, sandbox_id: str, **kwargs: object) -> str:  # type: ignore[override]
        del sandbox_id
        assert "owner" not in kwargs
        self.starts.append(dict(kwargs))
        return "/root/workspace"


_ALICE = SandboxOwner(user_id="alice", host_id="h" * 32, member_id="m" * 32, team_id="t" * 32)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("launcher_type", "host_config", "on_stage", "agent_name", "start_attempt", "expected"),
    [
        pytest.param(_NoClaimsOwned, None, None, None, None, set(), id="bare"),
        pytest.param(_NoClaimsOwned, None, print, None, None, {"on_stage"}, id="stage"),
        pytest.param(_NoClaimsOwned, {"a": 1}, None, None, None, {"host_config"}, id="config"),
        pytest.param(
            _NoClaimsOwned, {"a": 1}, print, None, None, {"host_config", "on_stage"}, id="both"
        ),
        # A claimed attempt reaches only a launcher with start claims: here it is dropped, and
        # the owner still goes through.
        pytest.param(_NoClaimsOwned, None, None, None, 3, set(), id="attempt-without-claims"),
        pytest.param(
            _NoClaimsClassifying,
            None,
            None,
            "coder",
            None,
            {"host_config", "on_stage", "agent_name"},
            id="classified",
        ),
    ],
)
async def test_every_start_names_the_owner_to_a_provider_without_start_claims(
    launcher_type: type[_NoClaimsOwned],
    host_config: dict[str, object] | None,
    on_stage: Any,
    agent_name: str | None,
    start_attempt: int | None,
    expected: set[str],
) -> None:
    from omnigent.server.managed_hosts import _start_sandbox_host

    launcher = launcher_type()
    assert launcher.capabilities.owner_context and not launcher.capabilities.start_claims
    await _start_sandbox_host(
        launcher,
        "sb-owned",
        token="tok",
        host_id="h" * 32,
        host_name="Alice's",
        server_url="http://omnigent.test",
        repos=(),
        host_config=host_config,
        agent_name=agent_name,
        on_stage=on_stage,
        start_attempt=start_attempt,
        owner=_ALICE,
    )
    (start,) = launcher.starts
    assert start["owner"] == _ALICE
    optional = {"host_config", "on_stage", "agent_name", "start_attempt"}
    assert {key for key in start if key in optional} == expected


@pytest.mark.asyncio
async def test_a_provider_without_owner_context_never_receives_the_keyword() -> None:
    from omnigent.server.managed_hosts import _start_sandbox_host

    launcher = _NoClaimsPlain()
    await _start_sandbox_host(
        launcher,
        "sb-plain",
        token="tok",
        host_id="h" * 32,
        host_name="Plain",
        server_url="http://omnigent.test",
        repos=(),
        host_config={"a": 1},
        owner=_ALICE,
    )
    assert launcher.starts and "owner" not in launcher.starts[0]


@dataclass
class _OwnedUntypedComputer(FakeUntypedComputer):
    """An untyped persistent provider (no admission, no start claims) that takes owners."""

    supports_owner_context: ClassVar[bool] = True
    owners: list[tuple[str, SandboxOwner | None]] = field(default_factory=list)

    def provision(self, name: str, *, owner: SandboxOwner | None = None) -> str:
        self.owners.append(("provision", owner))
        return super().provision(name)

    def is_running(self, sandbox_id: str, *, owner: SandboxOwner | None = None) -> bool | None:
        self.owners.append(("is_running", owner))
        return super().is_running(sandbox_id)

    def resume(self, sandbox_id: str, *, owner: SandboxOwner | None = None) -> None:
        self.owners.append(("resume", owner))
        super().resume(sandbox_id)

    def start_host(  # type: ignore[override]
        self,
        sandbox_id: str,
        *,
        token: str,
        host_id: str,
        host_name: str,
        server_url: str,
        repos: Sequence[RepoWorkspace] = (),
        host_config: dict[str, object] | None = None,
        on_stage: Any = None,
        owner: SandboxOwner,
    ) -> str:
        self.owners.append(("start_host", owner))
        return super().start_host(
            sandbox_id,
            token=token,
            host_id=host_id,
            host_name=host_name,
            server_url=server_url,
            repos=repos,
            host_config=host_config,
            on_stage=on_stage,
        )


@pytest.mark.asyncio
async def test_a_persistent_start_without_start_claims_names_the_owner(
    owned: tuple[FakeHost, OwnedComputer, dict[str, str]],
) -> None:
    """The untyped wake: no admission, no claim, a start for no attempt; Alice named throughout."""
    fake, _computer, ids = owned
    untyped = _OwnedUntypedComputer()
    loop = asyncio.get_running_loop()
    untyped.on_host_start = lambda token: call_soon(loop, fake.connect, token)
    assert not untyped.capabilities.start_claims and untyped.capabilities.owner_context
    persistent_hosts.configure(
        fake.store,
        fake.registry,
        ManagedSandboxDeployment.single(
            ManagedSandboxConfig(
                server_url="http://omnigent.test",
                launcher_factory=lambda: untyped,
                token_ttl_s=3600,
                provider=untyped.provider,
                persistent=PersistentHostPolicy(),
                online_timeout_s=10,
            )
        ),
    )
    created = await persistent_hosts.create_host(owner="alice", name="Untyped", provider=None)
    fake.host_id = created.host_id
    task = persistent_hosts._states[created.host_id].wake_task
    assert task is not None
    await asyncio.wait_for(task, 10)
    assert untyped.events == ["provision", "start_host"]
    alice = SandboxOwner(
        user_id="alice", host_id=created.host_id, member_id=ids["member"], team_id=ids["team"]
    )
    assert ("start_host", alice) in untyped.owners
    assert {owner for _, owner in untyped.owners} == {alice}
