"""Team membership, identity binding and actual-ownership lookups.

Runs on SQLite by default and on the configured server database when
``OMNIGENT_TEST_DB_URI`` is set, including the concurrent-writer races.
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Callable
from typing import Any

import pytest

from omnigent.db.account_authority import AccountAuthority
from omnigent.entities.team import account_admin_flag
from omnigent.errors import ErrorCode, OmnigentError
from omnigent.server.accounts_store import SqlAlchemyAccountStore
from omnigent.server.auth import LEVEL_EDIT, LEVEL_MANAGE, LEVEL_OWNER, LEVEL_READ
from omnigent.stores.conversation_store.sqlalchemy_store import SqlAlchemyConversationStore
from omnigent.stores.host_store import HostStore
from omnigent.stores.permission_store.sqlalchemy_store import SqlAlchemyPermissionStore
from omnigent.stores.team_store import (
    MAX_IDENTITIES_PER_MEMBER,
    IdentityBindingConflictError,
    LastActiveOwnerError,
    SqlAlchemyTeamStore,
)

_ISSUER = "https://team.example.cloudflareaccess.com"
_AUDIENCE = "aud-tag-1"


@pytest.fixture
def stores(db_uri: str) -> dict[str, Any]:
    return {
        "teams": SqlAlchemyTeamStore(db_uri),
        "accounts": SqlAlchemyAccountStore(db_uri),
        "permissions": SqlAlchemyPermissionStore(db_uri),
        "conversations": SqlAlchemyConversationStore(db_uri),
        "hosts": HostStore(db_uri),
    }


def _account(stores: dict[str, Any], user_id: str, *, is_admin: bool = False) -> None:
    stores["accounts"].create_user_with_password(user_id, "hash", is_admin=is_admin)


def _founder(stores: dict[str, Any], user_id: str = "owner") -> str:
    stores["accounts"].create_user_with_password(user_id, "hash", is_admin=True, team_owner=True)
    member = stores["teams"].get_member_for_account(user_id)
    assert member is not None and (member.role, member.state) == ("owner", "active")
    return member.id


def _race(*actions: Callable[[], object]) -> list[object]:
    """Run *actions* at the same moment; return each result or exception."""
    barrier = threading.Barrier(len(actions))
    results: list[object] = [None] * len(actions)

    def run(index: int) -> None:
        barrier.wait()
        try:
            results[index] = actions[index]()
        except Exception as exc:
            results[index] = exc

    threads = [threading.Thread(target=run, args=(index,)) for index in range(len(actions))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    return results


# ── Membership and the account admin flag ──────────────────────────


@pytest.mark.parametrize(
    ("role", "state", "flag"),
    [
        ("owner", "active", True),
        ("admin", "active", True),
        ("member", "active", False),
        ("admin", "invited", False),
        ("owner", "suspended", False),
        ("admin", "removed", False),
    ],
)
def test_account_admin_flag_follows_active_owner_and_admin_roles(role, state, flag) -> None:
    assert account_admin_flag(role, state) is flag


def test_first_admin_founds_the_team_and_later_accounts_join_explicitly(stores) -> None:
    owner_id = _founder(stores)
    team = stores["teams"].get_team()
    assert team is not None and len(team.id) == 32
    # A second first-admin claim cannot take over a team that has members.
    _account(stores, "bob", is_admin=True)
    stores["accounts"].delete_user("bob")
    stores["accounts"].create_user_with_password("bob", "hash", is_admin=True, team_owner=True)
    assert stores["teams"].get_member_for_account("bob") is None

    member = stores["teams"].add_member("bob", role="member", state="active")
    assert member.id not in (owner_id, "") and member.user_id == "bob"
    assert stores["accounts"].is_admin("bob") is False
    page, cursor = stores["teams"].list_members(limit=1)
    assert [m.user_id for m in page] == ["bob"] and cursor == "bob"
    page, cursor = stores["teams"].list_members(limit=1, after_user_id=cursor)
    assert [m.user_id for m in page] == ["owner"] and cursor is None


@pytest.mark.parametrize(
    ("user_id", "code"),
    [("local", ErrorCode.INVALID_INPUT), ("__public__", ErrorCode.INVALID_INPUT)],
)
def test_reserved_identities_cannot_join(stores, user_id, code) -> None:
    with pytest.raises(OmnigentError) as exc:
        stores["teams"].add_member(user_id, role="member")
    assert exc.value.code == code


def test_membership_requires_a_live_account_and_is_unique_per_account(stores) -> None:
    _founder(stores)
    with pytest.raises(OmnigentError) as exc:
        stores["teams"].add_member("nobody", role="member")
    assert exc.value.code == ErrorCode.INVALID_INPUT
    _account(stores, "bob")
    stores["teams"].add_member("bob", role="member")
    with pytest.raises(OmnigentError) as exc:
        stores["teams"].add_member("bob", role="admin")
    assert exc.value.code == ErrorCode.CONFLICT


def test_role_and_state_changes_keep_the_admin_flag_in_step(stores) -> None:
    _founder(stores)
    _account(stores, "bob")
    teams, accounts = stores["teams"], stores["accounts"]
    bob = teams.add_member("bob", role="admin")
    assert accounts.is_admin("bob") is False  # invited

    bob = teams.change_state(bob.id, "active")
    assert accounts.is_admin("bob") is True
    bob = teams.change_state(bob.id, "suspended")
    assert accounts.is_admin("bob") is False
    bob = teams.change_state(bob.id, "active")
    bob = teams.change_role(bob.id, "member")
    assert accounts.is_admin("bob") is False
    bob = teams.change_role(bob.id, "owner")
    assert accounts.is_admin("bob") is True

    removed = teams.change_state(bob.id, "removed")
    assert (removed.state, removed.is_current) == ("removed", False)
    assert accounts.is_admin("bob") is False
    assert accounts.get_user("bob") is not None  # removal does not delete the account
    assert teams.get_member_for_account("bob") is None
    assert teams.get_member(bob.id) == removed


@pytest.mark.parametrize(
    ("start", "target"),
    [
        ("invited", "suspended"),
        ("active", "invited"),
        ("suspended", "invited"),
        ("removed", "active"),
    ],
)
def test_lifecycle_refuses_other_transitions(stores, start, target) -> None:
    _founder(stores)
    _account(stores, "bob")
    teams = stores["teams"]
    member = teams.add_member(
        "bob", role="member", state="invited" if start == "invited" else "active"
    )
    if start in ("suspended", "removed"):
        member = teams.change_state(member.id, start)
    with pytest.raises(OmnigentError) as exc:
        teams.change_state(member.id, target)
    assert exc.value.code == ErrorCode.CONFLICT
    assert teams.change_state(member.id, start) == member  # the same state is a no-op


def test_unknown_member_is_not_found_and_a_removed_one_cannot_change(stores) -> None:
    with pytest.raises(OmnigentError) as exc:
        stores["teams"].change_role(uuid.uuid4().hex, "admin")
    assert exc.value.code == ErrorCode.NOT_FOUND
    _founder(stores)
    _account(stores, "bob")
    bob = stores["teams"].add_member("bob", role="member", state="active")
    assert stores["accounts"].delete_user("bob") is True
    with pytest.raises(OmnigentError) as exc:
        stores["teams"].change_role(bob.id, "admin")
    assert exc.value.code == ErrorCode.CONFLICT
    assert stores["teams"].change_state(bob.id, "removed").state == "removed"


# ── At least one active owner ──────────────────────────────────────


@pytest.mark.parametrize(
    "change",
    [
        lambda teams, member_id: teams.change_role(member_id, "admin"),
        lambda teams, member_id: teams.change_state(member_id, "suspended"),
        lambda teams, member_id: teams.change_state(member_id, "removed"),
    ],
    ids=["demote", "suspend", "remove"],
)
def test_the_last_active_owner_stays(stores, change) -> None:
    owner_id = _founder(stores)
    _account(stores, "bob")
    # An invited or suspended owner does not count.
    bob = stores["teams"].add_member("bob", role="owner")
    with pytest.raises(LastActiveOwnerError):
        change(stores["teams"], owner_id)
    stores["teams"].change_state(bob.id, "active")
    change(stores["teams"], owner_id)
    assert stores["teams"].get_member(bob.id).role == "owner"


def test_deleting_an_account_removes_its_membership_but_never_the_last_owner(stores) -> None:
    owner_id = _founder(stores)
    _account(stores, "second-admin", is_admin=True)
    accounts, teams = stores["accounts"], stores["teams"]
    # Another admin flag is not another owner.
    assert accounts.delete_user("owner") is False
    assert teams.get_member(owner_id).state == "active"

    _account(stores, "bob")
    bob = teams.add_member("bob", role="owner", state="active")
    assert accounts.delete_user("owner") is True
    assert teams.get_member(owner_id).state == "removed"
    assert teams.get_member(bob.id).state == "active"


@pytest.mark.parametrize("round_number", range(2))
@pytest.mark.parametrize("operation", ["demote", "suspend", "remove", "delete_account"])
def test_concurrent_changes_cannot_remove_both_owners(stores, operation, round_number) -> None:
    """Two writers each see the other owner; the team lock lets only one proceed."""
    owner_id = _founder(stores)
    _account(stores, "bob")
    bob_id = stores["teams"].add_member("bob", role="owner", state="active").id
    teams, accounts = stores["teams"], stores["accounts"]

    def act(member_id: str, user_id: str) -> Callable[[], object]:
        return {
            "demote": lambda: teams.change_role(member_id, "member"),
            "suspend": lambda: teams.change_state(member_id, "suspended"),
            "remove": lambda: teams.change_state(member_id, "removed"),
            "delete_account": lambda: accounts.delete_user(user_id),
        }[operation]

    results = _race(act(owner_id, "owner"), act(bob_id, "bob"))

    if operation == "delete_account":
        assert sorted(results, key=str) == [False, True]
    else:
        assert sum(isinstance(result, LastActiveOwnerError) for result in results) == 1, results
    active_owners = [
        member
        for member in (teams.get_member(owner_id), teams.get_member(bob_id))
        if (member.role, member.state) == ("owner", "active")
    ]
    assert len(active_owners) == 1


def test_concurrent_memberships_for_one_account_converge_to_one(stores) -> None:
    _founder(stores)
    _account(stores, "bob")
    results = _race(
        lambda: stores["teams"].add_member("bob", role="member"),
        lambda: stores["teams"].add_member("bob", role="admin"),
    )
    conflicts = [r for r in results if isinstance(r, OmnigentError)]
    assert len(conflicts) == 1 and conflicts[0].code == ErrorCode.CONFLICT
    members, _ = stores["teams"].list_members()
    assert [member.user_id for member in members] == ["bob", "owner"]


def test_team_roles_win_over_admin_flag_writes(stores) -> None:
    """A login-time roster or a flag write never changes a member's role (WP8.3 F-5)."""
    _founder(stores)
    _account(stores, "bob")
    _account(stores, "carol")
    teams, accounts, permissions = stores["teams"], stores["accounts"], stores["permissions"]
    bob = teams.add_member("bob", role="member", state="active")

    accounts.set_admin("bob", True)
    assert teams.get_member(bob.id).role == "member" and not accounts.is_admin("bob")
    teams.change_role(bob.id, "admin")
    permissions.set_admin("bob", False)
    assert teams.get_member(bob.id).role == "admin" and accounts.is_admin("bob")
    accounts.set_admin("owner", False)
    assert accounts.is_admin("owner")
    # A suspended member gains no admin flag, and an account outside the team keeps its flag.
    teams.change_state(bob.id, "suspended")
    accounts.set_admin("bob", True)
    assert not accounts.is_admin("bob")
    accounts.set_admin("carol", True)
    assert accounts.is_admin("carol") and teams.get_member_for_account("carol") is None


def test_a_roster_promotion_cannot_reverse_an_owners_demotion(stores, tmp_path) -> None:
    from omnigent.server.admin_list import AdminList, promote_if_listed

    _founder(stores)
    _account(stores, "bob")
    teams, accounts = stores["teams"], stores["accounts"]
    bob = teams.add_member("bob", role="admin", state="active")
    teams.change_role(bob.id, "member")
    roster = AdminList(tmp_path / "admins", extra=frozenset({"bob"}))
    assert promote_if_listed(roster, accounts, "bob") is False
    assert teams.get_member(bob.id).role == "member" and not accounts.is_admin("bob")


# ── External identities ────────────────────────────────────────────


def _bind(stores: dict[str, Any], member_id: str, subject: str = "subject-1", **kwargs: Any):
    values = {"issuer": _ISSUER, "subject": subject, "audience": _AUDIENCE}
    values.update(kwargs)
    return stores["teams"].bind_external_identity(
        member_id, provenance=values.pop("provenance", "invitation"), **values
    )


def test_identity_binds_looks_up_and_unbinds_by_issuer_and_subject(stores) -> None:
    owner_id = _founder(stores)
    teams = stores["teams"]
    binding = _bind(stores, owner_id, provenance="migration")
    assert (binding.member_id, binding.provenance, binding.bound_by_member_id) == (
        owner_id,
        "migration",
        None,
    )
    assert teams.get_external_identity(_ISSUER, "subject-1") == binding
    assert teams.member_for_external_identity(_ISSUER, "subject-1", audience=_AUDIENCE).id == (
        owner_id
    )
    # The audience recorded at binding is required; another issuer is another identity.
    assert teams.member_for_external_identity(_ISSUER, "subject-1", audience="other") is None
    assert (
        teams.member_for_external_identity(_ISSUER + "/", "subject-1", audience=_AUDIENCE) is None
    )
    assert teams.list_external_identities(owner_id) == [binding]
    assert teams.unbind_external_identity(_ISSUER, "subject-1") is True
    assert teams.unbind_external_identity(_ISSUER, "subject-1") is False
    assert teams.member_for_external_identity(_ISSUER, "subject-1", audience=_AUDIENCE) is None


def test_rebinding_converges_and_a_different_binding_conflicts(stores) -> None:
    owner_id = _founder(stores)
    _account(stores, "bob")
    bob = stores["teams"].add_member("bob", role="member", state="active")
    first = _bind(stores, owner_id)
    assert _bind(stores, owner_id) == first
    with pytest.raises(IdentityBindingConflictError):
        _bind(stores, bob.id)
    with pytest.raises(IdentityBindingConflictError):
        _bind(stores, owner_id, audience="another-application")


@pytest.mark.parametrize("round_number", range(3))
def test_concurrent_binds_of_one_identity_converge_or_conflict(stores, round_number) -> None:
    owner_id = _founder(stores)
    _account(stores, "bob")
    bob = stores["teams"].add_member("bob", role="member", state="active")

    same = _race(
        lambda: _bind(stores, owner_id, "s-same"), lambda: _bind(stores, owner_id, "s-same")
    )
    assert not any(isinstance(result, Exception) for result in same), same
    assert same[0] == same[1]

    different = _race(
        lambda: _bind(stores, owner_id, "s-two"), lambda: _bind(stores, bob.id, "s-two")
    )
    assert sum(isinstance(r, IdentityBindingConflictError) for r in different) == 1, different
    winner = stores["teams"].get_external_identity(_ISSUER, "s-two")
    assert winner is not None and winner.member_id in (owner_id, bob.id)
    assert [b for b in different if not isinstance(b, Exception)] == [winner]


def test_binding_needs_a_current_member_and_is_capped(stores) -> None:
    owner_id = _founder(stores)
    with pytest.raises(OmnigentError) as exc:
        _bind(stores, uuid.uuid4().hex)
    assert exc.value.code == ErrorCode.NOT_FOUND
    for index in range(MAX_IDENTITIES_PER_MEMBER):
        _bind(stores, owner_id, f"subject-{index}")
    with pytest.raises(OmnigentError) as exc:
        _bind(stores, owner_id, "one-too-many")
    assert exc.value.code == ErrorCode.CONFLICT
    with pytest.raises(OmnigentError) as exc:
        _bind(stores, owner_id, "x" * 256)
    assert exc.value.code == ErrorCode.INVALID_INPUT


def test_email_is_never_an_identity_key(stores) -> None:
    """A recycled email reaches a new subject, which no member is bound to."""
    stores["accounts"].create_user_with_password(
        "alice@example.com", "hash", is_admin=True, team_owner=True
    )
    alice = stores["teams"].get_member_for_account("alice@example.com")
    _bind(stores, alice.id, "original-person")
    teams = stores["teams"]
    # The provider renamed the person's email: the subject, and so the member, is unchanged.
    assert (
        teams.member_for_external_identity(_ISSUER, "original-person", audience=_AUDIENCE).id
        == alice.id
    )
    # Someone else now holds the old email: different subject, no member, no automatic link.
    assert teams.member_for_external_identity(_ISSUER, "new-person", audience=_AUDIENCE) is None
    assert (
        teams.member_for_external_identity(_ISSUER, "alice@example.com", audience=_AUDIENCE)
        is None
    )


def test_username_reuse_starts_a_new_member_without_old_bindings(stores) -> None:
    _founder(stores)
    _account(stores, "alice")
    teams, accounts = stores["teams"], stores["accounts"]
    old = teams.add_member("alice", role="admin", state="active")
    _bind(stores, old.id, "alice-subject")
    old_generation = accounts.get_user("alice").account_generation

    assert accounts.delete_user("alice") is True
    assert teams.get_member(old.id).state == "removed"
    assert teams.list_external_identities(old.id) == []
    assert teams.member_for_external_identity(_ISSUER, "alice-subject", audience=_AUDIENCE) is None
    with pytest.raises(OmnigentError):
        _bind(stores, old.id, "alice-subject")

    _account(stores, "alice")
    assert teams.get_member_for_account("alice") is None
    assert not accounts.is_admin("alice")
    new = teams.add_member("alice", role="member", state="active")
    assert new.id != old.id
    assert teams.member_for_owner(AccountAuthority("alice", old_generation, 0)) is None
    current = accounts.get_user("alice").account_generation
    assert teams.member_for_owner(AccountAuthority("alice", current, 0)).id == new.id


# ── Actual ownership ───────────────────────────────────────────────


def test_thread_owner_is_the_one_owner_grant_never_a_share_or_admin(stores) -> None:
    _founder(stores)
    _account(stores, "alice")
    _account(stores, "bob")
    conversations, permissions = stores["conversations"], stores["permissions"]
    thread = conversations.create_conversation(agent_id=None)
    for user_id, level in (
        ("bob", LEVEL_EDIT),
        ("owner", LEVEL_MANAGE),
        ("__public__", LEVEL_READ),
    ):
        permissions.grant(user_id, thread.id, level)
    # Only shares, plus an administrator who can see everything: no owner.
    assert conversations.get_actual_session_owner(thread.id) is None
    assert conversations.get_session_owner(thread.id) == "owner"  # the fallback this replaces

    permissions.grant("alice", thread.id, LEVEL_OWNER)
    owner = conversations.get_actual_session_owner(thread.id)
    alice = stores["accounts"].get_user("alice")
    assert owner == AccountAuthority("alice", alice.account_generation, 0)
    child = conversations.create_conversation(
        kind="sub_agent", parent_conversation_id=thread.id, sub_agent_name="helper"
    )
    assert conversations.get_actual_session_owner(child.id) == owner

    permissions.grant("bob", thread.id, LEVEL_OWNER)
    assert conversations.get_actual_session_owner(thread.id) is None  # ambiguous
    assert conversations.get_actual_session_owner(uuid.uuid4().hex) is None


def test_deleted_owner_owns_nothing(stores) -> None:
    _founder(stores)
    _account(stores, "alice")
    thread = stores["conversations"].create_conversation(agent_id=None)
    stores["permissions"].grant("alice", thread.id, LEVEL_OWNER)
    host = uuid.uuid4().hex
    stores["hosts"].upsert_on_connect(host, "alice-laptop", "alice")
    assert stores["hosts"].get_actual_host_owner(host).user_id == "alice"

    stores["accounts"].delete_user("alice")
    _account(stores, "alice")
    assert stores["conversations"].get_actual_session_owner(thread.id) is None
    assert stores["hosts"].get_actual_host_owner(host) is None


def test_host_owner_requires_the_current_registration(stores) -> None:
    _founder(stores)
    _account(stores, "alice")
    hosts = stores["hosts"]
    host = uuid.uuid4().hex
    hosts.upsert_on_connect(host, "alice-laptop", "alice")
    generation = stores["accounts"].get_user("alice").account_generation
    assert hosts.get_actual_host_owner(host) == AccountAuthority("alice", generation, 0)
    member = stores["teams"].add_member("alice", role="member", state="active")
    assert stores["teams"].member_for_owner(hosts.get_actual_host_owner(host)).id == member.id

    # A password reset that revokes sessions carries the host to the new generation.
    new_generation = stores["accounts"].update_password("alice", "hash-2", revoke_sessions=True)
    assert hosts.get_actual_host_owner(host) == AccountAuthority("alice", new_generation, 0)
    external = uuid.uuid4().hex
    hosts.upsert_on_connect(external, "proxy-laptop", "dana@example.com")
    assert hosts.get_actual_host_owner(external) == AccountAuthority("dana@example.com", None, 0)
    assert hosts.get_actual_host_owner(uuid.uuid4().hex) is None


# ── Inventory ──────────────────────────────────────────────────────


def _connection(db_uri: str, user_id: str, provider: str, secret: str) -> None:
    from sqlalchemy.orm import Session

    from omnigent.db.db_models import SqlConnection
    from omnigent.db.utils import get_or_create_engine

    with Session(get_or_create_engine(db_uri)) as session, session.begin():
        session.add(
            SqlConnection(
                user_id=user_id,
                provider=provider,
                account_id="",
                secret_enc=secret,
                metadata_json='{"login": "' + secret + '"}',
                created_at=1,
                updated_at=1,
            )
        )


@pytest.mark.parametrize("page_size", [1, 500])
def test_inventory_counts_owners_and_flags_what_it_cannot_assign(
    stores, db_uri: str, page_size: int
) -> None:
    from omnigent.stores.scheduled_task_store.sqlalchemy_store import (
        SqlAlchemyScheduledTaskStore,
    )

    owner_id = _founder(stores)
    _account(stores, "bob")
    conversations, permissions, hosts = (
        stores["conversations"],
        stores["permissions"],
        stores["hosts"],
    )

    def thread(*grants: tuple[str, int]) -> str:
        conversation = conversations.create_conversation(agent_id=None)
        for user_id, level in grants:
            permissions.grant(user_id, conversation.id, level)
        return conversation.id

    thread(("owner", LEVEL_OWNER), ("bob", LEVEL_EDIT))
    owned_thread = thread(("owner", LEVEL_OWNER))
    conversations.create_conversation(
        kind="sub_agent", parent_conversation_id=owned_thread, sub_agent_name="helper"
    )
    thread(("bob", LEVEL_OWNER))
    unowned = thread(("bob", LEVEL_EDIT))
    contested = thread(("owner", LEVEL_OWNER), ("bob", LEVEL_OWNER))
    legacy = thread(("local", LEVEL_OWNER))

    owner_host, bob_host, ghost_host = (uuid.uuid4().hex for _ in range(3))
    hosts.upsert_on_connect(owner_host, "owner-laptop", "owner")
    hosts.upsert_on_connect(bob_host, "bob-laptop", "bob")
    hosts.upsert_on_connect(ghost_host, "ghost-laptop", "ghost")
    _connection(db_uri, "owner", "github", "SECRET-CANARY-1")
    _connection(db_uri, "ghost", "github", "SECRET-CANARY-2")

    tasks = SqlAlchemyScheduledTaskStore(db_uri)
    agent_id = uuid.uuid4().hex
    tasks.create(
        uuid.uuid4().hex, "nightly", "PROMPT-CANARY", "FREQ=DAILY", "owner", agent_id, "UTC"
    )
    pinned_elsewhere = uuid.uuid4().hex
    tasks.create(
        pinned_elsewhere, "pinned", "p", "FREQ=DAILY", "owner", agent_id, "UTC", host_id=bob_host
    )
    single_user = uuid.uuid4().hex
    tasks.create(single_user, "local", "p", "FREQ=DAILY", None, agent_id, "UTC")
    retired = uuid.uuid4().hex
    tasks.create(retired, "old", "p", "FREQ=DAILY", None, agent_id, "UTC", state="deleted")

    inventory = stores["teams"].inventory(page_size=page_size)

    assert inventory.team_id == stores["teams"].get_team().id
    assert [(a.user_id, a.member_id, a.role) for a in inventory.accounts] == [
        ("bob", None, None),
        ("owner", owner_id, "owner"),
    ]
    assert inventory.thread_owners == {"owner": 2, "bob": 1}
    assert inventory.host_owners == {"owner": 1, "bob": 1}
    assert inventory.connection_owners == {"owner": 1}
    assert inventory.schedule_owners == {"owner": 1}
    assert {(f.resource, f.resource_id, f.reason, f.owners) for f in inventory.findings} == {
        ("account", "bob", "not_member", ("bob",)),
        ("thread", unowned, "no_owner", ()),
        ("thread", contested, "multiple_owners", ("bob", "owner")),
        ("thread", legacy, "legacy_local_owner", ("local",)),
        ("host", ghost_host, "unknown_owner", ("ghost",)),
        ("connection", "ghost/github/", "unknown_owner", ("ghost",)),
        ("schedule", pinned_elsewhere, "host_not_owned", ("owner",)),
        ("schedule", single_user, "no_owner", ()),
    }
    assert not inventory.clean
    # Identifiers and counts only: no password hashes, secrets, metadata or prompts.
    rendered = repr(inventory)
    for canary in ("hash", "SECRET-CANARY", "PROMPT-CANARY"):
        assert canary not in rendered


def test_inventory_reports_a_team_without_an_active_owner(stores) -> None:
    _account(stores, "alice", is_admin=True)
    _account(stores, "bob", is_admin=True)
    inventory = stores["teams"].inventory()
    assert ("team", "0", "no_active_owner", ("alice", "bob")) in {
        (f.resource, f.resource_id, f.reason, f.owners) for f in inventory.findings
    }

    member = stores["teams"].add_member("alice", role="owner", state="active")
    stores["accounts"].set_admin("bob", False)
    stores["teams"].add_member("bob", role="member", state="active")
    inventory = stores["teams"].inventory()
    assert inventory.clean, inventory.findings
    assert inventory.accounts[0].member_id == member.id


# ── Removal is registration-specific ───────────────────────────────


@pytest.mark.parametrize("store", ["accounts", "permissions"])
def test_admin_flag_writes_cannot_restore_a_removed_registration(stores, store) -> None:
    """A removed member stays non-admin through either store until re-admitted."""
    _founder(stores)
    _account(stores, "bob")
    teams, accounts = stores["teams"], stores["accounts"]
    bob = teams.add_member("bob", role="admin", state="active")
    teams.change_state(bob.id, "removed")

    stores[store].set_admin("bob", True)
    assert not accounts.is_admin("bob")
    # Rotating the password carries the removal to the new generation.
    accounts.update_password("bob", "hash-2", revoke_sessions=True)
    stores[store].set_admin("bob", True)
    assert not accounts.is_admin("bob")
    assert teams.get_member(bob.id).state == "removed"

    again = teams.add_member("bob", role="admin", state="active")
    assert accounts.is_admin("bob") and teams.get_member(again.id).role == "admin"
    # The team's role decides from then on; a flag write changes nothing.
    stores[store].set_admin("bob", False)
    assert accounts.is_admin("bob")


def test_a_later_registration_of_the_username_is_not_held_by_an_old_removal(stores) -> None:
    _founder(stores)
    _account(stores, "bob")
    teams, accounts = stores["teams"], stores["accounts"]
    bob = teams.add_member("bob", role="admin", state="active")
    teams.change_state(bob.id, "removed")
    assert accounts.delete_user("bob") is True
    _account(stores, "bob")

    accounts.set_admin("bob", True)
    assert accounts.is_admin("bob")
    assert teams.get_member_for_account("bob") is None


def test_a_removed_identity_stays_removed_when_it_first_gets_a_password(stores) -> None:
    """A passwordless identity provisioned with credentials is the same registration."""
    _founder(stores)
    teams, accounts, permissions = stores["teams"], stores["accounts"], stores["permissions"]
    permissions.ensure_user("erin@example.com")
    erin = teams.add_member("erin@example.com", role="admin", state="active")
    teams.change_state(erin.id, "removed")
    permissions.set_admin("erin@example.com", True)
    assert not accounts.is_admin("erin@example.com")

    generation = accounts.update_password("erin@example.com", "hash")
    assert generation is not None
    accounts.set_admin("erin@example.com", True)
    assert not accounts.is_admin("erin@example.com")
    # A never-enrolled identity keeps the plain admin-flag behaviour.
    permissions.ensure_user("frank@example.com")
    permissions.set_admin("frank@example.com", True)
    assert accounts.is_admin("frank@example.com")


def test_current_membership_follows_password_rotation(stores) -> None:
    owner_id = _founder(stores)
    accounts, teams = stores["accounts"], stores["teams"]
    new_generation = accounts.update_password("owner", "hash-2", revoke_sessions=True)
    owner = AccountAuthority("owner", new_generation, 0)
    assert teams.member_for_owner(owner).id == owner_id
    assert teams.get_member_for_account("owner").id == owner_id


# ── Inventory and registrations ────────────────────────────────────


def test_inventory_flags_work_recorded_before_the_account_registered(stores, db_uri: str) -> None:
    """External-identity hosts and schedules do not pass to a later password account."""
    from sqlalchemy import update

    from omnigent.db.db_models import SqlHost
    from omnigent.db.utils import get_or_create_engine
    from omnigent.stores.scheduled_task_store.sqlalchemy_store import (
        SqlAlchemyScheduledTaskStore,
    )

    _founder(stores)
    tasks = SqlAlchemyScheduledTaskStore(db_uri)
    agent_id = uuid.uuid4().hex
    external_host = uuid.uuid4().hex
    stores["hosts"].upsert_on_connect(external_host, "alice-laptop", "alice")
    external_schedule = uuid.uuid4().hex
    tasks.create(
        external_schedule, "x", "p", "FREQ=DAILY", "alice", agent_id, "UTC", host_id=external_host
    )
    _account(stores, "alice")
    stores["teams"].add_member("alice", role="member", state="active")
    assert stores["hosts"].get_actual_host_owner(external_host) is None
    # Work recorded under the new registration matches it.
    current_host = uuid.uuid4().hex
    stores["hosts"].upsert_on_connect(current_host, "alice-desktop", "alice")
    tasks.create(
        uuid.uuid4().hex, "y", "p", "FREQ=DAILY", "alice", agent_id, "UTC", host_id=current_host
    )

    # A host still carrying another generation belongs to an earlier registration.
    stale_host = uuid.uuid4().hex
    stores["hosts"].upsert_on_connect(stale_host, "alice-old", "alice")
    with get_or_create_engine(db_uri).begin() as connection:
        connection.execute(
            update(SqlHost)
            .where(SqlHost.host_id == stale_host)
            .values(account_generation="0" * 32)
        )

    inventory = stores["teams"].inventory()

    assert inventory.host_owners == {"alice": 1}
    assert inventory.schedule_owners == {"alice": 1}
    assert {(f.resource, f.resource_id, f.reason) for f in inventory.findings} == {
        ("host", external_host, "missing_registration"),
        ("host", stale_host, "earlier_registration"),
        ("schedule", external_schedule, "missing_registration"),
    }
