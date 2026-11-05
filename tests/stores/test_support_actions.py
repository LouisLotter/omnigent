"""An owner's or admin's support action, authorized inside the write that applies it.

Runs on SQLite by default and on the configured server database when
``OMNIGENT_TEST_DB_URI`` is set. Each write re-reads the team under its lock: the
actor's current role, and the member ID still naming the registration the
request read, in an allowed state.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest

from omnigent.entities.team import SupportAction
from omnigent.errors import ErrorCode, OmnigentError
from omnigent.server.accounts_store import SqlAlchemyAccountStore
from omnigent.stores.credential_store.sqlalchemy_store import CredentialStore
from omnigent.stores.host_store import HostStore
from omnigent.stores.team_store import SqlAlchemyTeamStore
from tests.server.test_credential_store import _FakeCipher
from tests.stores.test_team_member_removal import _member

_ANY = frozenset({"active", "suspended"})
_ACTIVE = frozenset({"active"})


@pytest.fixture
def stores(db_uri: str) -> dict[str, Any]:
    return {
        "teams": SqlAlchemyTeamStore(db_uri),
        "accounts": SqlAlchemyAccountStore(db_uri),
        "hosts": HostStore(db_uri),
    }


def _action(stores: dict[str, Any], actor: str, member: str, states=_ANY) -> SupportAction:
    teams = stores["teams"]
    found = teams.member_registration(user_id=member)
    assert found is not None
    return SupportAction(
        actor_member_id=teams.get_member_for_account(actor).id,
        member_id=found[0].id,
        member_generation=found[1],
        states=states,
    )


def _code(call: Any) -> ErrorCode:
    with pytest.raises(OmnigentError) as caught:
        call()
    return caught.value.code


def _connect(credentials: CredentialStore, user_id: str, *providers: str) -> None:
    for provider in providers:
        credentials.upsert(user_id, provider, secret={"token": "t"}, metadata={})


def test_a_membership_and_its_registration_are_read_together(stores) -> None:
    alice_id = _member(stores, "alice")
    teams, accounts = stores["teams"], stores["accounts"]
    member, generation = teams.member_registration(member_id=alice_id)
    assert (member.user_id, generation) == ("alice", accounts.get_user("alice").account_generation)
    assert teams.member_registration(user_id="alice") == (member, generation)
    assert teams.member_registration(member_id="nobody") is None
    with pytest.raises(ValueError):
        teams.member_registration()


def test_removing_connections_takes_only_the_members_listed_rows(stores, db_uri: str) -> None:
    _member(stores, "owner")
    _member(stores, "carol", role="admin")
    _member(stores, "alice")
    _member(stores, "bob")
    credentials = CredentialStore(db_uri, _FakeCipher())
    _connect(credentials, "alice", "github", "mcp:tracker", "mcp-pending:tracker")
    _connect(credentials, "bob", "mcp:tracker")
    action = _action(stores, "carol", "alice")

    removed = stores["accounts"].remove_member_connections(
        action, ["mcp:tracker", "mcp-pending:tracker"]
    )

    assert removed is True
    assert credentials.get("alice", "mcp:tracker") is None
    assert credentials.get("alice", "mcp-pending:tracker") is None
    assert credentials.get("alice", "github") is not None
    assert credentials.get("bob", "mcp:tracker") is not None
    assert stores["accounts"].remove_member_connections(action, ["mcp:tracker"]) is False


def test_a_support_write_sees_the_team_as_it_is_then(stores, db_uri: str) -> None:
    owner_id = _member(stores, "owner")
    carol_id = _member(stores, "carol", role="admin")
    alice_id = _member(stores, "alice")
    teams, accounts = stores["teams"], stores["accounts"]
    credentials = CredentialStore(db_uri, _FakeCipher())
    _connect(credentials, "alice", "github")
    remove = accounts.remove_member_connections

    # The actor demoted after the request read the team: refused.
    action = _action(stores, "carol", "alice")
    teams.change_role(carol_id, "member")
    assert _code(lambda: remove(action, ["github"])) == ErrorCode.FORBIDDEN
    teams.change_role(carol_id, "admin")

    # The member suspended: allowed for a suspended member, not for an active-only action.
    accounts.change_member_state(alice_id, "suspended", actor_member_id=owner_id)
    stale = _action(stores, "carol", "alice", states=_ANY)
    active_only = SupportAction(
        stale.actor_member_id, stale.member_id, stale.member_generation, _ACTIVE
    )
    assert _code(lambda: remove(active_only, ["github"])) == ErrorCode.NOT_FOUND
    assert credentials.get("alice", "github") is not None

    # Removed, and the username registered again by someone else.
    current = _action(stores, "carol", "alice")
    accounts.remove_member(alice_id, actor_member_id=owner_id)
    accounts.create_user_with_password("alice", "another-persons-hash")
    _connect(credentials, "alice", "github")
    assert _code(lambda: remove(current, ["github"])) in (ErrorCode.CONFLICT, ErrorCode.NOT_FOUND)
    assert credentials.get("alice", "github") is not None


def test_a_reset_follows_suspension_inside_its_write(stores) -> None:
    owner_id = _member(stores, "owner")
    carol_id = _member(stores, "carol", role="admin")
    _member(stores, "dave", role="admin")
    _member(stores, "alice")
    accounts, teams = stores["accounts"], stores["teams"]
    accounts.create_user_with_password("loner", "hash")  # an account with no membership
    reset = accounts.reset_password

    for target in ("owner", "dave"):
        before = accounts.get_password_hash(target)
        refused = _code(lambda t=target: reset(t, "new", actor_member_id=carol_id))
        assert refused == ErrorCode.FORBIDDEN
        assert accounts.get_password_hash(target) == before
    assert reset("alice", "new-alice", actor_member_id=carol_id) is not None
    assert reset("carol", "new-carol", actor_member_id=carol_id) is not None
    assert reset("dave", "new-dave", actor_member_id=owner_id) is not None
    assert reset("loner", "new-loner", actor_member_id=carol_id) is not None
    # No membership of their own: a member's password only.
    assert reset("alice", "again", actor_member_id=None) is not None
    assert _code(lambda: reset("dave", "x", actor_member_id=None)) == ErrorCode.FORBIDDEN
    # A demoted admin no longer resets a member's password.
    teams.change_role(carol_id, "member")
    before = accounts.get_password_hash("alice")
    assert _code(lambda: reset("alice", "x", actor_member_id=carol_id)) == ErrorCode.FORBIDDEN
    assert accounts.get_password_hash("alice") == before


def test_a_computer_for_someone_is_written_only_while_both_still_qualify(stores) -> None:
    owner_id = _member(stores, "owner")
    carol_id = _member(stores, "carol", role="admin")
    alice_id = _member(stores, "alice")
    hosts, teams, accounts = stores["hosts"], stores["teams"], stores["accounts"]

    def create(action: SupportAction, user_id: str = "alice") -> Any:
        return hosts.create_persistent_host(
            host_id=uuid.uuid4().hex,
            name=f"c-{uuid.uuid4().hex[:6]}",
            user_id=user_id,
            provider="fakepersist",
            support=action,
        )

    made = create(_action(stores, "carol", "alice", _ACTIVE))
    assert made.user_id == "alice"
    # Not for someone other than the member the action names.
    assert _code(lambda: create(_action(stores, "carol", "alice", _ACTIVE), "carol")) == (
        ErrorCode.NOT_FOUND
    )
    action = _action(stores, "carol", "alice", _ACTIVE)
    teams.change_role(carol_id, "member")
    assert _code(lambda: create(action)) == ErrorCode.FORBIDDEN
    teams.change_role(carol_id, "admin")
    action = _action(stores, "carol", "alice", _ACTIVE)
    accounts.change_member_state(alice_id, "suspended", actor_member_id=owner_id)
    assert _code(lambda: create(action)) in (ErrorCode.CONFLICT, ErrorCode.NOT_FOUND)
    assert [h.host_id for h in hosts.list_hosts("alice")] == [made.host_id]


def test_deleting_a_members_computer_checks_the_action_and_the_owner(stores) -> None:
    _member(stores, "owner")
    carol_id = _member(stores, "carol", role="admin")
    _member(stores, "alice")
    _member(stores, "bob")
    hosts, teams = stores["hosts"], stores["teams"]
    alices = hosts.create_persistent_host(
        host_id=uuid.uuid4().hex, name="a", user_id="alice", provider="fakepersist"
    )
    bobs = hosts.create_persistent_host(
        host_id=uuid.uuid4().hex, name="b", user_id="bob", provider="fakepersist"
    )
    action = _action(stores, "carol", "alice")
    # An action for Alice never deletes Bob's computer.
    assert _code(lambda: hosts.delete_host(bobs.host_id, support=action)) == ErrorCode.NOT_FOUND
    assert hosts.get_host(bobs.host_id) is not None
    teams.change_role(carol_id, "member")
    refused = _code(lambda: hosts.delete_host(alices.host_id, support=action))
    assert refused == ErrorCode.FORBIDDEN
    assert hosts.get_host(alices.host_id) is not None
    teams.change_role(carol_id, "admin")
    assert hosts.delete_host(alices.host_id, support=action) is not None
    assert hosts.get_host(alices.host_id) is None


def test_a_members_connections_and_threads_are_read_bound_to_the_membership(
    stores, db_uri: str
) -> None:
    from omnigent.server.auth import LEVEL_OWNER
    from omnigent.stores.conversation_store.sqlalchemy_store import SqlAlchemyConversationStore
    from omnigent.stores.permission_store.sqlalchemy_store import SqlAlchemyPermissionStore

    owner_id = _member(stores, "owner")
    alice_id = _member(stores, "alice")
    bob_id = _member(stores, "bob")
    teams, accounts = stores["teams"], stores["accounts"]
    credentials = CredentialStore(db_uri, _FakeCipher())
    conversations = SqlAlchemyConversationStore(db_uri)
    permissions = SqlAlchemyPermissionStore(db_uri)

    def owned_thread(user_id: str) -> str:
        thread = conversations.create_conversation(agent_id=None)
        permissions.grant(user_id, thread.id, LEVEL_OWNER)
        return thread.id

    _connect(credentials, "alice", "github", "mcp:tracker")
    alices_thread = owned_thread("alice")
    _member_row, generation = teams.member_registration(member_id=alice_id)
    _bob, bob_generation = teams.member_registration(member_id=bob_id)

    rows = accounts.member_connections(alice_id, generation, _ANY)
    assert sorted(row.provider for row in rows) == ["github", "mcp:tracker"]
    assert all(row.secret is None for row in rows)
    assert accounts.member_connections(bob_id, bob_generation, _ANY) == []
    assert teams.member_owned_threads(alice_id, generation) == [alices_thread]
    assert teams.member_owned_threads(bob_id, bob_generation) == []

    # Removed, and the username registered again by someone else with their own rows.
    accounts.remove_member(alice_id, actor_member_id=owner_id)
    accounts.create_user_with_password("alice", "another-persons-hash")
    _connect(credentials, "alice", "github")
    owned_thread("alice")
    assert accounts.member_connections(alice_id, generation, _ANY) is None
    assert teams.member_owned_threads(alice_id, generation) is None
