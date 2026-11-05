"""Removing a team member: one transaction, its record, and the rules it keeps.

Runs on SQLite by default and on the configured server database when
``OMNIGENT_TEST_DB_URI`` is set, including the concurrent removals.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest

from omnigent.errors import ErrorCode, OmnigentError
from omnigent.server import accounts_store as accounts_module
from omnigent.server.accounts_store import SqlAlchemyAccountStore
from omnigent.server.auth import LEVEL_EDIT, LEVEL_OWNER
from omnigent.stores.conversation_store.sqlalchemy_store import SqlAlchemyConversationStore
from omnigent.stores.host_store import HostStore
from omnigent.stores.permission_store.sqlalchemy_store import SqlAlchemyPermissionStore
from omnigent.stores.team_store import (
    InvitationUnavailableError,
    LastActiveOwnerError,
    SqlAlchemyTeamStore,
)
from tests.stores.test_team_store import _race

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


def _member(stores: dict[str, Any], user_id: str, role: str = "member") -> str:
    """An account with an active membership and a bound identity carrying its address."""
    founding = stores["accounts"].list_users() == []
    stores["accounts"].create_user_with_password(
        user_id, "hash", is_admin=founding, team_owner=founding
    )
    teams = stores["teams"]
    member = teams.get_member_for_account(user_id)
    if member is None:
        member = teams.add_member(user_id, role=role, state="active")
    teams.bind_external_identity(
        member.id,
        issuer=_ISSUER,
        subject=f"{user_id}-sub",
        audience=_AUDIENCE,
        provenance="invitation",
        email=f"{user_id}@example.test",
    )
    return member.id


def _computer(stores: dict[str, Any], user_id: str) -> str:
    hosts = stores["hosts"]
    host = hosts.create_persistent_host(
        host_id=uuid.uuid4().hex, name="Work computer", user_id=user_id, provider="fakepersist"
    )
    hosts.attach_persistent_sandbox(
        host.host_id, provider="fakepersist", sandbox_id=f"sb-{user_id}"
    )
    return host.host_id


def _thread(stores: dict[str, Any], owner: str) -> str:
    thread = stores["conversations"].create_conversation(agent_id=None)
    stores["permissions"].grant(owner, thread.id, LEVEL_OWNER)
    return thread.id


def test_removal_is_one_transaction_that_records_what_the_member_had(stores) -> None:
    owner_id = _member(stores, "owner")
    alice_id = _member(stores, "alice", role="admin")
    _member(stores, "bob")
    teams, accounts = stores["teams"], stores["accounts"]
    own_thread = _thread(stores, "alice")
    bobs_thread = _thread(stores, "bob")
    stores["permissions"].grant("alice", bobs_thread, LEVEL_EDIT)
    computer = _computer(stores, "alice")
    stores["hosts"].upsert_on_connect(uuid.uuid4().hex, "alice-laptop", "alice")
    teams.create_invitation(
        "alice@example.test", "member", created_by_member_id=owner_id, ttl_seconds=3600
    )
    _, dave_token = teams.create_invitation(
        "dave@example.test", "member", created_by_member_id=alice_id, ttl_seconds=3600
    )
    before = teams.admission_snapshot()
    assert "alice@example.test" in before.emails
    # A current member's own threads, read bound to the membership: what the removal records.
    _member_row, generation = teams.member_registration(member_id=alice_id)
    assert teams.member_owned_threads(alice_id, generation) == [own_thread]
    assert teams.member_owned_threads(alice_id, "another-registration") is None

    removal = accounts.remove_member(alice_id, actor_member_id=owner_id)

    assert accounts.get_user("alice") is None
    assert teams.get_member(alice_id).state == "removed"
    assert teams.get_member_for_account("alice") is None
    assert teams.member_for_external_identity(_ISSUER, "alice-sub", audience=_AUDIENCE) is None
    after = teams.admission_snapshot()
    assert after.revision > before.revision
    assert "alice@example.test" not in after.emails
    # The invitation she issued is Dave's, and stays.
    assert teams.invitation_for_token(dave_token).state == "pending"
    assert {i.email for i in teams.list_invitations()} == {"dave@example.test"}
    assert (removal.member_id, removal.user_id, removal.removed_by_member_id) == (
        alice_id,
        "alice",
        owner_id,
    )
    assert removal.admission_revision == after.revision
    # Her own thread, not the one shared with her; her cloud computer, not the laptop.
    assert (removal.thread_count, removal.computers, removal.computers_pending) == (1, 1, 1)
    assert teams.removal_threads(alice_id) == [own_thread]
    assert stores["hosts"].list_deleted_persistent_hosts("alice")[0].host_id == computer
    # Once the provider confirms, nothing is pending.
    stores["hosts"].mark_sandbox_terminated(computer, sandbox_id="sb-alice")
    assert teams.get_removal(alice_id).computers_pending == 0
    assert [r.member_id for r in teams.list_removals()] == [alice_id]


def test_an_invitation_to_their_address_is_revoked_whatever_its_case(stores) -> None:
    owner_id = _member(stores, "owner")
    teams, accounts = stores["teams"], stores["accounts"]
    accounts.create_user_with_password("carol", "hash")
    carol = teams.add_member("carol", role="member", state="active")
    teams.bind_external_identity(
        carol.id,
        issuer=_ISSUER,
        subject="carol-sub",
        audience=_AUDIENCE,
        provenance="invitation",
        email="Carol@Example.test",
    )
    teams.create_invitation(
        "carol@example.test", "member", created_by_member_id=owner_id, ttl_seconds=3600
    )
    accounts.remove_member(carol.id, actor_member_id=owner_id)
    assert teams.list_invitations() == []


def test_a_member_registered_without_a_proxy_cannot_rejoin_with_an_earlier_invitation(
    stores,
) -> None:
    """Without a front proxy no identity is bound; the redeemed invitation's address counts."""
    owner_id = _member(stores, "owner")
    teams, accounts = stores["teams"], stores["accounts"]
    _, first = teams.create_invitation(
        "member@example.test", "member", created_by_member_id=owner_id, ttl_seconds=3600
    )
    _, member = accounts.redeem_team_invitation(
        first, username="member", password_hash="hash", identity=None
    )
    assert teams.list_external_identities(member.id) == []
    # A second invitation to the same address, issued while they are a member.
    _, spare = teams.create_invitation(
        "Member@Example.test", "member", created_by_member_id=owner_id, ttl_seconds=3600
    )
    current = {m.id for m in teams.list_members(limit=100)[0]}

    accounts.remove_member(member.id, actor_member_id=owner_id)

    assert teams.invitation_for_token(spare).state == "revoked"
    for username in ("member", "someone-else"):
        with pytest.raises(InvitationUnavailableError):
            accounts.redeem_team_invitation(
                spare, username=username, password_hash="hash", identity=None
            )
    assert teams.get_member_for_account("member") is None
    assert {m.id for m in teams.list_members(limit=100)[0]} == current - {member.id}


def test_a_removal_that_fails_changes_nothing(stores, monkeypatch: pytest.MonkeyPatch) -> None:
    owner_id = _member(stores, "owner")
    bob_id = _member(stores, "bob")
    teams, accounts = stores["teams"], stores["accounts"]
    _thread(stores, "bob")
    teams.create_invitation(
        "bob@example.test", "member", created_by_member_id=owner_id, ttl_seconds=3600
    )
    revision = teams.admission_snapshot().revision

    def failing_record(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("the database went away")

    monkeypatch.setattr(accounts_module, "record_member_removal", failing_record)
    with pytest.raises(RuntimeError):
        accounts.remove_member(bob_id, actor_member_id=owner_id)

    assert accounts.get_user("bob") is not None
    assert teams.get_member(bob_id).state == "active"
    assert teams.member_for_external_identity(_ISSUER, "bob-sub", audience=_AUDIENCE) is not None
    assert [i.email for i in teams.list_invitations()] == ["bob@example.test"]
    assert teams.admission_snapshot().revision == revision
    assert teams.get_removal(bob_id) is None


def test_repeating_a_removal_returns_its_record_and_changes_nothing(stores) -> None:
    owner_id = _member(stores, "owner")
    bob_id = _member(stores, "bob")
    teams, accounts = stores["teams"], stores["accounts"]
    first = accounts.remove_member(bob_id, actor_member_id=owner_id)
    revision = teams.admission_snapshot().revision

    assert accounts.remove_member(bob_id, actor_member_id=owner_id) == first
    assert teams.admission_snapshot().revision == revision
    with pytest.raises(OmnigentError) as unknown:
        accounts.remove_member(uuid.uuid4().hex, actor_member_id=owner_id)
    assert unknown.value.code == ErrorCode.NOT_FOUND


def test_who_may_remove_whom_is_decided_inside_the_write(stores) -> None:
    owner_id = _member(stores, "owner")
    carol_id = _member(stores, "carol", role="admin")
    dave_id = _member(stores, "dave", role="admin")
    bob_id = _member(stores, "bob")
    accounts = stores["accounts"]
    for actor, target in ((carol_id, carol_id), (carol_id, owner_id), (carol_id, dave_id)):
        with pytest.raises(OmnigentError) as refused:
            accounts.remove_member(target, actor_member_id=actor)
        assert refused.value.code == ErrorCode.FORBIDDEN
    # A member is no administrator.
    with pytest.raises(OmnigentError) as member:
        accounts.remove_member(carol_id, actor_member_id=bob_id)
    assert member.value.code == ErrorCode.FORBIDDEN
    accounts.remove_member(bob_id, actor_member_id=carol_id)
    accounts.remove_member(dave_id, actor_member_id=owner_id)
    assert stores["teams"].get_member(bob_id).state == "removed"


def test_the_last_active_owner_and_the_installation_owner_are_kept(stores) -> None:
    owner_id = _member(stores, "owner")
    accounts, teams = stores["accounts"], stores["teams"]
    with pytest.raises(LastActiveOwnerError):
        accounts.remove_member(owner_id, actor_member_id=None)
    second_id = _member(stores, "second", role="owner")
    with pytest.raises(OmnigentError) as protected:
        accounts.remove_member(owner_id, actor_member_id=second_id, protected_user_id="owner")
    assert protected.value.code == ErrorCode.FORBIDDEN
    assert teams.get_member(owner_id).state == "active"
    accounts.remove_member(second_id, actor_member_id=owner_id, protected_user_id="owner")
    assert teams.get_member(second_id).state == "removed"


@pytest.mark.parametrize("round_number", range(2))
def test_two_owners_removing_each_other_keep_one(stores, round_number) -> None:
    owner_id = _member(stores, "owner")
    bob_id = _member(stores, "bob", role="owner")
    accounts, teams = stores["accounts"], stores["teams"]

    results = _race(
        lambda: accounts.remove_member(bob_id, actor_member_id=owner_id),
        lambda: accounts.remove_member(owner_id, actor_member_id=bob_id),
    )

    assert sum(isinstance(result, OmnigentError) for result in results) == 1, results
    states = sorted(teams.get_member(member_id).state for member_id in (owner_id, bob_id))
    assert states == ["active", "removed"]


def test_deleting_a_members_account_also_tells_the_proxy(stores) -> None:
    """Every account deletion that retires a membership moves the admission revision."""
    _member(stores, "owner")
    _member(stores, "bob")
    teams = stores["teams"]
    before = teams.admission_snapshot()
    assert stores["accounts"].delete_user("bob") is True
    after = teams.admission_snapshot()
    assert after.revision > before.revision
    assert "bob@example.test" not in after.emails
