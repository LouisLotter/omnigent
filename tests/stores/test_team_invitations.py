"""Team invitations, the admission list and suspension (WP8.3 store layer).

Each forbidden case sits next to the allowed one. Runs on SQLite by default and
on the configured server database when ``OMNIGENT_TEST_DB_URI`` is set,
including the concurrent-redemption race.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable
from typing import Any

import pytest

from omnigent.db.account_authority import account_authority_scope
from omnigent.db.db_models import SqlTeamInvitation, current_workspace_id
from omnigent.errors import ErrorCode, OmnigentError
from omnigent.server.accounts_store import SqlAlchemyAccountStore
from omnigent.server.external_identity import ExternalIdentity
from omnigent.stores.host_store import HostStore
from omnigent.stores.team_store import (
    IdentityBindingConflictError,
    InvitationEmailMismatchError,
    InvitationUnavailableError,
    LastActiveOwnerError,
    SqlAlchemyTeamStore,
)

_ISSUER = "https://team.example.cloudflareaccess.com"
_AUDIENCE = "aud-tag-1"


@pytest.fixture
def stores(db_uri: str) -> dict[str, Any]:
    accounts = SqlAlchemyAccountStore(db_uri)
    accounts.create_user_with_password("owner", "hash", is_admin=True, team_owner=True)
    return {
        "teams": SqlAlchemyTeamStore(db_uri),
        "accounts": accounts,
        "hosts": HostStore(db_uri),
        "db_uri": db_uri,
    }


def _identity(subject: str, email: str | None, *, owner: bool = False) -> ExternalIdentity:
    now = int(time.time())
    return ExternalIdentity(
        issuer=_ISSUER,
        subject=subject,
        audience=_AUDIENCE,
        email=email,
        installation_owner=owner,
        issued_at=now,
        expires_at=now + 60,
    )


def _owner_id(stores: dict[str, Any]) -> str:
    owner = stores["teams"].get_member_for_account("owner")
    assert owner is not None
    return owner.id


def _invite(stores: dict[str, Any], email: str, role: str = "member", **kwargs: Any):
    return stores["teams"].create_invitation(
        email, role, created_by_member_id=_owner_id(stores), ttl_seconds=3600, **kwargs
    )


def _join(stores: dict[str, Any], name: str, role: str = "member") -> str:
    email = f"{name}@example.test"
    _, token = _invite(stores, email, role)
    _, member = stores["accounts"].redeem_team_invitation(
        token, username=name, password_hash="hash", identity=_identity(f"{name}-sub", email)
    )
    return member.id


def _race(*actions: Callable[[], object]) -> list[object]:
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


# ── Invitations ────────────────────────────────────────────────────


def test_an_invitation_stores_only_a_digest_and_names_one_address(stores) -> None:
    invitation, token = _invite(stores, " Alice@Example.TEST ")
    assert invitation.email == "alice@example.test"
    assert (invitation.role, invitation.state) == ("member", "pending")
    assert stores["teams"].invitation_for_token(token) == invitation
    assert stores["teams"].invitation_for_token(token + "x") is None
    with stores["teams"]._session("test_read_invitation") as session:
        row = session.get(SqlTeamInvitation, (current_workspace_id(), invitation.id))
        assert row is not None and token.encode() not in row.token_digest


@pytest.mark.parametrize(
    ("email", "role"),
    [("not-an-address", "member"), ("a@example.test", "owner"), ("a@example.test", "guest")],
)
def test_a_malformed_invitation_is_refused(stores, email: str, role: str) -> None:
    with pytest.raises(OmnigentError) as caught:
        _invite(stores, email, role)
    assert caught.value.code is ErrorCode.INVALID_INPUT


def test_a_new_invitation_for_an_address_revokes_the_earlier_one(stores) -> None:
    first, first_token = _invite(stores, "alice@example.test")
    second, second_token = _invite(stores, "alice@example.test", "admin")
    assert stores["teams"].invitation_for_token(first_token).state == "revoked"
    assert stores["teams"].invitation_for_token(second_token).state == "pending"
    assert [i.id for i in stores["teams"].list_invitations()] == [second.id]
    assert first.id != second.id


def test_redemption_creates_the_account_membership_and_binding_at_once(stores) -> None:
    _, token = _invite(stores, "alice@example.test", "admin")
    account, member = stores["accounts"].redeem_team_invitation(
        token,
        username="alice",
        password_hash="hash",
        identity=_identity("alice-sub", "alice@example.test"),
    )
    assert account.id == "alice" and account.is_admin
    assert (member.user_id, member.role, member.state) == ("alice", "admin", "active")
    bound = stores["teams"].member_for_external_identity(_ISSUER, "alice-sub", audience=_AUDIENCE)
    assert bound == member
    binding = stores["teams"].get_external_identity(_ISSUER, "alice-sub")
    assert (binding.provenance, binding.email) == ("invitation", "alice@example.test")
    assert stores["teams"].invitation_for_token(token).state == "redeemed"


def test_a_used_invitation_cannot_be_redeemed_again(stores) -> None:
    _, token = _invite(stores, "alice@example.test")
    stores["accounts"].redeem_team_invitation(
        token,
        username="alice",
        password_hash="hash",
        identity=_identity("alice-sub", "alice@example.test"),
    )
    with pytest.raises(InvitationUnavailableError):
        stores["accounts"].redeem_team_invitation(
            token,
            username="mallory",
            password_hash="hash",
            identity=_identity("mallory-sub", "alice@example.test"),
        )
    assert stores["accounts"].get_user("mallory") is None


def test_another_address_cannot_redeem_and_the_invited_one_can(stores) -> None:
    _, token = _invite(stores, "erin@example.test")
    with pytest.raises(InvitationEmailMismatchError):
        stores["accounts"].redeem_team_invitation(
            token,
            username="dave",
            password_hash="hash",
            identity=_identity("dave-sub", "dave@example.test"),
        )
    assert stores["accounts"].get_user("dave") is None
    _, member = stores["accounts"].redeem_team_invitation(
        token,
        username="erin",
        password_hash="hash",
        identity=_identity("erin-sub", "erin@example.test"),
    )
    assert member.user_id == "erin"


def test_an_identity_already_bound_cannot_take_a_second_membership(stores) -> None:
    _join(stores, "alice")
    _, token = _invite(stores, "alice@example.test")
    with pytest.raises(IdentityBindingConflictError):
        stores["accounts"].redeem_team_invitation(
            token,
            username="alice2",
            password_hash="hash",
            identity=_identity("alice-sub", "alice@example.test"),
        )
    assert stores["accounts"].get_user("alice2") is None


def test_an_expired_or_revoked_invitation_is_unavailable(stores) -> None:
    invitation, token = _invite(stores, "alice@example.test")
    stores["teams"].revoke_invitation(invitation.id)
    with pytest.raises(InvitationUnavailableError):
        stores["accounts"].redeem_team_invitation(
            token, username="alice", password_hash="hash", identity=None
        )
    _, fresh = _invite(stores, "bob@example.test")
    with stores["teams"]._session_immediate("test_expire_invitation") as session:
        for row in session.query(SqlTeamInvitation).filter(
            SqlTeamInvitation.email == "bob@example.test"
        ):
            row.expires_at = int(time.time()) - 1
        session.commit()
    with pytest.raises(InvitationUnavailableError):
        stores["accounts"].redeem_team_invitation(
            fresh, username="bob", password_hash="hash", identity=None
        )
    assert stores["teams"].invitation_for_token(fresh).state == "expired"


def test_concurrent_redemptions_of_one_invitation_create_one_account(stores) -> None:
    _, token = _invite(stores, "alice@example.test")
    results = _race(
        *(
            lambda name=name: stores["accounts"].redeem_team_invitation(
                token,
                username=name,
                password_hash="hash",
                identity=_identity(f"{name}-sub", "alice@example.test"),
            )
            for name in ("alice", "alice-two")
        )
    )
    successes = [result for result in results if isinstance(result, tuple)]
    assert len(successes) == 1, results
    assert all(
        isinstance(result, (InvitationUnavailableError, OmnigentError))
        for result in results
        if not isinstance(result, tuple)
    )


def test_a_link_invitation_needs_its_account_and_binds_by_account_proof(stores) -> None:
    stores["accounts"].create_user_with_password("legacy", "hash")
    _, token = _invite(stores, "legacy@example.test", target_user_id="legacy")
    with pytest.raises(InvitationUnavailableError):
        stores["accounts"].link_team_invitation(
            token, user_id="owner", identity=_identity("x-sub", "legacy@example.test")
        )
    member = stores["accounts"].link_team_invitation(
        token, user_id="legacy", identity=_identity("legacy-sub", "legacy@example.test")
    )
    assert (member.user_id, member.state) == ("legacy", "active")
    binding = stores["teams"].get_external_identity(_ISSUER, "legacy-sub")
    assert binding.provenance == "account_proof"
    # A new-account redemption of a link invitation is refused.
    _, other = _invite(stores, "x@example.test", target_user_id=None)
    stores["accounts"].create_user_with_password("x-account", "hash")
    _, link = _invite(stores, "x2@example.test", target_user_id="x-account")
    with pytest.raises(InvitationEmailMismatchError):
        stores["accounts"].redeem_team_invitation(
            link, username="x-new", password_hash="hash", identity=None
        )
    assert other


# ── The installation owner's first entry ───────────────────────────


def test_the_owner_binds_once_with_the_pin_and_a_non_owner_cannot(stores) -> None:
    _join(stores, "alice")
    assert (
        stores["teams"].bind_installation_owner(
            "alice", _identity("fresh-sub", "alice@example.test", owner=True)
        )
        is None
    )
    assert (
        stores["teams"].bind_installation_owner(
            "owner", _identity("owner-sub", "owner@example.test", owner=False)
        )
        is None
    )
    member = stores["teams"].bind_installation_owner(
        "owner", _identity("owner-sub", "owner@example.test", owner=True)
    )
    assert member is not None and member.role == "owner"
    binding = stores["teams"].get_external_identity(_ISSUER, "owner-sub")
    assert (binding.provenance, binding.email) == ("migration", "owner@example.test")
    # Alice's identity is hers; a pin claim never moves it to the owner.
    assert (
        stores["teams"].bind_installation_owner(
            "owner", _identity("alice-sub", "alice@example.test", owner=True)
        )
        is None
    )


# ── Admission ──────────────────────────────────────────────────────


def test_admission_lists_active_members_and_pending_invitations_only(stores) -> None:
    owner = _owner_id(stores)
    stores["teams"].bind_external_identity(
        owner,
        issuer=_ISSUER,
        subject="owner-sub",
        audience=_AUDIENCE,
        provenance="migration",
        email="owner@example.test",
    )
    alice = _join(stores, "alice")
    _join(stores, "bob")
    _invite(stores, "carol@example.test", "admin")
    before = stores["teams"].admission_snapshot()
    assert before.emails == (
        "alice@example.test",
        "bob@example.test",
        "carol@example.test",
        "owner@example.test",
    )
    stores["accounts"].change_member_state(alice, "suspended")
    after = stores["teams"].admission_snapshot()
    assert "alice@example.test" not in after.emails
    assert "bob@example.test" in after.emails
    assert after.revision > before.revision
    stores["accounts"].change_member_state(alice, "active")
    assert "alice@example.test" in stores["teams"].admission_snapshot().emails


def test_every_admission_change_raises_the_revision(stores) -> None:
    revisions = [stores["teams"].admission_snapshot().revision]
    invitation, _ = _invite(stores, "alice@example.test")
    revisions.append(stores["teams"].admission_snapshot().revision)
    stores["teams"].revoke_invitation(invitation.id)
    revisions.append(stores["teams"].admission_snapshot().revision)
    member = _join(stores, "bob")
    revisions.append(stores["teams"].admission_snapshot().revision)
    stores["teams"].change_role(member, "admin")  # a role is not admission
    revisions.append(stores["teams"].admission_snapshot().revision)
    assert revisions[0] < revisions[1] < revisions[2] < revisions[3] == revisions[4]


# ── Suspension ─────────────────────────────────────────────────────


def test_suspension_refuses_every_credential_and_rotates_the_registration(stores) -> None:
    alice = _join(stores, "alice")
    bob = _join(stores, "bob")
    old_generation = stores["accounts"].get_user("alice").account_generation
    host_id = uuid.uuid4().hex
    with account_authority_scope("alice", old_generation):
        stores["hosts"].upsert_on_connect(host_id, "alice-laptop", "alice")
    assert stores["hosts"].get_host(host_id).account_generation == old_generation
    assert stores["accounts"].accepts_generation("alice", old_generation)

    stores["accounts"].change_member_state(alice, "suspended")

    new_generation = stores["accounts"].get_user("alice").account_generation
    assert new_generation != old_generation
    assert not stores["accounts"].accepts_generation("alice", old_generation)
    assert not stores["accounts"].accepts_generation("alice", new_generation)
    assert not stores["accounts"].get_user("alice").is_admin
    # Hosts move to the new registration, as on a password reset: nothing orphaned.
    assert stores["hosts"].get_host(host_id).account_generation == new_generation
    # Bob is untouched.
    bob_generation = stores["accounts"].get_user("bob").account_generation
    assert stores["accounts"].accepts_generation("bob", bob_generation)
    assert stores["teams"].get_member(bob).state == "active"

    stores["accounts"].change_member_state(alice, "active")
    assert stores["accounts"].accepts_generation("alice", new_generation)
    assert not stores["accounts"].accepts_generation("alice", old_generation)


def test_the_last_active_owner_cannot_be_suspended(stores) -> None:
    with pytest.raises(LastActiveOwnerError):
        stores["accounts"].change_member_state(_owner_id(stores), "suspended")
    second = _join(stores, "carol", "admin")
    stores["teams"].change_role(second, "owner")
    stores["accounts"].change_member_state(_owner_id(stores), "suspended")
    assert stores["teams"].get_member_for_account("owner").state == "suspended"
