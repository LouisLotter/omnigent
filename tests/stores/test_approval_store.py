"""Approver grants, protected rules and the protected-approval ledger.

Runs on SQLite by default and on the configured server database when
``OMNIGENT_TEST_DB_URI`` is set, including the concurrent-consumption race.
"""

from __future__ import annotations

import hashlib
import threading
import time
import uuid
from collections.abc import Callable

import pytest

from omnigent.errors import ErrorCode, OmnigentError
from omnigent.stores.approval_store import (
    TEAM_RESOURCE,
    ApprovalResource,
    Outcome,
    PolicyDomain,
    ResourceKind,
    SqlAlchemyApprovalStore,
    VerdictState,
)

_CEL = "omnigent.policies.builtins.cel.cel_policy"
_SERVICE = ApprovalResource(ResourceKind.SERVICE, "tracker")


@pytest.fixture
def store(db_uri: str) -> SqlAlchemyApprovalStore:
    return SqlAlchemyApprovalStore(db_uri)


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
        thread.join(timeout=60)
    return results


def _request(
    store: SqlAlchemyApprovalStore,
    *,
    digest: bytes = hashlib.sha256(b"bound").digest(),
    expires_in: int = 600,
) -> str:
    elicitation_id = f"elicit_{uuid.uuid4().hex}"
    store.record_request(
        elicitation_id=elicitation_id,
        session_id=uuid.uuid4().hex,
        binding_digest=digest,
        binding={"owner": "alice"},
        requirements=[{"kind": "personal", "resource": [1, "s"]}],
        expires_at=int(time.time()) + expires_in,
    )
    return elicitation_id


def _approve(store: SqlAlchemyApprovalStore, elicitation_id: str) -> None:
    answer = {"by": "alice", "action": "accept", "satisfies": [0]}
    approved = store.record_answer(elicitation_id, answer=answer, decline=False, complete=True)
    assert approved is not None and approved.state is VerdictState.APPROVED


def test_approvers_are_named_per_resource_and_default_initiator_rules_differ(store) -> None:
    """A thread allows its initiator by default; a service or the team does not."""
    thread = ApprovalResource(ResourceKind.SESSION, uuid.uuid4().hex)
    assert store.add_approver(_SERVICE, "carol", granted_by="admin")
    assert not store.add_approver(_SERVICE, "carol", granted_by="admin")
    store.add_approver(TEAM_RESOURCE, "dave", granted_by="admin")

    service, team, own = store.approvers_for([_SERVICE, TEAM_RESOURCE, thread])

    assert (service.approvers, team.approvers, own.approvers) == (("carol",), ("dave",), ())
    assert not service.initiator_may_approve() and not team.initiator_may_approve()
    assert own.initiator_may_approve()
    store.set_allow_initiator(_SERVICE, True, updated_by="admin")
    assert store.approvers_for([_SERVICE])[0].initiator_may_approve()
    assert store.resources_for_approver("carol") == [_SERVICE]


def test_removing_an_approver_changes_the_grant_fingerprint(store) -> None:
    """A verdict bound to the approvers no longer matches once one is removed."""
    store.add_approver(_SERVICE, "carol", granted_by="admin")
    before = store.approvers_for([_SERVICE])[0].fingerprint()
    assert store.remove_approver(_SERVICE, "carol")
    assert not store.remove_approver(_SERVICE, "carol")
    assert store.approvers_for([_SERVICE])[0].fingerprint() != before


def test_protected_rules_governing_a_thread_are_the_teams_and_its_owners(store) -> None:
    """Enabled team rules and the owner's personal rules; not other people's."""
    team = store.create_rule(
        domain=PolicyDomain.TEAM,
        owner_user_id=None,
        name="team",
        handler=_CEL,
        factory_params={"expression": "{}"},
        approval=_SERVICE,
        created_by="admin",
    )
    mine = store.create_rule(
        domain=PolicyDomain.PERSONAL,
        owner_user_id="alice",
        name="mine",
        handler=_CEL,
        factory_params=None,
        approval=None,
        created_by="alice",
    )
    store.create_rule(
        domain=PolicyDomain.PERSONAL,
        owner_user_id="bob",
        name="bobs",
        handler=_CEL,
        factory_params=None,
        approval=None,
        created_by="bob",
    )

    assert [rule.id for rule in store.rules_for("alice")] == [team.id, mine.id]
    assert [rule.id for rule in store.rules_for(None)] == [team.id]
    assert store.has_enabled_rules()
    before = team.fingerprint()
    updated = store.update_rule(team.id, enabled=False)
    assert updated is not None and updated.fingerprint() != before
    assert [rule.id for rule in store.rules_for("alice")] == [mine.id]
    assert store.delete_rule(mine.id) and not store.delete_rule(mine.id)


@pytest.mark.parametrize(
    ("domain", "owner", "approval"),
    [
        (PolicyDomain.PERSONAL, None, None),
        (PolicyDomain.TEAM, "alice", None),
        (PolicyDomain.PERSONAL, "alice", _SERVICE),
        (PolicyDomain.TEAM, None, ApprovalResource(ResourceKind.SESSION, "s")),
    ],
)
def test_rule_targets_are_validated(store, domain, owner, approval) -> None:
    """Personal rules have their person and no approvers; team rules name a service or the team."""
    with pytest.raises(OmnigentError) as raised:
        store.create_rule(
            domain=domain,
            owner_user_id=owner,
            name="bad",
            handler=_CEL,
            factory_params=None,
            approval=approval,
            created_by="x",
        )
    assert raised.value.code == ErrorCode.INVALID_INPUT


def test_request_is_recorded_once_per_binding(store) -> None:
    """A retry with the same binding converges; the same id for another binding is refused."""
    elicitation_id = _request(store)
    record = store.get_verdict(elicitation_id)
    assert record is not None and record.state is VerdictState.PENDING
    again = store.record_request(
        elicitation_id=elicitation_id,
        session_id=record.session_id,
        binding_digest=record.binding_digest,
        binding={},
        requirements=[],
        expires_at=record.expires_at,
    )
    assert again.created_at == record.created_at
    with pytest.raises(OmnigentError) as raised:
        store.record_request(
            elicitation_id=elicitation_id,
            session_id=record.session_id,
            binding_digest=hashlib.sha256(b"other").digest(),
            binding={},
            requirements=[],
            expires_at=record.expires_at,
        )
    assert raised.value.code == ErrorCode.CONFLICT


def test_answers_accumulate_until_complete_and_a_decline_is_final(store) -> None:
    """Partial answers keep it pending; completion approves; nothing changes a decision."""
    elicitation_id = _request(store)
    partial = store.record_answer(
        elicitation_id,
        answer={"by": "alice", "action": "accept", "satisfies": [0]},
        decline=False,
        complete=False,
    )
    assert partial is not None and partial.state is VerdictState.PENDING
    _approve(store, elicitation_id)
    late = store.record_answer(elicitation_id, answer={"by": "x"}, decline=True, complete=False)
    assert late is None

    declined_id = _request(store)
    declined = store.record_answer(
        declined_id, answer={"by": "carol"}, decline=True, complete=False
    )
    assert declined is not None and declined.state is VerdictState.DECLINED
    assert declined.decided_by == "carol"


def test_expired_approvals_take_no_answer_and_are_never_consumed(store) -> None:
    """After expiry an approval can be neither answered nor used."""
    unanswered = _request(store, expires_in=-1)
    assert store.record_answer(unanswered, answer={}, decline=False, complete=True) is None

    elicitation_id = _request(store)
    _approve(store, elicitation_id)
    record = store.get_verdict(elicitation_id)
    assert record is not None
    late = store.consume(
        elicitation_id,
        binding_digest=record.binding_digest,
        operation_id="op",
        now=record.expires_at,
    )
    assert not late.consumed and late.reason == "Approval expired; ask again"


def test_consumption_needs_the_exact_binding_and_happens_once(store) -> None:
    """Changed content is refused; the matching action consumes it, then nothing else can."""
    digest = hashlib.sha256(b"bound").digest()
    elicitation_id = _request(store, digest=digest)
    pending = store.consume(elicitation_id, binding_digest=digest, operation_id="op")
    assert not pending.consumed and pending.reason == "Tool call has not been approved"
    _approve(store, elicitation_id)

    changed = store.consume(
        elicitation_id, binding_digest=hashlib.sha256(b"changed").digest(), operation_id="op"
    )
    assert not changed.consumed
    assert changed.reason == "The action changed after it was approved; ask again"

    used = store.consume(elicitation_id, binding_digest=digest, operation_id="op")
    assert used.consumed and used.record is not None
    assert used.record.outcome is Outcome.EXECUTING and used.record.operation_id == "op"
    again = store.consume(elicitation_id, binding_digest=digest, operation_id="op")
    assert not again.consumed and again.reason == "Approval was already used"

    store.record_outcome(elicitation_id, outcome=Outcome.SUCCEEDED, body='{"result": 1}')
    record = store.get_verdict(elicitation_id)
    assert record is not None and record.outcome is Outcome.SUCCEEDED
    assert record.outcome_body == '{"result": 1}'


@pytest.mark.parametrize("round_number", range(3))
def test_parallel_consumption_runs_the_approval_once(store, round_number) -> None:
    """T10: of several simultaneous consumers exactly one gets the approval."""
    digest = hashlib.sha256(b"bound").digest()
    elicitation_id = _request(store, digest=digest)
    _approve(store, elicitation_id)

    results = _race(
        *(
            lambda operation=f"op-{index}": store.consume(
                elicitation_id, binding_digest=digest, operation_id=operation
            )
            for index in range(4)
        )
    )

    assert not [result for result in results if isinstance(result, Exception)]
    assert sum(1 for result in results if getattr(result, "consumed", False)) == 1


def test_parallel_answers_record_one_decision(store) -> None:
    """Concurrent accept and decline: one decides; the other finds it decided."""
    elicitation_id = _request(store)
    results = _race(
        lambda: store.record_answer(
            elicitation_id, answer={"by": "alice"}, decline=False, complete=True
        ),
        lambda: store.record_answer(
            elicitation_id, answer={"by": "carol"}, decline=True, complete=False
        ),
    )
    assert not [result for result in results if isinstance(result, Exception)]
    assert sum(1 for result in results if result is not None) == 1
    record = store.get_verdict(elicitation_id)
    assert record is not None and len(record.answers) == 1


def test_large_outcomes_are_not_kept_and_old_rows_are_purged(store) -> None:
    """A result too large to keep is dropped; expired rows go after the purge window."""
    digest = hashlib.sha256(b"bound").digest()
    elicitation_id = _request(store, digest=digest)
    _approve(store, elicitation_id)
    assert store.consume(elicitation_id, binding_digest=digest, operation_id="op").consumed
    store.record_outcome(elicitation_id, outcome=Outcome.SUCCEEDED, body="x" * 60_000)
    record = store.get_verdict(elicitation_id)
    assert record is not None and record.outcome_body is None

    assert store.purge_expired(before=record.expires_at + 1) >= 1
    assert store.get_verdict(elicitation_id) is None
    assert store.pending_verdicts() == []
