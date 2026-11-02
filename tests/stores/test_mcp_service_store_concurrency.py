"""A service edit and an account deletion running at once take locks in one order.

Account deletion locks the administrators' account rows, then deletes the target's
connections. A service edit that seals a new client secret must take the acting
administrator's account lock before the service and connection rows, or the two
transactions can deadlock. These tests run both for real on PostgreSQL, interleaved
at the point where the old order failed.
"""

from __future__ import annotations

import threading
import time

import pytest
import sqlalchemy as sa

from omnigent.db.account_authority import account_authority_scope
from omnigent.db.utils import get_or_create_engine
from omnigent.server.accounts_store import SqlAlchemyAccountStore
from omnigent.stores.credential_store.sqlalchemy_store import CredentialStore
from omnigent.stores.mcp_service_store import (
    SERVICE_PRINCIPAL,
    McpServiceOAuth,
    McpServiceSettings,
    McpServiceStore,
    client_provider,
    member_provider,
    stamped,
    version_guard,
)
from tests.server.test_credential_store import _FakeCipher


def _oauth(token_url: str) -> McpServiceSettings:
    return McpServiceSettings(
        title="Tracker",
        url="https://mcp.example.com/mcp",
        auth="oauth",
        timeout=60,
        tools=("read_ticket",),
        oauth=McpServiceOAuth(
            authorize_url="https://auth.example.com/authorize",
            token_url=token_url,
            client_id="omnigent",
            registration="manual",
            token_endpoint_auth="client_secret_post",
        ),
    )


def _waiting_on_locks(engine: sa.Engine) -> int:
    with engine.connect() as connection:
        return connection.execute(
            sa.text(
                "SELECT count(*) FROM pg_stat_activity "
                "WHERE wait_event_type = 'Lock' AND datname = current_database()"
            )
        ).scalar_one()


def test_a_service_edit_and_the_deletion_of_its_editor_both_complete(db_uri, monkeypatch):
    engine = get_or_create_engine(db_uri)
    if engine.dialect.name != "postgresql":
        pytest.skip("interleaves PostgreSQL row locks")
    accounts = SqlAlchemyAccountStore(db_uri)
    credentials = CredentialStore(db_uri, _FakeCipher())
    services = McpServiceStore(db_uri)
    for username in ("alice", "bob"):
        accounts.create_user_with_password(username, "not-a-real-hash", is_admin=True)
    alice = accounts.get_user("alice").account_generation
    bob = accounts.get_user("bob").account_generation
    with account_authority_scope("alice", alice):
        record = services.create(
            "tracker",
            _oauth("https://auth.example.com/token"),
            created_by="alice",
            client_secret=credentials.seal(
                SERVICE_PRINCIPAL, client_provider("tracker"), {"client_secret": "first"}
            ),
        )
        credentials.upsert(
            "alice",
            member_provider("tracker"),
            secret={"access_token": "alice-login"},
            metadata=stamped({}, record.credential_version),
            precondition=version_guard("tracker", record.credential_version),
        )

    # Alice's edit pauses once it has deleted the service's logins (Alice's among
    # them), the point after which the old order still wanted Alice's account lock.
    logins_deleted, resume = threading.Event(), threading.Event()
    original = McpServiceStore._delete_logins
    editor: list[threading.Thread] = []

    def paused_delete(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        if threading.current_thread() in editor and not logins_deleted.is_set():
            logins_deleted.set()
            assert resume.wait(30)
        return result

    monkeypatch.setattr(McpServiceStore, "_delete_logins", paused_delete)
    errors: list[BaseException] = []

    def edit() -> None:
        try:
            with account_authority_scope("alice", alice):
                sealed = credentials.seal(
                    SERVICE_PRINCIPAL, client_provider("tracker"), {"client_secret": "second"}
                )
                services.update(
                    "tracker",
                    _oauth("https://other.example.com/token"),
                    expected_version=record.credential_version,
                    client_secret=sealed,
                )
        except BaseException as exc:
            errors.append(exc)

    def delete_alice() -> None:
        try:
            with account_authority_scope("bob", bob):
                assert accounts.delete_user("alice") is True
        except BaseException as exc:
            errors.append(exc)

    editing = threading.Thread(target=edit)
    editor.append(editing)
    editing.start()
    assert logins_deleted.wait(30)
    deleting = threading.Thread(target=delete_alice)
    deleting.start()
    # The deletion is now blocked behind the edit's locks.
    deadline = time.monotonic() + 15
    while _waiting_on_locks(engine) == 0:
        assert time.monotonic() < deadline, "the account deletion never waited"
        time.sleep(0.05)
    resume.set()
    editing.join(30)
    deleting.join(30)

    assert not editing.is_alive() and not deleting.is_alive()
    assert errors == []
    edited = services.get("tracker")
    assert edited.oauth.token_url == "https://other.example.com/token"
    saved = credentials.get(SERVICE_PRINCIPAL, client_provider("tracker"), with_secret=True)
    assert saved.secret == {"client_secret": "second"}
    assert saved.metadata["service_version"] == edited.credential_version
    assert accounts.get_user("alice") is None
    assert credentials.get("alice", member_provider("tracker")) is None
