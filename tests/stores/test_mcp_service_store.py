"""Run-time MCP services: storage, credential versions and login removal."""

from __future__ import annotations

import pytest

from omnigent.server.auth import RESERVED_USER_MCP_SERVICE
from omnigent.stores import mcp_service_store as service_store_module
from omnigent.stores.credential_store.sqlalchemy_store import CredentialStore
from omnigent.stores.mcp_service_store import (
    SERVICE_PRINCIPAL,
    McpClientSecretMissing,
    McpServiceConflict,
    McpServiceNotFound,
    McpServiceOAuth,
    McpServiceSettings,
    McpServiceStore,
    McpServiceVersionChanged,
    client_provider,
    member_provider,
    pending_provider,
    stamped,
    version_guard,
)
from tests.server.test_credential_store import _FakeCipher

_OAUTH = McpServiceOAuth(
    authorize_url="https://auth.example.com/authorize",
    token_url="https://auth.example.com/token",
    client_id="omnigent",
    scopes=("tickets:read",),
    resource="https://mcp.example.com/mcp",
    registration="dynamic",
    token_endpoint_auth="client_secret_post",
)


def settings(**changes) -> McpServiceSettings:
    values = {
        "title": "Tracker",
        "url": "https://mcp.example.com/mcp",
        "auth": "bearer",
        "timeout": 60,
        "tools": ("read_ticket",),
    }
    return McpServiceSettings(**{**values, **changes})


@pytest.fixture
def stores(db_uri):
    return McpServiceStore(db_uri), CredentialStore(db_uri, _FakeCipher())


def save_login(credentials, record, user="alice", provider=None, version=None):
    credentials.upsert(
        user,
        provider or member_provider(record.service_id),
        secret={"access_token": f"token-{user}"},
        metadata=stamped({}, version or record.credential_version),
        precondition=version_guard(record.service_id, version or record.credential_version),
    )


def test_service_principal_is_a_reserved_user():
    assert SERVICE_PRINCIPAL == RESERVED_USER_MCP_SERVICE


def client_secret(credentials, value="client-secret", service_id="tracker"):
    return credentials.seal(
        SERVICE_PRINCIPAL, client_provider(service_id), {"client_secret": value}
    )


def test_create_get_list_and_conflict(stores):
    services, credentials = stores
    created = services.create(
        "tracker",
        settings(auth="oauth", oauth=_OAUTH),
        created_by="admin",
        client_secret=client_secret(credentials),
    )
    assert services.get("tracker") == created
    assert created.oauth == _OAUTH
    assert created.created_by == "admin"
    services.create("alpha", settings(auth="none", tools=()), created_by="admin")
    assert [r.service_id for r in services.list_page()] == ["alpha", "tracker"]
    assert [r.service_id for r in services.list_page(after="alpha")] == ["tracker"]
    assert [r.service_id for r in services.scan()[0]] == ["alpha", "tracker"]
    with pytest.raises(McpServiceConflict):
        services.create("tracker", settings(), created_by="admin")


def test_editing_title_or_tools_keeps_logins(stores):
    services, credentials = stores
    record = services.create("tracker", settings(), created_by="admin")
    save_login(credentials, record)
    result = services.update(
        "tracker", settings(title="Renamed", tools=("read_ticket", "list_tickets"))
    )
    assert not result.credentials_reset
    assert result.record.credential_version == record.credential_version
    assert result.record.tools == ("list_tickets", "read_ticket")
    assert credentials.get("alice", member_provider("tracker")) is not None


@pytest.mark.parametrize(
    "change",
    [
        {"url": "https://other.example.com/mcp"},
        {"auth": "none"},
        {"auth": "oauth", "oauth": _OAUTH},
    ],
    ids=["url", "login-type", "oauth-settings"],
)
def test_changing_destination_or_login_type_deletes_every_login(stores, change):
    services, credentials = stores
    record = services.create("tracker", settings(), created_by="admin")
    for user in ("alice", "bob"):
        save_login(credentials, record, user)
    save_login(credentials, record, "alice", pending_provider("tracker"))
    save_login(credentials, record, SERVICE_PRINCIPAL, client_provider("tracker"))
    credentials.upsert("alice", "mcp:other", secret={"access_token": "x"}, metadata={})

    sealed = client_secret(credentials, "new-secret") if "oauth" in change else None
    result = services.update("tracker", settings(**change), client_secret=sealed)

    assert result.credentials_reset
    assert result.record.credential_version != record.credential_version
    for provider in ("mcp:tracker", "mcp-pending:tracker"):
        assert credentials.list_all(provider=provider) == []
    client = credentials.get(SERVICE_PRINCIPAL, "mcp-client:tracker", with_secret=True)
    if sealed is None:
        assert client is None
    else:
        # The new client's secret, saved for the new version in the same transaction.
        assert client.secret == {"client_secret": "new-secret"}
        assert client.metadata["service_version"] == result.record.credential_version
    assert credentials.get("alice", "mcp:other") is not None


def test_a_login_for_replaced_settings_is_refused(stores):
    services, credentials = stores
    record = services.create("tracker", settings(), created_by="admin")
    services.update("tracker", settings(url="https://other.example.com/mcp"))
    with pytest.raises(McpServiceVersionChanged):
        save_login(credentials, record)
    assert credentials.get("alice", member_provider("tracker")) is None
    services.delete("tracker")
    with pytest.raises(McpServiceVersionChanged):
        save_login(credentials, record)


def test_update_with_an_outdated_version_is_refused(stores):
    services, _ = stores
    record = services.create("tracker", settings(), created_by="admin")
    services.update("tracker", settings(url="https://other.example.com/mcp"))
    with pytest.raises(McpServiceVersionChanged):
        services.update(
            "tracker", settings(title="Late"), expected_version=record.credential_version
        )
    with pytest.raises(McpServiceNotFound):
        services.update("missing", settings())


def test_delete_removes_service_tools_and_logins(stores):
    services, credentials = stores
    record = services.create("tracker", settings(), created_by="admin")
    save_login(credentials, record)
    assert services.delete("tracker") is True
    assert services.get("tracker") is None
    assert credentials.list_all(provider="mcp:tracker") == []
    assert services.delete("tracker") is False
    # The ID is free again, and the new service starts without old tools.
    again = services.create("tracker", settings(tools=()), created_by="admin")
    assert again.tools == ()


def test_removal_continues_past_one_batch_and_resumes_after_a_crash(stores, monkeypatch):
    services, credentials = stores
    monkeypatch.setattr(service_store_module, "BATCH_SIZE", 2)
    record = services.create("tracker", settings(), created_by="admin")
    for index in range(5):
        save_login(credentials, record, f"user-{index}")

    # A crash after the first batch leaves the row marked for a later sweep.
    monkeypatch.setattr(McpServiceStore, "finish_sweep", lambda self, service_id: None)
    changed = services.update("tracker", settings(url="https://other.example.com/mcp"))
    assert changed.record.pending_sweep
    assert len(credentials.list_all(provider="mcp:tracker")) == 3
    fresh = changed.record
    save_login(credentials, fresh, "new-user")
    _, pending = services.scan()
    assert pending == ["tracker"]

    monkeypatch.undo()
    monkeypatch.setattr(service_store_module, "BATCH_SIZE", 2)
    services.finish_sweeps(pending)
    remaining = credentials.list_all(provider="mcp:tracker")
    # Only the login saved for the new settings survives.
    assert [c.user_id for c in remaining] == ["new-user"]
    assert services.scan()[1] == []


def test_deleting_a_service_with_many_logins_leaves_no_tombstone(stores, monkeypatch):
    services, credentials = stores
    monkeypatch.setattr(service_store_module, "BATCH_SIZE", 2)
    record = services.create("tracker", settings(), created_by="admin")
    for index in range(5):
        save_login(credentials, record, f"user-{index}")
    assert services.delete("tracker")
    assert credentials.list_all(provider="mcp:tracker") == []
    assert services.scan() == ([], [])
    with pytest.raises(McpServiceVersionChanged):
        save_login(credentials, record)


def test_too_many_tools_are_refused(stores):
    services, _ = stores
    with pytest.raises(ValueError, match="at most"):
        services.create(
            "tracker", settings(tools=tuple(f"t{i}" for i in range(201))), created_by=None
        )


def _confidential(**changes) -> McpServiceSettings:
    return settings(auth="oauth", oauth=McpServiceOAuth(**{**_OAUTH.__dict__, **changes}))


def _saved_client_secret(credentials) -> tuple[str | None, str | None]:
    saved = credentials.get(SERVICE_PRINCIPAL, client_provider("tracker"), with_secret=True)
    if saved is None or saved.secret is None:
        return None, None
    return saved.secret.get("client_secret"), saved.metadata.get("service_version")


def test_a_kept_client_secret_moves_to_the_new_version_in_the_same_write(stores):
    services, credentials = stores
    record = services.create(
        "tracker", _confidential(), created_by="admin", client_secret=client_secret(credentials)
    )
    save_login(credentials, record)
    # New scopes: logins go, the same client keeps its secret under the new version.
    changed = services.update(
        "tracker",
        _confidential(scopes=("tickets:write",)),
        expected_version=record.credential_version,
        keep_client_secret=True,
    )
    assert changed.credentials_reset
    assert credentials.get("alice", member_provider("tracker")) is None
    assert _saved_client_secret(credentials) == (
        "client-secret",
        changed.record.credential_version,
    )
    # A metadata-only edit needs no secret and leaves it as it is.
    renamed = services.update("tracker", _confidential(scopes=("tickets:write",)))
    assert not renamed.credentials_reset
    assert _saved_client_secret(credentials)[0] == "client-secret"


def test_a_confidential_client_is_never_left_without_its_secret(stores):
    services, credentials = stores
    record = services.create(
        "tracker", _confidential(), created_by="admin", client_secret=client_secret(credentials)
    )
    with pytest.raises(McpClientSecretMissing):
        services.update("tracker", _confidential(token_url="https://other.example.com/token"))
    # Nothing changed: the refused write left the settings and the secret.
    assert services.get("tracker").credential_version == record.credential_version
    assert _saved_client_secret(credentials)[0] == "client-secret"
    # A secret saved for an older version cannot be kept.
    credentials.upsert(
        SERVICE_PRINCIPAL,
        client_provider("tracker"),
        secret={"client_secret": "old"},
        metadata=stamped({}, "0" * 32),
    )
    with pytest.raises(McpClientSecretMissing):
        services.update("tracker", _confidential(scopes=("x",)), keep_client_secret=True)


@pytest.mark.parametrize("interleaved", ["update", "delete-and-recreate"])
def test_a_secret_kept_for_settings_another_admin_replaced_is_refused(stores, interleaved):
    services, credentials = stores
    read = services.create(
        "tracker", _confidential(), created_by="admin", client_secret=client_secret(credentials)
    )
    # Another administrator commits between this request's read and its write.
    other = _confidential(token_url="https://evil.example.com/token", client_id="other")
    if interleaved == "update":
        services.update("tracker", other, client_secret=client_secret(credentials, "theirs"))
    else:
        services.delete("tracker")
        services.create(
            "tracker", other, created_by="b", client_secret=client_secret(credentials, "theirs")
        )
    current = services.get("tracker")
    with pytest.raises(McpServiceVersionChanged):
        services.update(
            "tracker",
            _confidential(scopes=("x",)),
            expected_version=read.credential_version,
            keep_client_secret=True,
        )
    with pytest.raises(McpServiceVersionChanged):
        services.update(
            "tracker",
            _confidential(),
            expected_version=read.credential_version,
            client_secret=client_secret(credentials, "mine"),
        )
    assert _saved_client_secret(credentials) == ("theirs", current.credential_version)
    assert services.get("tracker").oauth.token_url == "https://evil.example.com/token"
