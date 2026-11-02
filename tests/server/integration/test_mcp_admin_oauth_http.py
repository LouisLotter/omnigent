"""Add an OAuth MCP server from the page against a real HTTP service.

The demo service publishes protected-resource and authorization-server metadata
and accepts dynamic client registration, as hosted MCP connectors do.
"""

from __future__ import annotations

import logging
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest

from omnigent.server.mcp_destinations import DefaultMcpDestinationPolicy
from omnigent.server.mcp_registry import McpRegistry, McpRegistryConfig
from omnigent.stores.credential_store.sqlalchemy_store import CredentialStore
from omnigent.stores.mcp_service_store import (
    SERVICE_PRINCIPAL,
    McpServiceStore,
    client_provider,
    member_provider,
)
from tests.e2e.test_mcp_registry import demo_mcp as demo_mcp
from tests.server.test_credential_store import _FakeCipher

ADMIN = "/v1/mcp-registry/admin/services"


@pytest.fixture
def app(runtime_init, db_uri, tmp_path, demo_mcp, monkeypatch):
    from omnigent.runtime import get_agent_cache, get_agent_store, get_conversation_store
    from omnigent.server.app import create_app
    from omnigent.stores.artifact_store.local import LocalArtifactStore
    from omnigent.stores.file_store.sqlalchemy_store import SqlAlchemyFileStore

    monkeypatch.setenv("OMNIGENT_LOCAL_SINGLE_USER", "1")
    monkeypatch.setenv(
        "OMNIGENT_MCP_OAUTH_STATE_SECRET", "local-test-signing-key-32-characters-long"
    )
    registry = McpRegistry(
        McpRegistryConfig(public_url="http://localhost"),
        CredentialStore(db_uri, _FakeCipher()),
        service_store=McpServiceStore(db_uri),
        # The demo listens on loopback over plain HTTP.
        destination_policy=DefaultMcpDestinationPolicy(allow_private=True),
    )
    return create_app(
        agent_store=get_agent_store(),
        conversation_store=get_conversation_store(),
        agent_cache=get_agent_cache(),
        artifact_store=LocalArtifactStore(str(tmp_path / "artifacts")),
        file_store=SqlAlchemyFileStore(db_uri),
        mcp_registry=registry,
    )


async def sign_in(client: httpx.AsyncClient, service_id: str, demo_mcp: str) -> str:
    start = await client.get(f"/v1/connections/mcp-{service_id}/connect")
    assert start.status_code in (302, 307), start.text
    location = start.headers["location"]
    fields = {k: v[0] for k, v in parse_qs(urlsplit(location).query).items()}
    async with httpx.AsyncClient() as provider:
        consent = await provider.post(
            demo_mcp + "/authorize", data={**fields, "user": "alice@example.test"}
        )
    assert consent.status_code == 303, consent.text
    callback = await client.get(consent.headers["location"])
    assert f"mcp-{service_id}=connected" in callback.headers["location"], callback.headers
    return location


@pytest.mark.asyncio
async def test_discover_register_sign_in_allow_and_refresh(client, app, demo_mcp, caplog):
    caplog.set_level(logging.DEBUG)
    registry: McpRegistry = app.state.mcp_registry
    responses: list[httpx.Response] = []

    created = await client.post(
        ADMIN,
        json={
            "id": "tracker",
            "title": "Demo tracker",
            "url": demo_mcp + "/mcp",
            "auth": "oauth",
        },
    )
    responses.append(created)
    assert created.status_code == 201, created.text
    oauth = created.json()["oauth"]
    assert oauth["authorize_url"] == demo_mcp + "/authorize"
    assert oauth["token_url"] == demo_mcp + "/token"
    assert oauth["registration"] == "dynamic"
    assert oauth["client_id"].startswith("dyn-")
    assert oauth["has_client_secret"] is True
    assert oauth["scopes"] == ["tickets:read", "offline_access"]
    assert created.json()["draft"] is True
    client_secret = registry.store.get(
        SERVICE_PRINCIPAL, client_provider("tracker"), with_secret=True
    ).secret["client_secret"]

    # The new service's sign-in route exists without a restart.
    authorize = await sign_in(client, "tracker", demo_mcp)
    assert "resource=" in authorize and "code_challenge_method=S256" in authorize

    discovered = await client.post(f"{ADMIN}/tracker/discover")
    responses.append(discovered)
    assert discovered.status_code == 200, discovered.text
    names = sorted(tool["name"] for tool in discovered.json()["tools"])
    assert names == ["delete_ticket", "read_ticket", "whoami"]
    assert discovered.json()["allowed"] == []

    allowed = await client.put(
        f"{ADMIN}/tracker",
        json={
            "title": "Demo tracker",
            "url": demo_mcp + "/mcp",
            "auth": "oauth",
            "tools": ["whoami", "read_ticket"],
        },
    )
    responses.append(allowed)
    assert allowed.status_code == 200, allowed.text
    assert allowed.json()["credentials_reset"] is False
    assert allowed.json()["draft"] is False
    # Same authorization server: the registered client is kept.
    assert allowed.json()["oauth"]["client_id"] == oauth["client_id"]

    catalog = await client.get("/v1/mcp-registry/services")
    responses.append(catalog)
    entry = catalog.json()["data"][0]
    assert (entry["id"], entry["connected"], entry["tools"]) == (
        "tracker",
        True,
        ["read_ticket", "whoami"],
    )
    tested = await client.post("/v1/mcp-registry/services/tracker/test")
    responses.append(tested)
    assert sorted(tested.json()["tools"]) == ["read_ticket", "whoami"]

    # A refresh authenticates with the sealed client secret.
    store = registry.store
    grant = store.get("local", member_provider("tracker"), with_secret=True)
    store.update_secret(
        "local",
        member_provider("tracker"),
        secret=grant.secret,
        metadata={**grant.metadata, "expires_at": 0},
    )
    result = await registry.execute("tracker", "local", app.state, tool="whoami")
    assert '"refresh_count":1' in result.content[0].text.replace(" ", "")
    refreshed = store.get("local", member_provider("tracker"), with_secret=True)

    # Moving the service removes the login and the client; a new client is registered.
    moved = await client.put(
        f"{ADMIN}/tracker",
        json={
            "title": "Demo tracker",
            "url": demo_mcp.replace("127.0.0.1", "localhost") + "/mcp",
            "auth": "oauth",
            "tools": ["whoami"],
        },
    )
    responses.append(moved)
    assert moved.status_code == 200, moved.text
    assert moved.json()["credentials_reset"] is True
    assert moved.json()["oauth"]["client_id"] != oauth["client_id"]
    assert store.get("local", member_provider("tracker")) is None
    removed = await client.delete(f"{ADMIN}/tracker")
    responses.append(removed)
    assert store.get(SERVICE_PRINCIPAL, client_provider("tracker")) is None

    secrets_seen = [
        client_secret,
        grant.secret["access_token"],
        grant.secret["refresh_token"],
        refreshed.secret["access_token"],
        refreshed.secret["refresh_token"],
    ]
    for response in responses:
        for secret in secrets_seen:
            assert secret not in response.text
    for secret in secrets_seen:
        assert secret not in caplog.text


@pytest.mark.asyncio
async def test_strict_rules_refuse_a_loopback_service(client, app, demo_mcp):
    app.state.mcp_registry.destination_policy = DefaultMcpDestinationPolicy()
    refused = await client.post(
        ADMIN,
        json={"id": "tracker", "title": "Demo", "url": demo_mcp + "/mcp", "auth": "none"},
    )
    assert refused.status_code == 400
    assert "destination rule" in refused.json()["detail"]
