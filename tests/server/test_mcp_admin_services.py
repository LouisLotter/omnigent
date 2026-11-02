"""MCP services administrators add in Settings: authority, changes and live use.

Accounts-mode app, real routes: an administrator, a member and an administrator's
runner holding the delegated token ``POST /v1/runners/{id}/token`` mints.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient
from mcp.types import CallToolResult, TextContent, Tool

from omnigent.server import mcp_destinations
from omnigent.server.mcp_destinations import DefaultMcpDestinationPolicy
from omnigent.server.mcp_registry import McpRegistry, McpRegistryConfig, McpService
from omnigent.server.routes.connections_base import ConnectionError
from omnigent.stores.credential_store.sqlalchemy_store import CredentialStore
from omnigent.stores.mcp_service_store import (
    SERVICE_PRINCIPAL,
    McpServiceOAuth,
    McpServiceSettings,
    McpServiceStore,
    client_provider,
    member_provider,
    stamped,
)
from tests.server.test_account_revocation import create_session
from tests.server.test_accounts import _build_accounts_app, _login
from tests.server.test_credential_store import _FakeCipher
from tests.server.test_delegated_runner_tokens import _delegated_runner

ADMIN = "/v1/mcp-registry/admin/services"
CATALOG = "/v1/mcp-registry/services"
CANARY = "canary-token-7f3a9c"


@pytest.fixture
def public_dns(monkeypatch):
    """Every ``*.example.com`` name resolves to a public address; nothing else resolves."""

    async def resolve(host: str, port: int):
        literal = mcp_destinations._ip_literal(host)
        if literal is not None:
            return [literal]
        return [ipaddress.ip_address("93.184.216.34")] if host.endswith(".example.com") else []

    monkeypatch.setattr(mcp_destinations, "resolve", resolve)


@pytest.fixture
def accounts(tmp_path, monkeypatch, public_dns):
    """(admin, alice, captured create_app kwargs incl. the registry)."""
    import omnigent.server.app as app_module

    monkeypatch.setattr(app_module, "_WEB_UI_DIST", tmp_path / "no-web-ui")
    monkeypatch.delenv("OMNIGENT_OIDC_ISSUER", raising=False)
    original = app_module.create_app
    captured: dict[str, Any] = {}

    def capture(**kwargs):
        db_url = str(kwargs["account_store"]._engine.url)
        registry = McpRegistry(
            McpRegistryConfig(
                services=[
                    McpService(
                        id="shipped",
                        title="Shipped",
                        url="https://shipped.example.com/mcp",
                        auth="bearer",
                        tools=["ping"],
                    )
                ]
            ),
            CredentialStore(db_url, _FakeCipher()),
            service_store=McpServiceStore(db_url),
            destination_policy=DefaultMcpDestinationPolicy(),
        )
        captured.update(kwargs, registry=registry, db_url=db_url)
        return original(**kwargs, mcp_registry=registry)

    monkeypatch.setattr(app_module, "create_app", capture)
    built = _build_accounts_app(tmp_path, monkeypatch, init_admin_password="admin-pw-12345")
    client = next(built)
    try:
        admin = _login(client, "admin", "admin-pw-12345")
        invite = admin.post("/auth/invite", json={}).json()["token"]
        alice = TestClient(client.app)
        registered = alice.post(
            "/auth/register",
            json={"invite": invite, "username": "alice", "password": "alice-pw-1234"},
        )
        assert registered.status_code == 200, registered.text
        yield admin, alice, captured
    finally:
        built.close()


def bearer_service(**changes) -> dict[str, Any]:
    return {
        "id": "tracker",
        "title": "Tracker",
        "description": "Tickets",
        "url": "https://tracker.example.com/mcp",
        "auth": "bearer",
        "tools": ["read_ticket"],
        **changes,
    }


def settings_of(body: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in body.items() if key != "id"}


def test_only_a_signed_in_administrator_manages_services(accounts):
    admin, alice, captured = accounts
    created = admin.post(ADMIN, json=bearer_service())
    assert created.status_code == 201, created.text
    assert created.json()["source"] == "admin"

    # An administrator's runner holds a token that acts as the administrator.
    session = create_session(admin)
    assert session.status_code == 201, session.text
    runner = _delegated_runner(admin, captured, session.json()["session_id"])
    assert runner.get("/v1/me").json()["is_admin"] is True

    attempts = [
        ("get", ADMIN, None),
        ("get", "/v1/mcp-registry/admin/presets", None),
        ("post", ADMIN, bearer_service(id="other")),
        (
            "put",
            f"{ADMIN}/tracker",
            settings_of(bearer_service(url="https://evil.example.com/mcp")),
        ),
        ("delete", f"{ADMIN}/tracker", None),
        ("post", f"{ADMIN}/tracker/discover", None),
    ]
    for method, path, body in attempts:
        kwargs = {"json": body} if body is not None else {}
        member = getattr(alice, method)(path, **kwargs)
        assert member.status_code == 403, (path, member.text)
        delegated = getattr(runner, method)(path, **kwargs)
        assert delegated.status_code == 403, (path, delegated.text)
        assert "runner credential cannot manage MCP services" in delegated.text
    # Nothing the refused calls asked for happened.
    listed = admin.get(ADMIN).json()["data"]
    assert [(s["id"], s["url"]) for s in listed] == [
        ("shipped", "https://shipped.example.com/mcp"),
        ("tracker", "https://tracker.example.com/mcp"),
    ]
    # Members still use the published service.
    assert [s["id"] for s in alice.get(CATALOG).json()["data"]] == ["shipped", "tracker"]


def test_services_change_live_and_logins_go_when_the_destination_changes(accounts):
    admin, alice, captured = accounts
    registry: McpRegistry = captured["registry"]
    credentials = registry.store
    assert admin.post(ADMIN, json=bearer_service()).status_code == 201
    # A second server process sharing the database sees the change without a restart.
    other_process = McpRegistry(
        McpRegistryConfig(),
        CredentialStore(captured["db_url"], _FakeCipher()),
        service_store=McpServiceStore(captured["db_url"]),
        refresh_seconds=0,
    )
    assert "tracker" in other_process.managed_services()

    saved = alice.put(f"{CATALOG}/tracker/connection", json={"token": CANARY})
    assert saved.status_code == 200, saved.text
    assert alice.get(CATALOG).json()["data"][1]["connected"] is True

    renamed = admin.put(
        f"{ADMIN}/tracker", json=settings_of(bearer_service(title="Renamed", tools=["a", "b"]))
    )
    assert renamed.status_code == 200, renamed.text
    assert renamed.json()["credentials_reset"] is False
    entry = alice.get(CATALOG).json()["data"][1]
    assert (entry["title"], entry["tools"], entry["connected"]) == ("Renamed", ["a", "b"], True)

    stale = other_process.managed_services()["tracker"]
    moved = admin.put(
        f"{ADMIN}/tracker",
        json=settings_of(bearer_service(url="https://moved.example.com/mcp")),
    )
    assert moved.status_code == 200, moved.text
    assert moved.json()["credentials_reset"] is True
    assert credentials.get("alice", member_provider("tracker")) is None
    assert alice.get(CATALOG).json()["data"][1]["connected"] is False

    alice.put(f"{CATALOG}/tracker/connection", json={"token": CANARY})
    # A process still holding the old settings never sends the new login there.
    assert stale.url == "https://tracker.example.com/mcp"
    with pytest.raises(ConnectionError, match="Connect"):
        asyncio.run(other_process.token(stale, "alice", None))
    asyncio.run(other_process.refresh())
    fresh = other_process.managed_services()["tracker"]
    assert fresh.url == "https://moved.example.com/mcp"
    assert asyncio.run(other_process.token(fresh, "alice", None)) == CANARY
    retyped = admin.put(
        f"{ADMIN}/tracker",
        json=settings_of(bearer_service(url="https://moved.example.com/mcp", auth="none")),
    )
    assert retyped.json()["credentials_reset"] is True
    assert credentials.get("alice", member_provider("tracker")) is None
    assert alice.get(CATALOG).json()["data"][1]["connected"] is True  # auth none

    assert admin.delete(f"{ADMIN}/tracker").status_code == 200
    assert [s["id"] for s in alice.get(CATALOG).json()["data"]] == ["shipped"]
    asyncio.run(other_process.refresh())
    assert "tracker" not in other_process.managed_services()
    assert admin.delete(f"{ADMIN}/tracker").status_code == 404


def test_a_page_added_service_is_used_through_the_gateway_at_once(accounts):
    admin, alice, _captured = accounts
    backend = AsyncMock()
    backend.list_tools.return_value = [
        Tool(name="read_ticket", inputSchema={"type": "object"}),
        Tool(name="delete_ticket", inputSchema={"type": "object"}),
    ]
    backend.call_tool.return_value = CallToolResult(content=[TextContent(type="text", text="ok")])
    alice.app.state.mcp_gateway_backend = backend
    assert admin.post(ADMIN, json=bearer_service(auth="none")).status_code == 201

    created = create_session(alice)
    session_id = created.json()["session_id"]
    attached = alice.post(
        f"/v1/sessions/{session_id}/agent/mcp-servers",
        json={"name": "tracker", "transport": "registry"},
    )
    assert attached.status_code == 200, attached.text
    listed = alice.post(
        "/v1/mcp/tracker",
        headers={"X-Omnigent-Session-Id": session_id},
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
    )
    assert [t["name"] for t in listed.json()["result"]["tools"]] == ["read_ticket"]

    assert admin.delete(f"{ADMIN}/tracker").status_code == 200
    gone = alice.post(
        "/v1/mcp/tracker",
        headers={"X-Omnigent-Session-Id": session_id},
        json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
    )
    assert gone.status_code == 403, gone.text


def test_registry_file_services_stay_read_only_and_keep_working(accounts):
    admin, alice, captured = accounts
    listed = {s["id"]: s for s in admin.get(ADMIN).json()["data"]}
    assert listed["shipped"]["source"] == "config"
    assert listed["shipped"]["editable"] is False
    assert admin.post(ADMIN, json=bearer_service(id="shipped")).status_code == 409
    assert admin.put(f"{ADMIN}/shipped", json=settings_of(bearer_service())).status_code == 409
    assert admin.delete(f"{ADMIN}/shipped").status_code == 409
    assert alice.put(f"{CATALOG}/shipped/connection", json={"token": "x"}).status_code == 200
    assert alice.get(CATALOG).json()["data"][0]["connected"] is True

    # A run-time row with a file service's ID stays hidden, and a login stamped
    # for it is never sent to the file service's destination.
    registry: McpRegistry = captured["registry"]
    record = registry.service_store.create(
        "shipped",
        McpServiceSettings(title="x", url="https://x.example.com/mcp", auth="bearer", timeout=5),
        created_by="admin",
    )
    registry.reload()
    assert registry.lookup("shipped").url == "https://shipped.example.com/mcp"
    registry.store.upsert(
        "bob",
        member_provider("shipped"),
        secret={"access_token": CANARY},
        metadata=stamped({}, record.credential_version),
    )
    with pytest.raises(ConnectionError, match="Connect"):
        asyncio.run(registry.token(registry.services["shipped"], "bob", None))


def test_destination_rules_and_input_checks_refuse_bad_services(accounts):
    admin, _alice, _captured = accounts
    refused = {
        "http": bearer_service(url="http://tracker.example.com/mcp"),
        "credentials": bearer_service(url="https://u:p@tracker.example.com/mcp"),
        "query": bearer_service(url="https://tracker.example.com/mcp?key=1"),
        "private": bearer_service(url="https://10.0.0.1/mcp"),
        "unresolvable": bearer_service(url="https://tracker.invalid-tld.test/mcp"),
        "separator": bearer_service(id="a__b"),
        "oauth fields": bearer_service(oauth={"client_id": "x"}),
    }
    for name, body in refused.items():
        response = admin.post(ADMIN, json=body)
        assert response.status_code in (400, 422), (name, response.text)
        detail = response.json()["detail"]
        if isinstance(detail, str):
            # Explanations name the rule, never the URL.
            assert "example.com" not in detail and "10.0.0.1" not in detail, detail
    assert [s["id"] for s in admin.get(ADMIN).json()["data"]] == ["shipped"]


def test_no_login_reaches_a_response_or_a_log(accounts, caplog):
    admin, alice, _captured = accounts
    caplog.set_level(logging.DEBUG)
    responses = [admin.post(ADMIN, json=bearer_service())]
    responses.append(alice.put(f"{CATALOG}/tracker/connection", json={"token": CANARY}))
    responses += [
        alice.get(CATALOG),
        admin.get(ADMIN),
        alice.post(f"{CATALOG}/tracker/test"),
        admin.put(
            f"{ADMIN}/tracker", json=settings_of(bearer_service(url="https://b.example.com/mcp"))
        ),
        alice.get(CATALOG),
    ]
    for response in responses:
        assert CANARY not in response.text
    assert CANARY not in caplog.text


# ── Single-user app: the admin routes without accounts ──────────────────────


@pytest.fixture
def local_admin(db_uri, monkeypatch, public_dns):
    """An app whose only user may manage services, and its registry."""
    import httpx
    from fastapi import FastAPI

    from omnigent.server.routes.mcp_registry import create_mcp_registry_router

    monkeypatch.setenv("OMNIGENT_MCP_OAUTH_STATE_SECRET", "test-signing-key-32-characters-long")
    registry = McpRegistry(
        McpRegistryConfig(public_url="https://omnigent.example.com"),
        CredentialStore(db_uri, _FakeCipher()),
        service_store=McpServiceStore(db_uri),
        destination_policy=DefaultMcpDestinationPolicy(),
    )
    app = FastAPI()
    app.include_router(create_mcp_registry_router(registry, None), prefix="/v1")
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")
    return client, registry


def manual_oauth(**oauth) -> dict[str, Any]:
    return bearer_service(
        auth="oauth",
        oauth={
            "client_id": "ours",
            "client_secret": "our-secret",
            "authorize_url": "https://auth.example.com/authorize",
            "token_url": "https://auth.example.com/token",
            **oauth,
        },
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("interleaved", ["update", "delete-and-recreate"])
async def test_an_edit_racing_another_admins_change_never_moves_a_client_secret(
    local_admin, interleaved
):
    client, registry = local_admin
    store = registry.service_store
    created = await client.post(ADMIN, json=manual_oauth())
    assert created.status_code == 201, created.text
    real_update = store.update

    def theirs_first(service_id, settings, **kwargs):
        # Another administrator repoints the service after this request read it.
        current = store.get(service_id)
        other = McpServiceSettings(
            title="Theirs",
            url=current.url,
            auth="oauth",
            timeout=60,
            tools=("read_ticket",),
            oauth=McpServiceOAuth(
                authorize_url="https://evil.example.com/authorize",
                token_url="https://evil.example.com/token",
                client_id="theirs",
                token_endpoint_auth="client_secret_post",
            ),
        )
        sealed = registry.store.seal(
            SERVICE_PRINCIPAL, client_provider(service_id), {"client_secret": "their-secret"}
        )
        if interleaved == "update":
            real_update(service_id, other, client_secret=sealed)
        else:
            store.delete(service_id)
            store.create(service_id, other, created_by="b", client_secret=sealed)
        return real_update(service_id, settings, **kwargs)

    store.update = theirs_first
    # Keeps the saved secret (no new one entered) while changing the scopes.
    edited = await client.put(
        f"{ADMIN}/tracker",
        json=settings_of(manual_oauth(client_secret=None, scopes=["read"])),
    )
    assert edited.status_code == 409, edited.text
    store.update = real_update
    await registry.refresh(force=True)
    fresh = registry.managed_services()["tracker"]
    assert fresh.oauth.token_url == "https://evil.example.com/token"
    # The other administrator's secret, never this request's or the original one.
    assert registry.client_secret(fresh) == "their-secret"


SECRET = "canary-client-secret-5d1e"
TOKEN = "canary-bearer-token-9b2c"


@pytest.mark.asyncio
async def test_validation_errors_never_repeat_a_submitted_secret(local_admin, caplog):
    client, _registry = local_admin
    caplog.set_level(logging.DEBUG)
    assert (await client.post(ADMIN, json=bearer_service(auth="none"))).status_code == 201
    secret_bodies = {
        "oversized secret": manual_oauth(client_secret=SECRET + "x" * 4096),
        "secret of the wrong type": manual_oauth(client_secret=[SECRET]),
        "bad sibling field": {**manual_oauth(client_secret=SECRET), "timeout": 999},
        "missing parent field": {
            key: value
            for key, value in manual_oauth(client_secret=SECRET).items()
            if key != "title"
        },
        "unknown field": manual_oauth(client_secret=SECRET, client_secret_extra=SECRET),
        "oauth not an object": bearer_service(auth="oauth", oauth=SECRET),
    }
    responses = []
    for name, body in secret_bodies.items():
        created = await client.post(ADMIN, json=body)
        updated = await client.put(f"{ADMIN}/tracker", json=settings_of(body))
        for response in (created, updated):
            assert response.status_code == 422, (name, response.text)
            assert all({"loc", "msg", "type"} >= set(e) for e in response.json()["detail"])
            responses.append(response)
    token_bodies = [
        {"token": TOKEN + "x" * 16000},
        {"token": {"value": TOKEN}},
        {"token": TOKEN, "extra": TOKEN},
        {"tokn": TOKEN},
    ]
    for body in token_bodies:
        response = await client.put("/v1/mcp-registry/services/tracker/connection", json=body)
        # Malformed bodies are 422; a valid one fails later (this service takes no token).
        assert response.status_code in (400, 422), response.text
        responses.append(response)
    for raw in (f'"{SECRET}"', f'{{"oauth": "{SECRET}"', f'["{TOKEN}"]'):
        response = await client.post(
            ADMIN, content=raw, headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422, response.text
        responses.append(response)
    too_large = await client.post(
        ADMIN, content=b"{" + b" " * 70_000 + b"}", headers={"Content-Type": "application/json"}
    )
    assert too_large.status_code == 413
    for response in responses:
        assert SECRET not in response.text and TOKEN not in response.text, response.text
    assert SECRET not in caplog.text and TOKEN not in caplog.text


@pytest.mark.asyncio
async def test_only_json_requests_reach_the_secret_bearing_routes(local_admin):
    """A cross-site simple request (``text/plain``, no preflight) never changes a service."""
    import json as jsonlib

    client, registry = local_admin
    body = jsonlib.dumps(bearer_service(auth="none"))
    cross_site = {
        "Origin": "https://untrusted.example",
        "Sec-Fetch-Site": "cross-site",
        "Sec-Fetch-Mode": "no-cors",
    }
    for headers in (
        {**cross_site, "Content-Type": "text/plain"},
        {**cross_site, "Content-Type": "application/x-www-form-urlencoded"},
        {},  # no content type at all
    ):
        created = await client.post(ADMIN, content=body, headers=headers)
        assert created.status_code == 415, created.text
    await registry.refresh(force=True)
    assert registry.managed_services() == {}

    assert (
        await client.post(ADMIN, content=body, headers={"Content-Type": "application/json"})
    ).status_code == 201
    other = jsonlib.dumps(bearer_service(id="other", auth="bearer"))
    assert (
        await client.post(
            ADMIN, content=other, headers={"Content-Type": "application/vnd.api+json"}
        )
    ).status_code == 201

    retitled = jsonlib.dumps(settings_of(bearer_service(auth="none", title="Changed")))
    refused = await client.put(
        f"{ADMIN}/tracker", content=retitled, headers={**cross_site, "Content-Type": "text/plain"}
    )
    assert refused.status_code == 415
    token = jsonlib.dumps({"token": "canary-bearer-token"})
    for headers in ({**cross_site, "Content-Type": "text/plain"}, {}):
        saved = await client.put(
            "/v1/mcp-registry/services/other/connection", content=token, headers=headers
        )
        assert saved.status_code == 415, saved.text
    assert registry.store.get("local", member_provider("other")) is None
    await registry.refresh(force=True)
    assert registry.managed_services()["tracker"].title == "Tracker"

    assert (
        await client.put(
            f"{ADMIN}/tracker",
            content=retitled,
            headers={"Content-Type": "application/json; charset=utf-8"},
        )
    ).status_code == 200
    assert (
        await client.put(
            "/v1/mcp-registry/services/other/connection",
            content=token,
            headers={"Content-Type": "application/json"},
        )
    ).status_code == 200
