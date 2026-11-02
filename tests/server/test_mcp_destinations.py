"""Destination rules for run-time MCP services: at add time and at connect time."""

from __future__ import annotations

import ipaddress
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from omnigent.server import mcp_destinations
from omnigent.server.mcp_destinations import (
    DefaultMcpDestinationPolicy,
    McpDestinationError,
    check_destination,
    destination_refused,
    guarded_client,
)

STRICT = DefaultMcpDestinationPolicy()
PRIVATE_OK = DefaultMcpDestinationPolicy(allow_private=True)


@pytest.mark.parametrize(
    ("url", "problem"),
    [
        ("https://mcp.example.com/mcp", None),
        ("https://mcp.example.com:8443/mcp", None),
        ("https://8.8.8.8/mcp", None),
        ("http://mcp.example.com/mcp", "https"),
        ("ftp://mcp.example.com/mcp", "https"),
        ("https://user:pw@mcp.example.com/mcp", "credentials"),
        ("https://token@mcp.example.com/mcp", "credentials"),
        ("https://mcp.example.com/mcp?api_key=1", "query"),
        ("https://mcp.example.com/mcp?", "query"),
        ("https://mcp.example.com/mcp#frag", "query"),
        ("https://127.0.0.1/mcp", "public address"),
        ("https://10.1.2.3/mcp", "public address"),
        ("https://169.254.169.254/latest", "public address"),
        ("https://100.64.0.1/mcp", "public address"),
        ("https://[::1]/mcp", "public address"),
        ("https://[fd00::1]/mcp", "public address"),
        ("https://[::ffff:127.0.0.1]/mcp", "public address"),
        ("https://[64:ff9b::a00:1]/mcp", "public address"),
        ("https://localhost/mcp", "top-level domain"),
        ("https://intranet/mcp", "top-level domain"),
        ("https://app.localhost/mcp", "local or internal"),
        ("https://printer.local/mcp", "local or internal"),
        ("https://db.internal/mcp", "local or internal"),
        ("https://mcp.example.com/m cp", "plain ASCII"),
        ("https://mcp.exämple.com/mcp", "plain ASCII"),
        ("https://mcp.example.com:0/mcp", "port 0"),
        ("https://mcp.example.com:99999/mcp", "well-formed"),
        ("https:///mcp", "name a host"),
        ("https://mcp.example.com/" + "a" * 2100, "at most"),
    ],
)
def test_url_rules(url, problem):
    found = STRICT.url_problem(url)
    if problem is None:
        assert found is None
    else:
        assert found is not None and problem in found
        assert url not in found


def test_private_destinations_are_an_explicit_opt_in(monkeypatch):
    assert PRIVATE_OK.url_problem("http://127.0.0.1:8080/mcp") is None
    assert PRIVATE_OK.url_problem("http://user:pw@127.0.0.1/mcp") is not None
    monkeypatch.setenv(mcp_destinations.ALLOW_PRIVATE_DESTINATIONS_ENV, "1")
    assert DefaultMcpDestinationPolicy.from_env().allow_private
    monkeypatch.setenv(mcp_destinations.ALLOW_PRIVATE_DESTINATIONS_ENV, "yes")
    assert not DefaultMcpDestinationPolicy.from_env().allow_private


def fake_dns(monkeypatch, answers: dict[str, list[str]]) -> None:
    async def resolve(host: str, port: int):
        literal = mcp_destinations._ip_literal(host)
        if literal is not None:
            return [literal]
        return [ipaddress.ip_address(a) for a in answers.get(host, [])]

    monkeypatch.setattr(mcp_destinations, "resolve", resolve)


@pytest.mark.asyncio
async def test_a_name_resolving_to_a_private_address_is_refused_when_added(monkeypatch):
    fake_dns(
        monkeypatch,
        {
            "public.example.com": ["93.184.216.34"],
            "sneaky.example.com": ["10.0.0.5"],
            "mixed.example.com": ["93.184.216.34", "127.0.0.1"],
            "mapped.example.com": ["::ffff:192.168.1.1"],
        },
    )
    await check_destination(STRICT, "https://public.example.com/mcp", what="The server URL")
    for host in ("sneaky", "mixed", "mapped"):
        with pytest.raises(McpDestinationError, match="refuse") as caught:
            await check_destination(
                STRICT, f"https://{host}.example.com/mcp", what="The server URL"
            )
        assert host not in str(caught.value)
    with pytest.raises(McpDestinationError, match="does not resolve"):
        await check_destination(STRICT, "https://missing.example.com/mcp", what="The server URL")
    with pytest.raises(McpDestinationError, match="https"):
        await check_destination(STRICT, "http://public.example.com/mcp", what="The server URL")


class _Recorder(BaseHTTPRequestHandler):
    hosts: list[str] = []

    def do_GET(self):
        type(self).hosts.append(self.headers.get("Host", ""))
        if self.path == "/redirect":
            self.send_response(302)
            self.send_header("Location", "http://169.254.169.254/latest/meta-data")
            self.end_headers()
            return
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, *args):
        pass


@pytest.fixture
def local_server():
    _Recorder.hosts = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Recorder)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()


@pytest.mark.asyncio
async def test_connect_time_check_refuses_a_rebound_name_before_connecting(
    monkeypatch, local_server
):
    # Public when the service was saved; the resolver now answers with loopback.
    fake_dns(monkeypatch, {"rebind.example.com": ["127.0.0.1"]})
    async with guarded_client(STRICT, timeout=5) as client:
        with pytest.raises(httpx.ConnectError) as caught:
            await client.get(f"http://rebind.example.com:{local_server}/")
    assert destination_refused(caught.value)
    assert _Recorder.hosts == []


@pytest.mark.asyncio
async def test_guarded_client_pins_the_checked_address_and_never_redirects(
    monkeypatch, local_server
):
    fake_dns(monkeypatch, {"mcp.test": ["127.0.0.1"]})
    async with guarded_client(PRIVATE_OK, timeout=5) as client:
        response = await client.get(f"http://mcp.test:{local_server}/")
        redirected = await client.get(f"http://mcp.test:{local_server}/redirect")
    assert response.text == "ok"
    # The request still names the host it was addressed to.
    assert _Recorder.hosts[0] == f"mcp.test:{local_server}"
    assert redirected.status_code == 302
    assert len(_Recorder.hosts) == 2


@pytest.mark.asyncio
async def test_guarded_client_ignores_proxies_from_the_environment(monkeypatch, local_server):
    monkeypatch.setenv("HTTP_PROXY", "http://10.255.255.1:3128")
    monkeypatch.setenv("ALL_PROXY", "http://10.255.255.1:3128")
    fake_dns(monkeypatch, {"mcp.test": ["127.0.0.1"]})
    async with guarded_client(PRIVATE_OK, timeout=5) as client:
        assert (await client.get(f"http://mcp.test:{local_server}/")).text == "ok"
