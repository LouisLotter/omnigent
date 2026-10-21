"""Tests for the sandbox-side GitHub credential helper + host git and gh setup."""

from __future__ import annotations

import io
import os
import shlex
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import NoReturn

import httpx
import pytest
import yaml

import omnigent.git_credential_github as h
from omnigent.host.credential_broker import SOCKET_ENV_VAR
from omnigent.host.identity import HOST_TOKEN_ENV_VAR


@pytest.fixture(autouse=True)
def _force_sandbox(monkeypatch: pytest.MonkeyPatch) -> None:
    # The broker host integrations are sandbox-only; default every test to the
    # in-sandbox path. The not-in-sandbox no-op test clears IS_SANDBOX explicitly.
    monkeypatch.setenv("IS_SANDBOX", "1")


def test_get_prints_credentials_for_github(monkeypatch: pytest.MonkeyPatch) -> None:
    cred = {"connected": True, "username": "x-access-token", "token": "T"}
    monkeypatch.setattr(h, "_fetch", lambda *a, **k: cred)
    monkeypatch.setattr(sys, "stdin", io.StringIO("protocol=https\nhost=github.com\n\n"))
    out = io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    rc = h.main(["--server", "http://s", "--host-id", "hid", "--host-token", "tok", "get"])
    assert rc == 0
    assert "username=x-access-token" in out.getvalue()
    assert "password=T" in out.getvalue()


def test_get_declines_for_non_github_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO("protocol=https\nhost=gitlab.com\n\n"))
    out = io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    rc = h.main(["--server", "http://s", "--host-id", "hid", "--host-token", "tok", "get"])
    assert rc == 0 and out.getvalue() == ""  # declined → git falls through


def test_get_fails_closed_on_missing_host_or_non_https(monkeypatch: pytest.MonkeyPatch) -> None:
    # Fail closed: an empty/missing host or a non-https protocol must decline
    # (no output) so the brokered token never leaks to an unintended host.
    for stdin in ("protocol=https\n\n", "protocol=http\nhost=github.com\n\n", "\n"):
        monkeypatch.setattr(sys, "stdin", io.StringIO(stdin))
        out = io.StringIO()
        monkeypatch.setattr(sys, "stdout", out)
        rc = h.main(["--server", "http://s", "--host-id", "h", "--host-token", "t", "get"])
        assert rc == 0 and out.getvalue() == ""


def test_store_and_erase_are_noops(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    for op in ("store", "erase"):
        assert h.main(["--server", "http://s", "--host-id", "h", "--host-token", "t", op]) == 0


_HELPER_KEY = "credential.https://github.com.helper"
_LEGACY_CLEANUP = ("removed leftovers",)


@pytest.fixture
def started_brokers(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Path, Callable]]:
    """Record broker starts instead of binding sockets; the broker has its own tests."""
    started: list[tuple[Path, Callable]] = []

    def fake_start(path: Path, resolve: Callable) -> object:
        started.append((path, resolve))
        return _FAKE_BROKER

    monkeypatch.setattr(h.credential_broker, "start_credential_broker", fake_start)
    # A value inherited from a parent process is replaced or cleared.
    monkeypatch.setenv(SOCKET_ENV_VAR, "/stale/inherited.sock")
    return started


_FAKE_BROKER = object()


def _record_git_config(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, ...]]:
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(h, "_git_config", lambda *args: calls.append(args))
    monkeypatch.setattr(h, "_remove_leftover_credentials", lambda: calls.append(_LEGACY_CLEANUP))
    return calls


def _socket_helper_installed(calls: list[tuple[str, ...]], socket_path: Path) -> bool:
    helper = (
        f"!{shlex.quote(sys.executable)} -I -m omnigent.git_credential_github "
        f"--socket {shlex.quote(str(socket_path))}"
    )
    return calls[-2:] == [("--replace-all", _HELPER_KEY, ""), ("--add", _HELPER_KEY, helper)]


def test_configure_host_git_serves_a_connected_owner_through_the_broker(
    monkeypatch: pytest.MonkeyPatch, started_brokers: list[tuple[Path, Callable]]
) -> None:
    monkeypatch.setenv(HOST_TOKEN_ENV_VAR, "launch-tok")
    calls = _record_git_config(monkeypatch)
    cred = {"connected": True, "owner": "alice@example.com", "login": "octo"}
    monkeypatch.setattr(h, "_fetch", lambda *a, **k: cred)

    assert h.configure_host_git("http://srv", "host1") is _FAKE_BROKER

    socket_path = h.credential_broker.broker_socket_path("host1")
    assert [path for path, _ in started_brokers] == [socket_path]
    assert os.environ[SOCKET_ENV_VAR] == str(socket_path)
    # Leftovers go first; the github.com chain is reset BEFORE the socket helper
    # is added, so a wider-scope $GIT_TOKEN helper cannot shadow the broker.
    assert calls[0] == _LEGACY_CLEANUP
    assert _socket_helper_installed(calls[:3], socket_path)
    assert calls[3:] == [("user.email", "alice@example.com"), ("user.name", "octo")]
    assert not any("launch-tok" in arg for call in calls for arg in call)


def test_configure_host_git_fails_closed_when_probe_fails(
    monkeypatch: pytest.MonkeyPatch, started_brokers: list[tuple[Path, Callable]]
) -> None:
    # A transient probe failure is indistinguishable from "not linked", so the
    # broker still takes over github.com rather than fall back to a shared
    # $GIT_TOKEN identity; with no confirmed owner, no commit identity is set.
    monkeypatch.setenv(HOST_TOKEN_ENV_VAR, "launch-tok")
    calls = _record_git_config(monkeypatch)
    monkeypatch.setattr(h, "_fetch", lambda *a, **k: None)

    assert h.configure_host_git("http://srv", "host1") is _FAKE_BROKER

    socket_path = h.credential_broker.broker_socket_path("host1")
    assert [path for path, _ in started_brokers] == [socket_path]
    assert calls[0] == _LEGACY_CLEANUP
    assert len(calls) == 3 and _socket_helper_installed(calls, socket_path)


def test_configure_host_git_fails_closed_when_broker_cannot_start(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv(HOST_TOKEN_ENV_VAR, "launch-tok")
    monkeypatch.setenv(SOCKET_ENV_VAR, "/stale/inherited.sock")
    calls = _record_git_config(monkeypatch)
    monkeypatch.setattr(h, "_fetch", lambda *a, **k: {"connected": True})

    def refuse(*_args: object) -> NoReturn:
        raise RuntimeError("refusing to use /tmp/x: is a symlink")

    monkeypatch.setattr(h.credential_broker, "start_credential_broker", refuse)

    assert h.configure_host_git("http://srv", "host1") is None

    # The helper still names the socket and declines, so git cannot fall back
    # to the shared identity; gh gets no socket to ask.
    assert _socket_helper_installed(calls, h.credential_broker.broker_socket_path("host1"))
    assert SOCKET_ENV_VAR not in os.environ
    assert "did not start" in caplog.text


def test_configure_host_git_noop_without_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(HOST_TOKEN_ENV_VAR, raising=False)
    calls: list[object] = []
    monkeypatch.setattr(h.subprocess, "run", lambda *a, **k: calls.append(a))
    assert h.configure_host_git("http://srv", "host1") is None
    assert calls == []  # no token → nothing configured


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, json={"connected": False}),
        httpx.Response(404, json={"detail": "unknown credential provider"}),
    ],
    ids=["not-linked", "github-not-configured"],
)
def test_configure_host_git_clears_stale_broker_when_not_connected(
    monkeypatch: pytest.MonkeyPatch,
    started_brokers: list[tuple[Path, Callable]],
    response: httpx.Response,
) -> None:
    # Scoping: a confirmed not-connected owner (shared-$GIT_TOKEN / local model)
    # must NOT get the broker. It must also CLEAR any stale broker helper a prior
    # (inconclusive) probe installed — otherwise the reset it left strands
    # in-session git behind a declining broker. So after removing leftovers the
    # one write is the unset that restores the ambient/shared helper.
    monkeypatch.setenv(HOST_TOKEN_ENV_VAR, "launch-tok")
    calls = _record_git_config(monkeypatch)
    monkeypatch.setattr(h.httpx, "get", lambda *a, **k: response)

    assert h.configure_host_git("http://srv", "host1") is None

    assert calls == [_LEGACY_CLEANUP, ("--unset-all", _HELPER_KEY)]
    assert started_brokers == []
    assert SOCKET_ENV_VAR not in os.environ


@pytest.mark.parametrize("configure", [h.configure_clone_credentials, h.configure_host_git])
def test_unknown_404_keeps_broker_authoritative(
    monkeypatch: pytest.MonkeyPatch,
    started_brokers: list[tuple[Path, Callable]],
    configure: Callable[[str, str], object],
) -> None:
    # A 404 that is not the server's explicit "unknown credential provider"
    # (a proxy, a missing route) is inconclusive, not "GitHub not configured".
    monkeypatch.setenv(HOST_TOKEN_ENV_VAR, "launch-tok")
    monkeypatch.setenv("GIT_TOKEN", "shared-test-token")
    response = httpx.Response(404, json={"detail": "Not Found"})
    monkeypatch.setattr(h.httpx, "get", lambda *a, **k: response)
    calls = _record_git_config(monkeypatch)

    configure("http://srv", "host1")

    assert calls[-2] == ("--replace-all", _HELPER_KEY, "")
    assert calls[-1][:2] == ("--add", _HELPER_KEY)
    assert "omnigent.git_credential_github" in calls[-1][2]


def test_host_integrations_are_noops_outside_sandbox(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, started_brokers: list[tuple[Path, Callable]]
) -> None:
    # NOT in a managed sandbox → a complete no-op even for a connected owner, so
    # a local `omnigent host` never touches the developer's real ~/.gitconfig or
    # ~/.config/gh, and starts no broker.
    monkeypatch.delenv("IS_SANDBOX", raising=False)
    monkeypatch.setenv(HOST_TOKEN_ENV_VAR, "launch-tok")
    monkeypatch.setenv("GH_CONFIG_DIR", str(tmp_path / "gh"))
    (tmp_path / "gh").mkdir()
    hosts = "github.com:\n    oauth_token: dev-own-token\n    user: dev\n"
    (tmp_path / "gh" / "hosts.yml").write_text(hosts)
    calls: list[object] = []
    monkeypatch.setattr(h.subprocess, "run", lambda *a, **k: calls.append(a))
    monkeypatch.setattr(
        h, "_fetch", lambda *a, **k: {"connected": True, "token": "t", "login": "o"}
    )
    assert h.configure_host_git("http://srv", "host1") is None
    assert calls == []
    assert started_brokers == []
    assert (tmp_path / "gh" / "hosts.yml").read_text() == hosts


_LEGACY_HELPER = (
    "!python3 -m omnigent.git_credential_github --server http://srv "
    "--host-id host1 --host-token old-launch-token"
)


def _git_file(config: Path, *args: str) -> list[str]:
    result = subprocess.run(
        ["git", "config", "--file", str(config), *args],
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.stdout.splitlines()


@pytest.fixture
def git_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """A HOME for real git, with the default global config locations."""
    home = tmp_path / "home"
    (home / ".config" / "git").mkdir(parents=True)
    for name in ("GIT_CONFIG_GLOBAL", "XDG_CONFIG_HOME"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GH_CONFIG_DIR", str(tmp_path / "gh"))
    return home


def test_remove_leftover_credentials_drops_only_the_launch_token_helper(git_home: Path) -> None:
    # Real git. An earlier host's helper that baked the launch token goes from
    # both global files git reads (it wrote the XDG one when ~/.gitconfig was
    # absent); other helpers and settings stay.
    home_config = git_home / ".gitconfig"
    xdg_config = git_home / ".config" / "git" / "config"
    for value in ("", _LEGACY_HELPER, "!other-helper"):
        _git_file(home_config, "--add", _HELPER_KEY, value)
    _git_file(xdg_config, "--add", _HELPER_KEY, _LEGACY_HELPER)
    _git_file(xdg_config, "core.editor", "vi")

    h._remove_leftover_credentials()

    assert _git_file(home_config, "--get-all", _HELPER_KEY) == ["", "!other-helper"]
    assert _git_file(xdg_config, "--get-all", _HELPER_KEY) == []
    assert _git_file(xdg_config, "core.editor") == ["vi"]
    for config in (home_config, xdg_config):
        assert "old-launch-token" not in config.read_text()


@pytest.mark.parametrize("stale", [True, False], ids=["stale-lock", "live-lock"])
def test_remove_leftover_credentials_clears_a_stale_config_lock(
    git_home: Path, caplog: pytest.LogCaptureFixture, stale: bool
) -> None:
    # A writer killed mid-write leaves ~/.gitconfig.lock holding the config it
    # was writing, and git refuses every later write while it exists. A lock
    # older than a writer can hold one is removed; a fresh one is left to its
    # writer and the leftover is reported, not silently kept.
    home_config = git_home / ".gitconfig"
    _git_file(home_config, "--add", _HELPER_KEY, _LEGACY_HELPER)
    lock = home_config.with_name(".gitconfig.lock")
    lock.write_text(home_config.read_text())
    if stale:
        os.utime(lock, (0, 0))

    h._remove_leftover_credentials()

    assert lock.exists() is not stale
    assert ("old-launch-token" in home_config.read_text()) is not stale
    assert ("Could not remove the old launch-token git helper" in caplog.text) is not stale


def test_remove_leftover_credentials_honours_git_config_global(
    monkeypatch: pytest.MonkeyPatch, git_home: Path, tmp_path: Path
) -> None:
    override = tmp_path / "override.gitconfig"
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(override))
    _git_file(override, "--add", _HELPER_KEY, _LEGACY_HELPER)
    untouched = git_home / ".gitconfig"
    _git_file(untouched, "--add", _HELPER_KEY, _LEGACY_HELPER)

    h._remove_leftover_credentials()

    assert _git_file(override, "--get-all", _HELPER_KEY) == []
    # git does not read ~/.gitconfig while GIT_CONFIG_GLOBAL is set.
    assert _git_file(untouched, "--get-all", _HELPER_KEY) == [_LEGACY_HELPER]


_OLD_TOKEN = "ghu_old-owner-token"
# What the old writer left, before and after gh's multi-account migration.
_LEGACY_FLAT = {"oauth_token": _OLD_TOKEN, "user": "octo", "git_protocol": "https"}
_LEGACY_MIGRATED = {**_LEGACY_FLAT, "users": {"octo": {"oauth_token": _OLD_TOKEN}}}
# gh's own logins: a flat (pre-migration) login, and accounts the old writer
# merged its token into.
_OWN_FLAT = {"oauth_token": "gho_own", "user": "alice", "git_protocol": "ssh"}
_OWN_ACCOUNTS = {"users": {"alice": {"oauth_token": "gho_own"}, "bob": {}}, "user": "alice"}
_ENTERPRISE = {"oauth_token": "enterprise-tok", "user": "alice"}


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ({"github.com": _LEGACY_FLAT}, None),
        ({"github.com": _LEGACY_MIGRATED}, None),
        (
            {"github.com": _LEGACY_MIGRATED, "github.mycompany.com": _ENTERPRISE},
            {"github.mycompany.com": _ENTERPRISE},
        ),
        (
            {"github.com": {**_OWN_ACCOUNTS, "oauth_token": _OLD_TOKEN, "git_protocol": "https"}},
            {"github.com": {**_OWN_ACCOUNTS, "git_protocol": "https"}},
        ),
        ({"github.com": _OWN_FLAT}, {"github.com": _OWN_FLAT}),
        ({"github.com": _OWN_ACCOUNTS}, {"github.com": _OWN_ACCOUNTS}),
        ({"github.mycompany.com": _ENTERPRISE}, {"github.mycompany.com": _ENTERPRISE}),
    ],
    ids=[
        "legacy-flat",
        "legacy-migrated-by-gh",
        "legacy-and-enterprise",
        "legacy-merged-into-own-accounts",
        "own-flat-login",
        "own-accounts",
        "no-github-entry",
    ],
)
def test_remove_legacy_gh_token(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    before: dict[str, object],
    after: dict[str, object] | None,
) -> None:
    gh_dir = tmp_path / "gh"
    gh_dir.mkdir()
    hosts_path = gh_dir / "hosts.yml"
    hosts_path.write_text(yaml.safe_dump(before))
    # An interrupted earlier write left a temporary sibling holding a token.
    (gh_dir / ".hosts.abc123.yml").write_text(yaml.safe_dump({"github.com": _LEGACY_FLAT}))
    monkeypatch.setenv("GH_CONFIG_DIR", str(gh_dir))

    h._remove_legacy_gh_token()

    assert not (gh_dir / ".hosts.abc123.yml").exists()
    if after is None:
        assert not hosts_path.exists()
        return
    assert yaml.safe_load(hosts_path.read_text()) == after
    assert _OLD_TOKEN not in hosts_path.read_text()
    if after != before:
        assert hosts_path.stat().st_mode & 0o777 == 0o600


def test_github_resolver_vends_only_github_over_https(monkeypatch: pytest.MonkeyPatch) -> None:
    fetched: list[tuple[str, str, str]] = []

    def fake_fetch(server: str, host_id: str, token: str) -> dict:
        fetched.append((server, host_id, token))
        return {"connected": True, "username": "x-access-token", "token": "owner-token"}

    monkeypatch.setattr(h, "_fetch", fake_fetch)
    resolve = h._github_resolver("http://srv", "host1")
    monkeypatch.setenv(HOST_TOKEN_ENV_VAR, "launch-tok")
    for request in (
        {"protocol": "https", "host": "gitlab.com"},
        {"protocol": "http", "host": "github.com"},
        {"protocol": "https", "host": "github.com:8443"},
        {"protocol": "https"},
    ):
        assert resolve(request) is None
    assert fetched == []

    # Each request reads the launch token the host holds now.
    monkeypatch.setenv(HOST_TOKEN_ENV_VAR, "rotated-tok")
    assert resolve({"protocol": "https", "host": "github.com"}) == {
        "username": "x-access-token",
        "password": "owner-token",
    }
    assert fetched == [("http://srv", "host1", "rotated-tok")]

    monkeypatch.delenv(HOST_TOKEN_ENV_VAR)
    assert resolve({"protocol": "https", "host": "github.com"}) is None


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ({"username": "u", "password": "owner-token"}, "username=u\npassword=owner-token\n"),
        ({"password": "owner-token"}, "username=x-access-token\npassword=owner-token\n"),
        ({}, ""),
    ],
)
def test_socket_helper_relays_the_broker_reply(
    monkeypatch: pytest.MonkeyPatch, reply: dict[str, str], expected: str
) -> None:
    asked: list[tuple[str, dict[str, str]]] = []

    def fake_request(socket_path: str, request: dict[str, str]) -> dict[str, str]:
        asked.append((socket_path, request))
        return reply

    monkeypatch.setattr(h.credential_broker, "request_credential", fake_request)
    stdin = "protocol=https\nhost=github.com\npath=org/repo.git\nwwwauth[]=Basic\n\n"
    monkeypatch.setattr(sys, "stdin", io.StringIO(stdin))
    out = io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)

    assert h.main(["--socket", "/run/broker.sock", "get"]) == 0

    assert out.getvalue() == expected
    # Only what the broker needs to decide is forwarded.
    assert asked == [("/run/broker.sock", {"protocol": "https", "host": "github.com"})]


def test_socket_helper_declines_other_hosts_without_asking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        h.credential_broker, "request_credential", lambda *a: pytest.fail("asked the broker")
    )
    monkeypatch.setattr(sys, "stdin", io.StringIO("protocol=https\nhost=gitlab.com\n\n"))
    out = io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    assert h.main(["--socket", "/run/broker.sock", "get"]) == 0
    assert out.getvalue() == ""


def test_credential_url_targets_the_generic_provider_path() -> None:
    # The helper hits the provider-generic broker with provider=github.
    assert h._credential_url("http://s", "hid") == "http://s/v1/hosts/hid/credentials/github"
    assert h._credential_url("http://s/", "hid") == "http://s/v1/hosts/hid/credentials/github"


def test_fetch_sends_launch_token_as_header_not_url(monkeypatch: pytest.MonkeyPatch) -> None:
    # Exercise the real request construction (the other tests stub _fetch): the
    # launch token must ride the header so it can't land in server access logs.
    seen: dict = {}

    class _Resp:
        status_code = 200

        def json(self) -> dict:
            return {"connected": True, "token": "T", "username": "x-access-token"}

    def fake_get(url: str, headers: dict, timeout: float) -> _Resp:
        seen["url"], seen["headers"] = url, headers
        return _Resp()

    monkeypatch.setattr(h.httpx, "get", fake_get)
    data = h._fetch("http://s", "hid", "launch-tok")
    assert data is not None and data["token"] == "T"
    assert seen["url"] == "http://s/v1/hosts/hid/credentials/github"
    assert seen["headers"][h.MANAGED_HOST_TOKEN_HEADER] == "launch-tok"
    assert "launch-tok" not in seen["url"]


def test_configure_clone_credentials_wires_broker_when_connected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(HOST_TOKEN_ENV_VAR, "launch-tok")
    monkeypatch.setenv("GIT_TOKEN", "shared-test-token")
    calls: list[list[str]] = []
    monkeypatch.setattr(h.subprocess, "run", lambda args, **k: calls.append(args) or None)
    monkeypatch.setattr(h, "_fetch", lambda *a, **k: {"connected": True, "owner": "a@b.com"})
    assert h.configure_clone_credentials("http://srv", "host1") is True
    key = "credential.https://github.com.helper"
    reset_idx = next(
        i
        for i, c in enumerate(calls)
        if c[:5] == ["git", "config", "--global", "--replace-all", key] and c[-1] == ""
    )
    add_idx = next(
        i
        for i, c in enumerate(calls)
        if c[:5] == ["git", "config", "--global", "--add", key] and "host1" in c[-1]
    )
    assert reset_idx < add_idx


def test_configure_clone_credentials_noop_when_not_connected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Not connected → leave the ambient (image GIT_TOKEN) helper intact; no git config.
    monkeypatch.setenv(HOST_TOKEN_ENV_VAR, "launch-tok")
    calls: list[list[str]] = []
    monkeypatch.setattr(h.subprocess, "run", lambda args, **k: calls.append(args) or None)
    monkeypatch.setattr(h, "_fetch", lambda *a, **k: {"connected": False})
    assert h.configure_clone_credentials("http://srv", "host1") is False
    assert calls == []


def test_configure_clone_credentials_noop_without_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(HOST_TOKEN_ENV_VAR, raising=False)
    calls: list[object] = []
    monkeypatch.setattr(h.subprocess, "run", lambda *a, **k: calls.append(a))
    assert h.configure_clone_credentials("http://srv", "host1") is False
    assert calls == []


def test_configure_clone_credentials_fails_closed_when_probe_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A transient probe failure (``_fetch`` -> None) is indistinguishable from
    # "not linked", so fail closed: install the broker anyway rather than let the
    # clone silently fall back to the shared $GIT_TOKEN identity for what may be a
    # linked owner. Only a *successful* ``connected: false`` keeps the fallback.
    monkeypatch.setenv(HOST_TOKEN_ENV_VAR, "launch-tok")
    calls: list[list[str]] = []
    monkeypatch.setattr(h.subprocess, "run", lambda args, **k: calls.append(args) or None)
    monkeypatch.setattr(h, "_fetch", lambda *a, **k: None)
    assert h.configure_clone_credentials("http://srv", "host1") is True
    key = "credential.https://github.com.helper"
    assert any(
        c[:5] == ["git", "config", "--global", "--add", key] and "host1" in c[-1] for c in calls
    )


@pytest.mark.parametrize(
    ("status", "body", "expected"),
    [
        (404, '{"detail":"unknown credential provider"}', {"connected": False}),
        (404, '{"detail":"Not Found"}', None),
        (404, "<html>Not Found</html>", None),
        (404, "", None),
        (404, "[]", None),
        (404, '"unknown credential provider"', None),
        (401, '{"detail":"unknown credential provider"}', None),
        (503, '{"detail":"unknown credential provider"}', None),
        (200, '{"connected":true,"token":"T"}', {"connected": True, "token": "T"}),
        (200, '{"connected":false}', {"connected": False}),
        (200, "[]", None),
        (200, "invalid JSON", None),
    ],
)
def test_fetch_classifies_provider_response(
    monkeypatch: pytest.MonkeyPatch, status: int, body: str, expected: dict | None
) -> None:
    response = httpx.Response(status, content=body)
    monkeypatch.setattr(h.httpx, "get", lambda *a, **k: response)
    assert h._fetch("http://s", "hid", "tok") == expected


def test_configure_clone_credentials_keeps_shared_helper_on_provider_404(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(HOST_TOKEN_ENV_VAR, "launch-tok")
    calls: list[list[str]] = []
    monkeypatch.setattr(h.subprocess, "run", lambda args, **k: calls.append(args) or None)
    response = httpx.Response(404, json={"detail": "unknown credential provider"})
    monkeypatch.setattr(h.httpx, "get", lambda *a, **k: response)
    assert h.configure_clone_credentials("http://srv", "host1") is False
    assert calls == []  # ambient/shared helper chain untouched
