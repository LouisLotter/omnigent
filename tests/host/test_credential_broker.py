"""Tests for the host credential broker's socket, wire format and gh wrapper."""

from __future__ import annotations

import io
import os
import socketserver
import stat
import sys
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path

import pytest

from omnigent.host import credential_broker as broker

pytestmark = pytest.mark.posix_only


@pytest.fixture
def socket_root(monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A not-yet-created harness socket root under a short ``/tmp`` dir (``sun_path`` limit)."""
    with tempfile.TemporaryDirectory(prefix="og-cred-", dir="/tmp") as parent:
        root = Path(parent) / "run"
        monkeypatch.setenv("OMNIGENT_HARNESS_TMP_PARENT", str(root))
        yield root


class _Resolver:
    """Records requests and answers with *reply*, or raises it when it is an exception."""

    def __init__(self, reply: Mapping[str, str] | Exception | None) -> None:
        self.reply = reply
        self.requests: list[dict[str, str]] = []

    def __call__(self, request: dict[str, str]) -> Mapping[str, str] | None:
        self.requests.append(request)
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def _mode(path: Path) -> int:
    return stat.S_IMODE(os.lstat(path).st_mode)


@contextmanager
def _serving(path: Path, resolver: _Resolver) -> Iterator[socketserver.BaseServer]:
    server = broker.start_credential_broker(path, resolver)
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()


def test_broker_serves_requests_on_an_owner_only_socket(socket_root: Path) -> None:
    resolver = _Resolver({"username": "x-access-token", "password": "owner-token"})
    path = broker.broker_socket_path("host-1")
    with _serving(path, resolver):
        assert path.is_absolute() and path.parent == socket_root
        assert stat.S_ISSOCK(os.lstat(path).st_mode)
        assert _mode(socket_root) == 0o700
        assert _mode(path) == 0o600
        reply = broker.request_credential(str(path), {"protocol": "https", "host": "github.com"})
        assert reply == {"username": "x-access-token", "password": "owner-token"}
        assert resolver.requests == [{"protocol": "https", "host": "github.com"}]


def test_broker_tightens_an_existing_group_readable_root(socket_root: Path) -> None:
    socket_root.mkdir(mode=0o755)
    socket_root.chmod(0o755)
    with _serving(broker.broker_socket_path("h"), _Resolver(None)):
        assert _mode(socket_root) == 0o700


def test_broker_refuses_a_symlinked_root(socket_root: Path) -> None:
    real = socket_root.parent / "real"
    real.mkdir(mode=0o700)
    socket_root.symlink_to(real)
    with pytest.raises(RuntimeError, match="is a symlink"):
        broker.start_credential_broker(broker.broker_socket_path("h"), _Resolver(None))
    assert list(real.iterdir()) == []


@pytest.mark.parametrize("reply", [None, {}, RuntimeError("server said owner-token")])
def test_broker_declines_and_keeps_serving(
    socket_root: Path,
    reply: Mapping[str, str] | Exception | None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    resolver = _Resolver(reply)
    path = broker.broker_socket_path("h")
    with _serving(path, resolver):
        assert broker.request_credential(str(path), {"host": "github.com"}) == {}
        resolver.reply = {"password": "later"}
        assert broker.request_credential(str(path), {"host": "github.com"}) == {
            "password": "later"
        }
    # A failing resolver is logged by type only; its message could carry a credential.
    assert "owner-token" not in caplog.text


def test_restarted_host_recreates_the_socket(socket_root: Path) -> None:
    path = broker.broker_socket_path("host-1")
    first = broker.start_credential_broker(path, _Resolver({"password": "first"}))
    # A crashed or closed host leaves its socket file behind with nothing listening.
    first.shutdown()
    first.server_close()
    assert stat.S_ISSOCK(os.lstat(path).st_mode)
    assert broker.request_credential(str(path), {"host": "github.com"}) == {}

    with _serving(path, _Resolver({"password": "second"})):
        assert broker.request_credential(str(path), {"host": "github.com"}) == {
            "password": "second"
        }
        assert _mode(path) == 0o600
    assert [entry.name for entry in socket_root.iterdir()] == [path.name]


def test_overlapping_hosts_never_remove_each_others_socket(
    socket_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Host B starts and publishes while host A is between binding and
    # publishing. Whichever publishes last serves the path, and closing the
    # other leaves it serving.
    path = broker.broker_socket_path("host-1")
    real_replace = os.replace
    started: list[socketserver.BaseServer] = []

    def replace_after_a_competitor(source: str, target: str) -> None:
        monkeypatch.setattr(broker.os, "replace", real_replace)
        started.append(broker.start_credential_broker(path, _Resolver({"password": "b"})))
        real_replace(source, target)

    monkeypatch.setattr(broker.os, "replace", replace_after_a_competitor)
    host_a = broker.start_credential_broker(path, _Resolver({"password": "a"}))
    (host_b,) = started
    try:
        assert broker.request_credential(str(path), {"host": "github.com"}) == {"password": "a"}
        host_b.shutdown()
        host_b.server_close()
        assert broker.request_credential(str(path), {"host": "github.com"}) == {"password": "a"}
    finally:
        host_a.shutdown()
        host_a.server_close()
    # Only the published socket remains; no staging name is left behind.
    assert [entry.name for entry in socket_root.iterdir()] == [path.name]


def test_publish_failure_leaves_no_staging_socket(
    socket_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(source: str, target: str) -> None:
        raise PermissionError("read-only")

    monkeypatch.setattr(broker.os, "replace", refuse)
    with pytest.raises(PermissionError):
        broker.start_credential_broker(broker.broker_socket_path("h"), _Resolver(None))
    assert list(socket_root.iterdir()) == []


def test_socket_path_is_per_host_and_stable(socket_root: Path) -> None:
    assert broker.broker_socket_path("a") == broker.broker_socket_path("a")
    assert broker.broker_socket_path("a") != broker.broker_socket_path("b")
    assert len(str(broker.broker_socket_path("f" * 32))) < 104  # sun_path on macOS


def test_unreachable_broker_declines(socket_root: Path) -> None:
    assert broker.request_credential(str(socket_root / "missing.sock"), {"host": "x"}) == {}


def test_wire_format_is_bounded_and_refuses_line_injection() -> None:
    assert broker.format_fields({"username": "u", "password": "p"}) == b"username=u\npassword=p\n"
    assert broker.format_fields({"password": "p\nhost=evil.example"}) == b""
    assert broker.format_fields({"password": "p\0"}) == b""

    stream = io.BytesIO(b"protocol=https\r\nhost=github.com\nignored\n\nafter=blank\n")
    assert broker.read_fields(stream) == {"protocol": "https", "host": "github.com"}
    many = b"".join(f"k{index}=v\n".encode() for index in range(100))
    assert len(broker.read_fields(io.BytesIO(many))) == 32


# ── gh wrapper ──────────────────────────────────────────────────────────────


@pytest.fixture
def gh_layout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """A wrapper ``gh`` first on PATH and the real ``gh`` after it."""
    wrapper_dir = tmp_path / "wrapper"
    real_dir = tmp_path / "real"
    for directory in (wrapper_dir, real_dir):
        directory.mkdir()
        gh = directory / "gh"
        gh.write_text("#!/bin/sh\n")
        gh.chmod(0o755)
    monkeypatch.setenv(
        "PATH", os.pathsep.join(["relative/bin", str(wrapper_dir), str(real_dir), "/usr/bin"])
    )
    monkeypatch.setattr(sys, "argv", [str(wrapper_dir / "gh"), "pr", "view"])
    for name in ("GH_TOKEN", "GITHUB_TOKEN", broker.SOCKET_ENV_VAR):
        monkeypatch.delenv(name, raising=False)
    return wrapper_dir / "gh", real_dir / "gh"


class _Exec(Exception):
    """Raised by the fake ``execve`` so the wrapper stops where it would exec."""


def _run_wrapper(monkeypatch: pytest.MonkeyPatch) -> tuple[str, list[str], dict[str, str]]:
    calls: list[tuple[str, list[str], dict[str, str]]] = []

    def fake_execve(path: str, argv: list[str], env: dict[str, str]) -> None:
        calls.append((path, argv, env))
        raise _Exec

    monkeypatch.setattr(broker.os, "execve", fake_execve)
    with pytest.raises(_Exec):
        broker.gh_main()
    (call,) = calls
    return call


def test_gh_wrapper_passes_the_broker_token_to_the_next_gh(
    socket_root: Path, gh_layout: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    _, real_gh = gh_layout
    resolver = _Resolver({"username": "x-access-token", "password": "owner-token"})
    path = broker.broker_socket_path("h")
    monkeypatch.setenv(broker.SOCKET_ENV_VAR, str(path))
    with _serving(path, resolver):
        executable, argv, env = _run_wrapper(monkeypatch)
    assert executable == str(real_gh)
    assert argv == [str(real_gh), "pr", "view"]
    assert env["GH_TOKEN"] == "owner-token"
    assert resolver.requests == [{"protocol": "https", "host": "github.com"}]


@pytest.mark.parametrize(
    ("ambient", "broker_reply"),
    [
        ({}, None),  # no broker configured: gh runs unchanged
        ({}, {}),  # the broker declines
        ({"GH_TOKEN": "chosen"}, {"password": "owner-token"}),
        ({"GITHUB_TOKEN": "chosen"}, {"password": "owner-token"}),
    ],
)
def test_gh_wrapper_leaves_gh_alone_without_a_broker_token(
    socket_root: Path,
    gh_layout: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    ambient: dict[str, str],
    broker_reply: dict[str, str] | None,
) -> None:
    for name, value in ambient.items():
        monkeypatch.setenv(name, value)
    if broker_reply is None:
        _, _, env = _run_wrapper(monkeypatch)
    else:
        path = broker.broker_socket_path("h")
        monkeypatch.setenv(broker.SOCKET_ENV_VAR, str(path))
        with _serving(path, _Resolver(broker_reply)):
            _, _, env = _run_wrapper(monkeypatch)
    assert env.get("GH_TOKEN") == ambient.get("GH_TOKEN")
    assert env.get("GITHUB_TOKEN") == ambient.get("GITHUB_TOKEN")


def test_gh_wrapper_reports_a_missing_gh(
    gh_layout: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    _, real_gh = gh_layout
    real_gh.unlink()
    monkeypatch.setattr(broker.os, "execve", lambda *args: pytest.fail("nothing to exec"))
    with pytest.raises(SystemExit) as excinfo:
        broker.gh_main()
    assert excinfo.value.code == 127
    assert "not installed" in capsys.readouterr().err


def test_next_on_path_starts_after_the_wrapper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A second wrapper copy later on PATH hands off to the real gh after it,
    # rather than back to the first wrapper.
    dirs = [tmp_path / name for name in ("first", "second", "real")]
    for directory in dirs:
        directory.mkdir()
        (directory / "gh").write_text("#!/bin/sh\n")
        (directory / "gh").chmod(0o755)
    monkeypatch.setenv("PATH", os.pathsep.join(str(directory) for directory in dirs))
    assert broker._next_on_path("gh", str(dirs[0] / "gh")) == str(dirs[1] / "gh")
    assert broker._next_on_path("gh", str(dirs[1] / "gh")) == str(dirs[2] / "gh")
    # Called by absolute path from outside PATH: the first gh on PATH.
    assert broker._next_on_path("gh", str(tmp_path / "elsewhere" / "gh")) == str(dirs[0] / "gh")
