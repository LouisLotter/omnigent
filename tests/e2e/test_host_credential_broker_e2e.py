"""git and gh on a managed host get the owner's GitHub token from the host's broker.

The host's GitHub setup runs in this process against a loopback credentials
endpoint. Real git then clones and pushes, and gh runs through the image's
wrapper, as an agent would: without the launch token in its environment. The
github.com stand-in is :class:`FakeGitHub`. No LLM or Omnigent server is used.
"""

from __future__ import annotations

import json
import os
import shlex
import stat
import subprocess
import sys
import tempfile
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import yaml

from omnigent.git_credential_github import configure_host_git
from omnigent.host.connect import _build_runner_env
from omnigent.host.credential_broker import SOCKET_ENV_VAR
from omnigent.host.identity import HOST_TOKEN_ENV_VAR, MANAGED_HOST_TOKEN_HEADER
from omnigent.inner.agent_env import clean_agent_env
from tests.e2e._fake_github_https import FakeGitHub, make_bare_repo

pytestmark = pytest.mark.posix_only

_HOST_ID = "0123456789abcdef0123456789abcdef"
_LAUNCH_TOKEN = "launch-token-canary-3f9a"
_OWNER_TOKEN = "ghu_owner-token-canary-7c21"
_OLD_LAUNCH_TOKEN = "old-launch-token-canary-51d0"
_OLD_GITHUB_TOKEN = "ghu_old-github-token-canary-a4e2"
_SHARED_TOKEN = "shared-image-token-canary-9b77"
_CANARIES = (_LAUNCH_TOKEN, _OWNER_TOKEN, _OLD_LAUNCH_TOKEN, _OLD_GITHUB_TOKEN)
_ORG_REPO = "acme/widget"
_CLONE_URL = f"https://github.com/{_ORG_REPO}.git"
_HELPER_KEY = "credential.https://github.com.helper"
_ENTERPRISE = {"github.mycompany.com": {"oauth_token": "enterprise-kept", "user": "alice"}}
# Match the system-scope helper in deploy/docker/Dockerfile.
_IMAGE_CREDENTIAL_HELPER = (
    '!f() { [ "$1" = get ] || return 0; [ -n "$GIT_TOKEN" ] || return 0; '
    'printf "username=%s\\npassword=%s\\n" "${GIT_USERNAME:-x-access-token}" "$GIT_TOKEN"; }; f'
)
_TIMEOUT_S = 60


class _CredentialsEndpoint:
    """The server's host-credentials route: GitHub linked, failing, or not configured."""

    def __init__(self, scenario: str) -> None:
        self.scenario = scenario
        self.requests: list[bool] = []
        endpoint = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: object) -> None:
                pass

            def do_GET(self) -> None:
                endpoint.requests.append(
                    self.path == f"/v1/hosts/{_HOST_ID}/credentials/github"
                    and self.headers.get(MANAGED_HOST_TOKEN_HEADER) == _LAUNCH_TOKEN
                )
                status, body = endpoint.answer()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(body).encode())

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_port}"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def answer(self) -> tuple[int, dict[str, object]]:
        if self.scenario == "not_configured":
            return 404, {"detail": "unknown credential provider"}
        if self.scenario == "probe_fails" and len(self.requests) == 1:
            return 503, {"detail": "unavailable"}
        return 200, {
            "connected": True,
            "owner": "owner@example.test",
            "login": "octo",
            "username": "x-access-token",
            "token": _OWNER_TOKEN,
        }

    def __enter__(self) -> _CredentialsEndpoint:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture
def socket_root() -> Iterator[Path]:
    """A short ``/tmp`` harness socket root (``sun_path`` limit)."""
    with tempfile.TemporaryDirectory(prefix="og-e2e-", dir="/tmp") as parent:
        yield Path(parent) / "run"


def _git(*args: str, env: dict[str, str], cwd: Path | None = None) -> str:
    result = subprocess.run(
        ["git", *args], env=env, cwd=cwd, capture_output=True, text=True, timeout=_TIMEOUT_S
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def _seed_leftovers(home: Path) -> None:
    """What earlier hosts wrote: the launch-token helper and gh's token.

    The helper is in both global git files (the XDG one is used while
    ``~/.gitconfig`` is absent) and in the lock a writer killed mid-write left.
    """
    legacy_helper = (
        "!python3 -m omnigent.git_credential_github --server http://old "
        f"--host-id {_HOST_ID} --host-token {_OLD_LAUNCH_TOKEN}"
    )
    gitconfig = home / ".gitconfig"
    for value in ("", legacy_helper):
        _git("config", "--file", str(gitconfig), "--add", _HELPER_KEY, value, env={})
    xdg_gitconfig = home / ".config" / "git" / "config"
    xdg_gitconfig.parent.mkdir(parents=True)
    _git("config", "--file", str(xdg_gitconfig), "--add", _HELPER_KEY, legacy_helper, env={})
    lock = home / ".gitconfig.lock"
    lock.write_text(gitconfig.read_text())
    os.utime(lock, (0, 0))
    gh_dir = home / ".config" / "gh"
    gh_dir.mkdir(parents=True)
    # As gh's multi-account migration leaves the old writer's entry.
    legacy = {
        "oauth_token": _OLD_GITHUB_TOKEN,
        "user": "octo",
        "git_protocol": "https",
        "users": {"octo": {"oauth_token": _OLD_GITHUB_TOKEN}},
    }
    (gh_dir / "hosts.yml").write_text(yaml.safe_dump({"github.com": legacy, **_ENTERPRISE}))


def _install_gh(bin_root: Path) -> tuple[Path, Path]:
    """The image's gh wrapper (as in deploy/docker/Dockerfile) ahead of a recording gh."""
    wrapper_bin, real_bin = bin_root / "venv-bin", bin_root / "usr-local-bin"
    for directory in (wrapper_bin, real_bin):
        directory.mkdir(parents=True)
    (wrapper_bin / "gh").write_text(
        f"#!{sys.executable} -I\nfrom omnigent.host.credential_broker import gh_main\ngh_main()\n"
    )
    (real_bin / "gh").write_text(
        f"#!{sys.executable}\nimport json, os, sys\n"
        'print(json.dumps({"argv": sys.argv[1:], "gh_token": os.environ.get("GH_TOKEN")}))\n'
    )
    for gh in (wrapper_bin / "gh", real_bin / "gh"):
        gh.chmod(0o755)
    return wrapper_bin, real_bin


def _files_holding_canaries(*roots: Path) -> list[str]:
    hits = []
    for root in roots:
        for path in [root, *root.rglob("*")] if root.is_dir() else [root]:
            if path.is_symlink() or not path.is_file():
                continue
            content = path.read_bytes()
            hits += [f"{path}: {token}" for token in _CANARIES if token.encode() in content]
    return hits


@pytest.mark.parametrize("scenario", ["linked", "probe_fails", "not_configured"])
def test_git_and_gh_use_the_broker_and_leave_no_token_on_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, socket_root: Path, scenario: str
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    _seed_leftovers(home)
    system_config = tmp_path / "image-system.gitconfig"
    _git(
        "config",
        "--file",
        str(system_config),
        "credential.helper",
        _IMAGE_CREDENTIAL_HELPER,
        env={},
    )
    wrapper_bin, real_bin = _install_gh(tmp_path / "bin")
    trace = tmp_path / "git-trace.log"
    github_root = tmp_path / "github"
    bare = make_bare_repo(github_root, _ORG_REPO)
    brokered = scenario != "not_configured"

    for name in ("GIT_CONFIG_GLOBAL", "GIT_CONFIG_NOSYSTEM", "XDG_CONFIG_HOME", "GH_CONFIG_DIR"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("IS_SANDBOX", "1")
    monkeypatch.setenv(HOST_TOKEN_ENV_VAR, _LAUNCH_TOKEN)
    monkeypatch.setenv("OMNIGENT_HARNESS_TMP_PARENT", str(socket_root))
    monkeypatch.setenv(SOCKET_ENV_VAR, "")

    expected_token = _OWNER_TOKEN if brokered else _SHARED_TOKEN
    with (
        _CredentialsEndpoint(scenario) as endpoint,
        FakeGitHub(github_root, "x-access-token", expected_token) as github,
    ):
        broker = configure_host_git(endpoint.url, _HOST_ID)
        try:
            assert (broker is not None) == brokered
            # Commits are attributed to the owner only once the probe confirms them.
            owner_email = subprocess.run(
                ["git", "config", "--global", "user.email"],
                env={"HOME": str(home)},
                capture_output=True,
                text=True,
                timeout=_TIMEOUT_S,
            ).stdout.strip()
            assert owner_email == ("owner@example.test" if scenario == "linked" else "")
            # The agent's environment through the real filters: host to runner,
            # then runner to agent CLI. The launch token must not survive them.
            runner_env = _build_runner_env(
                os.environ,
                server_url=endpoint.url,
                runner_id="runner_e2e",
                binding_token="binding-e2e",
                workspace=str(home),
                parent_pid=os.getpid(),
            )
            agent_env = clean_agent_env(source=runner_env)
            assert HOST_TOKEN_ENV_VAR not in agent_env
            assert agent_env.get(SOCKET_ENV_VAR) == os.environ.get(SOCKET_ENV_VAR)
            # Test plumbing: the image's PATH order and system helper, its
            # shared token, the github.com stand-in, and git's command trace.
            agent_env.update(
                {
                    "PATH": os.pathsep.join([str(wrapper_bin), str(real_bin), agent_env["PATH"]]),
                    "GIT_CONFIG_SYSTEM": str(system_config),
                    "GIT_TERMINAL_PROMPT": "0",
                    "GIT_TOKEN": _SHARED_TOKEN,
                    "GIT_TRACE": str(trace),
                    "GIT_SSL_NO_VERIFY": "1",  # the stand-in's self-signed certificate
                    "https_proxy": github.proxy_url,
                    "no_proxy": "127.0.0.1,localhost",
                    **{
                        f"GIT_{role}_{field}": value
                        for role in ("AUTHOR", "COMMITTER")
                        for field, value in (("NAME", "agent"), ("EMAIL", "agent@example.test"))
                    },
                }
            )
            work = home / "work"
            _git("clone", "--", _CLONE_URL, str(work), env=agent_env)
            (work / "change.txt").write_text("agent change\n")
            _git("add", "change.txt", env=agent_env, cwd=work)
            _git("commit", "-m", "agent change", env=agent_env, cwd=work)
            _git("push", "origin", "HEAD:main", env=agent_env, cwd=work)
            assert _git("--git-dir", str(bare), "log", "-1", "--format=%s", env={}) == (
                "agent change\n"
            )

            gh = subprocess.run(
                ["gh", "api", "user"],
                env=agent_env,
                capture_output=True,
                text=True,
                timeout=_TIMEOUT_S,
            )
            assert gh.returncode == 0, gh.stderr
            assert json.loads(gh.stdout) == {
                "argv": ["api", "user"],
                "gh_token": _OWNER_TOKEN if brokered else None,
            }

            if brokered:
                assert broker is not None
                socket_path = Path(os.environ[SOCKET_ENV_VAR])
                assert stat.S_IMODE(socket_root.stat().st_mode) == 0o700
                assert stat.S_IMODE(os.lstat(socket_path).st_mode) == 0o600
                # The host crashes, leaving its socket file; the next start replaces it.
                broker.shutdown()
                broker.socket.close()
                broker = configure_host_git(endpoint.url, _HOST_ID)
                assert broker is not None
                _git("fetch", "origin", env=agent_env, cwd=work)
        finally:
            if broker is not None:
                broker.shutdown()
                broker.server_close()

    assert endpoint.requests and all(endpoint.requests)
    assert github.saw_authenticated_request
    # ~/.gitconfig holds the socket helper and the owner's identity, nothing else.
    config = _git("config", "--global", "--list", env={"HOME": str(home)}).splitlines()
    keys = {line.partition("=")[0] for line in config}
    if brokered:
        helper = (
            f"!{shlex.quote(sys.executable)} -I -m omnigent.git_credential_github "
            f"--socket {shlex.quote(os.environ[SOCKET_ENV_VAR])}"
        )
        helpers = _git("config", "--global", "--get-all", _HELPER_KEY, env={"HOME": str(home)})
        assert helpers.splitlines() == ["", helper]
        # After the restart's probe succeeds, probe_fails has the identity too.
        assert keys == {_HELPER_KEY, "user.email", "user.name"}
    else:
        assert keys == set()
    # gh's hosts.yml keeps only what earlier hosts did not write.
    assert yaml.safe_load((home / ".config" / "gh" / "hosts.yml").read_text()) == _ENTERPRISE
    # No launch or GitHub token on disk, and none in any command git ran.
    assert _files_holding_canaries(home, socket_root.parent, trace) == []
    # git ran the socket helper whenever the broker was in place.
    assert ("omnigent.git_credential_github --socket" in trace.read_text()) == brokered
