"""Git credential helper that gets the owner's GitHub token through the host.

Installed in a managed sandbox as git's ``credential.helper`` for
``github.com``. On each HTTPS auth challenge git runs this with ``get`` and the
request on stdin. The helper asks the host's credential broker
(:mod:`omnigent.host.credential_broker`) over a local Unix socket; the host
calls the server's host-facing credential endpoint
(:mod:`omnigent.server.routes.host_credentials`, with ``provider=github``)
with the launch token it holds in memory, and the helper prints back
``username`` / ``password``. The gh CLI asks the same broker through its
wrapper, :func:`omnigent.host.credential_broker.gh_main`.

Why this shape:
- **The host writes neither the launch token nor the GitHub token to disk or
  into an argv.** ``~/.gitconfig`` names only the socket, by absolute path, and
  the GitHub token lives only in the helper's stdout or gh's environment.
- It is **executor-agnostic**: every executor already starts the host with a
  server URL + launch token, so nothing GitHub-specific is injected per
  executor.

Threat model: this removes the tokens from disk, not the ability of in-sandbox
code to request one. Any same-user process can ask the broker (or read the
launch token from the host's environment) and obtain the owner's full-scope
user token for its lifetime; teardown stops future fetches but does not revoke
an already-fetched token. The trust boundary is the sandbox, not this helper.

Run as: ``git credential.helper`` →
``<python> -I -m omnigent.git_credential_github --socket <path>``. The
workspace-prep init container instead runs :func:`main` in-process with the
launch token from its environment (``--server/--host-id/--host-token``).
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import os
import shlex
import socketserver
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path

import httpx
import yaml

from omnigent.host import credential_broker
from omnigent.host.identity import HOST_TOKEN_ENV_VAR as _HOST_TOKEN_ENV_VAR
from omnigent.host.identity import MANAGED_HOST_TOKEN_HEADER

_logger = logging.getLogger(__name__)

_TIMEOUT_S = 15.0
_HELPER_KEY = "credential.https://github.com.helper"


def _in_sandbox() -> bool:
    """Whether we're running inside a managed sandbox (host image sets ``IS_SANDBOX=1``).

    The host integrations rewire git's github.com credential chain and start the
    credential broker. That's only ever wanted in a managed sandbox — a
    disposable, per-session filesystem. A local ``omnigent host`` shares the
    developer's real ``~/.gitconfig`` / ``~/.config/gh``, so every auto-apply must
    be a no-op there. ``IS_SANDBOX=1`` is baked into the managed host image and
    the k8s pod spec; it is absent on a local host.
    """
    return (os.environ.get("IS_SANDBOX") or "").strip() == "1"


def _read_git_request() -> dict[str, str]:
    """Parse git's ``key=value`` credential request from stdin (blank-line terminated)."""
    fields: dict[str, str] = {}
    for line in sys.stdin:
        line = line.rstrip("\n")
        if not line:
            break
        key, _, value = line.partition("=")
        fields[key] = value
    return fields


def _credential_url(server: str, host_id: str) -> str:
    return f"{server.rstrip('/')}/v1/hosts/{host_id}/credentials/github"


def _fetch(server: str, host_id: str, host_token: str) -> dict | None:
    """Fetch the credential endpoint JSON, or ``None`` if the result is inconclusive.

    Only the server's explicit no-provider 404 maps to ``connected: false``.
    A generic 404 from a proxy or missing route must not enable shared credentials.
    """
    try:
        resp = httpx.get(
            _credential_url(server, host_id),
            headers={MANAGED_HOST_TOKEN_HEADER: host_token},
            timeout=_TIMEOUT_S,
        )
    except httpx.HTTPError:
        return None
    if resp.status_code not in (200, 404):
        return None
    try:
        data = resp.json()
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    if resp.status_code == 404:
        if data.get("detail") != "unknown credential provider":
            return None
        return {"connected": False}
    return data


def _fetch_credential(server: str, host_id: str, host_token: str) -> tuple[str, str] | None:
    """Fetch ``(username, token)`` from the server, or ``None`` if unavailable."""
    data = _fetch(server, host_id, host_token)
    if not data or not data.get("connected") or not data.get("token"):
        return None
    return str(data.get("username") or "x-access-token"), str(data["token"])


def _git_config(*args: str) -> None:
    """Run ``git config --global`` best-effort (never raises; git may be absent)."""
    with contextlib.suppress(OSError, subprocess.SubprocessError):
        subprocess.run(
            ["git", "config", "--global", *args],
            check=False,
            capture_output=True,
            timeout=_TIMEOUT_S,
        )


def _install_broker_helper(server_url: str, host_id: str, token: str) -> None:
    """Make the per-user broker the sole github.com credential helper.

    Used by :func:`configure_clone_credentials`. The workspace-prep init
    container swaps this function, by name, for one that writes a helper reading
    the launch token from its environment into a temporary git config. The host
    uses :func:`_install_socket_helper`, which writes no token.

    Resets the github.com helper chain first: git accumulates credential helpers
    across scopes and runs them in parse order (system → global), stopping at the
    first that returns a full credential. A managed image installs a wider-scope
    helper that answers github.com from a shared ``$GIT_TOKEN``; parsed before
    this --global entry it would vend first and silently bypass the per-user
    broker. An empty value clears the inherited chain for github.com, so only the
    broker remains. ``--replace-all`` keeps re-runs idempotent.
    """
    helper = (
        "!python3 -m omnigent.git_credential_github "
        f"--server {shlex.quote(server_url)} --host-id {shlex.quote(host_id)} "
        f"--host-token {shlex.quote(token)}"
    )
    _git_config("--replace-all", _HELPER_KEY, "")
    _git_config("--add", _HELPER_KEY, helper)


def _install_socket_helper(socket_path: Path) -> None:
    """Make the host's broker socket the sole github.com credential helper.

    Resets the inherited github.com chain as :func:`_install_broker_helper`
    does. The helper names the interpreter and socket by absolute path, since
    agent processes may run another ``python3`` and lack the host's
    environment; ``-I`` keeps a repository's files off the import path.
    """
    helper = (
        f"!{shlex.quote(sys.executable)} -I -m omnigent.git_credential_github "
        f"--socket {shlex.quote(str(socket_path))}"
    )
    _git_config("--replace-all", _HELPER_KEY, "")
    _git_config("--add", _HELPER_KEY, helper)


def _gh_config_dir() -> Path:
    """The gh CLI config dir (``GH_CONFIG_DIR`` override, else ``~/.config/gh``)."""
    override = (os.environ.get("GH_CONFIG_DIR") or "").strip()
    if override:
        return Path(override)
    return Path(os.environ.get("XDG_CONFIG_HOME") or (Path.home() / ".config")) / "gh"


def _remove_legacy_gh_token() -> None:
    """Drop the github.com token earlier hosts wrote into gh's ``hosts.yml``.

    Those hosts wrote ``oauth_token`` / ``user`` / ``git_protocol`` for
    github.com; gh's own login also records the account under ``users``, so an
    entry with a token and no ``users`` map came from them. Their interrupted
    writes left ``.hosts.*.yml`` temporary files, removed too. Other hosts and
    accounts are kept; the rewrite is atomic and 0600.
    """
    config_dir = _gh_config_dir()
    for leftover in config_dir.glob(".hosts.*.yml"):
        with contextlib.suppress(OSError):
            leftover.unlink()
    hosts_path = config_dir / "hosts.yml"
    try:
        hosts = yaml.safe_load(hosts_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return
    if not isinstance(hosts, dict):
        return
    entry = hosts.get("github.com")
    if not isinstance(entry, dict) or "oauth_token" not in entry or "users" in entry:
        return
    del hosts["github.com"]
    if not hosts:
        with contextlib.suppress(OSError):
            hosts_path.unlink()
        return
    tmp: str | None = None
    try:
        fd, tmp = tempfile.mkstemp(dir=config_dir, prefix=".hosts.", suffix=".yml")
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            yaml.safe_dump(hosts, handle, default_flow_style=False, sort_keys=True)
        os.replace(tmp, hosts_path)
    except (OSError, yaml.YAMLError):
        if tmp is not None:
            with contextlib.suppress(OSError):
                os.unlink(tmp)


def _remove_leftover_credentials() -> None:
    """Remove tokens that earlier hosts wrote to disk.

    The launch token baked into a github.com helper in the global git config,
    and the GitHub token in gh's ``hosts.yml``.
    """
    _git_config("--unset-all", _HELPER_KEY, r"omnigent\.git_credential_github .*--host-token")
    _remove_legacy_gh_token()


def _github_resolver(server_url: str, host_id: str) -> credential_broker.Resolver:
    """Build the broker's resolver: github.com over https, per request, else decline."""

    def resolve(request: Mapping[str, str]) -> dict[str, str] | None:
        if request.get("host") != "github.com" or request.get("protocol") != "https":
            return None
        token = (os.environ.get(_HOST_TOKEN_ENV_VAR) or "").strip()
        if not token:
            return None
        credential = _fetch_credential(server_url, host_id, token)
        if credential is None:
            return None
        username, password = credential
        return {"username": username, "password": password}

    return resolve


def configure_host_git(server_url: str, host_id: str) -> socketserver.BaseServer | None:
    """Wire git and gh in a managed sandbox to the host's credential broker.

    Called by ``omnigent host`` at startup — executor-agnostic, since the host
    runs in every executor and holds ``$OMNIGENT_HOST_TOKEN``. First removes
    tokens that earlier hosts wrote to disk. When the owner has GitHub
    connected (the per-user Connect model), starts the broker, exports its
    socket path for gh's wrapper (:data:`credential_broker.SOCKET_ENV_VAR`),
    makes the socket the authoritative github.com ``credential.helper``, and
    sets the commit author to the owner. Best-effort: never raises (git may be
    absent, or GitHub not configured on the server).

    Connected-gated, mirroring :func:`configure_clone_credentials`: a confirmed
    ``connected: false`` means the owner hasn't linked GitHub (a shared-
    ``$GIT_TOKEN`` or local deployment). Taking over github.com would clear a
    shared ``$GIT_TOKEN`` helper and then decline, breaking in-session git — so
    that path instead **clears** any broker helper a prior (inconclusive) probe
    may have installed, restoring the ambient helper, and starts no broker. Only
    a connected owner, or an inconclusive probe (``None``, fail-closed like the
    clone), takes over github.com. If the broker cannot start, the helper still
    names its socket and declines, so git fails closed.

    Sandbox-only (see :func:`_in_sandbox`): a no-op outside a managed sandbox, so
    a local ``omnigent host`` never rewrites the developer's real ``~/.gitconfig``.

    :returns: The running broker, or ``None`` when none was started.
    """
    if not _in_sandbox():
        return None
    os.environ.pop(credential_broker.SOCKET_ENV_VAR, None)
    token = (os.environ.get(_HOST_TOKEN_ENV_VAR) or "").strip()
    if not token:
        return None
    _remove_leftover_credentials()
    data = _fetch(server_url, host_id, token)
    if data is not None and not data.get("connected"):
        # Confirmed not linked: clear any broker helper and the chain reset it
        # wrote, restoring the ambient shared-``$GIT_TOKEN`` helper.
        _git_config("--unset-all", _HELPER_KEY)
        return None
    socket_path = credential_broker.broker_socket_path(host_id)
    broker: socketserver.BaseServer | None = None
    try:
        broker = credential_broker.start_credential_broker(
            socket_path, _github_resolver(server_url, host_id)
        )
    except (OSError, RuntimeError) as exc:
        _logger.warning("GitHub credential broker did not start at %s: %s", socket_path, exc)
    else:
        os.environ[credential_broker.SOCKET_ENV_VAR] = str(socket_path)
    _install_socket_helper(socket_path)
    owner = str((data or {}).get("owner") or "")
    if data and data.get("connected") and "@" in owner:
        _git_config("user.email", owner)
        login = data.get("login")
        _git_config("user.name", str(login) if login else owner.split("@", 1)[0])
    return broker


def configure_clone_credentials(server_url: str, host_id: str) -> bool:
    """Wire the per-user broker for the initial workspace clone.

    Called by the managed-sandbox ``workspace-prep`` init container BEFORE it
    clones the workspace repo. Like :func:`configure_host_git`, this is
    connected-gated: it keeps the ambient credential chain — notably the image's
    shared ``$GIT_TOKEN`` helper — when the owner is *definitively* not linked, so
    a shared-token clone still works for them.

    Fails closed on an ambiguous probe: :func:`_fetch` returns ``None`` on any
    transient fault (timeout, 5xx, bad JSON), which is indistinguishable from
    "not linked". Treating that as unlinked would silently clone a *linked*
    owner's private repo under the shared image identity, defeating the per-user
    contract. So only a **definitive** negative — a ``connected: false``, or the
    endpoint's explicit 404 "unknown credential provider" — keeps the shared fallback; a
    connected owner — or an unresolved probe — installs the broker, so the clone
    authenticates per-user or fails visibly instead of quietly falling back to
    the shared token. Best-effort: never raises.

    :returns: ``True`` when the broker was wired (owner connected, or the probe
        was inconclusive); ``False`` only when the owner is confirmed not linked.
    """
    token = (os.environ.get(_HOST_TOKEN_ENV_VAR) or "").strip()
    if not token:
        return False
    data = _fetch(server_url, host_id, token)
    if data is not None and not data.get("connected"):
        return False
    _install_broker_helper(server_url, host_id, token)
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    # The host's broker socket; or the endpoint coordinates for an in-process
    # caller that holds the launch token itself.
    parser.add_argument("--socket")
    parser.add_argument("--server")
    parser.add_argument("--host-id")
    parser.add_argument("--host-token")
    # git passes the operation (get/store/erase) as the first positional arg.
    parser.add_argument("operation", nargs="?", default="get")
    args, _ = parser.parse_known_args(argv)

    # Only ``get`` returns credentials; ``store``/``erase`` are no-ops (nothing
    # is persisted — the token is re-fetched next time).
    if args.operation != "get":
        return 0

    request = _read_git_request()
    # Fail closed: only vend for github.com over https. A missing/empty host, a
    # different host, or a non-https protocol declines (return 0, no output) so
    # git falls through to the next helper — the brokered token can never leak
    # to another host even if this is ever wired as a global credential helper.
    if request.get("host") != "github.com" or request.get("protocol") != "https":
        return 0

    cred: tuple[str, str] | None = None
    if args.socket:
        reply = credential_broker.request_credential(
            args.socket, {"protocol": "https", "host": "github.com"}
        )
        if reply.get("password"):
            cred = (reply.get("username") or "x-access-token", reply["password"])
    elif args.server and args.host_id and args.host_token:
        cred = _fetch_credential(args.server, args.host_id, args.host_token)
    if cred is None:
        return 0  # decline; git tries the next helper / prompts
    username, token = cred
    sys.stdout.write(f"username={username}\npassword={token}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
