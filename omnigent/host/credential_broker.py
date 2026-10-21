"""Serve provider credentials from the host process over a local Unix socket.

A managed host holds its launch token in memory. Rather than writing that token,
or a credential it vends, into git or gh configuration, the host serves this
socket and fetches a fresh credential from the server for each request. Clients
name the socket by absolute path: git's credential helper has it in
``~/.gitconfig``, and the gh wrapper (:func:`gh_main`) reads
:data:`SOCKET_ENV_VAR`.

The socket is 0600 inside the owner-only harness socket root
(``/tmp/omnigent-<uid>`` by default), outside the home directory and workspace,
so snapshots of those never capture it. A restarted host replaces it at the
same path.

Threat model: this removes the on-disk copies only. Any process running as the
same user can still ask the broker for a credential while the host runs.

Wire format, one exchange per connection, in both directions: git's credential
format, ``key=value`` lines ended by a blank line or end of stream. An empty
reply declines.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import logging
import os
import secrets
import socket
import socketserver
import sys
import threading
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import NoReturn

_logger = logging.getLogger(__name__)

# Runner-visible path of the live broker socket. A path, not a secret, like
# ``SSH_AUTH_SOCK``; gh's wrapper reads it.
SOCKET_ENV_VAR = "OMNIGENT_HOST_BROKER_SOCKET"

# Maps a request's fields to reply fields, or ``None`` to decline.
Resolver = Callable[[dict[str, str]], Mapping[str, str] | None]

_MAX_FIELDS = 32
_MAX_LINE_BYTES = 8192
# Covers the server round trip a resolver makes (15 s in the GitHub resolver).
_CLIENT_TIMEOUT_S = 20.0
_REQUEST_READ_TIMEOUT_S = 5.0


def broker_socket_path(host_id: str) -> Path:
    """Return the host's socket path in the harness socket root.

    Keyed by host identity, so a restarted host serves the path that git's
    config and long-lived terminals already name.
    """
    # Imported here so the gh wrapper's import stays light.
    from omnigent.runtime.harnesses.paths import resolve_harness_tmp_parent

    digest = hashlib.sha256(host_id.encode()).hexdigest()[:16]
    return resolve_harness_tmp_parent() / f"credentials-{digest}.sock"


def read_fields(stream: io.BufferedIOBase) -> dict[str, str]:
    """Parse ``key=value`` lines up to a blank line or end of stream, bounded."""
    fields: dict[str, str] = {}
    for _ in range(_MAX_FIELDS):
        line = stream.readline(_MAX_LINE_BYTES).decode("utf-8", "replace").rstrip("\r\n")
        if not line:
            break
        key, separator, value = line.partition("=")
        if separator:
            fields[key] = value
    return fields


def format_fields(fields: Mapping[str, str]) -> bytes:
    """Serialize *fields*; a value that could inject another line declines all."""
    if any("\n" in f"{key}{value}" or "\0" in f"{key}{value}" for key, value in fields.items()):
        return b""
    return "".join(f"{key}={value}\n" for key, value in fields.items()).encode()


class _RequestHandler(socketserver.StreamRequestHandler):
    timeout = _REQUEST_READ_TIMEOUT_S

    def handle(self) -> None:
        resolve: Resolver = self.server.resolve  # type: ignore[attr-defined]
        try:
            reply = resolve(read_fields(self.rfile)) or {}
        except Exception as exc:  # noqa: BLE001 - a failed request declines.
            # Only the type: a resolver's message could carry a credential.
            _logger.warning("credential broker: request failed: %s", type(exc).__name__)
            reply = {}
        with contextlib.suppress(OSError):
            self.wfile.write(format_fields(reply))


def start_credential_broker(socket_path: Path, resolve: Resolver) -> socketserver.BaseServer:
    """Serve *resolve* on *socket_path* from a daemon thread.

    Validates the parent as an owner-only directory, binds the socket under a
    private name with mode 0600, then renames it over *socket_path*, replacing
    any socket an earlier or concurrent host process left there in one step.
    Closing the server leaves the path alone, since a newer host may own it.

    :returns: The running server.
    :raises RuntimeError: If the parent directory is not owner-only.
    :raises OSError: If the socket cannot be bound or published.
    """
    from omnigent.runtime.harnesses.paths import ensure_private_dir

    socket_path = Path(os.path.abspath(socket_path))
    # A configured root may be nested; only the socket's own directory must be owner-only.
    socket_path.parent.parent.mkdir(parents=True, exist_ok=True)
    ensure_private_dir(socket_path.parent, os.getuid())
    staging = socket_path.with_name(f".{os.getpid()}-{secrets.token_hex(4)}.sock")

    class _Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
        daemon_threads = True

    server = _Server(str(staging), _RequestHandler)
    server.resolve = resolve  # type: ignore[attr-defined]
    try:
        os.chmod(staging, 0o600)
        os.replace(staging, socket_path)
    except OSError:
        server.server_close()
        with contextlib.suppress(OSError):
            staging.unlink()
        raise
    threading.Thread(
        target=server.serve_forever, name="host-credential-broker", daemon=True
    ).start()
    return server


def request_credential(
    socket_path: str, request: Mapping[str, str], *, timeout: float = _CLIENT_TIMEOUT_S
) -> dict[str, str]:
    """Ask the broker at *socket_path*; an unreachable broker declines (``{}``)."""
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(timeout)
            connection.connect(socket_path)
            connection.sendall(format_fields(request) + b"\n")
            connection.shutdown(socket.SHUT_WR)
            with connection.makefile("rb") as reply:
                return read_fields(reply)
    except OSError:
        return {}


def _next_on_path(name: str, wrapper_path: str) -> str | None:
    """Return the first *name* on ``PATH`` after the wrapper's own entry."""
    own = os.path.realpath(wrapper_path)
    candidates = [
        os.path.join(directory, name)
        for directory in os.environ.get("PATH", "").split(os.pathsep)
        if os.path.isabs(directory)
    ]
    resolved = [os.path.realpath(candidate) for candidate in candidates]
    start = resolved.index(own) + 1 if own in resolved else 0
    for candidate, real in zip(candidates[start:], resolved[start:], strict=True):
        if real != own and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def gh_main(argv: list[str] | None = None) -> NoReturn:
    """Run the next ``gh`` on ``PATH`` with the broker's GitHub token as ``GH_TOKEN``.

    Installed as ``gh`` ahead of the real binary in a managed host image. A
    ``GH_TOKEN`` or ``GITHUB_TOKEN`` already set wins; without a broker, or when
    it declines, gh runs unchanged. gh scopes ``GH_TOKEN`` to github.com.
    """
    args = sys.argv[1:] if argv is None else argv
    real_gh = _next_on_path("gh", sys.argv[0])
    if real_gh is None:
        print("gh: the GitHub CLI is not installed", file=sys.stderr)
        raise SystemExit(127)
    env = dict(os.environ)
    socket_path = env.get(SOCKET_ENV_VAR)
    if socket_path and not (env.get("GH_TOKEN") or env.get("GITHUB_TOKEN")):
        reply = request_credential(socket_path, {"protocol": "https", "host": "github.com"})
        if reply.get("password"):
            env["GH_TOKEN"] = reply["password"]
    os.execve(real_gh, [real_gh, *args], env)
