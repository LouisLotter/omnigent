"""Reference key service for the credential store's HTTP cipher tests.

Not a test module. :class:`FakeKeyService` implements the endpoint side of the
protocol in ``designs/CREDENTIAL_STORE.md`` ("HTTP key service") independently
of the client code: it verifies signatures and timestamps itself, and seals each
record with AES-256-GCM under a key derived from the record's identity, the
construction the design suggests. Tests reach it in process through
:meth:`FakeKeyService.transport`, or over real HTTP through :meth:`FakeKeyService.serve`.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import httpx
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

PROTOCOL = "omnigent-key-service-v1"
FRESHNESS_WINDOW_S = 300
MAX_PLAINTEXT_BYTES = 32 * 1024
#: Path under which :meth:`FakeKeyService.serve` mounts the two operations.
BASE_PATH = "/key-service/v1"

_IV_BYTES = 12
_CIPHERTEXT_RE = re.compile(r"[A-Za-z0-9_-]+")

Reply = tuple[int, dict[str, Any]]


def _error(status: int, code: str) -> Reply:
    return status, {"error": {"code": code, "message": code.replace("_", " ")}}


def b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def b64url_decode(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def derivation_info(context: Mapping[str, str], key_id: str, key_version: int) -> bytes:
    """HKDF ``info`` for a record: canonical JSON of its identity and key."""
    info = {
        "context": dict(context),
        "key_id": key_id,
        "key_version": key_version,
        "purpose": "omnigent-credential-record",
    }
    return json.dumps(info, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def seal(
    root_key: bytes,
    *,
    context: Mapping[str, str],
    key_id: str,
    key_version: int,
    plaintext: str,
    iv: bytes,
) -> str:
    """Seal *plaintext* the suggested way: ``b64url(iv || AES-256-GCM(ciphertext || tag))``."""
    record_key = _record_key(root_key, context, key_id, key_version)
    return b64url(iv + AESGCM(record_key).encrypt(iv, plaintext.encode("utf-8"), None))


def _record_key(
    root_key: bytes, context: Mapping[str, str], key_id: str, key_version: int
) -> bytes:
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=derivation_info(context, key_id, key_version),
    )
    return hkdf.derive(root_key)


def _is_context(value: object) -> bool:
    return isinstance(value, dict) and all(
        isinstance(key, str) and isinstance(item, str) for key, item in value.items()
    )


class FakeKeyService:
    """An in-memory key service endpoint.

    :param signing_keys: Request-signing keys the endpoint accepts, by key ID.
    :param record_keys: Root record keys by ``(key_id, key_version)``.
    :param current: The record key new encryptions use.
    :param clock: The endpoint's clock, in Unix seconds.
    """

    def __init__(
        self,
        *,
        signing_keys: dict[str, bytes],
        record_keys: dict[tuple[str, int], bytes],
        current: tuple[str, int],
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.signing_keys = signing_keys
        self.record_keys = record_keys
        self.current = current
        self.clock = clock
        #: ``(operation, status)`` of every request served, refusals included.
        self.served: list[tuple[str, int]] = []

    def respond(self, operation: str, headers: Mapping[str, str], body: bytes) -> Reply:
        """Answer one request to ``<base>/<operation>`` with ``(status, JSON body)``."""
        reply = self._respond(operation, {k.lower(): v for k, v in headers.items()}, body)
        self.served.append((operation, reply[0]))
        return reply

    def _respond(self, operation: str, headers: dict[str, str], body: bytes) -> Reply:
        if operation not in ("encrypt", "decrypt"):
            return _error(404, "not_found")
        refusal = self._authenticate(operation, headers, body)
        if refusal is not None:
            return refusal
        try:
            request = json.loads(body)
        except ValueError:
            return _error(400, "invalid_request")
        if not isinstance(request, dict) or not _is_context(request.get("context")):
            return _error(400, "invalid_request")
        if operation == "encrypt":
            return self._encrypt(request)
        return self._decrypt(request)

    def _authenticate(self, operation: str, headers: dict[str, str], body: bytes) -> Reply | None:
        key_id = headers.get("x-omnigent-signing-key-id", "")
        timestamp = headers.get("x-omnigent-timestamp", "")
        signature = headers.get("x-omnigent-signature", "")
        signing_key = self.signing_keys.get(key_id)
        if signing_key is None or not timestamp.isdigit() or not signature:
            return _error(401, "unauthenticated")
        message = "\n".join(
            (PROTOCOL, operation, key_id, timestamp, hashlib.sha256(body).hexdigest())
        )
        expected = hmac.new(signing_key, message.encode("utf-8"), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, signature):
            return _error(401, "unauthenticated")
        if abs(self.clock() - int(timestamp)) > FRESHNESS_WINDOW_S:
            return _error(401, "stale_request")
        return None

    def _encrypt(self, request: dict[str, Any]) -> Reply:
        plaintext = request.get("plaintext")
        if not isinstance(plaintext, str):
            return _error(400, "invalid_request")
        if len(plaintext.encode("utf-8")) > MAX_PLAINTEXT_BYTES:
            return _error(413, "too_large")
        key_id, key_version = self.current
        ciphertext = seal(
            self.record_keys[self.current],
            context=request["context"],
            key_id=key_id,
            key_version=key_version,
            plaintext=plaintext,
            iv=os.urandom(_IV_BYTES),
        )
        return 200, {"key_id": key_id, "key_version": key_version, "ciphertext": ciphertext}

    def _decrypt(self, request: dict[str, Any]) -> Reply:
        key_id = request.get("key_id")
        key_version = request.get("key_version")
        ciphertext = request.get("ciphertext")
        if (
            not isinstance(key_id, str)
            or isinstance(key_version, bool)
            or not isinstance(key_version, int)
            or not isinstance(ciphertext, str)
        ):
            return _error(400, "invalid_request")
        root_key = self.record_keys.get((key_id, key_version))
        if root_key is None:
            return _error(409, "key_unavailable")
        if not _CIPHERTEXT_RE.fullmatch(ciphertext):
            return _error(422, "unusable")
        record_key = _record_key(root_key, request["context"], key_id, key_version)
        try:
            raw = b64url_decode(ciphertext)
            opened = AESGCM(record_key).decrypt(raw[:_IV_BYTES], raw[_IV_BYTES:], None)
            plaintext = opened.decode("utf-8")
        except (InvalidTag, ValueError):
            return _error(422, "unusable")
        return 200, {"plaintext": plaintext}

    def transport(
        self, *, alter_request: Callable[[httpx.Request], None] | None = None
    ) -> httpx.MockTransport:
        """An in-process transport to this endpoint; *alter_request* edits requests in transit."""

        def handle(request: httpx.Request) -> httpx.Response:
            if alter_request is not None:
                alter_request(request)
            operation = request.url.path.rsplit("/", 1)[-1]
            status, payload = self.respond(operation, request.headers, request.content)
            return httpx.Response(status, json=payload)

        return httpx.MockTransport(handle)

    @contextmanager
    def serve(self) -> Iterator[str]:
        """Serve this endpoint over HTTP on loopback; yields its base URL."""
        service = self

        class _Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                operation = self.path.removeprefix(f"{BASE_PATH}/")
                status, payload = service.respond(operation, dict(self.headers.items()), body)
                data = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, format: str, *args: Any) -> None:
                del format, args  # keep test output quiet

        server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_address[1]}{BASE_PATH}"
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
