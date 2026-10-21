"""HTTP key-service :class:`SecretCipher`, for deployments without AWS KMS or Vault.

Omnigent sends each secret, with its row identity as context, to a configured
endpoint that encrypts and decrypts with keys it never discloses, so the key
material stays out of Omnigent's process, as with KMS. Any platform that can
host a small authenticated endpoint can implement the service; the wire
protocol is specified in ``designs/CREDENTIAL_STORE.md`` ("HTTP key service").

Every request is signed with HMAC-SHA256 under a signing key from the
environment and carries the signing key's ID and a timestamp, so the endpoint
refuses forged and stale requests. The endpoint names the record key it used
(an ID and a version), and the stored ciphertext carries both, so the endpoint
can rotate keys and still open older records.

Only the endpoint's typed ``unusable`` verdict (a wrong context, or a corrupt or
tampered record) and an unrecognised stored value make :meth:`decrypt` return
``None``. Everything else (a refused signature, an unknown key, an unreachable
endpoint, a malformed reply) raises :class:`KeyServiceError`, so a store-wide
fault never reads as a per-user disconnect, which would re-encrypt over a
still-recoverable record on reconnect.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import re
import time
from collections.abc import Callable
from typing import Any
from urllib.parse import urlsplit

import httpx

from omnigent.stores.credential_store.secret_cipher import SecretContext, SecretTooLargeError

_logger = logging.getLogger(__name__)

#: Base URL of the key service; Omnigent posts to ``<url>/encrypt`` and ``<url>/decrypt``.
CREDENTIAL_HTTP_URL_ENV_VAR = "OMNIGENT_CREDENTIAL_HTTP_URL"
#: HMAC-SHA256 key that signs requests (its UTF-8 bytes, at least 32 of them).
CREDENTIAL_HTTP_SIGNING_KEY_ENV_VAR = "OMNIGENT_CREDENTIAL_HTTP_SIGNING_KEY"
#: ID of that signing key, sent with each request so the endpoint can rotate keys.
CREDENTIAL_HTTP_SIGNING_KEY_ID_ENV_VAR = "OMNIGENT_CREDENTIAL_HTTP_SIGNING_KEY_ID"

#: First line of every string to sign, so these signatures cannot be replayed
#: against another protocol that shares the key.
PROTOCOL = "omnigent-key-service-v1"
SIGNING_KEY_ID_HEADER = "X-Omnigent-Signing-Key-Id"
TIMESTAMP_HEADER = "X-Omnigent-Timestamp"
SIGNATURE_HEADER = "X-Omnigent-Signature"

MIN_SIGNING_KEY_BYTES = 32
#: Largest serialized secret sent; every conforming endpoint accepts this much.
MAX_PLAINTEXT_BYTES = 32 * 1024
_DEFAULT_TIMEOUT_S = 10.0

#: Stored form: ``oshttp1:<key_id>:<key_version>:<ciphertext>``.
_ENVELOPE_PREFIX = "oshttp1"
_KEY_ID_RE = re.compile(r"[A-Za-z0-9._-]{1,128}")
_KEY_VERSION_RE = re.compile(r"[1-9][0-9]{0,9}")
_MAX_KEY_VERSION = 2**31 - 1
_CIPHERTEXT_RE = re.compile(r"[A-Za-z0-9_-]+")
_ERROR_CODE_RE = re.compile(r"[a-z0-9_]{1,64}")


class KeyServiceError(RuntimeError):
    """The key service could not be used: a store-wide fault, never a per-record one."""


def build_http_secret_cipher() -> HttpSecretCipher | None:
    """Build the HTTP key-service cipher from the environment, or ``None``.

    ``None`` when none of its three settings is set, so the backend is simply
    not configured. Setting only some of them raises :class:`ValueError` rather
    than silently disabling the credential store.
    """
    settings = {
        name: os.environ.get(name, "").strip()
        for name in (
            CREDENTIAL_HTTP_URL_ENV_VAR,
            CREDENTIAL_HTTP_SIGNING_KEY_ENV_VAR,
            CREDENTIAL_HTTP_SIGNING_KEY_ID_ENV_VAR,
        )
    }
    if not any(settings.values()):
        return None
    missing = sorted(name for name, value in settings.items() if not value)
    if missing:
        raise ValueError(
            "the HTTP credential backend is only partly configured; "
            f"also set {', '.join(missing)}."
        )
    return HttpSecretCipher(
        settings[CREDENTIAL_HTTP_URL_ENV_VAR],
        signing_key=settings[CREDENTIAL_HTTP_SIGNING_KEY_ENV_VAR].encode("utf-8"),
        signing_key_id=settings[CREDENTIAL_HTTP_SIGNING_KEY_ID_ENV_VAR],
    )


def _validated_base_url(url: str) -> str:
    """Return *url* without a trailing slash, or raise if it can't be a key service URL."""
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ValueError("credential key service URL must be http(s) with a host.")
    if parts.username or parts.password or parts.query or parts.fragment:
        raise ValueError(
            "credential key service URL must not carry credentials, a query or a fragment."
        )
    # Reading .port raises ValueError for a malformed or out-of-range port.
    if parts.port == 0:
        raise ValueError("credential key service URL must not use port 0.")
    return url.rstrip("/")


def _string_to_sign(*, operation: str, signing_key_id: str, timestamp: int, body: bytes) -> str:
    """The protocol's canonical request string; see ``designs/CREDENTIAL_STORE.md``."""
    body_digest = hashlib.sha256(body).hexdigest()
    return "\n".join((PROTOCOL, operation, signing_key_id, str(timestamp), body_digest))


def _signature_headers(
    *, operation: str, signing_key: bytes, signing_key_id: str, timestamp: int, body: bytes
) -> dict[str, str]:
    """Headers that authenticate one request to the key service."""
    message = _string_to_sign(
        operation=operation, signing_key_id=signing_key_id, timestamp=timestamp, body=body
    )
    signature = hmac.new(signing_key, message.encode("utf-8"), hashlib.sha256).hexdigest()
    return {
        SIGNING_KEY_ID_HEADER: signing_key_id,
        TIMESTAMP_HEADER: str(timestamp),
        SIGNATURE_HEADER: signature,
    }


def _wire_context(context: SecretContext) -> dict[str, str]:
    """The row identity as sent; empty values are dropped, as the KMS and Vault ciphers do."""
    return {key: value for key, value in context.items() if value != ""}


def _valid_key_version(value: int) -> bool:
    return not isinstance(value, bool) and 1 <= value <= _MAX_KEY_VERSION


def _wrap(key_id: str, key_version: int, ciphertext: str) -> str:
    return f"{_ENVELOPE_PREFIX}:{key_id}:{key_version}:{ciphertext}"


def _unwrap(stored: str) -> tuple[str, int, str] | None:
    """Return ``(key_id, key_version, ciphertext)``, or ``None`` if *stored* is not ours."""
    parts = stored.split(":")
    if len(parts) != 4 or parts[0] != _ENVELOPE_PREFIX:
        return None
    _prefix, key_id, version_text, ciphertext = parts
    if not (
        _KEY_ID_RE.fullmatch(key_id)
        and _KEY_VERSION_RE.fullmatch(version_text)
        and _CIPHERTEXT_RE.fullmatch(ciphertext)
    ):
        return None
    key_version = int(version_text)
    if not _valid_key_version(key_version):
        return None
    return key_id, key_version, ciphertext


def _sealed_record(body: Any) -> tuple[str, int, str]:
    """``(key_id, key_version, ciphertext)`` from an encrypt reply; raises if malformed."""
    if isinstance(body, dict):
        key_id = body.get("key_id")
        key_version = body.get("key_version")
        ciphertext = body.get("ciphertext")
        if (
            isinstance(key_id, str)
            and _KEY_ID_RE.fullmatch(key_id)
            and isinstance(key_version, int)
            and _valid_key_version(key_version)
            and isinstance(ciphertext, str)
            and _CIPHERTEXT_RE.fullmatch(ciphertext)
        ):
            return key_id, key_version, ciphertext
    raise KeyServiceError("credential key service sent a malformed encrypt reply.")


def _error_code(body: Any) -> str | None:
    """The ``error.code`` of an error reply, or ``None`` when absent or malformed."""
    error = body.get("error") if isinstance(body, dict) else None
    code = error.get("code") if isinstance(error, dict) else None
    return code if isinstance(code, str) and _ERROR_CODE_RE.fullmatch(code) else None


class HttpSecretCipher:
    """:class:`SecretCipher` backed by a signed HTTP key service.

    :param url: Base URL of the key service (``http`` or ``https``).
    :param signing_key: HMAC-SHA256 key for request signatures.
    :param signing_key_id: ID the endpoint knows that key by.
    :param timeout: Per-request timeout in seconds.
    :param client: Optional pre-built ``httpx.Client`` (tests inject a transport).
    :param clock: Source of the request timestamp, in Unix seconds.
    """

    def __init__(
        self,
        url: str,
        *,
        signing_key: bytes,
        signing_key_id: str,
        timeout: float = _DEFAULT_TIMEOUT_S,
        client: httpx.Client | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if len(signing_key) < MIN_SIGNING_KEY_BYTES:
            raise ValueError(
                f"credential key service signing key must be at least "
                f"{MIN_SIGNING_KEY_BYTES} bytes."
            )
        if not _KEY_ID_RE.fullmatch(signing_key_id):
            raise ValueError(
                "credential key service signing key ID must be 1-128 characters "
                f"of letters, digits, '.', '_' or '-': {signing_key_id!r}"
            )
        self._base_url = _validated_base_url(url)
        self._signing_key = signing_key
        self._signing_key_id = signing_key_id
        self._client = client if client is not None else httpx.Client(timeout=timeout)
        self._clock = clock

    def __repr__(self) -> str:
        # Never include the signing key.
        return f"HttpSecretCipher(url={self._base_url!r}, signing_key_id={self._signing_key_id!r})"

    def encrypt(self, plaintext: str, *, context: SecretContext) -> str:
        if len(plaintext.encode("utf-8")) > MAX_PLAINTEXT_BYTES:
            raise SecretTooLargeError(
                f"Credential exceeds the key service limit of {MAX_PLAINTEXT_BYTES} "
                "serialized bytes."
            )
        status, body = self._call(
            "encrypt", {"context": _wire_context(context), "plaintext": plaintext}
        )
        if status == 413 and _error_code(body) == "too_large":
            raise SecretTooLargeError("Credential exceeds the key service's size limit.")
        if status != 200:
            raise self._refusal("encrypt", status, body)
        return _wrap(*_sealed_record(body))

    def decrypt(self, ciphertext: str, *, context: SecretContext) -> str | None:
        """Return the plaintext, or ``None`` when this record is unusable.

        ``None`` only for a stored value that is not this backend's envelope, or
        the endpoint's ``422 unusable`` verdict (wrong context, corrupt or
        tampered record). Any other outcome raises :class:`KeyServiceError`.
        """
        record = _unwrap(ciphertext)
        if record is None:
            _logger.warning(
                "credential key service: stored ciphertext is not a recognized envelope "
                "(corrupt/foreign) — the affected integration will read as disconnected "
                "until reconnected"
            )
            return None
        key_id, key_version, sealed = record
        status, body = self._call(
            "decrypt",
            {
                "context": _wire_context(context),
                "key_id": key_id,
                "key_version": key_version,
                "ciphertext": sealed,
            },
        )
        if status == 422 and _error_code(body) == "unusable":
            _logger.warning(
                "credential key service: record under key %s version %d does not open "
                "(wrong context or corrupt) — the affected integration will read as "
                "disconnected until reconnected",
                key_id,
                key_version,
            )
            return None
        if status != 200:
            raise self._refusal("decrypt", status, body)
        plaintext = body.get("plaintext") if isinstance(body, dict) else None
        if not isinstance(plaintext, str):
            raise KeyServiceError("credential key service sent a malformed decrypt reply.")
        return plaintext

    def _call(self, operation: str, payload: dict[str, Any]) -> tuple[int, Any]:
        """POST a signed request; return the status and the parsed JSON body, if any."""
        body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            **_signature_headers(
                operation=operation,
                signing_key=self._signing_key,
                signing_key_id=self._signing_key_id,
                timestamp=int(self._clock()),
                body=body,
            ),
        }
        try:
            response = self._client.post(
                f"{self._base_url}/{operation}", content=body, headers=headers
            )
        except httpx.HTTPError as exc:
            raise KeyServiceError(
                f"credential key service unreachable for {operation}: {type(exc).__name__}"
            ) from exc
        try:
            parsed = response.json()
        except ValueError:
            parsed = None
        return response.status_code, parsed

    @staticmethod
    def _refusal(operation: str, status: int, body: Any) -> KeyServiceError:
        # The endpoint's own message is not repeated: it is not ours to log.
        code = _error_code(body) or "no error code"
        return KeyServiceError(
            f"credential key service refused {operation}: HTTP {status} ({code})"
        )
