"""Tests for the HTTP key-service :class:`SecretCipher`.

The cipher talks to :class:`~tests.server.fake_key_service.FakeKeyService`, a
reference endpoint that checks signatures and seals records independently of
the client code: in process through an httpx transport, or over loopback HTTP
where the transport itself matters. The protocol's published test vectors
(``tests/fixtures/credential_key_service/protocol_v1.json``) pin the wire format
for other implementations of the endpoint.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from omnigent.stores.credential_store.http_cipher import (
    CREDENTIAL_HTTP_SIGNING_KEY_ENV_VAR,
    CREDENTIAL_HTTP_SIGNING_KEY_ID_ENV_VAR,
    CREDENTIAL_HTTP_URL_ENV_VAR,
    MAX_PLAINTEXT_BYTES,
    SIGNATURE_HEADER,
    HttpSecretCipher,
    KeyServiceError,
)
from omnigent.stores.credential_store.secret_cipher import (
    CREDENTIAL_CIPHER_ENV_VAR,
    CREDENTIAL_KMS_KEY_ENV_VAR,
    SecretCipher,
    SecretTooLargeError,
    build_secret_cipher,
)
from omnigent.stores.credential_store.vault_cipher import CREDENTIAL_VAULT_KEY_ENV_VAR
from tests.server.fake_key_service import FakeKeyService, b64url, b64url_decode

CTX_ALICE = {"workspace_id": "0", "user_id": "alice", "provider": "github", "account_id": ""}
CTX_BOB = {"workspace_id": "0", "user_id": "bob", "provider": "github", "account_id": ""}

SIGNING_KEY_ID = "signing-1"
SIGNING_KEY = b"test-signing-key-for-the-fake-key-service"
RECORD_KEY_ID = "records-test"
URL = "http://key-service.test/v1"

_FIXTURE = Path(__file__).parents[1] / "fixtures" / "credential_key_service" / "protocol_v1.json"


def _service(**overrides: Any) -> FakeKeyService:
    settings: dict[str, Any] = {
        "signing_keys": {SIGNING_KEY_ID: SIGNING_KEY},
        "record_keys": {(RECORD_KEY_ID, 1): bytes(range(32))},
        "current": (RECORD_KEY_ID, 1),
    }
    return FakeKeyService(**{**settings, **overrides})


def _cipher(
    service: FakeKeyService,
    *,
    signing_key: bytes = SIGNING_KEY,
    signing_key_id: str = SIGNING_KEY_ID,
    clock: Callable[[], float] | None = None,
    alter_request: Callable[[httpx.Request], None] | None = None,
) -> HttpSecretCipher:
    client = httpx.Client(transport=service.transport(alter_request=alter_request))
    extra = {"clock": clock} if clock is not None else {}
    return HttpSecretCipher(
        URL, signing_key=signing_key, signing_key_id=signing_key_id, client=client, **extra
    )


def _replying(status: int, body: bytes) -> HttpSecretCipher:
    """A cipher whose endpoint answers every request with *status* and raw *body*."""
    transport = httpx.MockTransport(lambda request: httpx.Response(status, content=body))
    return HttpSecretCipher(
        URL,
        signing_key=SIGNING_KEY,
        signing_key_id=SIGNING_KEY_ID,
        client=httpx.Client(transport=transport),
    )


# ── the SecretCipher contract ──────────────────────────────────────────────


def test_satisfies_port() -> None:
    assert isinstance(_cipher(_service()), SecretCipher)


def test_roundtrip_through_the_endpoint() -> None:
    service = _service()
    cipher = _cipher(service)
    stored = cipher.encrypt("ghu_secret_token", context=CTX_ALICE)
    assert stored.startswith(f"oshttp1:{RECORD_KEY_ID}:1:")
    assert "ghu_secret_token" not in stored
    assert cipher.decrypt(stored, context=CTX_ALICE) == "ghu_secret_token"
    assert service.served == [("encrypt", 200), ("decrypt", 200)]


def test_context_is_required() -> None:
    with pytest.raises(TypeError):
        _cipher(_service()).encrypt("x")  # type: ignore[call-arg]


def test_another_records_context_reads_as_not_linked() -> None:
    cipher = _cipher(_service())
    stored = cipher.encrypt("ghu_alice", context=CTX_ALICE)
    assert cipher.decrypt(stored, context=CTX_BOB) is None
    # account_id="" is dropped; a named account is a distinct, still-usable context.
    named = {**CTX_ALICE, "account_id": "org1"}
    assert cipher.decrypt(cipher.encrypt("y", context=named), context=named) == "y"
    assert cipher.decrypt(stored, context=named) is None


def test_corrupt_stored_value_reads_as_not_linked_without_a_call() -> None:
    service = _service()
    cipher = _cipher(service)
    for stored in ("not-a-real-ciphertext!!", "oshttp1:k:0:abc", "oshttp1:k:1:a/b", "osvault:x:y"):
        assert cipher.decrypt(stored, context=CTX_ALICE) is None
    assert service.served == []


def test_tampered_record_reads_as_not_linked() -> None:
    cipher = _cipher(_service())
    stored = cipher.encrypt("ghu_alice", context=CTX_ALICE)
    envelope, sealed = stored.rsplit(":", 1)
    flipped = bytearray(b64url_decode(sealed))
    flipped[-1] ^= 1
    assert cipher.decrypt(f"{envelope}:{b64url(bytes(flipped))}", context=CTX_ALICE) is None
    assert cipher.decrypt(f"{envelope}:AAAA", context=CTX_ALICE) is None


# ── store-wide failures raise, never read as not linked ────────────────────


def test_record_key_the_endpoint_does_not_hold_raises() -> None:
    """A record whose key ID the endpoint does not hold (repointed or retired) is store-wide."""
    stored = _cipher(_service()).encrypt("ghu_alice", context=CTX_ALICE)
    other = _service(record_keys={("records-other", 1): bytes(32)}, current=("records-other", 1))
    with pytest.raises(KeyServiceError, match="409 \\(key_unavailable\\)"):
        _cipher(other).decrypt(stored, context=CTX_ALICE)


def test_unknown_signing_key_id_raises() -> None:
    stored = _cipher(_service()).encrypt("ghu_alice", context=CTX_ALICE)
    cipher = _cipher(_service(), signing_key_id="signing-unknown")
    with pytest.raises(KeyServiceError, match="401 \\(unauthenticated\\)"):
        cipher.encrypt("x", context=CTX_ALICE)
    with pytest.raises(KeyServiceError, match="401 \\(unauthenticated\\)"):
        cipher.decrypt(stored, context=CTX_ALICE)


def test_wrong_signing_key_raises() -> None:
    cipher = _cipher(_service(), signing_key=b"some-other-signing-key-of-enough-length")
    with pytest.raises(KeyServiceError, match="401 \\(unauthenticated\\)"):
        cipher.encrypt("x", context=CTX_ALICE)


def test_unsigned_request_raises() -> None:
    def strip_signature(request: httpx.Request) -> None:
        del request.headers[SIGNATURE_HEADER]

    service = _service()
    stored = _cipher(service).encrypt("ghu_alice", context=CTX_ALICE)
    cipher = _cipher(service, alter_request=strip_signature)
    with pytest.raises(KeyServiceError, match="401 \\(unauthenticated\\)"):
        cipher.decrypt(stored, context=CTX_ALICE)
    with pytest.raises(KeyServiceError, match="401 \\(unauthenticated\\)"):
        cipher.encrypt("x", context=CTX_ALICE)


@pytest.mark.parametrize("skew_s", [-301, 301])
def test_expired_request_raises(skew_s: int) -> None:
    service = _service(clock=lambda: 1_800_000_000)
    stored = _cipher(service, clock=lambda: 1_800_000_000).encrypt("t", context=CTX_ALICE)
    cipher = _cipher(service, clock=lambda: 1_800_000_000 + skew_s)
    with pytest.raises(KeyServiceError, match="401 \\(stale_request\\)"):
        cipher.decrypt(stored, context=CTX_ALICE)
    with pytest.raises(KeyServiceError, match="401 \\(stale_request\\)"):
        cipher.encrypt("t", context=CTX_ALICE)


def test_unreachable_endpoint_raises() -> None:
    service = _service()
    with service.serve() as url:
        stored = HttpSecretCipher(
            url, signing_key=SIGNING_KEY, signing_key_id=SIGNING_KEY_ID
        ).encrypt("ghu_alice", context=CTX_ALICE)
    # The server is gone, so its port now refuses connections.
    cipher = HttpSecretCipher(url, signing_key=SIGNING_KEY, signing_key_id=SIGNING_KEY_ID)
    with pytest.raises(KeyServiceError, match="unreachable for decrypt: ConnectError"):
        cipher.decrypt(stored, context=CTX_ALICE)
    with pytest.raises(KeyServiceError, match="unreachable for encrypt"):
        cipher.encrypt("x", context=CTX_ALICE)


def test_timeout_raises() -> None:
    def time_out(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    cipher = HttpSecretCipher(
        URL,
        signing_key=SIGNING_KEY,
        signing_key_id=SIGNING_KEY_ID,
        client=httpx.Client(transport=httpx.MockTransport(time_out)),
    )
    with pytest.raises(KeyServiceError, match="ReadTimeout"):
        cipher.decrypt(f"oshttp1:{RECORD_KEY_ID}:1:AAAA", context=CTX_ALICE)


@pytest.mark.parametrize(
    ("status", "body"),
    [
        (500, b""),
        (502, b"<html>bad gateway</html>"),
        (422, b'{"error":{"code":"invalid_request"}}'),
        (422, b"not json"),
        (400, b'{"error":{"code":"unusable"}}'),
        (307, b""),
        (200, b'{"plaintext":7}'),
        (200, b"[]"),
    ],
)
def test_decrypt_failure_other_than_unusable_raises(status: int, body: bytes) -> None:
    """Only the endpoint's typed ``422 unusable`` verdict reads as not linked."""
    with pytest.raises(KeyServiceError):
        _replying(status, body).decrypt(f"oshttp1:{RECORD_KEY_ID}:1:AAAA", context=CTX_ALICE)


@pytest.mark.parametrize(
    "body",
    [
        b'{"key_id":"k","key_version":1}',
        b'{"key_id":"k:1","key_version":1,"ciphertext":"AAAA"}',
        b'{"key_id":"k","key_version":0,"ciphertext":"AAAA"}',
        b'{"key_id":"k","key_version":true,"ciphertext":"AAAA"}',
        b'{"key_id":"k","key_version":"1","ciphertext":"AAAA"}',
        b'{"key_id":"k","key_version":1,"ciphertext":"AA+/"}',
        b"not json",
    ],
)
def test_malformed_encrypt_reply_raises(body: bytes) -> None:
    with pytest.raises(KeyServiceError, match="malformed encrypt reply"):
        _replying(200, body).encrypt("x", context=CTX_ALICE)


def test_oversized_plaintext_is_refused() -> None:
    service = _service()
    with pytest.raises(SecretTooLargeError):
        _cipher(service).encrypt("x" * (MAX_PLAINTEXT_BYTES + 1), context=CTX_ALICE)
    assert service.served == []
    with pytest.raises(SecretTooLargeError):
        _replying(413, b'{"error":{"code":"too_large"}}').encrypt("x", context=CTX_ALICE)


# ── key rotation: the stored record carries the key ID and version ─────────


def test_record_key_rotation_keeps_older_records_readable() -> None:
    service = _service()
    cipher = _cipher(service)
    old = cipher.encrypt("ghu_old", context=CTX_ALICE)
    service.record_keys[(RECORD_KEY_ID, 2)] = bytes(range(1, 33))
    service.current = (RECORD_KEY_ID, 2)
    new = cipher.encrypt("ghu_new", context=CTX_ALICE)
    assert old.startswith(f"oshttp1:{RECORD_KEY_ID}:1:")
    assert new.startswith(f"oshttp1:{RECORD_KEY_ID}:2:")
    assert cipher.decrypt(old, context=CTX_ALICE) == "ghu_old"
    assert cipher.decrypt(new, context=CTX_ALICE) == "ghu_new"
    # The version is bound into the record: claiming another one does not open it.
    relabelled = new.replace(f":{RECORD_KEY_ID}:2:", f":{RECORD_KEY_ID}:1:")
    assert cipher.decrypt(relabelled, context=CTX_ALICE) is None


def test_signing_key_rotation_by_key_id() -> None:
    service = _service()
    service.signing_keys["signing-2"] = b"the-next-signing-key-for-the-fake-service"
    stored = _cipher(service).encrypt("t", context=CTX_ALICE)
    rotated = _cipher(
        service, signing_key=service.signing_keys["signing-2"], signing_key_id="signing-2"
    )
    assert rotated.decrypt(stored, context=CTX_ALICE) == "t"


# ── no plaintext or key in logs ────────────────────────────────────────────


def test_no_plaintext_or_key_in_logs_or_errors(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    plaintext = "ghu_plaintext_canary_0123"
    service = _service()
    errors: list[str] = []
    with service.serve() as url:
        cipher = HttpSecretCipher(url, signing_key=SIGNING_KEY, signing_key_id=SIGNING_KEY_ID)
        stored = cipher.encrypt(plaintext, context=CTX_ALICE)
        assert cipher.decrypt(stored, context=CTX_ALICE) == plaintext
        assert cipher.decrypt(stored, context=CTX_BOB) is None
        assert cipher.decrypt("corrupt", context=CTX_ALICE) is None
        refused = HttpSecretCipher(url, signing_key=SIGNING_KEY, signing_key_id="signing-unknown")
        with pytest.raises(KeyServiceError) as refusal:
            refused.encrypt(plaintext, context=CTX_ALICE)
        errors.append(str(refusal.value))
    with pytest.raises(KeyServiceError) as unreachable:
        cipher.decrypt(stored, context=CTX_ALICE)
    errors.append(str(unreachable.value))
    assert caplog.records, "the failure paths above log warnings"
    for text in [caplog.text, *errors, repr(cipher)]:
        assert plaintext not in text
        assert SIGNING_KEY.decode() not in text


# ── selection from the environment ─────────────────────────────────────────


def _clear_backend_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        CREDENTIAL_CIPHER_ENV_VAR,
        CREDENTIAL_KMS_KEY_ENV_VAR,
        CREDENTIAL_VAULT_KEY_ENV_VAR,
        CREDENTIAL_HTTP_URL_ENV_VAR,
        CREDENTIAL_HTTP_SIGNING_KEY_ENV_VAR,
        CREDENTIAL_HTTP_SIGNING_KEY_ID_ENV_VAR,
    ):
        monkeypatch.delenv(var, raising=False)


def _set_http_env(monkeypatch: pytest.MonkeyPatch, **overrides: str) -> None:
    settings = {
        CREDENTIAL_HTTP_URL_ENV_VAR: "https://keys.example.test/v1/",
        CREDENTIAL_HTTP_SIGNING_KEY_ENV_VAR: SIGNING_KEY.decode(),
        CREDENTIAL_HTTP_SIGNING_KEY_ID_ENV_VAR: SIGNING_KEY_ID,
    }
    for name, value in {**settings, **overrides}.items():
        monkeypatch.setenv(name, value)


def test_selected_by_name(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_backend_env(monkeypatch)
    monkeypatch.setenv(CREDENTIAL_CIPHER_ENV_VAR, "http")
    _set_http_env(monkeypatch)
    cipher = build_secret_cipher()
    assert isinstance(cipher, HttpSecretCipher)
    assert repr(cipher) == (
        "HttpSecretCipher(url='https://keys.example.test/v1', signing_key_id='signing-1')"
    )


def test_selected_but_unconfigured_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_backend_env(monkeypatch)
    monkeypatch.setenv(CREDENTIAL_CIPHER_ENV_VAR, "http")
    with pytest.raises(ValueError, match="not configured"):
        build_secret_cipher()


@pytest.mark.parametrize(
    "missing",
    [
        CREDENTIAL_HTTP_URL_ENV_VAR,
        CREDENTIAL_HTTP_SIGNING_KEY_ENV_VAR,
        CREDENTIAL_HTTP_SIGNING_KEY_ID_ENV_VAR,
    ],
)
@pytest.mark.parametrize("selected", ["http", ""])
def test_half_configured_backend_is_refused(
    monkeypatch: pytest.MonkeyPatch, missing: str, selected: str
) -> None:
    _clear_backend_env(monkeypatch)
    if selected:
        monkeypatch.setenv(CREDENTIAL_CIPHER_ENV_VAR, selected)
    _set_http_env(monkeypatch, **{missing: "  "})
    with pytest.raises(ValueError, match=missing):
        build_secret_cipher()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({CREDENTIAL_HTTP_URL_ENV_VAR: "file:///etc/keys"}, "http\\(s\\)"),
        ({CREDENTIAL_HTTP_URL_ENV_VAR: "https://user:pw@keys.test/v1"}, "credentials"),
        ({CREDENTIAL_HTTP_URL_ENV_VAR: "https://keys.test/v1?token=x"}, "query"),
        ({CREDENTIAL_HTTP_URL_ENV_VAR: "https://keys.test:99999/v1"}, "[Pp]ort"),
        ({CREDENTIAL_HTTP_SIGNING_KEY_ENV_VAR: "too-short"}, "at least 32 bytes"),
        ({CREDENTIAL_HTTP_SIGNING_KEY_ID_ENV_VAR: "has:colon"}, "key ID"),
    ],
)
def test_invalid_settings_are_refused(
    monkeypatch: pytest.MonkeyPatch, overrides: dict[str, str], message: str
) -> None:
    _clear_backend_env(monkeypatch)
    monkeypatch.setenv(CREDENTIAL_CIPHER_ENV_VAR, "http")
    _set_http_env(monkeypatch, **overrides)
    with pytest.raises(ValueError, match=message):
        build_secret_cipher()


def test_autodetected_alone_and_ambiguous_with_another(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_backend_env(monkeypatch)
    _set_http_env(monkeypatch)
    assert isinstance(build_secret_cipher(), HttpSecretCipher)
    monkeypatch.setenv(CREDENTIAL_KMS_KEY_ENV_VAR, "alias/x")
    with pytest.raises(ValueError, match="multiple credential backends"):
        build_secret_cipher()


# ── the published protocol vectors ─────────────────────────────────────────


def _fixture() -> dict[str, Any]:
    return json.loads(_FIXTURE.read_text())


def test_client_requests_match_the_protocol_vectors() -> None:
    fixture = _fixture()
    encrypt, decrypt = fixture["exchanges"]
    sent: list[httpx.Request] = []

    def reply(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        exchange = encrypt if request.url.path.endswith("/encrypt") else decrypt
        return httpx.Response(exchange["response"]["status"], json=exchange["response"]["body"])

    cipher = HttpSecretCipher(
        URL,
        signing_key=fixture["signing_key"]["key_utf8"].encode(),
        signing_key_id=fixture["signing_key"]["key_id"],
        client=httpx.Client(transport=httpx.MockTransport(reply)),
        clock=lambda: fixture["timestamp"],
    )
    plaintext = fixture["seal"]["plaintext"]
    context = {**fixture["seal"]["context"], "account_id": ""}
    stored = cipher.encrypt(plaintext, context=context)
    assert stored == encrypt["stored_record"]
    assert cipher.decrypt(stored, context=context) == plaintext
    for request, exchange in zip(sent, (encrypt, decrypt), strict=True):
        assert request.method == "POST"
        assert request.url.path == "/v1" + exchange["request"]["path"]
        assert request.content == exchange["request"]["body"].encode()
        for name, value in exchange["request"]["headers"].items():
            assert request.headers[name] == value


def test_reference_endpoint_matches_the_protocol_vectors() -> None:
    fixture = _fixture()
    signing, record = fixture["signing_key"], fixture["record_key"]
    for case in fixture["endpoint_cases"]:
        service = FakeKeyService(
            signing_keys={signing["key_id"]: signing["key_utf8"].encode()},
            record_keys={
                (record["key_id"], record["key_version"]): b64url_decode(record["key_base64url"])
            },
            current=(record["key_id"], record["key_version"]),
            clock=lambda now=case["now"]: now,
        )
        status, body = service.respond(case["operation"], case["headers"], case["body"].encode())
        expected = case["expect"]
        assert status == expected["status"], case["name"]
        if "code" in expected:
            assert body["error"]["code"] == expected["code"], case["name"]
        else:
            assert body == expected["body"], case["name"]
