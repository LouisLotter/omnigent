"""E2E: a key provider's native Codex turns read the key from a referenced file.

An Omnigent key provider's API key must never be written into the session's
private ``config.toml`` or its ``.omnigent-config-profile.toml`` journal: both
are saved with the Codex home. The key lives in a ``0600`` ``provider_key``
file beside the home instead, which the generated provider's ``auth`` command
reads, and which is removed when the app-server closes.

The journey drives the real ``codex`` app-server through two launches of one
session (the second resumes the first's thread, and is the one whose profile
journal snapshots the provider table), runs a turn on each against a loopback
provider that accepts only the configured key, then scans the whole bridge
directory, rollouts and Codex's databases included, for the key in every
encoded form a save could carry.

Self-contained: the provider is a loopback fake; no credentials, no network.
"""

from __future__ import annotations

import base64
import json
import socket
import stat
import threading
import urllib.parse
from pathlib import Path
from typing import Any

import pytest
import yaml

from omnigent.harnesses.codex_native.app_server import (
    CodexAppServerClient,
    build_codex_native_server,
    codex_session_meta_model_provider,
    preload_codex_thread_for_resume,
    resolve_native_codex_launch,
)
from tests.e2e._codex_loopback_resume import wait_for_turn_end
from tests.e2e._harness_probes import cli_unavailable_reason
from tests.e2e.test_codex_gateway_stale_bearer_e2e import FRESH_TOKEN, _FakeGateway

_PROVIDER_KEY = FRESH_TOKEN


def _encoded_forms(value: str) -> dict[str, bytes]:
    """The byte forms a saved file could hold *value* in."""
    raw = value.encode()
    forms = {
        "raw": raw,
        "json-escaped": json.dumps(value)[1:-1].encode(),
        "url-encoded": urllib.parse.quote(value, safe="").encode(),
        "hex": raw.hex().encode(),
    }
    for shift in range(3):
        # The value after ``shift`` other bytes: keep only the base64 characters
        # its neighbours cannot change.
        encoded = base64.b64encode(b"\0" * shift + raw)
        start = (shift * 4 + 2) // 3 + (1 if shift else 0)
        stable = encoded[start : len(encoded) - 4]
        forms[f"base64+{shift}"] = stable
        forms[f"base64url+{shift}"] = stable.replace(b"+", b"-").replace(b"/", b"_")
    return forms


def _key_hits(root: Path, value: str) -> list[tuple[str, str]]:
    """Return ``(file, form)`` for every regular file under *root* holding *value*."""
    forms = _encoded_forms(value)
    hits: list[tuple[str, str]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not path.is_file():
            continue
        data = path.read_bytes()
        hits.extend(
            (str(path.relative_to(root)), form) for form, needle in forms.items() if needle in data
        )
    return hits


def _loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def test_key_scan_finds_independently_encoded_keys(tmp_path: Path) -> None:
    """The scan used below finds a key however a saved file encodes it."""
    value = "sk-t/e+s=t'\"key:%1"
    raw = value.encode()
    planted = {
        "raw": b"token=" + raw,
        "json": json.dumps({"auth": {"token": value}}).encode(),
        "url": f"https://x.invalid/?key={urllib.parse.quote(value, safe='')}".encode(),
        "hex": b"0x" + raw.hex().encode(),
        # Different prefix lengths put the key at every base64 alignment.
        **{f"base64-{n}": base64.b64encode(b"Bearer "[:n] + raw) for n in range(4)},
        **{f"base64url-{n}": base64.urlsafe_b64encode(b"user:"[:n] + raw) for n in range(3)},
    }
    for name, payload in planted.items():
        (tmp_path / name).write_bytes(payload)
    (tmp_path / "unrelated").write_bytes(base64.b64encode(b"Bearer sk-another-key-entirely"))

    assert {path for path, _ in _key_hits(tmp_path, value)} == set(planted)


@pytest.mark.posix_only
@pytest.mark.timeout(180)
async def test_key_provider_turns_reference_the_key_and_save_none_of_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two launches complete turns with the referenced key; no saved file holds it."""
    reason = cli_unavailable_reason("codex")
    if reason is not None:
        pytest.skip(f"requires a runnable 'codex' CLI; {reason}")

    home = tmp_path / "home"
    source_home = tmp_path / "source-codex-home"
    workspace = tmp_path / "workspace"
    bridge_dir = tmp_path / "bridge"
    for directory in (home, source_home, workspace):
        directory.mkdir()
    (source_home / "config.toml").write_text('model = "mock-model"\n', encoding="utf-8")
    (source_home / "work.config.toml").write_text(
        'model_reasoning_effort = "low"\n', encoding="utf-8"
    )
    for name in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("CODEX_HOME", str(source_home))
    monkeypatch.setenv("OMNIGENT_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("OMNIGENT_DISABLE_KEYRING", "1")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    monkeypatch.setattr("omnigent.onboarding.ambient._ollama_reachable", lambda: False)

    gateway = _FakeGateway()
    gateway_thread = threading.Thread(target=gateway.serve_forever, daemon=True)
    gateway_thread.start()
    try:
        (tmp_path / "config.yaml").write_text(
            yaml.safe_dump(
                {
                    "providers": {
                        "loopback": {
                            "kind": "key",
                            "default": True,
                            "openai": {
                                "base_url": f"{gateway.host}/v1",
                                "api_key": _PROVIDER_KEY,
                                "models": {"default": "mock-model"},
                            },
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        launch = resolve_native_codex_launch(model=None)
        assert _PROVIDER_KEY not in launch.summary
        key_file = bridge_dir / "provider_key"

        thread_id: str | None = None
        for launch_number in range(2):
            server = build_codex_native_server(
                socket_path=bridge_dir / "app-server.sock",
                codex_home=bridge_dir / "codex-home",
                cwd=workspace,
                model=launch.model,
                profile=launch.profile,
                bridge_dir=bridge_dir,
                session_id="conv_referenced_provider_key",
                extra_config_overrides=[
                    *launch.config_overrides,
                    "analytics.enabled=false",
                    "feedback.enabled=false",
                    "features.plugins=false",
                    'otel.metrics_exporter="none"',
                    "check_for_update_on_startup=false",
                ],
                reconcile_process_registry=False,
                terminal_launch_args=("--profile", "work"),
            )
            server.listen_url = f"ws://127.0.0.1:{_loopback_port()}"
            client: CodexAppServerClient | None = None
            try:
                await server.start()
                assert key_file.read_text(encoding="utf-8") == _PROVIDER_KEY
                assert stat.S_IMODE(key_file.stat().st_mode) == 0o600
                # The scan sees the key exactly where it belongs while Codex runs.
                assert {path for path, _ in _key_hits(bridge_dir, _PROVIDER_KEY)} == {
                    "provider_key"
                }
                if thread_id is None:
                    client = CodexAppServerClient(
                        ws_url=server.listen_url, client_name="omnigent-referenced-key-e2e"
                    )
                    await client.connect()
                    started = await client.request(
                        "thread/start", {"model": "mock-model", "cwd": str(workspace)}
                    )
                    thread_id = started["result"]["thread"]["id"]
                else:
                    # The relaunch resumes the first launch's thread, as a session restart does.
                    client = await preload_codex_thread_for_resume(
                        server.listen_url,
                        thread_id,
                        retain_client=True,
                        cwd=workspace,
                        model_provider=codex_session_meta_model_provider(launch),
                    )
                assert client is not None
                await client.request(
                    "turn/start",
                    {
                        "threadId": thread_id,
                        "input": [{"type": "text", "text": f"hello {launch_number}"}],
                    },
                )
                terminal_event: dict[str, Any] = await wait_for_turn_end(client)
            finally:
                if client is not None:
                    await client.close()
                await server.close()
            assert terminal_event["method"] == "turn/completed", (
                terminal_event,
                server.recent_stderr,
            )
            assert terminal_event["params"]["turn"]["status"] == "completed", terminal_event
            assert not key_file.exists()

        assert gateway.auth_headers_seen
        assert set(gateway.auth_headers_seen) == {f"Bearer {_PROVIDER_KEY}"}
        codex_home = bridge_dir / "codex-home"
        assert "omnigent_provider" in (codex_home / ".omnigent-config-profile.toml").read_text(
            encoding="utf-8"
        )
        assert list(codex_home.glob("sessions/**/rollout-*.jsonl"))
        assert _key_hits(bridge_dir, _PROVIDER_KEY) == []
    finally:
        gateway.shutdown()
        gateway.server_close()
        gateway_thread.join(timeout=5)
