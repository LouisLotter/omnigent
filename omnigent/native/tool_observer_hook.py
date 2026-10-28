"""Shared, observational post-tool hook for native harnesses."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
import urllib.request
from pathlib import Path

# The hook bounds its own relay wait (``_RELAY_TIMEOUT_S`` per socket step), so
# the harness timeout only has to cover interpreter start-up, which takes
# seconds on a CPU-starved machine; a timeout there shows as a failed hook.
HOOK_TIMEOUT_S = 10
_RELAY_TIMEOUT_S = 1


def hook_settings(bridge_dir: Path, python: str, module: str) -> dict[str, object]:
    """Use each harness's existing owned hook module and trust registration."""
    return {
        "type": "command",
        "command": shlex.join(
            [python, "-I", "-m", module, "observe-tool", "--bridge-dir", str(bridge_dir)]
        ),
        "timeout": HOOK_TIMEOUT_S,
    }


def main(argv: list[str]) -> int:
    """Send bounded hook input to the local relay, with no model-visible output."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--bridge-dir", required=True)
    args = parser.parse_args(argv)
    try:
        raw = sys.stdin.read(1_048_577).encode("utf-8")
        if len(raw) > 1_048_576:
            return 0
        payload = json.loads(raw)
        if not isinstance(payload, dict) or payload.get("hook_event_name") != "PostToolUse":
            return 0
        relay = json.loads((Path(args.bridge_dir) / "tool_relay.json").read_text())
        request = urllib.request.Request(
            relay["url"].rstrip("/") + "/hook/observe-tool",
            data=raw,
            headers={
                "Authorization": "Bearer " + relay["token"],
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=_RELAY_TIMEOUT_S) as response:
            response.read()
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"omnigent tool observer: {type(exc).__name__}", file=sys.stderr)
    return 0
