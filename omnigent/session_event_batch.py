"""Wire contract for byte-bounded session event batches."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

# Limit the encoded JSON request body itself, not just the sum of event text.
# The count bound also keeps validation and dispatch work predictable.
MAX_SESSION_EVENT_REQUEST_BYTES = 10 * 1024 * 1024
MAX_SESSION_EVENT_BATCH_EVENTS = 100
# Names a session status edge on every retry of its POST, so the server
# applies the edge once.
IDEMPOTENCY_KEY_HEADER = "Idempotency-Key"


def encode_session_event_batch(events: Sequence[dict[str, Any]]) -> bytes:
    """Encode an event array exactly as it will be sent over HTTP."""
    return json.dumps(
        list(events),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
