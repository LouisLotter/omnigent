"""Save images and audio that registry MCP tools return as files the agent can use.

The gateway marks a result whose output carries media blocks in the tagged envelope.
Each block is written under the session's workspace, in ``.omnigent/mcp-files/<session>/``
(ignored by git), or under the data directory when the session has no workspace, and a
text block names the file. Images a harness can show stay in the result; other images
and audio are replaced by their note.
"""

from __future__ import annotations

import base64
import binascii
import contextlib
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path

from omnigent.process_logging import data_dir
from omnigent.runtime.mcp_tool_result import (
    decode_mcp_media_result,
    encode_mcp_image_result,
    native_image_payload,
)
from omnigent.util.json_types import JsonObject

#: Largest single media file saved, and the most one result may save.
MAX_MEDIA_BYTES = 50 * 1024 * 1024
MAX_RESULT_BYTES = 100 * 1024 * 1024
_UNSAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")
#: Extensions by MIME type; anything else is saved as ``.bin``.
_EXTENSIONS = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/gif": ".gif",
    "image/webp": ".webp",
    "image/bmp": ".bmp",
    "image/tiff": ".tiff",
    "image/avif": ".avif",
    "image/heic": ".heic",
    "image/svg+xml": ".svg",
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/ogg": ".ogg",
    "audio/opus": ".opus",
    "audio/flac": ".flac",
    "audio/aac": ".aac",
    "audio/mp4": ".m4a",
    "audio/webm": ".webm",
    "audio/pcm": ".pcm",
}


def media_dir(session_id: str, workspace: Path | None) -> Path:
    """Where a session's saved media go, e.g. ``<workspace>/.omnigent/mcp-files/conv_abc``."""
    name = _UNSAFE_NAME.sub("_", session_id)
    if name in ("", ".", ".."):
        name = "_"
    root = workspace / ".omnigent" / "mcp-files" if workspace is not None else None
    return (root or data_dir() / "mcp-files") / name


def save_media_result(
    output: str,
    *,
    session_id: str,
    service: str,
    tool: str,
    workspace: Path | None,
) -> str:
    """Return a gateway media result with its media saved as files and named in text.

    :param output: The registry tool's output: the tagged envelope the gateway marked.
    :param session_id: The session the call belongs to.
    :param service: The registry service ID, e.g. ``"openrouter"``.
    :param tool: The upstream tool name, e.g. ``"generate-image"``.
    :param workspace: The session's workspace, or ``None`` to use the data directory.
    :returns: The output unchanged when it is not an envelope with media; otherwise an
        envelope keeping the images a harness can show, or plain text (``{"error": …}``
        for an error result) when none remain.
    """
    try:
        parsed = json.loads(output)
    except ValueError:
        return output
    result = decode_mcp_media_result(parsed)
    if result is None:
        return output
    directory = media_dir(session_id, workspace)
    stem = f"{_UNSAFE_NAME.sub('_', service)}-{_UNSAFE_NAME.sub('_', tool)}-"
    stem += datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    budget = MAX_RESULT_BYTES
    blocks: list[JsonObject] = []
    for index, block in enumerate(result.content):
        if block["type"] == "text":
            blocks.append(block)
            continue
        note, size = _save_block(block, directory=directory, name=f"{stem}-{index}", budget=budget)
        budget -= size
        blocks.append({"type": "text", "text": note})
        if block["type"] == "image":
            canonical = native_image_payload(str(block["data"]), str(block["mimeType"]))
            if canonical is not None:
                blocks.append({**block, "data": canonical})
    if any(block["type"] == "image" for block in blocks):
        return encode_mcp_image_result(blocks, is_error=result.is_error)
    text = "\n".join(str(block["text"]) for block in blocks)
    return json.dumps({"error": text}) if result.is_error else text


def _save_block(block: JsonObject, *, directory: Path, name: str, budget: int) -> tuple[str, int]:
    """Write one media block; return the note for the agent and the bytes written."""
    kind = str(block["type"])
    mime_type = str(block["mimeType"])
    try:
        data = base64.b64decode("".join(str(block["data"]).split()), validate=True)
    except (binascii.Error, ValueError):
        return f"The {kind} ({mime_type}) could not be saved: its data is not valid base64.", 0
    if len(data) > min(MAX_MEDIA_BYTES, budget):
        return f"The {kind} ({mime_type}, {len(data)} bytes) was not saved: it is too large.", 0
    path = directory / f"{name}{_EXTENSIONS.get(mime_type.lower(), '.bin')}"
    try:
        _ensure_directory(directory)
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except OSError as exc:
        return f"The {kind} ({mime_type}) could not be saved: {exc.strerror or exc}.", 0
    try:
        with os.fdopen(descriptor, "wb") as file:
            file.write(data)
    except OSError as exc:
        with contextlib.suppress(OSError):
            path.unlink()
        return f"The {kind} ({mime_type}) could not be saved: {exc.strerror or exc}.", 0
    return f"Saved the {kind} to {path} ({mime_type}, {len(data)} bytes).", len(data)


def _ensure_directory(directory: Path) -> None:
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    # Inside a workspace the folder is ignored by git, the ignore file included.
    ignore = directory.parent / ".gitignore"
    if directory.parent.name == "mcp-files" and not ignore.exists():
        with contextlib.suppress(OSError):
            ignore.write_text("*\n")
