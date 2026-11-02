"""Save images and audio that registry MCP tools return as files the agent can use.

The gateway carries media blocks to the runner in the tagged result envelope. Each
block is written under ``<data dir>/mcp-files/<session id>/`` and a text block names
the file. Images stay in the result so vision-capable harnesses still see them;
audio blocks are replaced by their note, as no harness consumes audio results.
"""

from __future__ import annotations

import base64
import binascii
import json
import mimetypes
import os
import re
from datetime import UTC, datetime
from pathlib import Path

from omnigent.process_logging import data_dir
from omnigent.runtime.mcp_tool_result import decode_mcp_media_result, encode_mcp_image_result
from omnigent.util.json_types import JsonObject

#: Largest audio file saved; larger blocks are dropped with a note.
MAX_AUDIO_BYTES = 50 * 1024 * 1024
_UNSAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")
_EXTENSIONS = {"image/jpeg": ".jpg", "audio/mpeg": ".mp3", "audio/wav": ".wav"}


def media_dir(session_id: str) -> Path:
    """Where a session's saved media live, e.g. ``~/.omnigent/mcp-files/conv_abc``."""
    return data_dir() / "mcp-files" / _UNSAFE_NAME.sub("_", session_id)


def save_media_result(output: str, *, session_id: str, service: str, tool: str) -> str:
    """Return ``output`` with its media saved as files and named in text blocks.

    :param output: A registry tool's output, e.g. the tagged envelope or plain text.
    :param session_id: The session the call belongs to.
    :param service: The registry service ID, e.g. ``"openrouter"``.
    :param tool: The upstream tool name, e.g. ``"generate-image"``.
    :returns: The output unchanged when it carries no media; otherwise an envelope
        with image blocks kept, or plain text when only audio was returned.
    """
    try:
        parsed = json.loads(output)
    except ValueError:
        return output
    result = decode_mcp_media_result(parsed)
    if result is None:
        return output
    stem = f"{_UNSAFE_NAME.sub('_', service)}-{_UNSAFE_NAME.sub('_', tool)}-"
    stem += datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    blocks: list[JsonObject] = []
    for index, block in enumerate(result.content):
        if block["type"] == "text":
            blocks.append(block)
            continue
        note = _save_block(block, session_id=session_id, name=f"{stem}-{index}")
        blocks.append({"type": "text", "text": note})
        if block["type"] == "image":
            blocks.append(block)
    if any(block["type"] == "image" for block in blocks):
        return encode_mcp_image_result(blocks, is_error=result.is_error)
    text = "\n".join(str(block["text"]) for block in blocks)
    return json.dumps({"error": text}) if result.is_error else text


def _save_block(block: JsonObject, *, session_id: str, name: str) -> str:
    kind = str(block["type"])
    mime_type = str(block["mimeType"])
    try:
        data = base64.b64decode("".join(str(block["data"]).split()), validate=True)
    except (binascii.Error, ValueError):
        return f"The {kind} ({mime_type}) could not be saved: its data is not valid base64."
    if kind == "audio" and len(data) > MAX_AUDIO_BYTES:
        return f"The {kind} ({mime_type}, {len(data)} bytes) was not saved: it is too large."
    extension = _EXTENSIONS.get(mime_type) or mimetypes.guess_extension(mime_type) or ".bin"
    directory = media_dir(session_id)
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = directory / f"{name}{extension}"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as file:
        file.write(data)
    return f"Saved the {kind} to {path} ({mime_type}, {len(data)} bytes)."
