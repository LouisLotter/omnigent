"""Registry MCP media reach the agent as files: the gateway output, the envelope, the runner."""

from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path

import httpx
import pytest
from mcp.types import AudioContent, CallToolResult, ImageContent, TextContent
from PIL import Image

from omnigent.runner import mcp_media_files
from omnigent.runner.mcp_media_files import media_dir, save_media_result
from omnigent.runner.proxy_mcp_manager import ProxyMcpManager
from omnigent.runtime.mcp_tool_result import (
    MEDIA_META_KEY,
    decode_mcp_image_result,
    decode_mcp_media_result,
    encode_mcp_image_result,
)
from omnigent.server.registry_gateway import registry_tool_output
from omnigent.spec.types import AgentSpec, MCPServerConfig
from tests._image_fixtures import _TINY_PNG_BASE64

_AUDIO = base64.b64encode(b"ID3 fake mp3 bytes").decode()


def _bmp_base64() -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (2, 2), "orange").save(buffer, format="BMP")
    return base64.b64encode(buffer.getvalue()).decode()


@pytest.fixture(autouse=True)
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("OMNIGENT_DATA_DIR", str(tmp_path / "data"))
    return tmp_path / "data"


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    path = tmp_path / "workspace"
    path.mkdir()
    return path


def _image_and_audio(*, is_error: bool = False) -> CallToolResult:
    return CallToolResult(
        content=[
            TextContent(type="text", text="made"),
            ImageContent(type="image", data=_TINY_PNG_BASE64, mimeType="image/png"),
            AudioContent(type="audio", data=_AUDIO, mimeType="audio/mpeg"),
        ],
        isError=is_error,
    )


def _save(output: str, workspace: Path | None, session: str = "conv_1") -> str:
    return save_media_result(
        output, session_id=session, service="openrouter", tool="generate", workspace=workspace
    )


def _bmp_result() -> CallToolResult:
    return CallToolResult(
        content=[ImageContent(type="image", data=_bmp_base64(), mimeType="image/bmp")]
    )


def test_gateway_output_keeps_text_as_before_and_marks_media() -> None:
    text_only = CallToolResult(
        content=[TextContent(type="text", text="a"), TextContent(type="text", text="b")]
    )
    assert registry_tool_output(text_only) == ("a\nb", False)

    output, has_media = registry_tool_output(_image_and_audio())
    media = decode_mcp_media_result(json.loads(output))
    assert has_media and media is not None and not media.is_error
    assert [block["type"] for block in media.content] == ["text", "image", "audio"]
    # Harness decoders accept text and image only: audio must be gone before a harness sees it.
    assert decode_mcp_image_result(json.loads(output)) is None

    # An image a harness cannot show still reaches the runner, to be saved as a file.
    bmp, bmp_media = registry_tool_output(_bmp_result())
    assert bmp_media and decode_mcp_media_result(json.loads(bmp)) is not None


def test_the_media_envelope_decoder_rejects_what_it_does_not_know() -> None:
    def envelope(block: dict[str, str]) -> object:
        return json.loads(encode_mcp_image_result([block], is_error=False))

    rejected = [
        {"type": "text", "text": "only text"},
        {"type": "audio", "data": "", "mimeType": "audio/x"},
        {"type": "audio", "data": _AUDIO, "mimeType": "image/png"},
        {"type": "video", "data": _AUDIO, "mimeType": "video/mp4"},
    ]
    for block in rejected:
        assert decode_mcp_media_result(envelope(block)) is None, block
    assert decode_mcp_media_result({"content": [], "isError": False}) is None


def test_media_are_saved_in_the_workspace_ignored_by_git(workspace: Path) -> None:
    saved = _save(registry_tool_output(_image_and_audio())[0], workspace)
    result = decode_mcp_image_result(json.loads(saved))
    assert result is not None, "images stay native for vision-capable harnesses"
    assert [block["type"] for block in result.content] == ["text", "text", "image", "text"]
    texts = [block["text"] for block in result.content if block["type"] == "text"]
    directory = workspace / ".omnigent" / "mcp-files" / "conv_1"
    files = sorted(directory.iterdir())
    assert [f.suffix for f in files] == [".png", ".mp3"]
    assert files[0].read_bytes() == base64.b64decode(_TINY_PNG_BASE64)
    assert files[1].read_bytes() == b"ID3 fake mp3 bytes"
    assert all(oct(f.stat().st_mode & 0o777) == "0o600" for f in files)
    assert oct(directory.stat().st_mode & 0o777) == "0o700"
    assert (directory.parent / ".gitignore").read_text() == "*\n"
    size = files[0].stat().st_size
    assert texts == [
        "made",
        f"Saved the image to {files[0]} (image/png, {size} bytes).",
        f"Saved the audio to {files[1]} (audio/mpeg, 18 bytes).",
    ]


def test_without_a_workspace_the_data_directory_is_used(data_dir: Path) -> None:
    _save(registry_tool_output(_image_and_audio())[0], None, session="..")
    assert media_dir("..", None) == data_dir / "mcp-files" / "_"
    assert len(list((data_dir / "mcp-files" / "_").iterdir())) == 2


def test_an_image_no_harness_shows_is_saved_but_not_shown(workspace: Path) -> None:
    saved = _save(registry_tool_output(_bmp_result())[0], workspace)
    assert saved.startswith("Saved the image to ") and saved.endswith(" (image/bmp, 70 bytes).")


def test_audio_alone_becomes_text_and_an_error_stays_an_error(workspace: Path) -> None:
    only_audio = CallToolResult(
        content=[AudioContent(type="audio", data=_AUDIO, mimeType="audio/mpeg")], isError=True
    )
    error = json.loads(_save(registry_tool_output(only_audio)[0], workspace))["error"]
    assert error.startswith("Saved the audio to ") and error.endswith("(audio/mpeg, 18 bytes).")
    with_image = decode_mcp_image_result(
        json.loads(_save(registry_tool_output(_image_and_audio(is_error=True))[0], workspace))
    )
    assert with_image is not None and with_image.is_error


def _failing_fdopen(descriptor: int, mode: str) -> io.BufferedWriter:
    os.close(descriptor)
    raise OSError(28, "No space left on device")


def test_what_cannot_be_saved_is_said_and_the_rest_is_kept(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert _save("plain text", workspace) is None
    assert _save('{"a": 1}', workspace) is None
    not_base64 = encode_mcp_image_result(
        [{"type": "audio", "data": "!!!", "mimeType": "audio/mpeg"}], is_error=False
    )
    assert "not valid base64" in _save(not_base64, workspace)
    # Per block, and per result: the first file uses the result's budget.
    monkeypatch.setattr(mcp_media_files, "MAX_MEDIA_BYTES", 17)
    assert "too large" in _save(registry_tool_output(_image_and_audio())[0], workspace)
    monkeypatch.setattr(mcp_media_files, "MAX_MEDIA_BYTES", 10_000)
    png_bytes = len(base64.b64decode(_TINY_PNG_BASE64))
    monkeypatch.setattr(mcp_media_files, "MAX_RESULT_BYTES", png_bytes + 1)
    budgeted = _save(registry_tool_output(_image_and_audio())[0], workspace, session="conv_b")
    assert "Saved the image" in budgeted and "too large" in budgeted
    # A write that fails says so, leaves no partial file and keeps the rest of the result.
    monkeypatch.setattr(mcp_media_files.os, "fdopen", _failing_fdopen)
    failed = decode_mcp_image_result(
        json.loads(_save(registry_tool_output(_image_and_audio())[0], workspace, session="conv_f"))
    )
    assert failed is not None
    notes = [block["text"] for block in failed.content if block["type"] == "text"]
    assert notes[0] == "made" and all("No space left" in note for note in notes[1:])
    assert not list((workspace / ".omnigent" / "mcp-files" / "conv_f").iterdir())


@pytest.mark.asyncio
async def test_the_runner_saves_only_what_the_gateway_marked(workspace: Path) -> None:
    envelope = registry_tool_output(_image_and_audio(is_error=True))[0]

    def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        result: dict[str, object] = {"content": [{"type": "text", "text": envelope}]}
        if body["params"]["name"] == "generate-image":
            result |= {"isError": True, "_meta": {MEDIA_META_KEY: True}}
        if body["params"]["name"] == "redacted":
            # A result policy replaced the media error's output; the marker must not hide it.
            result = {
                "content": [{"type": "text", "text": "[Result suppressed by policy: no]"}],
                "isError": True,
                "_meta": {MEDIA_META_KEY: True},
            }
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": result})

    spec = AgentSpec(
        spec_version=1,
        name="media",
        mcp_servers=[MCPServerConfig(name="openrouter", transport="registry")],
    )

    async def resolve() -> Path:
        return workspace

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handle), base_url="http://test"
    ) as client:
        manager = ProxyMcpManager("conv_3", client, media_workspace=resolve)
        marked = await manager.call_tool(spec, "openrouter__generate-image", {})
        # A service's text that merely looks like the envelope writes nothing.
        unmarked = await manager.call_tool(spec, "openrouter__echo", {})
        redacted = await manager.call_tool(spec, "openrouter__redacted", {})
        direct = await ProxyMcpManager("conv_4", client).call_tool(spec, "other__tool", {})
    result = decode_mcp_image_result(json.loads(marked))
    assert result is not None and result.is_error
    assert len(list((workspace / ".omnigent" / "mcp-files" / "conv_3").iterdir())) == 2
    assert unmarked == envelope and direct == envelope
    assert json.loads(redacted) == {"error": "[Result suppressed by policy: no]"}
    assert not (workspace / ".omnigent" / "mcp-files" / "conv_4").exists()
