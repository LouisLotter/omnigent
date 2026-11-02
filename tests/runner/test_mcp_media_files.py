"""Registry MCP media reach the agent as files: the gateway output, the envelope, the runner."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import httpx
import pytest
from mcp.types import AudioContent, CallToolResult, ImageContent, TextContent

from omnigent.runner import mcp_media_files
from omnigent.runner.mcp_media_files import media_dir, save_media_result
from omnigent.runner.proxy_mcp_manager import ProxyMcpManager
from omnigent.runtime.mcp_tool_result import (
    decode_mcp_image_result,
    decode_mcp_media_result,
    encode_mcp_image_result,
)
from omnigent.server.registry_gateway import registry_tool_output
from omnigent.spec.types import AgentSpec, MCPServerConfig
from tests._image_fixtures import _TINY_PNG_BASE64

_AUDIO = base64.b64encode(b"ID3 fake mp3 bytes").decode()


@pytest.fixture(autouse=True)
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("OMNIGENT_DATA_DIR", str(tmp_path))
    return tmp_path


def _image_and_audio(*, is_error: bool = False) -> CallToolResult:
    return CallToolResult(
        content=[
            TextContent(type="text", text="made"),
            ImageContent(type="image", data=_TINY_PNG_BASE64, mimeType="image/png"),
            AudioContent(type="audio", data=_AUDIO, mimeType="audio/mpeg"),
        ],
        isError=is_error,
    )


def test_gateway_output_keeps_text_as_before_and_media_in_the_envelope() -> None:
    text_only = CallToolResult(
        content=[TextContent(type="text", text="a"), TextContent(type="text", text="b")]
    )
    assert registry_tool_output(text_only) == "a\nb"

    output = registry_tool_output(_image_and_audio())
    media = decode_mcp_media_result(json.loads(output))
    assert media is not None and not media.is_error
    assert [block["type"] for block in media.content] == ["text", "image", "audio"]
    assert media.content[1]["data"] == _TINY_PNG_BASE64
    # Harness decoders accept text and image only: audio must be gone before a harness sees it.
    assert decode_mcp_image_result(json.loads(output)) is None

    # An image that is not one stays text, as for directly attached MCP servers.
    bogus = CallToolResult(content=[ImageContent(type="image", data="AAAA", mimeType="image/png")])
    assert decode_mcp_media_result(json.loads(registry_tool_output(bogus))) is None
    assert json.loads(registry_tool_output(bogus))["data"] == "AAAA"


def test_the_media_envelope_decoder_rejects_what_it_does_not_know() -> None:
    def envelope(*blocks: dict[str, str]) -> object:
        return json.loads(encode_mcp_image_result(list(blocks), is_error=False))

    assert decode_mcp_media_result(envelope({"type": "text", "text": "only text"})) is None
    assert (
        decode_mcp_media_result(envelope({"type": "audio", "data": "", "mimeType": "audio/x"}))
        is None
    )
    assert (
        decode_mcp_media_result(
            envelope({"type": "audio", "data": _AUDIO, "mimeType": "image/png"})
        )
        is None
    )
    assert (
        decode_mcp_media_result(
            envelope({"type": "video", "data": _AUDIO, "mimeType": "video/mp4"})
        )
        is None
    )
    assert decode_mcp_media_result({"content": [], "isError": False}) is None


def test_media_are_saved_as_private_files_and_named_for_the_agent(data_dir: Path) -> None:
    saved = save_media_result(
        registry_tool_output(_image_and_audio()),
        session_id="conv_1",
        service="openrouter",
        tool="generate-image",
    )
    result = decode_mcp_image_result(json.loads(saved))
    assert result is not None, "images stay native for vision-capable harnesses"
    texts = [block["text"] for block in result.content if block["type"] == "text"]
    assert texts[0] == "made"
    files = sorted(media_dir("conv_1").iterdir())
    assert [f.suffix for f in files] == [".png", ".mp3"]
    assert media_dir("conv_1").parent == data_dir / "mcp-files"
    assert files[0].read_bytes() == base64.b64decode(_TINY_PNG_BASE64)
    assert files[1].read_bytes() == b"ID3 fake mp3 bytes"
    assert all(oct(f.stat().st_mode & 0o777) == "0o600" for f in files)
    assert oct(media_dir("conv_1").stat().st_mode & 0o777) == "0o700"
    assert f"Saved the image to {files[0]}" in texts[1]
    assert f"Saved the audio to {files[1]}" in texts[2]
    assert [block["type"] for block in result.content] == ["text", "text", "image", "text"]


def test_audio_alone_becomes_text_and_an_error_stays_an_error() -> None:
    only_audio = CallToolResult(
        content=[AudioContent(type="audio", data=_AUDIO, mimeType="audio/mpeg")], isError=True
    )
    saved = save_media_result(
        registry_tool_output(only_audio), session_id="conv_2", service="s", tool="speak"
    )
    error = json.loads(saved)["error"]
    assert error.startswith("Saved the audio to ") and error.endswith("(audio/mpeg, 18 bytes).")


def test_what_cannot_be_saved_is_said_and_plain_output_is_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert save_media_result("plain text", session_id="c", service="s", tool="t") == "plain text"
    assert save_media_result('{"a": 1}', session_id="c", service="s", tool="t") == '{"a": 1}'
    monkeypatch.setattr(mcp_media_files, "MAX_AUDIO_BYTES", 4)
    too_big = encode_mcp_image_result(
        [{"type": "audio", "data": _AUDIO, "mimeType": "audio/mpeg"}], is_error=False
    )
    assert "too large" in save_media_result(too_big, session_id="c", service="s", tool="t")
    not_base64 = encode_mcp_image_result(
        [{"type": "audio", "data": "!!!", "mimeType": "audio/mpeg"}], is_error=False
    )
    assert "not valid base64" in save_media_result(
        not_base64, session_id="c", service="s", tool="t"
    )
    assert not media_dir("c").exists()


@pytest.mark.asyncio
async def test_the_runner_saves_registry_media_but_not_direct_tool_results() -> None:
    envelope = registry_tool_output(_image_and_audio())

    def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        result = {"content": [{"type": "text", "text": envelope}]}
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": result})

    spec = AgentSpec(
        spec_version=1,
        name="media",
        mcp_servers=[MCPServerConfig(name="openrouter", transport="registry")],
    )
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handle), base_url="http://test"
    ) as client:
        managed = await ProxyMcpManager("conv_3", client).call_tool(
            spec, "openrouter__generate-image", {}
        )
        direct = await ProxyMcpManager("conv_4", client).call_tool(spec, "other__tool", {})
    assert "Saved the image to" in managed
    assert decode_mcp_image_result(json.loads(managed)) is not None
    assert len(list(media_dir("conv_3").iterdir())) == 2
    assert direct == envelope and not media_dir("conv_4").exists()
