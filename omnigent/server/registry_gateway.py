"""Catalog execution behind the existing session MCP authorization and policy path."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import Request
from mcp.types import AudioContent, CallToolResult, ImageContent

from omnigent.entities import Conversation
from omnigent.runtime.mcp_tool_result import encode_mcp_image_result, native_image_payload
from omnigent.server.auth import RESERVED_USER_LOCAL, local_single_user_enabled
from omnigent.server.mcp_gateway import gateway_backend
from omnigent.server.mcp_registry import McpRegistry, McpUpstreamError
from omnigent.server.routes.connections_base import ConnectionError
from omnigent.spec.types import AgentSpec, MCPServerConfig

_logger = logging.getLogger(__name__)


def registry_user(actor: dict[str, str] | None) -> str:
    user_id = (actor or {}).get("run_as")
    if user_id:
        return user_id
    if local_single_user_enabled():
        return RESERVED_USER_LOCAL
    raise ConnectionError("An authenticated user is required for registry MCP services")


def registry_tool_output(result: CallToolResult) -> tuple[str, bool]:
    """The gateway's tool output, and whether it carries media for the runner to save.

    Text-only results are joined as before. With image or audio blocks, the output is
    the tagged envelope: images the harnesses can show are canonicalized, others keep
    their bytes so the runner can still save them as files.
    """
    content: list[dict[str, Any]] = []
    has_media = False
    for block in result.content:
        if isinstance(block, ImageContent | AudioContent) and (
            block.data and block.mimeType.startswith(f"{block.type}/")
        ):
            dumped = block.model_dump(mode="json", exclude_none=True)
            if isinstance(block, ImageContent):
                canonical = native_image_payload(block.data, block.mimeType)
                dumped["data"] = canonical or block.data
            content.append(dumped)
            has_media = True
        else:
            content.append(
                {"type": "text", "text": getattr(block, "text", None) or block.model_dump_json()}
            )
    if has_media:
        return encode_mcp_image_result(content, is_error=bool(result.isError)), True
    return "\n".join(block["text"] for block in content), False


async def registry_tools(
    registry: McpRegistry,
    request: Request,
    spec: AgentSpec,
    user_id: str | None,
) -> list[dict[str, Any]]:
    tools = []
    user = registry_user({"run_as": user_id} if user_id else None)
    await registry.refresh()
    for config in spec.mcp_servers:
        if config.transport != "registry":
            continue
        try:
            service = registry.service(config.name, user)
            definitions = await gateway_backend(registry, request.app.state).list_tools(
                service, user
            )
            for tool in definitions:
                if tool.name in service.tools and (
                    config.tools is None or tool.name in config.tools
                ):
                    tools.append(
                        {
                            **tool.model_dump(by_alias=True, exclude_none=True),
                            "name": f"{config.name}__{tool.name}",
                        }
                    )
        except ConnectionError:
            continue
        except Exception:  # noqa: BLE001 — SDK errors may contain upstream credentials
            _logger.warning("Registry MCP discovery failed for service %s", config.name)
    return tools


async def execute_registry_tool(
    request: Request | None,
    conv: Conversation,
    config: MCPServerConfig,
    name: str,
    arguments: dict[str, Any],
    credential_user: str | None,
) -> dict[str, Any]:
    try:
        registry = getattr(request.app.state, "mcp_registry", None) if request else None
        if registry is None or request is None or conv.archived:
            raise ConnectionError("Registry MCP is unavailable for this session")
        tool = name[len(config.name) + 2 :]
        if config.tools is not None and tool not in config.tools:
            raise ConnectionError("Tool is not enabled for this session")
        user = registry_user({"run_as": credential_user} if credential_user else None)
        await registry.refresh()
        service = registry.service(config.name, user)
        if tool not in service.tools:
            raise ConnectionError("Tool is not allowed by the MCP registry")
        result = await gateway_backend(registry, request.app.state).call_tool(
            service, user, tool, arguments
        )
        output, has_media = registry_tool_output(result)
        _logger.info(
            "Registry MCP call service=%s tool=%s session=%s error=%s",
            config.name,
            tool,
            conv.id,
            result.isError,
        )
        executed: dict[str, Any] = {"result": {"output": output}, "isError": result.isError}
        if has_media:
            executed["media"] = True
        return executed
    except McpUpstreamError as exc:
        _logger.warning(
            "Registry MCP call failed service=%s tool=%s session=%s failure=%s http_status=%s",
            config.name,
            name,
            conv.id,
            exc.kind,
            exc.status_code,
        )
        return {"error": {"code": -32000, "message": str(exc)}}
    except ConnectionError as exc:
        return {"error": {"code": -32000, "message": str(exc)}}
    except Exception:  # noqa: BLE001 — do not expose upstream exception details
        _logger.warning("Registry MCP call failed service=%s session=%s", config.name, conv.id)
        return {
            "error": {
                "code": -32000,
                "message": (
                    "MCP request failed. Check the connection in Settings > MCP; "
                    "the operation was not retried."
                ),
            }
        }
