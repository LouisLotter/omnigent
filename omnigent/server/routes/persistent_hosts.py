"""Create, start and delete persistent managed hosts ("computers") without a session."""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from omnigent.server import persistent_hosts
from omnigent.server.auth import RESERVED_USER_LOCAL, AuthProvider
from omnigent.server.host_registry import HostRegistry
from omnigent.server.routes._auth_helpers import require_user
from omnigent.server.routes._host_launch import resolve_host_owner
from omnigent.server.routes.hosts import host_payload
from omnigent.stores.host_store import Host, HostStore


class PersistentHostCreate(BaseModel):
    """Body of ``POST /v1/hosts``.

    :param name: Display name shown in host pickers, e.g. ``"Work computer"``.
    :param provider: A configured persistent provider, or ``None`` for the first.
    """

    name: str = Field(min_length=1, max_length=64)
    provider: str | None = None

    @field_validator("name")
    @classmethod
    def _clean_name(cls, value: str) -> str:
        name = value.strip()
        if not name or any(not char.isprintable() for char in name):
            raise ValueError("name must be printable text")
        return name


def create_persistent_hosts_router(
    host_store: HostStore,
    host_registry: HostRegistry,
    *,
    auth_provider: AuthProvider | None = None,
) -> APIRouter:
    """Build the router that creates, starts and deletes persistent managed hosts."""
    router = APIRouter()

    async def _owned_persistent_host(request: Request, host_id: str) -> Host:
        user_id = require_user(request, auth_provider)
        host = await asyncio.to_thread(
            resolve_host_owner, user_id=user_id, host_id=host_id, host_store=host_store
        )
        if not persistent_hosts.is_persistent_host(host):
            raise HTTPException(status_code=400, detail="not a persistent host")
        return host

    @router.post("/hosts", status_code=201)
    async def create_persistent_host(
        request: Request, body: PersistentHostCreate
    ) -> dict[str, Any]:
        """Create a persistent host for the caller; its first start runs in the background.

        :returns: The new host, ``waking`` while it starts.
        :raises OmnigentError: 400 when no such persistent provider is
            configured; 409 when the caller already has a host with the name.
        """
        user_id = require_user(request, auth_provider)
        host = await persistent_hosts.create_host(
            owner=user_id if user_id is not None else RESERVED_USER_LOCAL,
            name=body.name,
            provider=body.provider,
        )
        return host_payload(host, host_registry)

    @router.post("/hosts/{host_id}/start", status_code=202)
    async def start_persistent_host(request: Request, host_id: str) -> dict[str, Any]:
        """Start a persistent host again, e.g. after its first start failed.

        :returns: The host, ``waking`` while it starts.
        """
        host = await _owned_persistent_host(request, host_id)
        persistent_hosts.start_host_now(host.host_id)
        return host_payload(host, host_registry)

    @router.delete("/hosts/{host_id}")
    async def delete_persistent_host(request: Request, host_id: str) -> dict[str, Any]:
        """Delete a persistent host and ask its provider to remove the sandbox.

        The host is removed from Omnigent first; a provider removal that fails
        is retried by the managed-sandbox reaper.
        """
        host = await _owned_persistent_host(request, host_id)
        await persistent_hosts.delete_host(host)
        return {"host_id": host.host_id, "deleted": True}

    return router
