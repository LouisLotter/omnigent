"""Create and remove persistent managed hosts without a session."""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from omnigent.server import persistent_hosts
from omnigent.server.auth import RESERVED_USER_LOCAL, AuthProvider
from omnigent.server.managed_hosts import launch_managed_host, terminate_managed_host
from omnigent.server.routes._auth_helpers import require_user
from omnigent.server.routes._host_launch import resolve_host_owner
from omnigent.stores.host_store import HostStore


class PersistentHostCreate(BaseModel):
    """Body of ``POST /v1/hosts/persistent``.

    :param provider: A configured provider with a persistent lifecycle, or
        ``None`` for the first such provider.
    """

    provider: str | None = None


def create_persistent_hosts_router(
    host_store: HostStore,
    *,
    auth_provider: AuthProvider | None = None,
) -> APIRouter:
    """Build the router that creates and removes persistent managed hosts."""
    router = APIRouter()

    @router.post("/hosts/persistent", status_code=201)
    async def create_persistent_host(
        request: Request, body: PersistentHostCreate
    ) -> dict[str, Any]:
        """Provision a persistent host for the caller and start it."""
        user_id = require_user(request, auth_provider)
        deployment = getattr(request.app.state, "sandbox_config", None)
        configs = [
            config
            for config in (deployment.offered() if deployment is not None else ())
            if config.persistent is not None
            and (body.provider is None or config.provider == body.provider)
        ]
        if deployment is None or not configs:
            raise HTTPException(status_code=400, detail="no persistent host provider configured")
        launch = await launch_managed_host(
            config=deployment,
            owner=user_id if user_id is not None else RESERVED_USER_LOCAL,
            host_store=host_store,
            provider=configs[0].provider,
            persistent=True,
        )
        return {"host_id": launch.host_id, "workspace": launch.workspace}

    @router.delete("/hosts/persistent/{host_id}")
    async def delete_persistent_host(request: Request, host_id: str) -> dict[str, Any]:
        """Delete a persistent host and terminate its sandbox."""
        user_id = require_user(request, auth_provider)
        host = await asyncio.to_thread(
            resolve_host_owner, user_id=user_id, host_id=host_id, host_store=host_store
        )
        if not persistent_hosts.is_persistent_host(host):
            raise HTTPException(status_code=400, detail="not a persistent host")
        await terminate_managed_host(
            host, host_store, getattr(request.app.state, "sandbox_config", None)
        )
        persistent_hosts.forget(host.host_id)
        return {"host_id": host.host_id, "deleted": True}

    return router
