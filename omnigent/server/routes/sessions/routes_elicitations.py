"""Elicitation routes: resolve and get elicitations."""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import (
    APIRouter,
    Request,
)

from omnigent.runner.routing import RunnerRouter
from omnigent.runtime import (
    get_approval_store,
    pending_elicitations,
)
from omnigent.runtime.policies.approval import _ELICITATION_MODE
from omnigent.server._elicitation_registry import (
    _harness_elicitation_owners,
    _harness_elicitation_registry,
    _harness_parked_elicitations,
    _harness_pre_resolved_elicitations,
    _ParkedHarnessElicitation,
    _PreResolvedHarnessElicitation,
)
from omnigent.server.approval_authority import (
    approval_view,
    gate_approval_answer,
    may_view_approval,
)
from omnigent.server.auth import (
    LEVEL_EDIT,
    AuthProvider,
)
from omnigent.server.routes._auth_helpers import (
    get_user_id as _get_user_id,
)
from omnigent.server.routes._auth_helpers import (
    refuse_delegated_credential as _refuse_delegated_credential,
)
from omnigent.server.routes._auth_helpers import (
    require_access_and_level as _require_access_and_level,
)
from omnigent.server.routes._errors import session_not_found as _session_not_found
from omnigent.server.routes._sessions.common import (
    _logger,
    get_server_runner_router,
    set_server_runner_router,
)
from omnigent.server.routes._sessions.helpers import (
    _apply_pending_policy_ask_writes,
)
from omnigent.server.routes._sessions.orchestration import (
    _resolve_elicitation,
)
from omnigent.server.schemas import (
    ElicitationResult,
)
from omnigent.stores import AgentStore, ConversationStore
from omnigent.stores.permission_store import PermissionStore

# Level needed to view or answer an elicitation. Elicitations carry
# collaborative prompts (``AskUserQuestion``, plan review, MCP forms) as
# well as policy ASK gates, and the in-band ``approval`` session event
# resolves the same elicitations at the events route's edit bar — so the
# two paths must stay at the same level. Raising it to owner is a product
# decision that needs a "waiting for the owner" UI state first.
ELICITATION_RESOLVE_LEVEL: int = LEVEL_EDIT


def register_elicitations_routes(
    router: APIRouter,
    *,
    conversation_store: ConversationStore,
    agent_store: AgentStore,
    runner_router: RunnerRouter | None = None,
    auth_provider: AuthProvider | None = None,
    permission_store: PermissionStore | None = None,
) -> None:
    """Register the elicitations routes on router."""

    async def _approver_may_view(
        session_id: str, elicitation_id: str, user_id: str | None
    ) -> bool:
        """Whether *user_id* answers this pending protected approval as an approver."""
        store = get_approval_store()
        if store is None or auth_provider is None:
            return False
        record = await asyncio.to_thread(store.get_verdict, elicitation_id)
        if record is None or record.session_id != session_id:
            return False
        return await asyncio.to_thread(
            may_view_approval, store, conversation_store, record, user_id
        )

    async def _approval_record(
        elicitation_id: str, *, session_id: str | None = None
    ) -> dict[str, Any]:
        """``{"approval": ...}`` for a recorded protected approval, else nothing."""
        store = get_approval_store()
        if store is None:
            return {}
        record = await asyncio.to_thread(store.get_verdict, elicitation_id)
        if record is None or (session_id is not None and record.session_id != session_id):
            return {}
        return {"approval": approval_view(record)}

    @router.post(
        "/sessions/{session_id}/elicitations/{elicitation_id}/resolve",
        # Internal elicitation flow — hidden from the public API reference.
        include_in_schema=False,
        status_code=202,
        # response_model=None: the body is a small acknowledgement
        # dict, not a domain model.
        response_model=None,
    )
    async def resolve_elicitation(
        request: Request,
        session_id: str,
        elicitation_id: str,
        body: ElicitationResult,
    ) -> dict[str, Any]:
        """
        Resolve an outstanding elicitation by its URL (URL-based
        elicitation).

        The dedicated, RESTful counterpart to delivering a verdict
        via the ``type == "approval"`` event on
        ``POST /v1/sessions/{id}/events``. An elicitation request
        published in ``mode == "url"`` carries this endpoint's path
        as its ``params.url``; the client hits it directly with the
        MCP :class:`ElicitationResult` body instead of POSTing a
        generic approval event. The verdict routes through the
        shared :func:`_resolve_elicitation`, so resolution semantics
        are identical to the event path.

        The ``elicitation_id`` is taken from the URL rather than the
        body, so the unguessable id (``secrets.token_hex(16)``) is
        the capability scoping the resolution — combined with the
        :data:`ELICITATION_RESOLVE_LEVEL` gate below (edit, not owner:
        any collaborator who can drive the agent may answer it) and
        the server-side ownership check inside
        :func:`_resolve_elicitation`.

        :param request: The inbound request, used for identity
            extraction.
        :param session_id: Session/conversation identifier,
            e.g. ``"conv_abc123"``.
        :param elicitation_id: Correlation id of the elicitation to
            resolve, e.g. ``"elicit_abc123"``. Taken from the URL
            path, not the body.
        :param body: The MCP-shaped verdict — ``action``
            (``"accept"`` / ``"decline"`` / ``"cancel"``) plus
            optional form ``content``.
        A protected approval (a policy ASK recorded in the approval ledger)
        is answered by its approvers instead: the thread's person, or the
        named approvers of the thread, service or team, who need no access to
        the thread. Only a browser sign-in answers one, and it resolves only
        once every required approver accepted (or one declined).

        :returns: ``{"queued": False}`` — resolution is synchronous
            and persists no conversation item. A protected approval still
            waiting for other approvers adds ``"awaiting"``: the kinds of
            approval still missing, comma-separated, e.g. ``"team"``.
        :raises OmnigentError: 404 if no session exists; 403 for a
            delegated credential such as a runner's token.
        """
        user_id = _get_user_id(request, auth_provider)
        gate = await gate_approval_answer(
            request,
            session_id=session_id,
            elicitation_id=elicitation_id,
            action=body.action,
            auth_provider=auth_provider,
            permission_store=permission_store,
            conversation_store=conversation_store,
        )
        if gate.protected:
            conv = await asyncio.to_thread(conversation_store.get_conversation, session_id)
            if conv is None:
                raise _session_not_found()
            if not gate.final:
                return {
                    "queued": False,
                    "awaiting": ",".join(gate.awaiting),
                    **(await _approval_record(elicitation_id)),
                }
            _resolve_data: dict[str, Any] = {
                "elicitation_id": elicitation_id,
                "action": gate.action,
            }
        else:
            access = await _require_access_and_level(
                user_id,
                session_id,
                ELICITATION_RESOLVE_LEVEL,
                permission_store,
                conversation_store,
            )
            _refuse_delegated_credential(request, auth_provider, "answer approvals")
            conv = access.conversation
            if conv is None:
                conv = await asyncio.to_thread(conversation_store.get_conversation, session_id)
                if conv is None:
                    raise _session_not_found()
            _resolve_data = {
                "elicitation_id": elicitation_id,
                **body.model_dump(exclude_none=True),
            }
        await _resolve_elicitation(
            session_id, _resolve_data, runner_router, conversation_store, decided_by=user_id
        )
        # Apply any policy writes deferred by the relay tool-call ASK gate
        # (e.g. a cost-budget checkpoint) now that the verdict is in.
        await _apply_pending_policy_ask_writes(
            session_id, conv, conversation_store, agent_store, _resolve_data
        )
        return {
            "queued": False,
            **(await _approval_record(elicitation_id) if gate.protected else {}),
        }

    @router.get(
        "/sessions/{session_id}/elicitations/{elicitation_id}",
        # Internal elicitation flow — hidden from the public API reference.
        include_in_schema=False,
        response_model=None,
    )
    async def get_elicitation(
        request: Request,
        session_id: str,
        elicitation_id: str,
    ) -> dict[str, Any]:
        """
        Return the state of a pending elicitation as JSON.

        Used by the frontend's standalone approval page
        (``/approve/:sessionId/:elicitationId``) to fetch the
        elicitation prompt and render approve/reject controls.
        The payload is read from the in-memory
        :mod:`omnigent.runtime.pending_elicitations` index — no
        database persistence required.

        :param request: The inbound request, used for identity
            extraction.
        :param session_id: Session/conversation identifier,
            e.g. ``"conv_abc123"``.
        :param elicitation_id: Correlation id of the elicitation,
            e.g. ``"elicit_abc123"``.
        :returns: JSON with ``status`` (``"pending"`` or
            ``"resolved"``), and when pending: ``message``,
            ``phase``, ``policy_name``, ``content_preview``. Native requests
            may include ``resolution`` (``"unanswered"`` or ``"confirmed"``)
            after clearing; absence alone does not prove native confirmation.
        :raises OmnigentError: 404 if the session does not exist.
        """
        user_id = _get_user_id(request, auth_provider)
        if not await _approver_may_view(session_id, elicitation_id, user_id):
            access = await _require_access_and_level(
                user_id,
                session_id,
                ELICITATION_RESOLVE_LEVEL,
                permission_store,
                conversation_store,
            )
            if access.conversation is None:
                conv = await asyncio.to_thread(conversation_store.get_conversation, session_id)
                if conv is None:
                    raise _session_not_found()

        found = pending_elicitations.lookup(elicitation_id, conversation_id=session_id)
        if found is None:
            resolution = pending_elicitations.native_resolution_for(session_id, elicitation_id)
            return {
                "status": "resolved",
                **({"resolution": resolution} if resolution else {}),
                **(await _approval_record(elicitation_id, session_id=session_id)),
            }

        _conv_id, event = found
        params_value = event.get("params")
        params = params_value if isinstance(params_value, dict) else {}
        if params.get("target_session_id") not in (None, session_id):
            return {"status": "resolved"}
        return {
            **(await _approval_record(elicitation_id, session_id=session_id)),
            "status": "pending",
            "message": params.get("message", "Approval required"),
            "phase": params.get("phase", ""),
            "policy_name": params.get("policy_name", ""),
            "content_preview": params.get("content_preview", ""),
            **{
                key: params[key]
                for key in ("native_delivery", "permission_directories")
                if key in params
            },
        }
