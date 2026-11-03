"""Who may call each route: people, runners, or anyone the route admits itself.

Every HTTP and WebSocket route is classified here. :class:`RouteAuthorityMiddleware`
looks up the matched route and refuses a caller whose principal the rule does
not admit. A route missing from the table is refused for everyone, so a new
endpoint cannot become reachable to runners by accident; the route-inventory
test lists any unclassified route.

A runner is a machine principal acting for its thread's owner. It never holds
an administrator's override (see :mod:`omnigent.db.account_authority`) and may
call only routes that name one of its token's scopes. Hosts have no HTTP token
yet; their routes authenticate with a launch token or the person's sign-in.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
from collections.abc import Iterable

from fastapi.routing import APIWebSocketRoute
from starlette.requests import HTTPConnection
from starlette.routing import Match, Mount, WebSocketRoute
from starlette.types import ASGIApp, Receive, Scope, Send

from omnigent.server.auth import AuthProvider
from omnigent.server.principals import (
    SCOPE_AGENTS,
    SCOPE_CATALOG,
    SCOPE_MCP,
    SCOPE_SCHEDULES,
    SCOPE_SESSION,
    SCOPE_SESSIONS_CREATE,
    SCOPE_SESSIONS_LIST,
    SCOPE_TUNNEL,
    Principal,
    PrincipalKind,
    recorded_principal,
)

ROUTE_UNCLASSIFIED_CODE = "route_unclassified"
MACHINE_FORBIDDEN_CODE = "machine_principal_forbidden"
_WS_FORBIDDEN_CLOSE_CODE = 4403


@dataclasses.dataclass(frozen=True)
class RouteRule:
    """Which principals may call a route.

    :param public: The guard admits every caller; the route authenticates on
        its own (a binding or launch token, a password form) or needs no
        identity. A machine principal that reaches it is still never an
        administrator.
    :param humans: People (and their delegated clients) may call it.
    :param runner_scope: The scope a runner token must hold, or ``None`` when
        runners are refused.
    """

    public: bool = False
    humans: bool = True
    runner_scope: str | None = None

    def admits(self, principal: Principal) -> bool:
        """Whether *principal* may call a route under this rule."""
        if self.public:
            return True
        if principal.kind is PrincipalKind.RUNNER:
            return self.runner_scope is not None and principal.allows(self.runner_scope)
        if principal.kind is PrincipalKind.HOST:
            return False
        return self.humans


PUBLIC = RouteRule(public=True)
HUMAN = RouteRule()


def runner(scope: str) -> RouteRule:
    """People, and runners holding *scope*."""
    return RouteRule(runner_scope=scope)


_SESSION = runner(SCOPE_SESSION)
_CATALOG = runner(SCOPE_CATALOG)

# Routes under one thread. Runners act on threads their person owns; the
# permission layer enforces that, and the exceptions below stay with people.
_SESSION_ROUTES: tuple[tuple[str, str, RouteRule], ...] = (
    ("DELETE", "", _SESSION),
    ("GET", "", _SESSION),
    ("PATCH", "", _SESSION),
    ("GET", "/agent", _SESSION),
    ("PUT", "/agent", _SESSION),
    ("GET", "/agent/contents", _SESSION),
    ("GET", "/agent/mcp-servers", _SESSION),
    ("POST", "/agent/mcp-servers", _SESSION),
    ("DELETE", "/agent/mcp-servers/{server_name}", _SESSION),
    ("PUT", "/agent/mcp-servers/{server_name}", _SESSION),
    ("POST", "/agent-title", _SESSION),
    ("POST", "/auto-title", _SESSION),
    ("POST", "/browser/action_claim/{action_id}", _SESSION),
    ("POST", "/browser/action_request", _SESSION),
    ("POST", "/browser/action_result/{action_id}", _SESSION),
    ("GET", "/child_sessions", _SESSION),
    ("DELETE", "/codex_goal", _SESSION),
    ("GET", "/codex_goal", _SESSION),
    ("PUT", "/codex_goal", _SESSION),
    ("PATCH", "/codex_goal/status", _SESSION),
    ("GET", "/comments", _SESSION),
    ("POST", "/comments", _SESSION),
    ("DELETE", "/comments/{comment_id}", _SESSION),
    ("PATCH", "/comments/{comment_id}", _SESSION),
    ("POST", "/comments/send", _SESSION),
    # A thread's approval records: who asked and who answered as what.
    ("GET", "/approvals", HUMAN),
    ("GET", "/approvals/{elicitation_id}", HUMAN),
    ("GET", "/elicitations/{elicitation_id}", _SESSION),
    # Answering an approval is a person's decision.
    ("POST", "/elicitations/{elicitation_id}/resolve", HUMAN),
    ("POST", "/events", _SESSION),
    # The bound runner registers each turn before its harness acts; the route
    # refuses any other caller.
    ("POST", "/turns", _SESSION),
    ("POST", "/hooks/antigravity-elicitation-request", _SESSION),
    ("POST", "/hooks/codex-elicitation-request", _SESSION),
    ("POST", "/hooks/cursor-permission-request", _SESSION),
    ("POST", "/hooks/native-permission-request", _SESSION),
    ("POST", "/hooks/permission-request", _SESSION),
    ("POST", "/hooks/pi-extension-ui", _SESSION),
    ("POST", "/hooks/route-subagent", _SESSION),
    ("POST", "/hooks/route-turn", _SESSION),
    ("GET", "/items", _SESSION),
    ("GET", "/labels", _SESSION),
    ("POST", "/mcp", _SESSION),
    ("POST", "/model-override/reset", _SESSION),
    ("GET", "/owner", _SESSION),
    ("GET", "/permissions", _SESSION),
    ("PUT", "/permissions", _SESSION),
    ("DELETE", "/permissions/{target_user_id}", _SESSION),
    ("GET", "/policies", _SESSION),
    ("POST", "/policies", _SESSION),
    ("GET", "/policies/{policy_id}", _SESSION),
    ("PATCH", "/policies/{policy_id}", _SESSION),
    ("DELETE", "/policies/{policy_id}", _SESSION),
    ("POST", "/policies/evaluate", _SESSION),
    ("PUT", "/read-state", _SESSION),
    ("GET", "/resources", _SESSION),
    ("GET", "/resources/environments", _SESSION),
    ("GET", "/resources/environments/{environment_id}", _SESSION),
    ("GET", "/resources/environments/{environment_id}/changes", _SESSION),
    ("GET", "/resources/environments/{environment_id}/content-search", _SESSION),
    ("GET", "/resources/environments/{environment_id}/diff/{relative_path:path}", _SESSION),
    ("GET", "/resources/environments/{environment_id}/filesystem", _SESSION),
    (
        "DELETE",
        "/resources/environments/{environment_id}/filesystem/{relative_path:path}",
        _SESSION,
    ),
    ("GET", "/resources/environments/{environment_id}/filesystem/{relative_path:path}", _SESSION),
    (
        "PATCH",
        "/resources/environments/{environment_id}/filesystem/{relative_path:path}",
        _SESSION,
    ),
    ("PUT", "/resources/environments/{environment_id}/filesystem/{relative_path:path}", _SESSION),
    ("GET", "/resources/environments/{environment_id}/search", _SESSION),
    ("GET", "/resources/environments/{environment_id}/search/{path:path}", _SESSION),
    ("POST", "/resources/environments/{environment_id}/shell", _SESSION),
    ("GET", "/resources/files", _SESSION),
    ("POST", "/resources/files", _SESSION),
    ("POST", "/resources/files:copy", _SESSION),
    ("DELETE", "/resources/files/{file_id}", _SESSION),
    ("GET", "/resources/files/{file_id}", _SESSION),
    ("GET", "/resources/files/{file_id}/content", _SESSION),
    ("GET", "/resources/github", _SESSION),
    ("POST", "/resources/github/attachments", _SESSION),
    ("GET", "/resources/github/changes", _SESSION),
    ("POST", "/resources/github/comments", _SESSION),
    ("POST", "/resources/github/details", _SESSION),
    ("GET", "/resources/github/diff", _SESSION),
    ("GET", "/resources/github/diff/{relative_path:path}", _SESSION),
    ("POST", "/resources/github/disable-auto-merge", _SESSION),
    ("GET", "/resources/github/labels", _SESSION),
    ("POST", "/resources/github/labels", _SESSION),
    ("POST", "/resources/github/merge", _SESSION),
    ("GET", "/resources/github/merge-options", _SESSION),
    ("POST", "/resources/github/preferences", _SESSION),
    ("POST", "/resources/github/prs", _SESSION),
    ("GET", "/resources/github/reviewers", _SESSION),
    ("POST", "/resources/github/reviewers", _SESSION),
    ("GET", "/resources/{resource_id}", _SESSION),
    ("GET", "/resources/terminals", _SESSION),
    ("POST", "/resources/terminals", _SESSION),
    ("DELETE", "/resources/terminals/{terminal_id}", _SESSION),
    ("GET", "/resources/terminals/{terminal_id}", _SESSION),
    # A person's interactive terminal; runners create terminals above.
    ("WS", "/resources/terminals/{terminal_id}/attach", HUMAN),
    ("POST", "/resources/terminals/{terminal_id}/transfer", _SESSION),
    ("GET", "/sign-in-link", _SESSION),
    ("GET", "/stream", _SESSION),
    ("POST", "/switch-agent", _SESSION),
    ("POST", "/ui/open-file", _SESSION),
)

# Routes under one host. Runners may read their person's hosts; registering,
# starting, deleting or steering a computer stays with people.
_HOST_ROUTES: tuple[tuple[str, str, RouteRule], ...] = (
    ("DELETE", "", HUMAN),
    ("GET", "", _CATALOG),
    ("GET", "/credentials/detected", HUMAN),
    # The host credential broker authenticates with the host's launch token
    # and vends only the host owner's own credential.
    ("GET", "/credentials/{provider}", PUBLIC),
    ("POST", "/directories", HUMAN),
    ("GET", "/filesystem", HUMAN),
    ("GET", "/filesystem/{path:path}", HUMAN),
    ("POST", "/harnesses/{harness}/credential", HUMAN),
    ("POST", "/harnesses/{harness}/install", HUMAN),
    ("GET", "/harnesses/{harness}/model-options", HUMAN),
    ("GET", "/harnesses/{harness}/skills/{name:path}", HUMAN),
    ("GET", "/harnesses/{harness}/startup", HUMAN),
    ("POST", "/harnesses/refresh", HUMAN),
    ("GET", "/mcp-servers", HUMAN),
    ("POST", "/mcp-servers/tools", HUMAN),
    ("GET", "/plugins", HUMAN),
    ("POST", "/runners", HUMAN),
    ("POST", "/start", HUMAN),
    # A host connects with its launch token or its person's sign-in; a runner
    # token can never register a computer.
    ("WS", "/tunnel", HUMAN),
    ("GET", "/worktrees", HUMAN),
)

_OTHER_ROUTES: tuple[tuple[str, str, RouteRule], ...] = (
    # No authority.
    ("GET", "/health", PUBLIC),
    ("GET", "/api/version", PUBLIC),
    ("GET", "/v1/info", PUBLIC),
    ("GET", "/.well-known/omnigent.json", PUBLIC),
    ("GET", "/openapi.json", PUBLIC),
    ("HEAD", "/openapi.json", PUBLIC),
    ("GET", "/docs", PUBLIC),
    ("HEAD", "/docs", PUBLIC),
    ("GET", "/docs/oauth2-redirect", PUBLIC),
    ("HEAD", "/docs/oauth2-redirect", PUBLIC),
    ("GET", "/redoc", PUBLIC),
    ("HEAD", "/redoc", PUBLIC),
    ("GET", "/v1/branding/logo/{variant}", PUBLIC),
    ("GET", "/", PUBLIC),
    ("GET", "/v1/me", _CATALOG),
    # Threads.
    ("GET", "/v1/sessions", runner(SCOPE_SESSIONS_LIST)),
    ("GET", "/v1/sessions/projects", runner(SCOPE_SESSIONS_LIST)),
    ("POST", "/v1/sessions", runner(SCOPE_SESSIONS_CREATE)),
    ("POST", "/v1/sessions/orchestrated", runner(SCOPE_SESSIONS_CREATE)),
    ("POST", "/v1/sessions/{source_id}/fork", runner(SCOPE_SESSIONS_CREATE)),
    ("WS", "/v1/sessions/updates", HUMAN),
    # Catalogues.
    ("GET", "/v1/agents", _CATALOG),
    ("GET", "/v1/agents/{agent_id}", _CATALOG),
    ("GET", "/v1/agents/{agent_id}/config", _CATALOG),
    # A runner may manage its person's own user agents; editing someone
    # else's agent is refused by the agent-ownership check.
    ("POST", "/v1/agents", runner(SCOPE_AGENTS)),
    ("PATCH", "/v1/agents/{agent_id}", runner(SCOPE_AGENTS)),
    ("DELETE", "/v1/agents/{agent_id}", runner(SCOPE_AGENTS)),
    ("GET", "/v1/skills", _CATALOG),
    ("GET", "/v1/harnesses", _CATALOG),
    ("GET", "/v1/policy-registry", _CATALOG),
    # Runners.
    ("GET", "/v1/runners", _CATALOG),
    ("GET", "/v1/runners/{runner_id}/status", _CATALOG),
    # Authenticated by the runner's binding token.
    ("POST", "/v1/runners/{runner_id}/token", PUBLIC),
    ("WS", "/v1/runners/{runner_id}/tunnel", runner(SCOPE_TUNNEL)),
    # Hosts.
    ("GET", "/v1/hosts", _CATALOG),
    ("POST", "/v1/hosts", HUMAN),
    # MCP gateway: the thread owner's own connection.
    ("POST", "/v1/mcp/{service_id}", runner(SCOPE_MCP)),
    # Schedules.
    ("GET", "/v1/scheduled-tasks", runner(SCOPE_SCHEDULES)),
    ("POST", "/v1/scheduled-tasks", runner(SCOPE_SCHEDULES)),
    ("GET", "/v1/scheduled-tasks/{scheduled_task_id}", runner(SCOPE_SCHEDULES)),
    ("PATCH", "/v1/scheduled-tasks/{scheduled_task_id}", runner(SCOPE_SCHEDULES)),
    ("DELETE", "/v1/scheduled-tasks/{scheduled_task_id}", runner(SCOPE_SCHEDULES)),
    ("GET", "/v1/scheduled-tasks/{scheduled_task_id}/runs", runner(SCOPE_SCHEDULES)),
    ("POST", "/v1/scheduled-tasks/{scheduled_task_id}/run", runner(SCOPE_SCHEDULES)),
    # A person's or an administrator's pages.
    ("WS", "/v1/dictation/stream", HUMAN),
    ("GET", "/v1/extensions", HUMAN),
    ("GET", "/v1/extensions/diagnostics", HUMAN),
    ("GET", "/v1/extensions/{extension_id}", HUMAN),
    ("GET", "/v1/extensions/{extension_id}/assets/{digest}/{asset_name}", HUMAN),
    ("POST", "/v1/imports", HUMAN),
    ("POST", "/v1/imports/local", HUMAN),
    ("POST", "/v1/imports/local/stream", HUMAN),
    ("GET", "/v1/sandbox-providers/{provider}/harnesses/{harness}/model-options", HUMAN),
    ("GET", "/v1/sharing", HUMAN),
    ("PUT", "/v1/sharing", HUMAN),
    ("GET", "/v1/usage", HUMAN),
    ("GET", "/v1/policies", HUMAN),
    ("POST", "/v1/policies", HUMAN),
    ("GET", "/v1/policies/{policy_id}", HUMAN),
    ("PATCH", "/v1/policies/{policy_id}", HUMAN),
    ("DELETE", "/v1/policies/{policy_id}", HUMAN),
    # Approvals: the approver inbox, approver grants and protected rules.
    ("GET", "/v1/approvals", HUMAN),
    ("GET", "/v1/approvers", HUMAN),
    ("POST", "/v1/approvers", HUMAN),
    ("DELETE", "/v1/approvers", HUMAN),
    ("PUT", "/v1/approval-settings", HUMAN),
    ("GET", "/v1/protected-policies", HUMAN),
    ("POST", "/v1/protected-policies", HUMAN),
    ("PATCH", "/v1/protected-policies/{policy_id}", HUMAN),
    ("DELETE", "/v1/protected-policies/{policy_id}", HUMAN),
    ("GET", "/v1/projects", HUMAN),
    ("POST", "/v1/projects", HUMAN),
    ("GET", "/v1/projects/order", HUMAN),
    ("PUT", "/v1/projects/order", HUMAN),
    ("GET", "/v1/projects/{project_id}", HUMAN),
    ("PATCH", "/v1/projects/{project_id}", HUMAN),
    ("DELETE", "/v1/projects/{project_id}", HUMAN),
    ("GET", "/v1/preferences/configuration", HUMAN),
    ("PUT", "/v1/preferences/configuration", HUMAN),
    ("GET", "/v1/preferences/shared", HUMAN),
    ("POST", "/v1/preferences/shared/operations", HUMAN),
    ("GET", "/v1/mcp-registry/services", HUMAN),
    ("PUT", "/v1/mcp-registry/services/{service_id}/connection", HUMAN),
    ("DELETE", "/v1/mcp-registry/services/{service_id}/connection", HUMAN),
    ("POST", "/v1/mcp-registry/services/{service_id}/test", HUMAN),
    ("GET", "/v1/mcp-registry/admin/services", HUMAN),
    ("GET", "/v1/mcp-registry/admin/presets", HUMAN),
    ("POST", "/v1/mcp-registry/admin/services", HUMAN),
    ("PUT", "/v1/mcp-registry/admin/services/{service_id}", HUMAN),
    ("DELETE", "/v1/mcp-registry/admin/services/{service_id}", HUMAN),
    ("POST", "/v1/mcp-registry/admin/services/{service_id}/discover", HUMAN),
)

# Families whose paths are only known at run time. Sign-in and credential
# issuance authenticate themselves, and machine tokens are refused there by the
# auth provider; connecting an account needs the person's browser.
PREFIX_RULES: tuple[tuple[str, RouteRule], ...] = (
    ("/auth/", PUBLIC),
    ("/oauth/", PUBLIC),
    ("/v1/connections/", HUMAN),
)

# Mounted applications, by mount name.
MOUNT_RULES: dict[str, RouteRule] = {"web-ui": PUBLIC}


def _keyed(
    prefix: str, routes: Iterable[tuple[str, str, RouteRule]]
) -> dict[tuple[str, str], RouteRule]:
    return {(method, prefix + suffix): rule for method, suffix, rule in routes}


ROUTE_RULES: dict[tuple[str, str], RouteRule] = {
    **_keyed("/v1/sessions/{session_id}", _SESSION_ROUTES),
    **_keyed("/v1/hosts/{host_id}", _HOST_ROUTES),
    **_keyed("", _OTHER_ROUTES),
}


def rule_for(method: str, path: str) -> RouteRule | None:
    """Return the rule for a route template, or ``None`` when unclassified.

    :param method: HTTP method, or ``"WS"`` for a WebSocket route.
    :param path: The route's path template, e.g. ``"/v1/sessions/{session_id}"``.
    """
    rule = ROUTE_RULES.get((method, path))
    if rule is not None:
        return rule
    for prefix, prefix_rule in PREFIX_RULES:
        if path.startswith(prefix):
            return prefix_rule
    return None


def route_keys(route: object, prefix: str = "") -> list[tuple[str, str]]:
    """Return the ``(method, path template)`` keys a route answers to.

    :param route: A Starlette/FastAPI route.
    :param prefix: A prefix the route is mounted under.
    """
    path = getattr(route, "path", None)
    if path is None or isinstance(route, Mount):
        return []
    if isinstance(route, (WebSocketRoute, APIWebSocketRoute)):
        return [("WS", prefix + path)]
    return [(method, prefix + path) for method in sorted(getattr(route, "methods", None) or ())]


def classify_added_routes(
    rules: dict[tuple[str, str], RouteRule],
    routes: Iterable[object],
    *,
    prefix: str = "",
    rule: RouteRule = HUMAN,
) -> None:
    """Classify routes an embedding application adds, unless already classified.

    :param rules: The application's extra rules (``app.state.route_authority_rules``).
    :param routes: The added routes.
    :param prefix: The prefix they are mounted under.
    :param rule: Their rule; people only by default.
    """
    for route in routes:
        for method, path in route_keys(route, prefix):
            if rule_for(method, path) is None:
                rules.setdefault((method, path), rule)


def matched_route_keys(app: ASGIApp, scope: Scope) -> tuple[str, str, str | None] | None:
    """Find the route the router will dispatch *scope* to.

    :returns: ``(method, path template, mount name)``, or ``None`` when no
        route fully matches (the router answers 404 or 405 itself).
    """
    router = getattr(app, "router", None)
    routes = getattr(router, "routes", ())
    method = "WS" if scope["type"] == "websocket" else scope.get("method", "GET")
    for route in routes:
        try:
            match, _ = route.matches(scope)
        except Exception:  # noqa: BLE001 — a matcher fault must fail closed below
            return (method, "<unmatchable>", None)
        if match is Match.FULL:
            if isinstance(route, Mount):
                return (method, route.path, route.name)
            return (method, getattr(route, "path", "<unknown>"), None)
    return None


class RouteAuthorityMiddleware:
    """Refuse callers a route's rule does not admit; fail closed when unclassified.

    Runs after authentication. For a machine principal it also marks the
    request so permission checks deny the administrator override and share
    grants, including in work the request starts.

    :param app: The downstream ASGI app.
    :param root_app: The application whose router decides the matched route.
    :param auth_provider: Resolves the request's principal, or ``None`` when
        authentication is off (every caller is then the local person).
    :param extra_rules: Rules for routes an embedding application adds.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        root_app: ASGIApp,
        auth_provider: AuthProvider | None,
        extra_rules: dict[tuple[str, str], RouteRule] | None = None,
    ) -> None:
        self._app = app
        self._root_app = root_app
        self._auth_provider = auth_provider
        self._extra_rules = extra_rules if extra_rules is not None else {}

    def _rule(self, method: str, path: str, mount: str | None) -> RouteRule | None:
        if mount is not None:
            return MOUNT_RULES.get(mount)
        return self._extra_rules.get((method, path)) or rule_for(method, path)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self._app(scope, receive, send)
            return
        matched = matched_route_keys(self._root_app, scope)
        if matched is None:
            await self._app(scope, receive, send)
            return
        method, path, mount = matched
        rule = self._rule(method, path, mount)
        if rule is None:
            await _deny(
                scope,
                send,
                403,
                ROUTE_UNCLASSIFIED_CODE,
                "This endpoint has no access classification and is unavailable.",
            )
            return
        principal = await self._principal(scope)
        if principal is not None and not rule.admits(principal):
            await _deny(
                scope,
                send,
                403,
                MACHINE_FORBIDDEN_CODE,
                f"A {principal.kind.value} credential cannot use this endpoint.",
            )
            return
        if principal is not None:
            from omnigent.debug_logging import add_audit_attrs

            add_audit_attrs(
                principal_kind=principal.kind.value,
                principal_runner_id=principal.runner_id,
                principal_host_id=principal.host_id,
            )
        if principal is None or not principal.is_machine:
            await self._app(scope, receive, send)
            return
        from omnigent.db.account_authority import machine_principal_scope

        with machine_principal_scope(True):
            await self._app(scope, receive, send)

    async def _principal(self, scope: Scope) -> Principal | None:
        if self._auth_provider is None:
            return None
        connection = HTTPConnection(scope)
        # Authentication middleware usually recorded it already; otherwise
        # authenticate off the event loop (it may read the database).
        recorded = recorded_principal(connection)
        if recorded is not None:
            return recorded
        return await asyncio.to_thread(self._auth_provider.principal, connection)


async def _deny(scope: Scope, send: Send, status: int, code: str, message: str) -> None:
    """Answer a refused request without reaching the route."""
    if scope["type"] == "websocket":
        await send({"type": "websocket.close", "code": _WS_FORBIDDEN_CLOSE_CODE, "reason": code})
        return
    body = json.dumps({"error": {"code": code, "message": message}}).encode()
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})
