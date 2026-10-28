"""Shared test helpers for the native-harness policy hooks.

Used by both ``tests/harnesses/claude_native/test_claude_native_hook.py`` and
``tests/harnesses/codex_native/test_codex_native_hook.py`` so the fail-closed failure-mode stub
lives in one place and can't drift if new modes are added.
"""

from __future__ import annotations

import httpx


def make_failing_client(mode: str) -> type:
    """
    Build an ``httpx.Client`` stub that fails the policy POST a given way.

    :param mode: One of ``"connect_error"`` (POST raises),
        ``"connect_timeout"`` (the connection attempt times out),
        ``"read_timeout"`` (the answer times out before any held-poll floor),
        ``"dropped_connection"`` (the server closes without answering),
        ``"non_2xx"`` (503 → ``raise_for_status``), ``"bad_request"`` (400),
        ``"empty_body"`` (200, no content), ``"malformed_json"`` (200,
        non-JSON body), ``"non_object_json"`` (200, a JSON list) or
        ``"no_verdict"`` (200, a JSON object without ``result``).
    :returns: A class usable as a drop-in for :class:`httpx.Client`.
    """

    class _FailingHttpxClient:
        def __init__(self, *, headers: dict[str, str], timeout: object) -> None:
            del headers, timeout

        def __enter__(self) -> _FailingHttpxClient:
            return self

        def __exit__(self, *args: object) -> None:
            del args

        def post(self, url: str, *, json: dict[str, object]) -> httpx.Response:
            del json
            req = httpx.Request("POST", url)
            if mode == "connect_error":
                raise httpx.ConnectError("AP unreachable", request=req)
            if mode == "connect_timeout":
                raise httpx.ConnectTimeout("AP connect timed out", request=req)
            if mode == "read_timeout":
                raise httpx.ReadTimeout("AP answer timed out", request=req)
            if mode == "dropped_connection":
                raise httpx.RemoteProtocolError("Server disconnected", request=req)
            if mode == "non_2xx":
                return httpx.Response(503, text="upstream down", request=req)
            if mode == "bad_request":
                return httpx.Response(400, text="bad request", request=req)
            if mode == "empty_body":
                return httpx.Response(200, content=b"", request=req)
            if mode == "non_object_json":
                return httpx.Response(200, text='["POLICY_ACTION_ALLOW"]', request=req)
            if mode == "no_verdict":
                return httpx.Response(200, text='{"reason": "no result field"}', request=req)
            if mode == "malformed_json":
                return httpx.Response(200, text="not json at all", request=req)
            raise ValueError(f"unknown failure mode {mode!r}")

    return _FailingHttpxClient
