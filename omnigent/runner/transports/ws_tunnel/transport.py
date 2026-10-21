"""Server-side ``WSTunnelTransport`` — httpx transport that tunnels
HTTP through a runner's WebSocket (Phase 4).

Per ``designs/RUNNER.md`` §3 "Sketch of the adapters", this is an
``httpx.AsyncBaseTransport`` subclass. Every existing call site
that uses ``httpx.AsyncClient`` keeps working unchanged — only the
transport object handed to the client differs.

Wire flow per request:
1. Allocate a fresh ``req_id`` (uuid4 hex).
2. Open reassembly state in the registry.
3. Send a :class:`RequestFrame` over the runner's WebSocket.
4. Await the :class:`ResponseHeadFrame` for status + headers.
5. Stream :class:`ResponseBodyFrame` chunks until
   :class:`ResponseEndFrame` or session abort.
6. Close the request in the registry.

If the runner is offline (no session in registry) → raise
``httpx.ConnectError``. If the tunnel closes before the request frame
is queued → :class:`TunnelRequestNotSentError`. If it closes after →
the abort propagates as a ``ConnectionError``, because the runner may
already have acted on the request.

The request's httpx timeouts apply as they would on a socket: ``write``
bounds queuing the request frame, and ``read`` bounds the wait for the
response head and for each body chunk. A request that times out is
cancelled on the runner and raises the matching ``httpx`` timeout, so a
half-open tunnel answers within the caller's budget instead of the
tunnel's liveness deadline. Use :func:`runner_request_may_have_been_delivered`
to tell a request that never left from one whose outcome is unknown.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator

import httpx

from omnigent.runner.transports.ws_tunnel.frames import (
    RequestCancelFrame,
    RequestFrame,
    ResponseBodyFrame,
    decode_body,
    encode_body,
    encode_frame,
)
from omnigent.runner.transports.ws_tunnel.registry import RequestState, TunnelRegistry


class TunnelRequestNotSentError(ConnectionError):
    """The tunnel closed before the request frame was queued, so the runner never saw it."""


def runner_request_may_have_been_delivered(error: BaseException) -> bool:
    """
    Return whether a failed request to a runner may still have reached it.

    ``False`` means the request certainly never left, so sending it again
    cannot repeat it. ``True`` means its outcome is unknown: the runner may
    have acted on it and the answer was lost.

    :param error: Exception raised by a runner client request, e.g. an
        ``httpx.ReadTimeout`` or the ``ConnectionError`` of a closed tunnel.
    :returns: ``True`` when the runner may have received the request.
    """
    from omnigent.native._native_post_delivery import post_may_have_been_delivered

    if isinstance(error, TunnelRequestNotSentError):
        return False
    if isinstance(error, httpx.HTTPError):
        return post_may_have_been_delivered(error)
    return isinstance(error, ConnectionError)


def _request_timeout(request: httpx.Request, kind: str) -> float | None:
    """
    Read one httpx timeout for *request*.

    :param request: Request whose ``timeout`` extension httpx filled in.
    :param kind: ``"read"`` or ``"write"``.
    :returns: Seconds, or ``None`` for no limit.
    """
    timeouts = request.extensions.get("timeout")
    value = timeouts.get(kind) if isinstance(timeouts, dict) else None
    return float(value) if isinstance(value, int | float) else None


async def _cancel_on_runner(registry: TunnelRegistry, state: RequestState, req_id: str) -> None:
    """
    Ask the runner to abort a request this side gave up on, best effort.

    :param registry: Registry owning the request's tunnel.
    :param state: Request state, carrying the tunnel generation it was sent on.
    :param req_id: Tunnel request id.
    """
    if not registry.request_is_open(state.session, req_id):
        return
    try:  # noqa: SIM105 — contextlib.suppress doesn't work with await
        await registry.send_text(
            state.session,
            encode_frame(RequestCancelFrame(id=req_id, reason="client_timeout")),
        )
    except Exception:  # noqa: BLE001 — best-effort cleanup
        pass


class _TunneledByteStream(httpx.AsyncByteStream):
    """Adapts the registry's body queue into an ``httpx.AsyncByteStream``."""

    def __init__(
        self,
        registry: TunnelRegistry,
        runner_id: str,
        req_id: str,
        state: RequestState,
        request: httpx.Request | None = None,
    ) -> None:
        self._registry = registry
        self._runner_id = runner_id
        self._req_id = req_id
        self._state = state
        self._request = request
        self._read_timeout = _request_timeout(request, "read") if request is not None else None

    async def __aiter__(self) -> AsyncIterator[bytes]:
        state = self._state
        try:
            while True:
                try:
                    item = await asyncio.wait_for(state.body_queue.get(), self._read_timeout)
                except TimeoutError as exc:
                    await _cancel_on_runner(self._registry, state, self._req_id)
                    raise httpx.ReadTimeout(
                        "runner sent no response body within the read timeout",
                        request=self._request,
                    ) from exc
                if state.aborted_with is not None:
                    raise state.aborted_with
                if item is None:
                    # Sentinel: end-event signalled, no more chunks.
                    break
                # Mypy/runtime: item must be a ResponseBodyFrame here.
                if isinstance(item, ResponseBodyFrame):
                    yield decode_body(item.body, item.encoding)
        finally:
            self._registry.close_request(
                self._runner_id,
                self._req_id,
                session=state.session,
            )

    async def aclose(self) -> None:
        # Close the request from the caller side — typically called
        # when the consumer's ``async with`` exits early (e.g. SSE
        # client disconnect). The transport translates this into a
        # request.cancel frame so the runner aborts.
        state = self._state
        if self._registry.request_is_open(state.session, self._req_id):
            try:  # noqa: SIM105 — contextlib.suppress doesn't work with await
                await self._registry.send_text(
                    state.session,
                    encode_frame(
                        RequestCancelFrame(
                            id=self._req_id,
                            reason="client_disconnected",
                        )
                    ),
                )
            except Exception:  # noqa: BLE001 — best-effort cleanup
                pass
        self._registry.close_request(
            self._runner_id,
            self._req_id,
            session=state.session,
        )


class WSTunnelTransport(httpx.AsyncBaseTransport):
    """httpx transport that tunnels each request through a runner WebSocket.

    Construct one transport per (registry, runner_id) pair (or share
    one via a thin lookup that resolves runner_id per request — that's
    a higher-level routing concern, not this transport's).

    :param registry: The :class:`TunnelRegistry` that owns the runner's
        live WebSocket and reassembly state.
    :param runner_id: Which runner this transport routes to.
    """

    def __init__(self, registry: TunnelRegistry, runner_id: str) -> None:
        self._registry = registry
        self._runner_id = runner_id

    async def wait_for_runner(self, timeout_s: float) -> bool:
        """
        Wait until this transport's runner has a live tunnel or the timeout expires.

        Lets a caller that just saw the runner go offline park until it
        re-registers, instead of retrying requests on a fixed interval.

        :param timeout_s: Maximum seconds to wait, e.g. ``90.0``.
        :returns: ``True`` when the runner is registered when the wait ends.
        """
        return (
            await self._registry.wait_for_runner(self._runner_id, timeout_s=timeout_s) is not None
        )

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        session = self._registry.get(self._runner_id)
        if session is None:
            # The runner is offline. Raising ConnectError matches
            # what httpx would emit for a TCP connect failure, so
            # the call site's exception handling for "runner went
            # away" is identical to "TCP connect refused."
            raise httpx.ConnectError(f"runner {self._runner_id!r} is offline")

        req_id = uuid.uuid4().hex
        # Read the request body up front. Streaming request bodies
        # would need a multi-frame send; v1 sends the whole body in
        # the request frame because all our request bodies are
        # tiny JSON.
        body = await request.aread() if request.content else b""
        content_type = request.headers.get("content-type", "application/json")
        body_str, encoding = encode_body(body, content_type) if body else (None, "utf-8")

        try:
            state = self._registry.open_request(
                self._runner_id,
                req_id,
                generation=request.extensions.get("runner_tunnel_generation"),
            )
        except KeyError as exc:
            raise httpx.ConnectError(f"runner {self._runner_id!r} is offline") from exc
        except ConnectionError as exc:
            raise TunnelRequestNotSentError(str(exc)) from exc
        try:
            try:
                await asyncio.wait_for(
                    self._registry.send_text(
                        state.session,
                        encode_frame(
                            RequestFrame(
                                id=req_id,
                                method=request.method,
                                path=request.url.path,
                                query_string=request.url.query.decode("utf-8"),
                                headers=[[k, v] for k, v in request.headers.items()],
                                body=body_str,
                                encoding=encoding,
                                # Best-effort hint for streaming responses;
                                # not load-bearing on the runner side.
                                stream=True,
                            )
                        ),
                    ),
                    _request_timeout(request, "write"),
                )
            except TimeoutError as exc:
                # The frame may still be queued once the tunnel's loop runs.
                raise httpx.WriteTimeout(
                    "runner tunnel did not accept the request within the write timeout",
                    request=request,
                ) from exc
            except ConnectionError as exc:
                raise TunnelRequestNotSentError(str(exc)) from exc
            # Block until the response head arrives, the tunnel aborts the
            # request, or the read timeout expires.
            try:
                head = await asyncio.wait_for(state.head_future, _request_timeout(request, "read"))
            except TimeoutError as exc:
                await _cancel_on_runner(self._registry, state, req_id)
                raise httpx.ReadTimeout(
                    "runner sent no response within the read timeout", request=request
                ) from exc
        except BaseException:
            # If we failed before getting head, clean up the slot so
            # we don't leak in_flight state.
            self._registry.close_request(self._runner_id, req_id, session=state.session)
            raise

        # Wrap the body queue as an httpx AsyncByteStream. The stream
        # owns close_request() — cleanup happens when the response
        # iterator finishes or the consumer's `async with` exits.
        stream = _TunneledByteStream(self._registry, self._runner_id, req_id, state, request)
        return httpx.Response(
            status_code=head.status,
            headers=[(k, v) for k, v in head.headers],
            stream=stream,
            request=request,
        )

    async def aclose(self) -> None:
        # Nothing to close — the transport doesn't own connections;
        # the registry does. Implementing this lets httpx call it
        # without exploding.
        pass
