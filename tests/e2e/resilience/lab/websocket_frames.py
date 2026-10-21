"""Just enough RFC 6455 framing for the fault proxy to act on whole WebSocket frames."""

from __future__ import annotations

import os
import re
import struct
from dataclasses import dataclass

OP_CONTINUATION = 0x0
OP_TEXT = 0x1
OP_BINARY = 0x2
OP_PING = 0x9
OP_PONG = 0xA
DATA_OPCODES = frozenset({OP_CONTINUATION, OP_TEXT, OP_BINARY})

_HEAD_LIMIT = 16 * 1024
_SWITCHING_PROTOCOLS = re.compile(rb"^HTTP/1\.[01] 101 ")
_UPGRADE_WEBSOCKET = re.compile(rb"(?im)^upgrade:[ \t]*websocket[ \t]*\r?$")


@dataclass(frozen=True)
class Frame:
    """One complete frame.

    :param opcode: Frame opcode, e.g. :data:`OP_PING`.
    :param payload: Unmasked payload of a control frame; empty for data frames,
        whose content the proxy never reads.
    :param raw: The frame exactly as it crossed the wire.
    """

    opcode: int
    payload: bytes
    raw: bytes


class FrameSplitter:
    """Split one direction of a WebSocket connection into whole frames.

    :param response_head: Whether the stream starts with the server's response
        head. It is held until complete, then passed through; frames follow only
        if it accepted a WebSocket upgrade, and otherwise every later byte
        passes through as it is.
    """

    def __init__(self, *, response_head: bool) -> None:
        self._buffer = bytearray()
        self._in_head = response_head
        self._passthrough = False
        #: Whether the response switched protocols, known once its head is complete.
        self.upgraded = not response_head
        #: Whether frames follow, known once the response head is complete.
        self.websocket = not response_head

    def feed(self, data: bytes) -> tuple[bytes, list[Frame]]:
        """Consume *data*.

        :param data: Bytes read from the socket.
        :returns: Bytes to forward as they are (the response head, or everything
            after a response that is not a WebSocket upgrade), then every frame now
            complete. A partial head or frame stays buffered until it completes.
        """
        if self._passthrough:
            return data, []
        self._buffer += data
        head = b""
        if self._in_head:
            end = self._buffer.find(b"\r\n\r\n")
            if end < 0 and len(self._buffer) <= _HEAD_LIMIT:
                return b"", []
            self._in_head = False
            if end >= 0:
                head = bytes(self._buffer[: end + 4])
                del self._buffer[: end + 4]
                self.upgraded = _SWITCHING_PROTOCOLS.match(head) is not None
                self.websocket = self.upgraded and _UPGRADE_WEBSOCKET.search(head) is not None
            if not self.websocket:
                self._passthrough = True
                rest, self._buffer = bytes(self._buffer), bytearray()
                return head + rest, []
        frames: list[Frame] = []
        while (frame := self._next_frame()) is not None:
            frames.append(frame)
        return head, frames

    def _next_frame(self) -> Frame | None:
        buffer = self._buffer
        if len(buffer) < 2:
            return None
        masked = bool(buffer[1] & 0x80)
        length = buffer[1] & 0x7F
        offset = 2
        if length == 126:
            if len(buffer) < 4:
                return None
            (length,) = struct.unpack_from("!H", buffer, 2)
            offset = 4
        elif length == 127:
            if len(buffer) < 10:
                return None
            (length,) = struct.unpack_from("!Q", buffer, 2)
            offset = 10
        mask_offset = offset
        if masked:
            offset += 4
        end = offset + length
        if len(buffer) < end:
            return None
        opcode = buffer[0] & 0x0F
        payload = b""
        if opcode not in DATA_OPCODES:
            payload = bytes(buffer[offset:end])
            if masked:
                payload = _apply_mask(payload, bytes(buffer[mask_offset:offset]))
        frame = Frame(opcode=opcode, payload=payload, raw=bytes(buffer[:end]))
        del buffer[:end]
        return frame


def encode_frame(opcode: int, payload: bytes, *, masked: bool) -> bytes:
    """Encode one final frame.

    :param opcode: Frame opcode, e.g. :data:`OP_PONG`.
    :param payload: Unmasked payload.
    :param masked: ``True`` for frames a client sends, as RFC 6455 requires.
    :returns: The frame's wire bytes.
    """
    head = bytearray([0x80 | opcode])
    mask_bit = 0x80 if masked else 0
    if len(payload) < 126:
        head.append(mask_bit | len(payload))
    elif len(payload) < 1 << 16:
        head += bytes([mask_bit | 126]) + struct.pack("!H", len(payload))
    else:
        head += bytes([mask_bit | 127]) + struct.pack("!Q", len(payload))
    if not masked:
        return bytes(head) + payload
    mask = os.urandom(4)
    return bytes(head) + mask + _apply_mask(payload, mask)


def _apply_mask(payload: bytes, mask: bytes) -> bytes:
    return bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))
