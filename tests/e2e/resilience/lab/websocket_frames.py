"""Just enough RFC 6455 framing for the fault proxy to act on whole WebSocket frames."""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass

OP_CONTINUATION = 0x0
OP_TEXT = 0x1
OP_BINARY = 0x2
OP_PING = 0x9
OP_PONG = 0xA
DATA_OPCODES = frozenset({OP_CONTINUATION, OP_TEXT, OP_BINARY})


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
    """Split one direction of an upgraded connection into whole frames.

    :param http_head: Whether the stream starts with an HTTP head (the server's
        ``101`` response) that passes through before the first frame.
    """

    def __init__(self, *, http_head: bool) -> None:
        self._buffer = bytearray()
        self._in_head = http_head

    def feed(self, data: bytes) -> tuple[bytes, list[Frame]]:
        """Consume *data*.

        :param data: Bytes read from the socket.
        :returns: Head bytes to forward as they are, then every frame now complete.
            A partial frame stays buffered until its remaining bytes arrive.
        """
        self._buffer += data
        head = b""
        if self._in_head:
            end = self._buffer.find(b"\r\n\r\n")
            if end < 0:
                head, self._buffer = bytes(self._buffer), bytearray()
                return head, []
            head = bytes(self._buffer[: end + 4])
            del self._buffer[: end + 4]
            self._in_head = False
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
