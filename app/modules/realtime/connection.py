"""Per-socket writer: bounded queue, droppable partials, sequence numbers, replay buffer."""

from __future__ import annotations

import asyncio
from collections import deque
from typing import Any

from fastapi import WebSocket

from app.platform.metrics import DROPPED_PARTIALS, WS_MESSAGES

from .protocol import encode, now_ms

DROPPABLE = frozenset({"utterance.partial", "typing", "signing"})
REPLAY_SIZE = 200


class ClientConn:
    def __init__(self, ws: WebSocket, session_id: str, subject: str, maxsize: int = 256):
        self.ws, self.session_id, self.subject = ws, session_id, subject
        self.member_id: str | None = None
        self.role: str = "observer"
        self.modality_out: str = "text"
        self.room_id: str | None = None
        self.q: asyncio.Queue[tuple[dict[str, Any], bytes | None]] = asyncio.Queue(maxsize)
        self.seq = 0
        self.last_peer_seq = 0
        self.replay: deque[tuple[int, str]] = deque(maxlen=REPLAY_SIZE)
        self.inflight_landmarks = 0
        self.closed = False

    def enqueue(self, msg: dict[str, Any], binary: bytes | None = None) -> bool:
        """Never blocks. Under pressure, drops droppable messages first; finals are kept."""
        if self.closed:
            return False
        if self.q.full():
            if msg.get("type") in DROPPABLE:
                DROPPED_PARTIALS.inc()
                return False
            kept = [m for m in self._drain() if m[0].get("type") not in DROPPABLE]
            DROPPED_PARTIALS.inc(max(0, self.q.maxsize - len(kept)))
            for m in kept[-(self.q.maxsize - 1) :]:
                self.q.put_nowait(m)
        self.q.put_nowait((msg, binary))
        return True

    def _drain(self) -> list[tuple[dict[str, Any], bytes | None]]:
        out = []
        while not self.q.empty():
            out.append(self.q.get_nowait())
        return out

    def envelope(self, msg: dict[str, Any]) -> str:
        self.seq += 1
        env = {"v": 1, "seq": self.seq, "ts": now_ms(), "ack": self.last_peer_seq, **msg}
        text = encode(env)
        self.replay.append((self.seq, text))
        return text

    async def writer(self) -> None:
        try:
            while True:
                msg, binary = await self.q.get()
                await self.ws.send_text(self.envelope(msg))
                WS_MESSAGES.labels("out", msg.get("type", "?")).inc()
                if binary is not None:
                    await self.ws.send_bytes(binary)
        except (asyncio.CancelledError, RuntimeError):
            self.closed = True
            raise

    def replay_after(self, last_seq: int) -> list[str]:
        return [t for s, t in self.replay if s > last_seq]
