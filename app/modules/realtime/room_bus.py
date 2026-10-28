"""Redis pub/sub fan-out across gateway replicas. Large binaries travel as Redis keys, not in pub/sub."""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any

import redis.asyncio as aioredis

from app.platform.logging import get_logger

from .local_registry import LocalRegistry

log = get_logger(__name__)
BLOB_TTL_S = 60


class RoomBus:
    """Implements the pipeline's Broadcaster port."""

    def __init__(self, redis: aioredis.Redis, local: LocalRegistry):
        self.r, self.local = redis, local
        self.pubsub = redis.pubsub()
        self.subscribed: set[str] = set()
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        await self.pubsub.subscribe("__keepalive__")
        self._task = asyncio.create_task(self._reader(), name="room-bus-reader")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
        await self.pubsub.aclose()

    async def join(self, room: str) -> None:
        if room not in self.subscribed:
            await self.pubsub.subscribe(f"room:{room}")
            self.subscribed.add(room)

    async def leave_if_empty(self, room: str) -> None:
        if room in self.subscribed and not self.local.members(room):
            await self.pubsub.unsubscribe(f"room:{room}")
            self.subscribed.discard(room)

    async def publish(
        self,
        room: str,
        msg: dict[str, Any],
        binary: bytes | None = None,
        exclude_member: str | None = None,
        only_roles: list[str] | None = None,
        only_member: str | None = None,
        only_modality_out: list[str] | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "msg": msg,
            "exclude": exclude_member,
            "only_roles": only_roles,
            "only_member": only_member,
            "only_out": only_modality_out,
        }
        if binary is not None:
            key = f"blob:{uuid.uuid4().hex}"
            await self.r.set(key, binary, ex=BLOB_TTL_S)
            payload["blob"] = key
        await self.r.publish(f"room:{room}", json.dumps(payload, default=str))

    async def _reader(self) -> None:
        blob_cache: dict[str, bytes] = {}
        while True:
            try:
                async for m in self.pubsub.listen():
                    if m["type"] != "message":
                        continue
                    channel = m["channel"].decode() if isinstance(m["channel"], bytes) else m["channel"]
                    if not channel.startswith("room:"):
                        continue
                    room = channel.split(":", 1)[1]
                    p = json.loads(m["data"])
                    binary = None
                    if p.get("blob"):
                        binary = blob_cache.get(p["blob"]) or await self.r.get(p["blob"])
                        if binary is not None:
                            blob_cache = {p["blob"]: binary}
                    for c in self.local.members(room):
                        if p["exclude"] and p["exclude"] == c.member_id:
                            continue
                        if p["only_member"] and p["only_member"] != c.member_id:
                            continue
                        if p["only_roles"] and c.role not in p["only_roles"]:
                            continue
                        if p["only_out"] and c.modality_out not in p["only_out"]:
                            continue
                        c.enqueue(p["msg"], binary)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("room_bus_reader_error")
                await asyncio.sleep(0.5)
