"""Redis-backed per-session state with TTL (survives a gateway restart for resume)."""

from __future__ import annotations

import json
from typing import Any

import redis.asyncio as aioredis

TTL_S = 120


class RedisSessionStore:
    def __init__(self, redis: aioredis.Redis):
        self.r = redis

    async def save(self, session_id: str, state: dict[str, Any]) -> None:
        await self.r.set(f"session:{session_id}:state", json.dumps(state, default=str), ex=TTL_S)

    async def load(self, session_id: str) -> dict[str, Any] | None:
        raw = await self.r.get(f"session:{session_id}:state")
        return json.loads(raw) if raw else None

    async def delete(self, session_id: str) -> None:
        await self.r.delete(f"session:{session_id}:state")
