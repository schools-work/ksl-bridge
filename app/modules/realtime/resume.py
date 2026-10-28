"""Session replay after reconnect: last messages kept in Redis for 2 minutes."""

from __future__ import annotations

import redis.asyncio as aioredis

TTL_S = 120
MAX_ITEMS = 200


async def save(r: aioredis.Redis, session_id: str, seq: int, text: str) -> None:
    key = f"session:{session_id}:replay"
    pipe = r.pipeline()
    pipe.rpush(key, f"{seq}\t{text}")
    pipe.ltrim(key, -MAX_ITEMS, -1)
    pipe.expire(key, TTL_S)
    await pipe.execute()


async def load_after(r: aioredis.Redis, session_id: str, last_seq: int) -> list[str]:
    items = await r.lrange(f"session:{session_id}:replay", 0, -1)
    out = []
    for raw in items:
        s, _, text = (raw.decode() if isinstance(raw, bytes) else raw).partition("\t")
        if int(s) > last_seq:
            out.append(text)
    return out


async def remember_owner(r: aioredis.Redis, session_id: str, subject: str, room: str | None) -> None:
    await r.hset(f"session:{session_id}:meta", mapping={"subject": subject, "room": room or ""})
    await r.expire(f"session:{session_id}:meta", TTL_S)


async def owner(r: aioredis.Redis, session_id: str) -> tuple[str, str] | None:
    meta = await r.hgetall(f"session:{session_id}:meta")
    if not meta:
        return None
    return meta[b"subject"].decode(), meta[b"room"].decode()
