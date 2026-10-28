"""Submit jobs to worker streams (XADD) and await replies on a per-gateway channel, with deadlines."""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any

import msgpack
import redis.asyncio as aioredis

from app.platform.logging import get_logger

log = get_logger(__name__)
STREAM_MAXLEN = 10_000


class RedisJobClient:
    """Implements the pipeline's JobClient port. Workers publish replies to `reply:<gateway-id>`."""

    def __init__(self, redis: aioredis.Redis):
        self.r = redis
        self.gateway_id = uuid.uuid4().hex[:12]
        self.channel = f"reply:{self.gateway_id}"
        self.pending: dict[str, asyncio.Future[dict[str, Any] | None]] = {}
        self._task: asyncio.Task[None] | None = None
        self._pubsub = redis.pubsub()
        self._beats: dict[str, float] = {}
        self._beats_at = 0.0

    async def start(self) -> None:
        await self._pubsub.subscribe(self.channel)
        self._task = asyncio.create_task(self._reader(), name="job-reply-reader")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
        await self._pubsub.aclose()

    async def available(self, task: str) -> bool:
        """A task is served if some worker heartbeat for it is fresh (< 30 s). Cached for 2 s.

        (Consumer-group membership is not a reliable liveness signal: Redis registers a consumer only once a
        message has been delivered to it.)"""
        now = time.time()
        if now - self._beats_at > 2.0:
            raw = await self.r.hgetall("workers:heartbeat")
            self._beats = {(k.decode() if isinstance(k, bytes) else k): float(v) for k, v in raw.items()}
            self._beats_at = now
        return any(name.split("@", 1)[0] == task and now - ts < 30 for name, ts in self._beats.items())

    async def call(self, task: str, payload: dict[str, Any], deadline_ms: int) -> dict[str, Any] | None:
        if not await self.available(task):
            return None
        job_id = uuid.uuid4().hex
        fut: asyncio.Future[dict[str, Any] | None] = asyncio.get_running_loop().create_future()
        self.pending[job_id] = fut
        body = msgpack.packb(
            {
                "job_id": job_id,
                "reply_to": self.channel,
                "input": payload,
                "deadline_ms": int(time.time() * 1000) + deadline_ms,
            },
            use_bin_type=True,
        )
        try:
            await self.r.xadd(f"jobs:{task}", {"p": body}, maxlen=STREAM_MAXLEN, approximate=True)
            return await asyncio.wait_for(fut, timeout=deadline_ms / 1000)
        except TimeoutError:
            log.warning("job_timeout", task=task)
            return None
        finally:
            self.pending.pop(job_id, None)

    async def _reader(self) -> None:
        while True:
            try:
                async for m in self._pubsub.listen():
                    if m["type"] != "message":
                        continue
                    msg = msgpack.unpackb(m["data"], raw=False)
                    fut = self.pending.get(msg.get("job_id", ""))
                    if fut and not fut.done():
                        fut.set_result(msg.get("output") if msg.get("ok", True) else None)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("job_reply_reader_error")
                await asyncio.sleep(0.5)
