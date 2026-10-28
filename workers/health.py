"""Worker heartbeats → Redis hash `workers:heartbeat` (read by /readyz)."""

from __future__ import annotations

import asyncio
import os
import socket
import time

import redis.asyncio as aioredis


async def heartbeat_forever(r: aioredis.Redis, task: str, every_s: float = 10.0) -> None:
    name = f"{task}@{socket.gethostname()}:{os.getpid()}"
    while True:
        await r.hset("workers:heartbeat", name, str(time.time()))
        await asyncio.sleep(every_s)
