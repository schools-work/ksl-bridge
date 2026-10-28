"""Redis token-bucket rate limiter (atomic Lua script)."""

from __future__ import annotations

import time

import redis.asyncio as aioredis

_LUA = """
local key, rate, burst, now, cost = KEYS[1], tonumber(ARGV[1]), tonumber(ARGV[2]), tonumber(ARGV[3]), tonumber(ARGV[4])
local b = redis.call('HMGET', key, 'tokens', 'ts')
local tokens = tonumber(b[1]) or burst
local ts = tonumber(b[2]) or now
tokens = math.min(burst, tokens + (now - ts) * rate)
local allowed = 0
if tokens >= cost then tokens = tokens - cost; allowed = 1 end
redis.call('HSET', key, 'tokens', tokens, 'ts', now)
redis.call('PEXPIRE', key, math.ceil(burst / rate * 1000) + 1000)
return {allowed, tostring(tokens)}
"""


class RateLimiter:
    def __init__(self, redis: aioredis.Redis, prefix: str = "rl"):
        self.r, self.prefix = redis, prefix
        self._script = redis.register_script(_LUA)

    async def allow(self, key: str, rate_per_s: float, burst: int, cost: float = 1.0) -> tuple[bool, float]:
        """Returns (allowed, retry_after_seconds)."""
        allowed, tokens = await self._script(keys=[f"{self.prefix}:{key}"], args=[rate_per_s, burst, time.time(), cost])
        if int(allowed) == 1:
            return True, 0.0
        return False, max(0.0, (cost - float(tokens)) / rate_per_s)


# policy table (requests per second, burst); see MASTER_PROMPT §6
POLICIES: dict[str, tuple[float, int]] = {
    "auth": (5 / 60, 5),  # 5 per minute
    "room_join": (1.0, 10),
    "upload": (0.2, 5),
    "ws_text": (5.0, 20),
}
