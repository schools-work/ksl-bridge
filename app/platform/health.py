"""Liveness and readiness endpoints."""

from __future__ import annotations

import time

from fastapi import APIRouter
from sqlalchemy import text

from .db import get_engine
from .redis import get_redis

router = APIRouter(tags=["health"])


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/readyz")
async def readyz() -> dict[str, object]:
    checks: dict[str, object] = {}
    try:
        async with get_engine().connect() as c:
            await c.execute(text("SELECT 1"))
        checks["db"] = "ok"
    except Exception as e:  # report, don't crash
        checks["db"] = f"error: {type(e).__name__}"
    try:
        await get_redis().ping()
        checks["redis"] = "ok"
        beats = await get_redis().hgetall("workers:heartbeat")
        now = time.time()
        checks["workers"] = {k.decode(): ("ok" if now - float(v) < 30 else "stale") for k, v in beats.items()}
    except Exception as e:
        checks["redis"] = f"error: {type(e).__name__}"
    ok = checks.get("db") == "ok" and checks.get("redis") == "ok"
    return {"status": "ready" if ok else "degraded", "checks": checks}
