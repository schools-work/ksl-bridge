"""Live models and the leaderboard."""

from __future__ import annotations

from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection


async def live(conn: AsyncConnection) -> list[dict[str, Any]]:
    rows = await conn.execute(
        text(
            "SELECT task::text, target::text, stage::text, name, version, format, is_bootstrap, "
            "started_at FROM ml.v_live_models ORDER BY task"
        )
    )
    return [dict(r) for r in rows.mappings()]


async def leaderboard(conn: AsyncConnection, task: str, metric: str) -> list[dict[str, Any]]:
    rows = await conn.execute(
        text(
            "SELECT name, version, suite, perturbation, metric, value, ci_low, ci_high, n, dataset, split, created_at "
            "FROM ml.v_leaderboard WHERE task = CAST(:t AS shared.model_task) AND metric = :m ORDER BY created_at DESC LIMIT 100"
        ),
        {"t": task, "m": metric},
    )
    return [dict(r) for r in rows.mappings()]
