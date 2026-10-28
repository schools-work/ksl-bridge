"""Roll back = re-promote the previous production version (its gates are already recorded)."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from app.platform.errors import NotFound

from . import promote


async def execute(conn: AsyncConnection, task: str, target: str, actor: UUID) -> UUID:
    rows = (
        await conn.execute(
            text(
                "SELECT model_version_id FROM ml.deployments WHERE task = CAST(:t AS shared.model_task) AND "
                "target = CAST(:g AS shared.deploy_target) AND stage = 'production' ORDER BY started_at DESC LIMIT 2"
            ),
            {"t": task, "g": target},
        )
    ).all()
    if len(rows) < 2:
        raise NotFound("no previous production version to roll back to")
    return await promote.execute(conn, UUID(str(rows[1][0])), target, actor)
