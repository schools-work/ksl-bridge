"""Admin queries over audit.events."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection


async def execute(
    conn: AsyncConnection, actor: UUID | None, action: str | None, limit: int = 100
) -> list[dict[str, Any]]:
    rows = await conn.execute(
        text(
            "SELECT id, at, actor_user_id, actor_device_id, action, target, meta FROM audit.events "
            "WHERE (CAST(:a AS uuid) IS NULL OR actor_user_id = CAST(:a AS uuid)) AND (CAST(:act AS text) IS NULL OR action = :act) "
            "ORDER BY at DESC LIMIT :n"
        ),
        {"a": str(actor) if actor else None, "act": action, "n": limit},
    )
    return [dict(r) for r in rows.mappings()]
