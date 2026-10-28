"""Queue the medoid-selection job (ml/train/select_medoids.py) that picks each gloss's dictionary clip."""

from __future__ import annotations

import json
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection


async def execute(conn: AsyncConnection, actor: UUID, glosses: list[str] | None = None) -> UUID:
    row = (
        await conn.execute(
            text(
                "INSERT INTO ops.jobs (kind, payload, created_by) VALUES ('select_medoids', CAST(:p AS jsonb), :u) RETURNING id"
            ),
            {"p": json.dumps({"glosses": glosses}), "u": actor},
        )
    ).first()
    assert row is not None
    return UUID(str(row[0]))
