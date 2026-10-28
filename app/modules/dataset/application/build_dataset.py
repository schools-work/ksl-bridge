"""Queue an immutable dataset build (ml/train side assembles and freezes it)."""

from __future__ import annotations

import json
from typing import Any
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection


async def execute(
    conn: AsyncConnection, actor: UUID, name: str, version: str, purpose: str, params: dict[str, Any]
) -> UUID:
    row = (
        await conn.execute(
            text(
                "INSERT INTO ops.jobs (kind, payload, created_by) VALUES ('build_dataset', CAST(:p AS jsonb), :u) RETURNING id"
            ),
            {"p": json.dumps({"name": name, "version": version, "purpose": purpose, "params": params}), "u": actor},
        )
    ).first()
    assert row is not None
    return UUID(str(row[0]))
