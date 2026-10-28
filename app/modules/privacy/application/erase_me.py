"""Right to erasure as anonymisation (privacy.erase_user) + object deletion job."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection


async def execute(conn: AsyncConnection, user_id: UUID, actor_id: UUID) -> dict[str, Any]:
    req = (
        await conn.execute(
            text("INSERT INTO privacy.erasure_requests (user_id, status) VALUES (:u, 'running') RETURNING id"),
            {"u": user_id},
        )
    ).first()
    assert req is not None
    report = (await conn.execute(text("SELECT privacy.erase_user(:u, :a)"), {"u": user_id, "a": actor_id})).scalar_one()
    await conn.execute(
        text(
            "UPDATE privacy.erasure_requests SET status = 'succeeded', completed_at = now(), "
            "report = CAST(:r AS jsonb) WHERE id = :id"
        ),
        {"r": __import__("json").dumps(report), "id": req[0]},
    )
    return dict(report)
