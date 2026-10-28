"""Schedule a per-user LoRA fine-tune once enough confirmed utterances exist (Guide Part V §6)."""

from __future__ import annotations

import json
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

MIN_CONFIRMED = 200


async def maybe_schedule(conn: AsyncConnection, user_id: UUID) -> UUID | None:
    if not (await conn.execute(text("SELECT privacy.has_consent(:u, 'personalisation')"), {"u": user_id})).scalar_one():
        return None
    n = (
        await conn.execute(
            text("SELECT count(*) FROM feedback.events WHERE user_id = :u AND kind = 'confirm'"), {"u": user_id}
        )
    ).scalar_one()
    if int(n) < MIN_CONFIRMED:
        return None
    row = (
        await conn.execute(
            text(
                "INSERT INTO ops.jobs (kind, payload, created_by) VALUES ('train_lora', CAST(:p AS jsonb), :u) RETURNING id"
            ),
            {"p": json.dumps({"user_id": str(user_id)}), "u": user_id},
        )
    ).first()
    return UUID(str(row[0])) if row else None
