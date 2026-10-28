"""Record a release-gate result (human_review and licence gates are recorded by people)."""

from __future__ import annotations

import json
from typing import Any
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection


async def execute(
    conn: AsyncConnection,
    model_id: UUID,
    gate: str,
    passed: bool,
    details: dict[str, Any],
    actor: UUID,
    evaluation_id: UUID | None = None,
) -> None:
    await conn.execute(
        text(
            "INSERT INTO ml.release_gates (model_version_id, gate, passed, details, evaluation_id, checked_by) "
            "VALUES (:m, :g, :p, CAST(:d AS jsonb), :e, :u) ON CONFLICT (model_version_id, gate) DO UPDATE SET "
            "passed = EXCLUDED.passed, details = EXCLUDED.details, evaluation_id = EXCLUDED.evaluation_id, "
            "checked_by = EXCLUDED.checked_by, checked_at = now()"
        ),
        {"m": model_id, "g": gate, "p": passed, "d": json.dumps(details), "e": evaluation_id, "u": actor},
    )
