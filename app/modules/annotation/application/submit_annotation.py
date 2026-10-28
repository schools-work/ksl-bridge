"""Submit an annotation; validates glosses against conventions; creates a review task for a Deaf reviewer."""

from __future__ import annotations

import json
import re
from typing import Any
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from app.platform.errors import Forbidden, Invalid

GLOSS_RE = re.compile(r"^[A-Z0-9][A-Z0-9:_.\-+]*$")  # docs/gloss_conventions.md


async def execute(conn: AsyncConnection, annotator: UUID, task_id: UUID, body: dict[str, Any]) -> UUID:
    task = (
        (
            await conn.execute(
                text("SELECT sample_id, kind, assigned_to FROM annotation.tasks WHERE id = :t FOR UPDATE"),
                {"t": task_id},
            )
        )
        .mappings()
        .first()
    )
    if task is None or task["assigned_to"] != annotator:
        raise Forbidden("task is not assigned to you")
    glosses = body.get("glosses")
    if glosses is not None:
        bad = [g for g in glosses if not GLOSS_RE.match(g)]
        if bad:
            raise Invalid(f"glosses violate conventions: {bad}")
    times = body.get("gloss_times")
    if times is not None and (glosses is None or len(times) != len(glosses)):
        raise Invalid("gloss_times must align with glosses")
    ranges = [f"[{a},{b})" for a, b in times] if times else None
    row = (
        await conn.execute(
            text(
                "INSERT INTO annotation.annotations (task_id, sample_id, annotator_id, glosses, gloss_times, text_en, text_sw, "
                "non_manual, self_confidence, time_spent_s) VALUES (:t, :s, :a, :g, CAST(:gt AS int4range[]), :en, :sw, "
                "CAST(:nm AS jsonb), :sc, :ts) RETURNING id"
            ),
            {
                "t": task_id,
                "s": task["sample_id"],
                "a": annotator,
                "g": glosses,
                "gt": ranges,
                "en": body.get("text_en"),
                "sw": body.get("text_sw"),
                "nm": json.dumps(body.get("non_manual")) if body.get("non_manual") else None,
                "sc": body.get("self_confidence"),
                "ts": body.get("time_spent_s"),
            },
        )
    ).first()
    assert row is not None
    await conn.execute(text("UPDATE annotation.tasks SET status = 'submitted' WHERE id = :t"), {"t": task_id})
    if task["kind"] == "gloss":
        await conn.execute(
            text(
                "INSERT INTO annotation.tasks (sample_id, kind, priority, required_role) "
                "VALUES (:s, 'review', 0.9, 'deaf_reviewer')"
            ),
            {"s": task["sample_id"]},
        )
    return UUID(str(row[0]))
