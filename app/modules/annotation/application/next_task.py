"""Assign the highest-priority open task the annotator may do (SKIP LOCKED: no double assignment)."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection


async def execute(
    conn: AsyncConnection, annotator: UUID, roles: frozenset[str], kind: str | None = None
) -> dict[str, Any] | None:
    row = (
        (
            await conn.execute(
                text(
                    "WITH t AS (SELECT id FROM annotation.tasks WHERE status = 'open' AND (CAST(:k AS text) IS NULL OR kind = :k) "
                    "AND required_role::text = ANY(:roles) "
                    "AND NOT EXISTS (SELECT 1 FROM annotation.annotations a WHERE a.task_id = tasks.id AND a.annotator_id = :u) "
                    "ORDER BY priority DESC, created_at FOR UPDATE SKIP LOCKED LIMIT 1) "
                    "UPDATE annotation.tasks SET status = 'assigned', assigned_to = :u, assigned_at = now() "
                    "FROM t WHERE annotation.tasks.id = t.id "
                    "RETURNING annotation.tasks.id, annotation.tasks.sample_id, annotation.tasks.kind"
                ),
                {"k": kind, "u": annotator, "roles": list(roles)},
            )
        )
        .mappings()
        .first()
    )
    if row is None:
        return None
    sample = (
        (
            await conn.execute(
                text(
                    "SELECT id, kind::text AS kind, domain::text AS domain, video_uri, landmarks_uri, text_en, glosses "
                    "FROM dataset.samples WHERE id = :s"
                ),
                {"s": row["sample_id"]},
            )
        )
        .mappings()
        .first()
    )
    return {"task": dict(row), "sample": dict(sample) if sample else None}
