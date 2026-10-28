"""Search the KSL dictionary by English/Swahili phrase (trigram) or gloss prefix."""

from __future__ import annotations

from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection


async def execute(conn: AsyncConnection, q: str, limit: int = 20) -> list[dict[str, Any]]:
    rows = await conn.execute(
        text(
            "SELECT g.gloss, g.categories, g.status::text AS status, g.is_critical, "
            "array_agg(DISTINCT t.phrase) FILTER (WHERE t.phrase IS NOT NULL) AS phrases, "
            "max(greatest(similarity(t.phrase, :q), CASE WHEN g.gloss ILIKE :pfx THEN 1 ELSE 0 END)) AS score "
            "FROM lexicon.glosses g LEFT JOIN lexicon.translations t ON t.gloss = g.gloss "
            "WHERE g.gloss ILIKE :pfx OR t.phrase % :q GROUP BY g.gloss ORDER BY score DESC LIMIT :n"
        ),
        {"q": q.lower(), "pfx": f"{q.upper()}%", "n": limit},
    )
    return [dict(r) for r in rows.mappings()]


async def pose_uri(conn: AsyncConnection, gloss: str) -> str | None:
    row = (
        await conn.execute(text("SELECT pose_uri FROM lexicon.sign_clips WHERE gloss = :g AND is_medoid"), {"g": gloss})
    ).first()
    return row[0] if row else None
