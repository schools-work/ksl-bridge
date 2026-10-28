"""'Signs we need more of': rank glosses by error mass and scarcity (learning-agent problem generator)."""

from __future__ import annotations

from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection


async def execute(conn: AsyncConnection, limit: int = 50) -> list[dict[str, Any]]:
    rows = await conn.execute(
        text(
            "WITH corr AS (SELECT unnest(corrected_glosses) AS gloss, count(*) AS corrections FROM feedback.events "
            "WHERE kind = 'correct' AND created_at > now() - interval '30 days' GROUP BY 1), "
            "have AS (SELECT unnest(glosses) AS gloss, count(*) AS samples FROM dataset.samples WHERE excluded_at IS NULL "
            "AND review_status = 'approved' GROUP BY 1) "
            "SELECT g.gloss, COALESCE(c.corrections, 0) AS corrections, COALESCE(h.samples, 0) AS samples, "
            "(COALESCE(c.corrections, 0) + 1.0) / (COALESCE(h.samples, 0) + 1.0) AS need "
            "FROM lexicon.glosses g LEFT JOIN corr c ON c.gloss = g.gloss LEFT JOIN have h ON h.gloss = g.gloss "
            "WHERE g.status <> 'rejected' ORDER BY need DESC LIMIT :n"
        ),
        {"n": limit},
    )
    return [dict(r) for r in rows.mappings()]
