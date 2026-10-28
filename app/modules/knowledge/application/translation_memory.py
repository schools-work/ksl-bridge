"""Translation memory: verified gloss⇄text pairs reused before any model guess (high precision)."""

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection


async def lookup(conn: AsyncConnection, glosses: list[str], domain: str) -> str | None:
    row = (
        await conn.execute(
            text(
                "UPDATE knowledge.translation_memory SET uses = uses + 1 WHERE id = (SELECT id FROM knowledge.translation_memory "
                "WHERE gloss_key = :k AND domain IN (CAST(:d AS shared.venue_type), 'general') ORDER BY (domain = "
                "CAST(:d AS shared.venue_type)) DESC, uses DESC LIMIT 1) RETURNING text_en"
            ),
            {"k": " ".join(glosses), "d": domain},
        )
    ).first()
    return row[0] if row else None


async def add(
    conn: AsyncConnection, glosses: list[str], text_en: str, domain: str, source: str, verified_by: str | None = None
) -> None:
    await conn.execute(
        text(
            "INSERT INTO knowledge.translation_memory (glosses, text_en, domain, source, verified_by) "
            "VALUES (:g, :t, CAST(:d AS shared.venue_type), :s, :v) ON CONFLICT (gloss_key, domain, text_en) DO NOTHING"
        ),
        {"g": glosses, "t": text_en, "d": domain, "s": source, "v": verified_by},
    )
