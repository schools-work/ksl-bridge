"""Derive knowledge.venue_vocabulary from the venue's menu chunks (RAG → recognition biasing)."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection


async def execute(conn: AsyncConnection, venue_id: UUID) -> int:
    rows = (
        await conn.execute(
            text(
                "SELECT c.id, c.meta->'item'->>'name' AS name, c.meta->'item'->>'category' AS category "
                "FROM knowledge.chunks c JOIN knowledge.documents d ON d.id = c.document_id "
                "WHERE d.venue_id = :v AND d.kind = 'menu' AND d.status = 'active' AND c.meta ? 'item'"
            ),
            {"v": venue_id},
        )
    ).all()
    n = 0
    for chunk_id, name, category in rows:
        if not name:
            continue
        gloss = (
            await conn.execute(
                text("SELECT gloss FROM lexicon.translations WHERE language = 'en' AND phrase = lower(:p) LIMIT 1"),
                {"p": name},
            )
        ).scalar_one_or_none()
        await conn.execute(
            text(
                "INSERT INTO knowledge.venue_vocabulary (venue_id, term, gloss, category, bias_nats, source_chunk_id) "
                "VALUES (:v, :t, :g, :c, 0.8, :ch) ON CONFLICT (venue_id, term) DO UPDATE SET gloss = EXCLUDED.gloss, "
                "category = EXCLUDED.category, source_chunk_id = EXCLUDED.source_chunk_id"
            ),
            {"v": venue_id, "t": name, "g": gloss, "c": (category or "ITEM").upper(), "ch": chunk_id},
        )
        n += 1
    return n
