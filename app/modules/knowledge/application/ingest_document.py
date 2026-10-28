"""Versioned, atomic (re)ingestion of a venue document: chunk → (embed) → store; old version superseded."""

from __future__ import annotations

import hashlib
import json
from typing import Any
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from app.platform.outbox import emit

from ..domain.chunking import markdown_sections, menu_rows, recursive_split


def chunk_document(kind: str, content: str | list[dict[str, Any]]) -> list[tuple[str, str, dict[str, Any]]]:
    if kind == "menu" and isinstance(content, list):
        return [("Menu", c, {"item": r}) for c, r in zip(menu_rows(content), content, strict=True)]
    assert isinstance(content, str)
    out: list[tuple[str, str, dict[str, Any]]] = []
    for section, body in markdown_sections(content) or [("", content)]:
        out.extend((section, c, {}) for c in recursive_split(body, 500, 50))
    return out


async def execute(
    conn: AsyncConnection,
    venue_id: UUID | None,
    kind: str,
    title: str,
    content: str | list[dict[str, Any]],
    actor: UUID,
    embed: Any | None = None,
) -> UUID:
    digest = hashlib.sha256(json.dumps(content, sort_keys=True).encode()).digest()
    prev = (
        await conn.execute(
            text(
                "SELECT id, version FROM knowledge.documents WHERE venue_id IS NOT DISTINCT FROM :v AND title = :t "
                "AND status = 'active' FOR UPDATE"
            ),
            {"v": venue_id, "t": title},
        )
    ).first()
    if prev:
        await conn.execute(text("UPDATE knowledge.documents SET status = 'superseded' WHERE id = :id"), {"id": prev[0]})
    row = (
        await conn.execute(
            text(
                "INSERT INTO knowledge.documents (venue_id, kind, title, content_sha256, version, created_by) "
                "VALUES (:v, :k, :t, :h, :ver, :u) RETURNING id"
            ),
            {"v": venue_id, "k": kind, "t": title, "h": digest, "ver": (prev[1] + 1) if prev else 1, "u": actor},
        )
    ).first()
    assert row is not None
    doc_id = UUID(str(row[0]))
    chunks = chunk_document(kind, content)
    vectors = await embed([c for _, c, _ in chunks]) if embed else [None] * len(chunks)
    for i, ((section, body, meta), vec) in enumerate(zip(chunks, vectors, strict=True)):
        await conn.execute(
            text(
                "INSERT INTO knowledge.chunks (document_id, section_path, ordinal, content, embedding, meta) "
                "VALUES (:d, :s, :o, :c, CAST(:e AS vector), CAST(:m AS jsonb))"
            ),
            {
                "d": doc_id,
                "s": section,
                "o": i,
                "c": body,
                "e": str(list(vec)) if vec is not None else None,
                "m": json.dumps(meta, default=str),
            },
        )
    await emit(
        conn,
        "knowledge.document",
        str(doc_id),
        "DocumentReindexed",
        {"venue_id": str(venue_id) if venue_id else None, "chunks": len(chunks)},
    )
    return doc_id
