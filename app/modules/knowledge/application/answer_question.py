"""Grounded, extractive answering with grading, one rewrite-and-retry, and 'say so' when nothing qualifies."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from ..domain.grading import best_sentence, grade, key_entities, rewrite
from ..infrastructure.repositories import SqlKnowledgeRepo


async def execute(
    conn: AsyncConnection,
    venue_id: UUID | None,
    question: str,
    room_id: UUID | None = None,
    qvec: list[float] | None = None,
) -> dict[str, Any]:
    repo = SqlKnowledgeRepo(conn)
    ents = key_entities(question)
    hits = await repo.search(venue_id, question, qvec)
    good = [h for h in hits if grade(ents, h["content"], h["score"])]
    rewritten = None
    if not good:
        rewritten = rewrite(question, await repo.synonyms())
        if rewritten:
            hits = await repo.search(venue_id, rewritten, qvec)
            good = [h for h in hits if grade(key_entities(rewritten), h["content"], h["score"])]
    answer = best_sentence(question, good[0]["content"]) if good else None
    await conn.execute(
        text(
            "INSERT INTO knowledge.retrieval_logs (room_id, venue_id, query, rewritten_query, chunk_ids, scores, "
            "graded_relevant, answer_chunk_id, answered) VALUES (:r, :v, :q, :rq, :ids, :sc, :g, :a, :ans)"
        ),
        {
            "r": room_id,
            "v": venue_id,
            "q": question,
            "rq": rewritten,
            "ids": [h["chunk_id"] for h in hits],
            "sc": [float(h["score"]) for h in hits],
            "g": bool(good),
            "a": good[0]["chunk_id"] if good else None,
            "ans": answer is not None,
        },
    )
    return {"answer": answer, "chunk_id": good[0]["chunk_id"] if good else None, "escalate": answer is None}
