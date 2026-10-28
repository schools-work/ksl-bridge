"""Active-learning queue (the learning agent's problem generator feeds on it)."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection


async def from_feedback(conn: AsyncConnection, feedback_id: UUID, reason: str, score: float) -> None:
    await conn.execute(
        text("INSERT INTO feedback.active_learning_queue (feedback_id, reason, score) VALUES (:f, :r, :s)"),
        {"f": feedback_id, "r": reason, "s": score},
    )


async def from_sample(
    conn: AsyncConnection, sample_id: UUID, reason: str, score: float, model_version_id: UUID | None = None
) -> None:
    await conn.execute(
        text(
            "INSERT INTO feedback.active_learning_queue (sample_id, reason, score, model_version_id) "
            "VALUES (:s, :r, :sc, :m)"
        ),
        {"s": sample_id, "r": reason, "sc": score, "m": model_version_id},
    )


async def promote_to_tasks(conn: AsyncConnection, limit: int = 100) -> int:
    """Turn the top open queue items that reference samples into annotation tasks."""
    res = await conn.execute(
        text(
            "WITH q AS (SELECT id, sample_id, score FROM feedback.active_learning_queue WHERE status = 'open' AND "
            "sample_id IS NOT NULL ORDER BY score DESC LIMIT :n FOR UPDATE SKIP LOCKED), "
            "ins AS (INSERT INTO annotation.tasks (sample_id, kind, priority, source_queue_id) "
            "SELECT sample_id, 'gloss', score, id FROM q RETURNING source_queue_id) "
            "UPDATE feedback.active_learning_queue SET status = 'tasked' WHERE id IN (SELECT source_queue_id FROM ins)"
        ),
        {"n": limit},
    )
    return res.rowcount or 0
