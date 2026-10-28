"""Record 👍/✏️/✗ on an utterance; corrections feed the active-learning queue."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from app.platform.metrics import FEEDBACK

from . import enqueue_active_learning


async def execute(
    conn: AsyncConnection,
    user_id: UUID,
    utterance_id: str,
    kind: str,
    corrected_glosses: list[str] | None,
    corrected_text: str | None,
) -> UUID | None:
    utt = (
        await conn.execute(
            text("SELECT id, created_at FROM conversation.utterances WHERE id = :u"), {"u": utterance_id}
        )
    ).first()
    if utt is None:
        return None  # nothing stored for this utterance (no consent / not linked)
    row = (
        await conn.execute(
            text(
                "INSERT INTO feedback.events (utterance_id, utterance_created_at, user_id, kind, corrected_glosses, corrected_text) "
                "VALUES (:u, :ts, :usr, CAST(:k AS shared.feedback_kind), :g, :t) RETURNING id"
            ),
            {"u": utt[0], "ts": utt[1], "usr": user_id, "k": kind, "g": corrected_glosses, "t": corrected_text},
        )
    ).first()
    assert row is not None
    FEEDBACK.labels(kind).inc()
    if kind in ("correct", "reject"):
        await enqueue_active_learning.from_feedback(conn, UUID(str(row[0])), "user_correction", score=1.0)
    return UUID(str(row[0]))
