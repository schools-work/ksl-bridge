"""Adjudication by a senior Deaf annotator; writes the gold labels onto the sample."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from app.platform.audit import record


async def execute(
    conn: AsyncConnection,
    adjudicator: UUID,
    sample_id: UUID,
    final_glosses: list[str],
    final_text_en: str | None,
    chosen_id: UUID | None,
    notes: str | None,
) -> None:
    await conn.execute(
        text(
            "INSERT INTO annotation.adjudications (sample_id, chosen_id, final_glosses, final_text_en, adjudicator_id, notes) "
            "VALUES (:s, :c, :g, :t, :a, :n) ON CONFLICT (sample_id) DO UPDATE SET chosen_id = EXCLUDED.chosen_id, "
            "final_glosses = EXCLUDED.final_glosses, final_text_en = EXCLUDED.final_text_en, "
            "adjudicator_id = EXCLUDED.adjudicator_id, notes = EXCLUDED.notes, decided_at = now()"
        ),
        {"s": sample_id, "c": chosen_id, "g": final_glosses, "t": final_text_en, "a": adjudicator, "n": notes},
    )
    await conn.execute(
        text(
            "UPDATE dataset.samples SET glosses = :g, text_en = COALESCE(:t, text_en), "
            "review_status = 'approved' WHERE id = :s"
        ),
        {"g": final_glosses, "t": final_text_en, "s": sample_id},
    )
    await conn.execute(text("UPDATE annotation.tasks SET status = 'closed' WHERE sample_id = :s"), {"s": sample_id})
    await record(conn, "annotation.adjudicate", str(sample_id), actor_user_id=adjudicator)
