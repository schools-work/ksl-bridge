"""Only Deaf reviewers approve glosses, clips and production rules (MASTER_PROMPT §7)."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from app.platform.audit import record
from app.platform.errors import NotFound
from app.platform.principal import Principal


async def execute(conn: AsyncConnection, p: Principal, gloss: str, approve: bool, notes: str | None = None) -> None:
    assert p.user_id and p.has_role("deaf_reviewer")
    res = await conn.execute(
        text(
            "UPDATE lexicon.glosses SET status = CAST(:s AS shared.review_status), approved_by = :u, "
            "notes = COALESCE(:n, notes) WHERE gloss = :g"
        ),
        {"s": "approved" if approve else "rejected", "u": p.user_id, "n": notes, "g": gloss},
    )
    if not res.rowcount:
        raise NotFound("gloss not found")
    await record(conn, "lexicon.review", gloss, actor_user_id=p.user_id, meta={"approved": approve})


async def approve_clip(conn: AsyncConnection, p: Principal, clip_id: UUID) -> None:
    assert p.user_id and p.has_role("deaf_reviewer")
    await conn.execute(
        text("UPDATE lexicon.sign_clips SET approved_by = :u WHERE id = :c"), {"u": p.user_id, "c": clip_id}
    )
