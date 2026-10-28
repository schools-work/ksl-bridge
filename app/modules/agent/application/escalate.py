"""Human in the loop: open/acknowledge/resolve escalations."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection


async def open_(conn: AsyncConnection, session_id: UUID, reason: str) -> UUID:
    row = (
        await conn.execute(
            text("INSERT INTO agent.escalations (session_id, reason) VALUES (:s, :r) RETURNING id"),
            {"s": session_id, "r": reason},
        )
    ).first()
    assert row is not None
    return UUID(str(row[0]))


async def acknowledge(conn: AsyncConnection, escalation_id: UUID, staff_id: UUID) -> None:
    await conn.execute(
        text("UPDATE agent.escalations SET acknowledged_by = :u, acknowledged_at = now() WHERE id = :e"),
        {"u": staff_id, "e": escalation_id},
    )


async def resolve(conn: AsyncConnection, escalation_id: UUID) -> None:
    await conn.execute(text("UPDATE agent.escalations SET resolved_at = now() WHERE id = :e"), {"e": escalation_id})
