"""Partition maintenance for partitioned tables (run monthly by the scheduler)."""

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

PARTITIONED = ("conversation.utterances", "inference.requests", "audit.events")


async def ensure_partitions(conn: AsyncConnection, months_ahead: int = 3) -> None:
    for table in PARTITIONED:
        await conn.execute(
            text("SELECT ops.create_monthly_partitions(CAST(:t AS regclass), now()::date, :m)"),
            {"t": table, "m": months_ahead},
        )
