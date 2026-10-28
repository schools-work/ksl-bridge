"""Transactional outbox: write events in the same transaction as the state change, relay to Redis."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from .principal import SYSTEM

log = logging.getLogger(__name__)

_INSERT = text(
    "INSERT INTO ops.outbox (aggregate, aggregate_id, event_type, event_version, payload) "
    "VALUES (:agg, :aid, :type, :ver, CAST(:payload AS jsonb))"
)


async def emit(
    conn: AsyncConnection, aggregate: str, aggregate_id: str, event_type: str, payload: dict[str, Any], version: int = 1
) -> None:
    await conn.execute(
        _INSERT,
        {
            "agg": aggregate,
            "aid": aggregate_id,
            "type": event_type,
            "ver": version,
            "payload": json.dumps(payload, default=str),
        },
    )


async def relay_forever(stream: str = "events", batch: int = 100, idle_s: float = 0.5) -> None:
    """Publish unpublished outbox rows to a Redis stream. Consumers must be idempotent (event id)."""
    from .db import transaction
    from .redis import get_redis

    r = get_redis()
    claim = text(
        "SELECT id, aggregate, aggregate_id, event_type, event_version, payload FROM ops.outbox "
        "WHERE published_at IS NULL ORDER BY id LIMIT :n FOR UPDATE SKIP LOCKED"
    )
    mark = text("UPDATE ops.outbox SET published_at = now(), attempts = attempts + 1 WHERE id = ANY(:ids)")
    while True:
        try:
            async with transaction(SYSTEM) as conn:
                rows = (await conn.execute(claim, {"n": batch})).mappings().all()
                for row in rows:
                    await r.xadd(
                        stream,
                        {
                            "id": str(row["id"]),
                            "aggregate": row["aggregate"],
                            "aggregate_id": row["aggregate_id"],
                            "type": row["event_type"],
                            "version": str(row["event_version"]),
                            "payload": json.dumps(row["payload"]),
                        },
                        maxlen=100_000,
                    )
                if rows:
                    await conn.execute(mark, {"ids": [r_["id"] for r_ in rows]})
            if not rows:
                await asyncio.sleep(idle_s)
        except asyncio.CancelledError:
            raise
        except Exception:  # keep relaying; failures are retried
            log.exception("outbox relay iteration failed")
            await asyncio.sleep(2.0)
