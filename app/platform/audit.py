"""Append-only audit trail helper (audit.events). Used by every module for privileged actions."""

from __future__ import annotations

import json
from typing import Any
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection


async def record(
    conn: AsyncConnection,
    action: str,
    target: str | None = None,
    *,
    actor_user_id: UUID | None = None,
    actor_device_id: UUID | None = None,
    meta: dict[str, Any] | None = None,
) -> None:
    await conn.execute(
        text(
            "INSERT INTO audit.events (actor_user_id, actor_device_id, action, target, meta) "
            "VALUES (:u, :d, :a, :t, CAST(:m AS jsonb))"
        ),
        {"u": actor_user_id, "d": actor_device_id, "a": action, "t": target, "m": json.dumps(meta or {}, default=str)},
    )
