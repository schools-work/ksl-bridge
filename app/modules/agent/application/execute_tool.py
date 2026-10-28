"""Tool execution with idempotency, confirmation and timeouts, logged in agent.tool_calls (DB-guarded)."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from ..infrastructure.tool_registry import resolve


class SessionToolExecutor:
    def __init__(self, conn: AsyncConnection, session_id: UUID, venue_id: UUID | None, room_id: UUID):
        self.c, self.session_id, self.venue_id, self.room_id = conn, session_id, venue_id, room_id

    async def __call__(self, name: str, args: dict[str, Any], confirmed: bool = False) -> Any:
        tool = (
            (
                await self.c.execute(
                    text(
                        "SELECT id, timeout_ms, side_effect::text, requires_confirmation "
                        "FROM agent.tools WHERE name = :n AND enabled"
                    ),
                    {"n": name},
                )
            )
            .mappings()
            .first()
        )
        if tool is None:
            return None
        key = f"{self.session_id}:{name}:{json.dumps(args, sort_keys=True)}:{uuid.uuid4().hex[:8]}"
        call_id = (
            await self.c.execute(
                text(
                    "INSERT INTO agent.tool_calls (session_id, tool_id, idempotency_key, arguments, confirmed_by_user) "
                    "VALUES (:s, :t, :k, CAST(:a AS jsonb), :conf) RETURNING id"
                ),
                {"s": self.session_id, "t": tool["id"], "k": key, "a": json.dumps(args), "conf": confirmed},
            )
        ).scalar_one()
        fn = resolve(name, self.c, self.venue_id, self.room_id)
        t0 = time.perf_counter()
        status, result, error = "succeeded", None, None
        try:
            result = await asyncio.wait_for(fn(**args), timeout=tool["timeout_ms"] / 1000)
        except TimeoutError:
            status, error = "timeout", "tool timed out"
        except Exception as e:  # tool failures never crash the conversation
            status, error = "failed", type(e).__name__
        await self.c.execute(
            text(
                "UPDATE agent.tool_calls SET status = CAST(:st AS shared.call_status), result = CAST(:r AS jsonb), "
                "error = :e, latency_ms = :l, completed_at = now() WHERE id = :id"
            ),
            {
                "st": status,
                "r": json.dumps(result, default=str),
                "e": error,
                "l": int((time.perf_counter() - t0) * 1000),
                "id": call_id,
            },
        )
        return result if status == "succeeded" else None
