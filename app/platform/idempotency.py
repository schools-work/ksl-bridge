"""Idempotency keys (ops.idempotency_keys): replay the stored response for a repeated request."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from .errors import Conflict


def request_hash(payload: Any) -> bytes:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).digest()


async def lookup(conn: AsyncConnection, scope: str, key: str, payload: Any) -> dict[str, Any] | None:
    row = (
        (
            await conn.execute(
                text(
                    "SELECT request_hash, response FROM ops.idempotency_keys WHERE scope = :s AND key = :k AND expires_at > now()"
                ),
                {"s": scope, "k": key},
            )
        )
        .mappings()
        .first()
    )
    if row is None:
        return None
    if bytes(row["request_hash"]) != request_hash(payload):
        raise Conflict("idempotency key reused with a different request")
    return dict(row["response"]) if row["response"] is not None else None


async def store(conn: AsyncConnection, scope: str, key: str, payload: Any, response: dict[str, Any]) -> None:
    await conn.execute(
        text(
            "INSERT INTO ops.idempotency_keys (scope, key, request_hash, response) VALUES (:s, :k, :h, CAST(:r AS jsonb)) "
            "ON CONFLICT (scope, key) DO NOTHING"
        ),
        {"s": scope, "k": key, "h": request_hash(payload), "r": json.dumps(response, default=str)},
    )
