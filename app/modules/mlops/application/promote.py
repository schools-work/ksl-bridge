"""Promote through ml.promote_model (refuses unless every mandatory gate has passed)."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncConnection

from app.platform.errors import Conflict


async def execute(
    conn: AsyncConnection,
    model_id: UUID,
    target: str,
    actor: UUID,
    decoder_id: UUID | None = None,
    min_app: str | None = None,
) -> UUID:
    try:
        dep = (
            await conn.execute(
                text("SELECT ml.promote_model(:m, CAST(:t AS shared.deploy_target), :a, :d, :v)"),
                {"m": model_id, "t": target, "a": actor, "d": decoder_id, "v": min_app},
            )
        ).scalar_one()
    except DBAPIError as e:
        raise Conflict(str(e.orig).split("\n")[0]) from e
    return UUID(str(dep))
