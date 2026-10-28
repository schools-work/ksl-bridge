"""/rooms/{id}/agent (start a service agent in a room), escalation acknowledgements."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from app.modules.identity.api.deps import current_principal, db, require_user
from app.platform.errors import NotFound
from app.platform.principal import Principal

from ..application import escalate

router = APIRouter(tags=["agent"])


@router.post("/rooms/{room_id}/agent", status_code=201)
async def start_agent(
    room_id: UUID, p: Principal = Depends(current_principal), conn: AsyncConnection = Depends(db)
) -> dict[str, Any]:
    room = (
        await conn.execute(
            text("SELECT venue_id, context_profile::text FROM conversation.rooms WHERE id = :r"), {"r": room_id}
        )
    ).first()
    if room is None:
        raise NotFound("room not found or not yours")
    d = (
        await conn.execute(
            text(
                "SELECT id FROM agent.definitions WHERE venue_type = CAST(:t AS shared.venue_type) "
                "AND enabled ORDER BY version DESC LIMIT 1"
            ),
            {"t": room[1]},
        )
    ).first()
    if d is None:
        raise NotFound(f"no agent definition for venue type {room[1]}")
    sid = (
        await conn.execute(
            text(
                "INSERT INTO agent.sessions (room_id, definition_id, venue_id) VALUES (:r, :d, :v) "
                "ON CONFLICT (room_id) WHERE goal_status = 'active' DO UPDATE SET frame = agent.sessions.frame RETURNING id"
            ),
            {"r": room_id, "d": d[0], "v": room[0]},
        )
    ).scalar_one()
    return {"agent_session_id": sid}


@router.post("/escalations/{escalation_id}/ack", status_code=204)
async def ack(escalation_id: UUID, p: Principal = Depends(require_user), conn: AsyncConnection = Depends(db)) -> None:
    assert p.user_id
    await escalate.acknowledge(conn, escalation_id, p.user_id)


@router.post("/escalations/{escalation_id}/resolve", status_code=204)
async def resolve(
    escalation_id: UUID, p: Principal = Depends(require_user), conn: AsyncConnection = Depends(db)
) -> None:
    await escalate.resolve(conn, escalation_id)
