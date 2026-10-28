"""/rooms, /rooms/join, /rooms/{id}, /rooms/{id}/close, /rooms/{id}/transcript"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncConnection

from app.modules.identity.api.deps import current_principal, db
from app.platform.errors import Forbidden, NotFound
from app.platform.principal import Principal

from ..application import close_room, create_room, join_room
from ..infrastructure.repositories import SqlRoomRepo

router = APIRouter(prefix="/rooms", tags=["rooms"])


class RoomIn(BaseModel):
    role: str = Field(default="signer", pattern="^(signer|speaker|device|observer)$")
    venue_id: UUID | None = None
    store_transcript: bool = False
    context_profile: str | None = Field(
        default=None, pattern="^(restaurant|hospital|bank|office|transport|education|general)$"
    )


class JoinIn(BaseModel):
    join_code: str = Field(min_length=6, max_length=8)
    role: str = Field(default="speaker", pattern="^(signer|speaker|device|observer)$")
    modality_in: str | None = Field(default=None, pattern="^(sign|speech|text|device)$")
    modality_out: str | None = Field(default=None, pattern="^(sign|speech|text|device)$")


@router.post("", status_code=201)
async def post_room(
    body: RoomIn, p: Principal = Depends(current_principal), conn: AsyncConnection = Depends(db)
) -> dict[str, Any]:
    if p.guest_id is not None:
        raise Forbidden("guests cannot create rooms")
    return await create_room.execute(conn, p, body.role, body.venue_id, body.store_transcript, body.context_profile)


@router.post("/join")
async def post_join(
    body: JoinIn, p: Principal = Depends(current_principal), conn: AsyncConnection = Depends(db)
) -> dict[str, Any]:
    return await join_room.by_code(conn, p, body.join_code, body.role, body.modality_in, body.modality_out)


@router.get("/{room_id}")
async def get_room(
    room_id: UUID, p: Principal = Depends(current_principal), conn: AsyncConnection = Depends(db)
) -> dict[str, Any]:
    state = await join_room.room_state(conn, room_id)
    if state is None:
        raise NotFound("not found or not yours")
    return state


@router.post("/{room_id}/close", status_code=204)
async def post_close(
    room_id: UUID, p: Principal = Depends(current_principal), conn: AsyncConnection = Depends(db)
) -> None:
    await close_room.execute(conn, room_id)


@router.get("/{room_id}/transcript")
async def get_transcript(
    room_id: UUID,
    limit: int = Query(50, ge=1, le=200),
    before: str | None = None,
    p: Principal = Depends(current_principal),
    conn: AsyncConnection = Depends(db),
) -> list[dict[str, Any]]:
    if await join_room.membership(conn, p, room_id) is None and not p.has_role("admin"):
        raise NotFound("not found or not yours")
    return await SqlRoomRepo(conn).transcript(room_id, limit, before)
