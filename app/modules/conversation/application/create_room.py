"""Create a room (by a user or a device); the creator joins as its first member."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncConnection

from app.platform.errors import Invalid
from app.platform.principal import Principal

from ..domain.member import DEFAULT_MODALITIES, ROLES
from ..domain.room import qr_payload
from ..infrastructure.repositories import SqlRoomRepo


async def execute(
    conn: AsyncConnection,
    p: Principal,
    role: str,
    venue_id: UUID | None,
    store_transcript: bool,
    context_profile: str | None = None,
) -> dict[str, Any]:
    if role not in ROLES:
        raise Invalid("unknown role")
    repo = SqlRoomRepo(conn)
    if p.device_id is not None:
        venue_id = p.venue_id
        role = "device"
    profile = context_profile or (await repo.venue_type(venue_id) if venue_id else "general")
    room = await repo.create(p, venue_id, profile, store_transcript)
    mi, mo = DEFAULT_MODALITIES[role]
    member_id = await repo.add_member(room["id"], p, role, mi, mo)
    return {
        "room_id": room["id"],
        "join_code": room["join_code"],
        "qr_payload": qr_payload(room["join_code"]),
        "member_id": member_id,
        "context_profile": profile,
    }
