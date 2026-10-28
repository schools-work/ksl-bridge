"""Join a room by code (users and devices; guests use /auth/guest) and membership checks."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncConnection

from app.platform.errors import Invalid, NotFound
from app.platform.principal import Principal

from ..domain.member import DEFAULT_MODALITIES, ROLES
from ..domain.room import normalize_code
from ..infrastructure.repositories import SqlRoomRepo


async def by_code(
    conn: AsyncConnection,
    p: Principal,
    join_code: str,
    role: str,
    modality_in: str | None = None,
    modality_out: str | None = None,
) -> dict[str, Any]:
    if role not in ROLES:
        raise Invalid("unknown role")
    repo = SqlRoomRepo(conn)
    room = await repo.find_by_code(normalize_code(join_code))
    if room is None or room["status"] != "open":
        raise NotFound("room not found or closed")
    existing = await repo.member_of(room["room_id"], p)
    if existing:
        return {"room_id": room["room_id"], "member_id": existing["id"], "already_member": True}
    dmi, dmo = DEFAULT_MODALITIES[role]
    member_id = await repo.add_member(room["room_id"], p, role, modality_in or dmi, modality_out or dmo)
    return {"room_id": room["room_id"], "member_id": member_id, "already_member": False}


async def membership(conn: AsyncConnection, p: Principal, room_id: UUID) -> dict[str, Any] | None:
    """The caller's live membership in the room (RLS-checked), or None."""
    return await SqlRoomRepo(conn).member_of(room_id, p)


async def room_state(conn: AsyncConnection, room_id: UUID) -> dict[str, Any] | None:
    repo = SqlRoomRepo(conn)
    room = await repo.get(room_id)
    if room is None:
        return None
    return {"room": room, "members": await repo.members(room_id)}
