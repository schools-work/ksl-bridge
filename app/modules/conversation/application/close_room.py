"""Close a room; members are marked as left. Retention applies to stored utterances."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncConnection

from app.platform.errors import NotFound
from app.platform.outbox import emit

from ..infrastructure.repositories import SqlRoomRepo


async def execute(conn: AsyncConnection, room_id: UUID) -> None:
    if not await SqlRoomRepo(conn).close(room_id):
        raise NotFound("room not found")
    await emit(conn, "room", str(room_id), "RoomClosed", {"room_id": str(room_id)})
