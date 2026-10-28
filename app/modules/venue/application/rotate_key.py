"""Rotate a device API key (old key stops working immediately)."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncConnection

from app.platform.audit import record
from app.platform.errors import Forbidden, NotFound

from ..infrastructure.api_keys import HmacDeviceKeys
from ..infrastructure.repositories import SqlVenueRepo


async def execute(conn: AsyncConnection, actor_id: UUID, venue_id: UUID, device_id: UUID) -> dict[str, object]:
    repo = SqlVenueRepo(conn)
    if not await repo.is_staff(venue_id, actor_id, "manager"):
        raise Forbidden("only venue managers can rotate keys")
    key, prefix, digest = HmacDeviceKeys().new_key()
    if not await repo.replace_device_key(venue_id, device_id, prefix, digest):
        raise NotFound("device not found")
    await record(conn, "venue.device_rotate_key", str(device_id), actor_user_id=actor_id)
    return {"device_id": device_id, "api_key": key}
