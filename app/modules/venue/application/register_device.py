"""Register a robot/kiosk. The API key is returned ONCE and stored only as HMAC(pepper, key)."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncConnection

from app.platform.audit import record
from app.platform.errors import Forbidden

from ..infrastructure.api_keys import HmacDeviceKeys
from ..infrastructure.repositories import SqlVenueRepo


async def execute(
    conn: AsyncConnection, actor_id: UUID, venue_id: UUID, name: str, kind: str, capabilities: dict[str, bool]
) -> dict[str, object]:
    repo = SqlVenueRepo(conn)
    if not await repo.is_staff(venue_id, actor_id, "manager"):
        raise Forbidden("only venue managers can register devices")
    key, prefix, digest = HmacDeviceKeys().new_key()
    device_id = await repo.insert_device(venue_id, name, kind, prefix, digest, capabilities)
    await record(conn, "venue.device_register", str(device_id), actor_user_id=actor_id)
    return {"device_id": device_id, "api_key": key, "note": "store this key now; it will not be shown again"}
