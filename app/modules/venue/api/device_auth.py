"""Exchange a device API key for a short-lived device JWT (scoped to the device's venue)."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, Header
from sqlalchemy.ext.asyncio import AsyncConnection

from app.modules.identity.api.deps import db_anonymous
from app.modules.identity.infrastructure.jwt import get_jwt
from app.platform.errors import Unauthorized
from app.platform.principal import Principal

from ..infrastructure.api_keys import HmacDeviceKeys
from ..infrastructure.repositories import SqlVenueRepo

router = APIRouter(tags=["devices"])


@router.post("/devices/token")
async def device_token(
    x_device_key: str = Header(..., alias="X-Device-Key"),
    x_app_version: str | None = Header(default=None, alias="X-App-Version"),
    conn: AsyncConnection = Depends(db_anonymous),
) -> dict[str, object]:
    keys = HmacDeviceKeys()
    prefix = keys.prefix_of(x_device_key)
    if prefix is None:
        raise Unauthorized("invalid device key")
    repo = SqlVenueRepo(conn)
    dev = await repo.device_by_prefix(prefix)
    if dev is None or dev["revoked_at"] is not None or not keys.verify(x_device_key, bytes(dev["api_key_hash"])):  # type: ignore[arg-type]
        raise Unauthorized("invalid device key")
    await repo.touch_device(UUID(str(dev["id"])), x_app_version)
    p = Principal(device_id=UUID(str(dev["id"])), venue_id=UUID(str(dev["venue_id"])), roles=frozenset({"device"}))
    return {"access_token": get_jwt().issue(p, ttl=3600), "expires_in": 3600, "venue_id": dev["venue_id"]}
