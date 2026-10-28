"""/venues, /venues/{id}/devices, /venues/{id}/staff"""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncConnection

from app.modules.identity.api.deps import db, require_user
from app.platform.principal import Principal

from ..application import create_venue, register_device, rotate_key, staff

router = APIRouter(prefix="/venues", tags=["venues"])


class VenueIn(BaseModel):
    name: str = Field(min_length=2, max_length=120)
    venue_type: str
    county_code: str | None = Field(default=None, pattern=r"^\d{3}$")


class DeviceIn(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    kind: str = Field(pattern="^(robot|kiosk|tablet)$")
    capabilities: dict[str, bool] = Field(default_factory=dict)


class StaffIn(BaseModel):
    user_id: UUID
    role: str = Field(pattern="^(manager|staff)$")


@router.post("", status_code=201)
async def post_venue(
    body: VenueIn, p: Principal = Depends(require_user), conn: AsyncConnection = Depends(db)
) -> dict[str, UUID]:
    assert p.user_id
    return {"venue_id": await create_venue.execute(conn, p.user_id, body.name, body.venue_type, body.county_code)}


@router.get("/mine")
async def my_venues(
    p: Principal = Depends(require_user), conn: AsyncConnection = Depends(db)
) -> list[dict[str, object]]:
    assert p.user_id
    return await staff.venues_of(conn, p.user_id)


@router.post("/{venue_id}/devices", status_code=201)
async def post_device(
    venue_id: UUID, body: DeviceIn, p: Principal = Depends(require_user), conn: AsyncConnection = Depends(db)
) -> dict[str, object]:
    assert p.user_id
    return await register_device.execute(conn, p.user_id, venue_id, body.name, body.kind, body.capabilities)


@router.post("/{venue_id}/devices/{device_id}/rotate-key")
async def post_rotate(
    venue_id: UUID, device_id: UUID, p: Principal = Depends(require_user), conn: AsyncConnection = Depends(db)
) -> dict[str, object]:
    assert p.user_id
    return await rotate_key.execute(conn, p.user_id, venue_id, device_id)


@router.post("/{venue_id}/staff", status_code=204)
async def post_staff(
    venue_id: UUID, body: StaffIn, p: Principal = Depends(require_user), conn: AsyncConnection = Depends(db)
) -> None:
    assert p.user_id
    await staff.add(conn, p.user_id, venue_id, body.user_id, body.role)
