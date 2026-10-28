"""Create a venue; the creator becomes its manager and gains the operator role."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncConnection

from app.platform.audit import record
from app.platform.errors import Invalid

from ..domain.venue import VENUE_TYPES
from ..infrastructure.repositories import SqlVenueRepo


async def execute(
    conn: AsyncConnection, owner_id: UUID, name: str, venue_type: str, county_code: str | None = None
) -> UUID:
    if venue_type not in VENUE_TYPES:
        raise Invalid("unknown venue type")
    repo = SqlVenueRepo(conn)
    vid = await repo.create(name, venue_type, owner_id, county_code)
    await repo.add_staff(vid, owner_id, "manager")
    await record(conn, "venue.create", str(vid), actor_user_id=owner_id)
    return vid
