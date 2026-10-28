"""Venue staff management."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncConnection

from app.platform.errors import Forbidden

from ..infrastructure.repositories import SqlVenueRepo


async def add(conn: AsyncConnection, actor_id: UUID, venue_id: UUID, user_id: UUID, role: str) -> None:
    repo = SqlVenueRepo(conn)
    if not await repo.is_staff(venue_id, actor_id, "manager"):
        raise Forbidden("only venue managers can add staff")
    await repo.add_staff(venue_id, user_id, role)


async def venues_of(conn: AsyncConnection, user_id: UUID) -> list[dict[str, object]]:
    return await SqlVenueRepo(conn).venues_of(user_id)
