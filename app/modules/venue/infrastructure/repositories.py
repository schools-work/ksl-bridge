"""SQL repositories for the venue schema."""

from __future__ import annotations

import json
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection


class SqlVenueRepo:
    def __init__(self, conn: AsyncConnection):
        self.c = conn

    async def create(self, name: str, venue_type: str, owner_id: UUID, county_code: str | None) -> UUID:
        row = (
            await self.c.execute(
                text(
                    "INSERT INTO venue.venues (name, venue_type, owner_id, county_code) "
                    "VALUES (:n, CAST(:t AS shared.venue_type), :o, :c) RETURNING id"
                ),
                {"n": name, "t": venue_type, "o": owner_id, "c": county_code},
            )
        ).first()
        assert row is not None
        return UUID(str(row[0]))

    async def add_staff(self, venue_id: UUID, user_id: UUID, role: str) -> None:
        await self.c.execute(
            text(
                "INSERT INTO venue.staff (venue_id, user_id, role) VALUES (:v, :u, :r) "
                "ON CONFLICT (venue_id, user_id) DO UPDATE SET role = EXCLUDED.role"
            ),
            {"v": venue_id, "u": user_id, "r": role},
        )

    async def is_staff(self, venue_id: UUID, user_id: UUID, role: str | None = None) -> bool:
        q = text(
            "SELECT 1 FROM venue.staff WHERE venue_id = :v AND user_id = :u AND (CAST(:r AS text) IS NULL OR role = :r)"
        )
        return (await self.c.execute(q, {"v": venue_id, "u": user_id, "r": role})).first() is not None

    async def venues_of(self, user_id: UUID) -> list[dict[str, object]]:
        rows = await self.c.execute(
            text(
                "SELECT v.id, v.name, v.venue_type::text AS venue_type, s.role FROM venue.staff s "
                "JOIN venue.venues v ON v.id = s.venue_id WHERE s.user_id = :u AND v.archived_at IS NULL"
            ),
            {"u": user_id},
        )
        return [dict(r) for r in rows.mappings()]

    async def get(self, venue_id: UUID) -> dict[str, object] | None:
        row = (
            (
                await self.c.execute(
                    text(
                        "SELECT id, name, venue_type::text AS venue_type, owner_id, settings FROM venue.venues WHERE id = :v"
                    ),
                    {"v": venue_id},
                )
            )
            .mappings()
            .first()
        )
        return dict(row) if row else None

    async def insert_device(
        self, venue_id: UUID, name: str, kind: str, prefix: str, digest: bytes, capabilities: dict[str, bool]
    ) -> UUID:
        row = (
            await self.c.execute(
                text(
                    "INSERT INTO venue.devices (venue_id, name, kind, api_key_prefix, api_key_hash, capabilities) "
                    "VALUES (:v, :n, :k, :p, :h, CAST(:c AS jsonb)) RETURNING id"
                ),
                {"v": venue_id, "n": name, "k": kind, "p": prefix, "h": digest, "c": json.dumps(capabilities)},
            )
        ).first()
        assert row is not None
        return UUID(str(row[0]))

    async def replace_device_key(self, venue_id: UUID, device_id: UUID, prefix: str, digest: bytes) -> bool:
        res = await self.c.execute(
            text(
                "UPDATE venue.devices SET api_key_prefix = :p, api_key_hash = :h WHERE id = :d AND venue_id = :v "
                "AND revoked_at IS NULL"
            ),
            {"p": prefix, "h": digest, "d": device_id, "v": venue_id},
        )
        return bool(res.rowcount)

    async def device_by_prefix(self, prefix: str) -> dict[str, object] | None:
        row = (
            (
                await self.c.execute(
                    text(
                        "SELECT id, venue_id, name, kind, api_key_hash, capabilities, revoked_at FROM venue.devices "
                        "WHERE api_key_prefix = :p"
                    ),
                    {"p": prefix},
                )
            )
            .mappings()
            .first()
        )
        return dict(row) if row else None

    async def touch_device(self, device_id: UUID, app_version: str | None) -> None:
        await self.c.execute(
            text(
                "UPDATE venue.devices SET last_seen_at = now(), app_version = COALESCE(:a, app_version) WHERE id = :d"
            ),
            {"d": device_id, "a": app_version},
        )
