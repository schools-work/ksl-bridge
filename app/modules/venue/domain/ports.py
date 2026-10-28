"""Ports for the venue module."""

from __future__ import annotations

from typing import Protocol
from uuid import UUID


class DeviceKeyHasher(Protocol):
    def new_key(self) -> tuple[str, str, bytes]: ...
    def verify(self, presented: str, stored_hash: bytes) -> bool: ...


class VenueRepo(Protocol):
    async def create(self, name: str, venue_type: str, owner_id: UUID, county_code: str | None) -> UUID: ...
    async def is_staff(self, venue_id: UUID, user_id: UUID) -> bool: ...
