"""Device (robot / kiosk / tablet) entity."""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID

DEVICE_KINDS = ("robot", "kiosk", "tablet")


@dataclass(slots=True)
class Device:
    id: UUID
    venue_id: UUID
    name: str
    kind: str
    capabilities: dict[str, bool] = field(default_factory=dict)

    @property
    def can_move(self) -> bool:
        return bool(self.capabilities.get("motion"))
