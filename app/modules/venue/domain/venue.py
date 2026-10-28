"""Venue entity and its policy settings."""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID

VENUE_TYPES = ("restaurant", "hospital", "bank", "office", "transport", "education", "general")
# loss multiplier when a reading is wrong (clarification policy, Guide Part V §4.2)
SEVERITY = {
    "general": 1.0,
    "restaurant": 1.0,
    "office": 1.0,
    "transport": 1.5,
    "education": 1.0,
    "bank": 2.0,
    "hospital": 4.0,
}


@dataclass(slots=True)
class Venue:
    id: UUID
    name: str
    venue_type: str
    owner_id: UUID
    settings: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.venue_type not in VENUE_TYPES:
            raise ValueError(f"unknown venue type {self.venue_type}")

    @property
    def severity(self) -> float:
        return SEVERITY[self.venue_type]
