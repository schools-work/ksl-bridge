"""Slot-filling frame (pure)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Frame:
    intent: str
    required: list[str]
    slots: dict[str, Any] = field(default_factory=dict)
    failed_turns: int = 0

    def missing(self) -> list[str]:
        return [s for s in self.required if s not in self.slots]

    def fill(self, slots: dict[str, Any]) -> None:
        self.slots.update({k: v for k, v in slots.items() if v not in (None, "", [])})

    def to_json(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "required": self.required,
            "slots": self.slots,
            "failed_turns": self.failed_turns,
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> Frame | None:
        if not d or "intent" not in d:
            return None
        return cls(d["intent"], list(d.get("required", [])), dict(d.get("slots", {})), int(d.get("failed_turns", 0)))
