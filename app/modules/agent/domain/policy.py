"""Agent policy: intent schemas, slot prompts, escalation rules (loaded from agent.definitions). Pure."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

DEFAULT_INTENTS: dict[str, list[str]] = {
    "order_food": ["item", "qty"],
    "ask_location": ["place"],
    "pay": ["method"],
    "book_appointment": ["department", "time"],
    "greet": [],
    "help": [],
}
DEFAULT_PROMPTS: dict[str, dict[str, Any]] = {
    "item": {"text": "What would you like?", "expect": ["FOOD", "DRINK"]},
    "qty": {"text": "How many?", "expect": ["NUMBER"]},
    "place": {"text": "Where do you want to go?", "expect": ["PLACE"]},
    "method": {"text": "Cash or M-Pesa?", "expect": ["PAYMENT"]},
    "department": {"text": "Which department?", "expect": ["HEALTH"]},
    "time": {"text": "What time?", "expect": ["TIME", "NUMBER"]},
}


@dataclass
class Policy:
    intents: dict[str, list[str]] = field(default_factory=lambda: dict(DEFAULT_INTENTS))
    prompts: dict[str, dict[str, Any]] = field(default_factory=lambda: dict(DEFAULT_PROMPTS))
    min_intent_confidence: float = 0.4
    upsell_drink: bool = False
    escalate_after_failed_turns: int = 2

    @classmethod
    def from_definition(
        cls, intents: dict[str, list[str]], prompts: dict[str, dict[str, Any]], policy: dict[str, Any]
    ) -> Policy:
        return cls(
            intents=intents or dict(DEFAULT_INTENTS),
            prompts={**DEFAULT_PROMPTS, **(prompts or {})},
            min_intent_confidence=float(policy.get("min_intent_confidence", 0.4)),
            upsell_drink=bool(policy.get("upsell_drink", False)),
            escalate_after_failed_turns=int(policy.get("escalate_after_failed_turns", 2)),
        )
