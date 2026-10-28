"""Dialogue context → expected categories → decoder bias map (Guide Part V §3.3). Pure."""

from __future__ import annotations

from dataclasses import dataclass, field

MAX_BIAS_NATS = 2.0  # matches the CHECK on ml.decoder_configs.bias_max_nats


@dataclass
class DialogueContext:
    venue_type: str = "general"
    last_turns: list[str] = field(default_factory=list)
    expected: set[str] = field(default_factory=set)
    venue_vocab: dict[str, tuple[str | None, str, float]] = field(default_factory=dict)  # term → (gloss, cat, bias)
    max_turns: int = 6

    def add_turn(self, text: str) -> None:
        self.last_turns = (self.last_turns + [text])[-self.max_turns :]

    def bias(self, category_index: dict[str, set[str]], strength: float = 1.5) -> dict[str, float]:
        boost: dict[str, float] = {}
        for cat in self.expected:
            for g in category_index.get(cat, ()):
                boost[g] = max(boost.get(g, 0.0), strength)
        for gloss, _cat, b in self.venue_vocab.values():
            if gloss:
                boost[gloss] = max(boost.get(gloss, 0.0), b)
        return {g: min(v, MAX_BIAS_NATS) for g, v in boost.items()}

    def fingerspell_lexicon(self) -> list[str]:
        return [term.upper() for term, (gloss, _c, _b) in self.venue_vocab.items() if gloss is None]
