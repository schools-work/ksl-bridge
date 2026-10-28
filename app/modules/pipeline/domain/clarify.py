"""Value-of-information clarification policy (AIMA Ch. 16; Guide Part V §4.2). Pure."""

from __future__ import annotations

from ksl_core.ir import Clarification, TokenSpan

SEVERITY = {
    "general": 1.0,
    "restaurant": 1.0,
    "office": 1.0,
    "education": 1.0,
    "transport": 1.5,
    "bank": 2.0,
    "hospital": 4.0,
}
CRITICAL_SLOTS = frozenset({"allergy", "medication", "amount", "dosage", "emergency"})


def expected_loss(p_best: float, venue: str, slots: set[str], critical_gloss: bool = False) -> float:
    loss_if_wrong = SEVERITY.get(venue, 1.0) * (3.0 if (slots & CRITICAL_SLOTS or critical_gloss) else 1.0)
    return (1.0 - p_best) * loss_if_wrong


def should_clarify(
    p_best: float, venue: str, slots: set[str], c_ask: float = 0.15, critical_gloss: bool = False
) -> bool:
    """Ask iff (1 − p)·loss_if_wrong > cost of asking."""
    return expected_loss(p_best, venue, slots, critical_gloss) > c_ask


def build_clarification(spans: list[TokenSpan], best_glosses: list[str]) -> Clarification:
    """Ask about the least-confident span when alternatives exist; otherwise fingerspell or repeat."""
    if spans:
        i = min(range(len(spans)), key=lambda k: spans[k].confidence)
        if spans[i].alternatives:
            opts = [spans[i].token] + [a for a, _ in spans[i].alternatives[:2]]
            return Clarification(
                kind="choose_one",
                span_index=i,
                prompt_text=f"Did you mean: {' / '.join(opts)}?",
                prompt_glosses=["IX-2", "MEAN", "WHICH"],
                options=opts,
            )
    if "<unk>" in best_glosses:
        return Clarification(
            kind="fingerspell", prompt_text="Please fingerspell that word.", prompt_glosses=["PLEASE", "FINGERSPELL"]
        )
    return Clarification(kind="repeat", prompt_text="Please sign that again.", prompt_glosses=["PLEASE", "AGAIN"])
