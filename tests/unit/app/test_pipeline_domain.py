import numpy as np

from app.modules.pipeline.domain.clarify import build_clarification, should_clarify
from app.modules.pipeline.domain.context import DialogueContext
from app.modules.pipeline.domain.decision import Candidate, choose_final, softmax_confidence
from app.modules.pipeline.domain.endpointing import Endpointer
from app.modules.pipeline.domain.session_state import SignState, SignStateMachine
from ksl_core.ir import TokenSpan


def test_value_of_information_depends_on_venue() -> None:
    assert should_clarify(0.7, "hospital", set())  # (0.3)*4 > 0.15
    assert not should_clarify(0.95, "restaurant", set())  # 0.05 < 0.15
    assert should_clarify(0.9, "restaurant", {"allergy"})


def test_clarification_options_from_weakest_span() -> None:
    spans = [
        TokenSpan(token="WANT", start_ms=0, end_ms=100, confidence=0.9),
        TokenSpan(token="WATER", start_ms=100, end_ms=200, confidence=0.5, alternatives=[("WAIT", 0.3)]),
    ]
    c = build_clarification(spans, ["WANT", "WATER"])
    assert c.kind == "choose_one" and c.options == ["WATER", "WAIT"] and c.span_index == 1
    assert build_clarification([], ["<unk>"]).kind == "fingerspell"


def test_bias_capped_at_two_nats() -> None:
    ctx = DialogueContext(
        expected={"DRINK"}, venue_vocab={"water": ("WATER", "DRINK", 5.0), "Oliech": (None, "PERSON", 0.5)}
    )
    b = ctx.bias({"DRINK": {"WATER", "SODA"}})
    assert max(b.values()) <= 2.0 and "SODA" in b and ctx.fingerspell_lexicon() == ["OLIECH"]


def test_decision_and_softmax() -> None:
    c = choose_final([Candidate(("A",), -1.0, "a", -1.0, "x", 0.2), Candidate(("B",), -1.2, "b", -0.5, "y", 0.9)])
    assert c.glosses == ("B",)
    assert abs(sum(softmax_confidence([-1.0, -2.0])) - 1) < 1e-9


def test_endpointer_and_state_machine() -> None:
    ep = Endpointer(fps=30, still_ms=100)
    f = np.zeros((107, 3), np.float32)
    events = []
    for t in range(40):
        g = f.copy()
        if t < 20:
            g[0:42, 0] = 0.1 * t
        events.append(ep.update(g))
    assert "start" in events and "end" in events
    sm = SignStateMachine()
    sm.to(SignState.SIGNING)
    sm.to(SignState.FINALISING)
    sm.to(SignState.IDLE)
