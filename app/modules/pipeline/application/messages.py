"""Build UtteranceIR values from worker outputs (shared by partial and final paths)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID

from ksl_core.ir import Hypothesis, Intent, Modality, TokenSpan, UtteranceIR

from ..domain.decision import softmax_confidence

if TYPE_CHECKING:
    from .session import PipelineSession


def _speaker(s: PipelineSession) -> UUID:
    return s.principal.user_id or s.principal.guest_id or s.principal.device_id or UUID(int=0)


def partial_ir(s: PipelineSession, out: dict[str, Any], revision: int) -> UtteranceIR:
    nbest = out["nbest"]
    conf = softmax_confidence([h["score"] for h in nbest])
    hyps = [
        Hypothesis(glosses=h["glosses"], log_prob=h["score"], confidence=c) for h, c in zip(nbest, conf, strict=True)
    ]
    assert s.room_id is not None
    return UtteranceIR(
        id=UUID(s.current_utterance_id) if s.current_utterance_id else UUID(int=0),
        room_id=s.room_id,
        speaker_id=_speaker(s),
        source=Modality.SIGN,
        language="ksl",
        created_at=datetime.now(UTC),
        is_final=False,
        revision=revision,
        best=hyps[0],
        nbest=hyps,
        model_versions=out.get("model_versions", {}),
    )


def final_ir(
    s: PipelineSession, out: dict[str, Any], text: str | None, intent: dict[str, Any] | None, calibrated: float | None
) -> UtteranceIR:
    ir = partial_ir(s, out, revision=s.sign.revision + 1)
    ir.is_final = True
    ir.best.text = text
    if calibrated is not None:
        ir.best.confidence = max(0.0, min(1.0, calibrated))
    ir.spans = [TokenSpan(**sp) for sp in out.get("spans", [])]
    if intent:
        ir.intent = Intent(name=intent["name"], confidence=intent["confidence"], slots=intent.get("slots", {}))
    return ir
