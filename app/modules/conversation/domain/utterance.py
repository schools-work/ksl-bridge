"""Helpers that build UtteranceIR values (wraps ksl_core.ir)."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from ksl_core.ir import Hypothesis, Modality, UtteranceIR


def from_text(
    room_id: UUID,
    speaker_id: UUID,
    text: str,
    lang: str = "en",
    source: Modality = Modality.TEXT,
    confidence: float = 1.0,
) -> UtteranceIR:
    return UtteranceIR(
        room_id=room_id,
        speaker_id=speaker_id,
        source=source,
        language=lang,
        created_at=datetime.now(UTC),
        is_final=True,
        best=Hypothesis(text=text, log_prob=0.0, confidence=confidence),
    )
