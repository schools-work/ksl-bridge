"""Speech in → ASR worker → same path as typed text (source = speech)."""

from __future__ import annotations

import base64
import time
from typing import TYPE_CHECKING

from app.settings import get_settings
from ksl_core._generated.ws_messages import ErrorCode
from ksl_core.ir import Modality

from . import on_text

if TYPE_CHECKING:
    from .session import PipelineSession

MIN_AUDIO_BYTES = 16000 * 2 // 4  # 250 ms


async def handle(s: PipelineSession) -> None:
    pcm, s.audio = bytes(s.audio), bytearray()
    if len(pcm) < MIN_AUDIO_BYTES or s.room_id is None:
        return
    t0 = time.perf_counter()
    out = await s.jobs.call(
        "asr",
        {
            "pcm16_b64": base64.b64encode(pcm).decode(),
            "sample_rate": 16000,
            "lang_hint": s.speech_lang,
            "domain": s.context.venue_type,
        },
        deadline_ms=get_settings().job_deadline_ms * 3,
    )
    if not out or not out.get("text"):
        s.reply(
            {
                "type": "error",
                "data": {
                    "code": int(ErrorCode.PIPELINE_UNAVAILABLE),
                    "message": "Speech recognition is unavailable; please type instead.",
                },
            }
        )
        return
    await on_text.handle(
        s,
        out["text"],
        out.get("lang", "en"),
        source=Modality.SPEECH,
        confidence=float(out.get("confidence", 0.8)),
        t0=t0,
    )
