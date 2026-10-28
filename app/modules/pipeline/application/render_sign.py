"""Broadcast finals; produce avatar pose sequences (Text→Gloss→Sign) and TTS for members who want them."""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING

from ksl_core.ir import UtteranceIR

if TYPE_CHECKING:
    from .session import PipelineSession

BIN_POSES = 0x03
BIN_TTS = 0x04


async def broadcast_final(s: PipelineSession, ir: UtteranceIR, from_agent: bool = False) -> None:
    assert s.room_id is not None
    await s.bus.publish(
        str(s.room_id),
        {
            "type": "utterance.final",
            "room": str(s.room_id),
            "id": str(ir.id),
            "data": {"utterance": ir.model_dump(mode="json"), "from_agent": from_agent},
        },
    )
    if ir.source.value == "sign" and ir.best.text:
        await _tts_for_room(s, ir)


async def render_sign_for_room(s: PipelineSession, ir: UtteranceIR) -> None:
    """Text → glosses → pose sequence, sent only to members whose output modality is sign."""
    if s.room_id is None or not ir.best.text:
        return
    t2g = await s.jobs.call(
        "t2g", {"text": ir.best.text, "lang": ir.language, "domain": s.context.venue_type}, deadline_ms=600
    )
    if not t2g:
        return
    glosses = t2g.get("glosses", [])
    prod = (
        await s.jobs.call(
            "sign2pose", {"glosses": glosses, "speed": 1.0, "question": t2g.get("question")}, deadline_ms=800
        )
        or {}
    )
    poses_b64 = prod.get("poses_f16_b64")
    binary = bytes([BIN_POSES]) + base64.b64decode(poses_b64) if poses_b64 else None
    await s.bus.publish(
        str(s.room_id),
        {
            "type": "sign.render",
            "room": str(s.room_id),
            "id": str(ir.id),
            "data": {
                "utterance_id": str(ir.id),
                "glosses": glosses,
                "captions": ir.best.text,
                "n_frames": int(prod.get("n_frames", 0)),
                "fps": float(prod.get("fps", 30.0)),
                "low_confidence": bool(t2g.get("low_confidence")),
                "critical": bool(t2g.get("critical")),
            },
        },
        binary=binary,
        only_modality_out=["sign"],
    )


async def _tts_for_room(s: PipelineSession, ir: UtteranceIR) -> None:
    assert s.room_id is not None
    out = await s.jobs.call("tts", {"text": ir.best.text, "lang": "en"}, deadline_ms=1500)
    if not out or not out.get("pcm16_b64"):
        return
    await s.bus.publish(
        str(s.room_id),
        {
            "type": "speech.audio",
            "room": str(s.room_id),
            "id": str(ir.id),
            "data": {"utterance_id": str(ir.id), "format": "pcm16", "sr": int(out.get("sample_rate", 22050))},
        },
        binary=bytes([BIN_TTS]) + base64.b64decode(out["pcm16_b64"]),
        only_modality_out=["speech"],
    )
