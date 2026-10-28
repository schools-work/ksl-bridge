"""Landmarks in → streaming normalisation → endpointing → periodic CSLR partials (Guide Part V §3)."""

from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING

import numpy as np

from app.platform.metrics import STAGE_LATENCY
from app.settings import get_settings
from ksl_core.landmarks import LandmarkBatch

from ..domain.session_state import SignState
from . import finalise_utterance
from .messages import partial_ir

if TYPE_CHECKING:
    from .session import PipelineSession

PARTIAL_EVERY_FRAMES = 8


async def handle(s: PipelineSession, batch: LandmarkBatch) -> None:
    t0 = time.perf_counter()
    norm = s.normalizer.update(batch.xyz, mirror=s.mirror)
    STAGE_LATENCY.labels("normalise").observe(time.perf_counter() - t0)
    for f in norm:
        ev = s.endpointer.update(f)
        if ev == "start" and s.sign.state is not SignState.SIGNING:
            s.sign.to(SignState.SIGNING)
            s.frames.clear()
            s.current_utterance_id = str(uuid.uuid4())
        if s.sign.state is SignState.SIGNING:
            s.frames.append(f)
            s.frames_since_partial += 1
        if ev == "end" and s.sign.state is SignState.SIGNING:
            await finalise_utterance.finalise(s)
            return
    if s.sign.state is SignState.SIGNING and len(s.frames) >= s.max_frames:
        await finalise_utterance.finalise(s)
        return
    if s.sign.state is SignState.SIGNING and s.frames_since_partial >= PARTIAL_EVERY_FRAMES:
        s.frames_since_partial = 0
        await _partial(s)


async def _partial(s: PipelineSession) -> None:
    if s.room_id is None or len(s.frames) < 8:
        return
    payload = {
        "xyz": np.stack(s.frames).astype(np.float16).tolist(),
        "final": False,
        "context_bias": s.context.bias(s.category_index if hasattr(s, "category_index") else {}),
        "venue_type": s.context.venue_type,
    }
    t0 = time.perf_counter()
    out = await s.jobs.call("cslr", payload, deadline_ms=get_settings().job_deadline_ms // 2)
    STAGE_LATENCY.labels("cslr_partial").observe(time.perf_counter() - t0)
    if not out or not out.get("nbest"):
        return
    ir = partial_ir(s, out, revision=s.sign.next_revision())
    await s.bus.publish(
        str(s.room_id),
        {
            "type": "utterance.partial",
            "room": str(s.room_id),
            "id": str(ir.id),
            "data": {"utterance": ir.model_dump(mode="json")},
        },
    )
