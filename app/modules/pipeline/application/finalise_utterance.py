"""Final sign path: CSLR final → gloss→text → NLU → clarification decision → broadcast → agent."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import numpy as np

from app.modules.conversation.application import record_utterance
from app.platform.db import transaction
from app.platform.metrics import CLARIFICATIONS, E2E_LATENCY, STAGE_LATENCY
from app.settings import get_settings
from ksl_core._generated.ws_messages import ErrorCode

from ..domain.clarify import build_clarification, expected_loss, should_clarify
from ..domain.session_state import SignState
from .messages import final_ir
from .render_sign import broadcast_final

if TYPE_CHECKING:
    from .session import PipelineSession


async def finalise(s: PipelineSession) -> None:
    if s.sign.state is not SignState.SIGNING or s.room_id is None:
        return
    s.sign.to(SignState.FINALISING)
    frames, s.frames = s.frames, []
    s.frames_since_partial = 0
    try:
        if len(frames) < 6:
            return
        t0 = time.perf_counter()
        settings = get_settings()
        out = await s.jobs.call(
            "cslr",
            {
                "xyz": np.stack(frames).astype(np.float16).tolist(),
                "final": True,
                "context_bias": s.context.bias(getattr(s, "category_index", {})),
                "venue_type": s.context.venue_type,
                "fs_lexicon": s.context.fingerspell_lexicon(),
            },
            deadline_ms=settings.job_deadline_ms,
        )
        STAGE_LATENCY.labels("cslr_final").observe(time.perf_counter() - t0)
        if not out or not out.get("nbest"):
            s.reply(
                {
                    "type": "error",
                    "data": {
                        "code": int(ErrorCode.PIPELINE_UNAVAILABLE),
                        "message": "Sign translation is temporarily unavailable; text chat still works.",
                    },
                }
            )
            return
        best = out["nbest"][0]["glosses"]
        tr = await s.jobs.call("g2t", {"glosses": best, "domain": s.context.venue_type}, deadline_ms=600) or {}
        text = tr.get("text") or " ".join(best)
        nlu = await s.jobs.call("nlu", {"text": text, "glosses": best, "domain": s.context.venue_type}, deadline_ms=300)
        ir = final_ir(s, out, text=text, intent=nlu, calibrated=out.get("calibrated_confidence"))
        ir.model_versions.update(tr.get("model_versions", {}))
        slots = set((nlu or {}).get("slots", {}).keys())
        critical = bool(out.get("critical"))
        if should_clarify(ir.best.confidence, s.context.venue_type, slots, settings.clarify_cost, critical):
            ir.needs_clarification = True
            ir.clarification = build_clarification(ir.spans, best)
            s.pending_clarifications[str(ir.id)] = {
                "ir": ir,
                "loss": expected_loss(ir.best.confidence, s.context.venue_type, slots, critical),
            }
            CLARIFICATIONS.labels(s.context.venue_type, ir.clarification.kind).inc()
        async with transaction(s.principal) as conn:
            await record_utterance.execute(conn, ir, s.member_id, force_link=ir.needs_clarification)
            if ir.needs_clarification and ir.clarification:
                await _store_clarification(conn, ir, s)
        if ir.needs_clarification and ir.clarification:
            s.reply(
                {
                    "type": "clarify.request",
                    "room": str(s.room_id),
                    "id": str(ir.id),
                    "data": {"utterance_id": str(ir.id), "clarification": ir.clarification.model_dump()},
                }
            )
            await s.bus.publish(
                str(s.room_id),
                {"type": "typing", "room": str(s.room_id), "data": {"member_id": s.member_id, "active": True}},
                exclude_member=s.member_id,
            )
        else:
            await broadcast_final(s, ir)
            await _agent_step(s, ir)
        E2E_LATENCY.labels("sign->text").observe(time.perf_counter() - t0)
        s.context.add_turn(text)
    finally:
        s.sign.to(SignState.IDLE)


async def _store_clarification(conn: Any, ir: Any, s: PipelineSession) -> None:
    from sqlalchemy import text as sql

    await conn.execute(
        sql(
            "INSERT INTO inference.clarifications (utterance_id, utterance_created_at, kind, options, span_index, p_best, "
            "expected_loss, venue_type) VALUES (:u, :ts, CAST(:k AS shared.clarify_kind), :o, :si, :p, :l, "
            "CAST(:v AS shared.venue_type))"
        ),
        {
            "u": ir.id,
            "ts": ir.created_at,
            "k": ir.clarification.kind,
            "o": ir.clarification.options,
            "si": ir.clarification.span_index,
            "p": ir.best.confidence,
            "l": s.pending_clarifications[str(ir.id)]["loss"],
            "v": s.context.venue_type,
        },
    )


async def resolve_clarification(s: PipelineSession, utterance_id: str, choice: str | None, retry: bool) -> None:
    pending = s.pending_clarifications.pop(utterance_id, None)
    if pending is None:
        return
    ir = pending["ir"]
    if retry or choice is None:
        s.reply(
            {
                "type": "clarify.request",
                "data": {
                    "utterance_id": utterance_id,
                    "clarification": {
                        "kind": "repeat",
                        "prompt_text": "Please sign again.",
                        "prompt_glosses": ["PLEASE", "AGAIN"],
                        "options": [],
                    },
                },
            }
        )
        return
    clar = ir.clarification
    if clar and clar.span_index is not None and choice in clar.options:
        glosses = list(ir.best.glosses)
        glosses[clar.span_index] = choice
        ir.best.glosses = glosses
        tr = await s.jobs.call("g2t", {"glosses": glosses, "domain": s.context.venue_type}, deadline_ms=600) or {}
        ir.best.text = tr.get("text") or " ".join(glosses)
    ir.best.confidence = max(ir.best.confidence, 0.95)  # confirmed by the signer
    ir.needs_clarification = False
    async with transaction(s.principal) as conn:
        from sqlalchemy import text as sql

        await conn.execute(
            sql(
                "UPDATE inference.clarifications SET answered_choice = :c, answered_at = now(), "
                "outcome = 'resolved' WHERE utterance_id = :u"
            ),
            {"c": choice, "u": ir.id},
        )
    await broadcast_final(s, ir)
    await _agent_step(s, ir)


async def _agent_step(s: PipelineSession, ir: Any) -> None:
    from . import on_text

    await on_text.agent_turn(s, ir)
