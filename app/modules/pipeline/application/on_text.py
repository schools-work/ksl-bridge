"""Text in (typed or ASR) → broadcast; produce signing for members who want sign; agent turns; venue context."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from sqlalchemy import text as sql

from app.modules.agent.application import dialogue_manager
from app.modules.conversation.application import record_utterance
from app.modules.conversation.domain.utterance import from_text
from app.platform.db import transaction
from app.platform.metrics import E2E_LATENCY
from ksl_core.ir import Intent, Modality, UtteranceIR
from ksl_core.text import normalize_text

from .render_sign import broadcast_final, render_sign_for_room

if TYPE_CHECKING:
    from .session import PipelineSession


async def load_venue_context(s: PipelineSession) -> None:
    """Venue vocabulary + gloss categories feed decoder biasing (RAG → recognition, Guide Part VII §3.7)."""
    venue_id = s.room_info.get("venue_id")
    async with transaction(s.principal) as conn:
        cats = await conn.execute(sql("SELECT gloss, categories FROM lexicon.glosses WHERE status <> 'rejected'"))
        index: dict[str, set[str]] = {}
        for g, categories in cats:
            for c in categories or []:
                index.setdefault(c, set()).add(g)
        s.category_index = index  # type: ignore[attr-defined]
        if venue_id:
            rows = await conn.execute(
                sql("SELECT term, gloss, category, bias_nats FROM knowledge.venue_vocabulary WHERE venue_id = :v"),
                {"v": venue_id},
            )
            s.context.venue_vocab = {r[0]: (r[1], r[2], float(r[3])) for r in rows}


async def handle(
    s: PipelineSession,
    text: str,
    lang: str,
    source: Modality = Modality.TEXT,
    confidence: float = 1.0,
    t0: float | None = None,
) -> UtteranceIR | None:
    if s.room_id is None:
        return None
    t0 = t0 or time.perf_counter()
    speaker = s.principal.user_id or s.principal.guest_id or s.principal.device_id
    assert speaker is not None
    ir = from_text(s.room_id, speaker, text.strip(), lang=lang, source=source, confidence=confidence)
    if s.role == "signer":  # the signer's own words drive the service agent → need intent + slots
        nlu = await s.jobs.call(
            "nlu", {"text": ir.best.text, "glosses": [], "domain": s.context.venue_type}, deadline_ms=300
        )
        if nlu and nlu.get("name"):
            ir.intent = Intent(name=nlu["name"], confidence=float(nlu["confidence"]), slots=nlu.get("slots", {}))
    async with transaction(s.principal) as conn:
        await record_utterance.execute(conn, ir, s.member_id)
    await broadcast_final(s, ir)
    s.context.add_turn(normalize_text(text, lang))
    await render_sign_for_room(s, ir)
    await agent_turn(s, ir)
    E2E_LATENCY.labels(f"{source.value}->text").observe(time.perf_counter() - t0)
    return ir


async def agent_turn(s: PipelineSession, ir: UtteranceIR) -> None:
    """If the room has an active agent session (robot/kiosk), let the dialogue manager respond."""
    if s.room_id is None or s.role == "device":
        return
    reply = await dialogue_manager.step_for_room(s.principal, s.room_id, ir)
    if reply is None:
        return
    s.context.expected = set(reply["expected"])
    agent_ir = from_text(s.room_id, s.room_id, reply["text"], lang="en", source=Modality.DEVICE)
    await broadcast_final(s, agent_ir, from_agent=True)
    await render_sign_for_room(s, agent_ir)
    for action in reply.get("actions", []):
        await s.bus.publish(
            str(s.room_id),
            {"type": "agent.action", "room": str(s.room_id), "data": {"action": action["type"], "params": action}},
            only_roles=["device"],
        )


async def handle_device(s: PipelineSession, msg: dict[str, Any]) -> None:
    """Robot/kiosk events become plain messages for the people in the room."""
    if s.room_id is None:
        return
    if msg["type"] == "device.event":
        texts = {
            "order_ready": "Your order is ready.",
            "item_unavailable": "Sorry, that item is not available.",
            "staff_arrived": "A staff member is coming to help you.",
        }
        text = texts.get(msg["event"])
        if text:
            await handle(s, text, "en", source=Modality.DEVICE)
