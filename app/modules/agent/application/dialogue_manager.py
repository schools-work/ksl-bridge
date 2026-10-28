"""Goal-based dialogue manager: ground slots with tools → ask for missing slots → execute (Guide Part VII §2)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from typing import Any
from uuid import UUID

from sqlalchemy import text

from app.platform.db import set_principal, transaction
from app.platform.principal import Principal
from ksl_core.ir import UtteranceIR

from ..domain.frame import Frame
from ..domain.policy import Policy
from ..domain.ports import ToolExecutor


@dataclass
class AgentReply:
    text: str
    expected: set[str]
    actions: list[dict[str, Any]] = field(default_factory=list)
    done: bool = False
    needs_confirmation: dict[str, Any] | None = None


class DialogueManager:
    def __init__(self, tools: ToolExecutor, policy: Policy, frame: Frame | None = None):
        self.tools, self.policy, self.frame = tools, policy, frame

    async def step(self, intent: str, conf: float, slots: dict[str, Any]) -> AgentReply:
        p = self.policy
        if conf < p.min_intent_confidence or intent == "unknown":
            if self.frame is not None:
                self.frame.failed_turns += 1
                if self.frame.failed_turns >= p.escalate_after_failed_turns:
                    return await self._escalate("low_confidence")
            return AgentReply("Sorry, I didn't understand. Could you sign it again?", set())
        if self.frame is None or (intent != self.frame.intent and intent in p.intents):
            self.frame = Frame(intent, list(p.intents.get(intent, [])))
        self.frame.fill(slots)
        if "item" in self.frame.slots and "item_id" not in self.frame.slots:
            hit = await self.tools("menu.lookup", {"name": str(self.frame.slots["item"])})
            if not hit:
                self.frame.slots.pop("item")
                return AgentReply(
                    "We don't have that. Would you like to see the menu?",
                    {"FOOD", "DRINK"},
                    actions=[{"type": "ui.show_menu"}],
                )
            self.frame.slots["item_id"], self.frame.slots["item"] = hit["id"], hit["name"]
        missing = self.frame.missing()
        if missing:
            prompt = p.prompts.get(missing[0], {"text": f"Please tell me the {missing[0]}.", "expect": []})
            return AgentReply(prompt["text"], set(prompt.get("expect", [])))
        return await self._execute()

    async def _execute(self) -> AgentReply:
        f = self.frame
        assert f is not None
        if f.intent == "order_food":
            qty = int(f.slots["qty"])
            res = await self.tools("order.add", {"item_id": int(f.slots["item_id"]), "qty": qty}, confirmed=True)
            self.frame = None
            follow = self.policy.upsell_drink and "drink" not in f.slots
            msg = f"{qty} {f.slots['item']}, added."
            if follow:
                return AgentReply(
                    f"{msg} Would you like a drink?",
                    {"DRINK", "YES_NO"},
                    actions=[{"type": "order_added", **(res or {})}],
                )
            return AgentReply(
                f"{msg} Anything else?", {"YES_NO"}, actions=[{"type": "order_added", **(res or {})}], done=True
            )
        if f.intent == "ask_location":
            ans = await self.tools("kb.answer", {"question": f"Where is the {f.slots['place']}?"})
            self.frame = None
            if ans:
                return AgentReply(str(ans), set(), done=True)
            return await self._escalate("kb_no_answer")
        if f.intent == "greet":
            self.frame = None
            return AgentReply("Hello! How can I help you?", {"FOOD", "DRINK", "HELP"})
        self.frame = None
        return await self._escalate("unknown_intent")

    async def _escalate(self, reason: str) -> AgentReply:
        await self.tools("staff.call", {"reason": reason})
        self.frame = None
        return AgentReply(
            "Let me call a staff member to help you.",
            set(),
            actions=[{"type": "staff.call", "reason": reason}],
            done=True,
        )


async def step_for_room(principal: Principal, room_id: UUID, ir: UtteranceIR) -> dict[str, Any] | None:
    """Load the room's active agent session, run one step, persist frame/turns. None if the room has no agent."""
    from .execute_tool import SessionToolExecutor

    if ir.intent is None:
        return None
    async with transaction(principal) as conn:
        sess = (
            (
                await conn.execute(
                    text(
                        "SELECT s.id, s.venue_id, s.frame, d.intents, d.slot_prompts, d.policy FROM agent.sessions s "
                        "JOIN agent.definitions d ON d.id = s.definition_id WHERE s.room_id = :r AND s.goal_status = 'active'"
                    ),
                    {"r": room_id},
                )
            )
            .mappings()
            .first()
        )
        if sess is None:
            return None
        if sess["venue_id"] is not None:  # venue KB is readable within this venue's rooms (RLS)
            await set_principal(conn, replace(principal, venue_id=sess["venue_id"]))
        policy = Policy.from_definition(sess["intents"], sess["slot_prompts"], sess["policy"])
        tools = SessionToolExecutor(conn, sess["id"], sess["venue_id"], room_id)
        dm = DialogueManager(tools, policy, Frame.from_json(sess["frame"]))
        reply = await dm.step(ir.intent.name, ir.intent.confidence, dict(ir.intent.slots))
        idx = (
            await conn.execute(
                text("SELECT COALESCE(max(turn_index), -1) + 1 FROM agent.turns WHERE session_id = :s"),
                {"s": sess["id"]},
            )
        ).scalar_one()
        await conn.execute(
            text(
                "INSERT INTO agent.turns (session_id, turn_index, actor, utterance_id, utterance_created_at, intent, "
                "intent_confidence, slots, reply_text, expected_categories) VALUES (:s, :i, 'user', :u, :ts, :in, :c, "
                "CAST(:sl AS jsonb), :rt, :ex)"
            ),
            {
                "s": sess["id"],
                "i": idx,
                "u": ir.id,
                "ts": ir.created_at,
                "in": ir.intent.name,
                "c": ir.intent.confidence,
                "sl": json.dumps(ir.intent.slots),
                "rt": reply.text,
                "ex": sorted(reply.expected),
            },
        )
        status = "escalated" if any(a.get("type") == "staff.call" for a in reply.actions) else "active"
        await conn.execute(
            text("UPDATE agent.sessions SET frame = CAST(:f AS jsonb), goal_status = :g WHERE id = :s"),
            {"f": json.dumps(dm.frame.to_json() if dm.frame else {}), "g": status, "s": sess["id"]},
        )
    return {"text": reply.text, "expected": sorted(reply.expected), "actions": reply.actions, "done": reply.done}
