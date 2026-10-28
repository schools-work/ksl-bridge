"""agent.tools ↔ Python callables, bound to the venue's adapter."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from .adapters.demo_restaurant import DemoRestaurantAdapter
from .adapters.generic_pos_rest import GenericPosRestAdapter

ADAPTERS = {"demo_restaurant": DemoRestaurantAdapter, "pos_generic_rest": GenericPosRestAdapter}


def resolve(name: str, conn: AsyncConnection, venue_id: UUID | None, room_id: UUID) -> Callable[..., Awaitable[Any]]:
    async def adapter() -> Any:
        row = (
            await conn.execute(
                text(
                    "SELECT vt.adapter FROM agent.venue_tools vt JOIN agent.tools t ON t.id = vt.tool_id "
                    "WHERE vt.venue_id = :v AND t.name = :n AND vt.enabled"
                ),
                {"v": venue_id, "n": name},
            )
        ).first()
        return ADAPTERS.get(row[0] if row else "demo_restaurant", DemoRestaurantAdapter)()

    async def menu_lookup(name: str) -> Any:
        return await (await adapter()).menu_lookup(name)

    async def order_add(item_id: int, qty: int) -> Any:
        return await (await adapter()).order_add(item_id, qty, str(room_id))

    async def staff_call(reason: str) -> Any:
        sess = (
            await conn.execute(
                text("SELECT id FROM agent.sessions WHERE room_id = :r AND goal_status = 'active'"), {"r": room_id}
            )
        ).first()
        if sess:
            await conn.execute(
                text("INSERT INTO agent.escalations (session_id, reason) VALUES (:s, :r)"),
                {
                    "s": sess[0],
                    "r": reason
                    if reason
                    in (
                        "low_confidence",
                        "unknown_intent",
                        "kb_no_answer",
                        "critical_domain",
                        "user_request",
                        "tool_failure",
                    )
                    else "user_request",
                },
            )
        return await (await adapter()).call_staff(str(room_id), reason)

    async def kb_answer(question: str) -> Any:
        from app.modules.knowledge.application import answer_question

        res = await answer_question.execute(conn, venue_id, question, room_id)
        return res["answer"]

    async def show_menu() -> Any:
        return {"shown": True}

    table: dict[str, Callable[..., Awaitable[Any]]] = {
        "menu.lookup": menu_lookup,
        "order.add": order_add,
        "staff.call": staff_call,
        "kb.answer": kb_answer,
        "ui.show_menu": show_menu,
    }
    if name not in table:
        raise KeyError(f"no implementation for tool {name}")
    return table[name]
