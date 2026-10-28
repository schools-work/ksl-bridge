from typing import Any

import pytest

from app.modules.agent.application.dialogue_manager import DialogueManager
from app.modules.agent.domain.policy import Policy
from app.modules.agent.infrastructure.adapters.demo_restaurant import DemoRestaurantAdapter


class FakeTools:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any], bool]] = []
        self.menu = DemoRestaurantAdapter()

    async def __call__(self, name: str, args: dict[str, Any], confirmed: bool = False) -> Any:
        self.calls.append((name, args, confirmed))
        if name == "menu.lookup":
            return await self.menu.menu_lookup(args["name"])
        if name == "order.add":
            return await self.menu.order_add(args["item_id"], args["qty"], "room")
        if name == "kb.answer":
            return None
        return {"ok": True}


@pytest.mark.asyncio
async def test_order_flow_asks_for_missing_slot_then_executes() -> None:
    tools = FakeTools()
    dm = DialogueManager(tools, Policy(upsell_drink=True))
    r1 = await dm.step("order_food", 0.9, {"item": "chicken burger"})
    assert r1.text == "How many?" and "NUMBER" in r1.expected
    r2 = await dm.step("order_food", 0.9, {"qty": 2})
    assert "added" in r2.text and "drink" in r2.text.lower()
    assert ("order.add", {"item_id": 17, "qty": 2}, True) in tools.calls


@pytest.mark.asyncio
async def test_unknown_item_and_escalation() -> None:
    tools = FakeTools()
    dm = DialogueManager(tools, Policy())
    r = await dm.step("order_food", 0.9, {"item": "pizza"})
    assert "don't have" in r.text
    r = await dm.step("ask_location", 0.9, {"place": "toilet"})
    assert any(a["type"] == "staff.call" for a in r.actions)
