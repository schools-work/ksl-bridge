"""In-memory demo adapter with a small Kenyan menu (demos and tests)."""

from __future__ import annotations

from typing import Any, ClassVar

_ORDERS: dict[str, list[tuple[int, int]]] = {}


class DemoRestaurantAdapter:
    MENU: ClassVar[list[dict[str, Any]]] = [
        {"id": 17, "name": "chicken burger", "price": 650},
        {"id": 3, "name": "chips", "price": 250},
        {"id": 21, "name": "water", "price": 100},
        {"id": 22, "name": "soda", "price": 120},
        {"id": 30, "name": "ugali beef", "price": 450},
        {"id": 31, "name": "chapati", "price": 50},
    ]

    async def menu_lookup(self, name: str) -> dict[str, Any] | None:
        n = name.lower().strip()
        return next((m for m in self.MENU if n == m["name"] or n in m["name"] or m["name"] in n), None)

    async def order_add(self, item_id: int, qty: int, room_id: str) -> dict[str, Any]:
        _ORDERS.setdefault(room_id, []).append((item_id, qty))
        total = sum(next(m["price"] for m in self.MENU if m["id"] == i) * q for i, q in _ORDERS[room_id])
        return {"order_lines": len(_ORDERS[room_id]), "total_ksh": total}

    async def call_staff(self, room_id: str, reason: str) -> dict[str, Any]:
        return {"notified": True, "reason": reason}
