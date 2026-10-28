"""Generic REST POS adapter. Credentials live encrypted in agent.venue_tools.config_enc."""

from __future__ import annotations

from typing import Any

import httpx


class GenericPosRestAdapter:
    def __init__(self, base_url: str = "", token: str = ""):
        self.base_url, self.token = base_url.rstrip("/"), token

    def _client(self) -> httpx.AsyncClient:
        if not self.base_url:
            raise RuntimeError("POS adapter not configured for this venue")
        return httpx.AsyncClient(base_url=self.base_url, timeout=2.0, headers={"Authorization": f"Bearer {self.token}"})

    async def menu_lookup(self, name: str) -> dict[str, Any] | None:
        async with self._client() as c:
            r = await c.get("/menu/search", params={"q": name})
            r.raise_for_status()
            items = r.json()
            return items[0] if items else None

    async def order_add(self, item_id: int, qty: int, room_id: str) -> dict[str, Any]:
        async with self._client() as c:
            r = await c.post(
                "/orders/lines",
                json={"external_ref": room_id, "item_id": item_id, "qty": qty},
                headers={"Idempotency-Key": f"{room_id}:{item_id}:{qty}"},
            )
            r.raise_for_status()
            return dict(r.json())

    async def call_staff(self, room_id: str, reason: str) -> dict[str, Any]:
        async with self._client() as c:
            r = await c.post("/staff/call", json={"ref": room_id, "reason": reason})
            r.raise_for_status()
            return {"notified": True}
