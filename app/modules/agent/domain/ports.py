"""Ports: venue adapters and tool execution."""

from __future__ import annotations

from typing import Any, Protocol


class VenueAdapter(Protocol):
    async def menu_lookup(self, name: str) -> dict[str, Any] | None: ...
    async def order_add(self, item_id: int, qty: int, room_id: str) -> dict[str, Any]: ...
    async def call_staff(self, room_id: str, reason: str) -> dict[str, Any]: ...


class ToolExecutor(Protocol):
    async def __call__(self, name: str, args: dict[str, Any], confirmed: bool = False) -> Any: ...
