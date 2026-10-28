"""Presence indicators: 'Wanjiku is signing…' / typing."""

from __future__ import annotations

from .protocol import server_msg
from .room_bus import RoomBus


async def signing(bus: RoomBus, room: str, member_id: str, active: bool) -> None:
    await bus.publish(
        room, server_msg("signing", {"member_id": member_id, "active": active}, room=room), exclude_member=member_id
    )


async def typing(bus: RoomBus, room: str, member_id: str, active: bool) -> None:
    await bus.publish(
        room, server_msg("typing", {"member_id": member_id, "active": active}, room=room), exclude_member=member_id
    )
