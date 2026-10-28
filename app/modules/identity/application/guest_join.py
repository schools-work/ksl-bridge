"""QR guest join: a hearing party joins a room with no account and gets a room-scoped token."""

from __future__ import annotations

import uuid

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from app.platform.db import set_principal
from app.platform.errors import NotFound
from app.platform.principal import Principal
from app.settings import get_settings

from ..infrastructure.jwt import get_jwt
from .dto import GuestJoinIn, GuestToken


async def execute(conn: AsyncConnection, body: GuestJoinIn) -> GuestToken:
    s = get_settings()
    code = body.join_code.upper()
    if "-" not in code:
        code = f"{code[:3]}-{code[3:]}"
    room = (
        (
            await conn.execute(
                text("SELECT room_id, status, expires_at FROM conversation.find_room_by_code(:c)"), {"c": code}
            )
        )
        .mappings()
        .first()
    )
    if room is None or room["status"] != "open":
        raise NotFound("room not found or closed")
    guest_id = uuid.uuid4()
    await conn.execute(
        text(
            "INSERT INTO identity.guests (id, display_name, expires_at) "
            "VALUES (:id, :n, now() + make_interval(secs => :ttl))"
        ),
        {"id": guest_id, "n": body.display_name, "ttl": s.guest_ttl_seconds},
    )
    principal = Principal(guest_id=guest_id, scope=f"room:{room['room_id']}")
    await set_principal(conn, principal)
    await conn.execute(
        text(
            "INSERT INTO conversation.room_members (room_id, guest_id, role, modality_in, modality_out) "
            "VALUES (:r, :g, 'speaker', 'speech', 'text')"
        ),
        {"r": room["room_id"], "g": guest_id},
    )
    token = get_jwt().issue(principal, ttl=s.guest_ttl_seconds)
    return GuestToken(access_token=token, guest_id=guest_id, room_id=room["room_id"], expires_in=s.guest_ttl_seconds)
