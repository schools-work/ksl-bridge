"""SQL repositories for the conversation schema."""

from __future__ import annotations

import json
from typing import Any
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from app.platform.principal import Principal
from ksl_core.ir import UtteranceIR


class SqlRoomRepo:
    def __init__(self, conn: AsyncConnection):
        self.c = conn

    async def venue_type(self, venue_id: UUID | None) -> str:
        if venue_id is None:
            return "general"
        row = (
            await self.c.execute(text("SELECT venue_type::text FROM venue.venues WHERE id = :v"), {"v": venue_id})
        ).first()
        return row[0] if row else "general"

    async def create(self, p: Principal, venue_id: UUID | None, profile: str, store: bool) -> dict[str, Any]:
        row = (
            (
                await self.c.execute(
                    text(
                        "INSERT INTO conversation.rooms (venue_id, created_by_user, created_by_device, context_profile, "
                        "store_transcript) VALUES (:v, :u, :d, CAST(:prof AS shared.venue_type), :s) RETURNING id, join_code"
                    ),
                    {"v": venue_id, "u": p.user_id, "d": p.device_id, "prof": profile, "s": store},
                )
            )
            .mappings()
            .first()
        )
        assert row is not None
        return dict(row)

    async def add_member(self, room_id: UUID, p: Principal, role: str, mi: str, mo: str) -> UUID:
        row = (
            await self.c.execute(
                text(
                    "INSERT INTO conversation.room_members (room_id, user_id, guest_id, device_id, role, modality_in, "
                    "modality_out) VALUES (:r, :u, :g, :d, :role, CAST(:mi AS shared.modality), CAST(:mo AS shared.modality)) "
                    "RETURNING id"
                ),
                {"r": room_id, "u": p.user_id, "g": p.guest_id, "d": p.device_id, "role": role, "mi": mi, "mo": mo},
            )
        ).first()
        assert row is not None
        return UUID(str(row[0]))

    async def find_by_code(self, code: str) -> dict[str, Any] | None:
        row = (
            (await self.c.execute(text("SELECT * FROM conversation.find_room_by_code(:c)"), {"c": code}))
            .mappings()
            .first()
        )
        return dict(row) if row else None

    async def get(self, room_id: UUID) -> dict[str, Any] | None:
        row = (
            (
                await self.c.execute(
                    text(
                        "SELECT id, venue_id, join_code, context_profile::text AS context_profile, status, store_transcript, "
                        "expires_at FROM conversation.rooms WHERE id = :r"
                    ),
                    {"r": room_id},
                )
            )
            .mappings()
            .first()
        )
        return dict(row) if row else None

    async def member_of(self, room_id: UUID, p: Principal) -> dict[str, Any] | None:
        row = (
            (
                await self.c.execute(
                    text(
                        "SELECT id, role, modality_in::text AS modality_in, modality_out::text AS modality_out "
                        "FROM conversation.room_members WHERE room_id = :r AND left_at IS NULL AND "
                        "((user_id IS NOT NULL AND user_id = :u) OR (guest_id IS NOT NULL AND guest_id = :g) "
                        "OR (device_id IS NOT NULL AND device_id = :d))"
                    ),
                    {"r": room_id, "u": p.user_id, "g": p.guest_id, "d": p.device_id},
                )
            )
            .mappings()
            .first()
        )
        return dict(row) if row else None

    async def members(self, room_id: UUID) -> list[dict[str, Any]]:
        rows = await self.c.execute(
            text(
                "SELECT m.id, m.role, m.modality_in::text AS modality_in, m.modality_out::text AS modality_out, "
                "COALESCE(u.display_name, g.display_name, d.name) AS name "
                "FROM conversation.room_members m LEFT JOIN identity.users u ON u.id = m.user_id "
                "LEFT JOIN identity.guests g ON g.id = m.guest_id LEFT JOIN venue.devices d ON d.id = m.device_id "
                "WHERE m.room_id = :r AND m.left_at IS NULL ORDER BY m.joined_at"
            ),
            {"r": room_id},
        )
        return [dict(r) for r in rows.mappings()]

    async def leave(self, member_id: UUID) -> None:
        await self.c.execute(
            text("UPDATE conversation.room_members SET left_at = now() WHERE id = :m AND left_at IS NULL"),
            {"m": member_id},
        )

    async def close(self, room_id: UUID) -> bool:
        res = await self.c.execute(
            text(
                "UPDATE conversation.rooms SET status = 'closed', closed_at = now() WHERE id = :r AND status = 'open'"
            ),
            {"r": room_id},
        )
        return bool(res.rowcount)

    async def stores_transcript(self, room_id: UUID) -> bool:
        row = (
            await self.c.execute(text("SELECT store_transcript FROM conversation.rooms WHERE id = :r"), {"r": room_id})
        ).first()
        return bool(row and row[0])

    async def insert_utterance(self, ir: UtteranceIR, member_id: str | None, with_content: bool) -> None:
        await self.c.execute(
            text(
                "INSERT INTO conversation.utterances (id, created_at, room_id, member_id, source, language, glosses, "
                "text_en, confidence, intent, non_manual, needs_clarification, revision, model_versions) VALUES "
                "(:id, :ts, :room, :mem, CAST(:src AS shared.modality), CAST(:lang AS shared.lang), :gl, :txt, :conf, "
                "CAST(:intent AS jsonb), CAST(:nm AS jsonb), :clar, :rev, CAST(:mv AS jsonb)) ON CONFLICT DO NOTHING"
            ),
            {
                "id": ir.id,
                "ts": ir.created_at,
                "room": ir.room_id,
                "mem": member_id,
                "src": ir.source.value,
                "lang": ir.language if ir.language in ("ksl", "en", "sw") else "en",
                "gl": ir.best.glosses if with_content else None,
                "txt": ir.best.text if with_content else None,
                "conf": ir.best.confidence,
                "intent": json.dumps(ir.intent.model_dump()) if (with_content and ir.intent) else None,
                "nm": json.dumps(ir.non_manual.model_dump()) if ir.non_manual else None,
                "clar": ir.needs_clarification,
                "rev": ir.revision,
                "mv": json.dumps(ir.model_versions),
            },
        )

    async def transcript(self, room_id: UUID, limit: int, before: str | None) -> list[dict[str, Any]]:
        rows = await self.c.execute(
            text(
                "SELECT id, created_at, source::text AS source, glosses, text_en, confidence FROM conversation.utterances "
                "WHERE room_id = :r AND (CAST(:b AS timestamptz) IS NULL OR created_at < CAST(:b AS timestamptz)) "
                "AND text_en IS NOT NULL ORDER BY created_at DESC LIMIT :n"
            ),
            {"r": room_id, "b": before, "n": limit},
        )
        return [dict(r) for r in rows.mappings()]
