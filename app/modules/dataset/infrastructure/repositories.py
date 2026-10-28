"""SQL for the dataset schema (service side; training reads through ml/data)."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection


class SqlDatasetRepo:
    def __init__(self, conn: AsyncConnection):
        self.c = conn

    async def signer_code_for(self, user_id: UUID) -> str | None:
        row = (
            await self.c.execute(text("SELECT signer_code FROM dataset.signers WHERE user_id = :u"), {"u": user_id})
        ).first()
        return row[0] if row else None

    async def active_contribution_consent(self, user_id: UUID) -> UUID | None:
        row = (
            await self.c.execute(
                text(
                    "SELECT id FROM privacy.consents WHERE user_id = :u AND withdrawn_at IS NULL AND scope IN "
                    "('contribute_landmarks', 'contribute_video', 'contribute_audio') ORDER BY granted_at DESC LIMIT 1"
                ),
                {"u": user_id},
            )
        ).first()
        return UUID(str(row[0])) if row else None

    async def insert_own_sample(self, signer_code: str, consent_id: UUID, body: dict[str, Any]) -> UUID:
        uri = body["object_uri"]
        is_video = uri.endswith((".mp4", ".mov"))
        row = (
            await self.c.execute(
                text(
                    "INSERT INTO dataset.samples (source_id, signer_code, session_id, consent_id, kind, domain, video_uri, "
                    "landmarks_uri, text_en) SELECT id, :sc, :sess, :cid, CAST(:k AS shared.sample_kind), "
                    "CAST(:d AS shared.venue_type), :v, :l, :t FROM dataset.sources WHERE slug = 'ksl-bridge-own' RETURNING id"
                ),
                {
                    "sc": signer_code,
                    "sess": body.get("session_id"),
                    "cid": consent_id,
                    "k": body.get("kind", "continuous"),
                    "d": body.get("domain"),
                    "v": uri if is_video else None,
                    "l": None if is_video else uri,
                    "t": body.get("script_english"),
                },
            )
        ).first()
        assert row is not None
        return UUID(str(row[0]))
