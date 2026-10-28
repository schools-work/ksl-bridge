"""Right of access: everything we hold about the caller, as JSON."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection


async def execute(conn: AsyncConnection, user_id: UUID) -> dict[str, Any]:
    async def rows(sql: str) -> list[dict[str, Any]]:
        return [dict(r) for r in (await conn.execute(text(sql), {"u": user_id})).mappings()]

    return {
        "user": await rows(
            "SELECT id, email::text, phone_e164, display_name, locale, preferred_in::text, "
            "preferred_out::text, dominant_hand::text, sign_speed, created_at FROM identity.users WHERE id = :u"
        ),
        "roles": await rows("SELECT role::text, granted_at, revoked_at FROM identity.user_roles WHERE user_id = :u"),
        "consents": await rows(
            "SELECT scope::text, text_version, granted_at, withdrawn_at FROM privacy.consents WHERE user_id = :u"
        ),
        "feedback": await rows(
            "SELECT kind::text, corrected_glosses, corrected_text, created_at FROM feedback.events WHERE user_id = :u"
        ),
        "personalisation": await rows(
            "SELECT kind, gloss, n_examples, enabled, created_at FROM feedback.personal_adaptations WHERE user_id = :u"
        ),
        "sessions": await rows(
            "SELECT device_label, issued_at, expires_at, revoked_at FROM identity.refresh_tokens "
            "WHERE user_id = :u ORDER BY issued_at DESC LIMIT 50"
        ),
    }
