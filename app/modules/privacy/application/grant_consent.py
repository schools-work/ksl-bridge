"""Grant a consent scope against the current consent text version."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from app.platform.audit import record
from app.platform.errors import Invalid

from ..domain.consent import validate_scope


async def execute(
    conn: AsyncConnection, user_id: UUID, scope: str, language: str = "en", evidence_uri: str | None = None
) -> UUID:
    try:
        validate_scope(scope)
    except ValueError as e:
        raise Invalid(str(e)) from e
    ver = (
        await conn.execute(
            text(
                "SELECT version FROM privacy.consent_texts WHERE scope = CAST(:s AS shared.consent_scope) AND "
                "language = CAST(:l AS shared.lang) AND effective_from <= now() ORDER BY effective_from DESC LIMIT 1"
            ),
            {"s": scope, "l": language},
        )
    ).first()
    if ver is None:
        raise Invalid("no consent text published for this scope/language")
    row = (
        await conn.execute(
            text(
                "INSERT INTO privacy.consents (user_id, scope, text_version, text_language, evidence_uri) "
                "VALUES (:u, CAST(:s AS shared.consent_scope), :v, CAST(:l AS shared.lang), :e) "
                "ON CONFLICT (user_id, scope) WHERE withdrawn_at IS NULL AND user_id IS NOT NULL DO UPDATE SET evidence_uri = "
                "COALESCE(EXCLUDED.evidence_uri, privacy.consents.evidence_uri) RETURNING id"
            ),
            {"u": user_id, "s": scope, "v": ver[0], "l": language, "e": evidence_uri},
        )
    ).first()
    assert row is not None
    await record(conn, "privacy.consent_grant", scope, actor_user_id=user_id)
    return UUID(str(row[0]))
