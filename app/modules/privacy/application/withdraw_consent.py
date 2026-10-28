"""Withdraw a consent; contribution withdrawals exclude the user's samples from future builds."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from app.platform.audit import record

from ..domain.consent import CONTRIBUTION_SCOPES


async def execute(conn: AsyncConnection, user_id: UUID, scope: str) -> int:
    await conn.execute(
        text(
            "UPDATE privacy.consents SET withdrawn_at = now() WHERE user_id = :u AND scope = CAST(:s AS shared.consent_scope) "
            "AND withdrawn_at IS NULL"
        ),
        {"u": user_id, "s": scope},
    )
    excluded = 0
    if scope in CONTRIBUTION_SCOPES:
        res = await conn.execute(
            text(
                "UPDATE dataset.samples s SET excluded_at = now(), excluded_reason = 'consent_withdrawn' "
                "FROM dataset.signers sg WHERE sg.signer_code = s.signer_code AND sg.user_id = :u AND s.excluded_at IS NULL"
            ),
            {"u": user_id},
        )
        excluded = res.rowcount or 0
    if scope == "personalisation":
        await conn.execute(
            text("UPDATE feedback.personal_adaptations SET enabled = false WHERE user_id = :u"), {"u": user_id}
        )
    await record(conn, "privacy.consent_withdraw", scope, actor_user_id=user_id, meta={"samples_excluded": excluded})
    return excluded
