"""/me/consents, /me/export, DELETE /me"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from app.modules.identity.api.deps import db, require_user
from app.platform.principal import Principal

from ..application import erase_me, export_my_data, grant_consent, withdraw_consent

router = APIRouter(tags=["privacy"])


class ConsentIn(BaseModel):
    scope: str
    language: str = Field(default="en", pattern="^(en|sw|ksl)$")
    evidence_uri: str | None = None


@router.get("/me/consents")
async def list_consents(
    p: Principal = Depends(require_user), conn: AsyncConnection = Depends(db)
) -> list[dict[str, Any]]:
    rows = await conn.execute(
        text(
            "SELECT scope::text, text_version, granted_at, withdrawn_at FROM privacy.consents "
            "WHERE user_id = :u ORDER BY granted_at DESC"
        ),
        {"u": p.user_id},
    )
    return [dict(r) for r in rows.mappings()]


@router.post("/me/consents", status_code=201)
async def post_consent(
    body: ConsentIn, p: Principal = Depends(require_user), conn: AsyncConnection = Depends(db)
) -> dict[str, str]:
    assert p.user_id
    return {
        "consent_id": str(await grant_consent.execute(conn, p.user_id, body.scope, body.language, body.evidence_uri))
    }


@router.delete("/me/consents/{scope}")
async def delete_consent(
    scope: str, p: Principal = Depends(require_user), conn: AsyncConnection = Depends(db)
) -> dict[str, int]:
    assert p.user_id
    return {"samples_excluded": await withdraw_consent.execute(conn, p.user_id, scope)}


@router.get("/me/export")
async def export(p: Principal = Depends(require_user), conn: AsyncConnection = Depends(db)) -> dict[str, Any]:
    assert p.user_id
    return await export_my_data.execute(conn, p.user_id)


@router.delete("/me")
async def delete_me(p: Principal = Depends(require_user), conn: AsyncConnection = Depends(db)) -> dict[str, Any]:
    assert p.user_id
    return await erase_me.execute(conn, p.user_id, p.user_id)
