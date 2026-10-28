"""/admin/audit (admin with 2FA)"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncConnection

from app.modules.identity.api.deps import db, require_roles
from app.platform.principal import Principal

from ..application import query

router = APIRouter(prefix="/admin/audit", tags=["audit"])


@router.get("")
async def list_events(
    actor: UUID | None = None,
    action: str | None = None,
    limit: int = Query(100, le=500),
    p: Principal = Depends(require_roles("admin", mfa=True)),
    conn: AsyncConnection = Depends(db),
) -> list[dict[str, Any]]:
    return await query.execute(conn, actor, action, limit)
