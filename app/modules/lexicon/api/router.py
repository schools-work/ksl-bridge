"""/dictionary (search), /dictionary/{gloss}/pose, review endpoints for Deaf reviewers."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncConnection

from app.modules.identity.api.deps import current_principal, db, require_roles
from app.platform.errors import NotFound
from app.platform.principal import Principal
from app.platform.s3 import presigned_get, split_uri

from ..application import approve_gloss, build_medoids, search_dictionary

router = APIRouter(prefix="/dictionary", tags=["dictionary"])


class ReviewIn(BaseModel):
    approve: bool
    notes: str | None = None


@router.get("")
async def search(
    q: str = Query(min_length=1, max_length=60),
    p: Principal = Depends(current_principal),
    conn: AsyncConnection = Depends(db),
) -> list[dict[str, Any]]:
    return await search_dictionary.execute(conn, q)


@router.get("/{gloss}/pose")
async def pose(
    gloss: str, p: Principal = Depends(current_principal), conn: AsyncConnection = Depends(db)
) -> dict[str, str]:
    uri = await search_dictionary.pose_uri(conn, gloss.upper())
    if uri is None:
        raise NotFound("no approved clip for this gloss yet")
    return {"url": presigned_get(*split_uri(uri))}


@router.post("/{gloss}/review", status_code=204)
async def review(
    gloss: str,
    body: ReviewIn,
    p: Principal = Depends(require_roles("deaf_reviewer", mfa=True)),
    conn: AsyncConnection = Depends(db),
) -> None:
    await approve_gloss.execute(conn, p, gloss.upper(), body.approve, body.notes)


@router.post("/medoids/rebuild", status_code=202)
async def rebuild(
    p: Principal = Depends(require_roles("ml_engineer", "admin", mfa=True)), conn: AsyncConnection = Depends(db)
) -> dict[str, str]:
    assert p.user_id
    return {"job_id": str(await build_medoids.execute(conn, p.user_id))}
