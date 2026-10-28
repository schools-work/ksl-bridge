"""/datasets/uploads, /datasets/samples, /datasets/builds"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncConnection

from app.modules.identity.api.deps import db, require_roles, require_user
from app.platform.principal import Principal

from ..application import build_dataset, create_upload, register_sample

router = APIRouter(prefix="/datasets", tags=["datasets"])


class UploadIn(BaseModel):
    content_type: str


class SampleIn(BaseModel):
    object_uri: str
    kind: str = Field(default="continuous", pattern="^(isolated|continuous|fingerspelling|speech)$")
    domain: str | None = Field(default=None, pattern="^(restaurant|hospital|bank|office|transport|education|general)$")
    session_id: UUID | None = None
    script_english: str | None = Field(default=None, max_length=500)


class BuildIn(BaseModel):
    name: str
    version: str = Field(pattern=r"^v\d+$")
    purpose: str
    params: dict[str, Any] = Field(default_factory=dict)


@router.post("/uploads")
async def post_upload(
    body: UploadIn, p: Principal = Depends(require_user), conn: AsyncConnection = Depends(db)
) -> dict[str, Any]:
    assert p.user_id
    return await create_upload.execute(conn, p.user_id, body.content_type)


@router.post("/samples", status_code=201)
async def post_sample(
    body: SampleIn, p: Principal = Depends(require_user), conn: AsyncConnection = Depends(db)
) -> dict[str, UUID]:
    assert p.user_id
    return {"sample_id": await register_sample.execute(conn, p.user_id, body.model_dump(mode="json"))}


@router.post("/builds", status_code=202)
async def post_build(
    body: BuildIn,
    p: Principal = Depends(require_roles("ml_engineer", "admin", mfa=True)),
    conn: AsyncConnection = Depends(db),
) -> dict[str, UUID]:
    assert p.user_id
    return {"job_id": await build_dataset.execute(conn, p.user_id, body.name, body.version, body.purpose, body.params)}
