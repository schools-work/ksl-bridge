"""/annotate/next, /annotate/{task_id}, /annotate/adjudicate"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncConnection

from app.modules.identity.api.deps import db, require_roles
from app.platform.principal import Principal

from ..application import adjudicate, next_task, submit_annotation

router = APIRouter(prefix="/annotate", tags=["annotation"])


class AnnotationIn(BaseModel):
    glosses: list[str] | None = None
    gloss_times: list[tuple[int, int]] | None = None
    text_en: str | None = Field(default=None, max_length=1000)
    text_sw: str | None = Field(default=None, max_length=1000)
    non_manual: dict[str, Any] | None = None
    self_confidence: int | None = Field(default=None, ge=1, le=5)
    time_spent_s: int | None = Field(default=None, ge=0)


class AdjudicateIn(BaseModel):
    sample_id: UUID
    final_glosses: list[str]
    final_text_en: str | None = None
    chosen_id: UUID | None = None
    notes: str | None = None


@router.get("/next")
async def get_next(
    kind: str | None = None,
    p: Principal = Depends(require_roles("annotator", "deaf_reviewer", mfa=True)),
    conn: AsyncConnection = Depends(db),
) -> dict[str, Any] | None:
    assert p.user_id
    return await next_task.execute(conn, p.user_id, p.roles, kind)


@router.post("/{task_id}", status_code=201)
async def post_annotation(
    task_id: UUID,
    body: AnnotationIn,
    p: Principal = Depends(require_roles("annotator", "deaf_reviewer", mfa=True)),
    conn: AsyncConnection = Depends(db),
) -> dict[str, UUID]:
    assert p.user_id
    return {"annotation_id": await submit_annotation.execute(conn, p.user_id, task_id, body.model_dump())}


@router.post("/adjudicate", status_code=204)
async def post_adjudicate(
    body: AdjudicateIn,
    p: Principal = Depends(require_roles("deaf_reviewer", mfa=True)),
    conn: AsyncConnection = Depends(db),
) -> None:
    assert p.user_id
    await adjudicate.execute(
        conn, p.user_id, body.sample_id, body.final_glosses, body.final_text_en, body.chosen_id, body.notes
    )
