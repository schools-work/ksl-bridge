"""/feedback (REST alternative to the WS message), /feedback/needed-signs"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncConnection

from app.modules.identity.api.deps import db, require_roles, require_user
from app.platform.principal import Principal

from ..application import problem_generator, record_feedback

router = APIRouter(prefix="/feedback", tags=["feedback"])


class FeedbackIn(BaseModel):
    utterance_id: str
    kind: str = Field(pattern="^(confirm|correct|reject|clarify_choice)$")
    corrected_glosses: list[str] | None = None
    corrected_text: str | None = Field(default=None, max_length=2000)


@router.post("", status_code=201)
async def post_feedback(
    body: FeedbackIn, p: Principal = Depends(require_user), conn: AsyncConnection = Depends(db)
) -> dict[str, Any]:
    assert p.user_id
    fid = await record_feedback.execute(
        conn, p.user_id, body.utterance_id, body.kind, body.corrected_glosses, body.corrected_text
    )
    return {"feedback_id": fid, "stored": fid is not None}


@router.get("/needed-signs")
async def needed(
    p: Principal = Depends(require_roles("ml_engineer", "admin", "annotator")), conn: AsyncConnection = Depends(db)
) -> list[dict[str, Any]]:
    return await problem_generator.execute(conn)
