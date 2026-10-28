"""/venues/{id}/kb/documents (ingest, list), /venues/{id}/kb/ask"""

from __future__ import annotations

from dataclasses import replace
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from app.modules.identity.api.deps import current_principal, db, require_user
from app.platform.db import set_principal
from app.platform.errors import Forbidden
from app.platform.principal import Principal

from ..application import answer_question, build_venue_vocabulary, ingest_document

router = APIRouter(prefix="/venues/{venue_id}/kb", tags=["knowledge"])


class DocumentIn(BaseModel):
    kind: str = Field(pattern="^(menu|faq|directions|policy|hours|other)$")
    title: str = Field(min_length=1, max_length=200)
    content: str | list[dict[str, Any]]


class AskIn(BaseModel):
    question: str = Field(min_length=2, max_length=500)


async def _require_staff(conn: AsyncConnection, venue_id: UUID, user_id: UUID) -> None:
    ok = (
        await conn.execute(
            text("SELECT 1 FROM venue.staff WHERE venue_id = :v AND user_id = :u"), {"v": venue_id, "u": user_id}
        )
    ).first()
    if not ok:
        raise Forbidden("venue staff only")


@router.post("/documents", status_code=201)
async def post_document(
    venue_id: UUID, body: DocumentIn, p: Principal = Depends(require_user), conn: AsyncConnection = Depends(db)
) -> dict[str, Any]:
    assert p.user_id
    await _require_staff(conn, venue_id, p.user_id)
    await set_principal(conn, replace(p, venue_id=venue_id))
    doc_id = await ingest_document.execute(conn, venue_id, body.kind, body.title, body.content, p.user_id)
    vocab = await build_venue_vocabulary.execute(conn, venue_id) if body.kind == "menu" else 0
    return {"document_id": doc_id, "venue_vocabulary_terms": vocab}


@router.get("/documents")
async def list_documents(
    venue_id: UUID, p: Principal = Depends(current_principal), conn: AsyncConnection = Depends(db)
) -> list[dict[str, Any]]:
    await set_principal(conn, replace(p, venue_id=venue_id))
    rows = await conn.execute(
        text(
            "SELECT id, kind, title, version, status, created_at FROM knowledge.documents "
            "WHERE venue_id = :v ORDER BY created_at DESC"
        ),
        {"v": venue_id},
    )
    return [dict(r) for r in rows.mappings()]


@router.post("/ask")
async def ask(
    venue_id: UUID, body: AskIn, p: Principal = Depends(current_principal), conn: AsyncConnection = Depends(db)
) -> dict[str, Any]:
    await set_principal(conn, replace(p, venue_id=venue_id))  # menus/FAQs are public within the venue
    return await answer_question.execute(conn, venue_id, body.question)
