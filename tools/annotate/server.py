"""Thin annotation UI backend (mounted at /annotate-ui in dev). Uses only annotation use cases."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from fastapi import Depends, FastAPI
from sqlalchemy.ext.asyncio import AsyncConnection

from app.modules.annotation.application import next_task, submit_annotation
from app.modules.identity.api.deps import db, require_roles
from app.platform.principal import Principal

ui = FastAPI(title="KSL Bridge annotation UI backend")
ANNOTATOR = require_roles("annotator", "deaf_reviewer", mfa=True)


@ui.get("/next")
async def get_next(
    kind: str = "gloss", p: Principal = Depends(ANNOTATOR), conn: AsyncConnection = Depends(db)
) -> dict[str, Any] | None:
    assert p.user_id
    return await next_task.execute(conn, p.user_id, p.roles, kind)


@ui.post("/{task_id}")
async def submit(
    task_id: UUID, body: dict[str, Any], p: Principal = Depends(ANNOTATOR), conn: AsyncConnection = Depends(db)
) -> dict[str, UUID]:
    assert p.user_id
    return {"annotation_id": await submit_annotation.execute(conn, p.user_id, task_id, body)}


def export_eaf(glosses: list[str], times: list[tuple[int, int]], media_url: str) -> str:
    """Minimal ELAN .eaf with one gloss tier (for offline annotation round-trips)."""
    slots, anns = [], []
    for k, (g, (s, e)) in enumerate(zip(glosses, times, strict=True)):
        slots += [
            f'<TIME_SLOT TIME_SLOT_ID="ts{2 * k}" TIME_VALUE="{s}"/>',
            f'<TIME_SLOT TIME_SLOT_ID="ts{2 * k + 1}" TIME_VALUE="{e}"/>',
        ]
        anns.append(
            f'<ANNOTATION><ALIGNABLE_ANNOTATION ANNOTATION_ID="a{k}" TIME_SLOT_REF1="ts{2 * k}" '
            f'TIME_SLOT_REF2="ts{2 * k + 1}"><ANNOTATION_VALUE>{g}</ANNOTATION_VALUE></ALIGNABLE_ANNOTATION></ANNOTATION>'
        )
    return (
        '<?xml version="1.0" encoding="UTF-8"?><ANNOTATION_DOCUMENT FORMAT="3.0" VERSION="3.0">'
        f'<HEADER MEDIA_FILE="" TIME_UNITS="milliseconds"><MEDIA_DESCRIPTOR MEDIA_URL="{media_url}"/></HEADER>'
        f'<TIME_ORDER>{"".join(slots)}</TIME_ORDER><TIER TIER_ID="Gloss" LINGUISTIC_TYPE_REF="default">{"".join(anns)}</TIER>'
        '<LINGUISTIC_TYPE LINGUISTIC_TYPE_ID="default" TIME_ALIGNABLE="true"/></ANNOTATION_DOCUMENT>'
    )
