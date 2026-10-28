"""/admin/models/* (ml_engineer/admin with 2FA)"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncConnection

from app.modules.identity.api.deps import db, require_roles
from app.platform.principal import Principal

from ..application import list_models, promote, record_gate, rollback

router = APIRouter(prefix="/admin/models", tags=["mlops"])
ML = require_roles("ml_engineer", "admin", mfa=True)


class GateIn(BaseModel):
    gate: str = Field(pattern="^(quality|no_regression|fairness|calibration|latency|safety_set|licence|human_review)$")
    passed: bool
    details: dict[str, Any] = Field(default_factory=dict)
    evaluation_id: UUID | None = None


class PromoteIn(BaseModel):
    target: str = Field(default="server", pattern="^(server|mobile)$")
    decoder_config_id: UUID | None = None
    min_app_version: str | None = None


@router.get("/live")
async def live(p: Principal = Depends(ML), conn: AsyncConnection = Depends(db)) -> list[dict[str, Any]]:
    return await list_models.live(conn)


@router.get("/leaderboard/{task}")
async def leaderboard(
    task: str, metric: str = "wer", p: Principal = Depends(ML), conn: AsyncConnection = Depends(db)
) -> list[dict[str, Any]]:
    return await list_models.leaderboard(conn, task, metric)


@router.post("/{model_id}/gates", status_code=204)
async def gate(model_id: UUID, body: GateIn, p: Principal = Depends(ML), conn: AsyncConnection = Depends(db)) -> None:
    assert p.user_id
    await record_gate.execute(conn, model_id, body.gate, body.passed, body.details, p.user_id, body.evaluation_id)


@router.post("/{model_id}/promote")
async def promote_model(
    model_id: UUID, body: PromoteIn, p: Principal = Depends(ML), conn: AsyncConnection = Depends(db)
) -> dict[str, UUID]:
    assert p.user_id
    return {
        "deployment_id": await promote.execute(
            conn, model_id, body.target, p.user_id, body.decoder_config_id, body.min_app_version
        )
    }


@router.post("/rollback/{task}")
async def do_rollback(
    task: str, target: str = "server", p: Principal = Depends(ML), conn: AsyncConnection = Depends(db)
) -> dict[str, UUID]:
    assert p.user_id
    return {"deployment_id": await rollback.execute(conn, task, target, p.user_id)}
