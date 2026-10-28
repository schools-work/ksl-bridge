"""GET /models/mobile/manifest: what the Kotlin app should download for offline mode."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from app.modules.identity.api.deps import current_principal, db
from app.platform.principal import Principal
from app.platform.s3 import presigned_get, split_uri

router = APIRouter(tags=["mobile"])


@router.get("/models/mobile/manifest")
async def manifest(p: Principal = Depends(current_principal), conn: AsyncConnection = Depends(db)) -> dict[str, Any]:
    rows = await conn.execute(
        text(
            "SELECT d.task::text AS task, mv.name, mv.version, mv.format, mv.artifact_uri, encode(mv.sha256, 'hex') AS sha256, "
            "mv.size_bytes, d.min_app_version FROM ml.deployments d JOIN ml.model_versions mv ON mv.id = d.model_version_id "
            "WHERE d.target = 'mobile' AND d.stage = 'production' AND d.ended_at IS NULL"
        )
    )
    models = []
    for r in rows.mappings():
        item = dict(r)
        try:
            item["url"] = presigned_get(*split_uri(item.pop("artifact_uri")), expires=3600)
        except Exception:
            item["url"] = None
        models.append(item)
    return {"manifest_version": 1, "landmark_spec_version": 1, "wifi_only": True, "models": models}
