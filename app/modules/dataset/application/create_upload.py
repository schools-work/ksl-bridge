"""Presigned upload URL for a contribution (requires an active contribution consent)."""

from __future__ import annotations

import uuid
from typing import Any
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from app.platform.errors import Forbidden, Invalid
from app.platform.s3 import presigned_put
from app.settings import get_settings

ALLOWED = {"video/mp4": "mp4", "video/quicktime": "mov", "application/octet-stream": "kslb", "audio/wav": "wav"}
SCOPE_FOR = {
    "mp4": "contribute_video",
    "mov": "contribute_video",
    "kslb": "contribute_landmarks",
    "wav": "contribute_audio",
}


async def execute(conn: AsyncConnection, user_id: UUID, content_type: str) -> dict[str, Any]:
    ext = ALLOWED.get(content_type)
    if ext is None:
        raise Invalid("unsupported content type")
    ok = (
        await conn.execute(
            text("SELECT privacy.has_consent(:u, CAST(:s AS shared.consent_scope))"),
            {"u": user_id, "s": SCOPE_FOR[ext]},
        )
    ).scalar_one()
    if not ok:
        raise Forbidden(f"consent '{SCOPE_FOR[ext]}' is required before uploading")
    key = f"contributions/{user_id}/{uuid.uuid4().hex}.{ext}"
    bucket = get_settings().s3_bucket_datasets
    return {
        "upload_url": presigned_put(bucket, key, content_type),
        "object_uri": f"s3://{bucket}/{key}",
        "expires_in": 900,
    }
