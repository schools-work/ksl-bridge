"""Register an uploaded contribution as a dataset sample and queue landmark extraction + annotation."""

from __future__ import annotations

import json
from typing import Any
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from app.platform.errors import Forbidden, Invalid

from ..infrastructure.repositories import SqlDatasetRepo


async def execute(conn: AsyncConnection, user_id: UUID, body: dict[str, Any]) -> UUID:
    if (
        not str(body.get("object_uri", "")).startswith("s3://")
        or f"/contributions/{user_id}/" not in body["object_uri"]
    ):
        raise Forbidden("object does not belong to the caller")
    repo = SqlDatasetRepo(conn)
    signer_code = await repo.signer_code_for(user_id)
    if signer_code is None:
        raise Invalid("contributor is not registered as a signer (dataset.signers); ask the data team")
    consent_id = await repo.active_contribution_consent(user_id)
    if consent_id is None:
        raise Forbidden("an active contribution consent is required")
    sample_id = await repo.insert_own_sample(signer_code, consent_id, body)
    await conn.execute(
        text("INSERT INTO ops.jobs (kind, payload, created_by) VALUES ('extract_landmarks', CAST(:p AS jsonb), :u)"),
        {"p": json.dumps({"sample_id": str(sample_id)}), "u": user_id},
    )
    await conn.execute(
        text("INSERT INTO annotation.tasks (sample_id, kind, priority) VALUES (:s, 'gloss', 1.0)"), {"s": sample_id}
    )
    return sample_id
