"""Resolve the live model for a task from ml.v_live_models and fetch + verify its artefact (sha256)."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LiveModel:
    task: str
    name: str
    version: str
    format: str
    artifact_uri: str
    is_bootstrap: bool
    sha256: bytes | None
    decoder: dict[str, Any] | None


def _connect():  # type: ignore[no-untyped-def]
    import sqlalchemy as sa

    url = os.environ.get("KSL_DATABASE_URL_SYNC", "postgresql://postgres:postgres@localhost:5432/ksl")
    if url.startswith("postgresql://"):
        url = "postgresql+psycopg://" + url[len("postgresql://") :]
    return sa.create_engine(url, pool_pre_ping=True)


def live_model(task: str) -> LiveModel | None:
    import sqlalchemy as sa

    with _connect().connect() as c:
        row = c.execute(
            sa.text(
                "SELECT d.task::text, mv.name, mv.version, mv.format, mv.artifact_uri, mv.is_bootstrap, mv.sha256, "
                "row_to_json(dc) AS decoder FROM ml.deployments d JOIN ml.model_versions mv ON mv.id = d.model_version_id "
                "LEFT JOIN ml.decoder_configs dc ON dc.id = d.decoder_config_id WHERE d.task = CAST(:t AS shared.model_task) "
                "AND d.target = 'server' AND d.stage = 'production' AND d.ended_at IS NULL"
            ),
            {"t": task},
        ).first()
    if row is None:
        return None
    return LiveModel(row[0], row[1], row[2], row[3], row[4], row[5], bytes(row[6]) if row[6] else None, row[7])


def fetch_artifact(m: LiveModel, cache_dir: str | None = None) -> Path:
    """Download (if needed) into the cache and verify sha256. Supports s3:// and local paths."""
    cache = Path(cache_dir or os.environ.get("KSL_MODEL_CACHE_DIR", "artifacts/model_cache")) / m.task / m.version
    cache.mkdir(parents=True, exist_ok=True)
    if m.artifact_uri.startswith("s3://"):
        import boto3

        bucket, _, key = m.artifact_uri[5:].partition("/")
        dest = cache / Path(key).name
        if not dest.exists():
            boto3.client(
                "s3",
                endpoint_url=os.environ.get("KSL_S3_ENDPOINT"),
                aws_access_key_id=os.environ.get("KSL_S3_ACCESS_KEY"),
                aws_secret_access_key=os.environ.get("KSL_S3_SECRET_KEY"),
            ).download_file(bucket, key, str(dest))
    else:
        dest = Path(m.artifact_uri)
    if m.sha256 and dest.is_file():
        h = hashlib.sha256(dest.read_bytes()).digest()
        if h != m.sha256:
            raise RuntimeError(f"artefact sha256 mismatch for {m.task} {m.version}")
    return dest
