"""Synchronous DB access for training/evaluation jobs (role ksl_trainer)."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any


def sync_url() -> str:
    url = os.environ.get("KSL_DATABASE_URL_SYNC", "postgresql://postgres:postgres@localhost:5432/ksl")
    if url.startswith("postgresql://"):
        url = "postgresql+psycopg://" + url[len("postgresql://") :]
    return url


@lru_cache
def engine() -> Any:
    import sqlalchemy as sa

    return sa.create_engine(sync_url(), pool_pre_ping=True)
