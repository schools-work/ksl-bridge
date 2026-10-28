"""Test configuration: loads .env.test, adds markers, shared fixtures."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for line in (ROOT / ".env.test").read_text().splitlines():
    if line and not line.startswith("#") and "=" in line:
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.split("#", 1)[0].strip())
# allow CI/local overrides
if os.environ.get("KSL_TEST_DATABASE_URL"):
    os.environ["KSL_DATABASE_URL"] = os.environ["KSL_TEST_DATABASE_URL"]
if os.environ.get("KSL_TEST_REDIS_URL"):
    os.environ["KSL_REDIS_URL"] = os.environ["KSL_TEST_REDIS_URL"]

# ------------------------------------------------ integration fixtures (integration / security / chaos)
import subprocess  # noqa: E402
import uuid  # noqa: E402
from collections.abc import Iterator  # noqa: E402

import pytest  # noqa: E402


def _psql(url: str, *args: str) -> None:
    subprocess.run(["psql", url, "-v", "ON_ERROR_STOP=1", "-q", *args], check=True, capture_output=True)


@pytest.fixture(scope="session")
def database_url() -> Iterator[str]:
    base = os.environ.get("KSL_TEST_DATABASE_URL")
    if not base:
        tc = pytest.importorskip("testcontainers.postgres")
        container = tc.PostgresContainer("pgvector/pgvector:pg16")
        container.start()
        base = container.get_connection_url().replace("psycopg2", "asyncpg")
    sync = base.replace("+asyncpg", "")
    admin = sync.rsplit("/", 1)[0] + "/postgres"
    name = f"ksl_it_{uuid.uuid4().hex[:8]}"
    _psql(admin, "-c", f"CREATE DATABASE {name}")
    url = sync.rsplit("/", 1)[0] + f"/{name}"
    _psql(url, "-f", str(ROOT / "db/ksl_bridge_v1.sql"))
    _psql(url, "-v", "app_password='it-pass'", "-f", str(ROOT / "db/seeds/dev_users.sql"))
    _psql(url, "-f", str(ROOT / "db/seeds/demo_restaurant.sql"))
    host = url.split("@", 1)[1]
    app_url = f"postgresql+asyncpg://ksl_app_login:it-pass@{host}"
    os.environ["KSL_DATABASE_URL"] = app_url
    os.environ["KSL_DATABASE_URL_SYNC"] = url
    yield app_url
    _psql(admin, "-c", f"DROP DATABASE IF EXISTS {name} WITH (FORCE)")


@pytest.fixture()
def client(database_url: str) -> Iterator[object]:
    import redis as redis_sync
    from fastapi.testclient import TestClient

    # fresh rate-limit buckets / streams per test (auth is limited to 5/min per client IP)
    redis_sync.from_url(os.environ.get("KSL_REDIS_URL", "redis://localhost:6379/15")).flushdb()

    from app.platform import db as dbmod
    from app.platform import redis as redismod
    from app.settings import get_settings

    get_settings.cache_clear()
    dbmod._engine = None
    redismod._client = None
    from app.main import create_app

    with TestClient(create_app()) as c:
        yield c
