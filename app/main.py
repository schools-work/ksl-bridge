"""FastAPI application factory + lifespan (DB, Redis, room bus, job client, outbox relay)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app

from app.modules.agent.api.router import router as agent_router
from app.modules.annotation.api.router import router as annotation_router
from app.modules.audit.api.router import router as audit_router
from app.modules.conversation.api.router import router as rooms_router
from app.modules.dataset.api.router import router as dataset_router
from app.modules.feedback.api.router import router as feedback_router
from app.modules.identity.api.router import router as auth_router
from app.modules.knowledge.api.router import router as kb_router
from app.modules.lexicon.api.router import router as lexicon_router
from app.modules.mlops.api.mobile import router as mobile_router
from app.modules.mlops.api.router import router as mlops_router
from app.modules.pipeline.infrastructure.job_client import RedisJobClient
from app.modules.privacy.api.router import router as privacy_router
from app.modules.realtime.endpoint import router as ws_router
from app.modules.realtime.local_registry import LocalRegistry
from app.modules.realtime.room_bus import RoomBus
from app.modules.venue.api.device_auth import router as device_auth_router
from app.modules.venue.api.router import router as venues_router
from app.platform.db import dispose_engine
from app.platform.errors import install_handlers
from app.platform.health import router as health_router
from app.platform.logging import configure_logging
from app.platform.outbox import relay_forever
from app.platform.redis import close_redis, get_redis
from app.platform.security_headers import SecurityHeadersMiddleware
from app.platform.tracing import configure_tracing
from app.settings import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    s = get_settings()
    configure_logging(s.log_level)
    configure_tracing(s.otel_exporter_otlp_endpoint)
    app.state.redis = get_redis()
    app.state.local = LocalRegistry()
    app.state.bus = RoomBus(app.state.redis, app.state.local)
    app.state.jobs = RedisJobClient(app.state.redis)
    await app.state.bus.start()
    await app.state.jobs.start()
    relay = asyncio.create_task(relay_forever(), name="outbox-relay") if s.env != "test" else None
    try:
        yield
    finally:
        if relay:
            relay.cancel()
        await app.state.jobs.stop()
        await app.state.bus.stop()
        await close_redis()
        await dispose_engine()


def create_app() -> FastAPI:
    s = get_settings()
    app = FastAPI(
        title="KSL Bridge API",
        version="1.0.0",
        lifespan=lifespan,
        description="Two-way Kenyan Sign Language ⇄ speech/text communication bridge. "
        "Real-time protocol: see contracts/realtime/ws_protocol_v1.json.",
    )
    app.add_middleware(SecurityHeadersMiddleware, hsts=s.is_production)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=s.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "PATCH", "DELETE"],
        allow_headers=["Authorization", "Content-Type", "Idempotency-Key", "X-Device-Key"],
    )
    install_handlers(app)
    for r in (
        health_router,
        auth_router,
        venues_router,
        device_auth_router,
        rooms_router,
        privacy_router,
        dataset_router,
        annotation_router,
        lexicon_router,
        kb_router,
        feedback_router,
        agent_router,
        mlops_router,
        mobile_router,
        audit_router,
        ws_router,
    ):
        app.include_router(r)
    if s.metrics_enabled:
        app.mount("/metrics", make_asgi_app())
    return app


app = create_app()
