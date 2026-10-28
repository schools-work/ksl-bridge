"""Async SQLAlchemy engine + per-request transactions that set the RLS GUCs (fail-closed RLS)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, create_async_engine

from app.settings import get_settings

from .principal import Principal

_engine: AsyncEngine | None = None


def get_engine() -> AsyncEngine:
    global _engine
    if _engine is None:
        s = get_settings()
        _engine = create_async_engine(s.database_url, pool_size=10, max_overflow=20, pool_pre_ping=True)
    return _engine


async def dispose_engine() -> None:
    global _engine
    if _engine is not None:
        await _engine.dispose()
        _engine = None


_SET_GUCS = text(
    "SELECT set_config('app.current_user_id', :uid, true), set_config('app.current_roles', :roles, true), "
    "set_config('app.current_venue_id', :vid, true), set_config('app.current_guest_id', :gid, true), "
    "set_config('app.current_device_id', :did, true)"
)


@asynccontextmanager
async def transaction(principal: Principal) -> AsyncIterator[AsyncConnection]:
    """One transaction with RLS GUCs set for `principal`. Commits on success, rolls back on error."""
    async with get_engine().begin() as conn:
        await conn.execute(
            _SET_GUCS,
            {
                "uid": str(principal.user_id or ""),
                "roles": ",".join(sorted(principal.roles)),
                "vid": str(principal.venue_id or ""),
                "gid": str(principal.guest_id or ""),
                "did": str(principal.device_id or ""),
            },
        )
        yield conn


async def set_principal(conn: AsyncConnection, principal: Principal) -> None:
    """Switch identity inside an open transaction (e.g. right after creating a user)."""
    await conn.execute(
        _SET_GUCS,
        {
            "uid": str(principal.user_id or ""),
            "roles": ",".join(sorted(principal.roles)),
            "vid": str(principal.venue_id or ""),
            "gid": str(principal.guest_id or ""),
            "did": str(principal.device_id or ""),
        },
    )
