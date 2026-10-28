"""/auth/*, /me, /.well-known/jwks.json"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncConnection

from app.platform.errors import NotFound, RateLimited, problem
from app.platform.principal import Principal
from app.platform.ratelimit import POLICIES, RateLimiter
from app.platform.redis import get_redis

from ..application import guest_join, login, refresh, register, totp
from ..infrastructure.jwt import get_jwt
from ..infrastructure.repositories import SqlUserRepo
from .deps import db, db_anonymous, require_user
from .schemas import GuestJoinIn, GuestToken, LoginIn, MeOut, MePatch, RefreshIn, RegisterIn, TokenPair

router = APIRouter(tags=["auth"])


async def _limit(request: Request, policy: str) -> None:
    rate, burst = POLICIES[policy]
    ip = request.client.host if request.client else "unknown"
    ok, retry = await RateLimiter(get_redis()).allow(f"{policy}:{ip}", rate, burst)
    if not ok:
        raise RateLimited("slow down", retry_after=max(1, int(retry)))


@router.post("/auth/register", status_code=201)
async def post_register(
    body: RegisterIn, request: Request, conn: AsyncConnection = Depends(db_anonymous)
) -> dict[str, UUID]:
    await _limit(request, "auth")
    return {"user_id": await register.execute(conn, body)}


@router.post("/auth/login", response_model=TokenPair)
async def post_login(body: LoginIn, request: Request, conn: AsyncConnection = Depends(db_anonymous)) -> TokenPair:
    await _limit(request, "auth")
    return await login.execute(conn, get_redis(), body)


@router.post("/auth/refresh", response_model=TokenPair, responses={401: {"description": "invalid or reused token"}})
async def post_refresh(body: RefreshIn, request: Request, conn: AsyncConnection = Depends(db_anonymous)) -> Any:
    await _limit(request, "auth")
    out = await refresh.execute(conn, body.refresh_token)
    if isinstance(out, TokenPair):
        return out
    # normal response (not an exception) so the transaction COMMITS the family revocation
    return problem(401, "unauthorized", "refresh token reuse detected; all sessions in this family were revoked")


@router.post("/auth/logout", status_code=204)
async def post_logout(body: RefreshIn, conn: AsyncConnection = Depends(db_anonymous)) -> None:
    await refresh.logout(conn, body.refresh_token)


@router.post("/auth/guest", response_model=GuestToken)
async def post_guest(body: GuestJoinIn, request: Request, conn: AsyncConnection = Depends(db_anonymous)) -> GuestToken:
    await _limit(request, "room_join")
    return await guest_join.execute(conn, body)


@router.post("/auth/2fa/enroll")
async def post_2fa_enroll(p: Principal = Depends(require_user), conn: AsyncConnection = Depends(db)) -> dict[str, str]:
    assert p.user_id
    return {"provisioning_uri": await totp.enroll(conn, p.user_id, str(p.user_id))}


@router.post("/auth/2fa/confirm")
async def post_2fa_confirm(
    code: str, p: Principal = Depends(require_user), conn: AsyncConnection = Depends(db)
) -> dict[str, bool]:
    assert p.user_id
    return {"enabled": await totp.confirm(conn, p.user_id, code)}


@router.get("/me", response_model=MeOut)
async def get_me(p: Principal = Depends(require_user), conn: AsyncConnection = Depends(db)) -> Any:
    assert p.user_id
    me = await SqlUserRepo(conn).me(p.user_id)
    if me is None:
        raise NotFound("user not found")
    return me


@router.patch("/me", response_model=MeOut)
async def patch_me(body: MePatch, p: Principal = Depends(require_user), conn: AsyncConnection = Depends(db)) -> Any:
    assert p.user_id
    fields = body.model_dump(exclude_none=True)
    repo = SqlUserRepo(conn)
    if fields:
        await repo.patch(p.user_id, fields)
    return await repo.me(p.user_id)


@router.get("/.well-known/jwks.json", tags=["auth"])
async def jwks() -> dict[str, Any]:
    return get_jwt().jwks()
