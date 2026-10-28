"""Password login with lockout and TOTP for privileged roles. Issues an access + refresh token pair."""

from __future__ import annotations

import uuid

import redis.asyncio as aioredis
from sqlalchemy.ext.asyncio import AsyncConnection

from app.platform.errors import RateLimited, Unauthorized
from app.platform.metrics import AUTH_EVENTS
from app.platform.principal import Principal
from app.settings import get_settings

from ..domain.tokens import OpaqueToken, lockout_seconds
from ..domain.user import MFA_REQUIRED_ROLES, Role, validate_email, validate_phone
from ..infrastructure.jwt import get_jwt
from ..infrastructure.passwords import Argon2Hasher
from ..infrastructure.repositories import SqlTokenRepo, SqlUserRepo
from . import totp as totp_uc
from .dto import LoginIn, TokenPair

GENERIC = "invalid credentials"


async def execute(conn: AsyncConnection, redis: aioredis.Redis, body: LoginIn) -> TokenPair:
    s = get_settings()
    try:
        email = validate_email(body.email) if body.email else None
        phone = validate_phone(body.phone) if body.phone else None
    except ValueError as e:
        raise Unauthorized(GENERIC) from e
    ident = email or phone or ""
    fails_key = f"auth:fails:{ident}"
    fails = int(await redis.get(fails_key) or 0)
    wait = lockout_seconds(fails)
    if wait and await redis.ttl(fails_key) > 0 and fails >= 5:
        raise RateLimited("too many failed attempts", retry_after=wait)

    users, hasher = SqlUserRepo(conn), Argon2Hasher()
    found = await users.find_login(email, phone)
    ok, rehash = (False, None)
    if found and found[1] and found[2] == "active":
        ok, rehash = hasher.verify(found[1], body.password)
    else:
        hasher.verify(hasher.dummy_hash, body.password)  # constant work: no user enumeration by timing
    if not ok or found is None:
        await redis.incr(fails_key)
        await redis.expire(fails_key, 3600)
        AUTH_EVENTS.labels("login_failed").inc()
        raise Unauthorized(GENERIC)

    user_id, _, _, roles = found
    roles_set = frozenset(roles)
    mfa = False
    if roles_set & {r.value for r in MFA_REQUIRED_ROLES}:
        if not body.totp or not await totp_uc.verify(conn, user_id, body.totp):
            raise Unauthorized("two-factor code required")
        mfa = True
    await redis.delete(fails_key)
    principal = Principal(user_id=user_id, roles=roles_set, mfa=mfa)
    from app.platform.db import set_principal

    await set_principal(conn, principal)
    if rehash:
        await users.update_password_hash(user_id, rehash)
    await users.touch_login(user_id)
    refresh = OpaqueToken.new()
    await SqlTokenRepo(conn).insert_refresh(
        user_id, uuid.uuid4(), refresh.digest, body.device_label, s.refresh_ttl_days
    )
    AUTH_EVENTS.labels("login_ok").inc()
    return TokenPair(
        access_token=get_jwt().issue(principal), refresh_token=refresh.raw, expires_in=s.access_ttl_seconds
    )


def is_self_service(role: str) -> bool:
    return role in (Role.SIGNER.value, Role.SPEAKER.value)
