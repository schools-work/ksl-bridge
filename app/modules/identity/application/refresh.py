"""Refresh with rotation; reuse of a rotated token revokes the whole family (identity.rotate_refresh_token).

IMPORTANT: on reuse we must NOT raise inside the request transaction: raising rolls back, and that would undo the
family revocation performed by the database function. The use case returns a RotationResult and the router answers
401 with a normal response, so the transaction commits the revocation.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncConnection

from app.platform.errors import Unauthorized
from app.platform.metrics import AUTH_EVENTS
from app.platform.principal import Principal
from app.settings import get_settings

from ..domain.tokens import OpaqueToken, RotationResult
from ..infrastructure.jwt import get_jwt
from ..infrastructure.repositories import SqlTokenRepo, SqlUserRepo
from .dto import TokenPair


async def execute(conn: AsyncConnection, raw_refresh: str) -> TokenPair | RotationResult:
    s = get_settings()
    new = OpaqueToken.new()
    result, user_id = await SqlTokenRepo(conn).rotate(OpaqueToken.hash(raw_refresh), new.digest, s.refresh_ttl_days)
    if result is RotationResult.REUSE_DETECTED:
        AUTH_EVENTS.labels("refresh_reuse").inc()
        return result  # commit the revocation; the router responds 401
    if result is not RotationResult.ROTATED or user_id is None:
        raise Unauthorized("invalid refresh token")
    roles = await SqlUserRepo(conn).roles_of(user_id)
    principal = Principal(user_id=user_id, roles=frozenset(roles))
    AUTH_EVENTS.labels("refresh_ok").inc()
    return TokenPair(access_token=get_jwt().issue(principal), refresh_token=new.raw, expires_in=s.access_ttl_seconds)


async def logout(conn: AsyncConnection, raw_refresh: str) -> None:
    await SqlTokenRepo(conn).revoke_family_of(OpaqueToken.hash(raw_refresh), "logout")
