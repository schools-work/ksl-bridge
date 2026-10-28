"""FastAPI dependencies: authenticate the bearer token, enforce roles/MFA, open an RLS transaction."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Coroutine
from typing import Any

from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncConnection

from app.platform.db import transaction
from app.platform.errors import Forbidden, Unauthorized
from app.platform.principal import ANONYMOUS, Principal

from ..infrastructure.jwt import get_jwt

_bearer = HTTPBearer(auto_error=False)


async def current_principal(cred: HTTPAuthorizationCredentials | None = Depends(_bearer)) -> Principal:
    if cred is None:
        raise Unauthorized("missing bearer token")
    try:
        return get_jwt().decode(cred.credentials)
    except Exception as e:
        raise Unauthorized("invalid token") from e


async def optional_principal(cred: HTTPAuthorizationCredentials | None = Depends(_bearer)) -> Principal:
    if cred is None:
        return ANONYMOUS
    return await current_principal(cred)


def require_roles(*roles: str, mfa: bool = False) -> Callable[..., Coroutine[Any, Any, Principal]]:
    async def dep(p: Principal = Depends(current_principal)) -> Principal:
        if not p.has_role(*roles):
            raise Forbidden("insufficient role")
        if mfa and not p.mfa:
            raise Forbidden("two-factor authentication required")
        return p

    return dep


async def require_user(p: Principal = Depends(current_principal)) -> Principal:
    if p.user_id is None:
        raise Forbidden("a user account is required")
    return p


async def db(p: Principal = Depends(current_principal)) -> AsyncIterator[AsyncConnection]:
    async with transaction(p) as conn:
        yield conn


async def db_anonymous() -> AsyncIterator[AsyncConnection]:
    async with transaction(ANONYMOUS) as conn:
        yield conn
