"""Domain-neutral error types → RFC 9457 problem+json responses."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from ksl_core.errors import KslError


class AppError(Exception):
    status = 400
    code = "bad_request"

    def __init__(self, detail: str = "", **extra: object):
        super().__init__(detail or self.code)
        self.detail = detail or self.code
        self.extra = extra


class Unauthorized(AppError):
    status, code = 401, "unauthorized"


class Forbidden(AppError):
    status, code = 403, "forbidden"


class NotFound(AppError):
    status, code = 404, "not_found"


class Conflict(AppError):
    status, code = 409, "conflict"


class Invalid(AppError):
    status, code = 422, "invalid"


class RateLimited(AppError):
    status, code = 429, "rate_limited"


class Unavailable(AppError):
    status, code = 503, "unavailable"


def problem(status: int, code: str, detail: str, **extra: object) -> JSONResponse:
    body = {
        "type": f"https://ksl-bridge.local/problems/{code}",
        "title": code,
        "status": status,
        "detail": detail,
        "error_code": code,
        **extra,
    }
    headers = {"Retry-After": str(int(extra["retry_after"]))} if "retry_after" in extra else None
    return JSONResponse(body, status_code=status, media_type="application/problem+json", headers=headers)


def install_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _app_error(_: Request, exc: AppError) -> JSONResponse:
        return problem(exc.status, exc.code, exc.detail, **exc.extra)

    @app.exception_handler(KslError)
    async def _ksl_error(_: Request, exc: KslError) -> JSONResponse:
        return problem(422, exc.code, str(exc))
