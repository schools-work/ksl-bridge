"""OTP delivery adapters. 'console' (dev only) logs codes; production uses an SMS gateway (non-AI)."""

from __future__ import annotations

import hashlib
import hmac
import secrets

import httpx

from app.platform.logging import get_logger
from app.settings import get_settings

log = get_logger(__name__)


def new_code() -> str:
    return f"{secrets.randbelow(10**6):06d}"


def hash_code(code: str) -> bytes:
    return hmac.new(get_settings().otp_pepper.encode(), code.encode(), hashlib.sha256).digest()


class ConsoleOtpSender:
    async def send(self, target: str, code: str, purpose: str) -> None:
        if get_settings().is_production:
            raise RuntimeError("console OTP sender is not allowed in production")
        print(f"[DEV OTP] {purpose} code for {target}: {code}")


class HttpSmsOtpSender:
    """Generic REST SMS gateway (configure provider URL/key per deployment)."""

    def __init__(self, base_url: str, api_key: str, sender_id: str):
        self.base_url, self.api_key, self.sender_id = base_url, api_key, sender_id

    async def send(self, target: str, code: str, purpose: str) -> None:
        async with httpx.AsyncClient(timeout=5) as c:
            r = await c.post(
                self.base_url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "to": target,
                    "from": self.sender_id,
                    "message": f"KSL Bridge code: {code}. It expires in 5 minutes.",
                },
            )
            r.raise_for_status()
        log.info("otp_sent", purpose=purpose)


def get_sender() -> ConsoleOtpSender | HttpSmsOtpSender:
    s = get_settings()
    if s.sms_provider == "console":
        return ConsoleOtpSender()
    return HttpSmsOtpSender(s.sms_provider, s.sms_api_key, s.sms_sender_id)
