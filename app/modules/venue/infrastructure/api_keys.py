"""Device API keys: kslb_dev_<prefix8>_<secret>; stored as HMAC-SHA256(pepper, key)."""

from __future__ import annotations

import hashlib
import hmac
import secrets

from app.settings import get_settings


class HmacDeviceKeys:
    def __init__(self, pepper: str | None = None):
        self.pepper = (pepper or get_settings().api_key_pepper).encode()

    def digest(self, key: str) -> bytes:
        return hmac.new(self.pepper, key.encode(), hashlib.sha256).digest()

    def new_key(self) -> tuple[str, str, bytes]:
        prefix = secrets.token_hex(4)
        key = f"kslb_dev_{prefix}_{secrets.token_urlsafe(32)}"
        return key, prefix, self.digest(key)

    @staticmethod
    def prefix_of(key: str) -> str | None:
        parts = key.split("_")
        if len(parts) < 4 or parts[0] != "kslb" or parts[1] != "dev" or len(parts[2]) != 8:
            return None
        return parts[2]

    def verify(self, presented: str, stored_hash: bytes) -> bool:
        return hmac.compare_digest(self.digest(presented), stored_hash)
