"""Refresh-token family policy (pure). Rotation + reuse detection happen in identity.rotate_refresh_token()."""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass
from enum import Enum


class RotationResult(str, Enum):
    ROTATED = "rotated"
    REUSE_DETECTED = "reuse_detected"
    INVALID = "invalid"


@dataclass(frozen=True, slots=True)
class OpaqueToken:
    raw: str
    digest: bytes

    @classmethod
    def new(cls) -> OpaqueToken:
        raw = secrets.token_urlsafe(48)
        return cls(raw, hashlib.sha256(raw.encode()).digest())

    @staticmethod
    def hash(raw: str) -> bytes:
        return hashlib.sha256(raw.encode()).digest()


MAX_FAILED_LOGINS = 10
LOCKOUT_SECONDS = (30, 60, 300, 900, 3600)  # exponential back-off ladder


def lockout_seconds(failures: int) -> int:
    if failures < 5:
        return 0
    return LOCKOUT_SECONDS[min(failures - 5, len(LOCKOUT_SECONDS) - 1)]
