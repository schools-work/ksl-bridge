"""User entity and value objects (framework-free)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from uuid import UUID


class Role(str, Enum):
    SIGNER = "signer"
    SPEAKER = "speaker"
    OPERATOR = "operator"
    ANNOTATOR = "annotator"
    DEAF_REVIEWER = "deaf_reviewer"
    ML_ENGINEER = "ml_engineer"
    ADMIN = "admin"


MFA_REQUIRED_ROLES = frozenset({Role.ADMIN, Role.ANNOTATOR, Role.DEAF_REVIEWER, Role.ML_ENGINEER})
SELF_SERVICE_ROLES = frozenset({Role.SIGNER, Role.SPEAKER})

_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_PHONE = re.compile(r"^\+[1-9][0-9]{7,14}$")


def validate_email(v: str) -> str:
    v = v.strip().lower()
    if not _EMAIL.match(v) or len(v) > 254:
        raise ValueError("invalid email")
    return v


def validate_phone(v: str) -> str:
    v = v.replace(" ", "")
    if v.startswith("07") or v.startswith("01"):
        v = "+254" + v[1:]  # Kenyan local format → E.164
    if not _PHONE.match(v):
        raise ValueError("invalid phone number (E.164)")
    return v


def validate_password(pw: str) -> str:
    if len(pw) < 10 or pw.lower() == pw or not any(c.isdigit() for c in pw):
        raise ValueError("password must be ≥ 10 characters with an upper-case letter and a digit")
    return pw


@dataclass(slots=True)
class User:
    id: UUID
    display_name: str
    email: str | None = None
    phone_e164: str | None = None
    roles: frozenset[Role] = field(default_factory=frozenset)
    preferred_in: str = "text"
    preferred_out: str = "text"
    dominant_hand: str | None = None
    sign_speed: float = 1.0
    status: str = "active"

    @property
    def requires_mfa(self) -> bool:
        return bool(self.roles & MFA_REQUIRED_ROLES)
