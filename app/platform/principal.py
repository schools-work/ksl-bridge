"""Who is making the request. Mapped onto the PostgreSQL RLS GUCs by app.platform.db."""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID


@dataclass(frozen=True, slots=True)
class Principal:
    user_id: UUID | None = None
    guest_id: UUID | None = None
    device_id: UUID | None = None
    venue_id: UUID | None = None
    roles: frozenset[str] = field(default_factory=frozenset)
    scope: str | None = None  # e.g. "room:<uuid>" for guest tokens
    mfa: bool = False

    @property
    def subject(self) -> str:
        for kind, v in (("user", self.user_id), ("guest", self.guest_id), ("device", self.device_id)):
            if v is not None:
                return f"{kind}:{v}"
        return "anonymous"

    def has_role(self, *roles: str) -> bool:
        return bool(self.roles & set(roles))


ANONYMOUS = Principal()
SYSTEM = Principal(roles=frozenset({"admin"}))
