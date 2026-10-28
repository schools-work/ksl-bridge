"""Room entity: join code, lifecycle."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID

JOIN_CODE = re.compile(r"^[A-HJ-NP-Z2-9]{3}-[A-HJ-NP-Z2-9]{3}$")


def normalize_code(code: str) -> str:
    c = code.strip().upper().replace(" ", "")
    if "-" not in c and len(c) == 6:
        c = f"{c[:3]}-{c[3:]}"
    return c


def qr_payload(join_code: str) -> str:
    return f"kslbridge://join?c={join_code}"


@dataclass(slots=True)
class Room:
    id: UUID
    join_code: str
    status: str
    context_profile: str
    store_transcript: bool
    expires_at: datetime
    venue_id: UUID | None = None

    def is_joinable(self, now: datetime | None = None) -> bool:
        return self.status == "open" and self.expires_at > (now or datetime.now(UTC))
