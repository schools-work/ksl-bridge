"""Room member roles and their default modalities."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

ROLES = ("signer", "speaker", "device", "observer")
DEFAULT_MODALITIES = {
    "signer": ("sign", "sign"),
    "speaker": ("speech", "text"),
    "device": ("device", "text"),
    "observer": ("text", "text"),
}


@dataclass(frozen=True, slots=True)
class Member:
    id: UUID
    room_id: UUID
    role: str
    modality_in: str
    modality_out: str
    user_id: UUID | None = None
    guest_id: UUID | None = None
    device_id: UUID | None = None

    @property
    def wants_sign(self) -> bool:
        return self.modality_out == "sign"

    @property
    def wants_speech(self) -> bool:
        return self.modality_out == "speech"
