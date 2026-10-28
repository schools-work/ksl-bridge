"""Request/response DTOs for identity use cases."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, Field


class RegisterIn(BaseModel):
    display_name: str = Field(min_length=1, max_length=80)
    email: str | None = None
    phone: str | None = None
    password: str | None = None
    role: str = Field(default="speaker", pattern="^(signer|speaker)$")
    preferred_in: str = Field(default="text", pattern="^(sign|speech|text)$")
    preferred_out: str = Field(default="text", pattern="^(sign|speech|text)$")
    dominant_hand: str | None = Field(default=None, pattern="^(left|right|ambidextrous)$")


class LoginIn(BaseModel):
    email: str | None = None
    phone: str | None = None
    password: str = Field(min_length=1, max_length=256)
    device_label: str | None = Field(default=None, max_length=80)
    totp: str | None = Field(default=None, pattern=r"^\d{6}$")


class TokenPair(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"  # noqa: S105 (OAuth token type, not a secret)
    expires_in: int


class RefreshIn(BaseModel):
    refresh_token: str = Field(min_length=20, max_length=200)


class GuestJoinIn(BaseModel):
    join_code: str = Field(pattern=r"^[A-Za-z0-9]{3}-?[A-Za-z0-9]{3}$")
    display_name: str = Field(default="Guest", max_length=80)


class GuestToken(BaseModel):
    access_token: str
    guest_id: UUID
    room_id: UUID
    expires_in: int


class MeOut(BaseModel):
    id: UUID
    display_name: str
    email: str | None
    phone_e164: str | None
    roles: list[str]
    preferred_in: str
    preferred_out: str
    dominant_hand: str | None
    sign_speed: float
    caption_scale: float
    locale: str


class MePatch(BaseModel):
    display_name: str | None = Field(default=None, min_length=1, max_length=80)
    preferred_in: str | None = Field(default=None, pattern="^(sign|speech|text)$")
    preferred_out: str | None = Field(default=None, pattern="^(sign|speech|text)$")
    dominant_hand: str | None = Field(default=None, pattern="^(left|right|ambidextrous)$")
    sign_speed: float | None = Field(default=None, ge=0.5, le=1.5)
    caption_scale: float | None = Field(default=None, ge=0.75, le=2.5)
    locale: str | None = Field(default=None, pattern="^(en-KE|sw-KE)$")
