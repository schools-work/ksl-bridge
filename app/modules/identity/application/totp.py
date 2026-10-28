"""TOTP 2FA for privileged roles (admin, annotator, deaf_reviewer, ml_engineer). Replay-protected."""

from __future__ import annotations

import time
from uuid import UUID

import pyotp
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from ..infrastructure.crypto import decrypt, encrypt


async def enroll(conn: AsyncConnection, user_id: UUID, account_name: str) -> str:
    """Create a secret and return the provisioning URI (show as a QR code once)."""
    secret = pyotp.random_base32()
    await conn.execute(
        text(
            "INSERT INTO identity.totp_credentials (user_id, secret_enc) VALUES (:u, :s) "
            "ON CONFLICT (user_id) DO UPDATE SET secret_enc = EXCLUDED.secret_enc, enabled_at = NULL"
        ),
        {"u": user_id, "s": encrypt(secret.encode())},
    )
    return pyotp.TOTP(secret).provisioning_uri(name=account_name, issuer_name="KSL Bridge")


async def confirm(conn: AsyncConnection, user_id: UUID, code: str) -> bool:
    if await verify(conn, user_id, code, require_enabled=False):
        await conn.execute(
            text("UPDATE identity.totp_credentials SET enabled_at = now() WHERE user_id = :u"), {"u": user_id}
        )
        return True
    return False


async def verify(conn: AsyncConnection, user_id: UUID, code: str, require_enabled: bool = True) -> bool:
    row = (
        (
            await conn.execute(
                text(
                    "SELECT secret_enc, enabled_at, last_used_step FROM identity.totp_credentials WHERE user_id = :u FOR UPDATE"
                ),
                {"u": user_id},
            )
        )
        .mappings()
        .first()
    )
    if row is None or (require_enabled and row["enabled_at"] is None):
        return False
    totp = pyotp.TOTP(decrypt(bytes(row["secret_enc"])).decode())
    step = int(time.time() // 30)
    for candidate in (step - 1, step, step + 1):
        if candidate <= (row["last_used_step"] or -1):
            continue
        if totp.at(candidate * 30) == code:
            await conn.execute(
                text("UPDATE identity.totp_credentials SET last_used_step = :s WHERE user_id = :u"),
                {"s": candidate, "u": user_id},
            )
            return True
    return False
