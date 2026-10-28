"""SQL repositories for the identity schema only."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from ..domain.tokens import RotationResult
from ..domain.user import Role, User


class SqlUserRepo:
    def __init__(self, conn: AsyncConnection):
        self.c = conn

    async def create(self, user: User, password_hash: str | None) -> None:
        await self.c.execute(
            text(
                "INSERT INTO identity.users (id, email, phone_e164, password_hash, display_name, preferred_in, "
                "preferred_out, dominant_hand) VALUES (:id, :email, :phone, :pw, :name, "
                "CAST(:pin AS shared.modality), CAST(:pout AS shared.modality), CAST(:hand AS shared.hand))"
            ),
            {
                "id": user.id,
                "email": user.email,
                "phone": user.phone_e164,
                "pw": password_hash,
                "name": user.display_name,
                "pin": user.preferred_in,
                "pout": user.preferred_out,
                "hand": user.dominant_hand,
            },
        )

    async def get(self, user_id: UUID) -> User | None:
        row = (
            (
                await self.c.execute(
                    text(
                        "SELECT id, display_name, email, phone_e164, preferred_in, preferred_out, dominant_hand, sign_speed, "
                        "status FROM identity.users WHERE id = :id"
                    ),
                    {"id": user_id},
                )
            )
            .mappings()
            .first()
        )
        if row is None:
            return None
        roles = await self.roles_of(user_id)
        return User(
            id=row["id"],
            display_name=row["display_name"],
            email=row["email"],
            phone_e164=row["phone_e164"],
            roles=frozenset(Role(r) for r in roles),
            preferred_in=row["preferred_in"],
            preferred_out=row["preferred_out"],
            dominant_hand=row["dominant_hand"],
            sign_speed=row["sign_speed"],
            status=row["status"],
        )

    async def me(self, user_id: UUID) -> dict[str, object] | None:
        row = (
            (
                await self.c.execute(
                    text(
                        "SELECT id, display_name, email::text AS email, phone_e164, preferred_in::text, preferred_out::text, "
                        "dominant_hand::text, sign_speed, caption_scale, locale FROM identity.users WHERE id = :id"
                    ),
                    {"id": user_id},
                )
            )
            .mappings()
            .first()
        )
        if row is None:
            return None
        return {**row, "roles": await self.roles_of(user_id)}

    async def patch(self, user_id: UUID, fields: dict[str, object]) -> None:
        casts = {"preferred_in": "shared.modality", "preferred_out": "shared.modality", "dominant_hand": "shared.hand"}
        sets = ", ".join(f"{k} = CAST(:{k} AS {casts[k]})" if k in casts else f"{k} = :{k}" for k in fields)
        await self.c.execute(
            text(f"UPDATE identity.users SET {sets} WHERE id = :id"),  # noqa: S608 (keys whitelisted by DTO)
            {**fields, "id": user_id},
        )

    async def find_login(self, email: str | None, phone: str | None) -> tuple[UUID, str | None, str, list[str]] | None:
        row = (
            await self.c.execute(text("SELECT * FROM identity.find_login(:e, :p)"), {"e": email, "p": phone})
        ).first()
        return (row[0], row[1], row[2], list(row[3])) if row else None

    async def roles_of(self, user_id: UUID) -> list[str]:
        rows = await self.c.execute(
            text("SELECT role::text FROM identity.user_roles WHERE user_id = :u AND revoked_at IS NULL"), {"u": user_id}
        )
        return [r[0] for r in rows]

    async def add_role(self, user_id: UUID, role: str, granted_by: UUID | None = None) -> None:
        await self.c.execute(
            text(
                "INSERT INTO identity.user_roles (user_id, role, granted_by) VALUES (:u, CAST(:r AS shared.user_role), :g) "
                "ON CONFLICT (user_id, role) DO UPDATE SET revoked_at = NULL"
            ),
            {"u": user_id, "r": role, "g": granted_by},
        )

    async def touch_login(self, user_id: UUID) -> None:
        await self.c.execute(text("UPDATE identity.users SET last_login_at = now() WHERE id = :u"), {"u": user_id})

    async def update_password_hash(self, user_id: UUID, hash_: str) -> None:
        await self.c.execute(
            text("UPDATE identity.users SET password_hash = :h WHERE id = :u"), {"h": hash_, "u": user_id}
        )


class SqlTokenRepo:
    def __init__(self, conn: AsyncConnection):
        self.c = conn

    async def insert_refresh(
        self, user_id: UUID, family_id: UUID, token_hash: bytes, device_label: str | None, ttl_days: int
    ) -> None:
        await self.c.execute(
            text(
                "INSERT INTO identity.refresh_tokens (user_id, family_id, token_hash, device_label, expires_at) "
                "VALUES (:u, :f, :h, :d, now() + make_interval(days => :ttl))"
            ),
            {"u": user_id, "f": family_id, "h": token_hash, "d": device_label, "ttl": ttl_days},
        )

    async def rotate(self, old_hash: bytes, new_hash: bytes, ttl_days: int) -> tuple[RotationResult, UUID | None]:
        row = (
            await self.c.execute(
                text("SELECT result, user_id FROM identity.rotate_refresh_token(:o, :n, make_interval(days => :ttl))"),
                {"o": old_hash, "n": new_hash, "ttl": ttl_days},
            )
        ).first()
        assert row is not None
        return RotationResult(row[0]), row[1]

    async def revoke_family_of(self, token_hash: bytes, reason: str) -> None:
        await self.c.execute(
            text(
                "UPDATE identity.refresh_tokens SET revoked_at = now(), revoke_reason = :r WHERE revoked_at IS NULL AND "
                "family_id = (SELECT family_id FROM identity.refresh_tokens WHERE token_hash = :h)"
            ),
            {"h": token_hash, "r": reason},
        )
