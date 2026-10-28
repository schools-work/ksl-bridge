"""Register a self-service account (signer or speaker). Admin/annotator roles are granted by admins only."""

from __future__ import annotations

import uuid

from sqlalchemy.ext.asyncio import AsyncConnection

from app.platform.db import set_principal
from app.platform.errors import Conflict, Invalid
from app.platform.principal import Principal

from ..domain.user import Role, User, validate_email, validate_password, validate_phone
from ..infrastructure.passwords import Argon2Hasher
from ..infrastructure.repositories import SqlUserRepo
from .dto import RegisterIn


async def execute(conn: AsyncConnection, body: RegisterIn) -> uuid.UUID:
    if not body.email and not body.phone:
        raise Invalid("email or phone is required")
    try:
        email = validate_email(body.email) if body.email else None
        phone = validate_phone(body.phone) if body.phone else None
        if body.password is not None:
            validate_password(body.password)
    except ValueError as e:
        raise Invalid(str(e)) from e
    repo = SqlUserRepo(conn)
    if await repo.find_login(email, phone) is not None:
        raise Conflict("an account with this email or phone already exists")
    user = User(
        id=uuid.uuid4(),
        display_name=body.display_name,
        email=email,
        phone_e164=phone,
        roles=frozenset({Role(body.role)}),
        preferred_in=body.preferred_in,
        preferred_out=body.preferred_out,
        dominant_hand=body.dominant_hand,
    )
    # RLS: identity.users only accepts a row whose id = app.current_user_id
    await set_principal(conn, Principal(user_id=user.id))
    pw_hash = Argon2Hasher().hash(body.password) if body.password else None
    await repo.create(user, pw_hash)
    await repo.add_role(user.id, body.role)
    return user.id
