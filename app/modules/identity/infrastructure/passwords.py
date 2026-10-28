"""Argon2id password hashing (time_cost 3, 64 MiB, parallelism 2)."""

from __future__ import annotations

from argon2 import PasswordHasher as _PH
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

_ph = _PH(time_cost=3, memory_cost=64 * 1024, parallelism=2)


class Argon2Hasher:
    dummy_hash = _ph.hash("dummy-password-for-constant-time")

    def hash(self, password: str) -> str:
        return _ph.hash(password)

    def verify(self, hash_: str, password: str) -> tuple[bool, str | None]:
        """Returns (ok, new_hash_if_parameters_changed)."""
        try:
            _ph.verify(hash_, password)
        except (VerifyMismatchError, VerificationError, InvalidHashError):
            return False, None
        return True, (_ph.hash(password) if _ph.check_needs_rehash(hash_) else None)
