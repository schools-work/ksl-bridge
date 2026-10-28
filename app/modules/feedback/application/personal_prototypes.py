"""Per-user sign prototypes (requires 'personalisation' consent, enforced by the DB trigger)."""

from __future__ import annotations

from uuid import UUID

import numpy as np
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection


def update_prototype(old: np.ndarray | None, n_old: int, new_embeddings: np.ndarray) -> tuple[np.ndarray, int]:
    """Running mean of L2-normalised embeddings, re-normalised."""
    new = new_embeddings / np.maximum(np.linalg.norm(new_embeddings, axis=1, keepdims=True), 1e-8)
    total = (old * n_old if old is not None else 0) + new.sum(0)
    n = n_old + len(new)
    proto = total / n
    return proto / max(float(np.linalg.norm(proto)), 1e-8), n


async def upsert(conn: AsyncConnection, user_id: UUID, gloss: str, base_model_id: UUID, embeddings: np.ndarray) -> None:
    row = (
        await conn.execute(
            text(
                "SELECT id, embedding::text, n_examples FROM feedback.personal_adaptations WHERE user_id = :u AND gloss = :g "
                "AND kind = 'prototype' AND base_model_id = :m"
            ),
            {"u": user_id, "g": gloss, "m": base_model_id},
        )
    ).first()
    old = np.array([float(x) for x in row[1].strip("[]").split(",")]) if row else None
    proto, n = update_prototype(old, row[2] if row else 0, embeddings)
    vec = "[" + ",".join(f"{x:.6f}" for x in proto) + "]"
    if row:
        await conn.execute(
            text(
                "UPDATE feedback.personal_adaptations SET embedding = CAST(:e AS vector), "
                "n_examples = :n WHERE id = :id"
            ),
            {"e": vec, "n": n, "id": row[0]},
        )
    else:
        await conn.execute(
            text(
                "INSERT INTO feedback.personal_adaptations (user_id, kind, gloss, base_model_id, embedding, n_examples, enabled) "
                "VALUES (:u, 'prototype', :g, :m, CAST(:e AS vector), :n, true)"
            ),
            {"u": user_id, "g": gloss, "m": base_model_id, "e": vec, "n": n},
        )
