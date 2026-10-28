"""Keep partitions 3 months ahead (schedule monthly)."""

from __future__ import annotations

import sqlalchemy as sa

from ml.db import engine

TABLES = ("conversation.utterances", "inference.requests", "audit.events")

if __name__ == "__main__":
    with engine().begin() as c:
        for t in TABLES:
            c.execute(sa.text("SELECT ops.create_monthly_partitions(CAST(:t AS regclass), now()::date, 3)"), {"t": t})
            print(f"ok {t}")
