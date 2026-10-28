"""Load a downloaded source into dataset.samples (+ signers), after the licence gate."""

from __future__ import annotations

import argparse

import sqlalchemy as sa

from ml.data.sources.base import check_allowed
from ml.data.sources.ekitabu_kaggle import EkitabuLoader
from ml.data.sources.google_asl_signs import GoogleAslSignsLoader
from ml.db import engine

LOADERS = {"ekitabu-ksl-video": EkitabuLoader, "google-asl-signs": GoogleAslSignsLoader}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=sorted(LOADERS))
    ap.add_argument("--root", required=True)
    ap.add_argument("--purpose", default="train", choices=["train", "pretrain", "eval"])
    a = ap.parse_args()
    check_allowed(a.source, a.purpose)
    loader = LOADERS[a.source](a.root)
    n = 0
    with engine().begin() as c:
        src_id = c.execute(sa.text("SELECT id FROM dataset.sources WHERE slug = :s"), {"s": a.source}).scalar_one()
        for rec in loader.iter_samples():
            signer = None
            if rec.signer_id:
                signer = f"S{abs(hash((a.source, rec.signer_id))) % 10**6:06d}"
                c.execute(
                    sa.text(
                        "INSERT INTO dataset.signers (signer_code, source_id) VALUES (:c, :s) ON CONFLICT DO NOTHING"
                    ),
                    {"c": signer, "s": src_id},
                )
            c.execute(
                sa.text(
                    "INSERT INTO dataset.samples (source_id, external_id, signer_code, kind, video_uri, glosses, review_status) "
                    "VALUES (:s, :e, :sg, CAST(:k AS shared.sample_kind), :v, :g, 'approved') ON CONFLICT DO NOTHING"
                ),
                {
                    "s": src_id,
                    "e": rec.external_id,
                    "sg": signer,
                    "k": rec.kind,
                    "v": f"file://{rec.path}",
                    "g": rec.glosses or None,
                },
            )
            n += 1
    print(f"imported {n} samples from {a.source}")


if __name__ == "__main__":
    main()
