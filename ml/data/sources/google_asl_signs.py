"""Google Isolated SLR 'ASL Signs' (Kaggle). PRETRAIN ONLY: motion priors, never KSL labels."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from ksl_core.landmarks import FACE_IDX, N_POINTS

from .base import SampleRecord, SourceLoader


def parquet_to_107(path: str) -> np.ndarray:
    """543-point long-format parquet → (T,107,3) in our layout."""
    import pandas as pd  # type: ignore[import-untyped]

    df = pd.read_parquet(path, columns=["frame", "type", "landmark_index", "x", "y", "z"])
    frames = sorted(df["frame"].unique())
    out = np.full((len(frames), N_POINTS, 3), np.nan, np.float32)
    fidx = {f: i for i, f in enumerate(frames)}
    face_map = {j: 67 + k for k, j in enumerate(FACE_IDX)}
    for row in df.itertuples(index=False):
        t = fidx[row.frame]
        if row.type == "left_hand":
            k = int(row.landmark_index)
        elif row.type == "right_hand":
            k = 21 + int(row.landmark_index)
        elif row.type == "pose" and row.landmark_index < 25:
            k = 42 + int(row.landmark_index)
        elif row.type == "face" and row.landmark_index in face_map:
            k = face_map[int(row.landmark_index)]
        else:
            continue
        out[t, k] = (row.x, row.y, row.z)
    return out


class GoogleAslSignsLoader(SourceLoader):
    slug = "google-asl-signs"

    def iter_samples(self) -> Iterator[SampleRecord]:
        import csv

        with open(self.root / "train.csv") as f:
            for r in csv.DictReader(f):
                yield SampleRecord(
                    external_id=r["sequence_id"],
                    signer_id=f"asl-{r['participant_id']}",
                    kind="isolated",
                    glosses=[],
                    path=self.root / r["path"],
                    meta={"asl_label": r["sign"], "pretrain_only": True},
                )
