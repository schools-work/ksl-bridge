"""Gloss → pose sequence for the avatar (dictionary medoid clips + Hermite blending)."""

from __future__ import annotations

import base64
from typing import Any

import numpy as np

from ml.models.production import produce

from .adapters import ModelAdapter, ModelInfo, register
from .runner import main


def _load_dictionary() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Medoid clips per gloss, fingerspelling letters and non-manual templates from lexicon.* (S3 .npy)."""
    import io

    import boto3
    import sqlalchemy as sa

    from .loader import _connect

    s3 = boto3.client("s3")

    def get(uri: str) -> np.ndarray:
        bucket, _, key = uri[5:].partition("/")
        return np.load(io.BytesIO(s3.get_object(Bucket=bucket, Key=key)["Body"].read()))

    clips, letters, faces = {}, {}, {}
    with _connect().connect() as c:
        for g, uri in c.execute(
            sa.text("SELECT gloss, pose_uri FROM lexicon.sign_clips WHERE is_medoid AND approved_by IS NOT NULL")
        ):
            clips[g] = get(uri)
        for ch, uri in c.execute(
            sa.text("SELECT letter, pose_uri FROM lexicon.fingerspell_letters WHERE approved_by IS NOT NULL")
        ):
            letters[ch] = get(uri)
        for name, uri in c.execute(
            sa.text("SELECT name, face_offsets_uri FROM lexicon.nonmanual_templates WHERE approved_by IS NOT NULL")
        ):
            faces[name] = get(uri)
    return clips, letters, faces


@register("sign2pose")
class ProductionAdapter(ModelAdapter):
    def __init__(self) -> None:
        self.info = ModelInfo("sign2pose", "dict_blend", "dict-1", "cpu")
        self.clips: dict[str, np.ndarray] = {}
        self.letters: dict[str, np.ndarray] = {}
        self.faces: dict[str, np.ndarray] = {}

    def load(self) -> None:
        try:
            self.clips, self.letters, self.faces = _load_dictionary()
        except Exception:
            self.clips, self.letters, self.faces = {}, {}, {}  # captions-only until clips are approved

    def infer_batch(self, inputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        outs = []
        for i in inputs:
            seq = produce(
                i["glosses"],
                self.clips,
                self.letters,
                speed=float(i.get("speed", 1.0)),
                question=i.get("question"),
                face_templates=self.faces,
            )
            missing = [g for g in i["glosses"] if not g.startswith("FS:") and g not in self.clips]
            outs.append(
                {
                    "n_frames": int(len(seq)),
                    "fps": 30.0,
                    "missing_glosses": missing,
                    "poses_f16_b64": base64.b64encode(seq.astype("<f2").tobytes()).decode() if len(seq) else None,
                }
            )
        return outs


if __name__ == "__main__":
    main("sign2pose", ProductionAdapter())
