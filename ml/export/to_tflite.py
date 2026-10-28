"""Export the on-device ISLR core / fingerspelling models to TFLite (via ai-edge-torch) for the Kotlin app."""

from __future__ import annotations

import argparse

import torch

from ksl_core.errors import DecisionRequired


def export_islr(bundle: str, out: str) -> None:
    try:
        import ai_edge_torch  # type: ignore[import-not-found]
    except ImportError as e:
        raise DecisionRequired(
            "TFLite export needs ai-edge-torch (or choose ExecuTorch); record the choice in an ADR"
        ) from e
    from ml.models.islr import ISLR

    ck = torch.load(bundle, map_location="cpu", weights_only=False)
    m = ISLR(len(ck["itos"]), **ck["enc_kw"]).eval()
    m.load_state_dict(ck["state_dict"])
    T = 64
    sample = (torch.zeros(1, T, 892), torch.ones(1, T, 892, dtype=torch.bool), torch.zeros(1, T, dtype=torch.bool))
    ai_edge_torch.convert(m, sample).export(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle")
    ap.add_argument("--out", default="artifacts/export/islr_core.tflite")
    a = ap.parse_args()
    export_islr(a.bundle, a.out)
