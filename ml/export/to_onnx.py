"""Export pose models to ONNX for ONNX Runtime / TensorRT serving."""

from __future__ import annotations

import argparse

import torch


def export_cslr(bundle: str, out: str, opset: int = 17) -> None:
    from ml.models.cslr import CSLR

    ck = torch.load(bundle, map_location="cpu", weights_only=False)
    m = CSLR(len(ck["itos"]), **ck["enc_kw"]).eval()
    m.load_state_dict(ck["state_dict"])

    class Wrapped(torch.nn.Module):
        def __init__(self, inner: CSLR):
            super().__init__()
            self.inner = inner

        def forward(self, x: torch.Tensor, mk: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
            return self.inner(x, mk, pad)[0].log_softmax(-1)

    T = 64
    torch.onnx.export(
        Wrapped(m),
        (torch.zeros(1, T, 892), torch.ones(1, T, 892, dtype=torch.bool), torch.zeros(1, T, dtype=torch.bool)),
        out,
        opset_version=opset,
        input_names=["features", "mask", "pad"],
        output_names=["log_probs"],
        dynamic_axes={
            "features": {0: "B", 1: "T"},
            "mask": {0: "B", 1: "T"},
            "pad": {0: "B", 1: "T"},
            "log_probs": {0: "B", 1: "T4"},
        },
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle")
    ap.add_argument("--out", default="artifacts/export/cslr.onnx")
    a = ap.parse_args()
    export_cslr(a.bundle, a.out)
