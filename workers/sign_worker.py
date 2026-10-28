"""CSLR (+ fingerspelling) worker. Stateless: each job decodes the utterance buffered so far."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from ksl_core.normalize import build_features
from ml.decoding.alignment import token_spans
from ml.decoding.ctc_beam import CTCPrefixBeamSearch
from ml.decoding.ngram_lm import KneserNeyLM

from .adapters import ModelAdapter, ModelInfo, ModelUnavailable, register
from .loader import fetch_artifact, live_model
from .runner import main


@register("cslr")
class CSLRAdapter(ModelAdapter):
    def __init__(self, device: str | None = None):
        self.device = device or os.environ.get("KSL_WORKER_DEVICE", "cpu")
        self.info = ModelInfo("cslr", "unloaded", "-", self.device)
        self.model: Any = None
        self.lm: KneserNeyLM | None = None
        self.itos: list[str] = []
        self.T = 1.0
        self.decoder: dict[str, Any] = {"beam": 8, "alpha": 0.6, "beta": 1.0, "prune_logp": -8.0}
        self.critical: set[str] = set()

    def load(self) -> None:
        import torch

        from ml.models.cslr import CSLR

        m = live_model("cslr")
        if m is None:
            return  # no deployed model yet: every job gets ModelUnavailable (text relay still works)
        ckpt = torch.load(fetch_artifact(m), map_location="cpu", weights_only=False)
        self.model = CSLR(vocab_size=len(ckpt["itos"]), **ckpt["enc_kw"]).eval().to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.itos, self.T = ckpt["itos"], float(ckpt.get("temperature", 1.0))
        self.critical = set(ckpt.get("critical_glosses", []))
        if m.decoder:
            self.decoder.update({k: m.decoder[k] for k in ("beam", "alpha", "beta", "prune_logp") if k in m.decoder})
        lm_m = live_model("gloss_lm")
        if lm_m is not None:
            self.lm = KneserNeyLM.load(fetch_artifact(lm_m))
        self.info = ModelInfo("cslr", m.name, m.version, self.device, m.is_bootstrap)

    def warmup(self) -> None:
        if self.model is not None:
            self.infer_batch([{"xyz": np.zeros((16, 107, 3)).tolist(), "final": False}])

    def infer_batch(self, inputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.model is None:
            raise ModelUnavailable("no production CSLR model deployed")
        import torch

        feats = [build_features(np.asarray(i["xyz"], dtype=np.float32)) for i in inputs]
        T = max(f.shape[0] for f, _ in feats)
        B, F = len(feats), feats[0][0].shape[1]
        X = torch.zeros(B, T, F)
        M = torch.zeros(B, T, F, dtype=torch.bool)
        P = torch.ones(B, T, dtype=torch.bool)
        for b, (f, m) in enumerate(feats):
            X[b, : len(f)] = torch.from_numpy(f)
            M[b, : len(f)] = torch.from_numpy(m)
            P[b, : len(f)] = False
        with torch.inference_mode():
            logits, _, pad = self.model(X.to(self.device), M.to(self.device), P.to(self.device))
            lp = (logits.float() / self.T).log_softmax(-1).cpu().numpy()
        outs = []
        for b, inp in enumerate(inputs):
            valid = int((~pad[b]).sum())
            dec = CTCPrefixBeamSearch(
                self.itos,
                lm=self.lm,
                beam=int(self.decoder["beam"]),
                alpha=float(self.decoder["alpha"]),
                beta=float(self.decoder["beta"]),
                prune_logp=float(self.decoder["prune_logp"]),
                context_bias={k: min(float(v), 2.0) for k, v in (inp.get("context_bias") or {}).items()},
            )
            dec.step(lp[b, :valid])
            nb = dec.nbest(5, final=bool(inp.get("final")))
            out: dict[str, Any] = {
                "nbest": [{"glosses": g, "score": s} for g, s in nb],
                "model_versions": {"cslr": self.info.version},
            }
            if inp.get("final") and nb:
                out["spans"] = token_spans(lp[b, :valid], self.itos, nb, frame_ms=1000 / 7.5)
                out["critical"] = any(g in self.critical for g in nb[0][0])
            outs.append(out)
        return outs


if __name__ == "__main__":
    main("cslr", CSLRAdapter())
