"""Evaluate a CSLR bundle on a manifest's test split: clean, robustness and fairness → ml.evaluations."""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from ksl_core.normalize import build_features, normalize_sequence
from ml.decoding.ctc_beam import CTCPrefixBeamSearch
from ml.decoding.ngram_lm import KneserNeyLM
from ml.models.cslr import CSLR

from .fairness import fairness_rows
from .metrics import wer
from .robustness import robustness_report


def load_cslr(bundle: str) -> tuple[CSLR, list[str]]:
    ck = torch.load(bundle, map_location="cpu", weights_only=False)
    m = CSLR(len(ck["itos"]), **ck["enc_kw"]).eval()
    m.load_state_dict(ck["state_dict"])
    return m, ck["itos"]


def make_decoder(model: CSLR, itos: list[str], lm: KneserNeyLM | None, beam: int, alpha: float, beta: float):  # type: ignore[no-untyped-def]
    @torch.no_grad()
    def run(xyz: np.ndarray) -> list[str]:
        f, m = build_features(normalize_sequence(xyz))
        logits, _, _ = model(
            torch.from_numpy(f)[None], torch.from_numpy(m)[None], torch.zeros(1, len(f), dtype=torch.bool)
        )
        dec = CTCPrefixBeamSearch(itos, lm=lm, beam=beam, alpha=alpha, beta=beta)
        dec.step(logits.log_softmax(-1)[0].numpy())
        nb = dec.nbest(1, final=True)
        return nb[0][0] if nb else []

    return run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="cslr")
    ap.add_argument("--bundle", required=False)
    ap.add_argument("--manifest", required=False)
    ap.add_argument("--lm")
    ap.add_argument("--beam", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=0.6)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--register", nargs=2, metavar=("MODEL_VERSION_ID", "BUILD_ID"))
    a = ap.parse_args()
    if a.task != "cslr" or not a.bundle or not a.manifest:
        raise SystemExit("usage: --task cslr --bundle <x.bundle.pt> --manifest <manifest.jsonl> [--lm lm.json]")
    model, itos = load_cslr(a.bundle)
    lm = KneserNeyLM.load(a.lm) if a.lm else None
    run = make_decoder(model, itos, lm, a.beam, a.alpha, a.beta)
    items = [json.loads(x) for x in open(a.manifest) if x.strip()]
    test = [
        dict(it, xyz=np.load(it["landmarks_path"]).astype(np.float32))
        for it in items
        if it["split"] == "test" and not it.get("is_synthetic")
    ]
    hyps = [run(it["xyz"]) for it in test]
    clean = wer([it["glosses"] for it in test], hyps)
    rob = robustness_report(run, test)
    fair = fairness_rows(test, hyps)
    print(json.dumps({"clean": clean, "robustness": rob, "fairness": fair}, indent=2))
    if a.register:
        from ml.train.registry import record_evaluation

        mid, build = a.register
        record_evaluation(
            mid, build, "test", "clean", [{"metric": k, "value": v, "n": len(test)} for k, v in clean.items()]
        )
        for name, r in rob.items():
            record_evaluation(
                mid,
                build,
                "test",
                "robustness",
                [
                    {
                        "metric": "wer",
                        "value": r["wer"],
                        "ci_low": r["ci_low"],
                        "ci_high": r["ci_high"],
                        "n": int(r["n"]),
                    }
                ],
                perturbation=name,
            )
        record_evaluation(mid, build, "test", "fairness", fair)


if __name__ == "__main__":
    main()
