"""Continuous sign recognition (CTC) with dynamic-chunk training for streaming."""

from __future__ import annotations

import argparse
import random

import torch

from ksl_core.vocab import Vocab
from ml.decoding.ctc_beam import greedy_decode
from ml.eval.metrics import wer
from ml.models.cslr import CSLR
from ml.models.pose_encoder import chunk_attn_mask

from . import registry
from .carbon import tracked
from .common import by_split, device, load_config, read_manifest, seed_everything
from .loop import train
from .pose_tasks import load_encoder_init, loaders


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="ml/configs/cslr_conformer_ctc.yaml")
    ap.add_argument("--no-registry", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    seed_everything(cfg["seed"])
    items = read_manifest(cfg["manifest"])
    splits = by_split(items)
    # synthetic sentences may train but never evaluate (MASTER_PROMPT §5.3)
    splits["val"] = [i for i in splits["val"] if not i.get("is_synthetic")]
    vocab = Vocab.build([g for it in splits["train"] for g in it["glosses"]])
    tr, va = loaders(splits, vocab, cfg, set(cfg.get("no_mirror_glosses", [])))
    model = CSLR(len(vocab), inter_ctc_layer=cfg.get("inter_ctc_layer", 4), **cfg["model"])
    load_encoder_init(model, cfg.get("init_from_mpm"))
    dev = device()
    chunks = cfg.get("dynamic_chunks", [4, 8, 16, 0])

    def loss_fn(m: torch.nn.Module, b: list) -> torch.Tensor:  # type: ignore[type-arg]
        X, M, P, _, Y, YL = b
        c = random.choice(chunks)  # noqa: S311 (training curriculum, not security)
        T_sub = (X.shape[1] + 3) // 4
        mask = chunk_attn_mask(T_sub, c, cfg.get("left_chunks", 4), X.device) if c else None
        logits, inter, pad = m(X, M, P, mask)
        return CSLR.loss(logits, inter, pad, Y.clamp(min=0), YL, cfg.get("inter_ctc_weight", 0.3))

    @torch.no_grad()
    def val_fn(m: torch.nn.Module) -> dict[str, float]:
        m.eval()
        refs, hyps = [], []
        for X, M, P, _, Y, YL in va:
            logits, _, pad = m(X.to(dev), M.to(dev), P.to(dev))
            lp = logits.float().log_softmax(-1).cpu().numpy()
            for b in range(len(Y)):
                valid = int((~pad[b]).sum())
                hyps.append(greedy_decode(lp[b, :valid], vocab.itos))
                refs.append(vocab.decode(Y[b, : YL[b]].tolist()))
        return {"wer": wer(refs, hyps)["wer"]}

    run = None
    if not a.no_registry:
        exp = registry.ensure_experiment("cslr", cfg["experiment"], cfg["hypothesis"], "RQ2")
        run = registry.start_run(exp, "cslr", cfg, cfg["seed"])
    (best, ckpt), fp = tracked(train)(
        model,
        loss_fn,
        tr,
        val_fn,
        cfg["epochs"],
        lr=cfg["lr"],
        device=dev,
        run_name=cfg["experiment"],
        log=(lambda k, v, s: registry.log_metric(run, k, v, s)) if run else None,
    )
    bundle = ckpt.with_suffix(".bundle.pt")
    torch.save(
        {
            "state_dict": torch.load(ckpt, weights_only=True),
            "itos": vocab.itos,
            "enc_kw": cfg["model"],
            "temperature": 1.0,
            "critical_glosses": cfg.get("critical_glosses", []),
        },
        bundle,
    )
    if run:
        registry.finish_run(run, True, fp, artifact_uri=str(bundle))
    print(f"best val WER (greedy) {best:.3f} → {bundle}")


if __name__ == "__main__":
    main()
