"""Conformer-CTC ASR training on Kenyan English + Swahili (Common Voice sw, AfriVoices-KE, own recordings)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ml.models.asr import ConformerCTC
from ml.models.asr_features import logmel, spec_augment

from .common import device, load_config, seed_everything
from .loop import train


class SpeechDataset(Dataset):  # type: ignore[type-arg]
    def __init__(self, rows: list[dict], sp: object, train: bool):  # type: ignore[type-arg]
        self.rows, self.sp, self.train = rows, sp, train

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        import soundfile as sf

        r = self.rows[i]
        wav, sr = sf.read(r["audio"], dtype="float32")
        if sr != 16000:
            import torchaudio

            wav = torchaudio.functional.resample(torch.from_numpy(wav), sr, 16000).numpy()
        f = logmel(torch.from_numpy(wav))
        if self.train:
            f = spec_augment(f)
        return f, torch.tensor([i + 5 for i in self.sp.encode(r["text"])])  # type: ignore[attr-defined]


def collate(b: list) -> tuple[torch.Tensor, ...]:  # type: ignore[type-arg]
    fs, ys = zip(*b, strict=True)
    T = max(f.shape[0] for f in fs)
    X = torch.zeros(len(fs), T, fs[0].shape[1])
    P = torch.ones(len(fs), T, dtype=torch.bool)
    for i, f in enumerate(fs):
        X[i, : len(f)], P[i, : len(f)] = f, False
    return X, P, torch.nn.utils.rnn.pad_sequence(list(ys), batch_first=True), torch.tensor([len(y) for y in ys])


def main() -> None:
    import sentencepiece as spm

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="ml/configs/asr_conformer.yaml")
    cfg = load_config(ap.parse_args().config)
    seed_everything(cfg["seed"])
    sp = spm.SentencePieceProcessor(model_file=cfg["spm_model"])
    rows = {
        s: [json.loads(x) for x in Path(cfg[f"{s}_jsonl"]).read_text().splitlines() if x.strip()]
        for s in ("train", "val")
    }
    tr = DataLoader(SpeechDataset(rows["train"], sp, True), cfg["batch_size"], shuffle=True, collate_fn=collate)
    va = DataLoader(SpeechDataset(rows["val"], sp, False), cfg["batch_size"], collate_fn=collate)
    model = ConformerCTC(sp.get_piece_size() + 5, **cfg["model"])
    dev = device()

    def loss_fn(m: torch.nn.Module, b: list) -> torch.Tensor:  # type: ignore[type-arg]
        X, P, Y, YL = b
        logits, pad = m(X, P)
        tg = torch.cat([y[:l] for y, l in zip(Y, YL, strict=True)])
        return F.ctc_loss(logits.float().log_softmax(-1).transpose(0, 1), tg, (~pad).sum(1), YL, zero_infinity=True)

    @torch.no_grad()
    def val_fn(m: torch.nn.Module) -> dict[str, float]:
        tot, n = 0.0, 0
        for b in va:
            tot += float(loss_fn(m, [t.to(dev) for t in b]))
            n += 1
        return {"val_ctc": tot / max(1, n)}

    best, ckpt = train(model, loss_fn, tr, val_fn, cfg["epochs"], lr=cfg["lr"], device=dev, run_name=cfg["experiment"])
    itos = ["<blank>", "<pad>", "<unk>", "<bos>", "<eos>"] + [sp.id_to_piece(i) for i in range(sp.get_piece_size())]
    torch.save(
        {"state_dict": torch.load(ckpt, weights_only=True), "itos": itos, "kw": cfg["model"]},
        ckpt.with_suffix(".bundle.pt"),
    )
    print(f"best val CTC {best:.3f}; evaluate WER with ml.eval.evaluate --task asr against the bootstrap model")


if __name__ == "__main__":
    main()
