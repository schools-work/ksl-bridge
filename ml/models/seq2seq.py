"""Gloss ⇄ text Transformer seq2seq with beam search (Guide Part IV §6)."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn


class Seq2Seq(nn.Module):
    def __init__(
        self,
        src_vocab: int,
        tgt_vocab: int,
        d: int = 256,
        heads: int = 4,
        enc_layers: int = 4,
        dec_layers: int = 4,
        p: float = 0.1,
        pad_id: int = 1,
        max_len: int = 512,
    ):
        super().__init__()
        self.src_emb = nn.Embedding(src_vocab, d, padding_idx=pad_id)
        self.tgt_emb = nn.Embedding(tgt_vocab, d, padding_idx=pad_id)
        self.pos = nn.Embedding(max_len, d)
        self.tf = nn.Transformer(d, heads, enc_layers, dec_layers, 4 * d, p, batch_first=True, norm_first=True)
        self.out = nn.Linear(d, tgt_vocab)
        self.out.weight = self.tgt_emb.weight
        self.d, self.pad = d, pad_id

    def _embed(self, emb: nn.Embedding, ids: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(ids.shape[1], device=ids.device)
        return emb(ids) * math.sqrt(self.d) + self.pos(pos)

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        causal = nn.Transformer.generate_square_subsequent_mask(tgt_in.shape[1], device=src.device, dtype=torch.bool)
        h = self.tf(
            self._embed(self.src_emb, src),
            self._embed(self.tgt_emb, tgt_in),
            tgt_mask=causal,
            src_key_padding_mask=src.eq(self.pad),
            tgt_key_padding_mask=tgt_in.eq(self.pad),
            memory_key_padding_mask=src.eq(self.pad),
        )
        return self.out(h)

    @torch.no_grad()
    def beam_search(
        self, src: torch.Tensor, bos: int, eos: int, beam: int = 4, max_len: int = 64, len_alpha: float = 0.7
    ) -> list[tuple[list[int], float]]:
        mem = self.tf.encoder(self._embed(self.src_emb, src), src_key_padding_mask=src.eq(self.pad))
        beams: list[tuple[list[int], float]] = [([bos], 0.0)]
        done: list[tuple[list[int], float]] = []
        for _ in range(max_len):
            cand: list[tuple[list[int], float]] = []
            for toks, sc in beams:
                t = torch.tensor([toks], device=src.device)
                causal = nn.Transformer.generate_square_subsequent_mask(t.shape[1], device=src.device, dtype=torch.bool)
                h = self.tf.decoder(
                    self._embed(self.tgt_emb, t), mem, tgt_mask=causal, memory_key_padding_mask=src.eq(self.pad)
                )
                lp = self.out(h[:, -1]).log_softmax(-1).squeeze(0)
                top = lp.topk(beam)
                cand += [(toks + [i], sc + v) for v, i in zip(top.values.tolist(), top.indices.tolist(), strict=True)]
            cand.sort(key=lambda x: -x[1])
            beams = []
            for toks, sc in cand[: beam * 2]:
                (done if toks[-1] == eos else beams).append((toks, sc))
                if len(beams) == beam:
                    break
            if len(done) >= beam or not beams:
                break
        done = done or beams

        def norm(t: list[int], s: float) -> float:
            return s / (((5 + len(t)) / 6) ** len_alpha)

        return sorted(((t, norm(t, s)) for t, s in done), key=lambda x: -x[1])


class Seq2SeqTranslator:
    """Inference wrapper: placeholders for FS:/NUM: tokens, gloss vocab → SentencePiece target."""

    def __init__(self, model: Seq2Seq, src_itos: list[str], sp: Any, bos: int = 3, eos: int = 4):
        self.model, self.src_stoi, self.sp, self.bos, self.eos = (
            model.eval(),
            {s: i for i, s in enumerate(src_itos)},
            sp,
            bos,
            eos,
        )

    @classmethod
    def from_checkpoint(cls, path: Any) -> Seq2SeqTranslator:
        import sentencepiece as spm

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model = Seq2Seq(len(ckpt["src_itos"]), ckpt["tgt_vocab"], **ckpt["kw"])
        model.load_state_dict(ckpt["state_dict"])
        sp = spm.SentencePieceProcessor(model_proto=ckpt["spm_proto"])
        return cls(model, ckpt["src_itos"], sp)

    def translate(self, glosses: list[str]) -> str:
        holds: dict[str, str] = {}
        src: list[str] = []
        for g in glosses:
            if g.startswith(("FS:", "NUM:")):
                key = f"<PH{len(holds)}>"
                holds[key] = g.split(":", 1)[1].title() if g.startswith("FS:") else g.split(":", 1)[1]
                src.append(key if key in self.src_stoi else "<unk>")
            else:
                src.append(g)
        ids = torch.tensor([[self.src_stoi.get(g, 2) for g in src]])
        best = self.model.beam_search(ids, self.bos, self.eos)[0][0]
        text = self.sp.decode([i for i in best if i not in (self.bos, self.eos)])
        for k, v in holds.items():
            text = text.replace(k, v)
        return str(text)
