"""Train a shared English+Swahili BPE (SentencePiece) for gloss2text/ASR/embedder targets."""

from __future__ import annotations

import argparse


def train(corpus: list[str], model_prefix: str, vocab_size: int = 8000) -> str:
    import sentencepiece as spm

    spm.SentencePieceTrainer.train(
        input=",".join(corpus),
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=1.0,
        pad_id=1,
        unk_id=2,
        bos_id=3,
        eos_id=4,
        user_defined_symbols=[f"<PH{i}>" for i in range(8)],
    )
    return f"{model_prefix}.model"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", nargs="+")
    ap.add_argument("--prefix", default="artifacts/bpe/en_sw_8k")
    ap.add_argument("--vocab-size", type=int, default=8000)
    a = ap.parse_args()
    print(train(a.corpus, a.prefix, a.vocab_size))
