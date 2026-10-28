import pytest

torch = pytest.importorskip("torch")

from ml.data.datasets import collate  # noqa: E402
from ml.models.cslr import CSLR  # noqa: E402
from ml.models.islr import ISLR  # noqa: E402
from ml.models.mpm import MaskedPoseModel  # noqa: E402
from ml.models.production import produce  # noqa: E402
from ml.models.seq2seq import Seq2Seq  # noqa: E402

pytestmark = pytest.mark.ml


def _batch(B: int = 2, T: int = 64):  # type: ignore[no-untyped-def]
    X = torch.randn(B, T, 892)
    M = torch.ones(B, T, 892, dtype=torch.bool)
    P = torch.zeros(B, T, dtype=torch.bool)
    P[1, 50:] = True
    return X, M, P


def test_islr_cslr_mpm_forward() -> None:
    X, M, P = _batch()
    isl = ISLR(10, d=64, layers=2, heads=4)
    assert isl(X, M, P, torch.tensor([1, 2])).shape == (2, 10)
    cs = CSLR(12, d=64, layers=4, heads=4, inter_ctc_layer=2)
    lg, inter, pad = cs(X, M, P)
    assert lg.shape == (2, 16, 12) and pad.shape == (2, 16)
    loss = CSLR.loss(lg, inter, pad, torch.tensor([[5, 6, 7], [5, 6, 0]]), torch.tensor([3, 2]))
    assert torch.isfinite(loss)
    assert torch.isfinite(MaskedPoseModel(d=64, layers=2, heads=4)(X, M, P))


def test_seq2seq_and_collate_and_production() -> None:
    import numpy as np

    s2s = Seq2Seq(20, 30, d=64, heads=4, enc_layers=1, dec_layers=1).eval()
    assert s2s(torch.tensor([[5, 6, 7]]), torch.tensor([[3, 8, 9]])).shape == (1, 3, 30)
    assert s2s.beam_search(torch.tensor([[5, 6, 7]]), 3, 4, beam=2, max_len=5)
    batch = [
        (torch.randn(10, 892), torch.ones(10, 892, dtype=torch.bool), torch.tensor([5, 6])),
        (torch.randn(7, 892), torch.ones(7, 892, dtype=torch.bool), torch.tensor([7])),
    ]
    X, M, P, L, Y, YL = collate(batch)
    assert X.shape == (2, 10, 892) and YL.tolist() == [2, 1]
    clip = np.random.default_rng(0).random((20, 107, 3)).astype(np.float32)
    seq = produce(
        ["A", "B", "FS:X"],
        {"A": clip, "B": clip},
        {"X": clip[:3]},
        question="q",
        face_templates={"q": np.zeros((40, 3))},
        speed=1.25,
    )
    assert seq.ndim == 3 and seq.shape[1:] == (107, 3)
