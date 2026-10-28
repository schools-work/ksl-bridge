"""Benchmark: custom CUDA kernel vs plain torch vs torch.compile (report all three honestly)."""

from __future__ import annotations

import torch

from ml.kernels.normalize_cuda import normalize_frames, normalize_frames_torch


def bench(fn, *args, iters: int = 200) -> float:  # type: ignore[no-untyped-def]
    for _ in range(10):
        fn(*args)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn(*args)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("needs a CUDA GPU")
    N, P = 4096, 107
    x = torch.randn(N, P, 3, device="cuda")
    center, scale, roll = (
        torch.randn(N, 3, device="cuda"),
        torch.rand(N, device="cuda") + 0.5,
        torch.randn(N, device="cuda"),
    )
    ref = normalize_frames_torch(x, center, scale, roll)
    assert torch.allclose(normalize_frames(x, center, scale, roll), ref, atol=1e-5)
    compiled = torch.compile(normalize_frames_torch)
    for name, fn in (("cuda_kernel", normalize_frames), ("torch", normalize_frames_torch), ("torch.compile", compiled)):
        print(f"{name:14s} {bench(fn, x, center, scale, roll):.4f} ms")


if __name__ == "__main__":
    main()
