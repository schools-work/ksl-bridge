"""Fused body-normalisation CUDA kernel via torch load_inline (Guide Part VIII §6.3). Learning exercise:
compare against the NumPy reference and torch.compile before adopting."""

from __future__ import annotations

import torch

CUDA_SRC = r"""
#include <torch/extension.h>
__global__ void normalize_kernel(const float* __restrict__ x, float* __restrict__ y, const float* __restrict__ center,
                                 const float* __restrict__ scale, const float* __restrict__ cs, int N, int P) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * P) return;
    int n = idx / P;
    const float* xi = x + idx * 3;
    float inv = 1.0f / scale[n];
    float px = (xi[0] - center[n*3+0]) * inv, py = (xi[1] - center[n*3+1]) * inv, pz = (xi[2] - center[n*3+2]) * inv;
    float c = cs[n*2+0], s = cs[n*2+1];
    float* yi = y + idx * 3;
    yi[0] = c * px - s * py; yi[1] = s * px + c * py; yi[2] = pz;
}
torch::Tensor normalize_frames(torch::Tensor x, torch::Tensor center, torch::Tensor scale, torch::Tensor cs) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.dtype() == torch::kFloat32, "x must be contiguous CUDA float32");
    auto y = torch::empty_like(x);
    int N = x.size(0), P = x.size(1), threads = 256, blocks = (N * P + threads - 1) / threads;
    normalize_kernel<<<blocks, threads>>>(x.data_ptr<float>(), y.data_ptr<float>(), center.data_ptr<float>(),
                                          scale.data_ptr<float>(), cs.data_ptr<float>(), N, P);
    return y;
}
"""
CPP_SRC = (
    "torch::Tensor normalize_frames(torch::Tensor x, torch::Tensor center, torch::Tensor scale, torch::Tensor cs);"
)
_ext = None


def _load():  # type: ignore[no-untyped-def]
    global _ext
    if _ext is None:
        from torch.utils.cpp_extension import load_inline

        _ext = load_inline(
            name="ksl_norm",
            cpp_sources=CPP_SRC,
            cuda_sources=CUDA_SRC,
            functions=["normalize_frames"],
            extra_cuda_cflags=["-O3"],
            verbose=False,
        )
    return _ext


def normalize_frames(x: torch.Tensor, center: torch.Tensor, scale: torch.Tensor, roll: torch.Tensor) -> torch.Tensor:
    cs = torch.stack([torch.cos(-roll), torch.sin(-roll)], -1).contiguous()
    return _load().normalize_frames(x.contiguous(), center.contiguous(), scale.contiguous(), cs)


def normalize_frames_torch(
    x: torch.Tensor, center: torch.Tensor, scale: torch.Tensor, roll: torch.Tensor
) -> torch.Tensor:
    """Reference implementation in plain PyTorch (also the torch.compile candidate)."""
    y = (x - center[:, None]) / scale[:, None, None]
    c, s = torch.cos(-roll)[:, None], torch.sin(-roll)[:, None]
    out = y.clone()
    out[..., 0] = c * y[..., 0] - s * y[..., 1]
    out[..., 1] = s * y[..., 0] + c * y[..., 1]
    return out
