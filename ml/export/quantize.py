"""Post-training dynamic INT8 quantisation (Linear layers) + size/accuracy report helper."""

from __future__ import annotations

import io

import torch


def dynamic_int8(model: torch.nn.Module) -> torch.nn.Module:
    return torch.ao.quantization.quantize_dynamic(model.eval(), {torch.nn.Linear}, dtype=torch.qint8)


def size_mb(model: torch.nn.Module) -> float:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / 1e6
