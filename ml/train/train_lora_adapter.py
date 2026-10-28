"""Per-user LoRA adapters (rank 8) on the CSLR attention projections (Guide Part V §6)."""

from __future__ import annotations

import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.A = nn.Parameter(torch.randn(r, base.in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        self.scale = alpha / r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale


def add_lora(model: nn.Module, r: int = 8) -> list[nn.Parameter]:
    """Wrap every Conformer feed-forward output Linear with LoRA; returns the trainable parameters."""
    params: list[nn.Parameter] = []
    for blk in model.enc.blocks:  # type: ignore[union-attr]
        for seq_name in ("ff1", "ff2"):
            seq = getattr(blk, seq_name)
            last = len(seq) - 1
            if isinstance(seq[last], nn.Linear):
                lora = LoRALinear(seq[last], r)
                seq[last] = lora
                params += [lora.A, lora.B]
    for p in model.parameters():
        p.requires_grad_(False)
    for p in params:
        p.requires_grad_(True)
    return params


def lora_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v for k, v in model.state_dict().items() if k.endswith((".A", ".B"))}
