"""Generic trainer: AMP (bf16), EMA, cosine LR with warm-up, grad clip, early stopping, MLflow logging."""

from __future__ import annotations

import copy
import math
import time
from collections.abc import Callable, Iterable
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.m = copy.deepcopy(model).eval()
        for p in self.m.parameters():
            p.requires_grad_(False)
        self.d = decay

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for e, p in zip(self.m.state_dict().values(), model.state_dict().values(), strict=True):
            if e.dtype.is_floating_point:
                e.mul_(self.d).add_(p.detach(), alpha=1 - self.d)
            else:
                e.copy_(p)


def cosine_lr(step: int, total: int, warmup: int, base: float) -> float:
    if step < warmup:
        return base * step / max(1, warmup)
    q = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1 + math.cos(math.pi * min(1.0, q)))


def _mlflow() -> Any:
    try:
        import mlflow

        return mlflow
    except ImportError:
        return None


def train(
    model: torch.nn.Module,
    loss_fn: Callable[[torch.nn.Module, list[Any]], torch.Tensor],
    train_dl: Iterable[Any],
    val_fn: Callable[[torch.nn.Module], dict[str, float]],
    epochs: int,
    lr: float = 1e-3,
    wd: float = 0.05,
    accum: int = 1,
    clip: float = 1.0,
    device: str = "cuda",
    run_name: str = "run",
    patience: int = 10,
    out_dir: str = "artifacts/checkpoints",
    lower_is_better: bool = True,
    log: Callable[[str, float, int], None] | None = None,
) -> tuple[float, Path]:
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.98))
    use_amp = device.startswith("cuda")
    ema = EMA(model)
    n_batches = len(train_dl) if hasattr(train_dl, "__len__") else 1000  # type: ignore[arg-type]
    total = max(1, epochs * n_batches // accum)
    step, bad = 0, 0
    best = float("inf") if lower_is_better else -float("inf")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ckpt = Path(out_dir) / f"{run_name}_best.pt"
    ml = _mlflow()
    ctx = ml.start_run(run_name=run_name) if ml else nullcontext()
    with ctx:
        for _ep in range(epochs):
            model.train()
            t0 = time.time()
            for i, batch in enumerate(train_dl):
                batch = [b.to(device, non_blocking=True) if torch.is_tensor(b) else b for b in batch]
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                    loss = loss_fn(model, batch) / accum
                loss.backward()
                if (i + 1) % accum == 0:
                    for g in opt.param_groups:
                        g["lr"] = cosine_lr(step, total, int(0.05 * total), lr)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    ema.update(model)
                    step += 1
                    if step % 50 == 0:
                        if ml:
                            ml.log_metric("train_loss", float(loss) * accum, step=step)
                        if log:
                            log("train_loss", float(loss) * accum, step)
            val = val_fn(ema.m)
            key = next(iter(val.values()))
            if ml:
                ml.log_metrics({f"val_{k}": v for k, v in val.items()}, step=step)
                ml.log_metric("epoch_seconds", time.time() - t0, step=step)
            if log:
                for k, v in val.items():
                    log(f"val_{k}", v, step)
            improved = key < best if lower_is_better else key > best
            if improved:
                best, bad = key, 0
                torch.save(ema.m.state_dict(), ckpt)
            else:
                bad += 1
                if bad >= patience:
                    break
    return best, ckpt
