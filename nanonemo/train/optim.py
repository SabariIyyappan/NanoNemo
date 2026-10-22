"""AdamW with decay/no-decay parameter groups, and the warmup + cosine LR schedule."""

import math

import torch
from torch import nn


def param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """Decay 2-D matmul weights only; norms and the (tied) embedding are not decayed."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():  # yields the tied embedding/lm_head weight once
        if not p.requires_grad:
            continue
        if p.dim() >= 2 and "tok_emb" not in name:
            decay.append(p)
        else:
            no_decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def build_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float = 0.1,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
    fused: bool = False,
) -> torch.optim.AdamW:
    return torch.optim.AdamW(param_groups(model, weight_decay), lr=lr, betas=betas, eps=eps, fused=fused)


def lr_at(step: int, peak_lr: float, total_steps: int, warmup_steps: int, min_ratio: float = 0.1) -> float:
    """Linear warmup to peak_lr, then cosine decay to min_ratio * peak_lr at the last step."""
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps  # step 0 already gets a nonzero LR
    progress = (step - warmup_steps) / max(1, total_steps - 1 - warmup_steps)
    progress = min(progress, 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))  # 1 -> 0
    return peak_lr * (min_ratio + (1.0 - min_ratio) * cosine)
