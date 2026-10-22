"""Overfit one batch: a correct model + optimizer must memorize a single fixed batch."""

import pytest
import torch

from nanonemo.model.config import ModelConfig
from nanonemo.model.model import NanoNemo


def _overfit(cfg: ModelConfig, device: str, B: int, T: int, lr: float, steps: int) -> list[float]:
    torch.manual_seed(0)
    model = NanoNemo(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    seq = torch.randint(0, cfg.vocab_size, (B, T + 1), device=device)
    x, y = seq[:, :-1], seq[:, 1:]  # predict the next token
    losses = []
    for _ in range(steps):
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


def test_small_model_overfits_one_batch_cpu():
    cfg = ModelConfig(n_layers=2, d_model=96, n_heads=6, n_kv_heads=2, d_ff=256, vocab_size=128, max_seq_len=32)
    losses = _overfit(cfg, "cpu", B=4, T=32, lr=3e-3, steps=100)
    assert losses[0] > 4.0  # ~ln(128) = 4.85 at init
    assert losses[-1] < 0.05


@pytest.mark.gpu
def test_full_model_overfits_one_batch_gpu():
    losses = _overfit(ModelConfig(), "cuda", B=4, T=128, lr=3e-4, steps=100)
    assert losses[0] > 10.5  # ~10.98 at init
    assert losses[-1] < 0.05
