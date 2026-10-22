import json

import numpy as np
import pytest
import torch

from nanonemo.data.shard import write_shard
from nanonemo.model.config import ModelConfig
from nanonemo.model.model import NanoNemo
from nanonemo.train.config import TrainConfig, load_config
from nanonemo.train.trainer import train

TINY = ModelConfig(n_layers=2, d_model=64, n_heads=4, n_kv_heads=2, d_ff=128, vocab_size=64, max_seq_len=32)


def test_grad_accumulation_equals_one_big_batch():
    torch.manual_seed(0)
    model = NanoNemo(TINY)
    seq = torch.randint(0, TINY.vocab_size, (4, 17))
    x, y = seq[:, :-1], seq[:, 1:]

    model(x, y)[1].backward()
    big = [p.grad.clone() for p in model.parameters()]
    model.zero_grad()

    for i in range(2):  # 2 micro-batches of 2, loss / accum
        (model(x[2 * i : 2 * i + 2], y[2 * i : 2 * i + 2])[1] / 2).backward()
    for g_big, p in zip(big, model.parameters()):
        torch.testing.assert_close(p.grad, g_big, rtol=1e-5, atol=1e-6)


def test_grad_accum_and_steps_from_config():
    cfg = TrainConfig(micro_batch=2, seq_len=1024, global_batch_tokens=65536, total_tokens=20_000_000)
    assert cfg.grad_accum() == 32
    assert cfg.grad_accum(world_size=2) == 16
    assert cfg.total_steps == 305
    assert cfg.warmup_steps == 6
    with pytest.raises(ValueError, match="not a multiple"):
        TrainConfig(micro_batch=3, seq_len=1024, global_batch_tokens=65536).grad_accum()


def test_smoke_config_loads():
    model_cfg, cfg = load_config("configs/smoke.yaml")
    assert model_cfg == ModelConfig()
    assert cfg.grad_accum() == 32 and cfg.total_steps == 305


def test_config_rejects_typos(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("train:\n  peak_rl: 1.0e-3\n")
    with pytest.raises(ValueError, match="peak_rl"):
        load_config(path)


def _patterned_dataset(tmp_path):
    """A repeating token pattern: learnable, so a working trainer must drive the loss down."""
    rng = np.random.default_rng(0)
    pattern = rng.integers(0, TINY.vocab_size, size=50).astype(np.uint16)
    splits = {}
    for split, reps in (("train", 400), ("val", 40)):
        tokens = np.tile(pattern, reps)
        info = write_shard(tmp_path / f"{split}_000000.bin", tokens, "gpt2")
        splits[split] = {"n_tokens": len(tokens), "shards": [{"path": f"{split}_000000.bin", **info}]}
    (tmp_path / "manifest.json").write_text(json.dumps({"tokenizer": "gpt2", "splits": splits}))
    return tmp_path


def test_trainer_reduces_loss_on_learnable_data(tmp_path):
    cfg = TrainConfig(
        data_dir=str(_patterned_dataset(tmp_path)), out_dir=str(tmp_path / "run"),
        seq_len=32, micro_batch=4, global_batch_tokens=256, total_tokens=256 * 60,
        peak_lr=3e-3, eval_every=20, eval_batches=4,
    )
    history = train(TINY, cfg, device="cpu")

    steps = [r for r in history if "loss" in r]
    evals = [r for r in history if "val_loss" in r]
    assert len(steps) == 60 and [r["step"] for r in evals] == [19, 39, 59]
    assert steps[0]["loss"] > 3.5  # ~ln(64) = 4.16
    assert steps[-1]["loss"] < 0.5
    assert evals[-1]["val_loss"] < evals[0]["val_loss"]
    assert all(r["grad_norm"] > 0 for r in steps)

    # Every record is persisted, and the exact config is saved next to it.
    lines = (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()
    assert len(lines) == len(history)
    saved_model_cfg, saved_cfg = load_config(tmp_path / "run" / "config.yaml")
    assert saved_model_cfg == TINY and saved_cfg == cfg
