import json

import numpy as np
import pytest

from nanonemo.data.shard import write_shard
from nanonemo.model.config import ModelConfig
from nanonemo.train.bench import benchmark
from nanonemo.train.config import TrainConfig, load_config

TINY = ModelConfig(n_layers=2, d_model=64, n_heads=4, n_kv_heads=2, d_ff=128, vocab_size=64, max_seq_len=32)


def _random_dataset(tmp_path):
    tokens = np.random.default_rng(0).integers(0, TINY.vocab_size, size=20_000).astype(np.uint16)
    info = write_shard(tmp_path / "train_000000.bin", tokens, "gpt2")
    splits = {"train": {"n_tokens": len(tokens), "shards": [{"path": "train_000000.bin", **info}]}}
    (tmp_path / "manifest.json").write_text(json.dumps({"tokenizer": "gpt2", "splits": splits}))
    return str(tmp_path)


def _cfg(tmp_path):
    return TrainConfig(data_dir=_random_dataset(tmp_path), seq_len=32, micro_batch=2, global_batch_tokens=128)


def test_benchmark_on_cpu(tmp_path):
    r = benchmark(TINY, _cfg(tmp_path), steps=4, trials=3, warmup_steps=2, device="cpu")

    assert r["bench"]["grad_accum"] == 2
    assert len(r["trials"]) == 3 and all(len(t["step_ms"]) == 4 for t in r["trials"])
    tps = r["tokens_per_s"]
    assert 0 < tps["min"] <= tps["median"] <= tps["max"]
    # Each trial's throughput is exactly its tokens over the sum of its step times.
    t = r["trials"][0]
    assert t["tokens_per_s"] == pytest.approx(4 * 128 / (sum(t["step_ms"]) / 1e3))
    assert r["mfu"] is None and r["peak_mem_gb"] is None  # no GPU peak on CPU
    assert r["config"]["model"]["d_model"] == 64
    json.dumps(r)  # the whole result must be saveable


def test_stage_configs_load():
    model_cfg, cfg = load_config("configs/stages/s0_fp32.yaml")
    assert model_cfg == ModelConfig()
    assert cfg.grad_accum() == 8


@pytest.mark.gpu
def test_benchmark_on_gpu(tmp_path):
    r = benchmark(TINY, _cfg(tmp_path), steps=100, trials=2, warmup_steps=2, device="cuda", monitor_interval_ms=50)

    assert r["peak_mem_gb"]["allocated"] > 0
    assert all(t["gpu"]["n_samples"] > 0 for t in r["trials"]), "GPU monitor saw no samples during trials"
    assert r["env"]["devices"], "env capture missing devices"
