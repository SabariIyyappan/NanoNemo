"""Multi-process training vs single-process, and checkpoint resume, through scripts/train.py.

Runs 2 ranks with torchrun on the CPU (gloo), so it needs no GPUs: the parallel layouts,
gradient sync and DCP checkpoints are the same code as on 2 GPUs with NCCL. The same
runs on real GPUs: NANONEMO_TEST_DEVICE=cuda pytest -m multigpu (Kaggle 2x T4).
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from nanonemo.data.shard import write_shard

ROOT = Path(__file__).resolve().parents[2]
DEVICE = os.environ.get("NANONEMO_TEST_DEVICE", "cpu")
pytestmark = pytest.mark.multigpu if DEVICE == "cuda" else pytest.mark.distributed

MODEL = dict(n_layers=2, d_model=64, n_heads=4, n_kv_heads=2, d_ff=128, vocab_size=512, max_seq_len=32)
TRAIN = dict(seq_len=32, micro_batch=2, global_batch_tokens=256, total_tokens=256 * 8, peak_lr=3e-3,
             warmup_frac=0.25, eval_every=1000, eval_batches=2, ckpt_async=False)


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    rng = np.random.default_rng(0)
    splits = {}
    for split, n in (("train", 40_000), ("val", 4_000)):
        tokens = np.tile(rng.integers(0, MODEL["vocab_size"], size=97), n // 97).astype(np.uint16)
        info = write_shard(d / f"{split}_000000.bin", tokens, "gpt2")
        splits[split] = {"n_tokens": len(tokens), "shards": [{"path": f"{split}_000000.bin", **info}]}
    (d / "manifest.json").write_text(json.dumps({"tokenizer": "gpt2", "splits": splits}))
    return str(d)


def _run(tmp_path, name, data_dir, nproc=1, model=None, device=DEVICE, torchrun=False, **train) -> list[float]:
    out = tmp_path / name
    cfg = {"model": {**MODEL, **(model or {})},
           "train": {**TRAIN, "data_dir": data_dir, "out_dir": str(out), **train}}
    if device == "cuda":
        cfg["train"]["precision"] = "fp16"
        cfg["model"]["attn_backend"] = "mem_efficient"
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(cfg))
    launcher = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={nproc}"]
    cmd = (launcher if nproc > 1 or torchrun else [sys.executable]) + [str(ROOT / "scripts/train.py"), "--config", str(path)]
    env = {**os.environ, "NANONEMO_DEVICE": device, "PYTHONPATH": str(ROOT), "OMP_NUM_THREADS": "1"}
    res = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-5000:]
    records = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
    return [r["loss"] for r in records if "loss" in r]


TOL = dict(rtol=1e-4, atol=1e-4) if DEVICE == "cpu" else dict(rtol=2e-2, atol=2e-2)


@pytest.fixture(scope="module")
def single(tmp_path_factory, data_dir):
    return _run(tmp_path_factory.mktemp("single"), "single", data_dir)


@pytest.mark.parametrize("parallel, extra", [
    ("ddp", {}),
    ("ddp", {"ddp_bucket_mb": 0.05, "ddp_grad_as_bucket_view": True}),
    ("fsdp", {}),
    ("fsdp", {"fsdp_prefetch": True, "fsdp_reshard_after_forward": False}),
    ("tp", {}),
], ids=["ddp", "ddp_tuned", "fsdp", "fsdp_prefetch", "tp"])
def test_two_ranks_match_one(tmp_path, data_dir, single, parallel, extra):
    losses = _run(tmp_path, parallel, data_dir, nproc=2, parallel=parallel, **extra)
    assert single[-1] < single[0], "reference run did not learn"
    # Same global batch each step (the loader splits it across ranks), same init: same losses.
    np.testing.assert_allclose(losses, single, **TOL)


@pytest.mark.parametrize("parallel, nproc", [("none", 1), ("ddp", 2), ("fsdp", 2)])
def test_resume_continues_the_same_run(tmp_path, data_dir, parallel, nproc):
    full = _run(tmp_path, "full", data_dir, nproc, parallel=parallel)
    first = _run(tmp_path, "part", data_dir, nproc, parallel=parallel, max_steps=3)
    both = _run(tmp_path, "part", data_dir, nproc, parallel=parallel, resume_from=str(tmp_path / "part/ckpt"))
    assert len(first) == 3
    # The resumed run restores weights, optimizer moments, LR schedule position and data
    # position, and appends to the same metrics file: together they are the uninterrupted run.
    np.testing.assert_allclose(both, full, **TOL)


def test_resume_into_a_different_layout(tmp_path, data_dir):
    # DCP reshards on load: an FSDP checkpoint from 2 ranks resumes in one process.
    full = _run(tmp_path, "full", data_dir, 2, parallel="fsdp")
    _run(tmp_path, "part", data_dir, 2, parallel="fsdp", max_steps=3)
    both = _run(tmp_path, "part", data_dir, 1, resume_from=str(tmp_path / "part/ckpt"))
    np.testing.assert_allclose(both, full, **TOL)


GPU = dict(rtol=2e-2, atol=2e-2)


@pytest.mark.gpu
def test_gpu_resume_with_fp16_and_cuda_graphs(tmp_path, data_dir):
    # Static .grad buffers (CUDA graphs) and the GradScaler's state must survive a resume.
    kw = dict(device="cuda", compile="reduce-overhead")
    full = _run(tmp_path, "full", data_dir, **kw)
    _run(tmp_path, "part", data_dir, max_steps=3, **kw)
    both = _run(tmp_path, "part", data_dir, resume_from=str(tmp_path / "part/ckpt"), **kw)
    np.testing.assert_allclose(both, full, **GPU)


@pytest.mark.gpu
@pytest.mark.parametrize("parallel", ["fsdp", "tp"])
def test_gpu_sharded_layouts_with_fp16(tmp_path, data_dir, parallel):
    # One rank under torchrun on one GPU: DTensor gradients through ShardedGradScaler and
    # clip_grad_norm_, which the CPU tests (FP32, no scaler) don't reach.
    single = _run(tmp_path, "single", data_dir, device="cuda")
    sharded = _run(tmp_path, parallel, data_dir, device="cuda", torchrun=True, parallel=parallel, compile="default")
    np.testing.assert_allclose(sharded, single, **GPU)
