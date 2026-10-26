"""Ladder-stage parity: each speedup must compute the same thing as the stage before it."""

import json

import numpy as np
import pytest
import torch

from nanonemo.data.loader import TokenLoader
from nanonemo.data.shard import write_shard
from nanonemo.model.config import ModelConfig
from nanonemo.model.layers import Attention
from nanonemo.train.config import TrainConfig, load_config
from nanonemo.train.optim import build_optimizer
from nanonemo.train.trainer import AMP_DTYPES, build_model, build_scaler, static_grads, train_step

# Big enough to exercise tensor-core tiles and a real softmax width; small enough to compile fast.
SMALL = ModelConfig(n_layers=2, d_model=256, n_heads=4, n_kv_heads=2, d_ff=512, vocab_size=512, max_seq_len=128)


def _patterned_dataset(tmp_path, vocab: int):
    rng = np.random.default_rng(0)
    tokens = np.tile(rng.integers(0, vocab, size=97), 400).astype(np.uint16)
    info = write_shard(tmp_path / "train_000000.bin", tokens, "gpt2")
    splits = {"train": {"n_tokens": len(tokens), "shards": [{"path": "train_000000.bin", **info}]}}
    (tmp_path / "manifest.json").write_text(json.dumps({"tokenizer": "gpt2", "splits": splits}))
    return str(tmp_path)


def _losses(model_cfg: ModelConfig, cfg: TrainConfig, steps: int) -> list[float]:
    torch.manual_seed(cfg.seed)
    model, step_model = build_model(model_cfg, cfg, "cuda")
    opt = build_optimizer(model, cfg.peak_lr, cfg.weight_decay)
    loader = TokenLoader(cfg.data_dir, "train", cfg.seq_len, cfg.micro_batch, seed=cfg.seed, device="cuda")
    amp_dtype, scaler = AMP_DTYPES[cfg.precision], build_scaler(cfg, "cuda")
    return [train_step(step_model, opt, loader, cfg.grad_accum(), cfg.grad_clip, amp_dtype, scaler, static_grads(cfg))[0]
            for _ in range(steps)]


def _cfg(tmp_path, **overrides) -> TrainConfig:
    return TrainConfig(data_dir=_patterned_dataset(tmp_path, SMALL.vocab_size), seq_len=128, micro_batch=4,
                       global_batch_tokens=1024, peak_lr=1e-3, **overrides)


KERNEL_SWITCHES = {"norm_kernel", "swiglu_kernel", "rope_kernel"}


def _changed(a, b) -> set[str]:
    (m0, t0), (m1, t1) = a, b
    return {k for k in vars(m0) if getattr(m0, k) != getattr(m1, k)} | {
        k for k in vars(t0) if getattr(t0, k) != getattr(t1, k)}


def test_stage_configs_differ_by_one_change():
    names = ["s0_fp32", "s1_fp16", "s2_mem_efficient", "s3_compile", "s4_cudagraphs", "s5_triton", "s6_cuda_norm", "s7_cuda_attn", "s8_fused_ce"]
    stages = [load_config(f"configs/stages/{n}.yaml") for n in names]
    for a, b in zip(stages, stages[1:]):
        changed = _changed(a, b)
        # S5 switches all three Triton kernels on together; that is its one change.
        assert len(changed) == 1 or changed == KERNEL_SWITCHES, changed


@pytest.mark.parametrize("kernel", ["norm", "swiglu", "rope"])
def test_s5_ablations_add_one_kernel_to_s4(kernel):
    s4 = load_config("configs/stages/s4_cudagraphs.yaml")
    assert _changed(s4, load_config(f"configs/ablations/s5_only_{kernel}.yaml")) == {f"{kernel}_kernel"}


def test_unknown_stage_settings_are_rejected():
    with pytest.raises(AssertionError, match="attn_backend"):
        ModelConfig(attn_backend="flash")  # needs sm_80+; not a Turing stage
    with pytest.raises(ValueError, match="precision"):
        TrainConfig(precision="fp8")
    with pytest.raises(ValueError, match="compile"):
        TrainConfig(compile="max_autotune")


@pytest.mark.gpu
@pytest.mark.parametrize("dtype, tol", [(torch.float32, 1e-5), (torch.float16, 2e-3)])
def test_mem_efficient_attention_matches_math(dtype, tol):
    torch.manual_seed(0)
    ref = Attention(ModelConfig(attn_backend="math")).cuda().to(dtype)
    fast = Attention(ModelConfig(attn_backend="mem_efficient")).cuda().to(dtype)
    fast.load_state_dict(ref.state_dict())
    from nanonemo.model.layers import rope_cache

    cos, sin = (t.cuda() for t in rope_cache(64, 512))
    x = torch.randn(2, 512, 768, device="cuda", dtype=dtype, requires_grad=True)
    x2 = x.detach().clone().requires_grad_()
    y_ref, y_fast = ref(x, cos, sin), fast(x2, cos, sin)
    torch.testing.assert_close(y_fast, y_ref, rtol=tol, atol=tol)
    g = torch.randn_like(y_ref)
    y_ref.backward(g)
    y_fast.backward(g)
    torch.testing.assert_close(x2.grad, x.grad, rtol=tol, atol=tol)


@pytest.mark.gpu
def test_fp16_amp_tracks_fp32(tmp_path):
    fp32 = _losses(SMALL, _cfg(tmp_path), steps=40)
    fp16 = _losses(SMALL, _cfg(tmp_path, precision="fp16"), steps=40)
    assert fp32[-1] < 0.5 * fp32[0], "reference run did not learn"
    # Same init, same batches: FP16 rounding nudges the trajectory but must not change it.
    np.testing.assert_allclose(fp16, fp32, rtol=0.02, atol=0.02)


@pytest.mark.gpu
@pytest.mark.parametrize("mode", ["default", "reduce-overhead"])
def test_compiled_training_tracks_eager(tmp_path, mode):
    torch._dynamo.reset()
    model_cfg = ModelConfig(**{**vars(SMALL), "attn_backend": "mem_efficient"})
    eager = _losses(model_cfg, _cfg(tmp_path, precision="fp16"), steps=15)
    compiled = _losses(model_cfg, _cfg(tmp_path, precision="fp16", compile=mode), steps=15)
    # Inductor reorders and fuses FP32/FP16 math, so this is a tolerance, not bitwise.
    np.testing.assert_allclose(compiled, eager, rtol=0.02, atol=0.02)


@pytest.mark.gpu
def test_triton_kernels_track_s4_training(tmp_path):
    torch._dynamo.reset()
    base = {**vars(SMALL), "attn_backend": "mem_efficient"}
    triton = {k: "triton" for k in KERNEL_SWITCHES}
    s4 = _losses(ModelConfig(**base), _cfg(tmp_path, precision="fp16", compile="reduce-overhead"), steps=15)
    s5 = _losses(ModelConfig(**{**base, **triton}), _cfg(tmp_path, precision="fp16", compile="reduce-overhead"), steps=15)
    np.testing.assert_allclose(s5, s4, rtol=0.02, atol=0.02)


@pytest.mark.parametrize("path", sorted(str(p) for p in __import__("pathlib").Path("configs").rglob("*.yaml")))
def test_every_config_loads(path):
    # A typo'd key or bad value fails here, not 20 minutes into a Kaggle session.
    load_config(path)
