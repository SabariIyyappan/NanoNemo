import pytest
from torch import nn

from nanonemo.model.config import ModelConfig
from nanonemo.model.model import NanoNemo
from nanonemo.utils.mfu import flops_per_token, matmul_params, mfu, peak_tflops, rated_boost_mhz

SMALL = ModelConfig(n_layers=2, d_model=64, n_heads=4, n_kv_heads=2, d_ff=128, vocab_size=64, max_seq_len=32)


@pytest.mark.parametrize("cfg", [SMALL, ModelConfig()])
def test_matmul_params_match_the_model(cfg):
    # Count from the built model: every Linear weight, with the tied lm_head/embedding once.
    model = NanoNemo(cfg)
    linear = sum(m.weight.numel() for m in model.modules() if isinstance(m, nn.Linear))
    assert matmul_params(cfg) == linear


def test_flops_per_token_for_the_spec_model():
    cfg = ModelConfig()
    # 114,150,144 params minus 25 RMSNorm gains of 768 = 114,130,944 matmul weights.
    assert matmul_params(cfg) == 114_150_144 - 25 * 768
    assert flops_per_token(cfg, 1024) == 6 * 114_130_944 + 12 * 12 * 768 * 1024


@pytest.mark.parametrize(
    "name, key",
    [
        ("NVIDIA GeForce RTX 2070 Super", "rtx 2070 super"),
        ("NVIDIA GeForce RTX 2070 SUPER Laptop GPU", "rtx 2070 super"),
        ("Tesla T4", "t4"),
        ("NVIDIA L4", "l4"),
    ],
)
def test_peak_lookup(name, key):
    from nanonemo.utils.mfu import PEAK_TFLOPS

    assert peak_tflops(name) is PEAK_TFLOPS[key]


def test_unknown_gpu_has_no_peak():
    assert peak_tflops("NVIDIA GeForce RTX 2070") is None  # non-Super: different SM count
    assert peak_tflops("NVIDIA A100-SXM4-80GB") is None


def test_mfu_arithmetic():
    # 10k tok/s at 1 GFLOP/token = 10 TFLOP/s, a quarter of a 40 TFLOP/s peak.
    assert mfu(10_000, 10**9, 40.0) == pytest.approx(0.25)


def test_rated_boost_lookup():
    assert rated_boost_mhz("NVIDIA GeForce RTX 2070 Super") == 1380
    assert rated_boost_mhz("Tesla T4") is None  # power-limited below boost by design
    assert rated_boost_mhz("NVIDIA A100-SXM4-80GB") is None
