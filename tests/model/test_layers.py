import pytest
import torch
import torch.nn.functional as F

from nanonemo.model.config import ModelConfig
from nanonemo.model.layers import RMSNorm


def test_config_defaults_match_spec():
    cfg = ModelConfig()
    assert cfg.head_dim == 64
    assert cfg.n_heads // cfg.n_kv_heads == 3
    assert cfg.vocab_size % 64 == 0


def test_rmsnorm_matches_formula():
    torch.manual_seed(0)
    norm = RMSNorm(16)
    with torch.no_grad():
        norm.weight.uniform_(0.5, 1.5)
    x = torch.randn(2, 3, 16)
    expected = x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * norm.weight
    torch.testing.assert_close(norm(x), expected)


def test_rmsnorm_matches_torch_builtin():
    torch.manual_seed(0)
    norm = RMSNorm(768)
    x = torch.randn(4, 8, 768)
    torch.testing.assert_close(norm(x), F.rms_norm(x, (768,), norm.weight, eps=1e-5))


def test_rmsnorm_output_has_unit_rms():
    x = torch.randn(5, 768) * 37.0
    y = RMSNorm(768)(x)
    torch.testing.assert_close(y.pow(2).mean(-1), torch.ones(5), rtol=1e-4, atol=1e-4)


@pytest.mark.gpu
def test_rmsnorm_fp16_large_inputs_do_not_overflow():
    # |x| = 1000 -> x**2 = 1e6, far past FP16's max of 65504.
    x = torch.full((2, 768), 1000.0, device="cuda", dtype=torch.float16)
    norm = RMSNorm(768).cuda()
    y = norm(x)
    assert y.dtype == torch.float16
    assert torch.isfinite(y).all()
    torch.testing.assert_close(y, torch.ones_like(y))
