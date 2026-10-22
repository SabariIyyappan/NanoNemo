import pytest
import torch
import torch.nn.functional as F

from nanonemo.model.config import ModelConfig
from nanonemo.model.layers import RMSNorm, apply_rope, rope_cache


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


def test_rope_position_zero_is_identity():
    cos, sin = rope_cache(64, 16)
    x = torch.randn(2, 3, 1, 64)
    torch.testing.assert_close(apply_rope(x, cos, sin), x)


def test_rope_preserves_vector_length():
    cos, sin = rope_cache(64, 128)
    x = torch.randn(2, 3, 128, 64)
    torch.testing.assert_close(apply_rope(x, cos, sin).norm(dim=-1), x.norm(dim=-1))


def test_rope_matches_complex_rotation():
    # Independent reference: pair (x[i], x[i+32]) as a complex number, multiply by e^{i*m*theta_i}.
    head_dim, T = 64, 32
    cos, sin = rope_cache(head_dim, T)
    x = torch.randn(1, 2, T, head_dim, dtype=torch.float64)
    half = head_dim // 2
    theta = 10000.0 ** (-torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim)
    angles = torch.outer(torch.arange(T, dtype=torch.float64), theta)
    z = torch.complex(x[..., :half], x[..., half:]) * torch.polar(torch.ones_like(angles), angles)
    expected = torch.cat((z.real, z.imag), dim=-1)
    torch.testing.assert_close(apply_rope(x, cos, sin), expected, rtol=1e-5, atol=1e-5)


def test_rope_scores_depend_only_on_relative_position():
    cos, sin = rope_cache(64, 64)
    q, k = torch.randn(64), torch.randn(64)

    def score(m, n):
        seq = torch.zeros(1, 1, 64, 64)
        qs, ks = seq.clone(), seq.clone()
        qs[0, 0, m], ks[0, 0, n] = q, k
        return (apply_rope(qs, cos, sin)[0, 0, m] @ apply_rope(ks, cos, sin)[0, 0, n]).item()

    # Same offset (m - n = 5) at different absolute positions -> same score.
    assert score(7, 2) == pytest.approx(score(40, 35), rel=1e-4, abs=1e-4)
    assert score(7, 2) != pytest.approx(score(7, 3), rel=1e-4, abs=1e-4)
