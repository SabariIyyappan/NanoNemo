"""Triton fused RoPE (Q and K in one launch) vs the plain-PyTorch apply_rope."""

import pytest
import torch

pytest.importorskip("triton")
from nanonemo.kernels.triton.rope import _rope, apply_rope_qk  # noqa: E402
from nanonemo.model.layers import apply_rope, rope_cache  # noqa: E402

pytestmark = pytest.mark.gpu

# (B, T, Hq, Hk, head_dim): the model, odd head counts / T, and a head_dim whose half isn't a power of 2.
SHAPES = [(2, 1024, 12, 4, 64), (1, 7, 3, 1, 64), (3, 33, 5, 5, 24), (1, 1, 1, 1, 2)]
TOL = {torch.float32: dict(rtol=1e-5, atol=1e-5), torch.float16: dict(rtol=1e-3, atol=1e-3)}


def _reference(q, k, cos, sin):
    # apply_rope takes (B, H, T, hd); return in the kernel's (B, T, H, hd) layout.
    rot = lambda x: apply_rope(x.transpose(1, 2), cos, sin).transpose(1, 2)  # noqa: E731
    return rot(q), rot(k)


def _inputs(B, T, hq, hk, hd, dtype):
    torch.manual_seed(0)
    q = torch.randn(B, T, hq, hd, device="cuda", dtype=dtype, requires_grad=True)
    k = torch.randn(B, T, hk, hd, device="cuda", dtype=dtype, requires_grad=True)
    cos, sin = (t.cuda() for t in rope_cache(hd, max(T, 64)))  # tables longer than T, as in the model
    return q, k, cos, sin


@pytest.mark.parametrize("B, T, hq, hk, hd", SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_matches_reference(B, T, hq, hk, hd, dtype):
    q, k, cos, sin = _inputs(B, T, hq, hk, hd, dtype)
    rq, rk = (t.detach().clone().requires_grad_() for t in (q, k))
    qo, ko = apply_rope_qk(q, k, cos, sin)
    rqo, rko = _reference(rq, rk, cos, sin)
    torch.testing.assert_close(qo, rqo, **TOL[dtype])
    torch.testing.assert_close(ko, rko, **TOL[dtype])
    gq, gk = torch.randn_like(qo), torch.randn_like(ko)
    torch.autograd.backward((qo, ko), (gq, gk))
    torch.autograd.backward((rqo, rko), (gq, gk))
    torch.testing.assert_close(q.grad, rq.grad, **TOL[dtype])
    torch.testing.assert_close(k.grad, rk.grad, **TOL[dtype])


def test_strided_input_matches_contiguous():
    # In the model the incoming gradient is a transposed view; the kernel reads it via strides.
    q, k, cos, sin = _inputs(2, 16, 4, 2, 64, torch.float32)
    q_t = q.detach().transpose(1, 2).contiguous().transpose(1, 2)  # same values, (B, H, T, hd) memory order
    k_t = k.detach().transpose(1, 2).contiguous().transpose(1, 2)
    assert not q_t.is_contiguous()
    for a, b in zip(apply_rope_qk(q_t, k_t, cos, sin), apply_rope_qk(q.detach(), k.detach(), cos, sin)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_inverse_undoes_rotation():
    q, k, cos, sin = _inputs(2, 16, 4, 2, 64, torch.float64)
    qo, ko = _rope(q, k, cos, sin, False)
    qb, kb = _rope(qo, ko, cos, sin, True)
    torch.testing.assert_close(qb, q)
    torch.testing.assert_close(kb, k)


def test_gradcheck_fp64():
    q, k, cos, sin = _inputs(2, 5, 3, 1, 8, torch.float64)
    assert torch.autograd.gradcheck(lambda q, k: apply_rope_qk(q, k, cos, sin), (q, k))


def test_opcheck():
    q, k, cos, sin = _inputs(2, 8, 4, 2, 64, torch.float16)
    torch.library.opcheck(_rope, (q, k, cos, sin, False))


def test_compiles_without_graph_breaks():
    q, k, cos, sin = _inputs(2, 64, 12, 4, 64, torch.float16)
    torch._dynamo.reset()
    fn = lambda q, k: apply_rope_qk(q, k, cos, sin)  # noqa: E731
    for a, b in zip(torch.compile(fn, fullgraph=True)(q, k), fn(q, k)):
        torch.testing.assert_close(a, b)
