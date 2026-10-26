"""CUDA FlashAttention-2 style kernel vs PyTorch's causal SDPA (the math backend)."""

import pytest
import torch
import torch.nn.functional as F

from nanonemo.kernels.cuda.flash_attn import _flash_attn, flash_attn

pytestmark = pytest.mark.gpu

# (B, H, KVH, T, hd): the model's shape, MHA (no GQA), an odd T, and a short head_dim.
SHAPES = [(2, 12, 4, 1024, 64), (2, 4, 4, 37, 64), (1, 1, 1, 1, 32), (3, 8, 2, 65, 32)]
TOL = dict(rtol=2e-2, atol=2e-2)  # fp16 + atomic-add accumulation order for dK/dV


def _inputs(B, H, KVH, T, hd):
    torch.manual_seed(0)
    q = torch.randn(B, H, T, hd, device="cuda", dtype=torch.float16, requires_grad=True)
    k = torch.randn(B, KVH, T, hd, device="cuda", dtype=torch.float16, requires_grad=True)
    v = torch.randn(B, KVH, T, hd, device="cuda", dtype=torch.float16, requires_grad=True)
    return q, k, v


def _reference(q, k, v):
    group = q.shape[1] // k.shape[1]
    k = k.repeat_interleave(group, dim=1)
    v = v.repeat_interleave(group, dim=1)
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)


@pytest.mark.parametrize("B, H, KVH, T, hd", SHAPES)
def test_matches_reference(B, H, KVH, T, hd):
    q, k, v = _inputs(B, H, KVH, T, hd)
    rq, rk, rv = (t.detach().clone().requires_grad_() for t in (q, k, v))

    y = flash_attn(q, k, v)
    ry = _reference(rq, rk, rv)
    torch.testing.assert_close(y, ry, **TOL)

    g = torch.randn_like(y)
    y.backward(g)
    ry.backward(g)
    torch.testing.assert_close(q.grad, rq.grad, **TOL)
    torch.testing.assert_close(k.grad, rk.grad, **TOL)
    torch.testing.assert_close(v.grad, rv.grad, **TOL)


def test_causal_future_keys_dont_leak():
    # Changing a key/value strictly after position i must not change output i.
    q, k, v = _inputs(1, 4, 1, 16, 32)
    y = flash_attn(q, k, v)
    k2, v2 = k.detach().clone(), v.detach().clone()
    k2[:, :, 8:] = torch.randn_like(k2[:, :, 8:])
    v2[:, :, 8:] = torch.randn_like(v2[:, :, 8:])
    y2 = flash_attn(q.detach(), k2, v2)
    torch.testing.assert_close(y[:, :, :8], y2[:, :, :8], rtol=0, atol=0)


def test_opcheck():
    q, k, v = _inputs(1, 4, 2, 48, 32)
    torch.library.opcheck(_flash_attn, (q, k, v))


def test_compiles_without_graph_breaks():
    q, k, v = _inputs(2, 12, 4, 64, 64)
    torch._dynamo.reset()
    fn = torch.compile(flash_attn, fullgraph=True)
    torch.testing.assert_close(fn(q, k, v), flash_attn(q, k, v), **TOL)


def test_compiles_with_strided_inputs():
    # In the model q/k arrive as transposed views; the fake impl must match the real output strides.
    q, k, v = (t.detach().transpose(1, 2).contiguous().transpose(1, 2).requires_grad_() for t in _inputs(2, 12, 4, 64, 64))
    assert not q.is_contiguous()
    torch._dynamo.reset()
    y = torch.compile(flash_attn, fullgraph=True)(q, k, v)
    y.sum().backward()
    torch.testing.assert_close(y, flash_attn(q, k, v), **TOL)
