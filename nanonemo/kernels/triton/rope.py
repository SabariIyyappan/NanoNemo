"""Fused RoPE for Q and K in Triton (rotate-half layout, like nanonemo.model.layers.apply_rope).

Inputs are (B, T, heads, head_dim), the layout the projections produce, before the
transpose to (B, heads, T, head_dim). One program per token: it loads that position's
cos/sin row once and rotates every Q head and every K head with it, in a single launch.

With x1, x2 the two halves of a head:   out = (x1*cos - x2*sin, x1*sin + x2*cos).
The backward is the same rotation by -theta (a rotation's transpose is its inverse),
so it's the same kernel with sin negated, and nothing needs to be saved but the tables.
"""

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _rotate(X, OUT, row_x, row_out, stride_xh, cos, sin, n_heads, HALF,
            H_PAD: tl.constexpr, HALF_PAD: tl.constexpr, ACC: tl.constexpr):
    h = tl.arange(0, H_PAD)[:, None]
    d = tl.arange(0, HALF_PAD)[None, :]
    mask = (h < n_heads) & (d < HALF)
    x_ptr = X + row_x + h * stride_xh + d
    o_ptr = OUT + row_out + h * (2 * HALF) + d
    x1 = tl.load(x_ptr, mask=mask, other=0.0).to(ACC)
    x2 = tl.load(x_ptr + HALF, mask=mask, other=0.0).to(ACC)
    tl.store(o_ptr, (x1 * cos - x2 * sin).to(OUT.dtype.element_ty), mask=mask)
    tl.store(o_ptr + HALF, (x1 * sin + x2 * cos).to(OUT.dtype.element_ty), mask=mask)


@triton.jit
def _rope_kernel(
    Q, K, QO, KO, COS, SIN,
    sq_b, sq_t, sq_h, sk_b, sk_t, sk_h, stride_cos,
    T, n_q, n_k, HALF,
    INVERSE: tl.constexpr, QH_PAD: tl.constexpr, KH_PAD: tl.constexpr, HALF_PAD: tl.constexpr, ACC: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    b, t = row // T, row % T
    d = tl.arange(0, HALF_PAD)[None, :]
    cos = tl.load(COS + t * stride_cos + d, mask=d < HALF, other=0.0).to(ACC)
    sin = tl.load(SIN + t * stride_cos + d, mask=d < HALF, other=0.0).to(ACC)
    if INVERSE:
        sin = -sin
    _rotate(Q, QO, b * sq_b + t * sq_t, row * n_q * 2 * HALF, sq_h, cos, sin, n_q, HALF, QH_PAD, HALF_PAD, ACC)
    _rotate(K, KO, b * sk_b + t * sk_t, row * n_k * 2 * HALF, sk_h, cos, sin, n_k, HALF, KH_PAD, HALF_PAD, ACC)


@torch.library.custom_op("nanonemo::rope", mutates_args=())
def _rope(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor, inverse: bool) -> tuple[Tensor, Tensor]:
    B, T, n_q, hd = q.shape
    n_k = k.shape[2]
    assert k.shape == (B, T, n_k, hd) and hd % 2 == 0
    # Any b/t/head strides are read in place; only the last dim must be contiguous.
    q = q if q.stride(-1) == 1 else q.contiguous()
    k = k if k.stride(-1) == 1 else k.contiguous()
    assert cos.stride(-1) == 1 and cos.stride() == sin.stride()
    assert cos.shape[0] >= T and cos.shape[1] == hd // 2
    qo = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    ko = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    half = hd // 2
    _rope_kernel[(B * T,)](
        q, k, qo, ko, cos, sin,
        q.stride(0), q.stride(1), q.stride(2), k.stride(0), k.stride(1), k.stride(2), cos.stride(0),
        T, n_q, n_k, half,
        INVERSE=inverse,
        QH_PAD=triton.next_power_of_2(n_q), KH_PAD=triton.next_power_of_2(n_k),
        HALF_PAD=triton.next_power_of_2(half),
        ACC=tl.float64 if q.dtype == torch.float64 else tl.float32,
        num_warps=4,
    )
    return qo, ko


@_rope.register_fake
def _(q, k, cos, sin, inverse):
    return torch.empty(q.shape, dtype=q.dtype, device=q.device), torch.empty(k.shape, dtype=k.dtype, device=k.device)


def _setup_context(ctx, inputs, output):
    _, _, cos, sin, inverse = inputs
    ctx.save_for_backward(cos, sin)
    ctx.inverse = inverse


def _backward(ctx, dq, dk):
    cos, sin = ctx.saved_tensors
    dq, dk = _rope(dq, dk, cos, sin, not ctx.inverse)
    return dq, dk, None, None, None


_rope.register_autograd(_backward, setup_context=_setup_context)


def apply_rope_qk(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    """Rotate q (B, T, Hq, hd) and k (B, T, Hk, hd) by position. cos/sin: (>= T, hd/2) tables."""
    return _rope(q, k, cos, sin, False)
