"""Fused SwiGLU activation in Triton: out = silu(g) * u, elementwise.

Forward reads g and u once and writes out once, computing in FP32 and rounding once.
Backward, with s = sigmoid(g):
    dg = dy * u * s * (1 + g * (1 - s))
    du = dy * g * s
s is recomputed from g rather than saved, so only g and u are kept for backward
(eager PyTorch also keeps silu(g), one more activation-sized tensor per layer).
"""

import torch
import triton
import triton.language as tl
from torch import Tensor

BLOCK = 1024


@triton.jit
def _fwd_kernel(G, U, OUT, n, ACC: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    g = tl.load(G + offs, mask=mask, other=0.0).to(ACC)
    u = tl.load(U + offs, mask=mask, other=0.0).to(ACC)
    s = 1.0 / (1.0 + tl.exp(-g))
    tl.store(OUT + offs, (g * s * u).to(OUT.dtype.element_ty), mask=mask)


@triton.jit
def _bwd_kernel(DY, G, U, DG, DU, n, ACC: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    dy = tl.load(DY + offs, mask=mask, other=0.0).to(ACC)
    g = tl.load(G + offs, mask=mask, other=0.0).to(ACC)
    u = tl.load(U + offs, mask=mask, other=0.0).to(ACC)
    s = 1.0 / (1.0 + tl.exp(-g))
    tl.store(DG + offs, (dy * u * s * (1.0 + g * (1.0 - s))).to(DG.dtype.element_ty), mask=mask)
    tl.store(DU + offs, (dy * g * s).to(DU.dtype.element_ty), mask=mask)


def _acc(dtype: torch.dtype):
    return tl.float64 if dtype == torch.float64 else tl.float32


@torch.library.custom_op("nanonemo::swiglu_fwd", mutates_args=())
def _fwd(g: Tensor, u: Tensor) -> Tensor:
    assert g.shape == u.shape and g.dtype == u.dtype, "gate and up must match"
    assert g.is_contiguous() and u.is_contiguous(), "inputs must be contiguous"
    out = torch.empty_like(g)
    n = g.numel()
    _fwd_kernel[(triton.cdiv(n, BLOCK),)](g, u, out, n, ACC=_acc(g.dtype), BLOCK=BLOCK, num_warps=4)
    return out


@_fwd.register_fake
def _(g, u):
    return torch.empty_like(g)


@torch.library.custom_op("nanonemo::swiglu_bwd", mutates_args=())
def _bwd(dy: Tensor, g: Tensor, u: Tensor) -> tuple[Tensor, Tensor]:
    assert dy.is_contiguous(), "grad must be contiguous"
    dg, du = torch.empty_like(g), torch.empty_like(u)
    n = g.numel()
    _bwd_kernel[(triton.cdiv(n, BLOCK),)](dy, g, u, dg, du, n, ACC=_acc(g.dtype), BLOCK=BLOCK, num_warps=4)
    return dg, du


@_bwd.register_fake
def _(dy, g, u):
    return torch.empty_like(g), torch.empty_like(u)


def _setup_context(ctx, inputs, output):
    ctx.save_for_backward(*inputs)


def _backward(ctx, dy):
    g, u = ctx.saved_tensors
    return _bwd(dy.contiguous(), g, u)


_fwd.register_autograd(_backward, setup_context=_setup_context)


def swiglu(g: Tensor, u: Tensor) -> Tensor:
    return _fwd(g.contiguous(), u.contiguous())
