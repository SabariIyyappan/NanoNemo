"""Python wrapper for the CUDA C++ fused residual add + RMSNorm (rmsnorm.cu).

Same op, signature and autograd as nanonemo.kernels.triton.rmsnorm; only the kernels differ,
so the two can be swapped with the model's norm_kernel switch and compared head to head.
"""

import torch
from torch import Tensor

from nanonemo.kernels.cuda import build


def _ext():
    return build("rmsnorm")


@torch.library.custom_op("nanonemo::cuda_add_rms_norm_fwd", mutates_args=())
def _fwd(x: Tensor, delta: Tensor | None, weight: Tensor, eps: float, out_dtype: torch.dtype) -> tuple[Tensor, Tensor, Tensor]:
    x = x.contiguous()
    delta = delta.contiguous() if delta is not None else None
    y, h, rstd = _ext().fwd(x, delta, weight.contiguous(), eps, out_dtype)
    return y, h, rstd


@_fwd.register_fake
def _(x, delta, weight, eps, out_dtype):
    h = x.new_empty(x.shape, dtype=torch.promote_types(x.dtype, delta.dtype)) if delta is not None else x.new_empty(0)
    return x.new_empty(x.shape, dtype=out_dtype), h, x.new_empty(x.shape[:-1], dtype=torch.float32)


@torch.library.custom_op("nanonemo::cuda_add_rms_norm_bwd", mutates_args=())
def _bwd(
    dy: Tensor, dh: Tensor | None, h: Tensor, weight: Tensor, rstd: Tensor,
    dx_dtype: torch.dtype, ddelta_dtype: torch.dtype | None,
) -> tuple[Tensor, Tensor, Tensor]:
    dx, dd, dw = _ext().bwd(dy.contiguous(), dh.contiguous() if dh is not None else None, h.contiguous(),
                            weight.contiguous(), rstd, dx_dtype, ddelta_dtype)
    return dx, dd, dw.to(weight.dtype)


@_bwd.register_fake
def _(dy, dh, h, weight, rstd, dx_dtype, ddelta_dtype):
    dd = h.new_empty(h.shape, dtype=ddelta_dtype) if ddelta_dtype is not None else h.new_empty(0)
    return h.new_empty(h.shape, dtype=dx_dtype), dd, torch.empty_like(weight)


def _setup_context(ctx, inputs, output):
    x, delta, weight, eps, out_dtype = inputs
    _, h, rstd = output
    ctx.save_for_backward(h if delta is not None else x, weight, rstd)
    ctx.x_dtype = x.dtype
    ctx.delta_dtype = delta.dtype if delta is not None else None


def _backward(ctx, dy, dh, _drstd):
    h, weight, rstd = ctx.saved_tensors
    has_delta = ctx.delta_dtype is not None
    dx, dd, dw = _bwd(dy, dh if has_delta else None, h, weight, rstd, ctx.x_dtype, ctx.delta_dtype)
    return dx, (dd if has_delta else None), dw, None, None


_fwd.register_autograd(_backward, setup_context=_setup_context)


def add_rms_norm(x: Tensor, delta: Tensor | None, weight: Tensor, eps: float) -> tuple[Tensor, Tensor]:
    """Returns (rmsnorm(x + delta) * weight, x + delta); delta=None normalizes x alone. See the Triton version."""
    out_dtype = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else x.dtype
    y, h, _ = _fwd(x, delta, weight, eps, out_dtype)
    return y, (x if delta is None else h)
