"""Fused residual add + RMSNorm in Triton.

Forward, one program per row (token):
    h = x + delta                              (the residual stream after a sublayer)
    y = h * rsqrt(mean(h^2) + eps) * w         (the next sublayer's normalized input)
Unfused, the add writes h and the norm reads it back. Here x and delta are read once and
h and y written once. Statistics are always computed in FP32 (FP64 for FP64 inputs).

Backward, with n = h * rstd and dn = dy * w:
    dh = rstd * (dn - n * mean(dn * n)) + dh_residual
    dw = sum over rows of dy * n
dx and ddelta are both dh (the add passes the gradient through to both inputs).
dw needs a sum over all rows, so each program handles a contiguous chunk of rows, keeps
its partial dw in registers, and writes one partial row; the partials are summed after.
"""

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _fwd_kernel(
    X, D, W, Y, H, RSTD,
    stride_x, stride_d, stride_y, stride_h,
    N, eps,
    HAS_DELTA: tl.constexpr, ACC: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    h = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(ACC)
    if HAS_DELTA:
        h += tl.load(D + row * stride_d + cols, mask=mask, other=0.0).to(ACC)
        # Normalize the residual as stored, so the result matches add-then-norm exactly.
        h = h.to(H.dtype.element_ty)
        tl.store(H + row * stride_h + cols, h, mask=mask)
        h = h.to(ACC)
    rstd = 1.0 / tl.sqrt(tl.sum(h * h, axis=0) / N + eps)
    tl.store(RSTD + row, rstd)
    w = tl.load(W + cols, mask=mask, other=0.0).to(ACC)
    tl.store(Y + row * stride_y + cols, (h * rstd * w).to(Y.dtype.element_ty), mask=mask)


@triton.jit
def _bwd_kernel(
    DY, DH, H, W, RSTD, DX, DD, DW,
    stride_dy, stride_dh, stride_h, stride_dx, stride_dd,
    M, N, rows_per_program,
    HAS_DH: tl.constexpr, HAS_DD: tl.constexpr, ACC: tl.constexpr, BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    w = tl.load(W + cols, mask=mask, other=0.0).to(ACC)
    dw = tl.zeros([BLOCK], dtype=ACC)
    row_start = pid * rows_per_program
    row_end = tl.minimum(row_start + rows_per_program, M)
    for r in range(row_start, row_end):
        row = r.to(tl.int64)
        dy = tl.load(DY + row * stride_dy + cols, mask=mask, other=0.0).to(ACC)
        rstd = tl.load(RSTD + row).to(ACC)
        n = tl.load(H + row * stride_h + cols, mask=mask, other=0.0).to(ACC) * rstd
        dw += dy * n
        dn = dy * w
        dh = (dn - n * (tl.sum(dn * n, axis=0) / N)) * rstd
        if HAS_DH:
            dh += tl.load(DH + row * stride_dh + cols, mask=mask, other=0.0).to(ACC)
        tl.store(DX + row * stride_dx + cols, dh.to(DX.dtype.element_ty), mask=mask)
        if HAS_DD:
            tl.store(DD + row * stride_dd + cols, dh.to(DD.dtype.element_ty), mask=mask)
    tl.store(DW + pid * N + cols, dw, mask=mask)


def _acc_dtype(dtype: torch.dtype):
    return (tl.float64, torch.float64) if dtype == torch.float64 else (tl.float32, torch.float32)


def _block_and_warps(N: int) -> tuple[int, int]:
    # One row per program; ~8 elements per thread keeps loads wide without spilling.
    BLOCK = triton.next_power_of_2(N)
    return BLOCK, min(max(BLOCK // 256, 1), 16)


@torch.library.custom_op("nanonemo::add_rms_norm_fwd", mutates_args=())
def _fwd(x: Tensor, delta: Tensor | None, weight: Tensor, eps: float, out_dtype: torch.dtype) -> tuple[Tensor, Tensor, Tensor]:
    N = x.shape[-1]
    x2 = x.reshape(-1, N)
    d2 = delta.reshape(-1, N) if delta is not None else x2
    assert x2.stride(-1) == 1 and d2.stride(-1) == 1, "last dim must be contiguous"
    acc_tl, acc_torch = _acc_dtype(x.dtype)
    y = torch.empty(x.shape, dtype=out_dtype, device=x.device)
    if delta is not None:
        h = torch.empty(x.shape, dtype=torch.promote_types(x.dtype, delta.dtype), device=x.device)
    else:
        h = x.new_empty(0)  # outputs may not alias inputs; the caller reuses x itself
    rstd = torch.empty(x.shape[:-1], dtype=acc_torch, device=x.device)
    BLOCK, num_warps = _block_and_warps(N)
    _fwd_kernel[(x2.shape[0],)](
        x2, d2, weight, y, h if delta is not None else y, rstd,
        x2.stride(0), d2.stride(0), N, N,
        N, eps,
        HAS_DELTA=delta is not None, ACC=acc_tl, BLOCK=BLOCK, num_warps=num_warps,
    )
    return y, h, rstd


@_fwd.register_fake
def _(x, delta, weight, eps, out_dtype):
    h = x.new_empty(x.shape, dtype=torch.promote_types(x.dtype, delta.dtype)) if delta is not None else x.new_empty(0)
    return x.new_empty(x.shape, dtype=out_dtype), h, x.new_empty(x.shape[:-1], dtype=_acc_dtype(x.dtype)[1])


@torch.library.custom_op("nanonemo::add_rms_norm_bwd", mutates_args=())
def _bwd(
    dy: Tensor, dh: Tensor | None, h: Tensor, weight: Tensor, rstd: Tensor,
    dx_dtype: torch.dtype, ddelta_dtype: torch.dtype | None,
) -> tuple[Tensor, Tensor, Tensor]:
    N = h.shape[-1]
    dy2, h2 = dy.reshape(-1, N), h.reshape(-1, N)
    dh2 = dh.reshape(-1, N) if dh is not None else dy2
    assert dy2.stride(-1) == 1 and dh2.stride(-1) == 1, "last dim must be contiguous"
    M = h2.shape[0]
    acc_tl, acc_torch = _acc_dtype(h.dtype)
    dx = torch.empty(h.shape, dtype=dx_dtype, device=h.device)
    dd = torch.empty(h.shape, dtype=ddelta_dtype, device=h.device) if ddelta_dtype is not None else h.new_empty(0)
    n_programs = min(M, torch.cuda.get_device_properties(h.device).multi_processor_count * 2)
    rows_per_program = triton.cdiv(M, n_programs)
    n_programs = triton.cdiv(M, rows_per_program)
    dw_partial = torch.empty(n_programs, N, dtype=acc_torch, device=h.device)
    BLOCK, num_warps = _block_and_warps(N)
    _bwd_kernel[(n_programs,)](
        dy2, dh2, h2, weight, rstd, dx, dd if ddelta_dtype is not None else dx, dw_partial,
        dy2.stride(0), dh2.stride(0), h2.stride(0), N, N,
        M, N, rows_per_program,
        HAS_DH=dh is not None, HAS_DD=ddelta_dtype is not None, ACC=acc_tl, BLOCK=BLOCK, num_warps=num_warps,
    )
    return dx, dd, dw_partial.sum(0).to(weight.dtype)


@_bwd.register_fake
def _(dy, dh, h, weight, rstd, dx_dtype, ddelta_dtype):
    dd = h.new_empty(h.shape, dtype=ddelta_dtype) if ddelta_dtype is not None else h.new_empty(0)
    return h.new_empty(h.shape, dtype=dx_dtype), dd, torch.empty_like(weight)


def _setup_context(ctx, inputs, output):
    x, delta, weight, eps, out_dtype = inputs
    _, h, rstd = output
    # Backward needs the normalized row: with an add that's the output h, otherwise x.
    ctx.save_for_backward(h if delta is not None else x, weight, rstd)
    ctx.x_dtype = x.dtype
    ctx.delta_dtype = delta.dtype if delta is not None else None


def _backward(ctx, dy, dh, _drstd):
    h, weight, rstd = ctx.saved_tensors
    has_delta = ctx.delta_dtype is not None
    dx, dd, dw = _bwd(dy.contiguous(), dh.contiguous() if has_delta else None, h, weight, rstd, ctx.x_dtype, ctx.delta_dtype)
    return dx, (dd if has_delta else None), dw, None, None


_fwd.register_autograd(_backward, setup_context=_setup_context)


def add_rms_norm(x: Tensor, delta: Tensor | None, weight: Tensor, eps: float) -> tuple[Tensor, Tensor]:
    """Returns (rmsnorm(x + delta) * weight, x + delta); delta=None normalizes x alone.

    Under autocast the normalized output is written directly in the autocast dtype: the
    next op is a matmul that would cast it there anyway, so this skips a separate cast pass.
    """
    out_dtype = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else x.dtype
    y, h, _ = _fwd(x, delta, weight, eps, out_dtype)
    return y, (x if delta is None else h)
