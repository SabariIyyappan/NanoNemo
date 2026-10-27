"""Fused LM head + cross-entropy: loss = CE(x @ W^T, targets) without the full (N, V) logits.

Unfused, the logits for one micro-batch of 2048 tokens x 50,304 vocab are 206 MB in FP16,
plus an FP32 copy for the loss and its FP32 gradient: ~1 GB of activations for one matmul.
Here rows go through in chunks:
    logits_c = x_c @ W^T                       (cuBLAS, chunk_rows x V)
    Triton, per row: lse, loss = lse - logit[target], then overwrite the logits in place
                     with g = softmax - onehot(target)
    dx_c = g @ W,  dW += g^T @ x_c             (cuBLAS)
so only one chunk of logits ever exists, and the backward is done during the forward.
g is O(1), stored in the compute dtype; the 1/N of the mean and the incoming gradient (the
GradScaler's scale under FP16) are applied in FP32 in backward, so g never underflows.
"""

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _ce_kernel(LOGITS, stride, TARGETS, LOSS, V, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    ptr = LOGITS + row * stride
    t = tl.load(TARGETS + row)
    m = tl.full([], float("-inf"), tl.float32)
    s = tl.full([], 0.0, tl.float32)
    for off in range(0, V, BLOCK):  # online logsumexp over the row
        cols = off + tl.arange(0, BLOCK)
        x = tl.load(ptr + cols, mask=cols < V, other=float("-inf")).to(tl.float32)
        m_new = tl.maximum(m, tl.max(x, 0))
        s = s * tl.exp(m - m_new) + tl.sum(tl.exp(x - m_new), 0)
        m = m_new
    lse = m + tl.log(s)
    tl.store(LOSS + row, lse - tl.load(ptr + t).to(tl.float32))
    for off in range(0, V, BLOCK):  # d loss_row / d logits = softmax - onehot
        cols = off + tl.arange(0, BLOCK)
        mask = cols < V
        x = tl.load(ptr + cols, mask=mask, other=0.0).to(tl.float32)
        g = tl.exp(x - lse) - (cols == t).to(tl.float32)
        tl.store(ptr + cols, g.to(LOGITS.dtype.element_ty), mask=mask)


@torch.library.custom_op("nanonemo::linear_ce", mutates_args=())
def _linear_ce(x: Tensor, weight: Tensor, targets: Tensor, compute_dtype: torch.dtype,
               chunk_rows: int) -> tuple[Tensor, Tensor, Tensor]:
    """Returns (mean loss, dx, dW), the gradients unscaled: d(sum of row losses)."""
    N, D = x.shape
    V = weight.shape[0]
    w = weight.to(compute_dtype)
    loss_rows = torch.empty(N, dtype=torch.float32, device=x.device)
    dx = torch.empty(N, D, dtype=torch.float32, device=x.device)
    dw = torch.zeros(V, D, dtype=torch.float32, device=x.device)
    for s in range(0, N, chunk_rows):
        e = min(s + chunk_rows, N)
        xc = x[s:e].to(compute_dtype)
        logits = xc @ w.t()
        _ce_kernel[(e - s,)](logits, logits.stride(0), targets[s:e], loss_rows[s:e], V, BLOCK=4096, num_warps=8)
        dx[s:e] = logits @ w
        dw += logits.t() @ xc
    return loss_rows.mean(), dx, dw


@_linear_ce.register_fake
def _(x, weight, targets, compute_dtype, chunk_rows):
    f32 = dict(dtype=torch.float32, device=x.device)
    return torch.empty((), **f32), torch.empty(x.shape, **f32), torch.empty(weight.shape, **f32)


def _setup_context(ctx, inputs, output):
    x, weight, *_ = inputs
    _, dx, dw = output
    ctx.save_for_backward(dx, dw)
    ctx.x_dtype, ctx.w_dtype, ctx.n = x.dtype, weight.dtype, x.shape[0]


def _backward(ctx, dloss, _ddx, _ddw):
    dx, dw = ctx.saved_tensors
    scale = dloss.float() / ctx.n
    return (dx * scale).to(ctx.x_dtype), (dw * scale).to(ctx.w_dtype), None, None, None


_linear_ce.register_autograd(_backward, setup_context=_setup_context)

# Explicit Inductor fallback. Without it, Inductor's implicit-fallback path formats every input's IR
# into a log message even with logging off; this op's input is the IR of the whole network, and
# printing it never finished (S8 compile ran >15 min and once grew to 30 GB of host RAM).
from torch._inductor.lowering import make_fallback  # noqa: E402

make_fallback(torch.ops.nanonemo.linear_ce.default)


def linear_cross_entropy(x: Tensor, weight: Tensor, targets: Tensor, chunk_rows: int = 512) -> Tensor:
    """Mean CE of (x @ weight^T) vs targets. x: (N, D), weight: (V, D), targets: (N,) int64.

    The matmuls run in the autocast dtype when autocast is on (like nn.Linear would), else x's.
    """
    dtype = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else x.dtype
    return _linear_ce(x, weight, targets, dtype, chunk_rows)[0]
