"""Python wrapper for the CUDA FlashAttention-2 style kernel (flash_attn.cu).

Registered as a torch.library custom op (forward returns (o, logsumexp), backward
recomputes scores from the saved logsumexp) so torch.compile sees one opaque node
and autograd gets a proper backward, the same pattern as the Triton kernels.
"""

import torch
from torch import Tensor

from nanonemo.kernels.cuda import build


def _ext():
    return build("flash_attn")


@torch.library.custom_op("nanonemo::flash_attn", mutates_args=())
def _flash_attn(q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
    q, k, v = (t if t.is_contiguous() else t.contiguous() for t in (q, k, v))
    o, lse = _ext().fwd(q, k, v)
    return o, lse


@_flash_attn.register_fake
def _(q, k, v):
    B, H, T, _ = q.shape  # the kernel returns contiguous tensors whatever the input strides
    return torch.empty(q.shape, dtype=q.dtype, device=q.device), torch.empty((B, H, T), dtype=torch.float32, device=q.device)


@torch.library.custom_op("nanonemo::flash_attn_bwd", mutates_args=())
def _flash_attn_bwd(d_o: Tensor, q: Tensor, k: Tensor, v: Tensor, o: Tensor, lse: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    d_o, q, k, v = (t if t.is_contiguous() else t.contiguous() for t in (d_o, q, k, v))
    dq, dk, dv = _ext().bwd(d_o, q, k, v, o, lse)
    return dq, dk, dv


@_flash_attn_bwd.register_fake
def _(d_o, q, k, v, o, lse):
    return tuple(torch.empty(t.shape, dtype=t.dtype, device=t.device) for t in (q, k, v))


def _setup_context(ctx, inputs, output):
    q, k, v = inputs
    o, lse = output
    ctx.save_for_backward(q, k, v, o, lse)


def _backward(ctx, d_o, _d_lse):
    q, k, v, o, lse = ctx.saved_tensors
    return _flash_attn_bwd(d_o, q, k, v, o, lse)


_flash_attn.register_autograd(_backward, setup_context=_setup_context)


def flash_attn(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    """Causal GQA attention. q: (B, H, T, hd) fp16, k/v: (B, KVH, T, hd) fp16, H % KVH == 0."""
    o, _ = _flash_attn(q, k, v)
    return o
