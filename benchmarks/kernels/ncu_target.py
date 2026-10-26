"""Run one of our kernels' forward + backward once at the model's shape, as an Nsight Compute target.

    scripts/run_container.sh benchmarks/kernels/ncu.sh          # all three, see that script

Shapes are one micro-batch of the S5 stage: 2 x 1024 tokens, FP16 autocast.
"""

import sys

import torch

from nanonemo.kernels.cuda.flash_attn import flash_attn
from nanonemo.kernels.triton.rmsnorm import add_rms_norm
from nanonemo.kernels.triton.rope import apply_rope_qk
from nanonemo.kernels.triton.swiglu import swiglu
from nanonemo.model.layers import rope_cache

M, D, F, T, HQ, HK, HD = 2048, 768, 2048, 1024, 12, 4, 64


def rmsnorm():
    x = torch.randn(M, D, device="cuda", requires_grad=True)
    d = torch.randn(M, D, device="cuda", dtype=torch.float16, requires_grad=True)
    w = torch.ones(D, device="cuda", requires_grad=True)
    with torch.autocast("cuda", dtype=torch.float16):
        y, h = add_rms_norm(x, d, w, 1e-5)
    torch.autograd.backward((y, h), (torch.randn_like(y), torch.randn_like(h)))


def swiglu_():
    g, u = (torch.randn(M, F, device="cuda", dtype=torch.float16, requires_grad=True) for _ in range(2))
    swiglu(g, u).backward(torch.randn(M, F, device="cuda", dtype=torch.float16))


def rope():
    q = torch.randn(2, T, HQ, HD, device="cuda", dtype=torch.float16, requires_grad=True)
    k = torch.randn(2, T, HK, HD, device="cuda", dtype=torch.float16, requires_grad=True)
    cos, sin = (t.cuda() for t in rope_cache(HD, T))
    qo, ko = apply_rope_qk(q, k, cos, sin)
    torch.autograd.backward((qo, ko), (torch.randn_like(qo), torch.randn_like(ko)))


def attn():
    q = torch.randn(2, HQ, T, HD, device="cuda", dtype=torch.float16, requires_grad=True)
    k, v = (torch.randn(2, HK, T, HD, device="cuda", dtype=torch.float16, requires_grad=True) for _ in range(2))
    flash_attn(q, k, v).backward(torch.randn_like(q))


if __name__ == "__main__":
    fn = {"rmsnorm": rmsnorm, "swiglu": swiglu_, "rope": rope, "flash_attn": attn}[sys.argv[1]]
    fn()  # first call compiles the kernels; ncu skips it via --launch-skip
    fn()
    torch.cuda.synchronize()
