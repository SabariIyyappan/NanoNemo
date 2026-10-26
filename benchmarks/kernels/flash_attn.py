"""Microbenchmark: causal GQA attention at the model's shape (12 Q heads, 4 KV heads, hd 64, FP16).

    scripts/run_container.sh python benchmarks/kernels/flash_attn.py

Providers: SDPA math (S0-S1), SDPA mem_efficient (S2-S5, CUTLASS), and our CUDA kernel (S7).
The SDPA providers include the repeat_interleave of K/V that GQA needs for them; ours reads
K/V heads directly. Bytes = the minimum traffic (read Q, K, V, write O; bwd also dO, dQ, dK, dV),
so %-of-bandwidth is a lower bound: attention is compute-bound, not memory-bound.
"""

import torch
import torch.nn.functional as F
from common import run
from torch.nn.attention import SDPBackend, sdpa_kernel

from nanonemo.kernels.cuda.flash_attn import flash_attn

H, KVH, HD = 12, 4, 64


def make_inputs(B, T):
    q = torch.randn(B, H, T, HD, device="cuda", dtype=torch.float16, requires_grad=True)
    k = torch.randn(B, KVH, T, HD, device="cuda", dtype=torch.float16, requires_grad=True)
    v = torch.randn(B, KVH, T, HD, device="cuda", dtype=torch.float16, requires_grad=True)
    return q, k, v


def sdpa(backend):
    def fn(q, k, v):
        k, v = k.repeat_interleave(H // KVH, dim=1), v.repeat_interleave(H // KVH, dim=1)
        with sdpa_kernel(backend):
            return F.scaled_dot_product_attention(q, k, v, is_causal=True)

    return fn


def providers(B, T):
    def make(fn):
        def build():
            q, k, v = make_inputs(B, T)
            return (lambda: (fn(q, k, v),)), [q, k, v]

        return build

    p = {"sdpa_mem_efficient": make(sdpa(SDPBackend.EFFICIENT_ATTENTION)), "cuda": make(flash_attn)}
    if T <= 2048:  # math materializes B*H*T*T scores: 1.6 GB at T=4096, too much next to the graphs
        p["sdpa_math"] = make(sdpa(SDPBackend.MATH))
    return p


def main():
    cases = []
    # 8k tokens per call at every length, so rows differ only in T (attention cost grows ~T^2).
    for B, T in ((32, 256), (16, 512), (8, 1024), (4, 2048), (2, 4096)):
        qo = B * H * T * HD * 2
        kv = B * KVH * T * HD * 2
        cases.append({"label": f"fp16 B{B} T{T}", "shape": [B, H, KVH, T, HD],
                      "bytes_fwd": 2 * qo + 2 * kv, "bytes_bwd": 3 * qo + 4 * kv, "providers": providers(B, T)})
    run("flash_attn", cases)


if __name__ == "__main__":
    main()
