"""Microbenchmark: RoPE on Q (12 heads) and K (4 heads), head_dim 64, FP16, seq 1024.

    scripts/run_container.sh python benchmarks/kernels/rope.py

Inputs are (B, T, H, hd), as the projections produce them. Bytes: fwd reads and writes
q and k once each; bwd the same for dq and dk (the cos/sin tables are tiny and L2-resident).
Liger rotates in place (no output allocation) and takes full-width cos/sin tables.
"""

import torch
from common import liger_available, run

from nanonemo.kernels.triton.rope import apply_rope_qk
from nanonemo.model.layers import apply_rope, rope_cache

T, HQ, HK, HD = 1024, 12, 4, 64


def reference(q, k, cos, sin):
    return apply_rope(q.transpose(1, 2), cos, sin), apply_rope(k.transpose(1, 2), cos, sin)


def make_inputs(B):
    q = torch.randn(B, T, HQ, HD, device="cuda", dtype=torch.float16, requires_grad=True)
    k = torch.randn(B, T, HK, HD, device="cuda", dtype=torch.float16, requires_grad=True)
    cos, sin = (t.cuda() for t in rope_cache(HD, T))
    return q, k, cos, sin


def providers(B):
    def eager():
        q, k, cos, sin = make_inputs(B)
        return (lambda: reference(q, k, cos, sin)), [q, k]

    def compiled():
        q, k, cos, sin = make_inputs(B)
        fn = torch.compile(reference)
        return (lambda: fn(q, k, cos, sin)), [q, k]

    def triton():
        q, k, cos, sin = make_inputs(B)
        return (lambda: apply_rope_qk(q, k, cos, sin)), [q, k]

    def liger():
        from liger_kernel.ops.rope import LigerRopeFunction

        q, k, cos, sin = make_inputs(B)
        cos_full, sin_full = torch.cat([cos, cos], -1), torch.cat([sin, sin], -1)
        return (lambda: LigerRopeFunction.apply(q.transpose(1, 2), k.transpose(1, 2), cos_full, sin_full)), [q, k]

    p = {"eager": eager, "compile": compiled, "triton": triton}
    if liger_available():
        p["liger"] = liger
    return p


def main():
    per_token = (HQ + HK) * HD * 2 * 2  # read + write, FP16
    cases = [{"label": f"fp16 B={B} T={T}", "shape": [B, T, HQ, HK, HD], "bytes_fwd": B * T * per_token,
              "bytes_bwd": B * T * per_token, "providers": providers(B)} for B in (2, 8)]
    run("rope", cases)


if __name__ == "__main__":
    main()
