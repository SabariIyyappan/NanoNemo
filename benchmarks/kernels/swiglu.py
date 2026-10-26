"""Microbenchmark: SwiGLU activation silu(g) * u (d_ff 2048, FP16 as under autocast).

    scripts/run_container.sh python benchmarks/kernels/swiglu.py

Bytes per element: fwd reads g, u and writes out (3 x 2); bwd reads dy, g, u and writes dg, du (5 x 2).
"""

import torch
from common import liger_available, run

from nanonemo.kernels.triton.swiglu import swiglu
from nanonemo.model.layers import silu_mul

F = 2048


def make_inputs(M):
    g = torch.randn(M, F, device="cuda", dtype=torch.float16, requires_grad=True)
    u = torch.randn(M, F, device="cuda", dtype=torch.float16, requires_grad=True)
    return g, u


def providers(M):
    def eager():
        g, u = make_inputs(M)
        return (lambda: (silu_mul(g, u),)), [g, u]

    def compiled():
        g, u = make_inputs(M)
        fn = torch.compile(silu_mul)
        return (lambda: (fn(g, u),)), [g, u]

    def triton():
        g, u = make_inputs(M)
        return (lambda: (swiglu(g, u),)), [g, u]

    def liger():
        from liger_kernel.ops.swiglu import LigerSiLUMulFunction

        g, u = make_inputs(M)
        return (lambda: (LigerSiLUMulFunction.apply(g, u),)), [g, u]

    p = {"eager": eager, "compile": compiled, "triton": triton}
    if liger_available():
        p["liger"] = liger
    return p


def main():
    cases = [{"label": f"fp16 {M}x{F}", "shape": [M, F], "bytes_fwd": M * F * 6, "bytes_bwd": M * F * 10,
              "providers": providers(M)} for M in (2048, 8192)]
    run("swiglu", cases)


if __name__ == "__main__":
    main()
