"""Microbenchmark: fused residual add + RMSNorm (d_model 768).

    scripts/run_container.sh python benchmarks/kernels/rmsnorm.py

Two settings:
  amp   the model's: FP32 residual stream, FP16 sublayer output, normalized output in FP16
        (the next matmul's input). Liger always writes the output in the residual's dtype,
        so its row includes the FP32 -> FP16 cast the model would otherwise need.
  fp16  everything FP16: like-for-like with Liger, which targets this layout.
Bytes are the minimum per element: amp fwd reads x 4 + delta 2, writes h 4 + y 2; bwd reads
dy 2 + dh 4 + h 4, writes dx 4 + ddelta 2. fp16 is 2 bytes for each of those.
"""

from types import SimpleNamespace

import torch
from common import liger_available, run

from nanonemo.kernels.cuda.rmsnorm import add_rms_norm as cuda_add_rms_norm
from nanonemo.kernels.triton.rmsnorm import add_rms_norm
from nanonemo.model.layers import RMSNorm

EPS, N = 1e-5, 768


def reference(x, d, w, out_dtype):
    h = x + d
    return RMSNorm.forward(SimpleNamespace(eps=EPS, weight=w), h).to(out_dtype), h


def make_inputs(M, x_dtype, d_dtype):
    x = torch.randn(M, N, device="cuda", dtype=x_dtype, requires_grad=True)
    d = torch.randn(M, N, device="cuda", dtype=d_dtype, requires_grad=True)
    w = torch.ones(N, device="cuda", requires_grad=True)
    return x, d, w


def providers(M, x_dtype, d_dtype):
    def eager():
        x, d, w = make_inputs(M, x_dtype, d_dtype)
        return (lambda: reference(x, d, w, torch.float16)), [x, d, w]

    def compiled():
        x, d, w = make_inputs(M, x_dtype, d_dtype)
        fn = torch.compile(reference)
        return (lambda: fn(x, d, w, torch.float16)), [x, d, w]

    def ours(fn):
        def make():
            x, d, w = make_inputs(M, x_dtype, d_dtype)

            def fwd():
                with torch.autocast("cuda", dtype=torch.float16):
                    return fn(x, d, w, EPS)

            return fwd, [x, d, w]

        return make

    def liger():
        from liger_kernel.ops.fused_add_rms_norm import LigerFusedAddRMSNormFunction

        x, d, w = make_inputs(M, x_dtype, d_dtype)

        def fwd():
            y, h = LigerFusedAddRMSNormFunction.apply(x, d, w, EPS, 0.0, "llama", False)
            return y.to(torch.float16), h

        return fwd, [x, d, w]

    p = {"eager": eager, "compile": compiled, "triton": ours(add_rms_norm), "cuda": ours(cuda_add_rms_norm)}
    if liger_available():
        p["liger"] = liger
    return p


def main():
    cases = []
    for setting, x_dtype, (fwd_b, bwd_b) in [("amp", torch.float32, (12, 16)), ("fp16", torch.float16, (8, 10))]:
        for M in (2048, 8192):
            cases.append({"label": f"{setting} {M}x{N}", "shape": [M, N], "bytes_fwd": M * N * fwd_b,
                          "bytes_bwd": M * N * bwd_b, "providers": providers(M, x_dtype, torch.float16)})
    run("rmsnorm", cases)


if __name__ == "__main__":
    main()
