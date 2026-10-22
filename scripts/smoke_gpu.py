"""GPU smoke test: torch, cuBLAS FP16 GEMM, and a Triton tensor-core kernel on this GPU.

Also a small, fast target for checking that nsys and ncu work:
    nsys profile -o results/smoke/smoke python scripts/smoke_gpu.py
    ncu -k regex:matmul_kernel --set full -o results/smoke/smoke python scripts/smoke_gpu.py
"""

import json

import torch
import triton
import triton.language as tl

from nanonemo.utils.env import capture_env


@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    # One program computes one BLOCK_M x BLOCK_N tile of C = A @ B (all row-major, contiguous).
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptr + rm[:, None] * K + (k + rk)[None, :])
        b = tl.load(b_ptr + (k + rk)[:, None] * N + rn[None, :])
        acc += tl.dot(a, b)  # FP16 inputs, FP32 accumulate -> tensor cores (mma) on sm_75
    tl.store(c_ptr + rm[:, None] * N + rn[None, :], acc.to(tl.float16))


def triton_matmul(a: torch.Tensor, b: torch.Tensor, return_kernel: bool = False):
    M, K = a.shape
    _, N = b.shape
    BLOCK = 64
    assert M % BLOCK == 0 and N % BLOCK == 0 and K % 32 == 0, "smoke kernel has no boundary masking"
    c = torch.empty(M, N, device=a.device, dtype=torch.float16)
    kernel = matmul_kernel[(M // BLOCK, N // BLOCK)](a, b, c, M, N, K, BLOCK_M=BLOCK, BLOCK_N=BLOCK, BLOCK_K=32)
    return (c, kernel) if return_kernel else c


def gemm_tflops(fn, n: int, iters: int = 20) -> float:
    for _ in range(3):  # warmup (includes Triton JIT compile on first call)
        fn()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()  # events are async; wait before reading the time
    seconds = start.elapsed_time(end) / 1e3 / iters
    return 2 * n**3 / seconds / 1e12


def main() -> None:
    env = capture_env()
    print(json.dumps({k: env[k] for k in ("torch", "triton", "cuda_runtime", "container", "torch_arch_list", "devices")}, indent=2))
    assert torch.cuda.is_available(), "CUDA not available"

    n = 2048
    torch.manual_seed(0)
    a = torch.randn(n, n, device="cuda", dtype=torch.float16)
    b = torch.randn(n, n, device="cuda", dtype=torch.float16)
    ref = a.float() @ b.float()

    out_cublas = a @ b
    out_triton, kernel = triton_matmul(a, b, return_kernel=True)
    # Triton > 3.2 lowers tl.dot on sm_75 to scalar FMA instead of tensor-core mma (docs/decisions.md).
    n_mma = kernel.asm["ptx"].count("mma.sync")
    print(f"Triton tl.dot uses tensor cores: {n_mma > 0} ({n_mma} mma.sync in PTX)")
    # FP16 output of a K=2048 dot product: compare relative to the magnitude of the result.
    scale = ref.abs().max().item()
    err_cublas = (out_cublas.float() - ref).abs().max().item() / scale
    err_triton = (out_triton.float() - ref).abs().max().item() / scale
    print(f"max rel err  cuBLAS: {err_cublas:.2e}  Triton: {err_triton:.2e}")
    assert err_cublas < 1e-2 and err_triton < 1e-2, "GEMM result mismatch"

    print(f"cuBLAS FP16 GEMM {n}^3: {gemm_tflops(lambda: a @ b, n):.1f} TFLOP/s")
    print(f"Triton FP16 GEMM {n}^3: {gemm_tflops(lambda: triton_matmul(a, b), n):.1f} TFLOP/s")
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
