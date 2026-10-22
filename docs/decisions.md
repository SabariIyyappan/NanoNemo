# Design decisions

A running log of choices and the reasons behind them. Newest entries at the bottom.

## Environment: NGC PyTorch container `25.11-py3`
- The container ships a matched PyTorch / CUDA / cuDNN / NCCL / Triton / Nsight stack, so version mismatches don't confound benchmarks.
- The tag is pinned to `25.11` (PyTorch 2.10, CUDA 13.0.2). It is built for `sm_75` (`TORCH_CUDA_ARCH_LIST=7.5 8.0 8.6 9.0 10.0 12.0+PTX`), which covers the Turing GPUs used here (RTX 2070 Super, T4).
- Why not the newer `26.09` (CUDA 13.4): on the R580 driver it runs in CUDA forward-compatibility mode, which NVIDIA supports only on datacenter GPUs. `25.11`'s CUDA 13.0 runs natively.
- Kaggle runs use Kaggle's own image plus a pinned requirements file.

## Triton has no tensor-core `tl.dot` on Turing after 3.2
- Measured with a 2048³ FP16 GEMM on the RTX 2070 Super:

| Triton | PTX for `tl.dot` | TFLOP/s |
|---|---|---|
| 3.0.0, 3.2.0 | `mma.sync.aligned.m16n8k8` (tensor cores) | ~18 |
| 3.3.x | FMA fallback, then a compile crash | — |
| 3.4.0+ (incl. 3.8.0) | scalar `fma.rn`, 0 `mma.sync` | ~1.5 |
| cuBLAS (reference) | `turing_fp16_s1688gemm` | 20.9 |

- Nsight Compute confirms 0 tensor-pipe instructions in the Triton 3.8 kernel.
- Consequences:
  - Matmul-heavy custom kernels (FlashAttention) are written in CUDA C++ with `mma.sync`.
  - Triton is used only for memory-bound kernels that don't need `tl.dot`.
  - Fused linear-CE uses cuBLAS for the matmul.

## Precision on Turing: FP16 autocast + dynamic loss scaling
Turing has FP16 tensor cores but no BF16 or FP8. BF16 and FP8 are evaluated separately on an Ada GPU (L4).

## Attention on Turing
- FlashAttention-2 and PyTorch's SDPA flash backend require Ampere or newer.
- The baseline uses SDPA's memory-efficient backend, with the math backend as the reference.
- A FlashAttention-2-style kernel (causal, native GQA, head_dim 64) is written in CUDA C++ for sm_75, using `mma.sync.m16n8k8` and `ldmatrix`, as its own optimization stage. CUDA C++ rather than Triton because of the `tl.dot` limitation above.

## CUDA C++ kernels: fused RMSNorm + residual, then FlashAttention
- RMSNorm comes first. It is small enough to optimize thoroughly (warp shuffles, vectorized loads, shared-memory reduction), and it gives a direct comparison against the Triton version of the same op.
- FlashAttention builds on the same extension and test machinery.

## Data cleaning: custom exact dedup, no NeMo Curator
Nemotron-CC has already been globally deduplicated and quality-filtered upstream. A lightweight exact-hash dedup and length filter over our slice is enough.

## Data source: Nemotron-CC v1, `quality=high`, real text only
- Downloaded as JSONL from Common Crawl's open host (`data.commoncrawl.org/contrib/Nemotron/Nemotron-CC/`).
- Synthetic rephrasings (`kind=synthetic`) are excluded, so results reflect training on real web text.
- Nemotron-CC-v2 on Hugging Face requires manual access approval. It can be swapped in later through config.
- Development uses the ungated `nvidia/Nemotron-Pretraining-Dataset-sample`.

## RoPE uses the rotate-half pairing
- Dimension `i` is rotated together with `i + head_dim/2` (the Hugging Face Llama layout), not with `i + 1` (Meta's original interleaved layout).
- The two are mathematically equivalent, but checkpoints are not interchangeable between them.
- Rotate-half reads two contiguous halves of each head, which keeps the later Triton RoPE kernel simple.
- The cos/sin tables and the rotation are computed in FP32, then the result is cast back to the input dtype.
