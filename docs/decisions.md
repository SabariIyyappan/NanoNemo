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

## Parameter count: 114.15M, architecture kept as specified
- Exact count: 114,150,144 total, 75,516,672 non-embedding (the 50304×768 embedding is tied with the output projection and counted once).
- This is slightly below the "≈ 115–125M" target. GQA's K/V projections are 3× smaller than in full multi-head attention (~9.4M fewer parameters), and the tied embedding saves another 38.6M.
- Widening `d_ff` to 2112 or adding a 13th layer would reach 115M+. The specified architecture was kept instead, since the shortfall is under 1%.

## Initial loss is ~10.98, not ln(V) = 10.83
- The final RMSNorm gives the hidden state unit RMS. Each logit is its dot product with a tied-embedding row (std 0.02), so logits start with std σ = 0.02·√768 ≈ 0.55.
- Random logits with spread σ cost about σ²/2 ≈ 0.15 nats more than a uniform prediction. The step-0 loss check expects ln(V) + σ²/2.

## MFU counts the tied lm_head matmul
- FLOPs per token = 6 × N_matmul + 12 × n_layers × d_model × seq_len (PaLM, appendix B), divided by the GPU's peak dense FP16 tensor-core throughput.
- The spec says "6 × non-embedding params". That convention assumes the embedding is only a lookup. Here the embedding is tied to the output projection, and the 768 → 50,304 lm_head matmul is real work: 6 × 38.6M ≈ 232 MFLOP per token, about 29% of the total. Leaving it out would understate MFU by about a third, so N_matmul counts it once. Norm gains are excluded (they do no matmul).
- For the spec model at seq 1024: 6 × 114,130,944 + 113,246,208 = 798,031,872 FLOPs per token.
- The attention term counts the full T × T score matrix, as PaLM does, even though causal attention needs only half of it. This matches the published convention, so numbers stay comparable with other reports.

## Peak FLOPs per GPU (MFU denominators)
| GPU | FP16 tensor (FP32 acc), TFLOP/s | FP32, TFLOP/s | Source |
|---|---|---|---|
| RTX 2070 Super laptop | 28.3 | 7.07 | 40 SMs × 1024 FLOP/clk/SM × 1.38 GHz rated boost, halved for FP32 accumulate (Turing whitepaper rates; NVIDIA mobile SUPER launch specs) |
| T4 | 65 | 8.1 | NVIDIA T4 datasheet |
| L4 | 121 (dense) | 30.3 | NVIDIA L4 datasheet (242 is the with-sparsity figure) |
- GeForce Turing runs FP16 tensor-core math with FP32 accumulation at half the FP16-accumulate rate. PyTorch accumulates in FP32, so the halved figure is the real ceiling. Datacenter parts (T4, L4) have no such halving.
- Peaks use the *rated* boost clock. The laptop GPU often boosts well past it (up to its 2100 MHz limit) when cool, so achieved clocks are recorded with every result rather than folded into the peak.
- MFU is also reported against the FP32 CUDA-core peak, which is the relevant ceiling for the FP32 S0 stage (Turing has no TF32).

## Benchmark methodology
- `scripts/bench_train.py` times the trainer's own `train_step` on real data: 5 warmup steps (excluded, but their time is recorded, since in later stages it includes compilation), then 3 trials of 50 optimizer steps. It reports the median and min/max across trials.
- Each step is bracketed by CUDA events. Throughput per trial is its tokens divided by the sum of its step times.
- Stage benchmark configs (`configs/stages/`) use a 16,384-token global batch (8 micro-batches of 2 at seq 1024) instead of the main run's 0.5M. Throughput per token doesn't depend on the accumulation count, and 150 timed steps take minutes instead of hours.
- An `nvidia-smi --loop-ms` subprocess samples SM clock, temperature, power and clock-event reasons during each trial. A trial is flagged if its lowest SM clock falls more than 10% below the median clock across trials, or if it reports a thermal or hardware slowdown. The software power cap is not a flag on its own: under full load a laptop GPU sits at its power limit all the time.

## The SDPA backend is pinned, never chosen automatically
- `F.scaled_dot_product_attention` picks a backend on its own. On Turing that is the memory-efficient kernel whenever the inputs allow it, even in FP32. Left alone, S0 would not be the math reference the ladder defines, and S2 would show no gain.
- `ModelConfig.attn_backend` (`math` or `mem_efficient`) is applied with `sdpa_kernel(...)`. A pinned backend that can't handle its inputs raises an error rather than silently falling back.
- FlashAttention (SDPA's `flash` backend) needs sm_80+, so it isn't an option on Turing. Our own CUDA C++ FlashAttention is stage S7.

## Mixed precision on Turing is FP16 + GradScaler
- Turing has no BF16 tensor cores. FP16's narrow range (max 65504, smallest normal 6e-5) means small gradients underflow, so the loss is scaled up before backward and the gradients scaled back down. A step whose gradients overflowed is skipped and the scale is halved.
- Gradients are unscaled before clipping, so the clip threshold applies to the true gradient norm.
- Weights, optimizer state and the residual stream stay FP32. Autocast runs matmuls and attention in FP16, and RMSNorm, RoPE and the loss upcast to FP32 explicitly.
- `precision: bf16` exists for the L4 stage (no scaler needed: BF16 has FP32's range).

## The training step syncs with the CPU once
- The per-micro-batch loss is summed on the GPU, and `.item()` is called once per optimizer step. A `.item()` per micro-batch made the CPU wait for the GPU 8 times per step, hiding nothing at S0 but defeating the asynchronous launches that S3 and S4 depend on.

## CUDA graphs keep gradients outside the graph's memory pool
- `torch.compile(mode="reduce-overhead")` replays captured CUDA graphs, which write their outputs into a private memory pool reused on every replay. With gradient accumulation, autograd adopts the first micro-batch's gradient tensor as `.grad`. That tensor lives in the pool, so the next replay would overwrite it. PyTorch detects this and raises an error.
- With CUDA graphs on, `.grad` is allocated once as ordinary memory and zeroed in place each step (`zero_grad(set_to_none=False)`), so each backward accumulates into it. The cost is ~0.43 GB of FP32 gradients that stay allocated through the forward pass, which is visible in S4's peak memory.
- The trainer marks each micro-batch as a new graph step (`torch.compiler.cudagraph_mark_step_begin()`), which does nothing without CUDA graphs.

## Laptop benchmarks need an absolute clock floor and automatic re-runs
- The laptop's platform power and thermal management can slow the GPU without the GPU being hot. Two cases seen in M4: (1) a whole run held at the 1140 MHz base clock at 80–86 W of 115 W, with no throttle reason reported; (2) a "SW thermal slowdown" that dropped the memory clock from 6801 to 810 MHz and power to ~30 W while the GPU read 48–50 °C.
- Case 1 is invisible to a trial-vs-trial comparison, because every trial agrees. So a trial is also flagged when its median SM clock is below the rated boost (1380 MHz) that the MFU peak assumes. This floor applies only to the laptop. A T4 runs below its boost clock under sustained load by design (70 W limit).
- `scripts/run_ladder.sh` re-runs a flagged stage after a 5-minute rest, up to 3 attempts. A GPU-temperature cooldown alone isn't enough, since the throttling sensor isn't the GPU's. Only unflagged results go into the ladder table; if a stage stays flagged, its row is marked.

## Custom kernels are `torch.library` custom ops, switched per kernel in `ModelConfig`
- Each Triton kernel is registered with `torch.library.custom_op`, plus `register_fake` (shapes/dtypes for tracing) and `register_autograd` (the backward is a second custom op). `torch.compile` then sees one opaque op and doesn't break the graph, and CUDA graphs capture the Triton launches like any other kernel. `torch.library.opcheck` tests every registration.
- The switches are `norm_kernel`, `swiglu_kernel`, `rope_kernel` (`torch` | `triton`) on `ModelConfig`, next to `attn_backend`. The plan's separate `KernelConfig` + `ops.py` registry would add a config section and an indirection layer for three switches; the layers simply pick their function at construction. `torch` stays the default, so the reference path is untouched.
- The S5 stage turns all three on together; `configs/ablations/s5_only_*.yaml` turn on one each, on top of S4, to measure each kernel alone.

## The residual stream travels between blocks as (x, delta)
- To fuse the residual add into the next RMSNorm, the add has to happen inside the norm. So each block returns `(x, delta)`: the stream and the last sublayer's output, not yet added. Each `AddRMSNorm` computes `h = x + delta`, then returns `(rmsnorm(h), h)`. The first block's norm gets `delta=None`; the final norm adds the last block's MLP output.
- With the `torch` path this runs exactly the same ops in the same order as before (the add, then the reference `RMSNorm`), so the reference model is unchanged.
- One kernel reads `x` and `delta` once and writes `h` and the normalized output once. Unfused, the add writes `h` and the norm reads it back.

## The Triton norm writes its output in the autocast dtype
- Under FP16 autocast the residual stream stays FP32, and the normalized output feeds a matmul that casts it to FP16 anyway. The Triton kernel writes it in FP16 directly: the same single rounding, 2 bytes per element written instead of 4, and no separate cast. The residual `h` is still written in FP32.

## Kernel microbenchmarks time GPU work only
- At these sizes (30–300 µs per call) `triton.testing.do_bench` measured mostly Python autograd and dispatch overhead: e.g. the RoPE forward+backward took 1.5 ms at batch 2 and 0.26 ms at batch 8. In training the GPU is never waiting on the CPU (S4 showed no launch overhead), so GPU time is what the ladder can gain.
- `benchmarks/kernels/common.py` captures each call (forward, or forward + backward) in a CUDA graph and replays it, timing each replay with CUDA events after flushing L2 with a 64 MB write. Without the flush, RoPE's batch-2 inputs (4 MB) would sit in the 4 MB L2 and overstate bandwidth.
- Capture needs PyTorch's recipe: warm up on a side stream first. The backward uses `torch.autograd.grad` (no accumulation into `.grad`).
- Bandwidth is the kernel's *minimum* bytes moved divided by time, against the DRAM peak (`PEAK_GBPS` in `nanonemo/utils/mfu.py`: 448 GB/s for the 2070S laptop, 320 for T4, 300 for L4).
- Clocks are sampled per row. A row with a slowdown reason or a memory clock more than 10% below the GPU's max memory clock is re-run after a rest. A first unmonitored run showed every provider slowing ~7× mid-run, to ~39 GB/s, consistent with the laptop's SW thermal slowdown (memory clock 6801 → 810 MHz).

## Liger-Kernel is the reference for kernel comparisons
- Liger-Kernel 0.8.5, installed with `--no-deps` so pip can't touch NGC's torch/Triton. Its fused add + RMSNorm, SwiGLU and RoPE match ours in function.
- Differences to keep in mind when reading the numbers: Liger's add + RMSNorm writes the normalized output in the residual's dtype (FP32 in our AMP setting), so its benchmark row includes the FP16 cast the matmul would need; Liger's RoPE rotates in place and takes full-width cos/sin tables.

## FlashAttention on Turing uses the WMMA API, not hand-written `mma.sync` PTX

The plan named `mma.sync.m16n8k8` + `ldmatrix`. The kernel uses `nvcuda::wmma` 16×16×16 FP16 fragments instead: the same tensor cores on sm_75, compiled to the same HMMA instructions, without inline PTX or hand-derived register layouts. The one place the opaque fragment layout gets in the way, rescaling the output rows during online softmax, is handled by loading a 16×16 "row index" matrix into an accumulator fragment once, so `rows.x[i]` tells each thread which row its element `i` belongs to. Raw `mma.sync` would give control over the layout (and allow keeping P in registers instead of a shared-memory round trip); that is the next step if the remaining gap to the CUTLASS kernel needs closing.

## The attention backward is parallel over key blocks, with atomics only for dQ

One block owns 32 keys of one KV head and loops over every query head in its GQA group (3) and every query tile at or after its keys. dK and dV for those keys accumulate in registers across the whole group, so GQA's sum over the group happens for free and needs no atomics or `repeat_interleave`. dQ is shared by all key blocks and is added with FP32 atomics, then cast. The alternative, a second kernel parallel over query blocks for dQ, recomputes S and dP; ncu showed L2 traffic at ~5% of peak, so the atomics are not the bottleneck.

## Shared-memory tiles are padded by 16 bytes per row

A 64-wide FP16 row is 128 bytes, so the 16 rows a WMMA load reads all start on the same bank. ncu put both attention kernels at 81–86% L1/shared-memory throughput with ~12% SM throughput; padding rows to 72 halves (FP16) / 36 floats (FP32) cut the forward from 643 to 293 µs and the backward from 2.4 to 1.5 ms.

## The CUDA RMSNorm is a drop-in for the Triton one

Same `torch.library` wiring and autograd (`nanonemo::cuda_add_rms_norm_fwd/bwd`), same contract (FP32 statistics, output in the autocast dtype, residual rounded to its stored dtype before normalizing), selected by `norm_kernel: cuda`. One warp per row, 16-byte vector loads, the row cached in registers, warp-shuffle reduction. Backward writes one dw partial row per warp (2 blocks per SM × 4 warps), summed after the kernel; 4 blocks per SM made that partial buffer large enough to cost 20% at 2048 rows.

## Fused linear + cross-entropy computes the gradient during the forward

The LM head matmul and the loss run chunk by chunk (512 rows): cuBLAS for the logits, a Triton kernel that writes each row's loss and overwrites the logits with `softmax - onehot`, then cuBLAS for dx and dW. The (B·T, vocab) logits never exist. The gradient is kept unscaled and O(1) in FP16; the 1/N of the mean and the incoming gradient (the GradScaler's scale) are applied in FP32 in backward, so small probabilities don't underflow FP16 the way `p / N` would. With `loss_kernel: fused`, `forward(idx, targets)` returns `logits=None`: nothing in training reads them.
