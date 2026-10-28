# Devlog

Short notes written while building: what we tried, what surprised us, and the numbers.

## M0: environment

- The NGC PyTorch container works on the RTX 2070 Super (sm_75). Nsight Systems traces and Nsight Compute counters both work inside it, after setting `NVreg_RestrictProfilingToAdminUsers=0` on the host.
- The first container tried (`26.09`, CUDA 13.4) printed "CUDA Forward Compatibility mode ENABLED" on the R580 driver. We switched to `25.11` (CUDA 13.0), which runs natively.
- **Surprise:** a 20-line Triton FP16 matmul was correct but ran at 1.8 TFLOP/s, against cuBLAS's 20.9. Three checks found the cause:
  - ncu reported `sm__inst_executed_pipe_tensor.sum = 0`.
  - The Triton IR had a `#blocked` layout instead of `nvidia_mma`.
  - The PTX had 1024 `fma.rn` and no `mma.sync`.
- Bisecting Triton versions found the cause: 3.0 and 3.2 emit `mma.sync.m16n8k8` (~18 TFLOP/s), while 3.4+ fall back to FMA on Turing. So the FlashAttention kernel will be written in CUDA C++, not Triton.
- Lesson: a correctness test can't tell you whether you're using the hardware. Check the counters.

## M1: data pipeline

- Shard format: a 256-byte header (magic, version, token count, tokenizer), then `uint16` tokens. Writes are atomic (`.tmp` + `fsync` + rename), and reads use `np.memmap`.
- **Bug caught before commit:** the first `prepare()` draft streamed val and train as two lazy generators over one document iterator. While the val pass ran, every train document's text accumulated in memory, and the train pass then tokenized all of it at once. The tests passed because the sample is tiny, but at 2.5B tokens this would have held the whole training set's text in RAM. It was rewritten as a single pass with a per-split `ShardWriter`. Memory is now bounded by one tokenize batch plus one shard.
- The train/val split is decided per document (hash of the doc id) *before* tokenizing, so no document straddles the boundary.
- Nemotron-CC high-quality sample: 785 docs → 32 too short (<200 chars) → **0 exact duplicates** (as expected; upstream already deduplicates) → 753 kept → 513,098 train / 6,052 val tokens.
- Loader: windows of 1025 tokens with stride 1024. Each epoch visits every window once, in a permutation seeded by (seed, epoch). The sample index `(micro_step·world + rank)·B + b` gives disjoint ranks, and the loader state is just `micro_step`. The resume test crosses an epoch boundary and matches the uninterrupted run exactly.
- Loader speed on the dev shards: 0.10 ms per 8×1024 batch (~81M tokens/s, including the pinned host-to-GPU copy). That's thousands of times faster than the model will consume tokens, so no worker processes are needed.
- `prepare()` now refuses to produce an empty split. With N docs, P(empty val) = (1 − val_frac)^N, which only matters for tiny inputs.
- Full run: 12 Nemotron-CC v1 `quality=high` files (one evenly spaced from each of 12 different crawl eras, 2013–2024), downloaded resumably (4.1 GB), then tokenized in 7m52s on 16 CPU cores, peak container RAM under 1 GB. 3,306,160 documents seen → 134,077 too short, 3,263 exact duplicates → 3,168,820 kept.
- Result: **2,415,765,275 train tokens** (25 shards) and **9,481,778 val tokens** (1 shard), 4.6 GB on disk. All 26 shards re-verified against the manifest's checksums.
- Train came in at 2.42B, not exactly 2.5B: the 12 downloaded files ran out of documents before the budget was reached (round-robin drained them evenly, then the source was exhausted). Still well above the spec's 2B-minimum fallback, so kept as is rather than fetching more files.
- Uploaded as a **private** Kaggle dataset `sabariiyyappand/nanonemo-nemotron-cc` (26 shards + manifest.json, 4.6 GB). Verified: all 27 files present, sizes match the local copies exactly, dataset status "ready", and the dataset page 404s when fetched unauthenticated (confirms it's private, not public).

## M2: reference model

- Built bottom-up, one tested block at a time: RMSNorm → RoPE → GQA attention → SwiGLU MLP → pre-norm block → full model. Each block is checked against an independently written reference: the formula by hand, `F.rms_norm`, a complex-number RoPE, and a per-head attention loop that indexes K/V head `h // 3` instead of repeating it.
- RMSNorm reduces in FP32. A GPU test feeds FP16 inputs of 1000 (x² = 1e6, past FP16's max of 65504); without the upcast the output collapses to 0.
- RoPE uses the rotate-half pairing. A test checks the defining property: the q·k score at positions (7, 2) equals the score at (40, 35).
- **Parameter count: 114,150,144** (75,516,672 non-embedding), slightly under the "≈ 115–125M" target because GQA shrinks K/V and the embedding is tied. Architecture kept as specified (see decisions.md).
- **Surprise:** the initial loss was 10.99, not ln(50304) = 10.83, and the first version of the test failed. This isn't a bug: the unit-RMS final norm times the tied embedding (std 0.02) gives logits with std 0.02·√768 ≈ 0.55, which adds about σ²/2 = 0.15 nats. The test now derives the expected value.
- Overfit one batch (AdamW, no weight decay, 100 steps): the small config on CPU goes 4.87 → 0.009 in about 1 s; the full model on the RTX 2070 Super goes 10.95 → 0.004 in 9 s (FP32, B=4, T=128, 2.5 GB peak).

## M2: trainer and smoke run

- Trainer (S0 reference): eager FP32, AdamW (β 0.9/0.95, wd 0.1 on matmul weights only), linear warmup then cosine decay to 10% of peak, gradient clipping at 1.0, gradient accumulation, fixed-batch validation loss, metrics to `metrics.jsonl` and W&B. The LR is a pure function of the step, so resume needs no scheduler state.
- A test confirms gradient accumulation is exact: 2 micro-batches of 2 give the same gradients as one batch of 4.
- FP32 memory on the RTX 2070 Super at seq 1024: micro-batch 1 → 2.81 GB, 2 → 4.31 GB, 4 → OOM. That is about 1.37 GB fixed (weights + AdamW state) plus about 1.5 GB of saved activations per 1024-token window, roughly half of it the 50,304-wide logits and their gradient.
- **Smoke run** (`configs/smoke.yaml`, commit `ace0058`): 305 steps × 65,536 tokens = 19,988,480 tokens (0.8% of one epoch), micro-batch 2 × 32 accumulation, peak LR 6e-4, 6 warmup steps.

| step | train loss | val loss | grad norm |
|---|---|---|---|
| 0 | 10.94 | — | 11.3 |
| 10 | 7.85 | — | 1.32 |
| 24/25 | 7.46 | 7.51 | 0.65 |
| 99/100 | 6.44 | 6.53 | 0.45 |
| 199/200 | 6.09 | 6.07 | 0.51 |
| 304 | 5.89 | 5.87 | 0.45 |

- Val loss fell at every one of the 13 evaluations and tracks train loss closely (no overfitting at 0.8% of an epoch). Grad norm spiked to 15.9 at step 1, then settled below 1; only 3 steps after step 50 exceeded the clip threshold (max 1.27).
- Throughput: median 6,109 tok/s, peak memory 4.74 GB, about 68 min wall time.
- **Unexplained:** between steps 90 and 227, bursts of 3–4 consecutive steps every ~15 steps ran at ~1.5–2k tok/s (4× slower); steps 228–304 were steady again. Not eval (excluded from the timing). Suspects: laptop GPU thermal or power throttling, or another process on the GPU. Clocks and temperature weren't logged, so the benchmark harness (M3) will record them alongside throughput.
- M2 acceptance met: param count printed, overfit-one-batch passes, the 20M-token smoke run's loss decreases, W&B logging works.

## M3: benchmark harness

- `scripts/bench_train.py` times the trainer's own `train_step` on real data: 5 warmup steps (timed, but excluded), then 3 trials of 50 optimizer steps, each step bracketed by CUDA events. It writes median and min/max across trials, MFU, peak memory, per-step times, GPU clock/temperature/power per trial, the resolved config and the full environment to `results/ladder/<gpu>/<config>.json`.
- MFU counts the tied lm_head matmul (the spec's "non-embedding" formula would drop ~29% of the real FLOPs): 798.0 MFLOP per token at seq 1024. Peaks: RTX 2070 Super laptop 28.3 TFLOP/s FP16 tensor (FP32 accumulate, half rate on GeForce Turing) and 7.07 FP32; see decisions.md.
- Harness check, S0 FP32 at micro-batch 2 (SDPA backend left to PyTorch's automatic choice, 16k-token steps):

| trial | tok/s | SM clock median (min–max) | temp max |
|---|---|---|---|
| 0 | 6,364 | 1530 (1365–1785) MHz | 67 °C |
| 1 | 6,222 | 1515 (1335–1770) MHz | 71 °C |
| 2 | 6,254 | 1500 (1335–1695) MHz | 73 °C |

- Median 6,254 tok/s (spread 2.3%), 4.99 TFLOP/s achieved: **17.6% MFU** against the FP16 tensor-core peak, 70.6% of the FP32 peak. Peak memory 4.74 GB allocated (5.18 reserved), identical to the M2 smoke run. Warmup 12.9 s for 5 steps.
- The SM clock runs *above* the 1380 MHz rated boost (~1515 MHz median) and swings ±15% sample to sample, while power sits at ~111 W of the 115 W limit and throughput stays flat. That is power-limited boost, not throttling. The first throttle rule (any sample >10% below the median) flagged two trials for it; the rule now flags only a sustained drop (trial median >10% below the median of trials) or a thermal/hardware slowdown reason.
- No 4× slowdown bursts like the M2 smoke run's: clocks drifted down only ~2% as the GPU warmed from 41 to 73 °C. The M2 bursts remain unexplained; the trainer now logs MFU, and any recurrence will show up in the per-trial clock record.
- The kernel microbenchmark helper (`benchmarks/kernels/`) moves to M5a, where the first kernel to benchmark arrives.

## M4: ladder stages S1–S4 on the RTX 2070 Super

- New switches, each tested for parity against the previous stage: `attn_backend` (math | mem_efficient, pinned with `sdpa_kernel`), `precision` (fp32 | fp16 + GradScaler | bf16), `compile` (default | max-autotune | reduce-overhead). One stage config per rung in `configs/stages/`; a test checks consecutive configs differ by exactly one setting.
- Parity tests (GPU): mem-efficient vs math attention, output and input-gradient, FP32 within 1e-5 and FP16 within 2e-3; FP16 AMP vs FP32 loss over 40 training steps within 2%; compiled (default and reduce-overhead) vs eager over 15 steps within 2%.
- **Pinning the backend mattered.** S0 with math attention runs at 5,468 tok/s and 5.86 GB; the M3 check, with PyTorch choosing the backend, ran at 6,254 tok/s and 4.74 GB. So the automatic choice was already the memory-efficient kernel, even in FP32, and S0 was not the reference the ladder defines.
- **Bug caught by the parity test:** CUDA graphs + gradient accumulation crashed with "accessing tensor output of CUDAGraphs that has been overwritten". Autograd adopted the first micro-batch's gradient (which lives in the graph's memory pool) as `.grad`, and the next replay overwrote it. Fix: with CUDA graphs on, `.grad` is allocated once outside the pool and zeroed in place. `cudagraph_mark_step_begin()` alone did not fix it.
- The training step now sums the loss on the GPU and syncs once per optimizer step instead of once per micro-batch.

Results (`python scripts/ladder_table.py rtx-2070-super`), seq 1024, micro-batch 2, 16,384-token steps, 3 × 50 timed steps after 5 warmup steps:

| Stage | Tokens/s (min–max) | MFU | Peak mem alloc / reserved | Loss @ step | Δ vs previous | Warmup | SM clock (lowest trial median) |
|---|---|---|---|---|---|---|---|
| s0_fp32 | 5,468 (5,453–5,549) | 15.4% | 5.86 / 6.30 GB | 6.772 @ 155 | — | 16 s | 1470 MHz |
| s1_fp16 | 9,563 (9,519–9,579) | 27.0% | 5.55 / 5.89 GB | 6.775 @ 155 | 1.75× | 9 s | 1590 MHz |
| s2_mem_efficient | 16,274 (16,267–16,319) | 45.9% | 4.30 / 4.69 GB | 6.774 @ 155 | 1.70× | 6 s | 1620 MHz |
| s3_compile | 22,994 (22,901–23,794) | 64.8% | 3.25 / 3.67 GB | 6.767 @ 155 | 1.41× | 24 s | 1590 MHz |
| s4_cudagraphs | 22,506 (22,283–22,878) | 63.5% | 1.99 / 4.05 GB | 6.771 @ 155 | 0.98× | 27 s | 1605 MHz |

- **4.2× from S0 to S3** (5,468 → 22,994 tok/s), at an unchanged loss: all five stages reach 6.767–6.775 at step 155 from the same init and data. FP16 never overflowed in 155 steps (the loss scale stayed at its initial 65,536).
- S1 FP16: 1.75×. The matmuls move to tensor cores, but math attention still writes the full 1024 × 1024 score matrix per head; memory barely drops (5.86 → 5.55 GB).
- S2 mem-efficient attention: 1.70× and −1.25 GB. The fused kernel never materializes the scores.
- S3 torch.compile: 1.41× and −1.05 GB. Inductor fuses RMSNorm, RoPE, SwiGLU and the residual adds into Triton kernels. Those ops are memory-bound and were each a separate read/write of the activations, so fusing saves both time and saved-activation memory. Attention stays on the CUTLASS kernel (Inductor reports it as a fallback kernel). Compile adds ~18 s to warmup.
- S4 CUDA graphs: **no gain** (0.98×, within the spread). The GPU was 100% busy and at its power limit (~113 W) in every stage: at 2,048 tokens per micro-batch the kernels are long enough that the CPU launches them faster than the GPU runs them, so there is no launch overhead for graphs to remove. Graphs should matter at small micro-batches or on a faster GPU. Kept as a stage; not used by default. Its "allocated" peak (1.99 GB) is misleading, since the graph pool is reserved during warmup; reserved (4.05 GB) is the fair number, 0.38 GB above S3, close to the ~0.43 GB of FP32 gradients that now stay allocated.
- MFU at S3 is 64.8% of the FP16 tensor-core peak *at the rated 1380 MHz boost*. The GPU actually ran at ~1600 MHz, so against the peak at the clock it really ran, it's ~56%.
- **Laptop throttling, caught by the harness:** the first ladder pass ran S0/S1 held at the 1140 MHz base clock (80–86 W of 115 W, no reason reported), and the second pass had S2 and S3 trials under a SW thermal slowdown (memory clock 6801 → 810 MHz, ~30 W) while the GPU read under 50 °C. The harness now flags a trial below the rated boost clock, and `scripts/run_ladder.sh` re-runs flagged stages after a rest. Every row above is unflagged (lowest trial clock 1470–1620 MHz). Discarded attempts are kept locally in `runs/`.
- T4 rows (Kaggle) not run yet: the instructions are in `docs/kaggle_workflow.md`. nsys traces per stage are also still to do.

## M4: the same ladder on a Kaggle T4

- One T4 of Kaggle's 2× T4 notebook, torch 2.11.0+cu128 (Kaggle's image; the laptop runs NGC's 2.10). Same configs, same data, run with `--data-dir` pointing at the attached dataset. No trial flagged.

| Stage | Tokens/s (min–max) | MFU | Peak mem alloc / reserved | Loss @ step | Δ vs previous | Warmup | SM clock (lowest trial median) |
|---|---|---|---|---|---|---|---|
| s0_fp32 | 2,731 (2,605–2,849) | 3.4% | 5.86 / 6.30 GB | 6.767 @ 155 | — | 29 s | 735 MHz |
| s1_fp16 | 6,294 (6,286–6,295) | 7.7% | 5.55 / 5.89 GB | 6.775 @ 155 | 2.30× | 14 s | 825 MHz |
| s2_mem_efficient | 11,224 (11,217–11,288) | 13.8% | 4.30 / 4.69 GB | 6.772 @ 155 | 1.78× | 9 s | 840 MHz |
| s3_compile | 15,509 (15,459–15,867) | 19.0% | 3.25 / 3.67 GB | 6.772 @ 155 | 1.38× | 37 s | 705 MHz |
| s4_cudagraphs | 15,370 (15,346–15,713) | 18.9% | 1.99 / 4.05 GB | 6.769 @ 155 | 0.99× | 34 s | 720 MHz |

- **Same loss at step 155 as the laptop** (6.767–6.775 on both GPUs, every stage), across two GPUs and two torch versions.
- **The T4 is power-limited to about half its boost clock.** It ran at 705–855 MHz against a rated 1590 MHz boost, drawing 64–66 W of its 70 W limit at 100% utilization, with only the power-cap reason set (no thermal flag at 77–82 °C). The 65 TFLOP/s datasheet peak assumes the boost clock, so MFU against it reads low (19% at S3). Against the peak at the clock it actually ran (~30 TFLOP/s at 735 MHz), S3 reaches ~41%.
- FP16 gains more on the T4 than on the laptop (2.30× vs 1.75×): the T4's FP16 tensor-core peak is 8× its FP32 peak, against 4× on GeForce Turing, which runs FP32-accumulate at half rate.
- CUDA graphs: no gain here either (0.99×). The T4 is slower than the laptop, so the GPU is even further ahead of the CPU's kernel launches.
- T4 vs laptop at S3: 15,509 vs 22,994 tok/s (0.67×), with identical memory use. The T4's higher datasheet peak doesn't help when its power limit holds it at ~735 MHz.
- Kaggle detail: the dataset now mounts at `/kaggle/input/datasets/<owner>/<slug>/`, not `/kaggle/input/<slug>/`; `docs/kaggle_workflow.md` is updated.

## M5a: fused Triton kernels (RMSNorm + residual add, SwiGLU, RoPE)

- Three Triton kernels, each forward + backward, each registered as a `torch.library` custom op so `torch.compile` and CUDA graphs treat it as one op (no graph breaks). Switched on per kernel by `norm_kernel` / `swiglu_kernel` / `rope_kernel` in `ModelConfig`; the `torch` default is the untouched reference path.
- **Fused residual add + RMSNorm.** To fuse the add into the norm, blocks now pass `(x, delta)` instead of `x`: `delta` is the previous sublayer's output, not yet added. Each norm computes `h = x + delta` and returns `(rmsnorm(h), h)`. Under FP16 autocast it writes the normalized output directly in FP16 (the next matmul would cast it anyway), while `h` stays FP32. Backward: one program per chunk of rows, each keeping its partial weight gradient in registers; the partials are summed afterwards.
- **SwiGLU.** Backward recomputes the sigmoid instead of saving it, so `silu(g)` is not kept as an extra activation.
- **RoPE.** One launch rotates all Q and K heads of a token and reads that position's cos/sin once. The backward is the same kernel with `sin` negated (a rotation's transpose is its inverse), so nothing is saved but the tables.
- **Tests (tests/kernels/, 62 + 6 model-level):** forward/backward vs the reference over a shape grid (odd sizes, strided inputs) in FP32 and FP16, FP64 `gradcheck`, `torch.library.opcheck`, `fullgraph=True` compile, and whole-model logits/gradient parity. A real bug the opcheck caught: the RoPE kernel assumed a contiguous last dim and failed on a transposed input. FP16 tolerances are set from observed error: the reference rounds the norm's gradient to FP16 before adding the residual's, the kernel adds in FP32 and rounds once (one element in 1.5M was 2 ulps apart at 0.002).

### Kernel microbenchmarks (RTX 2070 Super laptop, 448 GB/s peak)

GPU time only: each call is captured in a CUDA graph and replayed, L2 flushed before each replay. The first attempt used `triton.testing.do_bench`, which measured Python overhead (RoPE forward+backward read 1.5 ms vs 0.05 ms). Rows are clock-checked; a first unmonitored run slowed every provider ~7× mid-run (the SW thermal slowdown from M4), and the harness now rests and re-runs flagged rows. Achieved bandwidth is the minimum bytes moved over the time. Results: `results/kernels/rtx-2070-super/*.json`.

| Kernel | Setting | Eager | torch.compile | Ours (Triton) | Liger 0.8.5 |
|---|---|---|---|---|---|
| Add + RMSNorm fwd | AMP (FP32 stream, FP16 delta), 8192 rows | 23% | 66% | **86%** | 52% |
| Add + RMSNorm fwd+bwd | same | 15% | 49% | **85%** | 48% |
| Add + RMSNorm fwd | all FP16, 8192 rows | 15% | 69% | **86%** | 85% |
| Add + RMSNorm fwd+bwd | same | 10% | 60% | **82%** | 69% |
| SwiGLU fwd+bwd | FP16, 8192 × 2048 | 50% | 89% | 88% | 87% |
| RoPE fwd+bwd | FP16, B=8, T=1024 | 6% | 82% | 81% | 85% |

(% of the 448 GB/s DRAM peak.)

- RMSNorm is the one kernel with a clear win over `torch.compile` (+20 points of bandwidth): Inductor emits the add and the norm as separate kernels or a less coalesced reduction, ours is one pass. Against Liger the AMP gap is partly a different contract: Liger writes the normalized output in the residual's dtype (FP32), so the model would still need a cast to FP16.
- SwiGLU and RoPE are elementwise and bandwidth-bound; `torch.compile` already generates a fused kernel at the same bandwidth, so there is nothing to win. Ours and Liger's match it.
- **Nsight Compute** (`results/ncu/rtx-2070-super/*.csv`; `.ncu-rep` kept locally): memory-pipeline throughput 84–93% of peak for every kernel, and the DRAM bytes match the minimum-traffic model (RMSNorm fwd: 9.45 MB read, 8.78 MB written, predicted 9.4 MB each). The weak spot is RMSNorm backward: 24% warp occupancy (only 79 programs of 128 threads, 2 per SM) and ~70 µs against ~50 µs forward.
- At the real micro-batch of 2048 tokens, RMSNorm and RoPE are small (30–50 µs): launch gaps and clock behavior matter as much as bandwidth.

### S5 on training steps

| Stage | Tokens/s (min–max) | MFU | Peak mem alloc / reserved | Loss @ step 155 | Δ vs S4 |
|---|---|---|---|---|---|
| s4_cudagraphs | 22,506 (22,283–22,878) | 63.5% | 1.99 / 4.05 GB | 6.771 | — |
| s5_triton | 22,754 (22,333–23,255) | 64.2% | 1.99 / 4.05 GB | 6.769 | 1.01× |

- **No end-to-end gain.** 1.01× is inside the spread of both stages. `torch.compile` (S3) already fuses RMSNorm, SwiGLU, RoPE and the residual adds into Triton kernels, so hand-written ones replace kernels that were already fused. The RMSNorm microbenchmark gain (+20 points of bandwidth) is a fraction of ~50 µs per call, 25 calls per step, in a step dominated by matmuls and attention.
- Same loss as S4 at step 155, and no memory change: the sigmoid recompute and the dropped `silu(g)` saving don't show up in peak memory, which at S4/S5 is set by the CUDA-graph pool.
- The three single-kernel ablation configs (`configs/ablations/`) exist but were not run: with the combined effect inside the noise they could not separate anything.
- Where the custom kernels pay off is outside this setup: eager mode (S1/S2, where RMSNorm + RoPE + SwiGLU are many small kernels), and as the template for S6 (CUDA C++ RMSNorm).

## M5b/M5c: CUDA C++ kernels

**FlashAttention-2 for sm_75** (`nanonemo/kernels/cuda/flash_attn.cu`). Built in three versions, each measured at the model's shape (B=2, 12 Q / 4 KV heads, T=1024, hd 64, FP16) against PyTorch's CUTLASS memory-efficient kernel (S2's), which is 266 µs forward and 1,095 µs forward+backward:

| Version | fwd | fwd+bwd | What changed |
|---|---|---|---|
| 1. One query row per block, scalar FMA | 33,569 µs | 94,939 µs | Correct (tests pass) but a block-wide reduction per score |
| 2. Tiles + tensor cores (WMMA 16×16×16) | 1,514 µs | 3,969 µs | 64-query × 32-key tiles; dK/dV in registers per key block |
| 2b. Output kept in registers, cheaper softmax | 643 µs | 3,064 µs | Rescale the O fragments in registers via a row-index fragment; half a row per lane |
| 3. Shared-memory rows padded by 16 bytes | **293 µs** | **1,825 µs** | ncu: both kernels were at 81–86% L1/shared throughput, 12% SM: bank conflicts |

The forward now ties the CUTLASS kernel; the backward is 1.85× slower than CUTLASS's (1.53 vs 0.83 ms). Correctness: matches SDPA's output and gradients (fp16 tolerance) for MHA/GQA, odd T, hd 32/64; future keys provably don't leak; `opcheck` and `fullgraph` compile pass; whole-model loss identical to SDPA and weight gradients within 0.1%. A compile bug found on the way: the fake (meta) implementation copied `q`'s strides with `empty_like`, but the kernel returns contiguous tensors, so the compiled model failed at the first transposed input; fixed and covered by a test.

**S7 end to end**: 0.90× of S5 (20,495 vs 22,754 tok/s, clean clocks). Replacing CUTLASS's backward with a slower one costs more than nothing gained on the forward. Re-runs later hit the SM clock pinned at 1140 MHz, the mobile 2070 Super's base clock, with no throttle reason reported. `nvidia-smi -rgc` and a reboot did not help; the cause was the ACPI platform profile (`balanced`). Under `powerprofilesctl set performance` the GPU holds 1,650–1,900 MHz at its 115 W limit, and S5–S8 were re-measured back to back: **S7 21,011 tok/s, 0.88× of S6** (23,986).

**CUDA C++ RMSNorm** (`nanonemo/kernels/cuda/rmsnorm.cu`), the same op and autograd as the Triton one: warp per row, 16-byte vector loads, the row in registers, shuffle-only reduction. Forward ties Triton (79–86% of DRAM peak). The backward at first trailed by 20% at 2048 rows: 4 blocks per SM meant 640 per-warp dw partial rows to write and sum; 2 blocks per SM brought it level (within ±5% of Triton across the four cases). **S6**: 23,373 tok/s, 1.03× over S5, inside run-to-run noise, for the same reason as S5: Inductor already fuses this op well.

## M6: fused linear + cross-entropy

`nanonemo/kernels/triton/linear_ce.py`: 512-row chunks; cuBLAS for the logits, a Triton kernel that writes each row's loss and overwrites the chunk with `softmax − onehot`, cuBLAS for dx and dW. The (2048 × 50,304) logits never exist. The O(1) gradient is kept in FP16 and the 1/N and GradScaler scale are applied in FP32 in backward, so small probabilities don't underflow. Loss and gradients match `F.cross_entropy` (FP32: 1e-5 loss, 1e-4 grads; AMP: 1e-3 loss), and the test asserts peak memory under 60% of the unfused path at the model's shape. **S8**: 20,443 tok/s, 0.97× of S7; peak reserved memory 3.68 vs 4.02 GB. At micro-batch 2 the logits are only ~200 MB, so the memory saving is small and the extra chunked matmuls cost a few percent.

S8 first failed in a strange way: `torch.compile` never finished (>15 min), and once the process grew to 30 GB of host RAM and was OOM-killed. A stack dump (`faulthandler.dump_traceback_later`) showed Inductor inside `OperatorIssue.operator_str`, recursively printing IR nodes. For a custom op with no lowering, Inductor takes an implicit-fallback path that formats every argument's IR for an INFO log message, even with logging off. The input to `linear_ce` is the IR of the whole network, and printing it effectively never ends. Registering the op as an explicit fallback (`make_fallback`) skips that path: compile now takes 13 s.

## M7: multi-GPU code

DDP (bucket size, gradient-as-bucket-view, `no_sync` on all but the last micro-batch), FSDP2 (per-block `fully_shard`, FP32 gradient reduce, optional FP16 parameter gathers, explicit prefetch, per-block compile, gradient sync once per step), TP=2 + sequence parallel (column/row-parallel attention and MLP, norms on sequence shards, vocab-sharded tied embedding/LM head with `loss_parallel`). Checkpoints are DCP with async save and a `latest` pointer written only after a save completes; a wall-clock limit stops all ranks on the same step.

Tested without two GPUs: torchrun with 2 CPU ranks (gloo). Every layout matches single-process training to 1e-4 on the same global batch, and a run stopped at step 3 and resumed reproduces the uninterrupted one, including resuming an FSDP checkpoint from 2 ranks in 1 process. That last test caught a real bug: the loader position is per rank, so it is now converted through the global sample count when the world size changes. On the GPU (1 rank): FP16 resume under CUDA graphs, and FSDP/TP with the sharded GradScaler. TP + whole-model `torch.compile` failed in backward (the vocab-sharded logits' gradient came back as a plain tensor); under TP the blocks are compiled individually, as torchtitan does.

### M7 on Kaggle 2× T4

Run headlessly: `kaggle kernels push` with the notebook's cells, then `kaggle kernels output`. All 12 multigpu tests pass on real GPUs with NCCL. Throughput (64k-token steps): DDP 33.6k tok/s, DDP with 100 MB buckets 32.8k (same within noise), FSDP2 27.3k, FSDP2 + prefetch 27.8k, TP=2 + SP 19.2k. On this small model, FSDP's extra parameter all-gathers and TP's per-layer collectives cost more than DDP's single gradient all-reduce, and memory isn't a constraint (4.9 GB per GPU of 15). The 1-GPU row measured 2,300 tok/s, which is impossible (the same config does 28k on the laptop), so it was discarded and is re-run in a later session. The nsys rows weren't produced: Nsight Systems isn't installed on Kaggle's image.

LR sweep, 100M tokens each on DDP: 3e-4 → val loss 5.78, 6e-4 → 5.94, 1.2e-3 → 6.17. The main run uses **3e-4**.

## M9: main run (Kaggle 2× T4, DDP)

2,414,870,528 tokens (one epoch of the train split), 4,606 steps of 0.5M tokens, peak LR 3e-4 (cosine to 3e-5, 2% warmup), FP16 AMP + mem-efficient SDPA + `torch.compile`, DDP over 2 T4s. Two Kaggle sessions launched headlessly: session 1 stopped itself on the 11.5 h limit at step 2,860, wrote a checkpoint, and session 2 resumed from it (uploaded as a private dataset) and finished. **Final val loss 3.397** (train 3.38), median 36.2k tok/s, 22% MFU, 4.9 GB per GPU. The loss curve shows no step at the resume point: model, AdamW moments, LR schedule position, loader position and GradScaler state all came back exactly.

Things that went wrong on the way:
- Kaggle's checkpoint dataset flattened the uploaded `ckpt/` folder (`latest` at the dataset root), so the notebook's `ckpt/latest` glob found nothing. The session-2 notebook also searches for a `latest` next to `step_*/.metadata`, and refuses to start if no checkpoint is found rather than restarting from scratch.
- The run's own checkpoint `.metadata` is a pickle that names `pathlib._local.PosixPath` (Kaggle runs Python 3.13); the local container is 3.12 and couldn't unpickle it. `load_checkpoint` aliases the module.
- One downloaded file (`latest`, 11 bytes) came back as a Kaggle 502 error page; the shard files had their exact sizes.
- W&B through a Kaggle secret didn't work: secrets aren't served to notebooks pushed with the CLI. Both sessions' `metrics.jsonl` were logged to the W&B run afterwards from the laptop.

Session 2 also re-ran the M7 single-GPU row (18.1k tok/s; DDP's 33.6k is 93% scaling efficiency) and the T4 S5–S8 rows: 15.3k, 15.3k, 14.2k, 13.8k tok/s, the same shape as the laptop (S5/S6 ≈ S4, S7 0.93×, S8 0.97×). Kaggle's T4s ran at 650–900 MHz SM (thermal), so these are noisier than the laptop's.

The main run uses the S3 stack, not S5–S8: on both GPUs the custom kernels measured ≤1.06× (S7/S8 slower), and the multi-GPU tests and LR sweep were done on S3.
