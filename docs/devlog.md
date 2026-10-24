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
