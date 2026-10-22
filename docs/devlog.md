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
