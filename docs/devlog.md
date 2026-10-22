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
