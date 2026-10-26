# NanoNemo

A 114M-parameter Llama-style language model pretrained from scratch on ~2.4B tokens of NVIDIA Nemotron-CC, and the training made progressively faster on Turing GPUs (an RTX 2070 Super laptop, Tesla T4s) — every step measured, and every custom kernel checked against a plain-PyTorch reference.

## What's here

- **Model** (`nanonemo/model/`): decoder-only, pre-norm RMSNorm, RoPE, SwiGLU, grouped-query attention (12 query / 4 KV heads), tied embeddings. 12 layers, d_model 768, seq 1024, GPT-2 tokenizer (vocab padded to 50,304). The PyTorch reference path is kept and every optimized path is tested against it.
- **Optimization ladder** (`configs/stages/`): S0 eager FP32 → S1 FP16 AMP → S2 memory-efficient SDPA → S3 `torch.compile` → S4 CUDA graphs → S5 Triton kernels → S6 CUDA C++ RMSNorm → S7 CUDA C++ FlashAttention → S8 fused linear + cross-entropy. One change per stage; each stage is a YAML file differing from the previous one in one marked line.
- **Custom kernels** (`nanonemo/kernels/`), all `torch.library` custom ops (they compile with `fullgraph=True` and run under CUDA graphs):
  - Triton: fused residual-add + RMSNorm, SwiGLU, RoPE on Q and K in one launch, and the cross-entropy half of the fused LM-head loss.
  - CUDA C++: fused residual-add + RMSNorm (warp per row, 16-byte loads, shuffle reduction), and a FlashAttention-2 forward + backward for sm_75 on tensor cores (WMMA), causal, reading GQA's KV heads directly.
- **Multi-GPU** (`nanonemo/distributed/`): DDP with bucket tuning, FSDP2 (`fully_shard` per block, explicit prefetch, per-block compile), tensor parallel + sequence parallel with DTensor. DCP checkpoints (async, resumable on a different layout), a wall-clock session limit for Kaggle's 12-hour sessions.
- **Measurement**: CUDA-event step timing with GPU clock monitoring that flags throttled runs (`nanonemo/train/bench.py`), kernel microbenchmarks with CUDA-graph replay and L2 flushes (`benchmarks/kernels/`), Nsight Compute and Nsight Systems scripts, MFU that counts the tied LM head.
- **Evaluation**: checkpoints export to Hugging Face Llama format (logits verified identical) and run through lm-evaluation-harness with exactly the same settings as GPT-2 124M.

## Results

All tables below are generated from `results/` by `scripts/make_figures.py`; the full set (including every kernel row) is in [docs/results.md](docs/results.md). TBD marks runs not done yet.

<!-- results:start -->
### Optimization ladder: RTX 2070 Super (laptop)

| Stage | Tokens/s | MFU | Peak mem alloc / reserved | Loss (last step) | Δ vs previous |
|---|---|---|---|---|---|
| S0 eager FP32 | 5,468 | 15.4% | 5.86 / 6.30 GB | 6.772 | — |
| S1 FP16 AMP | 9,563 | 27.0% | 5.55 / 5.89 GB | 6.775 | 1.75× |
| S2 + mem-efficient SDPA | 16,274 | 45.9% | 4.30 / 4.69 GB | 6.774 | 1.70× |
| S3 + torch.compile | 22,994 | 64.8% | 3.25 / 3.67 GB | 6.767 | 1.41× |
| S4 + CUDA graphs | 22,506 | 63.5% | 1.99 / 4.05 GB | 6.771 | 0.98× |
| S5 + Triton kernels | 22,754 | 64.2% | 1.99 / 4.05 GB | 6.769 | 1.01× |
| S6 + CUDA C++ RMSNorm | 23,373 | 65.9% | 1.99 / 4.05 GB | 6.773 | 1.03× |
| S7 + CUDA C++ FlashAttention | 15,589 ⚠ clock-limited | 44.0% | 1.99 / 4.02 GB | 6.777 | 0.67× |
| S8 + fused linear-CE | TBD | TBD | TBD | TBD | TBD |

### Optimization ladder: Tesla T4

| Stage | Tokens/s | MFU | Peak mem alloc / reserved | Loss (last step) | Δ vs previous |
|---|---|---|---|---|---|
| S0 eager FP32 | 2,731 | 3.4% | 5.86 / 6.30 GB | 6.767 | — |
| S1 FP16 AMP | 6,294 | 7.7% | 5.55 / 5.89 GB | 6.775 | 2.30× |
| S2 + mem-efficient SDPA | 11,224 | 13.8% | 4.30 / 4.69 GB | 6.772 | 1.78× |
| S3 + torch.compile | 15,509 | 19.0% | 3.25 / 3.67 GB | 6.772 | 1.38× |
| S4 + CUDA graphs | 15,370 | 18.9% | 1.99 / 4.05 GB | 6.769 | 0.99× |
| S5 + Triton kernels | TBD | TBD | TBD | TBD | TBD |
| S6 + CUDA C++ RMSNorm | TBD | TBD | TBD | TBD | TBD |
| S7 + CUDA C++ FlashAttention | TBD | TBD | TBD | TBD | TBD |
| S8 + fused linear-CE | TBD | TBD | TBD | TBD | TBD |

![Ladder tokens/s](figures/ladder_tokens_per_s.png)

### Multi-GPU (Kaggle 2x T4)

TBD

### Precision (AWS L4)

| Precision | Tokens/s | MFU (vs dense FP16 TC peak) | Peak mem | Val loss after 20M tokens |
|---|---|---|---|---|
| FP16 AMP | TBD | TBD | TBD | TBD |
| BF16 AMP | TBD | TBD | TBD | TBD |
| BF16 + Transformer Engine linears | TBD | TBD | TBD | TBD |
| FP8 (TE, delayed scaling) | TBD | TBD | TBD | TBD |
| FP8 (TE, current scaling) | TBD | TBD | TBD | TBD |

### Model quality (zero-shot, lm-evaluation-harness 0.4.9, identical settings)

| Model | Tokens trained | HellaSwag (acc_norm) | ARC-Easy (acc_norm) | PIQA (acc_norm) | LAMBADA (acc) | Val loss |
|---|---|---|---|---|---|---|
| GPT-2 124M (HF `gpt2`, evaluated by us) | ~10B+ (published) | 31.1% ± 0.5% | 39.5% ± 1.0% | 62.5% ± 1.1% | 32.6% ± 0.7% | n/a |
| NanoNemo-114M (ours) | ~2.4B | TBD | TBD | TBD | TBD | TBD |

### Main run

TBD
<!-- results:end -->

Design decisions, with the measurements behind them, are in [docs/decisions.md](docs/decisions.md); the chronological record, including what didn't help, is in [docs/devlog.md](docs/devlog.md).

## Reproduce

Everything runs in an NGC PyTorch container (`docker/Dockerfile`, pinned to `25.11-py3`, PyTorch 2.10 / CUDA 13.0 / Triton 3.5).

```bash
scripts/run_container.sh python -m pytest -q                  # unit, kernel, parity, distributed (CPU gloo) tests
scripts/run_container.sh python -m nanonemo.data.prepare --source nemotron_cc --cc-files 12 \
    --out data/nemotron_cc --max-train-tokens 2500000000 --val-frac 0.004 --shard-tokens 100000000
scripts/run_ladder.sh                                         # every ladder stage, re-running throttled ones
scripts/run_container.sh python benchmarks/kernels/rmsnorm.py # kernel microbenchmarks (also swiglu, rope, flash_attn)
scripts/run_container.sh benchmarks/kernels/ncu.sh            # Nsight Compute reports for our kernels
scripts/run_container.sh python scripts/eval.py --model gpt2 --name gpt2-124m
scripts/run_container.sh python scripts/make_figures.py       # tables + figures from results/
```

Multi-GPU runs and the main run are on Kaggle 2× T4 with `notebooks/kaggle_train.ipynb`; the session-to-session checkpoint hand-off is in [docs/kaggle_workflow.md](docs/kaggle_workflow.md).

## Limitations

- Turing has no BF16 or FP8 tensor cores, so all training here is FP16 with loss scaling.
- The laptop GPU's clocks depend on the platform's thermal and power management; benchmark rows record clocks and are flagged when they drop, and flagged rows are not compared.
- The CUDA FlashAttention uses the WMMA API rather than hand-written `mma.sync` PTX; its forward matches PyTorch's CUTLASS memory-efficient kernel on the 2070 Super, its backward is slower (see the devlog).
- One data epoch of ~2.4B tokens is far less than GPT-2's training budget, so the quality comparison is about what that budget buys, not a claim of parity.
