"""Model FLOPs per token and MFU (model FLOPs utilization) against a per-GPU peak.

FLOPs per token = 6 * N_matmul + 12 * n_layers * d_model * seq_len   (PaLM, appendix B)
  - 6 * N: each matmul weight does 2 FLOPs per token forward and 4 backward.
  - N_matmul counts every weight a token is multiplied by. The embedding *lookup*
    is free, but our embedding is tied to lm_head, and the lm_head matmul is real
    work (~1/3 of the total), so the tied weight is counted once, as lm_head.
  - 12 * L * d * T: the attention score (QK^T) and value (PV) matmuls, which have
    no weights. Counted over the full T x T square, as PaLM does, even though
    causal attention needs only half of it.
"""

from nanonemo.model.config import ModelConfig


def matmul_params(cfg: ModelConfig) -> int:
    """Parameters that take part in a matmul per token (norm gains excluded)."""
    d, kv = cfg.d_model, cfg.n_kv_heads * cfg.head_dim
    attn = d * d + 2 * d * kv + d * d  # wq, wk, wv, wo
    mlp = 3 * d * cfg.d_ff  # w_gate, w_up, w_down
    return cfg.n_layers * (attn + mlp) + cfg.vocab_size * d  # + lm_head


def flops_per_token(cfg: ModelConfig, seq_len: int) -> int:
    """Training (forward + backward) FLOPs per token."""
    return 6 * matmul_params(cfg) + 12 * cfg.n_layers * cfg.d_model * seq_len


# Peak dense throughput in TFLOP/s at the rated boost clock.
#   fp16_tc: FP16 tensor cores with FP32 accumulate, which is what PyTorch's FP16
#            matmuls use. GeForce Turing runs FP32 accumulate at *half* its FP16-accumulate
#            rate; Tesla/datacenter parts (T4, L4) do not.
#   fp32:    plain FP32 on CUDA cores (Turing has no TF32), the ceiling for the FP32 S0 stage.
# Turing: 1024 FP16 tensor FLOPs/clock/SM with FP16 accumulate (Turing architecture
# whitepaper, appendix A: RTX 2070 = 36 SMs x 1024 x 1.62 GHz = 59.7 TFLOPS, 29.9 with FP32 acc).
PEAK_TFLOPS = {
    # RTX 2070 Super *laptop* (non-Max-Q): 40 SMs, 2560 cores, 1380 MHz rated boost
    # (NVIDIA 20-series SUPER mobile launch specs, 2020-04).
    #   fp16_tc = 40 * 1024 * 1.38e9 / 2 = 28.3;  fp32 = 2560 * 2 * 1.38e9 = 7.07.
    "rtx 2070 super": {"fp16_tc": 28.3, "fp32": 7.07},
    # T4: NVIDIA T4 datasheet, "mixed precision (FP16/FP32) 65 TFLOPS", "single precision 8.1".
    "t4": {"fp16_tc": 65.0, "fp32": 8.1},
    # L4: NVIDIA L4 datasheet lists FP16/BF16 tensor 242 TFLOPS *with sparsity*; dense is half.
    "l4": {"fp16_tc": 121.0, "bf16_tc": 121.0, "fp32": 30.3},
}


def peak_tflops(device_name: str) -> dict | None:
    """Peak table entry for a torch/nvidia-smi device name, or None if the GPU isn't listed."""
    name = device_name.lower().replace("nvidia", "").replace("geforce", "").replace("tesla", "")
    name = " ".join(name.replace("laptop gpu", "").split())
    # Longest key first, so "rtx 2070 super" is never shadowed by a shorter key.
    for key in sorted(PEAK_TFLOPS, key=len, reverse=True):
        if name == key or name.startswith(key + " "):
            return PEAK_TFLOPS[key]
    return None


def mfu(tokens_per_s: float, flops_per_tok: int, peak_tflops_value: float) -> float:
    return tokens_per_s * flops_per_tok / (peak_tflops_value * 1e12)
