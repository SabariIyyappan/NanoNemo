"""Model hyperparameters (spec §4)."""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    n_layers: int = 12
    d_model: int = 768
    n_heads: int = 12
    n_kv_heads: int = 4  # GQA: 3 query heads share each KV head
    d_ff: int = 2048  # SwiGLU intermediate size, ~8/3 * d_model rounded to a multiple of 64
    vocab_size: int = 50304  # GPT-2's 50,257 padded to a multiple of 64
    max_seq_len: int = 1024
    norm_eps: float = 1e-5
    rope_base: float = 10000.0
    init_std: float = 0.02
    # SDPA backend, pinned so a stage measures the kernel it claims to: "math" (the S0
    # reference: plain matmul + softmax) or "mem_efficient" (CUTLASS fused kernel, S2).
    # Pinning also stops a silent fallback: an unsupported input raises instead.
    attn_backend: str = "math"

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, "d_model must divide evenly into heads"
        assert self.n_heads % self.n_kv_heads == 0, "query heads must group evenly over KV heads"
        assert self.head_dim % 2 == 0, "RoPE rotates pairs of dimensions"
        assert self.attn_backend in ("math", "mem_efficient"), f"unknown attn_backend {self.attn_backend!r}"

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads
