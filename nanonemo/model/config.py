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
    # "torch" (reference) or "triton" (nanonemo/kernels/triton): fused residual add + RMSNorm,
    # SwiGLU activation, and RoPE on Q and K. The norm also has a "cuda" (C++) version.
    norm_kernel: str = "torch"
    swiglu_kernel: str = "torch"
    rope_kernel: str = "torch"
    # "torch" (SDPA, attn_backend above) or "cuda" (nanonemo/kernels/cuda/flash_attn.cu):
    # a hand-written FlashAttention-2 style kernel that reads K/V from their own GQA
    # head directly, so it ignores attn_backend.
    attn_kernel: str = "torch"
    # "torch" (lm_head, then F.cross_entropy on the full logits) or "fused"
    # (nanonemo/kernels/triton/linear_ce.py: chunked, never builds the (B*T, vocab) logits).
    # With "fused", forward(idx, targets) returns logits=None.
    loss_kernel: str = "torch"
    # "torch" (nn.Linear) or "te" (Transformer Engine te.Linear for the blocks' linear layers;
    # needed for FP8 training on an L4; see nanonemo/te).
    linear_impl: str = "torch"

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, "d_model must divide evenly into heads"
        assert self.n_heads % self.n_kv_heads == 0, "query heads must group evenly over KV heads"
        assert self.head_dim % 2 == 0, "RoPE rotates pairs of dimensions"
        assert self.attn_backend in ("math", "mem_efficient"), f"unknown attn_backend {self.attn_backend!r}"
        assert self.norm_kernel in ("torch", "triton", "cuda"), f"unknown norm_kernel {self.norm_kernel!r}"
        for name in ("swiglu_kernel", "rope_kernel"):
            assert getattr(self, name) in ("torch", "triton"), f"unknown {name} {getattr(self, name)!r}"
        assert self.attn_kernel in ("torch", "cuda"), f"unknown attn_kernel {self.attn_kernel!r}"
        assert self.loss_kernel in ("torch", "fused"), f"unknown loss_kernel {self.loss_kernel!r}"
        assert self.linear_impl in ("torch", "te"), f"unknown linear_impl {self.linear_impl!r}"

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads
