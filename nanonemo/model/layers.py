"""Reference (plain PyTorch) building blocks. Optimized kernels are checked against these."""

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from nanonemo.model.config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reduce in FP32: in FP16, x**2 overflows to inf once |x| > ~256.
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (xf * self.weight.float()).to(x.dtype)


class AddRMSNorm(RMSNorm):
    """Adds a pending sublayer output into the residual stream, then normalizes it.

    forward(x, delta) returns (rmsnorm(x + delta), x + delta); delta=None normalizes x alone.
    Taking the add here lets the "triton" kernel do both in one pass over memory.
    """

    def __init__(self, dim: int, eps: float = 1e-5, kernel: str = "torch"):
        super().__init__(dim, eps)
        self.fused = None
        if kernel == "triton":
            from nanonemo.kernels.triton.rmsnorm import add_rms_norm

            self.fused = add_rms_norm

    def forward(self, x: torch.Tensor, delta: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if self.fused is not None:
            return self.fused(x, delta, self.weight, self.eps)
        if delta is not None:
            x = x + delta
        return super().forward(x), x


def rope_cache(head_dim: int, max_seq_len: int, base: float = 10000.0, device=None):
    """cos/sin tables of shape (max_seq_len, head_dim // 2), in FP32."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    pos = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    angles = torch.outer(pos, inv_freq)  # angles[m, i] = m * theta_i
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate x of shape (B, H, T, head_dim). Pairs dimension i with i + head_dim/2 (rotate-half)."""
    T, half = x.shape[-2], x.shape[-1] // 2
    cos, sin = cos[:T], sin[:T]
    xf = x.float()
    x1, x2 = xf[..., :half], xf[..., half:]
    out = torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return out.to(x.dtype)


class Attention(nn.Module):
    """Causal grouped-query attention: n_heads query heads share n_kv_heads K/V heads."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads, self.n_kv_heads, self.head_dim = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        self.backend = {"math": SDPBackend.MATH, "mem_efficient": SDPBackend.EFFICIENT_ATTENTION}[cfg.attn_backend]
        self.wq = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)
        self.wk = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.wv = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model, bias=False)
        self.rope_qk = None
        if cfg.rope_kernel == "triton":
            from nanonemo.kernels.triton.rope import apply_rope_qk

            self.rope_qk = apply_rope_qk

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)  # (B, KVH, T, hd)
        if self.rope_qk is not None:
            q, k = self.rope_qk(q, k, cos, sin)  # rotates in (B, T, H, hd) layout, Q and K in one launch
            q, k = q.transpose(1, 2), k.transpose(1, 2)
        else:
            q, k = apply_rope(q.transpose(1, 2), cos, sin), apply_rope(k.transpose(1, 2), cos, sin)  # (B, H, T, hd)

        # Query head h uses K/V head h // group: [kv0, kv0, kv0, kv1, ...].
        group = self.n_heads // self.n_kv_heads
        k = k.repeat_interleave(group, dim=1)
        v = v.repeat_interleave(group, dim=1)

        with sdpa_kernel(self.backend):
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # (B, H, T, hd)
        return self.wo(y.transpose(1, 2).reshape(B, T, -1))


def silu_mul(g: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    return F.silu(g) * u


class MLP(nn.Module):
    """SwiGLU feed-forward: down(silu(gate(x)) * up(x))."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.w_gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w_up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w_down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)
        self.act = silu_mul
        if cfg.swiglu_kernel == "triton":
            from nanonemo.kernels.triton.swiglu import swiglu

            self.act = swiglu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(self.act(self.w_gate(x), self.w_up(x)))


class Block(nn.Module):
    """Pre-norm transformer block: each sublayer reads a normalized copy and adds back to the residual stream.

    The residual stream travels as (x, delta): delta is the previous sublayer's output, not
    yet added to x. Each norm does that add first, so the add and the norm can be one kernel.
    The block's output is x + delta; the next block's (or the final) norm adds it.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm_attn = AddRMSNorm(cfg.d_model, cfg.norm_eps, cfg.norm_kernel)
        self.attn = Attention(cfg)
        self.norm_mlp = AddRMSNorm(cfg.d_model, cfg.norm_eps, cfg.norm_kernel)
        self.mlp = MLP(cfg)

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, delta: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h, x = self.norm_attn(x, delta)
        h, x = self.norm_mlp(x, self.attn(h, cos, sin))
        return x, self.mlp(h)
