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

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, T, hd)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)  # (B, KVH, T, hd)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        # Query head h uses K/V head h // group: [kv0, kv0, kv0, kv1, ...].
        group = self.n_heads // self.n_kv_heads
        k = k.repeat_interleave(group, dim=1)
        v = v.repeat_interleave(group, dim=1)

        with sdpa_kernel(self.backend):
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # (B, H, T, hd)
        return self.wo(y.transpose(1, 2).reshape(B, T, -1))


class MLP(nn.Module):
    """SwiGLU feed-forward: down(silu(gate(x)) * up(x))."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.w_gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w_up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w_down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class Block(nn.Module):
    """Pre-norm transformer block: each sublayer reads a normalized copy and adds back to the residual stream."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm_attn = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.norm_mlp = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm_attn(x), cos, sin)
        x = x + self.mlp(self.norm_mlp(x))
        return x
