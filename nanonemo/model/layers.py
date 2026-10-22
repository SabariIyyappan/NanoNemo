"""Reference (plain PyTorch) building blocks. Optimized kernels are checked against these."""

import torch
from torch import nn


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
