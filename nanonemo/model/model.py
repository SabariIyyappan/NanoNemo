"""NanoNemo: Llama-style decoder-only transformer (reference implementation)."""

import math

import torch
import torch.nn.functional as F
from torch import nn

from nanonemo.model.config import ModelConfig
from nanonemo.model.layers import AddRMSNorm, Block, rope_cache


class NanoNemo(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm_out = AddRMSNorm(cfg.d_model, cfg.norm_eps, cfg.norm_kernel)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # tied input/output embedding

        # Not saved in checkpoints: rebuilt from the config, and moved with .to(device).
        cos, sin = rope_cache(cfg.head_dim, cfg.max_seq_len, cfg.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # GPT-2: shrink the projections that write into the residual stream, so the
        # stream's variance doesn't grow with the number of branches (2 per block).
        resid_std = cfg.init_std / math.sqrt(2 * cfg.n_layers)
        for block in self.blocks:
            nn.init.normal_(block.attn.wo.weight, std=resid_std)
            nn.init.normal_(block.mlp.w_down.weight, std=resid_std)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=self.cfg.init_std)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """idx, targets: (B, T) token ids. Returns (logits, loss); loss is None without targets."""
        T = idx.shape[1]
        assert T <= self.cfg.max_seq_len, f"sequence length {T} > max_seq_len {self.cfg.max_seq_len}"
        x, delta = self.tok_emb(idx), None
        for block in self.blocks:
            x, delta = block(x, self.rope_cos, self.rope_sin, delta)
        logits = self.lm_head(self.norm_out(x, delta)[0])
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def num_params(self, non_embedding: bool = False) -> int:
        # parameters() yields the tied embedding/lm_head weight once.
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()
        return n


if __name__ == "__main__":
    model = NanoNemo(ModelConfig())
    print(f"total params:         {model.num_params():,}")
    print(f"non-embedding params: {model.num_params(non_embedding=True):,}")
