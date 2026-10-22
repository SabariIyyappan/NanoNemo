import math

import torch

from nanonemo.model.config import ModelConfig
from nanonemo.model.model import NanoNemo

SMALL = ModelConfig(n_layers=2, d_model=96, n_heads=6, n_kv_heads=2, d_ff=256, vocab_size=128, max_seq_len=32)


def test_param_count_is_exact():
    cfg = ModelConfig()
    d, kv = cfg.d_model, cfg.n_kv_heads * cfg.head_dim
    per_block = (d * d + 2 * d * kv + d * d) + 3 * d * cfg.d_ff + 2 * d  # attn + mlp + 2 norms
    expected = cfg.vocab_size * d + cfg.n_layers * per_block + d  # tied embedding + blocks + final norm
    model = NanoNemo(cfg)
    assert model.num_params() == expected == 114_150_144
    assert model.num_params(non_embedding=True) == expected - cfg.vocab_size * d


def test_embedding_and_lm_head_share_storage():
    model = NanoNemo(SMALL)
    assert model.lm_head.weight is model.tok_emb.weight


def test_rope_tables_are_not_saved_in_state_dict():
    assert not any("rope" in k for k in NanoNemo(SMALL).state_dict())


def test_residual_projections_use_scaled_init():
    torch.manual_seed(0)
    cfg = ModelConfig()
    model = NanoNemo(cfg)
    resid_std = 0.02 / math.sqrt(2 * cfg.n_layers)
    assert abs(model.blocks[0].attn.wo.weight.std().item() - resid_std) < 2e-4
    assert abs(model.blocks[0].mlp.w_down.weight.std().item() - resid_std) < 2e-4
    assert abs(model.blocks[0].attn.wq.weight.std().item() - 0.02) < 2e-4


def test_initial_loss_is_near_uniform():
    torch.manual_seed(0)
    cfg = ModelConfig()
    model = NanoNemo(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 64))
    logits, loss = model(idx, targets=torch.roll(idx, -1, dims=1))
    assert logits.shape == (2, 64, cfg.vocab_size)
    # norm_out gives x unit RMS, so each logit = x . (tied embedding row, std 0.02) has
    # std sigma = 0.02 * sqrt(d_model) ~ 0.55. Random logits with spread sigma cost about
    # sigma^2 / 2 nats above the uniform ln(V): 10.83 + 0.15 = 10.98.
    sigma = cfg.init_std * math.sqrt(cfg.d_model)
    assert abs(loss.item() - (math.log(cfg.vocab_size) + sigma**2 / 2)) < 0.05


def test_model_is_causal_end_to_end():
    torch.manual_seed(0)
    model = NanoNemo(SMALL)
    idx = torch.randint(0, SMALL.vocab_size, (2, 20))
    idx2 = idx.clone()
    idx2[:, 12:] = torch.randint(0, SMALL.vocab_size, (2, 8))
    torch.testing.assert_close(model(idx2)[0][:, :12], model(idx)[0][:, :12])
