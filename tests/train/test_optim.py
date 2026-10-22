import pytest

from nanonemo.model.config import ModelConfig
from nanonemo.model.model import NanoNemo
from nanonemo.train.optim import build_optimizer, lr_at, param_groups


def test_param_groups_split_by_role():
    model = NanoNemo(ModelConfig())
    decay, no_decay = param_groups(model, 0.1)
    decay_ids = {id(p) for p in decay["params"]}

    assert id(model.tok_emb.weight) not in decay_ids  # embedding (= lm_head) not decayed
    assert id(model.norm_out.weight) not in decay_ids
    assert id(model.blocks[0].norm_attn.weight) not in decay_ids
    assert id(model.blocks[0].attn.wq.weight) in decay_ids
    assert id(model.blocks[0].mlp.w_down.weight) in decay_ids

    # Every parameter appears exactly once across the two groups.
    n_grouped = sum(p.numel() for g in (decay, no_decay) for p in g["params"])
    assert n_grouped == model.num_params()
    # Decayed = all block matmuls = non-embedding params minus the norm gains.
    n_norm = (2 * model.cfg.n_layers + 1) * model.cfg.d_model
    assert sum(p.numel() for p in decay["params"]) == model.num_params(non_embedding=True) - n_norm


def test_optimizer_hyperparameters_match_spec():
    opt = build_optimizer(NanoNemo(ModelConfig(n_layers=1)), lr=6e-4)
    assert [g["weight_decay"] for g in opt.param_groups] == [0.1, 0.0]
    assert opt.defaults["betas"] == (0.9, 0.95)
    assert opt.defaults["eps"] == 1e-8


def test_lr_schedule_shape():
    peak, total, warmup = 6e-4, 1000, 20
    lrs = [lr_at(s, peak, total, warmup) for s in range(total)]
    assert lrs[0] == pytest.approx(peak / warmup)  # first step is not wasted at lr=0
    assert lrs[warmup - 1] == pytest.approx(peak)
    assert lrs[warmup] == pytest.approx(peak)
    assert lrs[-1] == pytest.approx(0.1 * peak)
    assert all(a < b for a, b in zip(lrs[:warmup], lrs[1:warmup]))  # warmup strictly rises
    assert all(a >= b for a, b in zip(lrs[warmup:], lrs[warmup + 1 :]))  # decay never rises
    # Cosine midpoint: halfway through decay the LR is halfway between peak and floor.
    mid = warmup + (total - 1 - warmup) // 2
    assert lrs[mid] == pytest.approx(0.55 * peak, rel=1e-2)


def test_lr_holds_floor_past_the_end():
    assert lr_at(5000, 6e-4, 1000, 20) == pytest.approx(6e-5)
