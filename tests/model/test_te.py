"""Transformer Engine linears (M8) vs nn.Linear: same function with the same weights."""

import pytest
import torch

te = pytest.importorskip("transformer_engine.pytorch")
from nanonemo.model.config import ModelConfig  # noqa: E402
from nanonemo.model.model import NanoNemo  # noqa: E402

pytestmark = pytest.mark.gpu
SMALL = dict(n_layers=2, d_model=256, n_heads=4, n_kv_heads=2, d_ff=512, vocab_size=512, max_seq_len=128)


def _pair():
    torch.manual_seed(0)
    ref = NanoNemo(ModelConfig(**SMALL)).cuda()
    fast = NanoNemo(ModelConfig(**SMALL, linear_impl="te")).cuda()
    missing, unexpected = fast.load_state_dict(ref.state_dict(), strict=False)
    assert all(k.endswith("_extra_state") for k in missing) and not unexpected  # TE's FP8 metadata only
    idx = torch.randint(0, SMALL["vocab_size"], (2, 128), device="cuda")
    return ref, fast, idx, torch.roll(idx, -1, dims=1)


def test_te_linears_match_nn_linear():
    ref, fast, idx, tgt = _pair()
    ref_logits, ref_loss = ref(idx, tgt)
    logits, loss = fast(idx, tgt)
    torch.testing.assert_close(logits, ref_logits, rtol=1e-4, atol=1e-4)
    ref_loss.backward()
    loss.backward()
    ref_grads = dict(ref.named_parameters())
    for name, p in fast.named_parameters():
        torch.testing.assert_close(p.grad, ref_grads[name].grad, rtol=1e-3, atol=1e-5, msg=name)


def test_te_linears_get_the_same_init():
    torch.manual_seed(0)
    model = NanoNemo(ModelConfig(**SMALL, linear_impl="te"))
    w = model.blocks[0].mlp.w_up.weight
    assert abs(w.std().item() - 0.02) < 0.002


def _fp8_ok() -> bool:
    ok = te.is_fp8_available()
    return ok[0] if isinstance(ok, tuple) else ok


@pytest.mark.skipif(not _fp8_ok(), reason="FP8 needs sm_89+")
@pytest.mark.parametrize("recipe", ["delayed", "current"])
def test_fp8_forward_backward_tracks_bf16(recipe):
    from nanonemo.te import fp8_context

    _, fast, idx, tgt = _pair()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        _, ref_loss = fast(idx, tgt)
        with fp8_context(recipe)():
            _, loss = fast(idx, tgt)
    loss.backward()
    torch.testing.assert_close(loss, ref_loss, rtol=2e-2, atol=2e-2)
