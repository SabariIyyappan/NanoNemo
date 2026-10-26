"""The Hugging Face Llama export computes the same function as NanoNemo."""

import pytest
import torch

pytest.importorskip("transformers")
from transformers import LlamaForCausalLM  # noqa: E402

from nanonemo.eval.hf_export import export, load_checkpoint  # noqa: E402
from nanonemo.model.config import ModelConfig  # noqa: E402
from nanonemo.model.model import NanoNemo  # noqa: E402

SMALL = ModelConfig(n_layers=2, d_model=128, n_heads=4, n_kv_heads=2, d_ff=256, vocab_size=512, max_seq_len=64)


def _model():
    torch.manual_seed(0)
    model = NanoNemo(SMALL).eval()
    with torch.no_grad():  # norms start at 1; perturb them so a swapped norm would show
        for name, p in model.named_parameters():
            if "norm" in name:
                p.add_(0.1 * torch.randn_like(p))
    return model


def test_exported_model_has_the_same_logits(tmp_path):
    model = _model()
    hf = LlamaForCausalLM.from_pretrained(export(model, tmp_path, tokenizer=False)).eval()
    idx = torch.randint(0, SMALL.vocab_size, (2, 64))
    with torch.no_grad():
        ours, _ = model(idx)
        theirs = hf(idx).logits
    torch.testing.assert_close(theirs, ours, rtol=1e-4, atol=1e-4)
    assert hf.lm_head.weight.data_ptr() == hf.model.embed_tokens.weight.data_ptr()  # still tied


def test_load_from_training_checkpoint(tmp_path):
    from nanonemo.train.checkpoint import Checkpointer
    from nanonemo.train.optim import build_optimizer

    model = _model()
    opt = build_optimizer(model, 1e-3)
    Checkpointer(tmp_path, main=True, use_async=False).save(7, model, opt, {"step": 7})
    loaded = load_checkpoint(tmp_path / "step_000007", SMALL)
    for (n, a), (_, b) in zip(model.state_dict().items(), loaded.state_dict().items()):
        torch.testing.assert_close(a, b, rtol=0, atol=0, msg=n)
