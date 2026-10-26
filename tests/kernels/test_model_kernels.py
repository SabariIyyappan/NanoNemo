"""The whole model with our kernels vs the reference model, on identical weights and inputs."""

import pytest
import torch

pytest.importorskip("triton")
from nanonemo.model.config import ModelConfig  # noqa: E402
from nanonemo.model.model import NanoNemo  # noqa: E402

pytestmark = pytest.mark.gpu

SMALL = dict(n_layers=2, d_model=256, n_heads=4, n_kv_heads=2, d_ff=512, vocab_size=512, max_seq_len=128,
             attn_backend="mem_efficient")
TRITON = dict(norm_kernel="triton", swiglu_kernel="triton", rope_kernel="triton")
# Every custom kernel at once (S8). The CUDA attention is FP16-only, so these run under autocast.
ALL = dict(TRITON, norm_kernel="cuda", attn_kernel="cuda", loss_kernel="fused")


def _pair(**kernels):
    torch.manual_seed(0)
    ref = NanoNemo(ModelConfig(**SMALL)).cuda()
    fast = NanoNemo(ModelConfig(**SMALL, **kernels)).cuda()
    fast.load_state_dict(ref.state_dict())
    idx = torch.randint(0, SMALL["vocab_size"], (2, 128), device="cuda")
    return ref, fast, idx, torch.roll(idx, -1, dims=1)


def _grads(model):
    return {n: p.grad for n, p in model.named_parameters()}


@pytest.mark.parametrize("kernels", [TRITON, *({k: "triton"} for k in TRITON)], ids=["all", *TRITON])
def test_fp32_matches_reference(kernels):
    ref, fast, idx, tgt = _pair(**kernels)
    ref_logits, ref_loss = ref(idx, tgt)
    logits, loss = fast(idx, tgt)
    torch.testing.assert_close(logits, ref_logits, rtol=1e-4, atol=1e-4)
    ref_loss.backward()
    loss.backward()
    for name, g in _grads(fast).items():
        torch.testing.assert_close(g, _grads(ref)[name], rtol=1e-4, atol=1e-5, msg=name)


def test_fp16_autocast_matches_reference():
    ref, fast, idx, tgt = _pair(**TRITON)
    with torch.autocast("cuda", dtype=torch.float16):
        ref_logits, ref_loss = ref(idx, tgt)
        logits, loss = fast(idx, tgt)
    assert logits.dtype == torch.float16
    torch.testing.assert_close(logits, ref_logits, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(loss, ref_loss, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("kernels", [ALL, dict(norm_kernel="cuda"), dict(attn_kernel="cuda"), dict(loss_kernel="fused")],
                         ids=["all", "cuda_norm", "cuda_attn", "fused_ce"])
def test_fp16_autocast_loss_and_grads_match_reference(kernels):
    ref, fast, idx, tgt = _pair(**kernels)
    with torch.autocast("cuda", dtype=torch.float16):
        _, ref_loss = ref(idx, tgt)
        _, loss = fast(idx, tgt)
    torch.testing.assert_close(loss, ref_loss, rtol=1e-3, atol=1e-3)
    ref_loss.backward()
    loss.backward()
    for name, g in _grads(fast).items():
        r = _grads(ref)[name]
        torch.testing.assert_close(g, r, rtol=5e-2, atol=5e-2 * r.abs().max().item(), msg=name)


@pytest.mark.parametrize("kernels", [TRITON, ALL], ids=["triton", "all"])
def test_compiled_model_has_no_graph_breaks(kernels):
    _, fast, idx, tgt = _pair(**kernels)
    torch._dynamo.reset()
    compiled = torch.compile(fast, fullgraph=True)
    with torch.autocast("cuda", dtype=torch.float16):
        _, loss = compiled(idx, tgt)
        _, eager_loss = fast(idx, tgt)
    loss.backward()
    torch.testing.assert_close(loss, eager_loss, rtol=1e-3, atol=1e-3)
