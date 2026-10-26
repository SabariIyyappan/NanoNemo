"""Fused LM head + cross-entropy vs F.cross_entropy on the full logits."""

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("triton")
from nanonemo.kernels.triton.linear_ce import _linear_ce, linear_cross_entropy  # noqa: E402

pytestmark = pytest.mark.gpu


def _inputs(N, D, V, dtype=torch.float32):
    torch.manual_seed(0)
    x = torch.randn(N, D, device="cuda", dtype=dtype, requires_grad=True)
    w = (0.05 * torch.randn(V, D, device="cuda")).requires_grad_()
    t = torch.randint(0, V, (N,), device="cuda")
    return x, w, t


def _reference(x, w, t):
    return F.cross_entropy((x @ w.t().to(x.dtype)).float(), t)


# (N, D, V, chunk_rows): several chunks with a ragged last one, one chunk, V not a multiple of BLOCK.
@pytest.mark.parametrize("N, D, V, chunk", [(300, 64, 5000, 128), (16, 32, 100, 512), (64, 768, 50304, 32)])
def test_fp32_matches_reference(N, D, V, chunk):
    x, w, t = _inputs(N, D, V)
    rx, rw = (a.detach().clone().requires_grad_() for a in (x, w))
    loss = linear_cross_entropy(x, w, t, chunk_rows=chunk)
    ref = _reference(rx, rw, t)
    torch.testing.assert_close(loss, ref, rtol=1e-5, atol=1e-5)
    (2.0 * loss).backward()  # a non-unit incoming gradient, as with a GradScaler
    (2.0 * ref).backward()
    torch.testing.assert_close(x.grad, rx.grad, rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(w.grad, rw.grad, rtol=1e-4, atol=1e-6)


def test_autocast_matches_reference():
    x, w, t = _inputs(512, 768, 50304)
    rx, rw = (a.detach().clone().requires_grad_() for a in (x, w))
    with torch.autocast("cuda", dtype=torch.float16):
        loss = linear_cross_entropy(x, w, t)
        ref = F.cross_entropy(F.linear(rx, rw).float(), t)
    torch.testing.assert_close(loss, ref, rtol=1e-3, atol=1e-3)
    (1024.0 * loss).backward()
    (1024.0 * ref).backward()
    torch.testing.assert_close(x.grad, rx.grad, rtol=2e-2, atol=1e-3)
    torch.testing.assert_close(w.grad, rw.grad, rtol=2e-2, atol=1e-3)


def test_uses_less_peak_memory():
    def peak(fn):
        x, w, t = _inputs(2048, 768, 50304)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        with torch.autocast("cuda", dtype=torch.float16):
            loss = fn(x, w, t)
        loss.backward()
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() - base

    fused = peak(linear_cross_entropy)
    unfused = peak(lambda x, w, t: F.cross_entropy(F.linear(x, w).float(), t))
    assert fused < 0.6 * unfused, (fused / 2**20, unfused / 2**20)


def test_opcheck():
    x, w, t = _inputs(40, 32, 300)
    torch.library.opcheck(_linear_ce, (x, w, t, torch.float32, 16))


def test_compiles_without_graph_breaks():
    x, w, t = _inputs(256, 64, 1000)
    torch._dynamo.reset()
    fn = torch.compile(linear_cross_entropy, fullgraph=True)
    torch.testing.assert_close(fn(x, w, t), linear_cross_entropy(x, w, t))
