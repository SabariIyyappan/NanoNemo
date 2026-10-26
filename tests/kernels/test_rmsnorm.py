"""Triton fused add + RMSNorm vs the plain-PyTorch reference."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("triton")
from nanonemo.kernels.triton.rmsnorm import _fwd, add_rms_norm  # noqa: E402
from nanonemo.model.layers import RMSNorm  # noqa: E402

pytestmark = pytest.mark.gpu

EPS = 1e-5
# (rows, hidden): the model's shape, plus non-power-of-2, tiny, and wide hidden sizes.
SHAPES = [(2048, 768), (7, 768), (1, 33), (130, 1000), (64, 4096)]
# (residual dtype, delta dtype): FP32 stream + FP16 sublayer output is the AMP case.
DTYPES = [(torch.float32, torch.float32), (torch.float32, torch.float16), (torch.float16, torch.float16)]
# FP16 atol is 2 ulps at magnitude 2-4: for dx the reference rounds the norm's gradient to
# FP16 before adding the residual's gradient, while the kernel adds in FP32 and rounds once.
TOL = {torch.float32: dict(rtol=1e-5, atol=1e-5), torch.float16: dict(rtol=1e-3, atol=4e-3)}


def _reference(x, delta, weight):
    h = x + delta if delta is not None else x
    return RMSNorm.forward(SimpleNamespace(eps=EPS, weight=weight), h), h


def _inputs(M, N, x_dtype, d_dtype, with_delta=True):
    torch.manual_seed(0)
    x = torch.randn(M, N, device="cuda", dtype=x_dtype, requires_grad=True)
    d = torch.randn(M, N, device="cuda", dtype=d_dtype, requires_grad=True) if with_delta else None
    w = (1 + 0.1 * torch.randn(N, device="cuda")).requires_grad_()
    return x, d, w


def _clone(*ts):
    return [t.detach().clone().requires_grad_() if t is not None else None for t in ts]


@pytest.mark.parametrize("M, N", SHAPES)
@pytest.mark.parametrize("x_dtype, d_dtype", DTYPES)
@pytest.mark.parametrize("with_delta", [True, False])
def test_matches_reference(M, N, x_dtype, d_dtype, with_delta):
    x, d, w = _inputs(M, N, x_dtype, d_dtype, with_delta)
    rx, rd, rw = _clone(x, d, w)
    y, h = add_rms_norm(x, d, w, EPS)
    ry, rh = _reference(rx, rd, rw)
    tol = TOL[x_dtype]
    torch.testing.assert_close(y, ry, **tol)
    torch.testing.assert_close(h, rh, **tol)

    # Upstream gradient into both outputs, as in the model (next sublayer + residual).
    gy, gh = torch.randn_like(y), torch.randn_like(h)
    (y * gy).sum().backward(retain_graph=True)
    (h * gh).sum().backward()
    (ry * gy).sum().backward(retain_graph=True)
    (rh * gh).sum().backward()
    torch.testing.assert_close(x.grad, rx.grad, **tol)
    if with_delta:
        torch.testing.assert_close(d.grad, rd.grad, **TOL[d_dtype])
    # dw sums over all rows, so its error grows with M.
    torch.testing.assert_close(w.grad, rw.grad, rtol=tol["rtol"] * 10, atol=tol["atol"] * M**0.5)


def test_autocast_writes_normalized_output_in_fp16():
    x, d, w = _inputs(16, 768, torch.float32, torch.float16)
    with torch.autocast("cuda", dtype=torch.float16):
        y, h = add_rms_norm(x, d, w, EPS)
    assert y.dtype == torch.float16 and h.dtype == torch.float32
    ry, _ = _reference(x, d, w)
    torch.testing.assert_close(y, ry.half(), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("with_delta", [True, False])
def test_gradcheck_fp64(with_delta):
    x, d, w = _inputs(5, 37, torch.float64, torch.float64, with_delta)
    w = w.detach().double().requires_grad_()
    fn = (lambda x, d, w: add_rms_norm(x, d, w, EPS)) if with_delta else (lambda x, w: add_rms_norm(x, None, w, EPS)[0])
    args = (x, d, w) if with_delta else (x, w)
    assert torch.autograd.gradcheck(fn, args)


@pytest.mark.parametrize("with_delta", [True, False])
def test_opcheck(with_delta):
    # Checks the schema, the fake (meta) implementation, and the autograd registration.
    x, d, w = _inputs(8, 64, torch.float32, torch.float16, with_delta)
    torch.library.opcheck(_fwd, (x, d, w, EPS, torch.float32))


def test_compiles_without_graph_breaks():
    x, d, w = _inputs(64, 768, torch.float32, torch.float16)
    torch._dynamo.reset()
    compiled = torch.compile(lambda x, d, w: add_rms_norm(x, d, w, EPS), fullgraph=True)
    with torch.autocast("cuda", dtype=torch.float16):
        y, h = compiled(x, d, w)
        ry, rh = add_rms_norm(x, d, w, EPS)
    torch.testing.assert_close(y, ry)
    torch.testing.assert_close(h, rh)
