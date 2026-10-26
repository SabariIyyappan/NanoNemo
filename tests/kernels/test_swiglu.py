"""Triton fused SwiGLU vs the plain-PyTorch reference silu(g) * u."""

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("triton")
from nanonemo.kernels.triton.swiglu import _fwd, swiglu  # noqa: E402

pytestmark = pytest.mark.gpu

# The model's (B*T, d_ff), sizes that leave a partial last block, and a 3-D input.
SHAPES = [(2048, 2048), (7, 2048), (1, 1), (3, 1000), (2, 5, 333)]
# FP16: the reference rounds silu(g) to FP16 before the multiply; the kernel rounds once.
TOL = {torch.float32: dict(rtol=1e-5, atol=1e-5), torch.float16: dict(rtol=2e-3, atol=2e-3)}


def _inputs(shape, dtype):
    torch.manual_seed(0)
    # Scale 3 covers silu's linear, curved, and saturated regions.
    g = (3 * torch.randn(shape, device="cuda", dtype=torch.float32)).to(dtype).requires_grad_()
    u = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    return g, u


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_matches_reference(shape, dtype):
    g, u = _inputs(shape, dtype)
    rg, ru = (t.detach().clone().requires_grad_() for t in (g, u))
    out, ref = swiglu(g, u), F.silu(rg) * ru
    torch.testing.assert_close(out, ref, **TOL[dtype])
    dy = torch.randn_like(out)
    out.backward(dy)
    ref.backward(dy)
    torch.testing.assert_close(g.grad, rg.grad, **TOL[dtype])
    torch.testing.assert_close(u.grad, ru.grad, **TOL[dtype])


def test_gradcheck_fp64():
    g, u = _inputs((4, 33), torch.float64)
    assert torch.autograd.gradcheck(swiglu, (g, u))


def test_opcheck():
    g, u = _inputs((8, 64), torch.float16)
    torch.library.opcheck(_fwd, (g, u))


def test_compiles_without_graph_breaks():
    g, u = _inputs((64, 2048), torch.float16)
    torch._dynamo.reset()
    torch.testing.assert_close(torch.compile(swiglu, fullgraph=True)(g, u), swiglu(g, u))
