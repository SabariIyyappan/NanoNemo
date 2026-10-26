"""The warm-up CUDA kernel: builds, launches, covers partial last blocks, and rejects bad inputs."""

import pytest
import torch

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def ext():
    from nanonemo.kernels.cuda import build

    return build("warmup_add")


# 1 element, exactly one block, one past a block boundary, and a size that isn't a multiple of 256.
@pytest.mark.parametrize("n", [1, 256, 257, 1_000_003])
def test_add_matches_torch(ext, n):
    a, b = torch.randn(n, device="cuda"), torch.randn(n, device="cuda")
    torch.testing.assert_close(ext.add(a, b), a + b, rtol=0, atol=0)


def test_rejects_wrong_dtype(ext):
    a = torch.randn(8, device="cuda", dtype=torch.float16)
    with pytest.raises(RuntimeError, match="float32"):
        ext.add(a, a)
