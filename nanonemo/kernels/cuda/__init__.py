"""CUDA C++ kernels, compiled on first use with torch.utils.cpp_extension (nvcc + ninja).

The build is cached under ~/.cache/torch_extensions and rebuilt only when a source changes.
torch builds for the visible GPU's architecture (sm_75 on the 2070 Super and the T4).
"""

from functools import cache
from pathlib import Path

from torch.utils.cpp_extension import load

SRC = Path(__file__).parent


@cache
def build(name: str):
    """Compile nanonemo/kernels/cuda/<name>.cu (once per process) and return the module."""
    return load(name=f"nanonemo_{name}", sources=[str(SRC / f"{name}.cu")], extra_cuda_cflags=["-O3"], verbose=False)
