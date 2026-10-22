# NanoNemo dev image.
# The NGC base ships a matched PyTorch / CUDA / cuDNN / NCCL / Triton / Nsight stack.
# We pin an exact tag so every benchmark is reproducible. 25.11 is built for sm_75 (Turing)
# and its CUDA 13.0 runs natively on the R580 driver (no forward-compatibility shim).
FROM nvcr.io/nvidia/pytorch:25.11-py3

# Small extras only. NGC sets PIP_CONSTRAINT so pip cannot replace the bundled torch.
RUN pip install --no-cache-dir \
        tiktoken \
        wandb \
        pyyaml \
        pytest

WORKDIR /workspace/nanonemo
