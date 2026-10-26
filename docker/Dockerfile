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
        zstandard \
        kaggle \
        pytest

# Liger-Kernel: reference Triton kernels to benchmark ours against. --no-deps keeps
# its torch/triton requirements from touching the bundled versions.
RUN pip install --no-cache-dir --no-deps liger-kernel==0.8.5

# Evaluation (M10): lm-evaluation-harness and transformers. NanoNemo checkpoints are exported
# to Hugging Face Llama format so they run through the same lm-eval code path as GPT-2.
RUN pip install --no-cache-dir lm-eval==0.4.9 transformers==4.57.1

WORKDIR /workspace/nanonemo
