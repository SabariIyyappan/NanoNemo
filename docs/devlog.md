# Devlog

Short notes written while building: what we tried, what surprised us, and the numbers.

## M0: environment

- The NGC PyTorch container works on the RTX 2070 Super (sm_75). Nsight Systems traces and Nsight Compute counters both work inside it, after setting `NVreg_RestrictProfilingToAdminUsers=0` on the host.
- The first container tried (`26.09`, CUDA 13.4) printed "CUDA Forward Compatibility mode ENABLED" on the R580 driver. We switched to `25.11` (CUDA 13.0), which runs natively.
- **Surprise:** a 20-line Triton FP16 matmul was correct but ran at 1.8 TFLOP/s, against cuBLAS's 20.9. Three checks found the cause:
  - ncu reported `sm__inst_executed_pipe_tensor.sum = 0`.
  - The Triton IR had a `#blocked` layout instead of `nvidia_mma`.
  - The PTX had 1024 `fma.rn` and no `mma.sync`.
- Bisecting Triton versions found the cause: 3.0 and 3.2 emit `mma.sync.m16n8k8` (~18 TFLOP/s), while 3.4+ fall back to FMA on Turing. So the FlashAttention kernel will be written in CUDA C++, not Triton.
- Lesson: a correctness test can't tell you whether you're using the hardware. Check the counters.
