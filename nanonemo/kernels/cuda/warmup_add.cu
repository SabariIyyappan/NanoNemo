// Warm-up CUDA kernel: c = a + b, elementwise, FP32.
// Not used by the model. It exists to learn the mechanics every later kernel shares:
// a __global__ function, a <<<grid, block>>> launch, thread indexing, bounds checks,
// launching on PyTorch's current stream, and exposing the function to Python.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

// Runs on the GPU, once per thread. Every thread computes its own global index i
// from its block number and its position inside the block, and handles element i.
__global__ void add_kernel(const float* a, const float* b, float* c, int64_t n) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {  // the last block can stick out past the end of the data
        c[i] = a[i] + b[i];
    }
}

// Runs on the CPU. Checks inputs, allocates the output, and launches the kernel.
torch::Tensor add(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(a.scalar_type() == torch::kFloat32 && b.scalar_type() == torch::kFloat32, "inputs must be float32");
    TORCH_CHECK(a.sizes() == b.sizes(), "inputs must have the same shape");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous(), "inputs must be contiguous");

    auto c = torch::empty_like(a);
    int64_t n = a.numel();
    if (n == 0) return c;

    const int threads = 256;                                          // threads per block (8 warps)
    const int64_t blocks = (n + threads - 1) / threads;               // enough blocks to cover n
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();           // the stream PyTorch is using
    add_kernel<<<blocks, threads, 0, stream>>>(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();                                   // raise if the launch itself failed
    return c;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add", &add, "c = a + b (FP32, CUDA)");
}
