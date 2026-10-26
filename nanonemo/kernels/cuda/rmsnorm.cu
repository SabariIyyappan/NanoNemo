// Fused residual add + RMSNorm in CUDA C++, the same op as nanonemo/kernels/triton/rmsnorm.py:
//     h = x + delta,   y = h * rsqrt(mean(h^2) + eps) * w
// backward, with n = h * rstd and dn = dy * w:
//     dh = rstd * (dn - n * mean(dn * n)) + dh_residual,   dw = sum over rows of dy * n
//
// One warp per row, 4 rows per 128-thread block. Each lane owns chunks of V consecutive
// elements (V = 8: one 16-byte load for FP16, two for FP32), strided 32 chunks apart so a
// warp's loads are contiguous. The row stays in registers between the reduction and the
// write, so every input is read once and every output written once. The sum of squares is
// reduced with warp shuffles only: no shared memory, no __syncthreads.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <type_traits>

namespace {

using Half = at::Half;

template <int V, typename T>
__device__ __forceinline__ void load(const T* p, float* o) {
    if constexpr (V == 8 && std::is_same_v<T, float>) {
        const float4 a = reinterpret_cast<const float4*>(p)[0], b = reinterpret_cast<const float4*>(p)[1];
        o[0] = a.x, o[1] = a.y, o[2] = a.z, o[3] = a.w, o[4] = b.x, o[5] = b.y, o[6] = b.z, o[7] = b.w;
    } else if constexpr (V == 8) {
        const int4 raw = *reinterpret_cast<const int4*>(p);
        const __half2* h2 = reinterpret_cast<const __half2*>(&raw);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float2 f = __half22float2(h2[i]);
            o[2 * i] = f.x, o[2 * i + 1] = f.y;
        }
    } else {
#pragma unroll
        for (int i = 0; i < V; ++i) o[i] = static_cast<float>(p[i]);
    }
}

template <int V, typename T>
__device__ __forceinline__ void store(T* p, const float* v) {
    if constexpr (V == 8 && std::is_same_v<T, float>) {
        reinterpret_cast<float4*>(p)[0] = make_float4(v[0], v[1], v[2], v[3]);
        reinterpret_cast<float4*>(p)[1] = make_float4(v[4], v[5], v[6], v[7]);
    } else if constexpr (V == 8) {
        int4 raw;
        __half2* h2 = reinterpret_cast<__half2*>(&raw);
#pragma unroll
        for (int i = 0; i < 4; ++i) h2[i] = __floats2half2_rn(v[2 * i], v[2 * i + 1]);
        *reinterpret_cast<int4*>(p) = raw;
    } else {
#pragma unroll
        for (int i = 0; i < V; ++i) p[i] = static_cast<T>(v[i]);
    }
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}

template <int V, int MAXC, typename TX, typename TD, typename TH, typename TY, bool HAS_DELTA>
__global__ void __launch_bounds__(128) fwd_kernel(const TX* __restrict__ x, const TD* __restrict__ d,
                                                  const float* __restrict__ w, TY* __restrict__ y,
                                                  TH* __restrict__ h, float* __restrict__ rstd_out, int64_t M,
                                                  int N, float eps) {
    const int64_t row = (int64_t)blockIdx.x * 4 + threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    if (row >= M) return;
    const int chunks = N / V;
    float v[MAXC][V];
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < MAXC; ++i) {
        const int c = lane + i * 32;
        if (c >= chunks) break;
        const int64_t off = row * N + (int64_t)c * V;
        load<V>(x + off, v[i]);
        if constexpr (HAS_DELTA) {
            float dv[V];
            load<V>(d + off, dv);
#pragma unroll
            for (int j = 0; j < V; ++j) v[i][j] = static_cast<float>(static_cast<TH>(v[i][j] + dv[j]));
            store<V>(h + off, v[i]);  // normalize the residual as stored, like add-then-norm
        }
#pragma unroll
        for (int j = 0; j < V; ++j) ss += v[i][j] * v[i][j];
    }
    const float rstd = 1.f / sqrtf(warp_sum(ss) / N + eps);
    if (lane == 0) rstd_out[row] = rstd;
#pragma unroll
    for (int i = 0; i < MAXC; ++i) {
        const int c = lane + i * 32;
        if (c >= chunks) break;
        float wv[V];
        load<V>(w + (int64_t)c * V, wv);
#pragma unroll
        for (int j = 0; j < V; ++j) wv[j] *= v[i][j] * rstd;
        store<V>(y + row * N + (int64_t)c * V, wv);
    }
}

// Each warp handles rows warp, warp + 4, ... of its block's slice and keeps its own dw
// partial in registers, written to its own row of dw_partial (summed on the host side).
template <int V, int MAXC, typename TY, typename TH, typename TX, typename TD, bool HAS_DELTA>
__global__ void __launch_bounds__(128) bwd_kernel(const TY* __restrict__ dy, const TH* __restrict__ dh,
                                                  const TH* __restrict__ h, const float* __restrict__ w,
                                                  const float* __restrict__ rstd, TX* __restrict__ dx,
                                                  TD* __restrict__ dd, float* __restrict__ dw_partial, int64_t M,
                                                  int N, int64_t rows_per_block) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int chunks = N / V;
    const int64_t row_end = min((int64_t)(blockIdx.x + 1) * rows_per_block, M);
    float wv[MAXC][V], dw[MAXC][V];
#pragma unroll
    for (int i = 0; i < MAXC; ++i) {
        const int c = lane + i * 32;
        if (c < chunks) load<V>(w + (int64_t)c * V, wv[i]);
#pragma unroll
        for (int j = 0; j < V; ++j) dw[i][j] = 0.f;
    }
    for (int64_t row = (int64_t)blockIdx.x * rows_per_block + warp; row < row_end; row += 4) {
        const float r = rstd[row];
        float n[MAXC][V], dn[MAXC][V];
        float dot = 0.f;
#pragma unroll
        for (int i = 0; i < MAXC; ++i) {
            const int c = lane + i * 32;
            if (c >= chunks) break;
            const int64_t off = row * N + (int64_t)c * V;
            float g[V];
            load<V>(h + off, n[i]);
            load<V>(dy + off, g);
#pragma unroll
            for (int j = 0; j < V; ++j) {
                n[i][j] *= r;
                dw[i][j] += g[j] * n[i][j];
                dn[i][j] = g[j] * wv[i][j];
                dot += dn[i][j] * n[i][j];
            }
        }
        const float mean = warp_sum(dot) / N;
#pragma unroll
        for (int i = 0; i < MAXC; ++i) {
            const int c = lane + i * 32;
            if (c >= chunks) break;
            const int64_t off = row * N + (int64_t)c * V;
            float g[V];
            if constexpr (HAS_DELTA) load<V>(dh + off, g);
#pragma unroll
            for (int j = 0; j < V; ++j) {
                const float gj = (dn[i][j] - n[i][j] * mean) * r;
                g[j] = HAS_DELTA ? g[j] + gj : gj;
            }
            store<V>(dx + off, g);
            if constexpr (HAS_DELTA) store<V>(dd + off, g);
        }
    }
    float* out = dw_partial + ((int64_t)blockIdx.x * 4 + warp) * N;
#pragma unroll
    for (int i = 0; i < MAXC; ++i) {
        const int c = lane + i * 32;
        if (c < chunks) store<V>(out + (int64_t)c * V, dw[i]);
    }
}

// Calls f(T{}) with T the C++ type of a float32 / float16 tensor.
template <typename F>
void dispatch(at::ScalarType s, F&& f) {
    if (s == at::kFloat) f(float{});
    else if (s == at::kHalf) f(Half{});
    else TORCH_CHECK(false, "only float32 and float16 are supported, got ", s);
}

template <typename A, typename B>
using Promote = std::conditional_t<std::is_same_v<A, float> || std::is_same_v<B, float>, float, Half>;

// Picks the vector width and the per-lane chunk count (registers are sized at compile time).
template <typename F>
void dispatch_shape(int N, F&& f) {
    if (N % 8 == 0 && N <= 1024) f(std::integral_constant<int, 8>{}, std::integral_constant<int, 4>{});
    else if (N % 8 == 0 && N <= 4096) f(std::integral_constant<int, 8>{}, std::integral_constant<int, 16>{});
    else if (N <= 1024) f(std::integral_constant<int, 1>{}, std::integral_constant<int, 32>{});
    else TORCH_CHECK(false, "hidden size must be <= 1024, or a multiple of 8 and <= 4096; got ", N);
}

template <typename T>
T* ptr(torch::Tensor& t) { return reinterpret_cast<T*>(t.data_ptr()); }
template <typename T>
const T* cptr(const torch::Tensor& t) { return reinterpret_cast<const T*>(t.data_ptr()); }

}  // namespace

// Returns (y, h, rstd). delta may be None (then h is empty: the caller reuses x).
std::vector<torch::Tensor> fwd(torch::Tensor x, c10::optional<torch::Tensor> delta, torch::Tensor weight, double eps,
                               at::ScalarType out_dtype) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous(), "x must be a contiguous CUDA tensor");
    TORCH_CHECK(weight.scalar_type() == at::kFloat && weight.is_contiguous(), "weight must be contiguous float32");
    const bool has_delta = delta.has_value();
    if (has_delta) TORCH_CHECK(delta->is_contiguous() && delta->sizes() == x.sizes(), "delta must match x");
    const int N = x.size(-1);
    const int64_t M = x.numel() / N;
    TORCH_CHECK(weight.numel() == N, "weight must have hidden-size elements");
    auto y = torch::empty(x.sizes(), x.options().dtype(out_dtype));
    auto h = has_delta ? torch::empty(x.sizes(), x.options().dtype(at::promote_types(x.scalar_type(), delta->scalar_type())))
                       : x.new_empty({0});
    auto rstd = torch::empty(x.sizes().slice(0, x.dim() - 1), x.options().dtype(at::kFloat));
    if (M == 0) return {y, h, rstd};
    const auto& d = has_delta ? *delta : x;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    dispatch_shape(N, [&](auto v, auto c) {
        dispatch(x.scalar_type(), [&](auto tx) {
            dispatch(d.scalar_type(), [&](auto td) {
                dispatch(out_dtype, [&](auto ty) {
                    using TX = decltype(tx);
                    using TD = decltype(td);
                    using TY = decltype(ty);
                    using TH = Promote<TX, TD>;
                    auto launch = [&](auto kernel) {
                        kernel<<<(M + 3) / 4, 128, 0, stream>>>(cptr<TX>(x), cptr<TD>(d), cptr<float>(weight), ptr<TY>(y),
                                                                has_delta ? ptr<TH>(h) : nullptr, ptr<float>(rstd), M,
                                                                N, (float)eps);
                    };
                    if (has_delta) launch(fwd_kernel<decltype(v)::value, decltype(c)::value, TX, TD, TH, TY, true>);
                    else launch(fwd_kernel<decltype(v)::value, decltype(c)::value, TX, TD, TH, TY, false>);
                });
            });
        });
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {y, h, rstd};
}

// Returns (dx, ddelta, dw). dh/ddelta exist only when the forward had a delta; h is then the
// stored residual, otherwise x itself.
std::vector<torch::Tensor> bwd(torch::Tensor dy, c10::optional<torch::Tensor> dh, torch::Tensor h, torch::Tensor weight,
                               torch::Tensor rstd, at::ScalarType dx_dtype, c10::optional<at::ScalarType> dd_dtype) {
    TORCH_CHECK(dy.is_contiguous() && h.is_contiguous() && dy.sizes() == h.sizes(), "dy and h must be contiguous, same shape");
    const bool has_delta = dh.has_value();
    TORCH_CHECK(has_delta == dd_dtype.has_value(), "dh and ddelta_dtype go together");
    if (has_delta) TORCH_CHECK(dh->is_contiguous() && dh->scalar_type() == h.scalar_type(), "dh must be contiguous, h's dtype");
    const int N = h.size(-1);
    const int64_t M = h.numel() / N;
    auto dx = torch::empty(h.sizes(), h.options().dtype(dx_dtype));
    auto dd = has_delta ? torch::empty(h.sizes(), h.options().dtype(*dd_dtype)) : h.new_empty({0});
    const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    const int64_t blocks = std::max<int64_t>(1, std::min<int64_t>((M + 3) / 4, sms * 2));
    const int64_t rows_per_block = (M + blocks - 1) / blocks;
    auto dw_partial = torch::empty({blocks * 4, N}, h.options().dtype(at::kFloat));
    if (M == 0) return {dx, dd, torch::zeros_like(weight)};
    const auto& dhh = has_delta ? *dh : h;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    dispatch_shape(N, [&](auto v, auto c) {
        dispatch(dy.scalar_type(), [&](auto ty) {
            dispatch(dx_dtype, [&](auto tx) {
                dispatch(has_delta ? *dd_dtype : dx_dtype, [&](auto td) {
                    using TY = decltype(ty);
                    using TX = decltype(tx);
                    using TD = decltype(td);
                    // h's dtype: the promoted residual with a delta, x's own dtype without.
                    auto run = [&](auto th, auto kernel) {
                        using TH = decltype(th);
                        TORCH_CHECK(h.scalar_type() == (std::is_same_v<TH, float> ? at::kFloat : at::kHalf), "unexpected h dtype");
                        kernel<<<blocks, 128, 0, stream>>>(cptr<TY>(dy), cptr<TH>(dhh), cptr<TH>(h), cptr<float>(weight),
                                                           cptr<float>(rstd), ptr<TX>(dx),
                                                           has_delta ? ptr<TD>(dd) : nullptr, ptr<float>(dw_partial), M, N,
                                                           rows_per_block);
                    };
                    if (has_delta) run(Promote<TX, TD>{}, bwd_kernel<decltype(v)::value, decltype(c)::value, TY, Promote<TX, TD>, TX, TD, true>);
                    else run(TX{}, bwd_kernel<decltype(v)::value, decltype(c)::value, TY, TX, TX, TD, false>);
                });
            });
        });
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {dx, dd, dw_partial.sum(0)};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fwd", &fwd, "fused residual add + RMSNorm forward: (y, h, rstd)");
    m.def("bwd", &bwd, "fused residual add + RMSNorm backward: (dx, ddelta, dw)");
}
