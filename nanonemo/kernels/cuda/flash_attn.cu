// Causal, GQA-aware FlashAttention-2 for Turing (sm_75): softmax(QK^T/sqrt(d))V without
// ever materializing the T x T score matrix. Matmuls run on tensor cores through the WMMA
// API (16x16x16 FP16 tiles, FP32 accumulate); softmax bookkeeping is FP32.
//
// Forward: one block = 64 query rows of one (batch, head); 4 warps, 16 rows each. K/V tiles
// of 32 keys stream through shared memory; each warp keeps a running row max m, row sum l
// and an unnormalized output O (online softmax), rescaling O whenever m grows.
// Saves L = m + log(l) per row for the backward.
//
// Backward: one block = 32 keys of one (batch, KV head). It loops over every query head in
// the GQA group and every query tile at or after its keys (causal), recomputing
// P = exp(S - L) from the saved L instead of storing P. dK and dV for its keys accumulate in
// registers (no atomics: the block owns them, across the whole group); dQ is shared by
// many key blocks, so it is added into an FP32 buffer with atomics.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>

using namespace nvcuda;

namespace {

constexpr int THREADS = 128;  // 4 warps
using FragA = wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major>;
using FragAT = wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major>;  // reads a transpose
using FragB = wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major>;
using FragBT = wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major>;
using FragC = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;

// Shared-memory rows are padded past their logical width. Unpadded, every row of a 64-wide
// FP16 tile is 128 bytes, so the 16 rows a WMMA load touches all start on the same bank;
// the padding staggers them across banks.
template <int W> constexpr int pad_h() { return W + 8; }  // FP16 rows: +16 bytes
template <int W> constexpr int pad_f() { return W + 4; }  // FP32 rows: +16 bytes

// Copy rows [row0, row0 + ROWS) of a (T, HD) matrix into shared memory (row stride LD),
// 16 bytes per load, zero-filling rows past T so partial tiles need no special case.
template <int ROWS, int HD, int LD = pad_h<HD>()>
__device__ __forceinline__ void load_tile(half* dst, const half* src, int row0, int T) {
    constexpr int VEC = HD / 8;
    for (int idx = threadIdx.x; idx < ROWS * VEC; idx += THREADS) {
        int r = idx / VEC, c = (idx % VEC) * 8;
        int4 val = make_int4(0, 0, 0, 0);
        if (row0 + r < T) val = *reinterpret_cast<const int4*>(src + (int64_t)(row0 + r) * HD + c);
        *reinterpret_cast<int4*>(dst + r * LD + c) = val;
    }
}

template <int HD>
__global__ void __launch_bounds__(THREADS) fwd_kernel(const half* __restrict__ Q, const half* __restrict__ K,
                                                      const half* __restrict__ V, half* __restrict__ O,
                                                      float* __restrict__ L, int H, int KVH, int T, float scale) {
    constexpr int BR = 64, BC = 32;
    constexpr int LDH = pad_h<HD>(), LDP = pad_h<BC>(), LDS = pad_f<BC>();
    const int q0 = blockIdx.x * BR, h = blockIdx.y, b = blockIdx.z;
    const int kvh = h / (H / KVH);
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;

    // sQ is only needed to load the Q fragments; afterwards the same bytes hold S.
    constexpr int BUF = (BR * LDH * 2 > BR * LDS * 4) ? BR * LDH * 2 : BR * LDS * 4;
    __shared__ __align__(32) char sBuf[BUF];
    __shared__ __align__(32) half sK[BC * LDH];
    __shared__ __align__(32) half sV[BC * LDH];
    __shared__ __align__(32) half sP[BR * LDP];
    __shared__ float sCorr[BR];
    half* sQ = reinterpret_cast<half*>(sBuf);
    float* sS = reinterpret_cast<float*>(sBuf);

    const half* q_base = Q + ((int64_t)b * H + h) * T * HD;
    const half* k_base = K + ((int64_t)b * KVH + kvh) * T * HD;
    const half* v_base = V + ((int64_t)b * KVH + kvh) * T * HD;

    load_tile<BR, HD>(sQ, q_base, q0, T);
    __syncthreads();
    FragA qf[HD / 16];  // this warp's 16 query rows stay in registers for the whole loop
    for (int d = 0; d < HD / 16; ++d) wmma::load_matrix_sync(qf[d], sQ + warp * 16 * LDH + d * 16, LDH);

    float* wS = sS + warp * 16 * LDS;
    half* wP = sP + warp * 16 * LDP;
    float* wCorr = sCorr + warp * 16;

    // WMMA fragments have an opaque element layout. To rescale O's rows in registers we
    // need each element's row: load a 16x16 matrix whose (r, c) entry is r once, and read
    // the row index of element i back from rows.x[i].
    FragC rows;
    __syncthreads();  // all warps hold their Q fragments; sBuf can be overwritten
    for (int i = lane; i < 256; i += 32) wS[i] = (float)(i / 16);
    __syncwarp();
    wmma::load_matrix_sync(rows, wS, 16, wmma::mem_row_major);

    FragC o[HD / 16];
    for (int n = 0; n < HD / 16; ++n) wmma::fill_fragment(o[n], 0.f);

    // Softmax layout: lane handles row lane/2, columns (lane%2)*16 + [0, 16), visited in a
    // rotated order so the 32 lanes hit 32 different shared-memory banks.
    const int my_row = lane / 2, half_sel = lane % 2;
    float m = -INFINITY, l = 0.f;
    const int q = q0 + warp * 16 + my_row;
    const int last_q = min(q0 + BR, T) - 1;
    for (int k0 = 0; k0 <= last_q; k0 += BC) {  // causal: no key tile past the last query row
        __syncthreads();  // every warp is done reading the previous sV
        load_tile<BC, HD>(sK, k_base, k0, T);
        load_tile<BC, HD>(sV, v_base, k0, T);
        __syncthreads();

        for (int n = 0; n < BC / 16; ++n) {  // S = Q K^T (16 x BC per warp)
            FragC acc;
            wmma::fill_fragment(acc, 0.f);
            for (int d = 0; d < HD / 16; ++d) {
                FragBT kf;  // K^T: element (dim, key) read from row-major K
                wmma::load_matrix_sync(kf, sK + n * 16 * LDH + d * 16, LDH);
                wmma::mma_sync(acc, qf[d], kf, acc);
            }
            wmma::store_matrix_sync(wS + n * 16, acc, LDS, wmma::mem_row_major);
        }
        __syncwarp();

        float s[16];
        float mx = -INFINITY;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const int c = half_sel * 16 + ((j + my_row) & 15), key = k0 + c;
            s[j] = (key <= q && key < T) ? wS[my_row * LDS + c] * scale : -INFINITY;
            mx = fmaxf(mx, s[j]);
        }
        const float m_new = fmaxf(m, fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 1)));
        float sum = 0.f;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const int c = half_sel * 16 + ((j + my_row) & 15);
            const float p = (s[j] == -INFINITY) ? 0.f : __expf(s[j] - m_new);
            sum += p;
            wP[my_row * LDP + c] = __float2half(p);
        }
        sum += __shfl_xor_sync(0xffffffff, sum, 1);
        const float corr = (m_new == -INFINITY) ? 1.f : __expf(m - m_new);
        l = l * corr + sum;
        m = m_new;
        if (half_sel == 0) wCorr[my_row] = corr;
        __syncwarp();

        for (int n = 0; n < HD / 16; ++n) {  // O = O * corr + P V
            FragC pv;
            wmma::fill_fragment(pv, 0.f);
            for (int k = 0; k < BC / 16; ++k) {
                FragA pf;
                FragB vf;
                wmma::load_matrix_sync(pf, wP + k * 16, LDP);
                wmma::load_matrix_sync(vf, sV + k * 16 * LDH + n * 16, LDH);
                wmma::mma_sync(pv, pf, vf, pv);
            }
            for (int i = 0; i < pv.num_elements; ++i) o[n].x[i] = o[n].x[i] * wCorr[(int)rows.x[i]] + pv.x[i];
        }
        __syncwarp();
    }

    // Normalize by l and write out, staging one 16x16 tile at a time through wS.
    if (half_sel == 0) wCorr[my_row] = 1.f / l;
    if (half_sel == 0 && q < T) L[((int64_t)b * H + h) * T + q] = m + __logf(l);
    __syncwarp();
    half* o_base = O + ((int64_t)b * H + h) * T * HD;
    for (int n = 0; n < HD / 16; ++n) {
        wmma::store_matrix_sync(wS, o[n], 16, wmma::mem_row_major);
        __syncwarp();
        for (int i = lane; i < 256; i += 32) {
            const int r = i / 16, qr = q0 + warp * 16 + r;
            if (qr < T) o_base[(int64_t)qr * HD + n * 16 + i % 16] = __float2half(wS[i] * wCorr[r]);
        }
        __syncwarp();
    }
}

template <int HD>
__global__ void __launch_bounds__(THREADS) bwd_kernel(const half* __restrict__ Q, const half* __restrict__ K,
                                                      const half* __restrict__ V, const half* __restrict__ dO,
                                                      const float* __restrict__ L, const float* __restrict__ D,
                                                      float* __restrict__ dQ, float* __restrict__ dK,
                                                      float* __restrict__ dV, int H, int KVH, int T, float scale) {
    constexpr int BR = 32, BC = 32;
    constexpr int LDH = pad_h<HD>(), LDP = pad_h<BC>(), LDS = pad_f<BC>(), LDQ = pad_f<HD>();
    constexpr int TILES = (BC / 16) * (HD / 16);  // 16x16 tiles in a BC x HD (or BR x HD) matrix
    constexpr int TPW = TILES / 4;                // ... per warp
    const int k0 = blockIdx.x * BC, kvh = blockIdx.y, b = blockIdx.z;
    const int group = H / KVH;
    const int warp = threadIdx.x / 32;

    __shared__ __align__(32) half sK[BC * LDH];
    __shared__ __align__(32) half sV[BC * LDH];
    __shared__ __align__(32) half sQ[BR * LDH];
    __shared__ __align__(32) half sdO[BR * LDH];
    __shared__ __align__(32) half sP[BR * LDP];
    __shared__ __align__(32) half sdS[BR * LDP];
    // S and dP while they are live, then reused for this tile's dQ (and at the end dK/dV).
    __shared__ __align__(32) float sBuf[(2 * BR * LDS > BR * LDQ) ? 2 * BR * LDS : BR * LDQ];
    __shared__ float sLse[BR], sD[BR];
    float* sS = sBuf;
    float* sdP = sBuf + BR * LDS;
    float* sdQ = sBuf;

    const int64_t kv_off = ((int64_t)b * KVH + kvh) * T * HD;
    load_tile<BC, HD>(sK, K + kv_off, k0, T);
    load_tile<BC, HD>(sV, V + kv_off, k0, T);

    FragC dk[TPW], dv[TPW];
    for (int i = 0; i < TPW; ++i) wmma::fill_fragment(dk[i], 0.f), wmma::fill_fragment(dv[i], 0.f);

    const int mt = warp / 2, nt = warp % 2;  // this warp's tile of the 32x32 S / dP
    for (int g = 0; g < group; ++g) {
        const int h = kvh * group + g;
        const int64_t q_off = ((int64_t)b * H + h) * T * HD;
        const int64_t row_off = ((int64_t)b * H + h) * T;
        for (int q0 = (k0 / BR) * BR; q0 < T; q0 += BR) {  // causal: only queries at or after these keys
            __syncthreads();  // previous tile's reads of sQ/sdO/sdQ are done
            load_tile<BR, HD>(sQ, Q + q_off, q0, T);
            load_tile<BR, HD>(sdO, dO + q_off, q0, T);
            for (int r = threadIdx.x; r < BR; r += THREADS) {
                const bool ok = q0 + r < T;
                sLse[r] = ok ? L[row_off + q0 + r] : INFINITY;
                sD[r] = ok ? D[row_off + q0 + r] : 0.f;
            }
            __syncthreads();

            // S = Q K^T and dP = dO V^T, one 16x16 tile of each per warp
            {
                FragC s_acc, dp_acc;
                wmma::fill_fragment(s_acc, 0.f);
                wmma::fill_fragment(dp_acc, 0.f);
                for (int d = 0; d < HD / 16; ++d) {
                    FragA qf, dof;
                    FragBT kf, vf;
                    wmma::load_matrix_sync(qf, sQ + mt * 16 * LDH + d * 16, LDH);
                    wmma::load_matrix_sync(kf, sK + nt * 16 * LDH + d * 16, LDH);
                    wmma::mma_sync(s_acc, qf, kf, s_acc);
                    wmma::load_matrix_sync(dof, sdO + mt * 16 * LDH + d * 16, LDH);
                    wmma::load_matrix_sync(vf, sV + nt * 16 * LDH + d * 16, LDH);
                    wmma::mma_sync(dp_acc, dof, vf, dp_acc);
                }
                wmma::store_matrix_sync(sS + mt * 16 * LDS + nt * 16, s_acc, LDS, wmma::mem_row_major);
                wmma::store_matrix_sync(sdP + mt * 16 * LDS + nt * 16, dp_acc, LDS, wmma::mem_row_major);
            }
            __syncthreads();

            // P = exp(S*scale - L) (masked), dS = P * (dP - D) * scale
            for (int i = threadIdx.x; i < BR * BC; i += THREADS) {
                const int r = i / BC, c = i % BC, q = q0 + r, key = k0 + c;
                const float p = (key <= q && key < T) ? __expf(sS[r * LDS + c] * scale - sLse[r]) : 0.f;
                sP[r * LDP + c] = __float2half(p);
                sdS[r * LDP + c] = __float2half(p * (sdP[r * LDS + c] - sD[r]) * scale);
            }
            __syncthreads();

            for (int i = 0; i < TPW; ++i) {
                const int t = warp * TPW + i, tm = t / (HD / 16), tn = t % (HD / 16);
                // dV += P^T dO and dK += dS^T Q, summed over the query rows of this tile
                for (int k = 0; k < BR / 16; ++k) {
                    FragAT pt, dst;
                    FragB dof, qf;
                    wmma::load_matrix_sync(pt, sP + k * 16 * LDP + tm * 16, LDP);
                    wmma::load_matrix_sync(dof, sdO + k * 16 * LDH + tn * 16, LDH);
                    wmma::mma_sync(dv[i], pt, dof, dv[i]);
                    wmma::load_matrix_sync(dst, sdS + k * 16 * LDP + tm * 16, LDP);
                    wmma::load_matrix_sync(qf, sQ + k * 16 * LDH + tn * 16, LDH);
                    wmma::mma_sync(dk[i], dst, qf, dk[i]);
                }
                // dQ tile = dS K (S/dP in sBuf are dead now; sdQ reuses it)
                FragC dq;
                wmma::fill_fragment(dq, 0.f);
                for (int k = 0; k < BC / 16; ++k) {
                    FragA dsf;
                    FragB kf;
                    wmma::load_matrix_sync(dsf, sdS + tm * 16 * LDP + k * 16, LDP);
                    wmma::load_matrix_sync(kf, sK + k * 16 * LDH + tn * 16, LDH);
                    wmma::mma_sync(dq, dsf, kf, dq);
                }
                wmma::store_matrix_sync(sdQ + tm * 16 * LDQ + tn * 16, dq, LDQ, wmma::mem_row_major);
            }
            __syncthreads();

            for (int i = threadIdx.x; i < BR * HD; i += THREADS) {
                const int r = i / HD, c = i % HD, q = q0 + r;
                if (q < T) atomicAdd(&dQ[q_off + (int64_t)q * HD + c], sdQ[r * LDQ + c]);
            }
        }
    }

    // Write dK, dV through shared memory so rows past T can be masked.
    for (int pass = 0; pass < 2; ++pass) {
        __syncthreads();
        for (int i = 0; i < TPW; ++i) {
            const int t = warp * TPW + i, tm = t / (HD / 16), tn = t % (HD / 16);
            wmma::store_matrix_sync(sBuf + tm * 16 * LDQ + tn * 16, pass ? dv[i] : dk[i], LDQ, wmma::mem_row_major);
        }
        __syncthreads();
        float* out = (pass ? dV : dK) + kv_off;
        for (int i = threadIdx.x; i < BC * HD; i += THREADS) {
            const int r = i / HD, c = i % HD, key = k0 + r;
            if (key < T) out[(int64_t)key * HD + c] = sBuf[r * LDQ + c];
        }
    }
}

void check_inputs(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(q.scalar_type() == torch::kFloat16 && k.scalar_type() == torch::kFloat16 &&
                    v.scalar_type() == torch::kFloat16,
                "inputs must be float16");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "expected (B, H, T, hd) / (B, KVH, T, hd)");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(), "inputs must be contiguous");
    TORCH_CHECK(q.size(0) == k.size(0) && q.size(2) == k.size(2) && k.sizes() == v.sizes(),
                "K and V must share shape; Q must share batch and length");
    TORCH_CHECK(q.size(3) == k.size(3), "Q and K must share head_dim");
    TORCH_CHECK(q.size(3) == 32 || q.size(3) == 64, "head_dim must be 32 or 64");
    TORCH_CHECK(q.size(1) % k.size(1) == 0, "query heads must group evenly over KV heads");
    for (auto* t : {&q, &k, &v})
        TORCH_CHECK(reinterpret_cast<uintptr_t>(t->data_ptr()) % 16 == 0, "inputs must be 16-byte aligned");
}

const half* hp(const torch::Tensor& t) { return reinterpret_cast<const half*>(t.data_ptr<at::Half>()); }

}  // namespace

std::vector<torch::Tensor> flash_fwd(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
    check_inputs(q, k, v);
    const int B = q.size(0), H = q.size(1), T = q.size(2), hd = q.size(3), KVH = k.size(1);
    auto o = torch::empty_like(q);
    auto lse = torch::empty({B, H, T}, q.options().dtype(torch::kFloat32));
    if (T == 0) return {o, lse};

    const dim3 grid((T + 63) / 64, H, B);
    const float scale = 1.0f / sqrtf((float)hd);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    auto* op = reinterpret_cast<half*>(o.data_ptr<at::Half>());
    if (hd == 64)
        fwd_kernel<64><<<grid, THREADS, 0, stream>>>(hp(q), hp(k), hp(v), op, lse.data_ptr<float>(), H, KVH, T, scale);
    else
        fwd_kernel<32><<<grid, THREADS, 0, stream>>>(hp(q), hp(k), hp(v), op, lse.data_ptr<float>(), H, KVH, T, scale);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {o, lse};
}

std::vector<torch::Tensor> flash_bwd(torch::Tensor d_o, torch::Tensor q, torch::Tensor k, torch::Tensor v,
                                     torch::Tensor o, torch::Tensor lse) {
    check_inputs(q, k, v);
    TORCH_CHECK(d_o.is_contiguous() && d_o.sizes() == q.sizes() && d_o.scalar_type() == torch::kFloat16,
                "d_o must be a contiguous float16 tensor shaped like q");
    const int B = q.size(0), H = q.size(1), T = q.size(2), hd = q.size(3), KVH = k.size(1);
    // D = rowsum(dO * O): lets dS be computed from P alone, without re-running the softmax.
    auto D = (d_o.to(torch::kFloat32) * o.to(torch::kFloat32)).sum(-1).contiguous();
    auto dq = torch::zeros(q.sizes(), q.options().dtype(torch::kFloat32));
    auto dk = torch::empty(k.sizes(), k.options().dtype(torch::kFloat32));
    auto dv = torch::empty(v.sizes(), v.options().dtype(torch::kFloat32));
    if (T > 0) {
        const dim3 grid((T + 31) / 32, KVH, B);
        const float scale = 1.0f / sqrtf((float)hd);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        auto args = std::make_tuple(hp(q), hp(k), hp(v), hp(d_o), lse.data_ptr<float>(), D.data_ptr<float>(),
                                    dq.data_ptr<float>(), dk.data_ptr<float>(), dv.data_ptr<float>(), H, KVH, T, scale);
        if (hd == 64)
            std::apply([&](auto... a) { bwd_kernel<64><<<grid, THREADS, 0, stream>>>(a...); }, args);
        else
            std::apply([&](auto... a) { bwd_kernel<32><<<grid, THREADS, 0, stream>>>(a...); }, args);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {dq.to(torch::kFloat16), dk.to(torch::kFloat16), dv.to(torch::kFloat16)};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fwd", &flash_fwd, "FlashAttention-2 forward (causal, GQA): returns (o, logsumexp)");
    m.def("bwd", &flash_bwd, "FlashAttention-2 backward: returns (dq, dk, dv)");
}
