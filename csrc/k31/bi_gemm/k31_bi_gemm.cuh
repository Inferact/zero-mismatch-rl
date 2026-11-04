// K31 port of the batch-invariant split-K tcgen05 GEMM (vLLM
// csrc/batch_invariant/bi_gemm_sm100_splitk.cu). No torch includes here.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace k31bi {

// C[M,N] = A[M,K] @ W[N,K]^T. A row-major (lda), W row-major [N,K] (ldb =
// stride between rows of W, i.e. K-major B), C row-major (ldc). lda/ldb/ldc
// multiples of 8 elements, 16 B aligned pointers, N % 8 == 0.
// add (optional, bf16 [M, N], row stride ldadd): fused epilogue addend,
// C = round(fp32(sum) + fp32(add)) -- the fp32 add then the single rounding
// that torch's (y32 + add.float()).to(bf16) does, bit for bit.
struct Args {
  const void* a;
  const void* b;
  void* c;
  int64_t lda, ldb, ldc;
  int M, N, K;
  const void* add = nullptr;
  int64_t ldadd = 0;
};

#ifdef __CUDACC__
// 8 consecutive fp32 sums v[0..7] -> C[row, n .. n+7] (+ addend, one rounding).
template <bool kF32Out>
__device__ __forceinline__ void store8_out(void* c, int64_t ldc, int row, int n,
                                           const float (&v0)[8],
                                           const void* add, int64_t ldadd) {
  float v[8];
#pragma unroll
  for (int u = 0; u < 8; ++u) v[u] = v0[u];
  if (add != nullptr) {
    const uint4 a = *reinterpret_cast<const uint4*>(
        static_cast<const __nv_bfloat16*>(add) +
        static_cast<int64_t>(row) * ldadd + n);
    const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&a);
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      const float2 f = __bfloat1622float2(h[u]);
      v[2 * u] = __fadd_rn(v[2 * u], f.x);
      v[2 * u + 1] = __fadd_rn(v[2 * u + 1], f.y);
    }
  }
  if constexpr (kF32Out) {
    float* dst = static_cast<float*>(c) + static_cast<int64_t>(row) * ldc + n;
    reinterpret_cast<float4*>(dst)[0] = make_float4(v[0], v[1], v[2], v[3]);
    reinterpret_cast<float4*>(dst)[1] = make_float4(v[4], v[5], v[6], v[7]);
  } else {
    uint4 o;
    __nv_bfloat162 t;
    t = __floats2bfloat162_rn(v[0], v[1]);
    o.x = *reinterpret_cast<uint32_t*>(&t);
    t = __floats2bfloat162_rn(v[2], v[3]);
    o.y = *reinterpret_cast<uint32_t*>(&t);
    t = __floats2bfloat162_rn(v[4], v[5]);
    o.z = *reinterpret_cast<uint32_t*>(&t);
    t = __floats2bfloat162_rn(v[6], v[7]);
    o.w = *reinterpret_cast<uint32_t*>(&t);
    *reinterpret_cast<uint4*>(static_cast<__nv_bfloat16*>(c) +
                              static_cast<int64_t>(row) * ldc + n) = o;
  }
}
#endif

struct Plan {
  int S;        // chunks along K
  int k_chunk;  // multiple of the k-tile
};

constexpr int kMaxS = 16;
constexpr int kTileK = 64;  // TileN = 128 for every N (fixed: N-independent)

// The numerics are fixed by (K, S): K is cut into S chunks of k_chunk; each
// chunk is a fresh fp32 TMEM accumulation; chunk partials are summed left to
// right in fp32 (q0 + q1 + q2 ...). S_req is clamped to [1, kMaxS].
Plan make_plan(int K, int S_req);

// Launch: cluster below one wave of tiles, single-CTA "full" above, 2-CTA
// persistent for large M. Execution only; the result bits do not depend on it.
// out_fp32: store the fp32 sum instead of rounding it to bf16.
// tile_n: 128, or 64 for the 64-row plan (twice the tiles: small-M shapes
// with few 128-wide tiles); the chunks and their sum are unchanged.
const char* gemm(const Args& p, const Plan& pl, bool out_fp32,
                 cudaStream_t stream, int tile_n = 128);

// Large-M execution (k31_bi_gemm_2sm.cu): persistent 2-CTA 256 x BN tiles,
// same chunks and the same left-to-right chunk sum (bitwise equal). Tile
// config (BN, pairs) picked from (M, N) (or K31_BI_GEMM_2SM_CFG).
const char* gemm_2sm(const Args& p, const Plan& pl, bool out_fp32,
                     cudaStream_t stream);
// r5 (k31_bi_gemm_sw.cu): small-M swap-AB (weight = UMMA A, 128 x 16 tiles), chunk-balanced persistent launch,
// same chunks and the same left-to-right chunk sum.
// ws: fp32 [S, M, N] workspace; cnt: sw_counters(M, N) ints, all zero on entry (left all zero on exit);
// max_ctas: 0 = one CTA per SM.
const char* gemm_sw(const Args& p, const Plan& pl, bool out_fp32, float* ws, int* cnt, int max_ctas,
                    cudaStream_t stream);
int sw_counters(int M, int N);

// Its cost-model estimate at the BN it would pick, in the 64-row path's units
// (one 64x128x64 k-tile step of the 1-SM kernel).
int64_t cost_2sm(int M, int N, int kt_total);

}  // namespace k31bi
