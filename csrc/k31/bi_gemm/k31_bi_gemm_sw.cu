// SPDX-License-Identifier: Apache-2.0
// K31 bi-gemm r5: small-M swap-AB, chunk-balanced persistent launch of the split-K GEMM (execution only).
//
// Numerics are the r4 plan: K is cut into S chunks of k_chunk (unchanged); every chunk is ONE fp32 tcgen05
// accumulation from zero over k ascending in steps of 16 (UMMA K = 16, bf16 x bf16 -> fp32); the chunk partials
// are summed left to right in fp32 (run = q0; run += q1; ...), the optional bf16 addend is added in fp32, then
// one rounding. Only the operand roles change: the weight tile is the UMMA A operand (128 weight rows = output
// columns) and the activation tile the B operand (16 tokens), so C^T is accumulated. Each output element is
// still the same K = 16 dot-product steps with the same accumulator chain; gated bitwise equal to the r4 kernel
// (and the r4 gate already has the cuBLAS small-M kernels, which use either role, bitwise equal to it at S = 1).
//
// Why: at decode M (1..16 tokens) the r4 64-row tile spends a third of every pipeline stage on a mostly-empty
// activation tile and has only N / 128 output tiles (56 .. 96 on 152 SMs for N = 7168 .. 12288). Here a stage is
// 16 KB of weight + 2 KB of activations (12 stages in flight) and the (tile, chunk) units are spread evenly over
// min(units, SMs) persistent CTAs; chunk partials go to an fp32 workspace ws[chunk][m][n], the CTA completing the
// last chunk of a tile sums ws[0..S-1] in order and stores (self-resetting per-tile counters, CUDA-graph safe).
// CTA = 6 warps: 0-3 epilogue, 4 TMA producer, 5 MMA issuer.

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "cute/tensor.hpp"
#include "cute/arch/cluster_sm90.hpp"
#include "cute/arch/tmem_allocator_sm100.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/numeric_types.h"

#include "k31_bi_gemm.cuh"

namespace k31bi {

namespace {

using namespace cute;

using T = cutlass::bfloat16_t;
constexpr int kSwW = 128;  // weight rows per tile (UMMA M)
constexpr int kSwT = 16;   // tokens per tile (UMMA N)
constexpr int kSwThreads = 192;
constexpr int kSwEpi = 128;
constexpr int kSwBufs = 8;  // TMEM accumulators in flight (128 lanes x 16 fp32 columns each)

struct SwParams {
  int M, N, K;
  int64_t ldc;
  int k_chunk, S, m_tiles, units, ctas;
  void* c;
  const void* add;
  int64_t ldadd;
  float* ws;
  int* cnt;
};

template <bool kF32Out, int kStages>
struct SwCfg {
  static constexpr bool kF32Out_ = kF32Out;
  static constexpr int kStages_ = kStages;
  using TiledMma = decltype(make_tiled_mma(
      SM100_MMA_F16BF16_SS<T, T, float, kSwW, kSwT, UMMA::Major::K, UMMA::Major::K>{}));
  using MmaTiler = Shape<Int<kSwW>, Int<kSwT>, Int<kTileK>>;
  using MmaShapeA = decltype(partition_shape_A(TiledMma{}, Shape<Int<kSwW>, Int<kTileK>>{}));
  using MmaShapeB = decltype(partition_shape_B(TiledMma{}, Shape<Int<kSwT>, Int<kTileK>>{}));
  using Atom = UMMA::Layout_K_SW128_Atom<T>;
  using SmemLayoutA = decltype(UMMA::tile_to_mma_shape(Atom{}, append(MmaShapeA{}, Int<kStages>{})));
  using SmemLayoutB =
      decltype(UMMA::tile_to_mma_shape(Atom{}, append(MmaShapeB{}, Int<kStages>{}), Step<_1, _2, _3>{}));
  static constexpr uint32_t kTmaBytes = (kSwW * kTileK + kSwT * kTileK) * sizeof(T);
  struct SharedStorage {
    alignas(1024) ArrayEngine<T, cosize_v<SmemLayoutA>> A;
    alignas(1024) ArrayEngine<T, cosize_v<SmemLayoutB>> B;
    alignas(8) uint64_t full[kStages];
    alignas(8) uint64_t empty[kStages];
    alignas(8) uint64_t acc_full[kSwBufs];
    alignas(8) uint64_t acc_empty[kSwBufs];
    uint32_t tmem_base;
    int last;
  };
  static constexpr int kSmemBytes = sizeof(SharedStorage) + 1024;
};

__device__ __forceinline__ void sw_epi_bar() { asm volatile("bar.sync 1, %0;\n" ::"n"(kSwEpi) : "memory"); }

template <bool kF32Out>
__device__ __forceinline__ void store1_out(void* c, int64_t ldc, int m, int n, float v, const void* add,
                                           int64_t ldadd) {
  if (add != nullptr)  // same fp32 add as store8_out, then the single rounding
    v = __fadd_rn(v, __bfloat162float(static_cast<const __nv_bfloat16*>(add)[static_cast<int64_t>(m) * ldadd + n]));
  if constexpr (kF32Out)
    static_cast<float*>(c)[static_cast<int64_t>(m) * ldc + n] = v;
  else
    static_cast<__nv_bfloat16*>(c)[static_cast<int64_t>(m) * ldc + n] = __float2bfloat16_rn(v);
}

template <class Cfg, class TmaW, class TmaX>
__global__ void __launch_bounds__(kSwThreads, 1)
    k31_bi_gemm_sw_kernel(CUTE_GRID_CONSTANT TmaW const tma_w, CUTE_GRID_CONSTANT TmaX const tma_x,
                          SwParams p) {
#if defined(__CUDA_ARCH_FEAT_SM100_ALL) || defined(__CUDA_ARCH_FEAT_SM103_ALL)
  constexpr int kStages = Cfg::kStages_;
  constexpr bool kF32Out = Cfg::kF32Out_;
  using SharedStorage = typename Cfg::SharedStorage;
  extern __shared__ char smem_raw[];
  auto& smem = *reinterpret_cast<SharedStorage*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));

  const int warp = __shfl_sync(0xffffffffu, threadIdx.x / 32, 0);
  const int u0 = static_cast<int>(static_cast<int64_t>(blockIdx.x) * p.units / p.ctas);
  const int u1 = static_cast<int>(static_cast<int64_t>(blockIdx.x + 1) * p.units / p.ctas);
  const int kt_per_chunk = p.k_chunk / kTileK;
  const int kt_total = (p.K + kTileK - 1) / kTileK;

  Tensor mW = tma_w.get_tma_tensor(make_shape(p.N, p.K));
  Tensor mX = tma_x.get_tma_tensor(make_shape(p.M, p.K));
  Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), typename Cfg::SmemLayoutA{});
  Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), typename Cfg::SmemLayoutB{});
  typename Cfg::TiledMma tiled_mma;
  ThrMMA cta_mma = tiled_mma.get_slice(0);
  Tensor tCrA = cta_mma.make_fragment_A(sA);
  Tensor tCrB = cta_mma.make_fragment_B(sB);
  Tensor tCtAcc = tiled_mma.make_fragment_C(
      append(partition_shape_C(tiled_mma, Shape<Int<kSwW>, Int<kSwT>>{}), Int<kSwBufs>{}));
  constexpr int kTmemCols = kSwBufs * kSwT;  // 128

  using TmemAllocator = cute::TMEM::Allocator1Sm;
  TmemAllocator tmem_allocator{};
  if (warp == 4) tmem_allocator.allocate(kTmemCols, &smem.tmem_base);
  if (threadIdx.x == 0) {
    prefetch_tma_descriptor(tma_w.get_tma_descriptor());
    prefetch_tma_descriptor(tma_x.get_tma_descriptor());
    for (int s = 0; s < kStages; ++s) {
      initialize_barrier(smem.full[s], 1);
      initialize_barrier(smem.empty[s], 1);
    }
    for (int b = 0; b < kSwBufs; ++b) {
      initialize_barrier(smem.acc_full[b], 1);
      initialize_barrier(smem.acc_empty[b], kSwEpi);
    }
    cutlass::arch::fence_barrier_init();
  }
  __syncthreads();
  asm volatile("griddepcontrol.launch_dependents;");  // our dependents wait for our completion themselves
  tCtAcc.data() = smem.tmem_base;

  if (warp == 4) {
    if (elect_one_sync()) {
      // PDL: the weights never depend on the previous kernel, so the first kStages weight tiles are requested
      // before the grid-dependency wait; activations (and every write) only after it.
      int pre = 0;
      for (int u = u0; u < u1 && pre < kStages; ++u) {
        const int t = u / p.S, c = u - t * p.S;
        const int nt = t / p.m_tiles, mt = t - nt * p.m_tiles;
        Tensor gA = local_tile(mW, typename Cfg::MmaTiler{}, make_coord(nt, mt, _), Step<_1, X, _1>{});
        Tensor tCgA = cta_mma.partition_A(gA);
        auto [tAgA, tAsA] =
            tma_partition(tma_w, _0{}, Layout<_1>{}, group_modes<0, 3>(sA), group_modes<0, 3>(tCgA));
        const int kt0 = c * kt_per_chunk;
        const int kt1 = min(kt0 + kt_per_chunk, kt_total);
        for (int kt = kt0; kt < kt1 && pre < kStages; ++kt, ++pre) {
          set_barrier_transaction_bytes(smem.full[pre], Cfg::kTmaBytes);  // fresh stage: no empty wait
          copy(tma_w.with(smem.full[pre]), tAgA(_, kt), tAsA(_, pre));
        }
      }
      asm volatile("griddepcontrol.wait;" ::: "memory");
      int it = 0;
      for (int u = u0; u < u1; ++u) {
        const int t = u / p.S, c = u - t * p.S;
        const int nt = t / p.m_tiles, mt = t - nt * p.m_tiles;
        auto coord = make_coord(nt, mt, _);
        Tensor gA = local_tile(mW, typename Cfg::MmaTiler{}, coord, Step<_1, X, _1>{});
        Tensor gB = local_tile(mX, typename Cfg::MmaTiler{}, coord, Step<X, _1, _1>{});
        Tensor tCgA = cta_mma.partition_A(gA);
        Tensor tCgB = cta_mma.partition_B(gB);
        auto [tAgA, tAsA] =
            tma_partition(tma_w, _0{}, Layout<_1>{}, group_modes<0, 3>(sA), group_modes<0, 3>(tCgA));
        auto [tBgB, tBsB] =
            tma_partition(tma_x, _0{}, Layout<_1>{}, group_modes<0, 3>(sB), group_modes<0, 3>(tCgB));
        const int kt0 = c * kt_per_chunk;
        const int kt1 = min(kt0 + kt_per_chunk, kt_total);
        for (int kt = kt0; kt < kt1; ++kt, ++it) {
          const int stage = it % kStages;
          const int ph = (it / kStages) & 1;
          if (it >= pre) {
            wait_barrier(smem.empty[stage], ph ^ 1);
            set_barrier_transaction_bytes(smem.full[stage], Cfg::kTmaBytes);
            copy(tma_w.with(smem.full[stage]), tAgA(_, kt), tAsA(_, stage));
          }
          copy(tma_x.with(smem.full[stage]), tBgB(_, kt), tBsB(_, stage));
        }
      }
    }
    __syncwarp();
  } else if (warp == 5) {
    int it = 0, buf = 0, aph = 0;
    for (int u = u0; u < u1; ++u) {
      const int c = u % p.S;
      wait_barrier(smem.acc_empty[buf], aph ^ 1);
      Tensor acc = tCtAcc(_, _, _, buf);
      tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;
      const int kt0 = c * kt_per_chunk;
      const int kt1 = min(kt0 + kt_per_chunk, kt_total);
      for (int kt = kt0; kt < kt1; ++kt, ++it) {
        const int stage = it % kStages;
        const int ph = (it / kStages) & 1;
        wait_barrier(smem.full[stage], ph);
        CUTE_UNROLL
        for (int kb = 0; kb < size<2>(tCrA); ++kb) {
          gemm(tiled_mma, tCrA(_, _, kb, stage), tCrB(_, _, kb, stage), acc);
          tiled_mma.accumulate_ = UMMA::ScaleOut::One;
        }
        cutlass::arch::umma_arrive(&smem.empty[stage]);
      }
      cutlass::arch::umma_arrive(&smem.acc_full[buf]);
      if (++buf == kSwBufs) {
        buf = 0;
        aph ^= 1;
      }
    }
    __syncwarp();
  } else {
    TiledCopy t2r = make_tmem_copy(SM100_TMEM_LOAD_32dp32b16x{}, tCtAcc(_, _, _, _0{}));
    ThrCopy thr = t2r.get_slice(threadIdx.x);
    Tensor cC = make_identity_tensor(Shape<Int<kSwW>, Int<kSwT>>{});
    Tensor tDcC = thr.partition_D(cta_mma.partition_C(cC));
    Tensor rAcc = make_tensor<float>(shape(tDcC));
    static_assert(size(decltype(tDcC){}) == kSwT, "one weight row (16 tokens) per epilogue thread");
    const int64_t plane = static_cast<int64_t>(p.M) * p.N;
    int i = 0;
    for (int u = u0; u < u1; ++u, ++i) {
      const int t = u / p.S, c = u - t * p.S;
      const int nt = t / p.m_tiles, mt = t - nt * p.m_tiles;
      const int b0 = i % kSwBufs;
      wait_barrier(smem.acc_full[b0], (i / kSwBufs) & 1);
      copy(t2r, thr.partition_S(tCtAcc(_, _, _, b0)), rAcc);
      cutlass::arch::fence_view_async_tmem_load();
      arrive_barrier(smem.acc_empty[b0]);
      CUTE_UNROLL
      for (int e = 0; e < kSwT; ++e) {
        const int n = nt * kSwW + get<0>(tDcC(e));
        const int m = mt * kSwT + get<1>(tDcC(e));
        if (n < p.N && m < p.M) __stcg(p.ws + c * plane + static_cast<int64_t>(m) * p.N + n, rAcc(e));
      }
      __threadfence();
      sw_epi_bar();
      if (threadIdx.x == 0) smem.last = atomicAdd(p.cnt + t, 1) == p.S - 1;
      sw_epi_bar();
      if (smem.last) {  // every chunk partial of tile t is in ws: left-to-right sum, one rounding
        __threadfence();
        const int n = nt * kSwW + threadIdx.x;
        const int m_end = min(p.M, (mt + 1) * kSwT);
        if (n < p.N) {
          for (int m = mt * kSwT; m < m_end; ++m) {
            const float* src = p.ws + static_cast<int64_t>(m) * p.N + n;
            float q[kMaxS];  // all S partial loads in flight first, then the left-to-right sum
            CUTE_UNROLL
            for (int j = 0; j < kMaxS; ++j) q[j] = j < p.S ? __ldcg(src + j * plane) : 0.f;
            float v = q[0];
            CUTE_UNROLL
            for (int j = 1; j < kMaxS; ++j)
              if (j < p.S) v = v + q[j];
            store1_out<kF32Out>(p.c, p.ldc, m, n, v, p.add, p.ldadd);
          }
        }
        if (threadIdx.x == 0) p.cnt[t] = 0;  // all-zero again for the next launch
      }
    }
  }
  __syncthreads();
  if (warp == 4) {
    tmem_allocator.release_allocation_lock();
    tmem_allocator.free(smem.tmem_base, kTmemCols);
  }
#else
  asm volatile("trap;");
#endif
}

template <bool kF32Out>
const char* launch_sw(const Args& p, const Plan& pl, float* ws, int* cnt, int max_ctas, cudaStream_t stream) {
  constexpr int kStageBytes = (kSwW * kTileK + kSwT * kTileK) * sizeof(T);
  constexpr int kStages = std::min(16, (232448 - 3072 - 1024) / kStageBytes);
  using C = SwCfg<kF32Out, kStages>;
  Tensor mW = make_tensor(make_gmem_ptr(static_cast<const T*>(p.b)), make_shape(p.N, p.K),
                          make_stride(p.ldb, _1{}));
  auto tma_w = make_tma_atom(SM90_TMA_LOAD{}, mW, typename C::SmemLayoutA{}(_, _, _, _0{}),
                             Shape<Int<kSwW>, Int<kTileK>>{});
  Tensor mX = make_tensor(make_gmem_ptr(static_cast<const T*>(p.a)), make_shape(p.M, p.K),
                          make_stride(p.lda, _1{}));
  auto tma_x = make_tma_atom(SM90_TMA_LOAD{}, mX, typename C::SmemLayoutB{}(_, _, _, _0{}),
                             Shape<Int<kSwT>, Int<kTileK>>{});
  auto* kernel = &k31_bi_gemm_sw_kernel<C, decltype(tma_w), decltype(tma_x)>;
  static bool attr_set = false;
  if (!attr_set) {
    if (cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, C::kSmemBytes) != cudaSuccess)
      return "k31_bi_gemm_sw: cudaFuncSetAttribute failed";
    attr_set = true;
  }
  int sms = 0, dev = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  const int m_tiles = (p.M + kSwT - 1) / kSwT;
  const int n_tiles = (p.N + kSwW - 1) / kSwW;
  SwParams kp{p.M, p.N, p.K, p.ldc, pl.k_chunk, pl.S, m_tiles, n_tiles * m_tiles * pl.S, 0,
              p.c, p.add, p.ldadd, ws, cnt};
  kp.ctas = std::min(kp.units, max_ctas > 0 ? std::min(max_ctas, sms) : sms);
  static const bool pdl = [] {
    const char* e = std::getenv("K31_BI_GEMM_PDL");
    return e == nullptr || std::atoi(e) != 0;
  }();
  cudaLaunchConfig_t cfg{};
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.gridDim = dim3(kp.ctas);
  cfg.blockDim = dim3(kSwThreads);
  cfg.dynamicSmemBytes = C::kSmemBytes;
  cfg.stream = stream;
  cfg.attrs = attr;
  cfg.numAttrs = pdl ? 1 : 0;
  return cudaLaunchKernelEx(&cfg, kernel, tma_w, tma_x, kp) == cudaSuccess ? nullptr
                                                                           : "k31_bi_gemm_sw: launch failed";
}

}  // namespace

const char* gemm_sw(const Args& p, const Plan& pl, bool out_fp32, float* ws, int* cnt, int max_ctas,
                    cudaStream_t stream) {
  return out_fp32 ? launch_sw<true>(p, pl, ws, cnt, max_ctas, stream)
                  : launch_sw<false>(p, pl, ws, cnt, max_ctas, stream);
}

int sw_counters(int M, int N) { return ((M + kSwT - 1) / kSwT) * ((N + kSwW - 1) / kSwW); }

}  // namespace k31bi
