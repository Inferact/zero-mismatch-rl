// SPDX-License-Identifier: Apache-2.0
// Large-M path of k31_bi_gemm: persistent 2-CTA (tcgen05 cta_group::2) kernel,
// 256 x BN output tile per CTA pair (each CTA owns 128 rows x BN columns),
// BN in {256, 192, 160, 128} (auto: 256 / 160 / 128) and 1 or 2 pairs per cluster (2: B multicast)
// picked per launch by a wave cost model.
//
// Same numerics as the small-M kernels (k31_bi_gemm_kernel.cu): K is cut into
// the same S chunks of k_chunk; chunk j is a fresh fp32 TMEM accumulation over
// its k-tiles in ascending order (64-wide k-tiles, four K=16 UMMAs each); the
// chunk partials are summed left to right in fp32 (run = q0; run += q1; ...),
// then rounded once. Only the UMMA shape differs (256xBN 2-SM instead of
// 64x128 / 128x128 1-SM); bigemm_test.py checks the bits are identical.
//
// CTA = 10 warps: 0-7 epilogue (warp w reads TMEM lane quadrant w % 4,
// column half w / 4), 8 TMA producer (both CTAs, each loads its
// own A rows and B half into its own smem, completing on the leader's full
// barrier), 9 MMA issuer (leader CTA only). Two BN-column TMEM accumulators
// ping-pong across chunks, so chunk j+1's MMAs overlap the drain of chunk j and
// the store of one tile overlaps the next tile's main loop.

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

constexpr int kBM = 256, kBK = 64;
constexpr int kEpiWarps = 8;  // 2 per TMEM lane quadrant
constexpr int kThreads2 = (kEpiWarps + 2) * 32;
constexpr int kProdWarp = kEpiWarps, kMmaWarp = kEpiWarps + 1;

struct P2 {
  int M, N, K;
  int64_t ldc;
  int k_chunk, S;
  int m_tiles, n_tiles;
  void* c;
  const void* add;  // optional fused bf16 addend (see Args)
  int64_t ldadd;
};

// kPairs = 2: a 4-CTA cluster of two pairs on vertically adjacent tiles (same
// n_tile) multicasts B: each CTA loads half of its B half and TMA writes it
// into both pairs, cutting L2->SM traffic (the mainloop is load-bound).
template <bool kF32Out_, int kStages_, int kBN_, int kPairs_>
struct Cfg2 {
  static constexpr int kPairs = kPairs_;
  static constexpr bool kF32Out = kF32Out_;
  static constexpr int kStages = kStages_;
  static constexpr int kBN = kBN_;
  using TiledMma = decltype(make_tiled_mma(
      SM100_MMA_F16BF16_2x1SM_SS<T, T, float, kBM, kBN, UMMA::Major::K,
                                 UMMA::Major::K>{}));
  using MmaTiler = Shape<Int<kBM>, Int<kBN>, Int<kBK>>;
  using MmaShapeA =
      decltype(partition_shape_A(TiledMma{}, Shape<Int<kBM>, Int<kBK>>{}));
  using MmaShapeB =
      decltype(partition_shape_B(TiledMma{}, Shape<Int<kBN>, Int<kBK>>{}));
  using SmemLayoutA = decltype(UMMA::tile_to_mma_shape(
      UMMA::Layout_K_SW128_Atom<T>{}, append(MmaShapeA{}, Int<kStages>{})));
  using SmemLayoutB = decltype(UMMA::tile_to_mma_shape(
      UMMA::Layout_K_SW128_Atom<T>{}, append(MmaShapeB{}, Int<kStages>{})));
  // bytes landing in ONE CTA per stage; the leader expects both CTAs' bytes
  static constexpr uint32_t kCtaStageBytes =
      (size(MmaShapeA{}) + size(MmaShapeB{})) * sizeof(T);
  struct SharedStorage {
    alignas(1024) ArrayEngine<T, cosize_v<SmemLayoutA>> A;
    alignas(1024) ArrayEngine<T, cosize_v<SmemLayoutB>> B;
    alignas(8) uint64_t full[kStages];
    alignas(8) uint64_t empty[kStages];
    alignas(8) uint64_t acc_full[2];
    alignas(8) uint64_t acc_empty[2];
    uint32_t tmem_base;
  };
  static constexpr int kSmemBytes = sizeof(SharedStorage) + 1024;
};

__device__ __forceinline__ void tmem_ld16(uint32_t taddr, uint32_t (&r)[16]) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x16.b32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]),
        "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]), "=r"(r[9]),
        "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]),
        "=r"(r[15])
      : "r"(taddr));
}

__device__ __forceinline__ void tmem_ld8(uint32_t taddr, uint32_t (&r)[8]) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x8.b32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]),
        "=r"(r[5]), "=r"(r[6]), "=r"(r[7])
      : "r"(taddr));
}

__device__ __forceinline__ uint32_t pack2_bf16_2(float a, float b) {
  __nv_bfloat162 v = __floats2bfloat162_rn(a, b);
  return *reinterpret_cast<uint32_t*>(&v);
}

// 10 warps (allocated as 12): up to 168 registers, room for run[128] when
// TMEM is read 16 columns at a time (18 warps would cap at 96 and spill).
template <class C, class TmaA, class TmaB>
__global__ void __launch_bounds__(kThreads2, 1)
    k31_bi_gemm_2sm_kernel(CUTE_GRID_CONSTANT TmaA const tma_a,
                           CUTE_GRID_CONSTANT TmaB const tma_b, P2 p) {
#if defined(__CUDA_ARCH_FEAT_SM100_ALL) || defined(__CUDA_ARCH_FEAT_SM103_ALL)
  constexpr int kStages = C::kStages;
  constexpr int kBN = C::kBN;
  constexpr int kCols = kBN / 2;  // columns per epilogue warp
  static_assert(kCols % 8 == 0, "8/16-column TMEM loads");
  using SharedStorage = typename C::SharedStorage;
  extern __shared__ char smem_raw[];
  auto& smem = *reinterpret_cast<SharedStorage*>(
      (reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));

  const int warp = __shfl_sync(0xffffffffu, threadIdx.x / 32, 0);
  const int lane = threadIdx.x % 32;
  constexpr int kPairs = C::kPairs;
  const int rank = static_cast<int>(block_rank_in_cluster());
  const int peer = rank & 1;  // 0 = pair leader (issues the MMAs)
  const int mp = rank >> 1;   // pair index inside the cluster
  const int lead_rank = rank & ~1;
  const bool leader = peer == 0;
  // cluster c walks cluster tiles t = c, c + nclu, ...; tile t is kPairs
  // vertically adjacent 256-row tiles in column n_tile, one per pair.
  const int pair = blockIdx.x / (2 * kPairs), npairs = gridDim.x / (2 * kPairs);
  const int m_groups = (p.m_tiles + kPairs - 1) / kPairs;
  const int tiles = m_groups * p.n_tiles;
  const uint16_t pair_mask = static_cast<uint16_t>(0x3u << lead_rank);
  // stage release goes to every CTA whose TMA writes into this pair's smem
  const uint16_t empty_mask = kPairs == 1 ? 0x3 : 0xF;
  const int kt_per_chunk = p.k_chunk / kBK;
  const int kt_total = (p.K + kBK - 1) / kBK;

  typename C::TiledMma tiled_mma;
  ThrMMA cta_mma = tiled_mma.get_slice(peer);
  Tensor mA = tma_a.get_tma_tensor(make_shape(p.M, p.K));
  Tensor mB = tma_b.get_tma_tensor(make_shape(p.N, p.K));
  Tensor sA =
      make_tensor(make_smem_ptr(smem.A.begin()), typename C::SmemLayoutA{});
  Tensor sB =
      make_tensor(make_smem_ptr(smem.B.begin()), typename C::SmemLayoutB{});
  Tensor tCrA = cta_mma.make_fragment_A(sA);
  Tensor tCrB = cta_mma.make_fragment_B(sB);

  using TmemAllocator = cute::TMEM::Allocator2Sm;
  TmemAllocator tmem_allocator{};
  if (warp == 0) tmem_allocator.allocate(512, &smem.tmem_base);
  if (threadIdx.x == 32) {
    prefetch_tma_descriptor(tma_a.get_tma_descriptor());
    prefetch_tma_descriptor(tma_b.get_tma_descriptor());
    for (int s = 0; s < kStages; ++s) {
      initialize_barrier(smem.full[s], 1);
      initialize_barrier(smem.empty[s], kPairs);  // one commit per pair
    }
    for (int b = 0; b < 2; ++b) {
      initialize_barrier(smem.acc_full[b], 1);
      initialize_barrier(smem.acc_empty[b], 2 * kEpiWarps);
    }
    cutlass::arch::fence_barrier_init();
  }
  cluster_sync();  // barriers and TMEM visible to the whole cluster
  const uint32_t tmem_base = smem.tmem_base;

  if (warp == kProdWarp) {
    if (elect_one_sync()) {
      int it = 0;
      uint16_t mcast_b = 0;
      if constexpr (kPairs == 2)
        mcast_b = static_cast<uint16_t>((1u << peer) | (1u << (peer + 2)));
      for (int t = pair; t < tiles; t += npairs) {
        const int m_tile = (t / p.n_tiles) * kPairs + mp, n_tile = t % p.n_tiles;
        auto coord = make_coord(m_tile, n_tile, _);
        Tensor gA = local_tile(mA, typename C::MmaTiler{}, coord,
                               Step<_1, X, _1>{});
        Tensor gB = local_tile(mB, typename C::MmaTiler{}, coord,
                               Step<X, _1, _1>{});
        Tensor tCgA = cta_mma.partition_A(gA);
        Tensor tCgB = cta_mma.partition_B(gB);
        auto [tAgA, tAsA] =
            tma_partition(tma_a, _0{}, Layout<_1>{}, group_modes<0, 3>(sA),
                          group_modes<0, 3>(tCgA));
        auto [tBgB, tBsB] = tma_partition(
            tma_b, kPairs == 2 ? mp : 0, make_layout(Int<kPairs>{}),
            group_modes<0, 3>(sB), group_modes<0, 3>(tCgB));
        for (int kt = 0; kt < kt_total; ++kt, ++it) {
          const int stage = it % kStages;
          const int ph = (it / kStages) & 1;
          wait_barrier(smem.empty[stage], ph ^ 1);
          if (leader)
            set_barrier_transaction_bytes(smem.full[stage],
                                          2 * C::kCtaStageBytes);
          copy(tma_a.with(smem.full[stage]), tAgA(_, kt), tAsA(_, stage));
          if constexpr (kPairs == 2)
            copy(tma_b.with(smem.full[stage], mcast_b), tBgB(_, kt),
                 tBsB(_, stage));
          else
            copy(tma_b.with(smem.full[stage]), tBgB(_, kt), tBsB(_, stage));
        }
      }
    }
  } else if (warp == kMmaWarp) {
    if (leader) {
      Tensor acc = tiled_mma.make_fragment_C(
          partition_shape_C(tiled_mma, Shape<Int<kBM>, Int<kBN>>{}));
      int it = 0, cg = 0;
      for (int t = pair; t < tiles; t += npairs) {
        for (int c = 0; c < p.S; ++c, ++cg) {
          const int b = cg & 1;
          wait_barrier(smem.acc_empty[b], ((cg >> 1) & 1) ^ 1);
          asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");
          acc.data() = tmem_base + b * kBN;
          tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;
          const int kt0 = c * kt_per_chunk;
          const int kt1 = min(kt0 + kt_per_chunk, kt_total);
          for (int kt = kt0; kt < kt1; ++kt, ++it) {
            const int stage = it % kStages;
            const int ph = (it / kStages) & 1;
            wait_barrier(smem.full[stage], ph);
            CUTE_UNROLL
            for (int kb = 0; kb < size<2>(tCrA); ++kb) {
              gemm(tiled_mma, tCrA(_, _, kb, stage), tCrB(_, _, kb, stage),
                   acc);
              tiled_mma.accumulate_ = UMMA::ScaleOut::One;
            }
            cutlass::arch::umma_arrive_multicast_2x1SM(&smem.empty[stage],
                                                       empty_mask);
          }
          cutlass::arch::umma_arrive_multicast_2x1SM(&smem.acc_full[b],
                                                     pair_mask);
        }
      }
    }
  } else {
    // Epilogue: row = this CTA's 128-row half, quadrant (warp % 4) of it.
    const int q = warp & 3, h = warp >> 2;  // lane quadrant, column half
    const uint32_t lane_base = static_cast<uint32_t>(q * 32) << 16;
    int cg = 0;
    for (int t = pair; t < tiles; t += npairs) {
      const int m_tile = (t / p.n_tiles) * kPairs + mp, n_tile = t % p.n_tiles;
      const int row = m_tile * kBM + peer * (kBM / 2) + q * 32 + lane;
      const int col0 = n_tile * kBN + h * kCols;
      float run[kCols];
      for (int c = 0; c < p.S; ++c, ++cg) {
        const int b = cg & 1;
        wait_barrier(smem.acc_full[b], (cg >> 1) & 1);
        asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");
        {
        CUTE_UNROLL
        for (int s = 0; s < kCols / 16; ++s) {
          uint32_t v[16];
          tmem_ld16(tmem_base + lane_base + b * kBN + h * kCols + s * 16, v);
          asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
          CUTE_UNROLL
          for (int e = 0; e < 16; ++e)
            run[s * 16 + e] = c == 0 ? __uint_as_float(v[e])
                                     : run[s * 16 + e] + __uint_as_float(v[e]);
        }
        if constexpr (kCols % 16 != 0) {
          constexpr int s0 = kCols - 8;
          uint32_t v[8];
          tmem_ld8(tmem_base + lane_base + b * kBN + h * kCols + s0, v);
          asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
          CUTE_UNROLL
          for (int e = 0; e < 8; ++e)
            run[s0 + e] = c == 0 ? __uint_as_float(v[e])
                                 : run[s0 + e] + __uint_as_float(v[e]);
        }
        }
        asm volatile("tcgen05.fence::before_thread_sync;\n" ::: "memory");
        __syncwarp();
        if (lane == 0)
          cutlass::arch::ClusterBarrier::arrive(&smem.acc_empty[b], lead_rank,
                                                1);
      }
      if (row < p.M) {
        CUTE_UNROLL
        for (int i = 0; i < kCols; i += 8) {
          const int n = col0 + i;
          if (n < p.N) {
            float v[8];
            CUTE_UNROLL
            for (int u = 0; u < 8; ++u) v[u] = run[i + u];
            store8_out<C::kF32Out>(p.c, p.ldc, row, n, v, p.add, p.ldadd);
          }
        }
      }
    }
  }
  __syncthreads();
  cluster_sync();  // no CTA leaves while a cluster peer may still signal it
  if (warp == 0) {
    tmem_allocator.release_allocation_lock();
    tmem_allocator.free(tmem_base, 512);
  }
#else
  asm volatile("trap;");
#endif
}

int num_sms2() {
  static int cached = 0;
  if (cached == 0) {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&cached, cudaDevAttrMultiProcessorCount, dev);
  }
  return cached;
}

int g_clusters4 = 0;  // set by launch2<..., 2> once queried
// (GB200: 36 four-CTA clusters co-reside, one less than SMs / 4)
int clusters4() { return g_clusters4 > 0 ? g_clusters4 : num_sms2() / 4 - 1; }


template <bool kF32Out, int kBN, int kPairs>
const char* launch2(const Args& p, const Plan& pl, cudaStream_t stream) {
  constexpr int kStageBytesCta =
      Cfg2<kF32Out, 1, kBN, kPairs>::kCtaStageBytes;  // per stage, any count
  constexpr int kStages = std::min(8, (232448 - 4096) / kStageBytesCta);
  using C = Cfg2<kF32Out, kStages, kBN, kPairs>;
  typename C::TiledMma tiled_mma;
  auto cluster_layout_vmnk = tiled_divide(
      make_layout(Shape<Int<2 * kPairs>, _1, _1>{}),
      make_tile(typename C::TiledMma::AtomThrID{}));
  Tensor mA = make_tensor(make_gmem_ptr(static_cast<const T*>(p.a)),
                          make_shape(p.M, p.K), make_stride(p.lda, _1{}));
  Tensor mB = make_tensor(make_gmem_ptr(static_cast<const T*>(p.b)),
                          make_shape(p.N, p.K), make_stride(p.ldb, _1{}));
  auto tma_a = make_tma_atom_A_sm100(
      SM100_TMA_2SM_LOAD{}, mA, typename C::SmemLayoutA{}(_, _, _, _0{}),
      typename C::MmaTiler{}, tiled_mma, cluster_layout_vmnk);
  using TmaOpB = std::conditional_t<kPairs == 2, SM100_TMA_2SM_LOAD_MULTICAST,
                                    SM100_TMA_2SM_LOAD>;
  auto tma_b = make_tma_atom_B_sm100(
      TmaOpB{}, mB, typename C::SmemLayoutB{}(_, _, _, _0{}),
      typename C::MmaTiler{}, tiled_mma, cluster_layout_vmnk);
  auto* kernel =
      &k31_bi_gemm_2sm_kernel<C, decltype(tma_a), decltype(tma_b)>;
  static bool attr_set = false;
  if (!attr_set) {
    if (cudaFuncSetAttribute(kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             C::kSmemBytes) != cudaSuccess)
      return "k31_bi_gemm_2sm: cudaFuncSetAttribute failed";
    attr_set = true;
  }
  P2 kp{p.M, p.N, p.K, p.ldc, pl.k_chunk, pl.S,
        (p.M + kBM - 1) / kBM, (p.N + kBN - 1) / kBN, p.c, p.add, p.ldadd};
  const int tiles = ((kp.m_tiles + kPairs - 1) / kPairs) * kp.n_tiles;
  cudaLaunchConfig_t cfg{};
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim = {static_cast<unsigned>(2 * kPairs), 1, 1};
  cfg.blockDim = dim3(kThreads2);
  cfg.dynamicSmemBytes = C::kSmemBytes;
  cfg.stream = stream;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  // Persistent grid = clusters the device co-schedules (GPC packing can fit
  // fewer 4-CTA clusters than SMs / 4; a second wave would double the time).
  static int cap = 0;
  if (cap == 0) {
    cfg.gridDim = dim3(2 * kPairs, 1, 1);
    int n = 0;
    cap = cudaOccupancyMaxActiveClusters(&n, kernel, &cfg) == cudaSuccess && n > 0
              ? n
              : num_sms2() / (2 * kPairs);
    cudaGetLastError();
  }
  if constexpr (kPairs == 2) g_clusters4 = cap;
  const int npairs = std::max(1, std::min(tiles, cap));  // clusters
  cfg.gridDim = dim3(2 * kPairs * npairs, 1, 1);
  return cudaLaunchKernelEx(&cfg, kernel, tma_a, tma_b, kp) == cudaSuccess
             ? nullptr
             : "k31_bi_gemm_2sm: launch failed";
}

// Wave cost model (execution only: every output element sums the same
// products in the same order for any tile shape). Per-SM work of one 2-CTA
// k-step (128 x BN x 64) in 64-row k-tile units is f * BN / 128: f = 1.25 for
// one pair per cluster (GB200: 0.21 us per 128x128x64 vs 0.165 us per 64-row
// k-tile) and 1.125 with B multicast (GB200 M=8192, N=4224 / 2880: 0.84 /
// 0.95x the one-pair time at BN = 160); plus the 64-row model's fixed cost
// per wave. BN = 192 / 224 run but are never picked (slowest in every sweep).
int64_t cost_cfg(int M, int N, int kt_total, int bn, int pairs) {
  const int clusters = std::max(1, pairs == 1 ? num_sms2() / 2 : clusters4());
  const int m_tiles = (M + kBM - 1) / kBM;
  const int tiles = ((m_tiles + pairs - 1) / pairs) * ((N + bn - 1) / bn);
  const int64_t waves = (tiles + clusters - 1) / clusters;
  const int f8 = pairs == 2 ? 9 : 10;  // f in eighths
  return waves * (static_cast<int64_t>(kt_total) * bn * f8 / 1024 + 28);
}

// Picks (BN, pairs); cfg_force (env K31_BI_GEMM_2SM_CFG = pairs * 1000 + BN)
// is for timing. Ties keep the narrower tile and one pair.
int pick_cfg(int M, int N, int kt_total, int64_t* cost) {
  static const int cfg_force = [] {
    const char* e = std::getenv("K31_BI_GEMM_2SM_CFG");
    return e != nullptr ? std::atoi(e) : 0;
  }();
  int best = 1128;
  int64_t best_cost = -1;
  for (int pairs : {1, 2})
    for (int bn : {128, 160, 256}) {
      const int64_t c = cost_cfg(M, N, kt_total, bn, pairs);
      if (best_cost < 0 || c < best_cost) {
        best_cost = c;
        best = pairs * 1000 + bn;
      }
    }
  if (cfg_force > 0) {
    best = cfg_force;
    best_cost = cost_cfg(M, N, kt_total, cfg_force % 1000, cfg_force / 1000);
  }
  if (cost != nullptr) *cost = best_cost;
  return best;
}

template <bool kF32Out>
const char* launch2_cfg(const Args& p, const Plan& pl, cudaStream_t stream) {
  const int cfg = pick_cfg(p.M, p.N, (p.K + kBK - 1) / kBK, nullptr);
  switch (cfg) {
    case 1128:
      return launch2<kF32Out, 128, 1>(p, pl, stream);
    case 1160:
      return launch2<kF32Out, 160, 1>(p, pl, stream);
    case 1192:
      return launch2<kF32Out, 192, 1>(p, pl, stream);
    case 1256:
      return launch2<kF32Out, 256, 1>(p, pl, stream);
    case 2128:
      return launch2<kF32Out, 128, 2>(p, pl, stream);
    case 2160:
      return launch2<kF32Out, 160, 2>(p, pl, stream);
    case 2192:
      return launch2<kF32Out, 192, 2>(p, pl, stream);
    case 2256:
      return launch2<kF32Out, 256, 2>(p, pl, stream);
    default:
      return "k31_bi_gemm_2sm: unknown tile config";
  }
}

}  // namespace

int64_t cost_2sm(int M, int N, int kt_total) {
  int64_t c = 0;
  pick_cfg(M, N, kt_total, &c);
  return c;
}

const char* gemm_2sm(const Args& p, const Plan& pl, bool out_fp32,
                     cudaStream_t stream) {
  return out_fp32 ? launch2_cfg<true>(p, pl, stream)
                  : launch2_cfg<false>(p, pl, stream);
}

}  // namespace k31bi
