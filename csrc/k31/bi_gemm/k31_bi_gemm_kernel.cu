// SPDX-License-Identifier: Apache-2.0
// K31 port of the chunked split-K tcgen05 GEMM (vLLM
// csrc/batch_invariant/bi_gemm_sm100_splitk.cu). Changes vs the original:
//   - bf16 inputs, K-major weight ([N, K] row-major) only;
//   - TileN fixed at 128 (k-tile 64) for every N, so the split plan is a
//     function of (K, S) only;
//   - optional fp32 output (same fp32 sum, no bf16 rounding) for the router;
//   - no bias;
//   - explicit S from the caller (per-(K, N) table lives in Python).
//
// Numerics: K is cut into S chunks of k_chunk; chunk j is accumulated from
// zero in TMEM by the same UMMA sequence whatever M is; the chunk partials
// are added in fp32 left to right (run = q0; run += q1; ...), then rounded
// once. Launch modes reproducing that sum exactly:
//   cluster: G = ceil(S / cpc) CTAs per output tile, cpc (1 or 2) chunks
//            each; every chunk's fp32 partial rows are pushed separately over
//            DSMEM (st.async) into slot j of the CTA owning each row slice,
//            which adds slots 0..S-1 in order;
//   full:    one CTA walks every chunk, keeping the running sum in registers
//            (64- or 128-row tile; the 2-CTA 256-row tile is in
//            k31_bi_gemm_2sm.cu).
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
constexpr int kTileM = 64;
constexpr int kTileN = 128;
constexpr int kThreads = 192;
constexpr int kMaxCpc = 2;  // chunks per CTA in cluster mode (each sent as is)
constexpr int kEpiThreads = 128;

struct KParams {
  int M, N, K;
  int64_t ldc;
  int k_chunk, S, cpc;  // cpc == S is full mode
  int rp;               // cluster mode: tile rows owned by each rank
  void* c;
  const void* add;  // optional fused bf16 addend (see Args)
  int64_t ldadd;
};

template <bool kF32Out, int kStages, int kTileM, int kRecvR, int kTileN, int kMinBlocks>
struct Cfg {
  static constexpr int kTileN_ = kTileN;
  static constexpr int kMinBlocks_ = kMinBlocks;
  static constexpr bool kF32Out_ = kF32Out;
  static constexpr int kStages_ = kStages;
  static constexpr int kTileM_ = kTileM;
  using TiledMma = decltype(make_tiled_mma(
      SM100_MMA_F16BF16_SS<T, T, float, kTileM, kTileN, UMMA::Major::K,
                           UMMA::Major::K>{}));
  using MmaTiler = Shape<Int<kTileM>, Int<kTileN>, Int<kTileK>>;
  using MmaShapeA = decltype(partition_shape_A(
      TiledMma{}, Shape<Int<kTileM>, Int<kTileK>>{}));
  using MmaShapeB = decltype(partition_shape_B(
      TiledMma{}, Shape<Int<kTileN>, Int<kTileK>>{}));
  using AtomA = UMMA::Layout_K_SW128_Atom<T>;
  using AtomB = UMMA::Layout_K_SW128_Atom<T>;
  using SmemLayoutA = decltype(UMMA::tile_to_mma_shape(
      AtomA{}, append(MmaShapeA{}, Int<kStages>{})));
  using SmemLayoutB = decltype(UMMA::tile_to_mma_shape(
      AtomB{}, append(MmaShapeB{}, Int<kStages>{}), Step<_1, _2, _3>{}));
  static constexpr uint32_t kTmaBytes =
      (kTileM * kTileK + kTileN * kTileK) * sizeof(T);
  static constexpr int kTmemBufs = (512 / kTileN) * (128 / kTileM);
  static constexpr int kPartStride = kTileN + 4;
  static constexpr int kRecvRows = kRecvR;
  static_assert(kMaxCpc <= kTmemBufs, "one TMEM accumulator per chunk");
  struct SharedStorage {
    alignas(1024) ArrayEngine<T, cosize_v<SmemLayoutA>> A;
    alignas(1024) ArrayEngine<T, cosize_v<SmemLayoutB>> B;
    alignas(16) float recv[kRecvRows * kPartStride];
    alignas(8) uint64_t full[kStages];
    alignas(8) uint64_t empty[kStages];
    alignas(8) uint64_t acc_full[kTmemBufs];
    alignas(8) uint64_t acc_empty[kTmemBufs];
    alignas(8) uint64_t recv_full;
    uint32_t tmem_base;
  };
  static constexpr int kSmemBytes = sizeof(SharedStorage) + 1024;
};

__device__ __forceinline__ void dsmem_st_async(uint32_t dst, float4 v,
                                               uint32_t bar) {
  asm volatile(
      "st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32"
      " [%0], {%1, %2, %3, %4}, [%5];\n" ::"r"(dst),
      "r"(__float_as_uint(v.x)), "r"(__float_as_uint(v.y)),
      "r"(__float_as_uint(v.z)), "r"(__float_as_uint(v.w)), "r"(bar)
      : "memory");
}

__device__ __forceinline__ uint32_t pack2_bf16(float a, float b) {
  __nv_bfloat162 v = __floats2bfloat162_rn(a, b);
  return *reinterpret_cast<uint32_t*>(&v);
}

template <class Cfg, class TmaA, class TmaB>
__global__ void __launch_bounds__(kThreads, Cfg::kMinBlocks_)
    k31_bi_gemm_kernel(CUTE_GRID_CONSTANT TmaA const tma_a,
                       CUTE_GRID_CONSTANT TmaB const tma_b, KParams p) {
#if defined(__CUDA_ARCH_FEAT_SM100_ALL) || defined(__CUDA_ARCH_FEAT_SM103_ALL)
  constexpr int kStages = Cfg::kStages_;
  constexpr bool kF32Out = Cfg::kF32Out_;
  constexpr int kTileM = Cfg::kTileM_;
  constexpr int kTileN = Cfg::kTileN_;
  using SharedStorage = typename Cfg::SharedStorage;

  extern __shared__ char smem_raw[];
  auto& smem = *reinterpret_cast<SharedStorage*>(
      (reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));

  // shfl makes the warp index provably uniform (else ptxas wraps every
  // tcgen05.mma in a WARPSYNC.COLLECTIVE/ELECT block).
  const int warp = __shfl_sync(0xffffffffu, threadIdx.x / 32, 0);
  const int m_tile = blockIdx.x, n_tile = blockIdx.y, z = blockIdx.z;
  const int chunk0 = z * p.cpc;
  const int chunk1 = min(chunk0 + p.cpc, p.S);
  const int kt_per_chunk = p.k_chunk / kTileK;
  const int kt_total = (p.K + kTileK - 1) / kTileK;

  Tensor mA = tma_a.get_tma_tensor(make_shape(p.M, p.K));
  Tensor mB = tma_b.get_tma_tensor(make_shape(p.N, p.K));
  auto coord = make_coord(m_tile, n_tile, _);
  Tensor gA =
      local_tile(mA, typename Cfg::MmaTiler{}, coord, Step<_1, X, _1>{});
  Tensor gB =
      local_tile(mB, typename Cfg::MmaTiler{}, coord, Step<X, _1, _1>{});
  Tensor sA =
      make_tensor(make_smem_ptr(smem.A.begin()), typename Cfg::SmemLayoutA{});
  Tensor sB =
      make_tensor(make_smem_ptr(smem.B.begin()), typename Cfg::SmemLayoutB{});

  typename Cfg::TiledMma tiled_mma;
  ThrMMA cta_mma = tiled_mma.get_slice(0);
  Tensor tCgA = cta_mma.partition_A(gA);
  Tensor tCgB = cta_mma.partition_B(gB);
  Tensor tCrA = cta_mma.make_fragment_A(sA);
  Tensor tCrB = cta_mma.make_fragment_B(sB);
  Tensor tCtAcc = tiled_mma.make_fragment_C(
      append(partition_shape_C(tiled_mma, Shape<Int<kTileM>, Int<kTileN>>{}),
             Int<Cfg::kTmemBufs>{}));
  const bool full = p.cpc == p.S;
  // full mode: chunk i accumulates in TMEM buffer i % nbuf (4 deep).
  const int nbuf = full ? std::min(4, Cfg::kTmemBufs) : p.cpc;
  const int m_valid = min(kTileM, p.M - m_tile * kTileM);
  const int rank = full ? 0 : static_cast<int>(block_rank_in_cluster());
  const int rows_me = full ? 0 : max(0, min(p.rp, m_valid - rank * p.rp));
  constexpr int kBufsPerCol = 128 / kTileM;
  int tmem_cols = 32;
  while (tmem_cols < (nbuf + kBufsPerCol - 1) / kBufsPerCol * kTileN)
    tmem_cols *= 2;

  auto [tAgA, tAsA] =
      tma_partition(tma_a, _0{}, Layout<_1>{}, group_modes<0, 3>(sA),
                    group_modes<0, 3>(tCgA));
  auto [tBgB, tBsB] =
      tma_partition(tma_b, _0{}, Layout<_1>{}, group_modes<0, 3>(sB),
                    group_modes<0, 3>(tCgB));

  using TmemAllocator = cute::TMEM::Allocator1Sm;
  TmemAllocator tmem_allocator{};
  if (warp == 4) {
    tmem_allocator.allocate(tmem_cols, &smem.tmem_base);
  }
  if (threadIdx.x == 0) {
    prefetch_tma_descriptor(tma_a.get_tma_descriptor());
    prefetch_tma_descriptor(tma_b.get_tma_descriptor());
    for (int s = 0; s < kStages; ++s) {
      initialize_barrier(smem.full[s], 1);
      initialize_barrier(smem.empty[s], 1);
    }
    for (int b = 0; b < Cfg::kTmemBufs; ++b) {
      initialize_barrier(smem.acc_full[b], 1);
      initialize_barrier(smem.acc_empty[b], kEpiThreads);
    }
    initialize_barrier(smem.recv_full, 1);
    cutlass::arch::fence_barrier_init();
    if (!full) {
      cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(
          &smem.recv_full, p.S * rows_me * kTileN * sizeof(float));
    }
  }
  __syncthreads();
  asm volatile("griddepcontrol.launch_dependents;");  // dependents wait for our completion themselves
  if (!full) cluster_arrive();
  tCtAcc.data() = smem.tmem_base;

  if (warp == 4) {
    if (elect_one_sync()) {
      int pre = 0;  // PDL: weight (B) tiles of the first stages before the grid-dependency wait
      for (int c = chunk0; c < chunk1 && pre < kStages; ++c) {
        const int kt0 = c * kt_per_chunk;
        const int kt1 = min(kt0 + kt_per_chunk, kt_total);
        for (int kt = kt0; kt < kt1 && pre < kStages; ++kt, ++pre) {
          set_barrier_transaction_bytes(smem.full[pre], Cfg::kTmaBytes);  // fresh stage: no empty wait
          copy(tma_b.with(smem.full[pre]), tBgB(_, kt), tBsB(_, pre));
        }
      }
      asm volatile("griddepcontrol.wait;" ::: "memory");
      int it = 0;
      for (int c = chunk0; c < chunk1; ++c) {
        const int kt0 = c * kt_per_chunk;
        const int kt1 = min(kt0 + kt_per_chunk, kt_total);
        for (int kt = kt0; kt < kt1; ++kt, ++it) {
          const int stage = it % kStages;
          const int ph = (it / kStages) & 1;
          if (it >= pre) {
            wait_barrier(smem.empty[stage], ph ^ 1);
            set_barrier_transaction_bytes(smem.full[stage], Cfg::kTmaBytes);
            copy(tma_b.with(smem.full[stage]), tBgB(_, kt), tBsB(_, stage));
          }
          copy(tma_a.with(smem.full[stage]), tAgA(_, kt), tAsA(_, stage));
        }
      }
    }
    __syncwarp();
    if (!full) cluster_wait();
  } else if (warp == 5) {
    {
      int it = 0;
      int buf = 0, aph = 0;
      for (int c = chunk0; c < chunk1; ++c) {
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
        if (++buf == nbuf) {
          buf = 0;
          aph ^= 1;
        }
      }
    }
    __syncwarp();
    if (!full) cluster_wait();
  } else {
    using TmemLoad = std::conditional_t<kTileM == 128, SM100_TMEM_LOAD_32dp32b32x,
                                        SM100_TMEM_LOAD_16dp32b32x>;
    TiledCopy t2r = make_tmem_copy(TmemLoad{}, tCtAcc(_, _, _, _0{}));
    ThrCopy thr = t2r.get_slice(threadIdx.x);
    Tensor cC = make_identity_tensor(Shape<Int<kTileM>, Int<kTileN>>{});
    Tensor tDcC = group_modes<1, 4>(thr.partition_D(cta_mma.partition_C(cC)));
    constexpr int kSlabCols = 32 * (128 / kTileM);
    constexpr int kSlabs = kTileN / kSlabCols;
    static_assert(size<1>(decltype(tDcC){}) == kSlabs, "slab partition");
    static_assert(size<0>(decltype(tDcC){}) == 32, "32 values per load");
    constexpr int NF = kSlabs * 32;
    const int row_in_tile = get<0>(tDcC(_0{}, _0{}));
    const int row = m_tile * kTileM + row_in_tile;
    int scol[kSlabs];
    CUTE_UNROLL
    for (int s = 0; s < kSlabs; ++s) scol[s] = get<1>(tDcC(_0{}, s));

    Tensor rSlab = make_tensor<float>(shape(tDcC(_, _0{})));
    const int col0 = n_tile * kTileN;
    const int n_chunks = chunk1 - chunk0;
    float run[NF];
    if (full) {
      // run = q0; run += q1; run += q2; ... streamed one 32-column slab at a
      // time; each TMEM buffer is released as soon as it is read.
      for (int i = 0; i < n_chunks; ++i) {
        const int b0 = i % nbuf;
        wait_barrier(smem.acc_full[b0], (i / nbuf) & 1);
        Tensor t0 = group_modes<1, 4>(thr.partition_S(tCtAcc(_, _, _, b0)));
        CUTE_UNROLL
        for (int s = 0; s < kSlabs; ++s) {
          copy(t2r, t0(_, s), rSlab);
          cutlass::arch::fence_view_async_tmem_load();
          CUTE_UNROLL
          for (int e = 0; e < 32; ++e)
            run[s * 32 + e] = i == 0 ? rSlab(e) : run[s * 32 + e] + rSlab(e);
        }
        arrive_barrier(smem.acc_empty[b0]);
      }
      if (row < p.M) {
        CUTE_UNROLL
        for (int i = 0; i < NF; i += 8) {
          const int n = col0 + scol[i / 32] + i % 32;
          if (n < p.N) {
            float v[8];
            CUTE_UNROLL
            for (int u = 0; u < 8; ++u) v[u] = run[i + u];
            store8_out<kF32Out>(p.c, p.ldc, row, n, v, p.add, p.ldadd);
          }
        }
      }
    } else {
      // Chunk chunk0 + i (TMEM buffer i, never reused) goes unsummed to
      // slot chunk0 + i of the row owner; the owner does the whole sum.
      const int owner = row_in_tile / p.rp;
      const int rr = row_in_tile - owner * p.rp;
      const uint32_t bar =
          set_block_rank(cast_smem_ptr_to_uint(&smem.recv_full), owner);
      for (int i = 0; i < n_chunks; ++i) {
        wait_barrier(smem.acc_full[i], 0);
        Tensor t0 = group_modes<1, 4>(thr.partition_S(tCtAcc(_, _, _, i)));
        if (i == 0) cluster_wait();  // peers' recv barriers are initialized
        const uint32_t dst0 = set_block_rank(
            cast_smem_ptr_to_uint(smem.recv + ((chunk0 + i) * p.rp + rr) *
                                                  Cfg::kPartStride),
            owner);
        CUTE_UNROLL
        for (int s = 0; s < kSlabs; ++s) {
          copy(t2r, t0(_, s), rSlab);
          cutlass::arch::fence_view_async_tmem_load();
          if (row_in_tile < m_valid) {
            CUTE_UNROLL
            for (int e = 0; e < 32; e += 4)
              dsmem_st_async(dst0 + (scol[s] + e) * 4,
                             make_float4(rSlab(e), rSlab(e + 1), rSlab(e + 2),
                                         rSlab(e + 3)),
                             bar);
          }
        }
      }
    }
  }

  if (!full) {
    const int slot_floats = p.rp * Cfg::kPartStride;
    wait_barrier(smem.recv_full, 0);
    constexpr int kGroups = kTileN / 8;
    const int col0 = n_tile * kTileN;
    for (int idx = threadIdx.x; idx < rows_me * kGroups; idx += kThreads) {
      const int rr = idx / kGroups;
      const int c = (idx % kGroups) * 8;
      const int row = rank * p.rp + rr;
      const int n = col0 + c;
      if (n >= p.N) continue;
      float v[8];
      for (int j = 0; j < p.S; ++j) {  // left to right over all S chunks
        const float* src =
            smem.recv + j * slot_floats + rr * Cfg::kPartStride + c;
        const float4 a = *reinterpret_cast<const float4*>(src);
        const float4 b = *reinterpret_cast<const float4*>(src + 4);
        const float w[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
        CUTE_UNROLL
        for (int u = 0; u < 8; ++u) v[u] = j == 0 ? w[u] : v[u] + w[u];
      }
      store8_out<kF32Out>(p.c, p.ldc, m_tile * kTileM + row, n, v, p.add,
                          p.ldadd);
    }
  }
  __syncthreads();
  if (warp == 4) {
    tmem_allocator.release_allocation_lock();
    tmem_allocator.free(smem.tmem_base, tmem_cols);
  }
#else
  asm volatile("trap;");
#endif
}

int num_sms() {
  static int cached = 0;
  if (cached == 0) {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&cached, cudaDevAttrMultiProcessorCount, dev);
  }
  return cached;
}

// kTM = 64: cluster or full mode. kTM = 128: full mode only (large M); same
// chunk plan and tree, one 128-row UMMA instead of two 64-row ones per k-step.
// kTN / kMinB: tile N and CTAs per SM (kMinB = 2 selects a 4-stage pipeline).
// kClu: the cluster-capable instantiation (receive buffer for S slots of
// ceil(64 / G) rows each); the full-only one keeps that smem for stages.
// dry != nullptr: only return the cost-model estimate (64-row k-tile units).
template <bool kF32Out, int kTM, int kTN, int kMinB, bool kClu>
const char* launch(const Args& p, const Plan& pl, int mode,
                   cudaStream_t stream, int64_t* dry = nullptr) {
  constexpr int kTileM = kTM;
  constexpr int kTileN = kTN;
  // 64-row: up to kMaxCpc chunks per CTA; 128-row: one (S * ceil(128 / S)).
  constexpr int kRecvR =
      !kClu ? 1 : (kTM == 64 ? kMaxCpc : 1) * (kTM + kMaxS - 1);
  constexpr int kStageBytes = (kTileM * kTileK + kTileN * kTileK) * sizeof(T);
  constexpr int kRecvBytes = kRecvR * (kTileN + 4) * sizeof(float);
  constexpr int kStages =
      kMinB == 1 ? (232448 - 3072 - kRecvBytes) / kStageBytes : 4;
  using C = Cfg<kF32Out, kStages, kTM, kRecvR, kTN, kMinB>;
  Tensor mA = make_tensor(make_gmem_ptr(static_cast<const T*>(p.a)),
                          make_shape(p.M, p.K), make_stride(p.lda, _1{}));
  auto tma_a = make_tma_atom(SM90_TMA_LOAD{}, mA,
                             typename C::SmemLayoutA{}(_, _, _, _0{}),
                             Shape<Int<kTileM>, Int<kTileK>>{});
  Tensor mB = make_tensor(make_gmem_ptr(static_cast<const T*>(p.b)),
                          make_shape(p.N, p.K), make_stride(p.ldb, _1{}));
  auto tma_b = make_tma_atom(SM90_TMA_LOAD{}, mB,
                             typename C::SmemLayoutB{}(_, _, _, _0{}),
                             Shape<Int<kTileN>, Int<kTileK>>{});
  auto* kernel = &k31_bi_gemm_kernel<C, decltype(tma_a), decltype(tma_b)>;
  static bool attr_set = false;
  if (!attr_set) {
    if (cudaFuncSetAttribute(kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             C::kSmemBytes) != cudaSuccess ||
        cudaFuncSetAttribute(kernel,
                             cudaFuncAttributeNonPortableClusterSizeAllowed,
                             1) != cudaSuccess)
      return "k31_bi_gemm: cudaFuncSetAttribute failed";
    attr_set = true;
  }
  const int m_tiles = (p.M + kTileM - 1) / kTileM;
  const int n_tiles = (p.N + kTileN - 1) / kTileN;
  const int tiles = m_tiles * n_tiles;
  bool full = pl.S == 1 || !kClu;
  if (!full) {
    if (mode == 1)
      full = true;
    else if (mode == 0)
      full = tiles >= num_sms();
  }
  cudaLaunchConfig_t cfg{};
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  cfg.blockDim = dim3(kThreads);
  cfg.dynamicSmemBytes = C::kSmemBytes;
  cfg.stream = stream;
  // Scheduling only (the chunk sum is fixed): chunks per CTA by a wave cost
  // model against how many G-CTA clusters the device co-schedules.
  static int cap[kMaxS + 1] = {};
  const int kt_chunk = pl.k_chunk / kTileK;
  constexpr int kFixed = 28;
  constexpr int kReduce = 5;
  int cpc = pl.S;
  int64_t est = -1;
  if (!full) {
    // also weighs one CTA walking every chunk.
    const int waves_full = (tiles + num_sms() - 1) / num_sms();
    int64_t best = static_cast<int64_t>(waves_full) *
                   (pl.S * kt_chunk + kFixed);
    for (int c = 1; c <= kMaxCpc && c < pl.S; ++c) {
      const int G = (pl.S + c - 1) / c;
      if (G > kMaxS || pl.S * ((kTileM + G - 1) / G) > kRecvR) continue;
      if (cap[G] == 0) {
        cfg.gridDim = dim3(1, 1, G);
        attr[0].val.clusterDim = {1, 1, static_cast<unsigned>(G)};
        cfg.attrs = attr;
        cfg.numAttrs = 1;
        int n = 0;
        cap[G] =
            cudaOccupancyMaxActiveClusters(&n, kernel, &cfg) == cudaSuccess &&
                    n > 0
                ? n
                : -1;
        cudaGetLastError();
      }
      if (cap[G] < 0) continue;
      const int waves = (tiles + cap[G] - 1) / cap[G];
      const int64_t cost =
          static_cast<int64_t>(waves) * (c * kt_chunk + kFixed + kReduce);
      if (best < 0 || cost < best) {
        best = cost;
        cpc = c;
      }
    }
    if (best < 0 || cpc == pl.S) full = true;  // no cluster shape won/fits
    est = best;
  }
  if (full) cpc = pl.S;
  if (dry != nullptr) {
    *dry = !full ? est
                 : static_cast<int64_t>((tiles + num_sms() - 1) / num_sms()) *
                       (pl.S * kt_chunk + kFixed);
    return nullptr;
  }
  if constexpr (kClu) {  // full mode: run the instantiation without recv smem
    if (full)
      return launch<kF32Out, kTM, kTN, kMinB, false>(p, pl, 1, stream);
  }
  const int G = (pl.S + cpc - 1) / cpc;
  KParams kp{p.M, p.N, p.K, p.ldc, pl.k_chunk, pl.S, cpc,
             G == 1 ? 0 : (kTileM + G - 1) / G, p.c, p.add, p.ldadd};
  cfg.gridDim = dim3(m_tiles, n_tiles, G);
  attr[0].val.clusterDim = {1, 1, static_cast<unsigned>(G)};
  static const bool pdl = [] {
    const char* e = std::getenv("K31_BI_GEMM_PDL");
    return e == nullptr || std::atoi(e) != 0;
  }();
  cudaLaunchAttribute la[2];
  int na = 0;
  if (G > 1) la[na++] = attr[0];
  if (pdl) {
    la[na].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    la[na].val.programmaticStreamSerializationAllowed = 1;
    ++na;
  }
  cfg.attrs = la;
  cfg.numAttrs = na;
  return cudaLaunchKernelEx(&cfg, kernel, tma_a, tma_b, kp) == cudaSuccess
             ? nullptr
             : "k31_bi_gemm: launch failed";
}

}  // namespace

Plan make_plan(int K, int S_req) {
  const int S0 = std::max(1, std::min(kMaxS, S_req));
  const int per = (K + S0 - 1) / S0;
  Plan pl;
  pl.k_chunk = (per + kTileK - 1) / kTileK * kTileK;
  pl.S = (K + pl.k_chunk - 1) / pl.k_chunk;
  return pl;
}

const char* gemm(const Args& p, const Plan& pl, bool out_fp32,
                 cudaStream_t stream, int tile_n) {
  // 128-row tiles once they alone fill the GPU.
  // Execution only: bitwise equal to the 64-row paths (bigemm_test.py gate).
  static const int tile128 = [] {
    const char* e = std::getenv("K31_BI_GEMM_TILE128");
    return e != nullptr ? std::atoi(e) : 1;
  }();
  // (A 128x64 tile with two CTAs per SM measured ~1.7x slower than 128x128 at
  // M=8192: the small tiles are shared-memory-bandwidth bound.)
  // From M = 256 on (K31_BI_GEMM_2SM_MIN_M; 0 = never) the 2-CTA persistent
  // path runs when the wave cost model says it beats the 64-row plan.
  static const int min_m_2sm = [] {
    const char* e = std::getenv("K31_BI_GEMM_2SM_MIN_M");
    return e != nullptr ? std::atoi(e) : 256;
  }();
  if (min_m_2sm > 0 && p.M >= min_m_2sm) {
    // 2-CTA when the wave cost model rates it no worse than the 64-row plan.
    int64_t c64 = 0;
    if (out_fp32)
      launch<true, 64, 128, 1, true>(p, pl, 0, stream, &c64);
    else
      launch<false, 64, 128, 1, true>(p, pl, 0, stream, &c64);
    if (cost_2sm(p.M, p.N, (p.K + kTileK - 1) / kTileK) <= c64)
      return gemm_2sm(p, pl, out_fp32, stream);
  }
  const int n_tiles = (p.N + kTileN - 1) / kTileN;
  // Not at M <= 64: there a 128-row tile only doubles the MMA work of a
  // weight-read-bound GEMM (TP=1 sweep, GB300: kda_in N=49376 M=1 full128
  // 174 us vs full64 118 us; lm_head N=163840 554 vs 354 us).
  const bool big = tile128 && p.M > 64 &&
                   ((p.M + 127) / 128) * n_tiles >= num_sms();
  if (big)
    return out_fp32 ? launch<true, 128, 128, 1, false>(p, pl, 1, stream)
                    : launch<false, 128, 128, 1, false>(p, pl, 1, stream);
  // (64-row plan; the cost models above already preferred it.)
  if (tile_n == 64)
    return out_fp32 ? launch<true, 64, 64, 1, true>(p, pl, 0, stream)
                    : launch<false, 64, 64, 1, true>(p, pl, 0, stream);
  return out_fp32
             ? launch<true, 64, 128, 1, true>(p, pl, 0, stream)
             : launch<false, 64, 128, 1, true>(p, pl, 0, stream);
}

}  // namespace k31bi
