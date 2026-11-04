// K31 TP/EP-invariant fixed-tree all-reduce over MNNVL (prototype, CUDA, no Triton).
//
// Contract implemented here:
//   * G = 16 leaves per reduced element. Rank r of a W-rank group (W | 16) owns leaves
//     [r*16/W, (r+1)*16/W) and puts ONE value per element on the wire: the leaf itself when it
//     owns one leaf (bf16 wire), or the fp32 root of its local balanced subtree (fp32 wire).
//   * The receiver combines the W wire values with the balanced pairwise tree
//     level k pairs (i, i + 2^k) for i % 2^(k+1) == 0, all adds in fp32 (round-to-nearest-even,
//     no FMA: there are no multiplies), and rounds once at the root to the output dtype.
//     Because every rank's value is a contiguous subtree, the root is the same 16-leaf tree for
//     every W in {1, 2, 4, 8, 16}.
//   * Signed zero: wire values and the output have -0 replaced by +0 (Lamport sentinel; the
//     trainer applies the same canonicalisation to the root, which is equivalent).
//
// Transport: Lamport push (FlashInfer/TRT-LLM style). Each rank STORES its value into the peers'
// symmetric buffers; the receiver polls its LOCAL buffer until no element equals the -0 sentinel.
// No barriers. Three rotating buffers per region; call k uses buffer k%3 and clears buffer
// (k+2)%3 (consumed at call k-1, next written at call k+2, which peers can only reach after this
// rank's call k+1 push), so the clear runs right after this call's push, overlapping the wait.
// Epoch + per-buffer used size live in a small local device array (epoch claimed at kernel entry),
// so the kernels are CUDA-graph safe.
//   one-shot : push full vector to every peer slot [me][n]; every rank reduces all n.
//   two-shot : push chunk c to owner c slot [me][chunk]; owner reduces its chunk, pushes the
//              root to every peer's gather buffer; everyone polls the gather buffer.
// Both paths evaluate the SAME per-element tree, so the switch point does not change bits.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cstdint>

namespace {

#ifndef K31_OCC
#define K31_OCC 1
#endif
constexpr int kThreads = 512;
constexpr int kMaxWorld = 16;


struct Params {
  uint64_t peer_base[kMaxWorld];  // symmetric buffer base of every rank
  uint64_t local_base;
  uint64_t mc_base;                // NVLS multicast VA of the buffer, 0 = unicast
  int64_t* state;                  // local: see STATE_* below
  const void* in;
  void* out;
  int64_t n;                       // elements
  int64_t os_off, os_epoch_bytes;  // one-shot region
  int64_t rs_off, rs_epoch_bytes;  // two-shot scatter region
  int64_t ag_off, ag_epoch_bytes;  // two-shot gather region
  int me;
};

enum : int {
  ST_EPOCH_OS = 0, ST_CNT_OS = 1, ST_OS_BYTES = 2,       // [2..4]
  ST_EPOCH_TS = 5, ST_CNT_TS = 6, ST_RS_BYTES = 7,       // [7..9]
  ST_AG_BYTES = 10,                                      // [10..12]
  ST_N = 16
};

template <typename T> struct Wire;
template <> struct Wire<float> {
  static constexpr int kVec = 4;
  static constexpr uint32_t kWordSentinel = 0x80000000u;
};
template <> struct Wire<__nv_bfloat16> {
  static constexpr int kVec = 8;
  static constexpr uint32_t kWordSentinel = 0x80008000u;
};

__device__ inline uint4 ld_volatile(const void* p) {
  uint4 v;
  asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p));
  return v;
}

__device__ inline void st_mc(uint64_t p, uint4 v) {
  asm volatile("multimem.st.relaxed.sys.global.v4.f32 [%0], {%1,%2,%3,%4};" ::"l"(p), "r"(v.x), "r"(v.y),
               "r"(v.z), "r"(v.w)
               : "memory");
}

__device__ inline void st_plain(void* p, uint4 v) {
  asm volatile("st.global.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(p), "r"(v.x), "r"(v.y), "r"(v.z),
               "r"(v.w)
               : "memory");
}

// ---- per-dtype helpers on a 16-byte packet ----
template <typename T> __device__ inline bool dirty(uint4 v);
template <> __device__ inline bool dirty<float>(uint4 v) {
  return v.x == 0x80000000u || v.y == 0x80000000u || v.z == 0x80000000u || v.w == 0x80000000u;
}
__device__ inline bool hneg0(uint32_t w) {
  return (w & 0xffffu) == 0x8000u || (w >> 16) == 0x8000u;
}
template <> __device__ inline bool dirty<__nv_bfloat16>(uint4 v) {
  return hneg0(v.x) || hneg0(v.y) || hneg0(v.z) || hneg0(v.w);
}

template <typename T> __device__ inline uint4 canon(uint4 v);
__device__ inline uint32_t canon32(uint32_t w) { return (w & 0x7fffffffu) ? w : 0u; }
__device__ inline uint32_t canon16x2(uint32_t w) {
  uint32_t lo = w & 0xffffu, hi = w >> 16;
  lo = (lo & 0x7fffu) ? lo : 0u;
  hi = (hi & 0x7fffu) ? hi : 0u;
  return lo | (hi << 16);
}
template <> __device__ inline uint4 canon<float>(uint4 v) {
  return make_uint4(canon32(v.x), canon32(v.y), canon32(v.z), canon32(v.w));
}
template <> __device__ inline uint4 canon<__nv_bfloat16>(uint4 v) {
  return make_uint4(canon16x2(v.x), canon16x2(v.y), canon16x2(v.z), canon16x2(v.w));
}

template <typename T> __device__ inline void unpack(uint4 v, float* f);
template <> __device__ inline void unpack<float>(uint4 v, float* f) {
  f[0] = __uint_as_float(v.x); f[1] = __uint_as_float(v.y);
  f[2] = __uint_as_float(v.z); f[3] = __uint_as_float(v.w);
}
__device__ inline float bf_lo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ inline float bf_hi(uint32_t w) { return __uint_as_float(w & 0xffff0000u); }
template <> __device__ inline void unpack<__nv_bfloat16>(uint4 v, float* f) {
  f[0] = bf_lo(v.x); f[1] = bf_hi(v.x); f[2] = bf_lo(v.y); f[3] = bf_hi(v.y);
  f[4] = bf_lo(v.z); f[5] = bf_hi(v.z); f[6] = bf_lo(v.w); f[7] = bf_hi(v.w);
}

// Pack kVec(Tout) floats. Tout = bf16 packs 8 floats into 16 B; Tout = float packs 4.
template <typename T> __device__ inline uint4 pack(const float* f);
template <> __device__ inline uint4 pack<float>(const float* f) {
  return make_uint4(__float_as_uint(f[0]), __float_as_uint(f[1]), __float_as_uint(f[2]),
                    __float_as_uint(f[3]));
}
__device__ inline uint32_t bf2(float a, float b) {
  __nv_bfloat16 x = __float2bfloat16_rn(a), y = __float2bfloat16_rn(b);
  return (uint32_t)__bfloat16_as_ushort(x) | ((uint32_t)__bfloat16_as_ushort(y) << 16);
}
template <> __device__ inline uint4 pack<__nv_bfloat16>(const float* f) {
  return make_uint4(bf2(f[0], f[1]), bf2(f[2], f[3]), bf2(f[4], f[5]), bf2(f[6], f[7]));
}

// Balanced pairwise tree over W values, in place, fp32. Same association as reference_tree():
// level s: v[i] = v[i] + v[i+s] for i % (2s) == 0.
template <int W>
__device__ inline float tree(float* v) {
#pragma unroll
  for (int s = 1; s < W; s *= 2) {
#pragma unroll
    for (int i = 0; i < W; i += 2 * s) v[i] = __fadd_rn(v[i], v[i + s]);
  }
  return v[0];
}

__device__ inline void clear_region(char* base, int64_t bytes, uint32_t pattern) {
  uint4 p = make_uint4(pattern, pattern, pattern, pattern);
  int64_t nv = bytes / 16;
  for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < nv;
       i += (int64_t)gridDim.x * blockDim.x)
    reinterpret_cast<uint4*>(base)[i] = p;
}

// Epoch of this call, claimed at kernel entry: every thread reads the epoch, the CTA barrier
// makes sure all of them have it, then thread 0 counts the CTA in (the other threads go on); the
// last CTA to arrive (all CTAs have read the epoch by then) resets the counter and bumps the epoch
// for the next call. The next call is a later kernel on the same stream, so plain stores are
// visible to it; no fence anywhere.
__device__ inline int64_t epoch_claim(int64_t* state, int cnt_idx, int epoch_idx) {
  const int64_t e = *reinterpret_cast<volatile int64_t*>(state + epoch_idx);
  if (e < 0) __trap();  // consumes the value: the load has returned before this thread reaches the barrier
  __syncthreads();
  if (threadIdx.x == 0) {
    unsigned long long prev = atomicAdd(reinterpret_cast<unsigned long long*>(state + cnt_idx), 1ull);
    if (prev == gridDim.x - 1) {
      state[cnt_idx] = 0;
      state[epoch_idx] = e + 1;
    }
  }
  return e;
}

// Reduce kVecOut output elements at wire-element offset e (relative to each slot) from W slots
// in a local buffer [W][slot_elems]. Tin/Tout vector widths differ when Tin=bf16, Tout=float or
// vice versa; we always process kE = 8 elements per step (two packets for fp32).
template <typename T> struct Pk { static constexpr int kE = 16 / sizeof(T); };

// Tree root for 8 consecutive elements over NS slots of a local Lamport buffer (slot k at
// slot0 + k*slot_bytes). Slots are taken in groups of up to 8: the group's packets are loaded
// back to back (volatile, so they bypass L1), re-loaded until none holds the sentinel, then fed
// in slot order into a binary-counter stack, i.e. the balanced tree (left + right at every
// node), optionally after scaling slot k by wt[k] (fp32 multiply, round to nearest).
template <int NS, typename Tin, bool kWeighted>
__device__ inline void poll_tree8(const char* slot0, int64_t slot_bytes, const float* wt, float* out8) {
  constexpr int kP = 8 / Pk<Tin>::kE;
  constexpr int kG = NS < 8 ? NS : 8;
  constexpr int kLv = (NS == 16 ? 5 : NS == 8 ? 4 : NS == 4 ? 3 : NS == 2 ? 2 : 1);
  float st[kLv][8];
#pragma unroll
  for (int g0 = 0; g0 < NS; g0 += kG) {
    uint4 x[kG][kP];
    bool bad;
    do {
      bad = false;
#pragma unroll
      for (int i = 0; i < kG; ++i)
#pragma unroll
        for (int q = 0; q < kP; ++q) x[i][q] = ld_volatile(slot0 + (g0 + i) * slot_bytes + 16 * q);
#pragma unroll
      for (int i = 0; i < kG; ++i)
#pragma unroll
        for (int q = 0; q < kP; ++q) bad |= dirty<Tin>(x[i][q]);
    } while (bad);
#pragma unroll
    for (int i = 0; i < kG; ++i) {
      const int k = g0 + i;
      float cur[8];
#pragma unroll
      for (int q = 0; q < kP; ++q) unpack<Tin>(x[i][q], &cur[q * Pk<Tin>::kE]);
      if constexpr (kWeighted) {
#pragma unroll
        for (int e = 0; e < 8; ++e) cur[e] = __fmul_rn(wt[k], cur[e]);
      }
      int lvl = 0;
#pragma unroll
      for (int b = 0; b < kLv; ++b) {
        if (b == lvl && ((k >> b) & 1)) {
#pragma unroll
          for (int e = 0; e < 8; ++e) cur[e] = __fadd_rn(st[b][e], cur[e]);
          lvl = b + 1;
        }
      }
#pragma unroll
      for (int b = 0; b < kLv; ++b)
        if (b == lvl) {
#pragma unroll
          for (int e = 0; e < 8; ++e) st[b][e] = cur[e];
        }
    }
  }
#pragma unroll
  for (int e = 0; e < 8; ++e) out8[e] = st[kLv - 1][e];
}

template <int W, typename Tin>
__device__ inline void poll_reduce8(const char* slot0, int64_t slot_bytes, int64_t e, float* out8) {
  poll_tree8<W, Tin, false>(slot0 + e * (int64_t)sizeof(Tin), slot_bytes, nullptr, out8);
}

template <typename T>
__device__ inline void load8(const char* src, uint4* pk) {
#pragma unroll
  for (int q = 0; q < 8 / Pk<T>::kE; ++q) pk[q] = *reinterpret_cast<const uint4*>(src + 16 * q);
}

template <typename T>
__device__ inline void store8_canon_from_f(char* dst, const float* f) {
  // round to T, canonicalise -0 -> +0, store (local or remote)
#pragma unroll
  for (int q = 0; q < 8 / Pk<T>::kE; ++q) st_plain(dst + 16 * q, canon<T>(pack<T>(f + q * Pk<T>::kE)));
}

template <int W, typename Tin, typename Tout>
__global__ void __launch_bounds__(kThreads, K31_OCC) oneshot_kernel(Params P) {
  cudaGridDependencySynchronize();
  cudaTriggerProgrammaticLaunchCompletion();
  const int64_t e = epoch_claim(P.state, ST_CNT_OS, ST_EPOCH_OS);
  const int cur = (int)(e % 3), drt = (int)((e + 2) % 3);
  const int64_t drt_bytes = P.state[ST_OS_BYTES + drt];
  const int64_t n = P.n;
  const int64_t slot_bytes = n * (int64_t)sizeof(Tin);
  const int64_t nunits = n / 8;
  const int64_t tid0 = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t stride = (int64_t)gridDim.x * blockDim.x;
  // phase 1: push my value into slot [me] of every rank's current buffer
  for (int64_t u = tid0; u < nunits; u += stride) {
    uint4 pk[8 / Pk<Tin>::kE];
    load8<Tin>(reinterpret_cast<const char*>(P.in) + u * 8 * sizeof(Tin), pk);
#pragma unroll
    for (int q = 0; q < 8 / Pk<Tin>::kE; ++q) pk[q] = canon<Tin>(pk[q]);
    const int64_t off = P.os_off + cur * P.os_epoch_bytes + P.me * slot_bytes + u * 8 * sizeof(Tin);
    if (P.mc_base) {
#pragma unroll
      for (int q = 0; q < 8 / Pk<Tin>::kE; ++q) st_mc(P.mc_base + off + 16 * q, pk[q]);
    } else {
#pragma unroll
      for (int p = 0; p < W; ++p) {
        char* dst = reinterpret_cast<char*>(P.peer_base[p]) + off;
#pragma unroll
        for (int q = 0; q < 8 / Pk<Tin>::kE; ++q) st_plain(dst + 16 * q, pk[q]);
      }
    }
  }
  // clear the buffer consumed at the previous call (overlaps the wait for the peers; no peer can
  // write it before this rank's NEXT call has pushed, see header)
  clear_region(reinterpret_cast<char*>(P.local_base) + P.os_off + drt * P.os_epoch_bytes, drt_bytes,
               Wire<Tin>::kWordSentinel);
  // phase 2: poll local slots, tree, store output
  const char* local = reinterpret_cast<const char*>(P.local_base) + P.os_off + cur * P.os_epoch_bytes;
  for (int64_t u = tid0; u < nunits; u += stride) {
    float r[8];
    poll_reduce8<W, Tin>(local, slot_bytes, u * 8, r);
    char* dst = reinterpret_cast<char*>(P.out) + u * 8 * sizeof(Tout);
#pragma unroll
    for (int q = 0; q < 8 / Pk<Tout>::kE; ++q)
      *reinterpret_cast<uint4*>(dst + 16 * q) = canon<Tout>(pack<Tout>(r + q * Pk<Tout>::kE));
  }
  if (threadIdx.x == 0 && blockIdx.x == 0) P.state[ST_OS_BYTES + cur] = W * slot_bytes;
}

template <int W, typename Tin, typename Tout>
__global__ void __launch_bounds__(kThreads, K31_OCC) twoshot_kernel(Params P) {
  cudaGridDependencySynchronize();
  cudaTriggerProgrammaticLaunchCompletion();
  const int64_t e = epoch_claim(P.state, ST_CNT_TS, ST_EPOCH_TS);
  const int cur = (int)(e % 3), drt = (int)((e + 2) % 3);
  const int64_t drt_rs = P.state[ST_RS_BYTES + drt], drt_ag = P.state[ST_AG_BYTES + drt];
  const int64_t n = P.n, chunk = n / W;
  const int64_t nunits = n / 8, cunits = chunk / 8;
  const int64_t tid0 = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t stride = (int64_t)gridDim.x * blockDim.x;
  const int64_t rs_slot_bytes = chunk * (int64_t)sizeof(Tin);
  // phase 1: scatter chunk c to owner c, slot [me]. The chunk order is rotated by rank (rank me
  // starts with owner me+1), so at any moment the 16 ranks target 16 different owners instead of
  // all hammering owner 0 (incast) -- the grid-stride sweep covers ~1 chunk per step at prefill.
  for (int64_t v = tid0; v < nunits; v += stride) {
    const int64_t cr = v / cunits, j = v - cr * cunits;
    const int64_t c = (cr + P.me + 1) & (W - 1);
    const int64_t u = c * cunits + j;
    uint4 pk[8 / Pk<Tin>::kE];
    load8<Tin>(reinterpret_cast<const char*>(P.in) + u * 8 * sizeof(Tin), pk);
    char* dst = reinterpret_cast<char*>(P.peer_base[c]) + P.rs_off + cur * P.rs_epoch_bytes +
                P.me * rs_slot_bytes + j * 8 * sizeof(Tin);
#pragma unroll
    for (int q = 0; q < 8 / Pk<Tin>::kE; ++q) st_plain(dst + 16 * q, canon<Tin>(pk[q]));
  }
  clear_region(reinterpret_cast<char*>(P.local_base) + P.rs_off + drt * P.rs_epoch_bytes, drt_rs,
               Wire<Tin>::kWordSentinel);
  clear_region(reinterpret_cast<char*>(P.local_base) + P.ag_off + drt * P.ag_epoch_bytes, drt_ag,
               Wire<Tout>::kWordSentinel);
  // phase 2: reduce my chunk, push the root to every rank's gather buffer
  const char* rs_local =
      reinterpret_cast<const char*>(P.local_base) + P.rs_off + cur * P.rs_epoch_bytes;
  for (int64_t j = tid0; j < cunits; j += stride) {
    float r[8];
    poll_reduce8<W, Tin>(rs_local, rs_slot_bytes, j * 8, r);
    const int64_t off = (P.me * chunk + j * 8) * (int64_t)sizeof(Tout);
    if (P.mc_base) {
#pragma unroll
      for (int q = 0; q < 8 / Pk<Tout>::kE; ++q)
        st_mc(P.mc_base + P.ag_off + cur * P.ag_epoch_bytes + off + 16 * q,
              canon<Tout>(pack<Tout>(r + q * Pk<Tout>::kE)));
    } else {
#pragma unroll
      for (int p = 0; p < W; ++p) {
        char* dst = reinterpret_cast<char*>(P.peer_base[p]) + P.ag_off + cur * P.ag_epoch_bytes + off;
        store8_canon_from_f<Tout>(dst, r);
      }
    }
  }
  // phase 3: gather
  const char* ag_local =
      reinterpret_cast<const char*>(P.local_base) + P.ag_off + cur * P.ag_epoch_bytes;
  for (int64_t u = tid0; u < nunits; u += stride) {
    const char* src = ag_local + u * 8 * sizeof(Tout);
    char* dst = reinterpret_cast<char*>(P.out) + u * 8 * sizeof(Tout);
#pragma unroll
    for (int q = 0; q < 8 / Pk<Tout>::kE; ++q) {
      uint4 x;
      do { x = ld_volatile(src + 16 * q); } while (dirty<Tout>(x));
      *reinterpret_cast<uint4*>(dst + 16 * q) = x;
    }
  }
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    P.state[ST_RS_BYTES + cur] = n * (int64_t)sizeof(Tin);
    P.state[ST_AG_BYTES + cur] = n * (int64_t)sizeof(Tout);
  }
}

// Local fp32 subtree of `leaves` [L, n] (any float dtype): the same tree as above, used by the
// test to build what a rank owning L leaves puts on the wire. Exposed so the trainer side can
// call literally the same device code.
__device__ inline float to_float(float x) { return x; }
__device__ inline float to_float(__nv_bfloat16 x) { return __bfloat162float(x); }

template <int L, typename Tl>
__global__ void subtree_kernel(const Tl* leaves, float* out, int64_t n) {
  for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += (int64_t)gridDim.x * blockDim.x) {
    float v[L];
#pragma unroll
    for (int l = 0; l < L; ++l) v[l] = to_float(leaves[l * n + i]);
    out[i] = tree<L>(v);
  }
}


// ------------------------------------------------------------------------------------------
// MoE combine with an EP-invariant canonical order.
// out[t, :] = bf16( tree16_c( leaf[t, c, :] ) ), -0 -> +0, c = canonical slot (rank of the slot's
// expert id among the token's 16 ids, ascending). rows [S, H] bf16 = unweighted GEMM2 output in
// FlashInfer's permuted order, row = e2p[t, k]. Two leaf forms, one contract:
//   Tw = bf16 (no LoRA):  leaf = fp32(w[t,k]) * fp32(y[t,k]), formed by the owner (weights in
//       canonical order from `scales`); the wire carries the unweighted bf16 row
//   Tw = fp32 (expert LoRA): leaf = fp32(w * fp32(y)) + fp32(d2), formed by the pushing rank with
//       the arithmetic of k31_moe_lora_fuse slot_kernel; d2 [T*16, H] bf16 in expanded order
// The no-LoRA leaf is the LoRA leaf with d2 = 0 (x + 0 == x up to the sign of zero, and the root
// is canonicalised), so both paths implement the same tree.
//   e2p    [T, 16] int32 expanded (t, k) -> permuted row, -1 = expert not local
//   ids    [T, 16] int32 or int64 expert ids (router slot order)
//   scales [T, 16] fp32 router weights (router slot order)
// Every (t, k) is local on exactly one EP rank, which pushes the leaf row to the owner(s) of slot
// (t, canon(t, k)); ids / weights are replicated. Nothing depends on which rank holds which
// expert, so EP = 1, 4, 8, 16 give the same bits. The fp32 form uses its own buffers (sentinels
// differ per wire dtype).
struct MoeParams {
  uint64_t peer_base[kMaxWorld];
  uint64_t local_base;
  uint64_t mc_base;
  int64_t* state;  // [0]/[1] two-shot epoch/cnt, [2..4] rs bytes, [5..7] ag bytes,
                   // [8..10] os bytes, [11]/[12] one-shot epoch/cnt
  const __nv_bfloat16* rows;
  const __nv_bfloat16* d2;  // fp32 wire only
  const int* e2p;
  const void* ids;
  const float* scales;
  __nv_bfloat16* out;  // [T, H]
  int64_t T, H;
  int64_t rs_off, rs_eb, ag_off, ag_eb, os_off, os_eb;
  int me;
  int ids64;
};

// expert ids are < 2^31 (K31 has 896 experts)
__device__ inline int moe_id(const MoeParams& M, int64_t i) {
  return M.ids64 ? (int)reinterpret_cast<const long long*>(M.ids)[i] : reinterpret_cast<const int*>(M.ids)[i];
}

// canonical position of slot q = t*16 + k, from the already loaded ids (all 32 lanes must call)
__device__ inline int moe_canon_pos(int mine, int other) {
  const int lane = threadIdx.x & 31;
  const unsigned b = __ballot_sync(0xffffffffu, lane < 16 && other < mine);
  return __popc(b);
}

// 16 router weights of token t in canonical order (bitonic sort by id in registers)
__device__ inline void moe_canon_weights(const MoeParams& M, int64_t t, float* wt) {
  int key[16];
#pragma unroll
  for (int k = 0; k < 16; ++k) {
    key[k] = moe_id(M, t * 16 + k);
    wt[k] = M.scales[t * 16 + k];
  }
#pragma unroll
  for (int kk = 2; kk <= 16; kk <<= 1) {
#pragma unroll
    for (int j = kk >> 1; j > 0; j >>= 1) {
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        const int l = i ^ j;
        if (l > i) {
          const bool asc = (i & kk) == 0;
          const bool sw = asc ? (key[i] > key[l]) : (key[i] < key[l]);
          const int a = key[i], b = key[l];
          const float x = wt[i], y = wt[l];
          key[i] = sw ? b : a;
          key[l] = sw ? a : b;
          wt[i] = sw ? y : x;
          wt[l] = sw ? x : y;
        }
      }
    }
  }
}

template <typename Tw> struct MoeLeaf { static constexpr bool kWeighted = false; };
template <> struct MoeLeaf<__nv_bfloat16> { static constexpr bool kWeighted = true; };

// Phase 1: push the wire rows of this rank's local slots. One warp per (slot, half row); a lane
// takes 8-column groups and loads a whole batch before storing any of it (the stores are asm
// "memory" clobbers: a load after one would only issue once the previous group had arrived).
// st(slot_dst, col, packet, v) stores packet `packet` of the 8 columns at `col` of canonical slot
// row `slot_dst` on its receiver(s).
template <typename Tw, typename St>
__device__ inline void moe_push(const MoeParams& M, St st) {
  constexpr int kB = 8;
  const int lane = threadIdx.x & 31;
  const int64_t warp0 = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int64_t nwarp = ((int64_t)gridDim.x * blockDim.x) >> 5;
  const int64_t hu = M.H / 8, half = (hu + 1) / 2;
  for (int64_t it = warp0; it < M.T * 32; it += nwarp) {
    const int64_t q = it >> 1;
    const int e2p_q = M.e2p[q];  // the three loads are independent: issued together
    const int mine = moe_id(M, q), other = moe_id(M, (q & ~15ll) + (lane & 15));
    if (e2p_q < 0) continue;  // warp-uniform: expert not local
    const int64_t dst = (q & ~15ll) + moe_canon_pos(mine, other);
    const __nv_bfloat16* y = M.rows + (int64_t)e2p_q * M.H;
    const __nv_bfloat16* d = MoeLeaf<Tw>::kWeighted ? nullptr : M.d2 + q * M.H;
    const float w = MoeLeaf<Tw>::kWeighted ? 0.f : M.scales[q];
    const int64_t g_lo = (it & 1) * half, g_hi = min(hu, g_lo + half);
    for (int64_t g0 = g_lo + lane; g0 < g_hi; g0 += 32 * kB) {
      uint4 xv[kB], dv[kB];
#pragma unroll
      for (int b = 0; b < kB; ++b) {
        const int64_t g = g0 + 32 * b;
        if (g < g_hi) {
          xv[b] = *reinterpret_cast<const uint4*>(y + g * 8);
          if constexpr (!MoeLeaf<Tw>::kWeighted) dv[b] = *reinterpret_cast<const uint4*>(d + g * 8);
        }
      }
#pragma unroll
      for (int b = 0; b < kB; ++b) {
        const int64_t g = g0 + 32 * b;
        if (g >= g_hi) continue;
        if constexpr (MoeLeaf<Tw>::kWeighted) {
          st(dst, g * 8, 0, canon<Tw>(xv[b]));
        } else {
          float yf[8], df[8], o[8];
          unpack<__nv_bfloat16>(xv[b], yf);
          unpack<__nv_bfloat16>(dv[b], df);
#pragma unroll
          for (int i = 0; i < 8; ++i) o[i] = __fadd_rn(__fmul_rn(w, yf[i]), df[i]);  // = slot_kernel
          st(dst, g * 8, 0, canon<float>(pack<float>(o)));
          st(dst, g * 8, 1, canon<float>(pack<float>(o + 4)));
        }
      }
    }
  }
}

template <int W, typename Tw>
__global__ void __launch_bounds__(kThreads, K31_OCC) moe_twoshot_kernel(MoeParams M) {
  cudaGridDependencySynchronize();
  cudaTriggerProgrammaticLaunchCompletion();
  constexpr int64_t sz = sizeof(Tw);
  constexpr bool kW = MoeLeaf<Tw>::kWeighted;
  const int64_t e = epoch_claim(M.state, 1, 0);
  const int cur = (int)(e % 3), drt = (int)((e + 2) % 3);
  const int64_t drt_rs = M.state[2 + drt], drt_ag = M.state[5 + drt];
  const int64_t H = M.H, ch = H / W, cu = ch / 8, hu = H / 8;
  const int64_t tid0 = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t stride = (int64_t)gridDim.x * blockDim.x;
  // phase 1: column chunk c of every local slot row -> owner c
  moe_push<Tw>(M, [&](int64_t dst, int64_t col, int pk, uint4 v) {
    const int64_t c = col / ch;
    st_plain(reinterpret_cast<char*>(M.peer_base[c]) + M.rs_off + cur * M.rs_eb + (dst * ch + col - c * ch) * sz +
                 16 * pk, v);
  });
  clear_region(reinterpret_cast<char*>(M.local_base) + M.rs_off + drt * M.rs_eb, drt_rs, Wire<Tw>::kWordSentinel);
  clear_region(reinterpret_cast<char*>(M.local_base) + M.ag_off + drt * M.ag_eb, drt_ag, 0x80008000u);
  // phase 2: owner reduces its column chunk of every token, multicasts the root
  const char* rs = reinterpret_cast<const char*>(M.local_base) + M.rs_off + cur * M.rs_eb;
  int64_t t_w = -1;
  float wt[16];
  for (int64_t u = tid0; u < M.T * cu; u += stride) {
    const int64_t t = u / cu, j = u - t * cu;
    if (kW && t != t_w) {
      moe_canon_weights(M, t, wt);
      t_w = t;
    }
    float r[8];
    poll_tree8<16, Tw, kW>(rs + (t * 16 * ch + j * 8) * sz, ch * sz, wt, r);
    const int64_t off = (t * H + M.me * ch + j * 8) * 2;
    uint4 v = canon<__nv_bfloat16>(pack<__nv_bfloat16>(r));
    if (M.mc_base) {
      st_mc(M.mc_base + M.ag_off + cur * M.ag_eb + off, v);
    } else {
#pragma unroll
      for (int q = 0; q < W; ++q)
        st_plain(reinterpret_cast<char*>(M.peer_base[q]) + M.ag_off + cur * M.ag_eb + off, v);
    }
  }
  // phase 3: gather
  const char* ag = reinterpret_cast<const char*>(M.local_base) + M.ag_off + cur * M.ag_eb;
  for (int64_t u = tid0; u < M.T * hu; u += stride) {
    uint4 x;
    do { x = ld_volatile(ag + u * 16); } while (dirty<__nv_bfloat16>(x));
    reinterpret_cast<uint4*>(M.out)[u] = x;
  }
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    M.state[2 + cur] = M.T * 16 * ch * sz;
    M.state[5 + cur] = M.T * H * 2;
  }
}

template <int W, typename Tw>
__global__ void __launch_bounds__(kThreads, K31_OCC) moe_oneshot_kernel(MoeParams M) {
  cudaGridDependencySynchronize();
  cudaTriggerProgrammaticLaunchCompletion();
  constexpr int64_t sz = sizeof(Tw);
  constexpr bool kW = MoeLeaf<Tw>::kWeighted;
  const int64_t e = epoch_claim(M.state, 12, 11);
  const int cur = (int)(e % 3), drt = (int)((e + 2) % 3);
  const int64_t drt_os = M.state[8 + drt];
  const int64_t H = M.H, hu = H / 8;
  const int64_t tid0 = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t stride = (int64_t)gridDim.x * blockDim.x;
  // phase 1: every local slot row -> every rank's slot buffer
  moe_push<Tw>(M, [&](int64_t dst, int64_t col, int pk, uint4 v) {
    const int64_t off = M.os_off + cur * M.os_eb + (dst * H + col) * sz + 16 * pk;
    if (M.mc_base) {
      st_mc(M.mc_base + off, v);
    } else {
#pragma unroll
      for (int p = 0; p < W; ++p) st_plain(reinterpret_cast<char*>(M.peer_base[p]) + off, v);
    }
  });
  clear_region(reinterpret_cast<char*>(M.local_base) + M.os_off + drt * M.os_eb, drt_os, Wire<Tw>::kWordSentinel);
  // phase 2: every rank reduces every token
  const char* os = reinterpret_cast<const char*>(M.local_base) + M.os_off + cur * M.os_eb;
  int64_t t_w = -1;
  float wt[16];
  for (int64_t u = tid0; u < M.T * hu; u += stride) {
    const int64_t t = u / hu, cc = u - t * hu;
    if (kW && t != t_w) {
      moe_canon_weights(M, t, wt);
      t_w = t;
    }
    float r[8];
    poll_tree8<16, Tw, kW>(os + (t * 16 * H + cc * 8) * sz, H * sz, wt, r);
    reinterpret_cast<uint4*>(M.out)[u] = canon<__nv_bfloat16>(pack<__nv_bfloat16>(r));
  }
  if (threadIdx.x == 0 && blockIdx.x == 0) M.state[8 + cur] = M.T * 16 * H * sz;
}


int g_num_sms = 0;
int g_max_ctas = 0;  // 0 = all SMs (set_max_ctas)
int grid_for(int64_t units) {
  if (!g_num_sms) {
    cudaDeviceGetAttribute(&g_num_sms, cudaDevAttrMultiProcessorCount, 0);
  }
  int64_t g = (units + kThreads - 1) / kThreads;
  int64_t cap = (int64_t)K31_OCC * g_num_sms;  // all CTAs co-resident (launch bounds)
  if (g_max_ctas > 0 && g_max_ctas < cap) cap = g_max_ctas;
  if (g > cap) g = cap;
  if (g < 1) g = 1;
  return (int)g;
}


template <typename... KArgs, typename... Args>
void k31_launch(void (*kern)(KArgs...), int grid, cudaStream_t s, Args... args) {
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(grid);
  cfg.blockDim = dim3(kThreads);
  cfg.dynamicSmemBytes = 0;
  cfg.stream = s;
  cfg.attrs = nullptr;
  cfg.numAttrs = 0;
  cudaLaunchKernelEx(&cfg, kern, args...);
}

template <typename Tin, typename Tout>
void launch(bool twoshot, int W, const Params& P, cudaStream_t s) {
  // one-shot: enough CTAs that clearing the W-slot buffer costs ~2 stores per thread
  const int64_t units = twoshot ? P.n / 8 : std::max<int64_t>(P.n / 8, W * P.n * (int64_t)sizeof(Tin) / 32);
  const int grid = grid_for(units);
#define K31_CASE(WW)                                                        \
  case WW:                                                                  \
    if (twoshot) k31_launch(twoshot_kernel<WW, Tin, Tout>, grid, s, P); \
    else k31_launch(oneshot_kernel<WW, Tin, Tout>, grid, s, P);         \
    break;
  switch (W) {
    K31_CASE(2) K31_CASE(4) K31_CASE(8) K31_CASE(16)
    default: TORCH_CHECK(false, "world size must be 2/4/8/16");
  }
#undef K31_CASE
}

}  // namespace

// in: [n] wire dtype (bf16 or fp32); out: [n] bf16 or fp32. peers: int64 [W] symmetric bases.
void tree_all_reduce(torch::Tensor in, torch::Tensor out, torch::Tensor peers, int64_t local_base, int64_t mc_base,
                     torch::Tensor state, int64_t me, bool twoshot, int64_t os_off,
                     int64_t os_epoch_bytes, int64_t rs_off, int64_t rs_epoch_bytes,
                     int64_t ag_off, int64_t ag_epoch_bytes) {
  const int W = (int)peers.numel();
  TORCH_CHECK(in.is_contiguous() && out.is_contiguous());
  TORCH_CHECK(in.numel() == out.numel());
  const int64_t n = in.numel();
  TORCH_CHECK(n % 8 == 0, "n must be a multiple of 8");
  if (twoshot) {
    TORCH_CHECK(n % (8 * W) == 0, "two-shot needs n % (8W) == 0");
  }
  Params P{};
  TORCH_CHECK(peers.device().is_cpu(), "peers must be a CPU int64 tensor");
  auto pc = peers;
  for (int i = 0; i < W; ++i) P.peer_base[i] = (uint64_t)pc.data_ptr<int64_t>()[i];
  P.local_base = (uint64_t)local_base;
  P.mc_base = (uint64_t)mc_base;
  P.state = state.data_ptr<int64_t>();
  P.in = in.data_ptr();
  P.out = out.data_ptr();
  P.n = n;
  P.os_off = os_off; P.os_epoch_bytes = os_epoch_bytes;
  P.rs_off = rs_off; P.rs_epoch_bytes = rs_epoch_bytes;
  P.ag_off = ag_off; P.ag_epoch_bytes = ag_epoch_bytes;
  P.me = (int)me;
  if (!twoshot) {
    TORCH_CHECK(W * n * (int64_t)in.element_size() <= os_epoch_bytes, "one-shot region too small");
  } else {
    TORCH_CHECK(n * (int64_t)in.element_size() <= rs_epoch_bytes, "scatter region too small");
    TORCH_CHECK(n * (int64_t)out.element_size() <= ag_epoch_bytes, "gather region too small");
  }
  auto s = c10::cuda::getCurrentCUDAStream().stream();
  const bool ib = in.scalar_type() == at::kBFloat16, ob = out.scalar_type() == at::kBFloat16;
  if (ib && ob) launch<__nv_bfloat16, __nv_bfloat16>(twoshot, W, P, s);
  else if (!ib && ob) launch<float, __nv_bfloat16>(twoshot, W, P, s);
  else if (!ib && !ob) launch<float, float>(twoshot, W, P, s);
  else launch<__nv_bfloat16, float>(twoshot, W, P, s);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Fill [base, base+bytes) of the LOCAL symmetric buffer with the sentinel of `bf16` ? bf16 : fp32.
void fill_sentinel(int64_t base, int64_t bytes, bool bf16) {
  uint32_t pat = bf16 ? 0x80008000u : 0x80000000u;
  std::vector<uint32_t> h(1 << 20, pat);
  for (int64_t off = 0; off < bytes; off += (int64_t)h.size() * 4) {
    int64_t b = std::min<int64_t>(bytes - off, (int64_t)h.size() * 4);
    cudaMemcpy(reinterpret_cast<char*>(base) + off, h.data(), b, cudaMemcpyHostToDevice);
  }
}

void set_max_ctas(int64_t n) { g_max_ctas = (int)n; }

torch::Tensor local_subtree(torch::Tensor leaves) {  // [L, n] -> fp32 [n]
  const int L = (int)leaves.size(0);
  const int64_t n = leaves.size(1);
  auto out = torch::empty({n}, leaves.options().dtype(torch::kFloat32));
  auto s = c10::cuda::getCurrentCUDAStream().stream();
  int grid = (int)std::min<int64_t>((n + 255) / 256, 4096);
  const bool b = leaves.scalar_type() == at::kBFloat16;
#define K31_L(LL)                                                                                   \
  case LL:                                                                                          \
    if (b) subtree_kernel<LL, __nv_bfloat16><<<grid, 256, 0, s>>>(                                 \
        reinterpret_cast<const __nv_bfloat16*>(leaves.data_ptr()), out.data_ptr<float>(), n);       \
    else subtree_kernel<LL, float><<<grid, 256, 0, s>>>(leaves.data_ptr<float>(), out.data_ptr<float>(), n); \
    break;
  switch (L) {
    K31_L(1) K31_L(2) K31_L(4) K31_L(8) K31_L(16)
    default: TORCH_CHECK(false, "L must be a power of two <= 16");
  }
#undef K31_L
  return out;
}


// rows: [S, H] bf16 unweighted GEMM2 output (permuted, row = e2p). d2: empty -> bf16 wire, leaf =
// w*y formed by the owner; [T*16, H] bf16 (expert-LoRA delta, expanded order) -> fp32 wire, leaf =
// w*y + d2 formed by the pushing rank. e2p [T, 16] int32, ids [T, 16] int32/int64, scales [T, 16]
// fp32, out [T, H] bf16. Buffers of one region set always see the same wire dtype (sentinel).
void moe_combine(torch::Tensor rows, torch::Tensor d2, torch::Tensor e2p, torch::Tensor ids, torch::Tensor scales,
                 torch::Tensor out, torch::Tensor peers, int64_t local_base, int64_t mc_base, torch::Tensor state,
                 int64_t me, bool twoshot, int64_t rs_off, int64_t rs_eb, int64_t ag_off, int64_t ag_eb,
                 int64_t os_off, int64_t os_eb) {
  const int W = (int)peers.numel();
  TORCH_CHECK(peers.device().is_cpu());
  const int64_t T = out.size(0), H = out.size(1);
  const bool f32 = d2.numel() > 0;
  TORCH_CHECK(rows.scalar_type() == at::kBFloat16 && out.scalar_type() == at::kBFloat16);
  TORCH_CHECK(rows.is_contiguous() && out.is_contiguous() && rows.size(1) == H);
  TORCH_CHECK(!f32 || (d2.scalar_type() == at::kBFloat16 && d2.is_contiguous() && d2.numel() == T * 16 * H),
              "d2 must be [T*16, H] bf16 in expanded order");
  const int64_t sz = f32 ? 4 : 2;
  TORCH_CHECK(e2p.scalar_type() == at::kInt && e2p.is_contiguous() && e2p.size(0) == T && e2p.size(1) == 16);
  TORCH_CHECK((ids.scalar_type() == at::kInt || ids.scalar_type() == at::kLong) && ids.is_contiguous() &&
              ids.size(0) == T && ids.size(1) == 16);
  TORCH_CHECK(scales.scalar_type() == at::kFloat && scales.is_contiguous() && scales.size(0) == T &&
              scales.size(1) == 16);
  MoeParams M{};
  for (int i = 0; i < W; ++i) M.peer_base[i] = (uint64_t)peers.data_ptr<int64_t>()[i];
  M.local_base = (uint64_t)local_base;
  M.mc_base = (uint64_t)mc_base;
  M.state = state.data_ptr<int64_t>();
  M.rows = reinterpret_cast<const __nv_bfloat16*>(rows.data_ptr());
  M.d2 = f32 ? reinterpret_cast<const __nv_bfloat16*>(d2.data_ptr()) : nullptr;
  M.e2p = e2p.data_ptr<int>();
  M.ids = ids.data_ptr();
  M.ids64 = ids.scalar_type() == at::kLong;
  M.scales = scales.data_ptr<float>();
  M.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  M.T = T;
  M.H = H;
  M.rs_off = rs_off; M.rs_eb = rs_eb; M.ag_off = ag_off; M.ag_eb = ag_eb;
  M.os_off = os_off; M.os_eb = os_eb;
  M.me = (int)me;
  TORCH_CHECK(H % (8 * W) == 0);
  if (twoshot) {
    TORCH_CHECK(T * 16 * H * sz / W <= rs_eb && T * H * 2 <= ag_eb, "moe regions too small");
  } else {
    TORCH_CHECK(T * 16 * H * sz <= os_eb, "moe one-shot region too small");
  }
  if (T == 0) return;
  auto s = c10::cuda::getCurrentCUDAStream().stream();
  const int64_t units = std::max<int64_t>(std::max<int64_t>(T * 32 * 32, T * H / 8), twoshot ? 0 : T * H * sz / 2);
  const int grid = grid_for(units);
#define K31_M(WW)                                                                     \
  case WW:                                                                            \
    if (f32) {                                                                        \
      if (twoshot) k31_launch(moe_twoshot_kernel<WW, float>, grid, s, M);             \
      else k31_launch(moe_oneshot_kernel<WW, float>, grid, s, M);                     \
    } else {                                                                          \
      if (twoshot) k31_launch(moe_twoshot_kernel<WW, __nv_bfloat16>, grid, s, M);     \
      else k31_launch(moe_oneshot_kernel<WW, __nv_bfloat16>, grid, s, M);             \
    }                                                                                 \
    break;
  switch (W) {
    K31_M(1) K31_M(2) K31_M(4) K31_M(8) K31_M(16)
    default: TORCH_CHECK(false, "bad world");
  }
#undef K31_M
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("tree_all_reduce", &tree_all_reduce);
  m.def("fill_sentinel", &fill_sentinel);
  m.def("local_subtree", &local_subtree);
  m.def("moe_combine", &moe_combine);
  m.def("set_max_ctas", &set_max_ctas);
}
