// K31 TP-ordered reductions for attention TP > 1 under BI (CUDA, no Triton).
//
// Contract: a TP-W row-parallel layer reproduces the TP = 1 bi-gemm bits exactly.
//   * The TP = 1 GEMM of a row-parallel weight [N, K] runs S = split_count(K) chunks of k_chunk = K / S;
//     every chunk is an fp32 accumulation from zero, the chunk partials are summed LEFT TO RIGHT in fp32
//     (run = q0; run += q1; ...), the LoRA leaf adds fp32(delta) after the last chunk, one rounding to bf16.
//   * With S % W == 0 rank r holds chunks [r*L, (r+1)*L), L = S / W, and computes exactly those partials
//     (k31_bi_gemm.chunk_partial on its own K slice is bitwise the in-kernel partial).
//   * ordered_all_reduce: the owner of an element block reads the W*L partials in global chunk order
//     (rank-major, chunk-minor), sums them left to right with __fadd_rn (no FMA, -fmad=false), adds
//     fp32(delta) when given, rounds once to bf16 (RN), and multicasts the result. Every rank gets the same
//     bits as the TP = 1 kernel, signed zeros included (no canonicalisation anywhere).
//   * gather_window: per rank, only the heads of its grp-aligned head window around its own heads, written
//     straight into the zero-padded window (sparse MLA q of k31 smla-tpgroup; a copy).
//   * gather_cols: x_r [T, kl] of every rank -> out [T, W*kl] with out[t, r*kl + c] = x_r[t, c] (a copy;
//     the row-parallel LoRA shrink needs the full input to run the TP = 1 punica shrink).
//   * split_cols: the row-parallel LoRA shrink split over ranks (k31_lin_lora lin_lora_part): each rank gets,
//     for each of its 8/W shrink splits, the input columns of that split's chain (a copy).
//
// Transport: Lamport push over torch symmetric memory (as k31_tree_allreduce.cu). The sentinel is the
// all-ones word: 0xFFFFFFFF for fp32, 0xFFFF for bf16. Both are NaNs that no arithmetic produces (hardware
// NaNs are 0x7FFFFFFF / 0x7FFF), so finite data, -0 included, travels unchanged. Three rotating buffers
// per region, epoch and per-buffer used bytes in a local device array (CUDA-graph safe), see the tree file.
// A push into a peer's buffer of epoch e needs the peer past its epoch e - 2, which clears that buffer.
// gather_cols pushes to every rank, so finishing its epoch e - 1 already needs every rank past e - 2;
// gather_window and split_cols push to some ranks only. Every region therefore needs an all-rank call
// (gather_cols, ordered_all_reduce) between its epochs e - 2 and e. An MLA layer runs gather_window ->
// split_cols -> fp32 partial gather -> ordered_all_reduce, so the fp32 partials of the split shrink are gathered
// on their own region and epoch (gather_cols part = true).
// The all-rank call between two partial pushes can be missing: a breakable CUDA graph capture runs the eager
// segments (sparse MLA: gather_window, split_cols) and only records the graph segments around them. So
// gather_window and split_cols also carry an epoch marker: every CTA writes the gather epoch e into its slot of
// every rank's marker array and waits until every rank's slot is at least e - 1 before pushing, i.e. every peer
// has finished its epoch e - 2 kernel, which cleared the buffer this call writes. The marker array (one int64
// per rank, filled with the sentinel = -1) only grows, so it needs no clearing.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>

#include <cstdint>
#include <vector>

namespace {

constexpr int kThreads = 512;
constexpr int kMaxWorld = 16;
constexpr uint32_t kSent = 0xFFFFFFFFu;  // fill pattern of every region (fp32 and bf16 alike)

struct OrdParams {
  uint64_t peer_base[kMaxWorld];
  uint64_t local_base;
  uint64_t mc_base;  // NVLS multicast VA of the buffer, 0 = unicast
  int64_t* state;
  const float* in;     // [L, n] fp32 partials of this rank (chunk-major)
  const __nv_bfloat16* delta;  // [n] or nullptr
  __nv_bfloat16* out;  // [n]
  int64_t n;
  int64_t rs_off, rs_eb, ag_off, ag_eb, os_off, os_eb;
  int me;
};

// state: [0] epoch two-shot, [1] cnt, [2..4] rs bytes, [5..7] ag bytes,
//        [8] epoch one-shot, [9] cnt, [10..12] os bytes,
//        [16] epoch gather, [17] cnt, [18..20] gather bytes, [24] epoch partials, [25] cnt, [26..28] partial bytes
enum : int { S_E2 = 0, S_C2 = 1, S_RS = 2, S_AG = 5, S_E1 = 8, S_C1 = 9, S_OS = 10, S_EG = 16, S_CG = 17, S_GB = 18,
             S_EP = 24, S_N = 32 };
// a gather_cols region's state block at sb: epoch sb, cnt sb + 1, bytes sb + 2 ..
static_assert(S_CG == S_EG + 1 && S_GB == S_EG + 2 && S_EP + 5 <= S_N);

__device__ inline uint4 ld_volatile(const void* p) {
  uint4 v;
  asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}

__device__ inline void st_mc(uint64_t p, uint4 v) {
  asm volatile("multimem.st.relaxed.sys.global.v4.f32 [%0], {%1,%2,%3,%4};" ::"l"(p), "r"(v.x), "r"(v.y), "r"(v.z),
               "r"(v.w)
               : "memory");
}

__device__ inline void st_plain(void* p, uint4 v) {
  asm volatile("st.global.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}

__device__ inline bool dirty_f32(uint4 v) { return v.x == kSent || v.y == kSent || v.z == kSent || v.w == kSent; }
__device__ inline bool h_sent(uint32_t w) { return (w & 0xffffu) == 0xffffu || (w >> 16) == 0xffffu; }
__device__ inline bool dirty_bf16(uint4 v) { return h_sent(v.x) || h_sent(v.y) || h_sent(v.z) || h_sent(v.w); }

__device__ inline float bf_lo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ inline float bf_hi(uint32_t w) { return __uint_as_float(w & 0xffff0000u); }
__device__ inline uint32_t bf2(float a, float b) {
  __nv_bfloat16 x = __float2bfloat16_rn(a), y = __float2bfloat16_rn(b);
  return (uint32_t)__bfloat16_as_ushort(x) | ((uint32_t)__bfloat16_as_ushort(y) << 16);
}

// gather epoch e: announce e to every rank, then wait until every rank has announced at least e - 1
template <int W>
__device__ inline void epoch_marker(const uint64_t* peer_base, uint64_t local_base, int64_t m_off, int me,
                                    int64_t e) {
  if (threadIdx.x < W) {
    int64_t* dst = reinterpret_cast<int64_t*>(peer_base[threadIdx.x] + m_off) + me;
    asm volatile("st.volatile.global.s64 [%0], %1;" ::"l"(dst), "l"(e) : "memory");
    const int64_t* src = reinterpret_cast<const int64_t*>(local_base + m_off) + threadIdx.x;
    int64_t v;
    do { asm volatile("ld.volatile.global.s64 %0, [%1];" : "=l"(v) : "l"(src) : "memory"); } while (v < e - 1);
  }
  __syncthreads();
}

__device__ inline void clear_region(char* base, int64_t bytes) {
  const uint4 p = make_uint4(kSent, kSent, kSent, kSent);
  const int64_t nv = bytes / 16;
  for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < nv; i += (int64_t)gridDim.x * blockDim.x)
    reinterpret_cast<uint4*>(base)[i] = p;
}

// see k31_tree_allreduce.cu: epoch claimed at kernel entry, last CTA bumps it for the next call
__device__ inline int64_t epoch_claim(int64_t* state, int cnt_idx, int epoch_idx) {
  const int64_t e = *reinterpret_cast<volatile int64_t*>(state + epoch_idx);
  if (e < 0) __trap();
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

// 8 consecutive elements: poll NS fp32 slots (slot k at slot0 + k*slot_bytes, two 16 B packets each) in
// groups of up to 8 (loads back to back, re-loaded until no sentinel), then sum them LEFT TO RIGHT in slot
// order: run = s0; run = run + s1; ...
template <int NS>
__device__ inline void poll_seqsum8(const char* slot0, int64_t slot_bytes, float* run) {
  constexpr int kG = NS < 8 ? NS : 8;
#pragma unroll
  for (int g0 = 0; g0 < NS; g0 += kG) {
    uint4 x[kG][2];
    bool bad;
    do {
      bad = false;
#pragma unroll
      for (int i = 0; i < kG; ++i) {
        x[i][0] = ld_volatile(slot0 + (g0 + i) * slot_bytes);
        x[i][1] = ld_volatile(slot0 + (g0 + i) * slot_bytes + 16);
      }
#pragma unroll
      for (int i = 0; i < kG; ++i) bad |= dirty_f32(x[i][0]) || dirty_f32(x[i][1]);
    } while (bad);
#pragma unroll
    for (int i = 0; i < kG; ++i) {
      const float v[8] = {__uint_as_float(x[i][0].x), __uint_as_float(x[i][0].y), __uint_as_float(x[i][0].z),
                          __uint_as_float(x[i][0].w), __uint_as_float(x[i][1].x), __uint_as_float(x[i][1].y),
                          __uint_as_float(x[i][1].z), __uint_as_float(x[i][1].w)};
      if (g0 + i == 0) {
#pragma unroll
        for (int e = 0; e < 8; ++e) run[e] = v[e];
      } else {
#pragma unroll
        for (int e = 0; e < 8; ++e) run[e] = __fadd_rn(run[e], v[e]);
      }
    }
  }
}

template <bool kDelta>
__device__ inline uint4 finish8(float* run, const __nv_bfloat16* delta, int64_t e0) {
  if constexpr (kDelta) {
    const uint4 d = *reinterpret_cast<const uint4*>(delta + e0);
    const float df[8] = {bf_lo(d.x), bf_hi(d.x), bf_lo(d.y), bf_hi(d.y), bf_lo(d.z), bf_hi(d.z), bf_lo(d.w), bf_hi(d.w)};
#pragma unroll
    for (int e = 0; e < 8; ++e) run[e] = __fadd_rn(run[e], df[e]);
  }
  return make_uint4(bf2(run[0], run[1]), bf2(run[2], run[3]), bf2(run[4], run[5]), bf2(run[6], run[7]));
}

// two-shot: chunk c of every local partial -> owner c, slot [me*L + l]; owner sums its chunk over the W*L
// slots in global order, multicasts the bf16 root; everyone gathers.
template <int W, int L, bool kDelta>
__global__ void __launch_bounds__(kThreads, 1) ord_twoshot_kernel(OrdParams P) {
  cudaGridDependencySynchronize();
  cudaTriggerProgrammaticLaunchCompletion();
  constexpr int NS = W * L;
  const int64_t e = epoch_claim(P.state, S_C2, S_E2);
  const int cur = (int)(e % 3), drt = (int)((e + 2) % 3);
  const int64_t drt_rs = P.state[S_RS + drt], drt_ag = P.state[S_AG + drt];
  const int64_t n = P.n, chunk = n / W;
  const int64_t c4 = chunk / 4;  // 16 B fp32 vectors per chunk
  const int64_t tid0 = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t stride = (int64_t)gridDim.x * blockDim.x;
  const int64_t slot_bytes = chunk * 4;
  // phase 1: scatter, owner order rotated by rank (no incast)
  for (int64_t v = tid0; v < (int64_t)L * W * c4; v += stride) {
    const int64_t l = v / (W * c4), r = v - l * (W * c4);
    const int64_t cr = r / c4, j = r - cr * c4;
    const int64_t c = (cr + P.me + 1) % W;
    const uint4 x = *reinterpret_cast<const uint4*>(P.in + l * n + c * chunk + j * 4);
    char* dst = reinterpret_cast<char*>(P.peer_base[c]) + P.rs_off + cur * P.rs_eb + (P.me * L + l) * slot_bytes + j * 16;
    st_plain(dst, x);
  }
  clear_region(reinterpret_cast<char*>(P.local_base) + P.rs_off + drt * P.rs_eb, drt_rs);
  clear_region(reinterpret_cast<char*>(P.local_base) + P.ag_off + drt * P.ag_eb, drt_ag);
  // phase 2: reduce my chunk (8 elements per step), multicast the bf16 root
  const char* rs = reinterpret_cast<const char*>(P.local_base) + P.rs_off + cur * P.rs_eb;
  for (int64_t j = tid0; j < chunk / 8; j += stride) {
    float run[8];
    poll_seqsum8<NS>(rs + j * 32, slot_bytes, run);
    const int64_t e0 = P.me * chunk + j * 8;
    const uint4 o = finish8<kDelta>(run, P.delta, e0);
    const int64_t off = P.ag_off + cur * P.ag_eb + e0 * 2;
    if (P.mc_base) {
      st_mc(P.mc_base + off, o);
    } else {
#pragma unroll
      for (int p = 0; p < W; ++p) st_plain(reinterpret_cast<char*>(P.peer_base[p]) + off, o);
    }
  }
  // phase 3: gather
  const char* ag = reinterpret_cast<const char*>(P.local_base) + P.ag_off + cur * P.ag_eb;
  for (int64_t u = tid0; u < n / 8; u += stride) {
    uint4 x;
    do { x = ld_volatile(ag + u * 16); } while (dirty_bf16(x));
    reinterpret_cast<uint4*>(P.out)[u] = x;
  }
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    P.state[S_RS + cur] = (int64_t)NS * slot_bytes;
    P.state[S_AG + cur] = n * 2;
  }
}

// one-shot: every local partial -> every rank, slot [me*L + l]; every rank sums all n.
template <int W, int L, bool kDelta>
__global__ void __launch_bounds__(kThreads, 1) ord_oneshot_kernel(OrdParams P) {
  cudaGridDependencySynchronize();
  cudaTriggerProgrammaticLaunchCompletion();
  constexpr int NS = W * L;
  const int64_t e = epoch_claim(P.state, S_C1, S_E1);
  const int cur = (int)(e % 3), drt = (int)((e + 2) % 3);
  const int64_t drt_os = P.state[S_OS + drt];
  const int64_t n = P.n, n4 = n / 4;
  const int64_t tid0 = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t stride = (int64_t)gridDim.x * blockDim.x;
  const int64_t slot_bytes = n * 4;
  for (int64_t v = tid0; v < (int64_t)L * n4; v += stride) {
    const int64_t l = v / n4, j = v - l * n4;
    const uint4 x = *reinterpret_cast<const uint4*>(P.in + l * n + j * 4);
    const int64_t off = P.os_off + cur * P.os_eb + (P.me * L + l) * slot_bytes + j * 16;
    if (P.mc_base) {
      st_mc(P.mc_base + off, x);
    } else {
#pragma unroll
      for (int p = 0; p < W; ++p) st_plain(reinterpret_cast<char*>(P.peer_base[p]) + off, x);
    }
  }
  clear_region(reinterpret_cast<char*>(P.local_base) + P.os_off + drt * P.os_eb, drt_os);
  const char* os = reinterpret_cast<const char*>(P.local_base) + P.os_off + cur * P.os_eb;
  for (int64_t u = tid0; u < n / 8; u += stride) {
    float run[8];
    poll_seqsum8<NS>(os + u * 32, slot_bytes, run);
    reinterpret_cast<uint4*>(P.out)[u] = finish8<kDelta>(run, P.delta, u * 8);
  }
  if (threadIdx.x == 0 && blockIdx.x == 0) P.state[S_OS + cur] = (int64_t)NS * slot_bytes;
}

struct WindowParams {
  uint64_t peer_base[kMaxWorld];
  uint64_t local_base;
  int64_t* state;
  const __nv_bfloat16* in;  // [T, hw*D]: heads [me*hw, (me+1)*hw)
  __nv_bfloat16* out;       // [T, wout*D]
  int64_t T, D;
  int64_t g_off, g_eb, m_off;
  int hw, grp, wout, me;
};

// head window of rank r: the grp-aligned heads around its heads [r*hw, (r+1)*hw), within H = W*hw
template <int W>
__device__ inline void head_window(int r, const WindowParams& G, int& lo, int& hi) {
  lo = r * G.hw / G.grp * G.grp;
  hi = min(((r + 1) * G.hw + G.grp - 1) / G.grp * G.grp, W * G.hw);
}

// every head goes to the ranks whose window holds it (window head h - lo); a rank reads its window heads and
// writes zeros above them. Shares the gather region and epoch.
template <int W>
__global__ void __launch_bounds__(kThreads, 1) gather_window_kernel(WindowParams G) {
  cudaGridDependencySynchronize();
  cudaTriggerProgrammaticLaunchCompletion();
  const int64_t e = epoch_claim(G.state, S_CG, S_EG);
  const int cur = (int)(e % 3), drt = (int)((e + 2) % 3);
  const int64_t drt_b = G.state[S_GB + drt];
  epoch_marker<W>(G.peer_base, G.local_base, G.m_off, G.me, e);
  const int64_t d8 = G.D / 8, n8 = G.hw * d8, w8 = G.wout * d8;
  const int64_t tid0 = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t stride = (int64_t)gridDim.x * blockDim.x;
  int wlo[W], whi[W];
#pragma unroll
  for (int r = 0; r < W; ++r) head_window<W>(r, G, wlo[r], whi[r]);
  for (int64_t v = tid0; v < G.T * n8; v += stride) {
    const int64_t t = v / n8, c = v - t * n8, i = c / d8;
    const int h = G.me * G.hw + (int)i;
    const uint4 x = *reinterpret_cast<const uint4*>(G.in + t * n8 * 8 + c * 8);
#pragma unroll
    for (int r = 0; r < W; ++r) {
      if (h >= wlo[r] && h < whi[r]) {
        const int64_t off = G.g_off + cur * G.g_eb + ((t * (whi[r] - wlo[r]) + h - wlo[r]) * d8 + c - i * d8) * 16;
        st_plain(reinterpret_cast<char*>(G.peer_base[r]) + off, x);
      }
    }
  }
  clear_region(reinterpret_cast<char*>(G.local_base) + G.g_off + drt * G.g_eb, drt_b);
  int lo, hi;
  head_window<W>(G.me, G, lo, hi);
  const int64_t wn8 = (hi - lo) * d8;
  const char* g = reinterpret_cast<const char*>(G.local_base) + G.g_off + cur * G.g_eb;
  for (int64_t u = tid0; u < G.T * w8; u += stride) {
    const int64_t t = u / w8, c = u - t * w8;
    uint4 x = make_uint4(0u, 0u, 0u, 0u);
    if (c < wn8) {
      do { x = ld_volatile(g + (t * wn8 + c) * 16); } while (dirty_bf16(x));
    }
    reinterpret_cast<uint4*>(G.out)[u] = x;
  }
  if (threadIdx.x == 0 && blockIdx.x == 0) G.state[S_GB + cur] = G.T * wn8 * 16;
}

struct GatherParams {
  uint64_t peer_base[kMaxWorld];
  uint64_t local_base;
  uint64_t mc_base;
  int64_t* state;
  const __nv_bfloat16* in;  // [T, kl]
  __nv_bfloat16* out;       // [T, W*kl]
  int64_t T, kl;            // kl in 2-byte units (an fp32 row of n values: kl = 2n)
  int64_t g_off, g_eb;
  int me;
  int sb;  // state block: S_EG (gather region) or S_EP (fp32 partial region)
};

template <int W, bool F32>
__global__ void __launch_bounds__(kThreads, 1) gather_cols_kernel(GatherParams G) {
  cudaGridDependencySynchronize();
  cudaTriggerProgrammaticLaunchCompletion();
  const int64_t e = epoch_claim(G.state, G.sb + 1, G.sb);
  const int cur = (int)(e % 3), drt = (int)((e + 2) % 3);
  const int64_t drt_b = G.state[G.sb + 2 + drt];
  const int64_t kl8 = G.kl / 8, K = (int64_t)W * G.kl;
  const int64_t tid0 = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t stride = (int64_t)gridDim.x * blockDim.x;
  for (int64_t v = tid0; v < G.T * kl8; v += stride) {
    const int64_t t = v / kl8, c = v - t * kl8;
    const uint4 x = *reinterpret_cast<const uint4*>(G.in + t * G.kl + c * 8);
    const int64_t off = G.g_off + cur * G.g_eb + (t * K + G.me * G.kl + c * 8) * 2;
    if (G.mc_base) {
      st_mc(G.mc_base + off, x);
    } else {
#pragma unroll
      for (int p = 0; p < W; ++p) st_plain(reinterpret_cast<char*>(G.peer_base[p]) + off, x);
    }
  }
  clear_region(reinterpret_cast<char*>(G.local_base) + G.g_off + drt * G.g_eb, drt_b);
  const char* g = reinterpret_cast<const char*>(G.local_base) + G.g_off + cur * G.g_eb;
  for (int64_t u = tid0; u < G.T * K / 8; u += stride) {
    uint4 x;
    do { x = ld_volatile(g + u * 16); } while (F32 ? dirty_f32(x) : dirty_bf16(x));
    reinterpret_cast<uint4*>(G.out)[u] = x;
  }
  if (threadIdx.x == 0 && blockIdx.x == 0) G.state[G.sb + 2 + cur] = G.T * K * 2;
}

struct SplitParams {
  uint64_t peer_base[kMaxWorld];
  uint64_t local_base;
  int64_t* state;
  const __nv_bfloat16* in;  // [T, kl]: input columns [me*kl, (me+1)*kl) of K = W*kl
  __nv_bfloat16* out;       // [T, (8/W)*kc]
  int64_t T, kl, kc;
  int64_t g_off, g_eb, m_off;
  int me;
};

// k31_lin_lora split layout: 256-column block b belongs to shrink split b % 8, split s to rank s / NL (NL = 8/W
// splits per rank); in its rank's output, split s takes columns (s % NL)*kc .., input column k of the split at
// (k / 2048)*256 + k % 256 (its chain order). Columns past K are written 0. Shares the gather region and epoch.
template <int W>
__global__ void __launch_bounds__(kThreads, 1) split_cols_kernel(SplitParams G) {
  cudaGridDependencySynchronize();
  cudaTriggerProgrammaticLaunchCompletion();
  constexpr int NL = 8 / W;
  const int64_t e = epoch_claim(G.state, S_CG, S_EG);
  const int cur = (int)(e % 3), drt = (int)((e + 2) % 3);
  const int64_t drt_b = G.state[S_GB + drt];
  epoch_marker<W>(G.peer_base, G.local_base, G.m_off, G.me, e);
  const int64_t kl8 = G.kl / 8, K = (int64_t)W * G.kl, ld = NL * G.kc, ld8 = ld / 8;
  const int64_t tid0 = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t stride = (int64_t)gridDim.x * blockDim.x;
  for (int64_t v = tid0; v < G.T * kl8; v += stride) {
    const int64_t t = v / kl8, c = (v - t * kl8) * 8, k = G.me * G.kl + c;
    const int s = (int)((k >> 8) & 7);
    const uint4 x = *reinterpret_cast<const uint4*>(G.in + t * G.kl + c);
    const int64_t off = G.g_off + cur * G.g_eb + (t * ld + (s % NL) * G.kc + (k >> 11) * 256 + (k & 255)) * 2;
    st_plain(reinterpret_cast<char*>(G.peer_base[s / NL]) + off, x);
  }
  clear_region(reinterpret_cast<char*>(G.local_base) + G.g_off + drt * G.g_eb, drt_b);
  const char* g = reinterpret_cast<const char*>(G.local_base) + G.g_off + cur * G.g_eb;
  for (int64_t u = tid0; u < G.T * ld8; u += stride) {
    const int64_t j = (u % ld8) * 8, l = j / G.kc, jj = j - l * G.kc;
    uint4 x = make_uint4(0u, 0u, 0u, 0u);
    if ((jj >> 8) * 2048 + (G.me * NL + l) * 256 + (jj & 255) < K) {
      do { x = ld_volatile(g + u * 16); } while (dirty_bf16(x));
    }
    reinterpret_cast<uint4*>(G.out)[u] = x;
  }
  if (threadIdx.x == 0 && blockIdx.x == 0) G.state[S_GB + cur] = G.T * ld * 2;
}

int g_num_sms = 0;
int g_max_ctas = 0;
int grid_for(int64_t units) {
  if (!g_num_sms) cudaDeviceGetAttribute(&g_num_sms, cudaDevAttrMultiProcessorCount, 0);
  int64_t g = (units + kThreads - 1) / kThreads;
  int64_t cap = g_num_sms;  // all CTAs co-resident (Lamport polling)
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

template <int W, int L>
void launch_ord(bool two, bool dl, const OrdParams& P, cudaStream_t s) {
  // one-shot: enough CTAs that clearing the NS-slot buffer costs ~2 stores per thread
  const int64_t units = two ? std::max<int64_t>(P.n / 8, (int64_t)L * P.n / 4)
                            : std::max<int64_t>((int64_t)L * P.n / 4, (int64_t)W * L * P.n / 8);
  const int grid = grid_for(units);
  if (two) {
    if (dl) k31_launch(ord_twoshot_kernel<W, L, true>, grid, s, P);
    else k31_launch(ord_twoshot_kernel<W, L, false>, grid, s, P);
  } else {
    if (dl) k31_launch(ord_oneshot_kernel<W, L, true>, grid, s, P);
    else k31_launch(ord_oneshot_kernel<W, L, false>, grid, s, P);
  }
}

template <int W>
void launch_gather(bool f32, int grid, cudaStream_t s, const GatherParams& G) {
  if (f32) k31_launch(gather_cols_kernel<W, true>, grid, s, G);
  else k31_launch(gather_cols_kernel<W, false>, grid, s, G);
}

}  // namespace

// partials [L, n] fp32 (this rank's chunks, global chunk me*L + l), delta [n] bf16 or empty, out [n] bf16.
void ordered_all_reduce(torch::Tensor partials, torch::Tensor delta, torch::Tensor out, torch::Tensor peers,
                        int64_t local_base, int64_t mc_base, torch::Tensor state, int64_t me, bool twoshot,
                        int64_t rs_off, int64_t rs_eb, int64_t ag_off, int64_t ag_eb, int64_t os_off, int64_t os_eb) {
  const int W = (int)peers.numel();
  TORCH_CHECK(peers.device().is_cpu() && peers.scalar_type() == at::kLong);
  TORCH_CHECK(partials.scalar_type() == at::kFloat && partials.is_contiguous() && partials.dim() == 2);
  TORCH_CHECK(out.scalar_type() == at::kBFloat16 && out.is_contiguous());
  const int L = (int)partials.size(0);
  const int64_t n = partials.size(1);
  TORCH_CHECK(out.numel() == n && n % (8 * W) == 0, "ordered_all_reduce: n must be a multiple of 8*W");
  const bool dl = delta.numel() > 0;
  if (dl) TORCH_CHECK(delta.scalar_type() == at::kBFloat16 && delta.is_contiguous() && delta.numel() == n);
  if (twoshot) {
    TORCH_CHECK((int64_t)L * n * 4 <= rs_eb && n * 2 <= ag_eb, "ordered_all_reduce: two-shot regions too small");
  } else {
    TORCH_CHECK((int64_t)W * L * n * 4 <= os_eb, "ordered_all_reduce: one-shot region too small");
  }
  OrdParams P{};
  for (int i = 0; i < W; ++i) P.peer_base[i] = (uint64_t)peers.data_ptr<int64_t>()[i];
  P.local_base = (uint64_t)local_base;
  P.mc_base = (uint64_t)mc_base;
  P.state = state.data_ptr<int64_t>();
  P.in = partials.data_ptr<float>();
  P.delta = dl ? reinterpret_cast<const __nv_bfloat16*>(delta.data_ptr()) : nullptr;
  P.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  P.n = n;
  P.rs_off = rs_off; P.rs_eb = rs_eb; P.ag_off = ag_off; P.ag_eb = ag_eb; P.os_off = os_off; P.os_eb = os_eb;
  P.me = (int)me;
  if (n == 0) return;
  auto s = c10::cuda::getCurrentCUDAStream().stream();
#define K31_O(WW, LL) \
  if (W == WW && L == LL) { launch_ord<WW, LL>(twoshot, dl, P, s); C10_CUDA_KERNEL_LAUNCH_CHECK(); return; }
  K31_O(2, 1) K31_O(2, 2) K31_O(2, 4) K31_O(2, 8)
  K31_O(4, 1) K31_O(4, 2) K31_O(4, 4)
  K31_O(8, 1) K31_O(8, 2)
  K31_O(16, 1)
#undef K31_O
  TORCH_CHECK(false, "ordered_all_reduce: unsupported (W, L) = (", W, ", ", L, ")");
}

// x [T, kl] bf16 or fp32 -> out [T, W*kl] (fp32: the sentinel test is per 32-bit word); part: the region at
// g_off is the fp32 partial region (its own epoch)
void gather_cols(torch::Tensor x, torch::Tensor out, torch::Tensor peers, int64_t local_base, int64_t mc_base,
                 torch::Tensor state, int64_t me, int64_t g_off, int64_t g_eb, bool part) {
  const int W = (int)peers.numel();
  const bool f32 = x.scalar_type() == at::kFloat;
  TORCH_CHECK((f32 || x.scalar_type() == at::kBFloat16) && x.is_contiguous() && x.dim() == 2);
  TORCH_CHECK(out.scalar_type() == x.scalar_type() && out.is_contiguous());
  const int64_t T = x.size(0), kl = x.size(1) * (f32 ? 2 : 1);
  TORCH_CHECK(kl % 8 == 0 && out.numel() * (f32 ? 2 : 1) == T * W * kl);
  TORCH_CHECK(T * W * kl * 2 <= g_eb, "gather_cols: region too small");
  GatherParams G{};
  for (int i = 0; i < W; ++i) G.peer_base[i] = (uint64_t)peers.data_ptr<int64_t>()[i];
  G.local_base = (uint64_t)local_base;
  G.mc_base = (uint64_t)mc_base;
  G.state = state.data_ptr<int64_t>();
  G.in = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  G.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  G.T = T; G.kl = kl; G.g_off = g_off; G.g_eb = g_eb; G.me = (int)me; G.sb = part ? S_EP : S_EG;
  if (T == 0) return;
  auto s = c10::cuda::getCurrentCUDAStream().stream();
  const int grid = grid_for(std::max<int64_t>(T * W * kl / 8, T * W * kl / 32));
  switch (W) {
    case 2: launch_gather<2>(f32, grid, s, G); break;
    case 4: launch_gather<4>(f32, grid, s, G); break;
    case 8: launch_gather<8>(f32, grid, s, G); break;
    case 16: launch_gather<16>(f32, grid, s, G); break;
    default: TORCH_CHECK(false, "gather_cols: world must be 2/4/8/16");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// x [T, kl] (columns [me*kl, (me+1)*kl) of K = W*kl) -> out [T, (8/W)*kc], kc = ceil(K / 2048)*256: the input
// columns of this rank's k31_lin_lora shrink splits (split_cols_kernel)
void split_cols(torch::Tensor x, torch::Tensor out, torch::Tensor peers, int64_t local_base, torch::Tensor state,
                int64_t me, int64_t g_off, int64_t g_eb, int64_t m_off) {
  const int W = (int)peers.numel();
  TORCH_CHECK(W == 2 || W == 4 || W == 8, "split_cols: world must be 2/4/8");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.is_contiguous() && x.dim() == 2);
  TORCH_CHECK(out.scalar_type() == at::kBFloat16 && out.is_contiguous() && out.dim() == 2);
  const int64_t T = x.size(0), kl = x.size(1), kc = (W * kl + 2047) / 2048 * 256;
  TORCH_CHECK(kl % 8 == 0 && out.size(0) == T && out.size(1) == 8 / W * kc, "split_cols: shape");
  TORCH_CHECK(T * out.size(1) * 2 <= g_eb, "split_cols: region too small");
  SplitParams G{};
  for (int i = 0; i < W; ++i) G.peer_base[i] = (uint64_t)peers.data_ptr<int64_t>()[i];
  G.local_base = (uint64_t)local_base;
  G.state = state.data_ptr<int64_t>();
  G.in = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  G.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  G.T = T; G.kl = kl; G.kc = kc; G.g_off = g_off; G.g_eb = g_eb; G.m_off = m_off; G.me = (int)me;
  if (T == 0) return;
  auto s = c10::cuda::getCurrentCUDAStream().stream();
  const int grid = grid_for(std::max<int64_t>(T * kl / 8, T * out.size(1) / 8));
  switch (W) {
    case 2: k31_launch(split_cols_kernel<2>, grid, s, G); break;
    case 4: k31_launch(split_cols_kernel<4>, grid, s, G); break;
    case 8: k31_launch(split_cols_kernel<8>, grid, s, G); break;
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void fill_sentinel(int64_t base, int64_t bytes) {
  std::vector<uint32_t> h(1 << 20, kSent);
  for (int64_t off = 0; off < bytes; off += (int64_t)h.size() * 4) {
    int64_t b = std::min<int64_t>(bytes - off, (int64_t)h.size() * 4);
    C10_CUDA_CHECK(cudaMemcpy(reinterpret_cast<char*>(base) + off, h.data(), b, cudaMemcpyHostToDevice));
  }
}

// q [T, hw*D] (heads [me*hw, (me+1)*hw) of H = W*hw) -> out [T, wout*D]: the heads lo .. hi - 1 of all ranks, then
// zeros, lo .. hi = this rank's head window (grp-aligned around its heads, hi - lo <= wout for every rank)
void gather_window(torch::Tensor q, torch::Tensor out, torch::Tensor peers, int64_t local_base, torch::Tensor state,
                   int64_t me, int64_t D, int64_t grp, int64_t g_off, int64_t g_eb, int64_t m_off) {
  const int W = (int)peers.numel();
  TORCH_CHECK(q.scalar_type() == at::kBFloat16 && q.is_contiguous() && q.dim() == 2);
  TORCH_CHECK(out.scalar_type() == at::kBFloat16 && out.is_contiguous() && out.dim() == 2);
  const int64_t T = q.size(0), hw = q.size(1) / D, wout = out.size(1) / D;
  TORCH_CHECK(D % 8 == 0 && grp > 0 && q.size(1) == hw * D && out.size(0) == T && out.size(1) == wout * D,
              "gather_window: shape");
  int64_t wmax = 0;
  for (int64_t r = 0; r < W; ++r)
    wmax = std::max(wmax, std::min(((r + 1) * hw + grp - 1) / grp * grp, W * hw) - r * hw / grp * grp);
  TORCH_CHECK(wmax <= wout, "gather_window: a head window is wider than the output");
  TORCH_CHECK(T * wmax * D * 2 <= g_eb, "gather_window: region too small");
  WindowParams G{};
  for (int i = 0; i < W; ++i) G.peer_base[i] = (uint64_t)peers.data_ptr<int64_t>()[i];
  G.local_base = (uint64_t)local_base;
  G.state = state.data_ptr<int64_t>();
  G.in = reinterpret_cast<const __nv_bfloat16*>(q.data_ptr());
  G.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  G.T = T; G.D = D; G.g_off = g_off; G.g_eb = g_eb; G.m_off = m_off;
  G.hw = (int)hw; G.grp = (int)grp; G.wout = (int)wout; G.me = (int)me;
  if (T == 0) return;
  auto s = c10::cuda::getCurrentCUDAStream().stream();
  const int grid = grid_for(std::max<int64_t>(T * hw * D / 8, T * wout * D / 8));
  switch (W) {
    case 2: k31_launch(gather_window_kernel<2>, grid, s, G); break;
    case 4: k31_launch(gather_window_kernel<4>, grid, s, G); break;
    case 8: k31_launch(gather_window_kernel<8>, grid, s, G); break;
    case 16: k31_launch(gather_window_kernel<16>, grid, s, G); break;
    default: TORCH_CHECK(false, "gather_window: world must be 2/4/8/16");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void set_max_ctas(int64_t n) { g_max_ctas = (int)n; }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ordered_all_reduce", &ordered_all_reduce);
  m.def("gather_cols", &gather_cols);
  m.def("split_cols", &split_cols);
  m.def("fill_sentinel", &fill_sentinel);
  m.def("set_max_ctas", &set_max_ctas);
  m.def("gather_window", &gather_window);
}
