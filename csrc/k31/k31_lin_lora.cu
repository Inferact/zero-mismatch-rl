// K31 overlay linlora-cuda: punica add_lora_linear (batch-invariant path, one adapter slot, add_inputs=False)
// as one CUDA kernel: shrink (split-K 8) -> ordered split reduce -> expand.
//
// Replaces three Triton launches (lora_shrink partials, _lora_shrink_reduce_kernel, lora_expand) with the
// same arithmetic per output element:
//
//   shrink : p_s[r] (s = 0..7) = one fp32 accumulator chained over mma.m16n8k16 (bf16 x bf16 -> fp32) in K
//            order over the K blocks of split s (BLOCK_K = 256 blocks s, s + 8, s + 16, ..; Triton folds
//            `acc += tl.dot(a, b)` into the dot accumulator)
//   reduce : t[r]    = ((((0 + p_0) + p_1) + ..) + p_7) * scale            (fp32, in split order)
//   expand : out[n]  = bf16( mma(bf16(t[0:16]), B[n, 0:16]) from 0, then mma(bf16(t[16:32]), B[n, 16:32]) )
//
// Elements past K are zero-filled (Triton masks them to 0); whole 128-wide chunks past K are skipped, which
// can only change the sign of a zero partial, and the reduce starts from +0, so the reduced value is the same.
// Rows never mix and the K order is fixed by K alone, so a token's result does not depend on the batch.
//
// One thread-block cluster of 8 CTAs per 16-row tile (4 tiles per CTA from BIG_M rows, the same per-row
// arithmetic; from WHOLE_M rows one CTA per row tile runs the 8 splits one after the other, lin_lora_whole_kernel):
// CTA s computes split s for all S*32 shrink columns,
// the splits are summed in order through distributed shared memory (every CTA sums all 8 the same way), and
// CTA s then expands its share of the output columns. Like punica, a token gets its delta only when its slot is
// listed in active_lora_ids[0 .. num_active_loras) (and is not -1); every other row is written +0, so the output
// needs no zero fill.
//
// Row-parallel input split over TP ranks (lin_lora_part / lin_lora_red, one slice): split s runs on one rank
// only, on the input columns of its chain (moved there by k31_tp_ordered split_cols), with the same code, so
// the same mma operands in the same order give the same fp32 partial; the partials of the 8 splits are then
// gathered and lin_lora_red runs the reduce and expand above on them: the same bits as lin_lora on the
// gathered input.
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

#include <cstdint>
#include <vector>

namespace k31ll {

namespace cg = cooperative_groups;
using bf16 = __nv_bfloat16;

constexpr int RANK = 32, SPLIT = 8, CK = 128, SLD = CK + 8, MAXS = 7;
constexpr int SPAN = SPLIT * 2 * CK;  // K columns per round of the 8 splits (one 256-block each)
// from BIG_M rows (2 * BIG_M for the split reduce) a CTA takes 4 row tiles: prefill chunks, launch shape only; the
// split shrink keeps 1 (its grid is a single split per rank at TP 8)
constexpr int BIG_M = 512;
// from WHOLE_M rows (WHOLE_M / 4 over more than 2 slice grids) one CTA runs all 8 splits of its rows (no cluster,
// the x tile read once per slice): prefill
constexpr int WHOLE_M = 2560;

// launch shape per slice count and 16-row tiles per CTA (none of it changes a value)
template <int S, int MT = 1>
struct Cfg {
  static constexpr int NW = S <= 2 && MT == 1 ? 8 : 16;  // warps
  static constexpr int NT = 32 * NW;
  static constexpr int G = NW / 4;                        // shrink warp groups
  static constexpr int NCH = (S * MT + G - 1) / G;        // shrink chains per warp: (slice, row tile) j * G + group
  static constexpr int U = NW == 8 ? 8 : 4;               // expand n8 tiles per warp per round
  static constexpr int NB = S <= 2 && MT == 1 ? 2 : 1;    // shrink fragment buffers (2: next chunk's read early)
  static constexpr int R = 16 * MT;                       // rows per CTA
  static constexpr int STAGE = (R + S * RANK) * SLD * 2;
  static constexpr int STATIC = R * S * RANK * 4 + R * (S * RANK + 8) * 2;
  static constexpr int ST0 = (220 * 1024 - STATIC) / STAGE;
  static constexpr int ST = ST0 > 8 ? 8 : ST0;  // cp.async ring depth
  static_assert(ST >= 3, "slice count too large for shared memory");
  static constexpr int SMEM = ST * STAGE;
};

__device__ __forceinline__ void mma16816(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ unsigned sptr(const void* p) {
  return static_cast<unsigned>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(sptr(p)));
}
__device__ __forceinline__ void cp16(void* smem, const void* gmem, bool pred) {
  const int n = pred ? 16 : 0;  // zero-fill
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(sptr(smem)), "l"(gmem), "r"(n));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

struct Args {
  const bf16* x;        // [M, K] (MODE 1: [M, ldx], the compact columns of the splits)
  const bf16* a[MAXS];  // [R, K] per slice
  const bf16* b[MAXS];  // [N_s, R] per slice
  int n[MAXS];          // N_s
  int off[MAXS];        // output column offset of slice s
  int tiles[MAXS + 1];  // prefix sums of ceil(N_s / 8)
  const int* tlm;       // [M] slot per token (-1: no LoRA)
  const int* ids;       // punica active_lora_ids: a token is processed iff its slot is one of ids[0..nids)
  int nids;
  bf16* out;  // [M, ldo]
  int M, K, ldo;
  float scale;
  float* part;  // MODE 1: [M, ldp] partials out; MODE 2: [M, ldp] the 8 splits' partials in
  int ldp, ldx, kc, split0;
};

// grid.z > 1: one single-slice problem per z (slices run as independent clusters)
struct Pack {
  Args v[MAXS];
};

// expand n8 tile tt -> slice (-1 past the end) and first column within the slice
template <int S>
__device__ __forceinline__ int tile_slice(const Args& P, int tt, int& n0) {
  int s = 0;
#pragma unroll
  for (int i = 1; i < S; ++i) s += tt >= P.tiles[i];
  n0 = (tt - P.tiles[s]) * 8;
  return tt < P.tiles[S] ? s : -1;
}
// B fragments of U n8 tiles of the expand (tile tt = base + sp * u)
template <int S, int MT>
struct BFrag {
  uint32_t b[Cfg<S, MT>::U][4];
};
template <int S, int MT>
__device__ __forceinline__ void fetch_b(const Args& P, int base, int sp, int g, int t, BFrag<S, MT>& f) {
  using C = Cfg<S, MT>;
#pragma unroll
  for (int u = 0; u < C::U; ++u) {
    int n0;
    const int s = tile_slice<S>(P, base + sp * u, n0);
    const int sc = s < 0 ? 0 : s, n = n0 + g;
    const bool ok = s >= 0 && n < P.n[sc];
    const uint32_t* bp = reinterpret_cast<const uint32_t*>(P.b[sc] + (long)(ok ? n : 0) * RANK + 2 * t);
    f.b[u][0] = ok ? __ldg(bp) : 0u;
    f.b[u][1] = ok ? __ldg(bp + 4) : 0u;
    f.b[u][2] = ok ? __ldg(bp + 8) : 0u;
    f.b[u][3] = ok ? __ldg(bp + 12) : 0u;
  }
}

// ---- expand: the n8 tiles base + sp * u (base = ebase, ebase + sp * U, ..) of rows r0 .. r0 + 16 * MT, fb_cur holding
// the first base's B fragments; rows < M without an active slot get +0
template <int S, int MT>
__device__ __forceinline__ void expand_rows(const Args& P, const bf16 (*tsm)[S * RANK + 8], int r0, int ebase, int sp,
                                            BFrag<S, MT>& fb_cur) {
  using C = Cfg<S, MT>;
  const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  auto active = [&](int r) {
    if (r >= P.M) return false;
    const int l = __ldg(P.tlm + r);
    if (l < 0) return false;
    bool f = false;
    for (int i = 0; i < P.nids; ++i) f |= __ldg(P.ids + i) == l;
    return f;
  };
  bool oa[MT], ob[MT];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) {
    oa[mt] = active(r0 + mt * 16 + g);
    ob[mt] = active(r0 + mt * 16 + g + 8);
  }
  const int ntiles = P.tiles[S];
  const int estride = sp * C::U;
  BFrag<S, MT> fb_nxt;
  for (int base = ebase; base < ntiles; base += estride) {
    if (base + estride < ntiles) fetch_b<S, MT>(P, base + estride, sp, g, t, fb_nxt);
#pragma unroll
    for (int u = 0; u < C::U; ++u) {
      int n0;
      const int s = tile_slice<S>(P, base + sp * u, n0);
      if (s < 0) break;
      const int n = n0 + 2 * t, ns = P.n[s], col = P.off[s] + n;
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        // A operand: this slice's 32 reduced values of rows g, g + 8 of row tile mt (the slice is warp-uniform)
        const int ga = mt * 16 + g, ra = r0 + ga, rb = ra + 8;
        uint32_t ta[2][4];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int k = s * RANK + h * 16 + 2 * t;
          ta[h][0] = *reinterpret_cast<const uint32_t*>(&tsm[ga][k]);
          ta[h][1] = *reinterpret_cast<const uint32_t*>(&tsm[ga + 8][k]);
          ta[h][2] = *reinterpret_cast<const uint32_t*>(&tsm[ga][k + 8]);
          ta[h][3] = *reinterpret_cast<const uint32_t*>(&tsm[ga + 8][k + 8]);
        }
        float acc[4] = {0.f, 0.f, 0.f, 0.f};
        mma16816(acc, ta[0], fb_cur.b[u][0], fb_cur.b[u][1]);
        mma16816(acc, ta[1], fb_cur.b[u][2], fb_cur.b[u][3]);
        const bool wa = ra < P.M, wb = rb < P.M;
        bf16* oa_p = P.out + (long)ra * P.ldo;
        bf16* ob_p = P.out + (long)rb * P.ldo;
        const __nv_bfloat162 va = __floats2bfloat162_rn(oa[mt] ? acc[0] : 0.f, oa[mt] ? acc[1] : 0.f);
        const __nv_bfloat162 vb = __floats2bfloat162_rn(ob[mt] ? acc[2] : 0.f, ob[mt] ? acc[3] : 0.f);
        // pair only at a 4-byte-aligned address: rows start at ra * ldo (ob_p = oa_p + 16 * ldo bytes, same alignment)
        if (n + 1 < ns && (reinterpret_cast<uintptr_t>(oa_p + col) & 3) == 0) {
          if (wa) *reinterpret_cast<__nv_bfloat162*>(oa_p + col) = va;
          if (wb) *reinterpret_cast<__nv_bfloat162*>(ob_p + col) = vb;
        } else if (n < ns) {
          if (wa) oa_p[col] = va.x;
          if (wb) ob_p[col] = vb.x;
          if (n + 1 < ns) {
            if (wa) oa_p[col + 1] = va.y;
            if (wb) ob_p[col + 1] = vb.y;
          }
        }
      }
    }
    fb_cur = fb_nxt;
  }
}

// MODE 0: the whole delta, CTA `split` of a cluster of 8 per tile of MT 16-row tiles (MT: launch shape only).
// MODE 1: shrink only: CTA (l, tile) runs split split0 + l on x[m, l * kc + q * CK + c] = input column
//         chunk_of(q) * CK + c (k31_tp_ordered split_cols) and stores its fp32 partial at part[m, l * S * RANK ..].
// MODE 2: no shrink: the reduce of part[m, s * S * RANK ..] (s = 0..7) and the expand, CTA `split` (blockIdx.x)
//         taking the expand tiles of split `split` as in MODE 0.
template <int S, int MODE, int MT = 1>
__device__ __forceinline__ void lin_lora_body(const Args& P, const int split) {
  using C = Cfg<S, MT>;
  constexpr int NW = C::NW, NT = C::NT, ST = C::ST, G = C::G, NCH = C::NCH, U = C::U, NB = C::NB, R = C::R;
  extern __shared__ __align__(16) unsigned char smem_raw[];
  __shared__ __align__(16) float part[R][S * RANK];    // this split's partial
  __shared__ __align__(16) bf16 tsm[R][S * RANK + 8];  // reduced, bf16 (expand A operand)
  auto xs = reinterpret_cast<bf16(*)[SLD]>(smem_raw);
  cg::cluster_group cluster = cg::this_cluster();
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
  const int r0 = blockIdx.y * R, K = P.K;
  __builtin_assume(tid >= 0 && tid < NT);  // the launch bound, lost when inlined from the kernels below

  // ---- shrink: chunks of this split in K order (block b = split + 8j covers chunks 2b, 2b + 1)
  const int nck = (K + CK - 1) / CK;
  auto chunk_of = [&](int q) { return 2 * (split + SPLIT * (q >> 1)) + (q & 1); };
  int nq = 0;
  while (MODE != 2 && chunk_of(nq) < nck) ++nq;
  const long ldx = MODE == 1 ? P.ldx : K;
  auto load = [&](int stage, int q, int what) {  // what: 1 = x rows, 2 = A rows, 3 = both
    const int k0 = chunk_of(q) * CK, x0 = MODE == 1 ? (int)blockIdx.x * P.kc + q * CK : k0;
    bf16* base = &xs[stage * (R + S * RANK)][0];
    constexpr int NCP = (R + S * RANK) * (CK / 8);
#pragma unroll
    for (int i = 0; i < (NCP + NT - 1) / NT; ++i) {
      const int idx = tid + NT * i;
      if (NCP % NT == 0 || idx < NCP) {
        const int row = idx >> 4, c = (idx & 15) * 8, k = k0 + c;
        const bool kin = k < K;
        if (row < R) {
          if (what & 1) {
            const int gr = r0 + row;
            const bool live = kin && gr < P.M;
            cp16(base + row * SLD + c, P.x + (live ? gr : 0) * ldx + (kin ? x0 + c : 0), live);
          }
        } else if (what & 2) {
          const int ar = row - R, sl = ar / RANK;
          cp16(base + row * SLD + c, P.a[sl] + (long)(ar % RANK) * K + (kin ? k : 0), kin);
        }
      }
    }
  };
  // weights first (A of the prologue chunks, the expand's first B fragments), then the inputs
#pragma unroll
  for (int s = 0; s < ST - 1; ++s)
    if (s < nq) load(s, s, 2);
  const int ebase = split + SPLIT * warp;
  BFrag<S, MT> fb_cur;
  if constexpr (MODE != 1) fetch_b<S, MT>(P, ebase, SPLIT * NW, g, t, fb_cur);
#pragma unroll
  for (int s = 0; s < ST - 1; ++s) {
    if (s < nq) load(s, s, 1);
    cp_commit();  // group s = x of chunk s (+ all prologue A in group 0)
  }
  // warp w: rank columns (w & 3) * 8 .. + 8 of the chains jj = G * j + w / 4 < S * MT, chain jj = slice jj % S
  // of row tile jj / S
  const int grp = warp >> 2;
  const bool computes = grp < S * MT;
  float c[NCH][4];
#pragma unroll
  for (int j = 0; j < NCH; ++j) c[j][0] = c[j][1] = c[j][2] = c[j][3] = 0.f;
  const int xr = lane & 15, xc = (lane >> 4) * 8;
  const int arow = (warp & 3) * 8 + (lane & 7), ac = (lane >> 3) * 8;
  // operand fragments of one chunk; the next chunk's are read from shared memory before the current
  // chunk's mma chain is issued (the asm statements keep program order)
  uint32_t fa[NB][4][NCH][2][4], fbb[NB][4][NCH][4];
  auto frag = [&](int buf, int stage) {
    const bf16* base = &xs[stage * (R + S * RANK)][0];
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
#pragma unroll
      for (int j = 0; j < NCH; ++j) {
        const int jj = G * j + grp, i = jj % S, xrow = (jj / S) * 16 + xr;
        if (jj < S * MT) {
          ldsm_x4(fa[buf][kk][j][0], base + xrow * SLD + kk * 32 + xc);
          ldsm_x4(fa[buf][kk][j][1], base + xrow * SLD + kk * 32 + 16 + xc);
          ldsm_x4(fbb[buf][kk][j], base + (R + i * RANK + arow) * SLD + kk * 32 + ac);
        }
      }
    }
  };
  cp_wait<ST - 2>();
  __syncthreads();
  if (NB == 2 && computes && nq > 0) frag(0, 0);
#pragma unroll 2
  for (int q = 0; q < nq; ++q) {
    const int cur = NB == 2 ? (q & 1) : 0;
    cp_wait<ST - 3>();  // chunks <= q + 1 have landed
    __syncthreads();    // and every warp is done reading the slot refilled below
    {
      const int nx = q + ST - 1;
      if (nx < nq) load(nx % ST, nx, 3);
      cp_commit();
    }
    if (computes) {
      if (NB == 1)
        frag(0, q % ST);
      else if (q + 1 < nq)
        frag(cur ^ (NB - 1), (q + 1) % ST);
#pragma unroll
      for (int kk = 0; kk < 4; ++kk) {
#pragma unroll
        for (int j = 0; j < NCH; ++j) {
          if (G * j + grp < S * MT) {
            mma16816(c[j], fa[cur][kk][j][0], fbb[cur][kk][j][0], fbb[cur][kk][j][1]);
            mma16816(c[j], fa[cur][kk][j][1], fbb[cur][kk][j][2], fbb[cur][kk][j][3]);
          }
        }
      }
    }
  }
  cp_wait<0>();
  if (computes && MODE != 2) {
#pragma unroll
    for (int j = 0; j < NCH; ++j) {
      const int jj = G * j + grp;
      if (jj < S * MT) {
        const int col = (jj % S) * RANK + (warp & 3) * 8 + 2 * t, ga = (jj / S) * 16 + g;
        if constexpr (MODE == 1) {
          float* pp = P.part + (long)blockIdx.x * S * RANK + col;
          if (r0 + ga < P.M) *reinterpret_cast<float2*>(pp + (long)(r0 + ga) * P.ldp) = make_float2(c[j][0], c[j][1]);
          if (r0 + ga + 8 < P.M)
            *reinterpret_cast<float2*>(pp + (long)(r0 + ga + 8) * P.ldp) = make_float2(c[j][2], c[j][3]);
        } else {
          part[ga][col] = c[j][0];
          part[ga][col + 1] = c[j][1];
          part[ga + 8][col] = c[j][2];
          part[ga + 8][col + 1] = c[j][3];
        }
      }
    }
  }
  if constexpr (MODE == 1) return;
  if constexpr (MODE == 0) cluster.sync();

  // ---- reduce: every CTA sums the 8 splits in order (same result everywhere)
  constexpr int NV = R * S * RANK / 4;
  for (int v = tid; v < NV; v += NT) {
    const int row = v / (S * RANK / 4), col = (v % (S * RANK / 4)) * 4;
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int s = 0; s < SPLIT; ++s) {
      float4 p = make_float4(0.f, 0.f, 0.f, 0.f);
      if constexpr (MODE == 0)
        p = *reinterpret_cast<const float4*>(cluster.map_shared_rank(&part[row][col], s));
      else if (r0 + row < P.M)
        p = *reinterpret_cast<const float4*>(P.part + (long)(r0 + row) * P.ldp + s * S * RANK + col);
      acc.x = __fadd_rn(acc.x, p.x);
      acc.y = __fadd_rn(acc.y, p.y);
      acc.z = __fadd_rn(acc.z, p.z);
      acc.w = __fadd_rn(acc.w, p.w);
    }
    tsm[row][col] = __float2bfloat16_rn(__fmul_rn(acc.x, P.scale));
    tsm[row][col + 1] = __float2bfloat16_rn(__fmul_rn(acc.y, P.scale));
    tsm[row][col + 2] = __float2bfloat16_rn(__fmul_rn(acc.z, P.scale));
    tsm[row][col + 3] = __float2bfloat16_rn(__fmul_rn(acc.w, P.scale));
  }
  if constexpr (MODE == 0)
    cluster.sync();  // remote reads done (no CTA leaves while its partial is read); tsm complete
  else
    __syncthreads();

  expand_rows<S, MT>(P, tsm, r0, ebase, SPLIT * NW, fb_cur);
}

// grid = (8, ceil(M / (16 * MT)), 1 or number of slices), cluster = (8, 1, 1)
template <int S, int MT>
__global__ void __cluster_dims__(SPLIT, 1, 1) __launch_bounds__(Cfg<S, MT>::NT)
    lin_lora_kernel(const __grid_constant__ Pack PK) {
  lin_lora_body<S, 0, MT>(PK.v[blockIdx.z], (int)cg::this_cluster().block_rank());
}

// MODE 1: grid = (splits of this rank, ceil(M / (16 * MT))); MODE 2: grid = (8, ceil(M / (16 * MT)))
template <int S, int MODE, int MT>
__global__ void __launch_bounds__(Cfg<S, MT>::NT) lin_lora_tp_kernel(const __grid_constant__ Args P) {
  lin_lora_body<S, MODE, MT>(P, MODE == 1 ? P.split0 + (int)blockIdx.x : (int)blockIdx.x);
}

// Whole-chain launch (large batches): one CTA per tile of MT 16-row tiles runs the 8 splits one after the other: split
// sp's chunks in K order into a chain from zero (exactly CTA sp's MODE 0 chain: same fragments, same mma order), the
// 8 partials in shared memory, then the in-order reduce and the expand of every output column of its rows. No cluster;
// x is read once per slice grid. grid = (1, ceil(M / (16 * MT)), 1 or number of slices).
template <int S, int MT>
struct WCfg {
  using C = Cfg<S, MT>;
  static constexpr int STAGE = (C::R + S * RANK) * SLD * 2;
  static constexpr int PART = SPLIT * C::R * S * RANK * 4;
  static constexpr int ST = (220 * 1024 - PART - C::R * (S * RANK + 8) * 2) / STAGE;
  static_assert(ST >= 3, "whole-chain launch: shared memory");
  static constexpr int SMEM = PART + ST * STAGE;
};

template <int S, int MT>
__global__ void __launch_bounds__(Cfg<S, MT>::NT) lin_lora_whole_kernel(const __grid_constant__ Pack PK) {
  using C = Cfg<S, MT>;
  constexpr int NW = C::NW, NT = C::NT, G = C::G, NCH = C::NCH, R = C::R, ST = WCfg<S, MT>::ST;
  const Args& P = PK.v[blockIdx.z];
  extern __shared__ __align__(16) unsigned char smem_raw[];
  auto part = reinterpret_cast<float(*)[R][S * RANK]>(smem_raw);  // [SPLIT][R][S * RANK]
  auto xs = reinterpret_cast<bf16(*)[SLD]>(smem_raw + WCfg<S, MT>::PART);
  __shared__ __align__(16) bf16 tsm[R][S * RANK + 8];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
  const int r0 = blockIdx.y * R, K = P.K;
  __builtin_assume(tid >= 0 && tid < NT);

  // flattened (split, q) sequence: split sp owns chunks 2 * (sp + SPLIT * (q >> 1)) + (q & 1) < nck
  const int nck = (K + CK - 1) / CK;
  auto chunk_of = [&](int sp, int q) { return 2 * (sp + SPLIT * (q >> 1)) + (q & 1); };
  int qend[SPLIT];  // flattened index one past split sp's last chunk
  int ntot = 0;
#pragma unroll
  for (int sp = 0; sp < SPLIT; ++sp) {
    int q = 0;
    while (chunk_of(sp, q) < nck) ++q;
    ntot += q;
    qend[sp] = ntot;
  }
  auto chunk_at = [&](int i) {
    int sp = 0, q0 = 0;
#pragma unroll
    for (int j = 0; j < SPLIT - 1; ++j)
      if (i >= qend[j]) {
        sp = j + 1;
        q0 = qend[j];
      }
    return chunk_of(sp, i - q0);
  };
  auto load = [&](int stage, int i) {
    const int k0 = chunk_at(i) * CK;
    bf16* base = &xs[stage * (R + S * RANK)][0];
    constexpr int NCP = (R + S * RANK) * (CK / 8);
#pragma unroll
    for (int j = 0; j < (NCP + NT - 1) / NT; ++j) {
      const int idx = tid + NT * j;
      if (NCP % NT == 0 || idx < NCP) {
        const int row = idx >> 4, c = (idx & 15) * 8, k = k0 + c;
        const bool kin = k < K;
        if (row < R) {
          const int gr = r0 + row;
          const bool live = kin && gr < P.M;
          cp16(base + row * SLD + c, P.x + (long)(live ? gr : 0) * K + (kin ? k : 0), live);
        } else {
          const int ar = row - R, sl = ar / RANK;
          cp16(base + row * SLD + c, P.a[sl] + (long)(ar % RANK) * K + (kin ? k : 0), kin);
        }
      }
    }
  };
#pragma unroll
  for (int st = 0; st < ST - 1; ++st) {
    if (st < ntot) load(st, st);
    cp_commit();
  }
  BFrag<S, MT> fb_cur;
  fetch_b<S, MT>(P, warp, NW, g, t, fb_cur);
  // warp w: rank columns (w & 3) * 8 .. + 8 of the chains jj = G * j + w / 4 < S * MT (slice jj % S, row tile jj / S)
  const int grp = warp >> 2;
  float c[NCH][4];
#pragma unroll
  for (int j = 0; j < NCH; ++j) c[j][0] = c[j][1] = c[j][2] = c[j][3] = 0.f;
  const int xr = lane & 15, xc = (lane >> 4) * 8;
  const int arow = (warp & 3) * 8 + (lane & 7), ac = (lane >> 3) * 8;
  int sp = 0;
  for (int i = 0; i < ntot; ++i) {
    cp_wait<ST - 2>();
    __syncthreads();
    {
      const int nx = i + ST - 1;
      if (nx < ntot) load(nx % ST, nx);
      cp_commit();
    }
    const bf16* base = &xs[(i % ST) * (R + S * RANK)][0];
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
#pragma unroll
      for (int j = 0; j < NCH; ++j) {
        const int jj = G * j + grp;
        if (jj < S * MT) {
          uint32_t fa[2][4], fb[4];
          const int xrow = (jj / S) * 16 + xr;
          ldsm_x4(fa[0], base + xrow * SLD + kk * 32 + xc);
          ldsm_x4(fa[1], base + xrow * SLD + kk * 32 + 16 + xc);
          ldsm_x4(fb, base + (R + (jj % S) * RANK + arow) * SLD + kk * 32 + ac);
          mma16816(c[j], fa[0], fb[0], fb[1]);
          mma16816(c[j], fa[1], fb[2], fb[3]);
        }
      }
    }
    while (sp < SPLIT && i + 1 == qend[sp]) {  // split sp done (then any empty ones, +0): partial out, chain from 0
#pragma unroll
      for (int j = 0; j < NCH; ++j) {
        const int jj = G * j + grp;
        if (jj < S * MT) {
          const int col = (jj % S) * RANK + (warp & 3) * 8 + 2 * t, ga = (jj / S) * 16 + g;
          part[sp][ga][col] = c[j][0];
          part[sp][ga][col + 1] = c[j][1];
          part[sp][ga + 8][col] = c[j][2];
          part[sp][ga + 8][col + 1] = c[j][3];
        }
        c[j][0] = c[j][1] = c[j][2] = c[j][3] = 0.f;
      }
      ++sp;
    }
  }
  cp_wait<0>();
  __syncthreads();
  // reduce: the 8 splits in order from +0 (MODE 0)
  for (int v = tid; v < R * S * RANK / 4; v += NT) {
    const int row = v / (S * RANK / 4), col = (v % (S * RANK / 4)) * 4;
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int q = 0; q < SPLIT; ++q) {
      const float4 pv = *reinterpret_cast<const float4*>(&part[q][row][col]);
      acc.x = __fadd_rn(acc.x, pv.x);
      acc.y = __fadd_rn(acc.y, pv.y);
      acc.z = __fadd_rn(acc.z, pv.z);
      acc.w = __fadd_rn(acc.w, pv.w);
    }
    tsm[row][col] = __float2bfloat16_rn(__fmul_rn(acc.x, P.scale));
    tsm[row][col + 1] = __float2bfloat16_rn(__fmul_rn(acc.y, P.scale));
    tsm[row][col + 2] = __float2bfloat16_rn(__fmul_rn(acc.z, P.scale));
    tsm[row][col + 3] = __float2bfloat16_rn(__fmul_rn(acc.w, P.scale));
  }
  __syncthreads();
  expand_rows<S, MT>(P, tsm, r0, warp, NW, fb_cur);
}

}  // namespace k31ll

using k31ll::bf16;

template <int S, int MT>
static void launch_mt(const k31ll::Pack& a, int nz, cudaStream_t st) {
  using C = k31ll::Cfg<S, MT>;
  static int done[64] = {};
  int dev = 0;
  C10_CUDA_CHECK(cudaGetDevice(&dev));
  if (!done[dev]) {
    C10_CUDA_CHECK(
        cudaFuncSetAttribute(k31ll::lin_lora_kernel<S, MT>, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM));
    done[dev] = 1;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(k31ll::SPLIT, (a.v[0].M + C::R - 1) / C::R, nz);
  cfg.blockDim = dim3(C::NT);
  cfg.dynamicSmemBytes = C::SMEM;
  cfg.stream = st;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, k31ll::lin_lora_kernel<S, MT>, a));
}

template <int S, int MT>
static void launch_whole(const k31ll::Pack& a, int nz, cudaStream_t st) {
  using W = k31ll::WCfg<S, MT>;
  static int done[64] = {};
  int dev = 0;
  C10_CUDA_CHECK(cudaGetDevice(&dev));
  if (!done[dev]) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(k31ll::lin_lora_whole_kernel<S, MT>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, W::SMEM));
    done[dev] = 1;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(1, (a.v[0].M + W::C::R - 1) / W::C::R, nz);
  cfg.blockDim = dim3(W::C::NT);
  cfg.dynamicSmemBytes = W::SMEM;
  cfg.stream = st;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, k31ll::lin_lora_whole_kernel<S, MT>, a));
}

template <int S>
static void launch(const k31ll::Pack& a, int nz, cudaStream_t st) {
  if (a.v[0].M >= (nz > 2 ? k31ll::WHOLE_M / 4 : k31ll::WHOLE_M)) return launch_whole<S, S == 1 ? 4 : 2>(a, nz, st);
  if (a.v[0].M >= k31ll::BIG_M) return launch_mt<S, 4>(a, nz, st);
  launch_mt<S, 1>(a, nz, st);
}

template <int MODE, int MT>
static void launch_tp_mt(const k31ll::Args& a, int gx) {
  using C = k31ll::Cfg<1, MT>;
  constexpr int SM = MODE == 1 ? C::SMEM : 0;
  static int done[64] = {};
  int dev = 0;
  C10_CUDA_CHECK(cudaGetDevice(&dev));
  if (SM && !done[dev]) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(k31ll::lin_lora_tp_kernel<1, MODE, MT>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, SM));
    done[dev] = 1;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(gx, (a.M + C::R - 1) / C::R, 1);
  cfg.blockDim = dim3(C::NT);
  cfg.dynamicSmemBytes = SM;
  cfg.stream = c10::cuda::getCurrentCUDAStream().stream();
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, k31ll::lin_lora_tp_kernel<1, MODE, MT>, a));
}

template <int MODE>
static void launch_tp(const k31ll::Args& a, int gx) {
  if constexpr (MODE == 2)
    if (a.M >= 2 * k31ll::BIG_M) return launch_tp_mt<MODE, 4>(a, gx);
  launch_tp_mt<MODE, 1>(a, gx);
}

// out[m, off_s + n] = expand(reduce(shrink(x[m])))  for the active tokens, +0 for the others
void lin_lora(torch::Tensor out, torch::Tensor x, std::vector<torch::Tensor> a, std::vector<torch::Tensor> b,
              torch::Tensor tlm, torch::Tensor ids, int64_t nids, double scale, int64_t fuse_max) {
  TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == at::kBFloat16 && x.dim() == 2, "x");
  TORCH_CHECK(out.is_cuda() && out.is_contiguous() && out.scalar_type() == at::kBFloat16 && out.dim() == 2, "out");
  TORCH_CHECK(tlm.is_cuda() && tlm.is_contiguous() && tlm.scalar_type() == at::kInt, "tlm");
  TORCH_CHECK(ids.is_cuda() && ids.is_contiguous() && ids.scalar_type() == at::kInt && nids <= ids.numel(), "ids");
  const int S = a.size();
  TORCH_CHECK(S >= 1 && S <= k31ll::MAXS && (int)b.size() == S, "slices");
  k31ll::Args P{};
  P.M = x.size(0);
  P.K = x.size(1);
  TORCH_CHECK(P.K > 0 && P.K % 8 == 0 && tlm.numel() >= P.M && out.size(0) == P.M, "shape");
  P.x = (const bf16*)x.data_ptr();
  int off = 0, tiles = 0;
  for (int s = 0; s < S; ++s) {
    TORCH_CHECK(a[s].is_contiguous() && a[s].scalar_type() == at::kBFloat16 && a[s].dim() == 2 &&
                    a[s].size(0) == k31ll::RANK && a[s].size(1) == P.K,
                "a");
    TORCH_CHECK(b[s].is_contiguous() && b[s].scalar_type() == at::kBFloat16 && b[s].dim() == 2 &&
                    b[s].size(1) == k31ll::RANK,
                "b");
    P.a[s] = (const bf16*)a[s].data_ptr();
    P.b[s] = (const bf16*)b[s].data_ptr();
    P.n[s] = b[s].size(0);
    P.off[s] = off;
    P.tiles[s] = tiles;
    off += P.n[s];
    tiles += (P.n[s] + 7) / 8;
  }
  for (int s = S; s <= k31ll::MAXS; ++s) P.tiles[s] = tiles;
  TORCH_CHECK(out.size(1) == off, "out columns");
  P.tlm = tlm.data_ptr<int>();
  P.ids = ids.data_ptr<int>();
  P.nids = (int)nids;
  P.out = (bf16*)out.data_ptr();
  P.ldo = out.size(1);
  P.scale = (float)scale;
  if (P.M == 0) return;
  auto st = c10::cuda::getCurrentCUDAStream().stream();
  k31ll::Pack PK{};
  if (S > 2 || S > fuse_max) {  // one single-slice cluster grid per slice, same values
    for (int s = 0; s < S; ++s) {
      k31ll::Args& Q = PK.v[s];
      Q = P;
      Q.a[0] = P.a[s];
      Q.b[0] = P.b[s];
      Q.n[0] = P.n[s];
      Q.off[0] = P.off[s];
      Q.tiles[0] = 0;
      for (int i = 1; i <= k31ll::MAXS; ++i) Q.tiles[i] = (P.n[s] + 7) / 8;
    }
    return launch<1>(PK, S, st);
  }
  PK.v[0] = P;
  switch (S) {
    case 1: return launch<1>(PK, 1, st);
    case 2: return launch<2>(PK, 1, st);
  }
}

// One slice, input split over ranks. x [M, nl * kc]: for splits split0 .. split0 + nl - 1 of a K-column input,
// the columns of each split's chain (split_cols layout, kc = ceil(K / SPAN) * 256) -> part [M, nl * 32] fp32:
// their shrink partials, lora_A a [32, K]
void lin_lora_part(torch::Tensor part, torch::Tensor x, torch::Tensor a, int64_t split0, int64_t K) {
  TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == at::kBFloat16 && x.dim() == 2, "x");
  TORCH_CHECK(part.is_cuda() && part.is_contiguous() && part.scalar_type() == at::kFloat && part.dim() == 2 &&
                  part.size(0) == x.size(0) && part.size(1) % k31ll::RANK == 0,
              "part");
  TORCH_CHECK(a.is_contiguous() && a.scalar_type() == at::kBFloat16 && a.dim() == 2 && a.size(0) == k31ll::RANK &&
                  a.size(1) == K,
              "a");
  const int nl = part.size(1) / k31ll::RANK;
  const int64_t kc = (K + k31ll::SPAN - 1) / k31ll::SPAN * 256;
  TORCH_CHECK(K % 8 == 0 && nl >= 1 && split0 >= 0 && split0 + nl <= k31ll::SPLIT && x.size(1) == nl * kc, "shape");
  k31ll::Args P{};
  P.M = x.size(0);
  P.K = K;
  P.x = (const bf16*)x.data_ptr();
  P.a[0] = (const bf16*)a.data_ptr();
  P.part = part.data_ptr<float>();
  P.ldp = part.size(1);
  P.ldx = x.size(1);
  P.kc = kc;
  P.split0 = split0;
  if (P.M == 0) return;
  launch_tp<1>(P, nl);
}

// part [M, 8 * 32] fp32: the shrink partials of the 8 splits (split-major) of one slice -> out [M, N]: the
// reduce and expand of lin_lora, lora_B b [N, 32]
void lin_lora_red(torch::Tensor out, torch::Tensor part, torch::Tensor b, torch::Tensor tlm, torch::Tensor ids,
                  int64_t nids, double scale) {
  TORCH_CHECK(part.is_cuda() && part.is_contiguous() && part.scalar_type() == at::kFloat && part.dim() == 2 &&
                  part.size(1) == k31ll::SPLIT * k31ll::RANK,
              "part");
  TORCH_CHECK(out.is_cuda() && out.is_contiguous() && out.scalar_type() == at::kBFloat16 && out.dim() == 2 &&
                  out.size(0) == part.size(0) && out.size(1) == b.size(0),
              "out");
  TORCH_CHECK(b.is_contiguous() && b.scalar_type() == at::kBFloat16 && b.dim() == 2 && b.size(1) == k31ll::RANK, "b");
  TORCH_CHECK(tlm.is_cuda() && tlm.is_contiguous() && tlm.scalar_type() == at::kInt && tlm.numel() >= part.size(0),
              "tlm");
  TORCH_CHECK(ids.is_cuda() && ids.is_contiguous() && ids.scalar_type() == at::kInt && nids <= ids.numel(), "ids");
  k31ll::Args P{};
  P.M = part.size(0);
  P.b[0] = (const bf16*)b.data_ptr();
  P.n[0] = b.size(0);
  for (int i = 1; i <= k31ll::MAXS; ++i) P.tiles[i] = (P.n[0] + 7) / 8;
  P.tlm = tlm.data_ptr<int>();
  P.ids = ids.data_ptr<int>();
  P.nids = (int)nids;
  P.out = (bf16*)out.data_ptr();
  P.ldo = out.size(1);
  P.scale = (float)scale;
  P.part = part.data_ptr<float>();
  P.ldp = part.size(1);
  if (P.M == 0) return;
  launch_tp<2>(P, k31ll::SPLIT);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("lin_lora", &lin_lora);
  m.def("lin_lora_part", &lin_lora_part);
  m.def("lin_lora_red", &lin_lora_red);
}
