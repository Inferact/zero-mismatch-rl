// K31 CF-BI contract v5 prefill engine `prefill` (bv 316 / 332 = BV 16 / 32), two kernels per call:
//   K1s: per (chunk, head): Akk/Aqk on mma.sync, blocked WY (UT, W), Aqk^T, KD, GEND; last partial chunk to cache.
//   K2s: per (V-tile, sequence, head): chunk scan with every product on mma.sync m16n8k8 tf32; WG and KD/Aqk^T
//        loads as cp.async.bulk + mbarrier, P3 on warps 0..3 concurrent with intra on warps 4..7.
#include "kda_cfbi6.cu"

namespace cf6p {
using namespace cf5;
constexpr int KS = 16;

template <int KCX>
DEV void cmrc(int e, int& row, int& kk) {
  const int cm = e >> 5, w = e & 31;
  row = (cm / (KCX / 4)) * 8 + (w >> 2);
  kk = (cm % (KCX / 4)) * 4 + (w & 3);
}
// [K/16][R][16] core-matrix order
DEV size_t cm16(int row, int kk, int R) {
  return (size_t)(kk >> 4) * R * 16 + (((row >> 3) * 4 + ((kk & 15) >> 2)) * 32 + (row & 7) * 4 + (kk & 3));
}
DEV float4 ld4(const float* p) { return *reinterpret_cast<const float4*>(p); }

constexpr int K3_G = 2 * BT * LDB * 2;
constexpr int K3_SMR = K3_G + BT * LDF * 4;
DEV int swE(int v, int k) { return v * D + (k ^ ((v & 7) << 2)); }
DEV void mb_expect(uint64_t* m, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(umma::smem_u32(m)), "r"(bytes)
               : "memory");
}
DEV void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* m) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" ::"r"(
                   umma::smem_u32(dst)),
               "l"(src), "r"(bytes), "r"(umma::smem_u32(m))
               : "memory");
}

// =============================================================================================
// K2s (v5 round 2): K2 with every product on mma.sync m16n8k8 tf32 (== UMMA bitwise), small BV, 1 CTA/SM.
//   P   = [W; GQ] (128 x K=128) x E_vt^T : warp w = m-tile w (rows 16w..), all BV/8 n-tiles
//   U   = UT - P1 (warps 0..3, fragment owners) ; o = P2 (+ M2) + serial (warps 4..7, fragment owners)
//   P3  = KD (128 k x K=64 j) x U          : warp w = m-tile w (k rows 16w..)
//   cp.async: group A = WG of the next chunk (issued after P), group B = KD + AQKT of the next chunk.
// =============================================================================================
template <int BV>
struct K2s {
  static constexpr int LU = BV + 8;
  static constexpr int OFF_WG = 0;
  static constexpr int OFF_KD = OFF_WG + 16384 * 4;
  static constexpr int OFF_AQ = OFF_KD + 8192 * 4;
  static constexpr int OFF_E = OFF_AQ + BT * BT * 4;
  static constexpr int OFF_U = OFF_E + BV * D * 4;
  static constexpr int OFF_MB = ((OFF_U + BT * LU * 4 + 63) / 64) * 64;
  static constexpr int SMEM = OFF_MB + 64;
  static constexpr int NT = BV / 8;
};
DEV int cmi(int row, int kk) {  // index inside a [rows][16] cmrc<16> slice
  return ((row >> 3) * 4 + (kk >> 2)) * 32 + (row & 7) * 4 + (kk & 3);
}
DEV void step3s(float* big, float* sml, const float* a, const float* b) {
  const uint32_t ah[4] = {__float_as_uint(a[0]), __float_as_uint(a[1]), __float_as_uint(a[2]), __float_as_uint(a[3])};
  const uint32_t al[4] = {mm::lob(a[0]), mm::lob(a[1]), mm::lob(a[2]), mm::lob(a[3])};
  const uint32_t bh[2] = {__float_as_uint(b[0]), __float_as_uint(b[1])};
  const uint32_t bl[2] = {mm::lob(b[0]), mm::lob(b[1])};
  mm::mma8(sml, al, bh);
  mm::mma8(sml, ah, bl);
  mm::mma8(big, ah, bh);
}

// =============================================================================================
// K1s (v5 round 3): K1 without TMEM. Akk/Aqk on mma.sync (unit = subchunk s x n-tile, ks and qs tiles share the
// kjr B fragment), blocked WY cross-block products with B fragments / results moved by warp shuffles (no sX/sM),
// 2 CTAs/SM.
// =============================================================================================
constexpr int KS_V = K3_SMR + ((3 * BT * 4 + 127) / 128) * 128;
constexpr int KS_AKK = KS_V + BT * LDB * 2;
constexpr int KS_SMEM = KS_AKK + BT * LDT * 4;
__global__ void __launch_bounds__(256, 2) k1s_kernel(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v, const __nv_bfloat16* __restrict__ g,
    const __nv_bfloat16* __restrict__ beta, const float* __restrict__ A_log,
    const float* __restrict__ bias, const int* __restrict__ chunk_tab, float* __restrict__ WG,
    float* __restrict__ UT, float* __restrict__ AQKT, float* __restrict__ KDc, float* __restrict__ GEND,
    float* __restrict__ cache, int H, float lower, float scale) {
  extern __shared__ __align__(1024) unsigned char sm[];
  __nv_bfloat16* rawQ = reinterpret_cast<__nv_bfloat16*>(sm);
  __nv_bfloat16* rawK = rawQ + BT * LDB;
  float* sG = reinterpret_cast<float*>(sm + K3_G);
  float* sB = reinterpret_cast<float*>(sm + K3_SMR);
  float* sRq = sB + BT;
  float* sRk = sRq + BT;
  float* sAkkT = reinterpret_cast<float*>(sm + KS_AKK);
  __nv_bfloat16* rawV = reinterpret_cast<__nv_bfloat16*>(sm + KS_V);
  const int c = blockIdx.x, h = blockIdx.y, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int seq = chunk_tab[c * 4], n = chunk_tab[c * 4 + 2], is_last = chunk_tab[c * 4 + 3];
  const long t0 = chunk_tab[c * 4 + 1];
  const long ch = (long)c * H + h;
  for (int p = tid; p < 2 * BT * 16; p += 256) {
    const int which = p >> 10, t = (p >> 4) & 63, c8 = p & 15;
    __nv_bfloat16* dst = rawQ + which * BT * LDB + t * LDB + c8 * 8;
    const __nv_bfloat16* src = which == 0 ? q : k;
    if (t < n) cp16(dst, src + ((t0 + t) * H + h) * D + c8 * 8);
    else *reinterpret_cast<uint4*>(dst) = make_uint4(0, 0, 0, 0);
  }
  for (int p = tid; p < BT * 16; p += 256) {
    const int t = p >> 4, c8 = p & 15;
    __nv_bfloat16* dst = rawV + t * LDB + c8 * 8;
    if (t < n) cp16(dst, v + ((t0 + t) * H + h) * D + c8 * 8);
    else *reinterpret_cast<uint4*>(dst) = make_uint4(0, 0, 0, 0);
  }
  cp_commit();
  {
    const float nah = head_nah(A_log[h]), low2 = __fmul_rn(lower, L2E);
    const int c8 = tid & 15;
    float bs[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) bs[e] = bias[h * D + c8 * 8 + e];
    uint4 gv[4];
#pragma unroll
    for (int it = 0; it < 4; ++it) {
      const int t = (tid >> 4) + 16 * it;
      gv[it] = t < n ? *reinterpret_cast<const uint4*>(g + ((t0 + t) * H + h) * D + c8 * 8) : make_uint4(0, 0, 0, 0);
    }
#pragma unroll
    for (int it = 0; it < 4; ++it) {
      const int t = (tid >> 4) + 16 * it;
      const __nv_bfloat16* e8 = reinterpret_cast<const __nv_bfloat16*>(&gv[it]);
#pragma unroll
      for (int e = 0; e < 8; ++e) sG[t * LDF + c8 * 8 + e] = gate2(bf(e8[e]), bs[e], nah, low2);
    }
  }
  cp_wait<0>();
  __syncthreads();
  if (tid < 128) {
    float G = 0.f;
#pragma unroll 16
    for (int t = 0; t < BT; ++t) {
      const float a = sG[t * LDF + tid];
      G = t == 0 ? a : __fadd_rn(G, a);
      sG[t * LDF + tid] = G;
    }
  } else {
    const int row = (tid - 128) & 63;
    const __nv_bfloat16* src = (tid < 192 ? rawQ : rawK) + row * LDB;
    float s = 0.f;
#pragma unroll
    for (int c8 = 0; c8 < 16; ++c8) {
      const uint4 u = *reinterpret_cast<const uint4*>(src + c8 * 8);
      const __nv_bfloat16* e8 = reinterpret_cast<const __nv_bfloat16*>(&u);
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const float x = bf(e8[e]);
        s = fmaf(x, x, s);
      }
    }
    if (tid < 192) {
      sRq[row] = __fmul_rn(rnorm(s), scale);
      sB[row] = row < n ? sigm2(bf(beta[(t0 + row) * H + h])) : 0.f;
    } else {
      sRk[row] = rnorm(s);
    }
  }
  __syncthreads();
  // ---- Akk / Aqk on mma.sync: unit (s, nt): rows t of subchunk s (ks tile + qs tile), j in n-tile nt
  {
    const int g8 = lane >> 2, t4 = lane & 3;
    for (int u = warp; u < 20; u += 8) {
      const int s = u < 2 ? 0 : u < 6 ? 1 : u < 12 ? 2 : 3;
      const int nt = u - (s == 0 ? 0 : s == 1 ? 2 : s == 2 ? 6 : 12);
      const int r0 = 16 * s, ta = r0 + g8, tb = ta + 8, j = 8 * nt + g8;
      const float rka = sRk[ta], rkb = sRk[tb], rqa = sRq[ta], rqb = sRq[tb], rkj = sRk[j];
      float bk[4] = {-0.f, -0.f, -0.f, -0.f}, sk[4] = {-0.f, -0.f, -0.f, -0.f};
      float bq[4] = {-0.f, -0.f, -0.f, -0.f}, sq[4] = {-0.f, -0.f, -0.f, -0.f};
#pragma unroll 2
      for (int ks = 0; ks < 16; ++ks) {
        float ak[4], aq[4], bb[2];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int t = (e & 1) ? tb : ta, kk = 8 * ks + t4 + ((e & 2) ? 4 : 0);
          const float fr = ex2(__fsub_rn(sG[t * LDF + kk], sG[r0 * LDF + kk]));
          ak[e] = __fmul_rn(__fmul_rn(bf(rawK[t * LDB + kk]), (e & 1) ? rkb : rka), fr);
          aq[e] = __fmul_rn(__fmul_rn(bf(rawQ[t * LDB + kk]), (e & 1) ? rqb : rqa), fr);
        }
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int kk = 8 * ks + t4 + 4 * e;
          bb[e] = __fmul_rn(__fmul_rn(bf(rawK[j * LDB + kk]), rkj), ex2(__fsub_rn(sG[r0 * LDF + kk], sG[j * LDF + kk])));
        }
        step3s(bk, sk, ak, bb);
        step3s(bq, sq, aq, bb);
      }
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int t = e < 2 ? ta : tb, jj = 8 * nt + 2 * t4 + (e & 1);
        if (jj < t) sAkkT[jj * LDT + t] = __fmul_rn(sB[t], __fadd_rn(bk[e], sk[e]));
        if (jj <= t) AQKT[(ch * BT + jj) * BT + t] = __fadd_rn(bq[e], sq[e]);
      }
    }
  }
  __syncthreads();
  {
    const int i = tid & 127, hf = tid >> 7;
    float X[BT];
    if (hf == 0) {
#pragma unroll
      for (int t = 0; t < BT; ++t) X[t] = __fmul_rn(sB[t], bf(rawV[t * LDB + i]));
    } else {
#pragma unroll
      for (int t = 0; t < BT; ++t)
        X[t] = __fmul_rn(sB[t], __fmul_rn(ex2(sG[t * LDF + i]), __fmul_rn(bf(rawK[t * LDB + i]), sRk[t])));
    }
#pragma unroll
    for (int B = 0; B < 4; ++B) {
      if (B > 0) {
        // M[r][c] = sum_{kk < 16B} Akk[16B+r][kk] X[kk][c] for the warp's own 32 columns c = 32*warp + 8*nt + g
        const int g8 = lane >> 2, t4 = lane & 3;
        float o4[4][4];
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
          float big[4] = {-0.f, -0.f, -0.f, -0.f}, sml[4] = {-0.f, -0.f, -0.f, -0.f};
#pragma unroll
          for (int ks = 0; ks < 2 * B; ++ks) {
            const int k0 = 8 * ks;
            const float a[4] = {sAkkT[(k0 + t4) * LDT + 16 * B + g8], sAkkT[(k0 + t4) * LDT + 16 * B + g8 + 8],
                                sAkkT[(k0 + t4 + 4) * LDT + 16 * B + g8], sAkkT[(k0 + t4 + 4) * LDT + 16 * B + g8 + 8]};
            float bb[2] = {0.f, 0.f};
#pragma unroll
            for (int r = 0; r < 4; ++r) {
              const float v0 = __shfl_sync(0xffffffffu, X[k0 + r], 8 * nt + g8);
              const float v1 = __shfl_sync(0xffffffffu, X[k0 + 4 + r], 8 * nt + g8);
              if (t4 == r) {
                bb[0] = v0;
                bb[1] = v1;
              }
            }
            step3s(big, sml, a, bb);
          }
#pragma unroll
          for (int e = 0; e < 4; ++e) o4[nt][e] = __fadd_rn(big[e], sml[e]);
        }
        // deliver M[r][lane] to the column owner (lane): src lane (r & 7) * 4 + ((lane & 7) >> 1), elem (r >= 8) * 2 + (lane & 1)
        const int ntm = lane >> 3, cc = lane & 7;
#pragma unroll
        for (int r = 0; r < 16; ++r) {
          const int src = (r & 7) * 4 + (cc >> 1);
          float m0 = 0.f, m1 = 0.f;
#pragma unroll
          for (int nt = 0; nt < 4; ++nt) {
            const float x0 = __shfl_sync(0xffffffffu, o4[nt][(r >= 8) * 2], src);
            const float x1 = __shfl_sync(0xffffffffu, o4[nt][(r >= 8) * 2 + 1], src);
            if (ntm == nt) {
              m0 = x0;
              m1 = x1;
            }
          }
          X[16 * B + r] = __fsub_rn(X[16 * B + r], (cc & 1) ? m1 : m0);
        }
      }
#pragma unroll
      for (int jj = 0; jj < 15; ++jj) {
        const float4* ar = reinterpret_cast<const float4*>(sAkkT + (16 * B + jj) * LDT + 16 * B);
        float a[16];
#pragma unroll
        for (int q4 = 0; q4 < 4; ++q4) {
          if (4 * q4 + 3 > jj) {
            const float4 x4 = ar[q4];
            a[4 * q4] = x4.x;
            a[4 * q4 + 1] = x4.y;
            a[4 * q4 + 2] = x4.z;
            a[4 * q4 + 3] = x4.w;
          }
        }
        const float nx = -X[16 * B + jj];
        int r = jj + 1;
        if ((16 - r) & 1) {
          X[16 * B + r] = fmaf(a[r], nx, X[16 * B + r]);
          ++r;
        }
#pragma unroll
        for (; r < 16; r += 2) fma2(X[16 * B + r], X[16 * B + r + 1], a[r], a[r + 1], nx, nx);
      }
    }
    if (hf == 0) {
#pragma unroll
      for (int t = 0; t < BT; ++t) UT[(ch * BT + t) * D + i] = t < n ? X[t] : 0.f;
    } else {
#pragma unroll
      for (int t = 0; t < BT; ++t) WG[ch * 16384 + cm16(t, i, 128)] = t < n ? X[t] : 0.f;
    }
    if (is_last && n < BT) {
#pragma unroll
      for (int t = 0; t < BT; ++t) {
        if (t < n) {
          float* Cr = cache + (((long)seq * H + h) * BT + t) * ROWF;
          if (hf == 0) {
            Cr[C_UT + i] = X[t];
            Cr[C_KN + i] = __fmul_rn(bf(rawK[t * LDB + i]), sRk[t]);
          } else {
            Cr[C_W + i] = X[t];
            Cr[C_G + i] = sG[t * LDF + i];
          }
        }
      }
    }
  }
  {
#pragma unroll 4
    for (int u = tid; u < 2048; u += 256) {
      const int kc = u >> 8, f = (u & 255) * 4;
      int t, kk;
      cmrc<KS>(f, t, kk);
      const int kq = kc * KS + kk;
      const float4 G4 = ld4(sG + t * LDF + kq);
      const uint2 qr = *reinterpret_cast<const uint2*>(rawQ + t * LDB + kq);
      const __nv_bfloat16* q4 = reinterpret_cast<const __nv_bfloat16*>(&qr);
      const float rq = sRq[t];
      float4 val = make_float4(0.f, 0.f, 0.f, 0.f);
      if (t < n)
        val = make_float4(__fmul_rn(ex2(G4.x), __fmul_rn(bf(q4[0]), rq)), __fmul_rn(ex2(G4.y), __fmul_rn(bf(q4[1]), rq)),
                          __fmul_rn(ex2(G4.z), __fmul_rn(bf(q4[2]), rq)), __fmul_rn(ex2(G4.w), __fmul_rn(bf(q4[3]), rq)));
      *reinterpret_cast<float4*>(WG + ch * 16384 + kc * 2048 + 1024 + f) = val;
    }
    if (n == BT) {
#pragma unroll 4
      for (int u = tid; u < 2048; u += 256) {
        const int jc = u >> 9, f = (u & 511) * 4;
        int kr, jj;
        cmrc<KS>(f, kr, jj);
        const float Ge = sG[(BT - 1) * LDF + kr];
        float r4[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int j = jc * KS + jj + e;
          r4[e] = __fmul_rn(__fmul_rn(bf(rawK[j * LDB + kr]), sRk[j]), ex2(__fsub_rn(Ge, sG[j * LDF + kr])));
        }
        *reinterpret_cast<float4*>(KDc + ch * 8192 + jc * 2048 + f) = make_float4(r4[0], r4[1], r4[2], r4[3]);
      }
      if (tid < 128) GEND[ch * D + tid] = ex2(sG[(BT - 1) * LDF + tid]);
    }
  }
}


template <int BV>
__global__ void __launch_bounds__(256, 1) k2s_kernel(
    const int* __restrict__ seq_chunks, const int* __restrict__ chunk_tab, const float* __restrict__ WG,
    const float* __restrict__ UT, const float* __restrict__ AQKT, const float* __restrict__ KDc,
    const float* __restrict__ GEND, __nv_bfloat16* __restrict__ out, float* __restrict__ Ebuf,
    float* __restrict__ cache, int* __restrict__ posbuf, int H) {
  using C = K2s<BV>;
  constexpr int NT = C::NT, LU = C::LU;
  extern __shared__ __align__(1024) unsigned char sm[];
  float* sWG = reinterpret_cast<float*>(sm + C::OFF_WG);
  float* sKD = reinterpret_cast<float*>(sm + C::OFF_KD);
  float* sAq = reinterpret_cast<float*>(sm + C::OFF_AQ);
  float* sE = reinterpret_cast<float*>(sm + C::OFF_E);
  float* sU = reinterpret_cast<float*>(sm + C::OFF_U);
  const int vt = blockIdx.x, n = blockIdx.y, h = blockIdx.z, v0 = vt * BV;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g8 = lane >> 2, t4 = lane & 3;
  float* E = Ebuf + ((long)n * H + h) * D * D;
  const int cb0 = seq_chunks[n], cb1 = seq_chunks[n + 1];
  uint64_t* mbA = reinterpret_cast<uint64_t*>(sm + C::OFF_MB);
  uint64_t* mbB = mbA + 1;
  auto issueA = [&](int c) {
    if (tid == 0 && c < cb1) {
      mb_expect(mbA, 16384 * 4);
      bulk_g2s(sWG, WG + ((long)c * H + h) * 16384, 16384 * 4, mbA);
    }
  };
  auto issueB = [&](int c) {
    if (tid == 0 && c < cb1) {
      const long chp = (long)c * H + h;
      const bool fl = chunk_tab[c * 4 + 2] == BT;
      mb_expect(mbB, BT * BT * 4 + (fl ? 8192 * 4 : 0));
      if (fl) bulk_g2s(sKD, KDc + chp * 8192, 8192 * 4, mbB);
      bulk_g2s(sAq, AQKT + chp * BT * BT, BT * BT * 4, mbB);
    }
  };
  if (tid == 0) {
    umma::mbar_init(mbA, 1);
    umma::mbar_init(mbB, 1);
    umma::fence_mbar_init();
  }
  __syncthreads();
  issueA(cb0);
  issueB(cb0);
  uint32_t phA = 0, phB = 0;
  for (int p = tid; p < BV * D / 4; p += 256) {
    const int vv = p / (D / 4), k4 = p % (D / 4);
    *reinterpret_cast<float4*>(sE + swE(vv, 4 * k4)) = ld4(E + (long)(v0 + vv) * D + 4 * k4);
  }
  __syncthreads();
  int pos = 0;
  const int r0 = 16 * warp + g8, r1 = r0 + 8;  // A rows of this lane (P: rows of [W;GQ]; P3: k)
  for (int c = cb0; c < cb1; ++c) {
    const int nrow = chunk_tab[c * 4 + 2];
    const long t0 = chunk_tab[c * 4 + 1];
    const long ch = (long)c * H + h;
    const bool full = nrow == BT;
    float ut[NT][4];
    if (warp < 4) {
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        const int col = v0 + nt * 8 + 2 * t4;
        const float2 a = *reinterpret_cast<const float2*>(UT + (ch * BT + r0) * D + col);
        const float2 b = *reinterpret_cast<const float2*>(UT + (ch * BT + r1) * D + col);
        ut[nt][0] = a.x;
        ut[nt][1] = a.y;
        ut[nt][2] = b.x;
        ut[nt][3] = b.y;
      }
    }
    float ge[2][2] = {{0.f, 0.f}, {0.f, 0.f}};
    if (full && warp < 4) {
#pragma unroll
      for (int mi = 0; mi < 2; ++mi) {
        ge[mi][0] = GEND[ch * D + 16 * (2 * warp + mi) + g8];
        ge[mi][1] = GEND[ch * D + 16 * (2 * warp + mi) + g8 + 8];
      }
    }
    umma::mbar_wait(mbA, phA);
    phA ^= 1;
    // ---- P
    float o[NT][4];
    {
      float big[NT][4], sml[NT][4];
#pragma unroll
      for (int nt = 0; nt < NT; ++nt)
#pragma unroll
        for (int e = 0; e < 4; ++e) big[nt][e] = sml[nt][e] = -0.f;
#pragma unroll 4
      for (int s = 0; s < 16; ++s) {
        const float* sl = sWG + (s >> 1) * 2048;
        const int kk = (s & 1) * 8 + t4;
        const float a[4] = {sl[cmi(r0, kk)], sl[cmi(r1, kk)], sl[cmi(r0, kk + 4)], sl[cmi(r1, kk + 4)]};
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
          const int vv = nt * 8 + g8;
          const float b[2] = {sE[swE(vv, 8 * s + t4)], sE[swE(vv, 8 * s + t4 + 4)]};
          step3s(big[nt], sml[nt], a, b);
        }
      }
#pragma unroll
      for (int nt = 0; nt < NT; ++nt)
#pragma unroll
        for (int e = 0; e < 4; ++e) o[nt][e] = __fadd_rn(big[nt][e], sml[nt][e]);
    }
    if (warp < 4) {
#pragma unroll
      for (int nt = 0; nt < NT; ++nt)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int t = e < 2 ? r0 : r1, vv = nt * 8 + 2 * t4 + (e & 1);
          const float u = __fsub_rn(ut[nt][e], o[nt][e]);
          sU[t * LU + vv] = u;
          if (!full && t < nrow) cache[(((long)n * H + h) * BT + t) * ROWF + C_U + v0 + vv] = u;
        }
    }
    __syncthreads();
    issueA(c + 1);
    umma::mbar_wait(mbB, phB);
    phB ^= 1;
    // ---- intra (warps 4..7: rows t = 16B + {g, g+8}, B = warp - 4)
    if (warp >= 4) {
      const int B = warp - 4, ta = 16 * B + g8, tb = ta + 8;
      if (B > 0) {
        float big[NT][4], sml[NT][4];
#pragma unroll
        for (int nt = 0; nt < NT; ++nt)
#pragma unroll
          for (int e = 0; e < 4; ++e) big[nt][e] = sml[nt][e] = -0.f;
        for (int s = 0; s < 2 * B; ++s) {
          const int kk = 8 * s + t4;
          const float a[4] = {sAq[kk * BT + ta], sAq[kk * BT + tb], sAq[(kk + 4) * BT + ta], sAq[(kk + 4) * BT + tb]};
#pragma unroll
          for (int nt = 0; nt < NT; ++nt) {
            const int vv = nt * 8 + g8;
            const float b[2] = {sU[kk * LU + vv], sU[(kk + 4) * LU + vv]};
            step3s(big[nt], sml[nt], a, b);
          }
        }
#pragma unroll
        for (int nt = 0; nt < NT; ++nt)
#pragma unroll
          for (int e = 0; e < 4; ++e) o[nt][e] = __fadd_rn(o[nt][e], __fadd_rn(big[nt][e], sml[nt][e]));
      }
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const int t = hh ? tb : ta;
        for (int j = 16 * B; j <= t; ++j) {
          const float a = sAq[j * BT + t];
#pragma unroll
          for (int nt = 0; nt < NT; ++nt) {
            const float2 x = *reinterpret_cast<const float2*>(sU + j * LU + nt * 8 + 2 * t4);
            o[nt][2 * hh] = fmaf(a, x.x, o[nt][2 * hh]);
            o[nt][2 * hh + 1] = fmaf(a, x.y, o[nt][2 * hh + 1]);
          }
        }
        if (t < nrow) {
#pragma unroll
          for (int nt = 0; nt < NT; ++nt) {
            __nv_bfloat162 pr;
            pr.x = __float2bfloat16(o[nt][2 * hh]);
            pr.y = __float2bfloat16(o[nt][2 * hh + 1]);
            *reinterpret_cast<__nv_bfloat162*>(out + ((t0 + t) * H + h) * D + v0 + nt * 8 + 2 * t4) = pr;
          }
        }
      }
    }
    // ---- P3 + state update (warps 0..3: k m-tiles 2w, 2w+1)
    if (full) {
      if (warp < 4) {
#pragma unroll
        for (int mi = 0; mi < 2; ++mi) {
          const int ka = 16 * (2 * warp + mi) + g8, kb = ka + 8;
          float big[NT][4], sml[NT][4];
#pragma unroll
          for (int nt = 0; nt < NT; ++nt)
#pragma unroll
            for (int e = 0; e < 4; ++e) big[nt][e] = sml[nt][e] = -0.f;
#pragma unroll 4
          for (int s = 0; s < 8; ++s) {
            const float* sl = sKD + (s >> 1) * 2048;
            const int kk = (s & 1) * 8 + t4;
            const float a[4] = {sl[cmi(ka, kk)], sl[cmi(kb, kk)], sl[cmi(ka, kk + 4)], sl[cmi(kb, kk + 4)]};
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
              const int vv = nt * 8 + g8;
              const float b[2] = {sU[(8 * s + t4) * LU + vv], sU[(8 * s + t4 + 4) * LU + vv]};
              step3s(big[nt], sml[nt], a, b);
            }
          }
#pragma unroll
          for (int nt = 0; nt < NT; ++nt)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
              const int kr = e < 2 ? ka : kb, vv = nt * 8 + 2 * t4 + (e & 1);
              float* pe = sE + swE(vv, kr);
              *pe = __fadd_rn(__fmul_rn(*pe, ge[mi][e >> 1]), __fadd_rn(big[nt][e], sml[nt][e]));
            }
        }
      }
      pos = 0;
    } else {
      pos = nrow;
    }
    __syncthreads();
    issueB(c + 1);
  }
  if (tid == 0 && vt == 0) posbuf[n * H + h] = pos;
  __syncthreads();
  for (int p = tid; p < BV * D / 4; p += 256) {
    const int vv = p / (D / 4), k4 = p % (D / 4);
    *reinterpret_cast<float4*>(E + (long)(v0 + vv) * D + 4 * k4) = ld4(sE + swE(vv, 4 * k4));
  }
}

template <int BV>
static void launch_k2s(int N, int H, cudaStream_t st, const int* sc, const int* tab, const float* WG,
                       const float* UT, const float* AQKT, const float* KDc, const float* GEND, __nv_bfloat16* out,
                       float* E, float* cache, int* pos) {
  const int smem = K2s<BV>::SMEM;
  cudaFuncSetAttribute(k2s_kernel<BV>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  k2s_kernel<BV><<<dim3(D / BV, N, H), 256, smem, st>>>(sc, tab, WG, UT, AQKT, KDc, GEND, out, E, cache, pos, H);
}

}  // namespace cf6p

std::vector<torch::Tensor> prefill6(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor g,
                                      torch::Tensor beta, torch::Tensor A_log, torch::Tensor bias,
                                      torch::Tensor chunk_tab, torch::Tensor seq_chunks, torch::Tensor E,
                                      torch::Tensor cache, torch::Tensor pos, double lower, int64_t bv) {
  using namespace cf6p;
  TORCH_CHECK(bv == 316 || bv == 332, "prefill: bv must be 316 or 332, got ", bv);
  const int NT = chunk_tab.size(0), N = seq_chunks.size(0) - 1, H = q.size(2);
  auto opt = q.options().dtype(torch::kFloat32);
  auto WG = torch::empty({NT, H, 16384}, opt), UT = torch::empty({NT, H, BT, D}, opt);
  auto AQKT = torch::empty({NT, H, BT, BT}, opt), KDc = torch::empty({NT, H, 8192}, opt);
  auto GEND = torch::empty({NT, H, D}, opt);
  auto out = torch::zeros_like(v);
  auto st = c10::cuda::getCurrentCUDAStream().stream();
  cudaFuncSetAttribute(k1s_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, KS_SMEM);
  k1s_kernel<<<dim3(NT, H), 256, KS_SMEM, st>>>(
      (const __nv_bfloat16*)q.data_ptr(), (const __nv_bfloat16*)k.data_ptr(), (const __nv_bfloat16*)v.data_ptr(),
      (const __nv_bfloat16*)g.data_ptr(), (const __nv_bfloat16*)beta.data_ptr(), A_log.data_ptr<float>(),
      bias.data_ptr<float>(), chunk_tab.data_ptr<int>(), WG.data_ptr<float>(), UT.data_ptr<float>(),
      AQKT.data_ptr<float>(), KDc.data_ptr<float>(), GEND.data_ptr<float>(), cache.data_ptr<float>(), H,
      (float)lower, 1.f / sqrtf((float)D));
  const int* sc = seq_chunks.data_ptr<int>();
  const int* tab = chunk_tab.data_ptr<int>();
  if (bv == 316)
    launch_k2s<16>(N, H, st, sc, tab, WG.data_ptr<float>(), UT.data_ptr<float>(), AQKT.data_ptr<float>(),
                   KDc.data_ptr<float>(), GEND.data_ptr<float>(), (__nv_bfloat16*)out.data_ptr(), E.data_ptr<float>(),
                   cache.data_ptr<float>(), pos.data_ptr<int>());
  else
    launch_k2s<32>(N, H, st, sc, tab, WG.data_ptr<float>(), UT.data_ptr<float>(), AQKT.data_ptr<float>(),
                   KDc.data_ptr<float>(), GEND.data_ptr<float>(), (__nv_bfloat16*)out.data_ptr(), E.data_ptr<float>(),
                   cache.data_ptr<float>(), pos.data_ptr<int>());
  return {out};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("prefill", &prefill6); }
