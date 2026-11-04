// Round 10 (perf P11): pure decode (rows_fast_slots: one token per row) past one wave (N*H > SMs) runs dsync9 =
// dsync7 at two CTAs per SM (R rows read from global, E staged where the {kn, G} rows were, beta read with its token
// stride); several tokens per row (rows_fast) keep dsync7.
// Round 9 (perf P8): dsync7 loads E straight into its P1/P2 fragments (plain global loads issued before Akk)
// and bulk-copies the R rows at kernel start next to the KG rows, so nothing waits for an E staging copy; P3 loads
// its E values before the products and the fragments are reloaded after P3 only when the row has another token.
// K31 CF-BI contract v5 decode engine `rows_fast`, round 8 = dsync7 (N*H*4 > 148) + dsplit<32> V-tiled CTAs for
// small batches. Round 7 (= dsync6 + E staged by per-row bulk copies into the
// R region, moved to registers, then the R rows are bulk-copied over it; first-token G rows loaded early).
// Round 6: (= dsync3 + first-token inputs loaded before posbuf, cache rows staged by per-row cp.async.bulk). Round-3 header: every product on mma.sync m16n8k8 tf32 (== UMMA).
// One CTA (256 threads, 8 warps, 1 CTA/SM) per (sequence, head); tokens in order.
//   * E is held in registers as P1/P2 B fragments (warp w: v rows [16w, 16w+16), all k), loaded once.
//   * cache rows j < pos are bulk-copied (cp.async 16B) into smem at kernel start: sKG[j] = {kn, G},
//     sR[j] = {UT, W, U}; rows produced by this kernel are written to smem and global.
//   * Akk/Aqk: A = {ks, qs}, B = kjr rows (warp w: j in [8w, 8w+8)); cross-block (v5) M from sR;
//     P3 at chunk end: A = KD rows k (warp w: k in [16w, 16w+16)), B = U rows, K = 64.
#include "kda_cfbi6.cu"

namespace cf6d8 {
using namespace cf5;
DEV float ldv(const float* p) {
  float x;
  asm volatile("ld.global.nc.f32 %0, [%1];" : "=f"(x) : "l"(p));
  return x;
}
DEV float ldc(const float* p) {
  float x;
  asm volatile("ld.global.f32 %0, [%1];" : "=f"(x) : "l"(p));
  return x;
}
DEV void step3(float* big, float* sml, const float* a, const float* b) {
  const uint32_t ah[4] = {__float_as_uint(a[0]), __float_as_uint(a[1]), __float_as_uint(a[2]), __float_as_uint(a[3])};
  const uint32_t al[4] = {mm::lob(a[0]), mm::lob(a[1]), mm::lob(a[2]), mm::lob(a[3])};
  const uint32_t bh[2] = {__float_as_uint(b[0]), __float_as_uint(b[1])};
  const uint32_t bl[2] = {mm::lob(b[0]), mm::lob(b[1])};
  mm::mma8(sml, al, bh);
  mm::mma8(sml, ah, bl);
  mm::mma8(big, ah, bh);
}

DEV void mbx(uint64_t* m, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(umma::smem_u32(m)), "r"(bytes) : "memory");
}
DEV void bulk(void* dst, const void* src, uint32_t bytes, uint64_t* m) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" ::"r"(
                   umma::smem_u32(dst)),
               "l"(src), "r"(bytes), "r"(umma::smem_u32(m))
               : "memory");
}

constexpr int LDE = 132;
constexpr int LKG = 260;  // {kn 128 | G 128} + 4   (bank = 4*row + col: conflict-free for (g8, t4))
constexpr int LR = 392;   // {UT 128 | W 128 | U 128} + 8 (bank = 8*row + col)
constexpr int R_UT = 0, R_W = D, R_U = 2 * D;
constexpr int OFF_KG = 0;
constexpr int OFF_R = OFF_KG + BT * LKG * 4;
constexpr int OFF_V = OFF_R + BT * LR * 4;
constexpr int NV = 18;
constexpr int OFF_MB = OFF_V + NV * D * 4 + 2 * BT * 4 + 64;
constexpr int SMEM = OFF_MB + 64;

__global__ void __launch_bounds__(256, 1) dsync7_kernel(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v, const __nv_bfloat16* __restrict__ g,
    const __nv_bfloat16* __restrict__ beta, const float* __restrict__ A_log,
    const float* __restrict__ bias, __nv_bfloat16* __restrict__ out, const int* __restrict__ cu,
    float* __restrict__ Ebuf, float* __restrict__ cache, int* __restrict__ posbuf, int H, float lower,
    float scale, const int* __restrict__ slots, long sEn, long sCn, long sPn, long sQ) {
  extern __shared__ __align__(1024) unsigned char sm[];
  float* sKG = reinterpret_cast<float*>(sm + OFF_KG);
  float* sR = reinterpret_cast<float*>(sm + OFF_R);
  float* V = reinterpret_cast<float*>(sm + OFF_V);
  float *sQv = V, *sKv = V + D, *sG = V + 2 * D, *sGr = V + 3 * D, *sKn = V + 4 * D, *sQn = V + 5 * D;
  float *sKs = V + 6 * D, *sQs = V + 7 * D, *sKjp = V + 8 * D, *sW = V + 9 * D, *sGQ = V + 10 * D;
  float *sUT = V + 11 * D, *sP1 = V + 12 * D, *sP2 = V + 13 * D, *sM = V + 14 * D, *sM2 = V + 16 * D;
  float* sAkk = V + NV * D;
  float* sAqk = sAkk + BT;
  float* sc = sAqk + BT;
  const int n = blockIdx.x, h = blockIdx.y, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g8 = lane >> 2, t4 = lane & 3;
  // kda-cfbi-slot: row n works on pool entry slots[n] in place (null slot = padding: no state access)
  const long sn = slots ? (long)slots[n] : (long)n;
  if (slots && sn == 0) {  // padding row: zero output, no state access
    for (long t = cu[n]; t < cu[n + 1]; ++t)
      if (tid < D) out[(t * H + h) * D + tid] = __float2bfloat16(0.f);
    return;
  }
  if (cu[n] >= cu[n + 1]) return;  // empty row: no TMA issued, state unchanged
  float* E = Ebuf + sn * sEn + (long)h * D * D;
  float* C = cache + sn * sCn + (long)h * BT * ROWF;
  uint64_t* mKG = reinterpret_cast<uint64_t*>(sm + OFF_MB);
  uint64_t* mR = mKG + 1;
  const long tf = cu[n];
  float fa = 0.f, fb = 0.f, fc = 0.f, fbeta = 0.f;
  if (tf < cu[n + 1]) {
    const long th = tf * H + h, tq = tf * sQ + (long)h * D;
    if (tid < 128) {
      fa = bf(g[th * D + tid]);
      fb = bf(q[tq + tid]);
      fc = bf(k[tq + tid]);
    } else {
      fa = bf(v[tq + tid - 128]);
    }
    if (tid == 0) fbeta = bf(beta[th]);
  }
  int pos = posbuf[sn * sPn + h];
  float gprev = 0.f, grr = 0.f;
  if (tid < 128) {
    const int rr0 = (pos / SUB) * SUB;
    if (pos > 0) gprev = ldc(C + (pos - 1) * ROWF + C_G + tid);
    if (rr0 != pos) grr = ldc(C + rr0 * ROWF + C_G + tid);
  }
  if (tid == 0) {
    umma::mbar_init(mKG, 1);
    umma::mbar_init(mR, 1);
    umma::fence_mbar_init();
    mbx(mKG, pos * 2 * D * 4);
    mbx(mR, pos * 3 * D * 4);
  }
  __syncthreads();
  if (tid >= 128 && tid < 192)
    for (int j = tid - 128; j < pos; j += 64) bulk(sKG + j * LKG, C + j * ROWF + C_KN, 2 * D * 4, mKG);
  else if (tid >= 192)
    for (int j = tid - 192; j < pos; j += 64) bulk(sR + j * LR, C + j * ROWF + C_UT, 3 * D * 4, mR);
  float eb[2][16][2];
  const float nah = head_nah(A_log[h]), low2 = __fmul_rn(lower, L2E);
  const float bias_i = bias[h * D + (tid & 127)];
  bool first = true;
  for (long t = cu[n]; t < cu[n + 1]; ++t) {
    const long th = t * H + h, tq = t * sQ + (long)h * D;
    const int rr = (pos / SUB) * SUB, b0 = pos & ~(SUB - 1);
    if (tid < 128) {
      const int i = tid;
      const float a2 = gate2(first ? fa : bf(g[th * D + i]), bias_i, nah, low2);
      sQv[i] = first ? fb : bf(q[tq + i]);
      sKv[i] = first ? fc : bf(k[tq + i]);
      if (first) {
        sG[i] = pos == 0 ? a2 : __fadd_rn(gprev, a2);
        sGr[i] = rr == pos ? sG[i] : grr;
      } else {
        sG[i] = pos == 0 ? a2 : __fadd_rn(sKG[(pos - 1) * LKG + D + i], a2);
        sGr[i] = rr == pos ? sG[i] : sKG[rr * LKG + D + i];
      }
    } else {
      sUT[tid - 128] = first ? fa : bf(v[tq + tid - 128]);
    }
    const bool was_first = first;
    first = false;
    __syncthreads();
    if (tid == 0 || tid == 32) {
      const float* x = tid == 0 ? sQv : sKv;
      float ss = 0.f;
#pragma unroll
      for (int k4 = 0; k4 < D / 4; ++k4) {
        const float4 y = *reinterpret_cast<const float4*>(x + 4 * k4);
        ss = fmaf(y.x, y.x, ss);
        ss = fmaf(y.y, y.y, ss);
        ss = fmaf(y.z, y.z, ss);
        ss = fmaf(y.w, y.w, ss);
      }
      if (tid == 0) {
        sc[0] = __fmul_rn(rnorm(ss), scale);
        sc[2] = sigm2(was_first ? fbeta : bf(beta[th]));
      } else {
        sc[1] = rnorm(ss);
      }
    }
    __syncthreads();
    if (tid < 128) {
      const int i = tid;
      const float G = sG[i], Gr = sGr[i];
      const float qn = __fmul_rn(sQv[i], sc[0]), kn = __fmul_rn(sKv[i], sc[1]);
      const float fr = ex2(__fsub_rn(G, Gr));
      sKn[i] = kn;
      sQn[i] = qn;
      sKs[i] = __fmul_rn(kn, fr);
      sQs[i] = __fmul_rn(qn, fr);
      sKjp[i] = __fmul_rn(kn, ex2(__fsub_rn(Gr, G)));
    }
    if (was_first) umma::mbar_wait(mKG, 0);
    __syncthreads();
    if (was_first) {  // E straight into the P1/P2 B fragments, in flight through Akk / cross / WY
#pragma unroll
      for (int nt = 0; nt < 2; ++nt) {
        const float* er = E + (16 * warp + 8 * nt + g8) * D;
#pragma unroll
        for (int s = 0; s < 16; ++s) {
          eb[nt][s][0] = ldc(er + 8 * s + t4);
          eb[nt][s][1] = ldc(er + 8 * s + t4 + 4);
        }
      }
    }
    // ---- Akk / Aqk
    const int jB = 8 * warp + g8;
    if (8 * warp <= pos) {
      float big[4] = {-0.f, -0.f, -0.f, -0.f}, sml[4] = {-0.f, -0.f, -0.f, -0.f};
      const float* kr = sKG + jB * LKG;
#pragma unroll
      for (int s = 0; s < 16; ++s) {
        const int k0 = 8 * s;
        float a[4], b[2];
        a[0] = g8 == 0 ? sKs[k0 + t4] : g8 == 1 ? sQs[k0 + t4] : 0.f;
        a[1] = 0.f;
        a[2] = g8 == 0 ? sKs[k0 + t4 + 4] : g8 == 1 ? sQs[k0 + t4 + 4] : 0.f;
        a[3] = 0.f;
        if (jB < pos) {
          b[0] = __fmul_rn(kr[k0 + t4], ex2(__fsub_rn(sGr[k0 + t4], kr[D + k0 + t4])));
          b[1] = __fmul_rn(kr[k0 + t4 + 4], ex2(__fsub_rn(sGr[k0 + t4 + 4], kr[D + k0 + t4 + 4])));
        } else if (jB == pos) {
          b[0] = sKjp[k0 + t4];
          b[1] = sKjp[k0 + t4 + 4];
        } else {
          b[0] = 0.f;
          b[1] = 0.f;
        }
        step3(big, sml, a, b);
      }
      if (g8 < 2) {
        const int j = 8 * warp + 2 * t4;
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const float d = __fadd_rn(big[e], sml[e]);
          if (j + e <= pos) {
            if (g8 == 0) sAkk[j + e] = __fmul_rn(sc[2], d);
            else sAqk[j + e] = d;
          }
        }
      }
    }
    if (was_first) umma::mbar_wait(mR, 0);
    __syncthreads();
    // ---- cross-block products (v5): warp w: WY n-tiles 4w..4w+3 (cols of {UT|W}), intra n-tiles 2w, 2w+1 (U)
    if (b0 > 0) {
      float big[6][4], sml[6][4];
#pragma unroll
      for (int q = 0; q < 6; ++q)
#pragma unroll
        for (int e = 0; e < 4; ++e) big[q][e] = sml[q][e] = -0.f;
#pragma unroll
      for (int s = 0; s < 6; ++s)
        if (8 * s < b0) {
          const int k0 = 8 * s, j0 = k0 + t4;
          const float ak[4] = {g8 == 0 ? sAkk[j0] : 0.f, 0.f, g8 == 0 ? sAkk[j0 + 4] : 0.f, 0.f};
          const float aq[4] = {g8 == 0 ? sAqk[j0] : 0.f, 0.f, g8 == 0 ? sAqk[j0 + 4] : 0.f, 0.f};
#pragma unroll
          for (int q4 = 0; q4 < 4; ++q4) {
            const int c = (warp * 4 + q4) * 8 + g8;  // 0..255 -> R_UT/R_W contiguous
            const float b[2] = {sR[j0 * LR + c], sR[(j0 + 4) * LR + c]};
            step3(big[q4], sml[q4], ak, b);
          }
#pragma unroll
          for (int q2 = 0; q2 < 2; ++q2) {
            const int c = R_U + (warp * 2 + q2) * 8 + g8;
            const float b[2] = {sR[j0 * LR + c], sR[(j0 + 4) * LR + c]};
            step3(big[4 + q2], sml[4 + q2], aq, b);
          }
        }
      if (lane < 4) {
#pragma unroll
        for (int q4 = 0; q4 < 4; ++q4) {
          const int c0 = (warp * 4 + q4) * 8;
          sM[c0 + 2 * lane] = __fadd_rn(big[q4][0], sml[q4][0]);
          sM[c0 + 2 * lane + 1] = __fadd_rn(big[q4][1], sml[q4][1]);
        }
#pragma unroll
        for (int q2 = 0; q2 < 2; ++q2) {
          const int c0 = (warp * 2 + q2) * 8;
          sM2[c0 + 2 * lane] = __fadd_rn(big[4 + q2][0], sml[4 + q2][0]);
          sM2[c0 + 2 * lane + 1] = __fadd_rn(big[4 + q2][1], sml[4 + q2][1]);
        }
      }
      __syncthreads();
    }
    // ---- WY row
    {
      const float b = sc[2];
      const int i = tid & 127;
      float x = tid < 128 ? __fmul_rn(b, sUT[i]) : __fmul_rn(b, __fmul_rn(ex2(sG[i]), sKn[i]));
      if (b0 > 0) x = __fsub_rn(x, sM[tid]);
      for (int j = b0; j < pos; ++j) x = fmaf(-sAkk[j], sR[j * LR + tid], x);
      sR[pos * LR + tid] = x;
      if (tid < 128) {
        C[pos * ROWF + C_UT + i] = x;
        C[pos * ROWF + C_KN + i] = sKn[i];
        sKG[pos * LKG + i] = sKn[i];
        sUT[i] = x;
      } else {
        C[pos * ROWF + C_W + i] = x;
        C[pos * ROWF + C_G + i] = sG[i];
        sKG[pos * LKG + D + i] = sG[i];
        sW[i] = x;
        sGQ[i] = __fmul_rn(ex2(sG[i]), sQn[i]);
      }
    }
    __syncthreads();
    // ---- P1 / P2
    {
      float big[2][4], sml[2][4];
#pragma unroll
      for (int nt = 0; nt < 2; ++nt)
#pragma unroll
        for (int e = 0; e < 4; ++e) big[nt][e] = sml[nt][e] = -0.f;
#pragma unroll
      for (int s = 0; s < 16; ++s) {
        const int k0 = 8 * s;
        float a[4];
        a[0] = g8 == 0 ? sW[k0 + t4] : g8 == 1 ? sGQ[k0 + t4] : 0.f;
        a[1] = 0.f;
        a[2] = g8 == 0 ? sW[k0 + t4 + 4] : g8 == 1 ? sGQ[k0 + t4 + 4] : 0.f;
        a[3] = 0.f;
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) step3(big[nt], sml[nt], a, eb[nt][s]);
      }
      if (g8 < 2) {
#pragma unroll
        for (int nt = 0; nt < 2; ++nt)
#pragma unroll
          for (int e = 0; e < 2; ++e) {
            const int vv = 16 * warp + 8 * nt + 2 * t4 + e;
            (g8 == 0 ? sP1 : sP2)[vv] = __fadd_rn(big[nt][e], sml[nt][e]);
          }
      }
    }
    __syncthreads();
    if (tid < 128) {
      const int vv = tid;
      const float u = __fsub_rn(sUT[vv], sP1[vv]);
      C[pos * ROWF + C_U + vv] = u;
      sR[pos * LR + R_U + vv] = u;
      float o = sP2[vv];
      if (b0 > 0) o = __fadd_rn(o, sM2[vv]);
      for (int j = b0; j < pos; ++j) o = fmaf(sAqk[j], sR[j * LR + R_U + vv], o);
      o = fmaf(sAqk[pos], u, o);
      out[th * D + vv] = __float2bfloat16(o);
    }
    __syncthreads();
    if (pos == BT - 1) {
      // ---- P3: E'[v][k] = E[v][k]*ex2(G[k]) + sum_j KD[k][j] U[j][v]
      const int kA = 16 * warp + g8, kB = kA + 8;
      const float gA = sG[kA], gB = sG[kB];
      const float geA = ex2(gA), geB = ex2(gB);
      for (int pass = 0; pass < 2; ++pass) {
        float big[8][4], sml[8][4], eo[8][4];
#pragma unroll
        for (int nt = 0; nt < 8; ++nt)
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            big[nt][e] = sml[nt][e] = -0.f;
            eo[nt][e] = ldc(E + (64 * pass + 8 * nt + 2 * t4 + (e & 1)) * D + (e < 2 ? kA : kB));
          }
#pragma unroll 2
        for (int s = 0; s < 8; ++s) {
          const int j0 = 8 * s + t4, j1 = j0 + 4;
          const float* r0 = sKG + j0 * LKG;
          const float* r1 = sKG + j1 * LKG;
          float a[4];
          a[0] = __fmul_rn(r0[kA], ex2(__fsub_rn(gA, r0[D + kA])));
          a[1] = __fmul_rn(r0[kB], ex2(__fsub_rn(gB, r0[D + kB])));
          a[2] = __fmul_rn(r1[kA], ex2(__fsub_rn(gA, r1[D + kA])));
          a[3] = __fmul_rn(r1[kB], ex2(__fsub_rn(gB, r1[D + kB])));
#pragma unroll
          for (int nt = 0; nt < 8; ++nt) {
            const int vv = 64 * pass + 8 * nt + g8;
            const float b[2] = {sR[j0 * LR + R_U + vv], sR[j1 * LR + R_U + vv]};
            step3(big[nt], sml[nt], a, b);
          }
        }
#pragma unroll
        for (int nt = 0; nt < 8; ++nt)
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const int kr = e < 2 ? kA : kB, vv = 64 * pass + 8 * nt + 2 * t4 + (e & 1);
            const float d = __fadd_rn(big[nt][e], sml[nt][e]);
            E[vv * D + kr] = __fadd_rn(__fmul_rn(eo[nt][e], e < 2 ? geA : geB), d);
          }
      }
      __syncthreads();
      if (t + 1 < cu[n + 1]) {  // the next token of this row runs P1/P2 on the new E
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
          const float* er = E + (16 * warp + 8 * nt + g8) * D;
#pragma unroll
          for (int s = 0; s < 16; ++s) {
            eb[nt][s][0] = ldc(er + 8 * s + t4);
            eb[nt][s][1] = ldc(er + 8 * s + t4 + 4);
          }
        }
      }
      pos = 0;
    } else {
      ++pos;
    }
    __syncthreads();
  }
  if (tid == 0) posbuf[sn * sPn + h] = pos;
}

// =============================================================================================
// dsync9 (perf P11): dsync7 for pure decode (one token per row) at two CTAs per SM.
//   dsync7 keeps the cache rows {UT, W, U} of the open chunk in smem (100 KB) next to the {kn, G} rows (66.5 KB)
//   and E in registers, which pins it to one 256-thread CTA per SM (5-6 waves at 64 rows x 12 heads). A one-token
//   call reads every R row element once, so dsync9 reads them from global (cross-block operands, WY and output
//   chains); E is staged into the {kn, G} region once Akk is done with it, and at a chunk end P3 reloads the {kn, G}
//   rows there and the U rows into their own region. 109.6 KB of smem and <= 128 registers: two CTAs per SM.
//   Same instructions on the same operand values as dsync7 (copied phase by phase), so the same bits.
// =============================================================================================
constexpr int LU = D + 8;  // U row stride of the P3 staging (bank = 8*row + col)
constexpr int X9 = (BT * LKG > D * LDE ? BT * LKG : D * LDE) * 4;  // {kn, G} rows, later the E staging
constexpr int OFF_U9 = X9;
constexpr int OFF_V9 = OFF_U9 + BT * LU * 4;
constexpr int OFF_MB9 = OFF_V9 + NV * D * 4 + 2 * BT * 4 + 64;
constexpr int SMEM9 = OFF_MB9 + 64;

__global__ void __launch_bounds__(256, 2) dsync9_kernel(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v, const __nv_bfloat16* __restrict__ g,
    const __nv_bfloat16* __restrict__ beta, const float* __restrict__ A_log,
    const float* __restrict__ bias, __nv_bfloat16* __restrict__ out, const int* __restrict__ cu,
    float* __restrict__ Ebuf, float* __restrict__ cache, int* __restrict__ posbuf, int H, float lower,
    float scale, const int* __restrict__ slots, long sEn, long sCn, long sPn, long sQ, long sB) {
  extern __shared__ __align__(1024) unsigned char sm[];
  float* sKG = reinterpret_cast<float*>(sm);
  float* sE = reinterpret_cast<float*>(sm);
  float* sU = reinterpret_cast<float*>(sm + OFF_U9);
  float* V = reinterpret_cast<float*>(sm + OFF_V9);
  float *sQv = V, *sKv = V + D, *sG = V + 2 * D, *sGr = V + 3 * D, *sKn = V + 4 * D, *sQn = V + 5 * D;
  float *sKs = V + 6 * D, *sQs = V + 7 * D, *sKjp = V + 8 * D, *sW = V + 9 * D, *sGQ = V + 10 * D;
  float *sUT = V + 11 * D, *sP1 = V + 12 * D, *sP2 = V + 13 * D, *sM = V + 14 * D, *sM2 = V + 16 * D;
  float* sAkk = V + NV * D;
  float* sAqk = sAkk + BT;
  float* sc = sAqk + BT;
  const int n = blockIdx.x, h = blockIdx.y, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g8 = lane >> 2, t4 = lane & 3;
  const long sn = slots ? (long)slots[n] : (long)n;
  if (slots && sn == 0) {  // padding row: zero output, no state access
    for (long t = cu[n]; t < cu[n + 1]; ++t)
      if (tid < D) out[(t * H + h) * D + tid] = __float2bfloat16(0.f);
    return;
  }
  const long t = cu[n];
  if (t >= cu[n + 1]) return;  // empty row: state unchanged
  if (cu[n + 1] - t != 1) __trap();  // one token per row
  float* E = Ebuf + sn * sEn + (long)h * D * D;
  float* C = cache + sn * sCn + (long)h * BT * ROWF;
  uint64_t* mKG = reinterpret_cast<uint64_t*>(sm + OFF_MB9);
  uint64_t* mE = mKG + 1;
  uint64_t* mP3 = mKG + 2;
  const long th = t * H + h, tq = t * sQ + (long)h * D;
  float fa, fb = 0.f, fc = 0.f, fbeta = 0.f;
  if (tid < 128) {
    fa = bf(g[th * D + tid]);
    fb = bf(q[tq + tid]);
    fc = bf(k[tq + tid]);
  } else {
    fa = bf(v[tq + tid - 128]);
  }
  if (tid == 0) fbeta = bf(beta[t * sB + h]);
  const int pos = posbuf[sn * sPn + h];
  const int rr = (pos / SUB) * SUB, b0 = pos & ~(SUB - 1);
  float gprev = 0.f, grr = 0.f;
  if (tid < 128) {
    if (pos > 0) gprev = ldc(C + (pos - 1) * ROWF + C_G + tid);
    if (rr != pos) grr = ldc(C + rr * ROWF + C_G + tid);
  }
  if (tid == 0) {
    umma::mbar_init(mKG, 1);
    umma::mbar_init(mE, 1);
    umma::mbar_init(mP3, 1);
    umma::fence_mbar_init();
    mbx(mKG, pos * 2 * D * 4);
  }
  __syncthreads();
  if (tid >= 128 && tid < 192)
    for (int j = tid - 128; j < pos; j += 64) bulk(sKG + j * LKG, C + j * ROWF + C_KN, 2 * D * 4, mKG);
  const float nah = head_nah(A_log[h]), low2 = __fmul_rn(lower, L2E);
  const float bias_i = bias[h * D + (tid & 127)];
  if (tid < 128) {
    const int i = tid;
    const float a2 = gate2(fa, bias_i, nah, low2);
    sQv[i] = fb;
    sKv[i] = fc;
    sG[i] = pos == 0 ? a2 : __fadd_rn(gprev, a2);
    sGr[i] = rr == pos ? sG[i] : grr;
  } else {
    sUT[tid - 128] = fa;
  }
  __syncthreads();
  if (tid == 0 || tid == 32) {
    const float* x = tid == 0 ? sQv : sKv;
    float ss = 0.f;
#pragma unroll
    for (int k4 = 0; k4 < D / 4; ++k4) {
      const float4 y = *reinterpret_cast<const float4*>(x + 4 * k4);
      ss = fmaf(y.x, y.x, ss);
      ss = fmaf(y.y, y.y, ss);
      ss = fmaf(y.z, y.z, ss);
      ss = fmaf(y.w, y.w, ss);
    }
    if (tid == 0) {
      sc[0] = __fmul_rn(rnorm(ss), scale);
      sc[2] = sigm2(fbeta);
    } else {
      sc[1] = rnorm(ss);
    }
  }
  __syncthreads();
  if (tid < 128) {
    const int i = tid;
    const float G = sG[i], Gr = sGr[i];
    const float qn = __fmul_rn(sQv[i], sc[0]), kn = __fmul_rn(sKv[i], sc[1]);
    const float fr = ex2(__fsub_rn(G, Gr));
    sKn[i] = kn;
    sQn[i] = qn;
    sKs[i] = __fmul_rn(kn, fr);
    sQs[i] = __fmul_rn(qn, fr);
    sKjp[i] = __fmul_rn(kn, ex2(__fsub_rn(Gr, G)));
  }
  umma::mbar_wait(mKG, 0);
  __syncthreads();
  // ---- Akk / Aqk (dsync7)
  const int jB = 8 * warp + g8;
  if (8 * warp <= pos) {
    float big[4] = {-0.f, -0.f, -0.f, -0.f}, sml[4] = {-0.f, -0.f, -0.f, -0.f};
    const float* kr = sKG + jB * LKG;
#pragma unroll
    for (int s = 0; s < 16; ++s) {
      const int k0 = 8 * s;
      float a[4], b[2];
      a[0] = g8 == 0 ? sKs[k0 + t4] : g8 == 1 ? sQs[k0 + t4] : 0.f;
      a[1] = 0.f;
      a[2] = g8 == 0 ? sKs[k0 + t4 + 4] : g8 == 1 ? sQs[k0 + t4 + 4] : 0.f;
      a[3] = 0.f;
      if (jB < pos) {
        b[0] = __fmul_rn(kr[k0 + t4], ex2(__fsub_rn(sGr[k0 + t4], kr[D + k0 + t4])));
        b[1] = __fmul_rn(kr[k0 + t4 + 4], ex2(__fsub_rn(sGr[k0 + t4 + 4], kr[D + k0 + t4 + 4])));
      } else if (jB == pos) {
        b[0] = sKjp[k0 + t4];
        b[1] = sKjp[k0 + t4 + 4];
      } else {
        b[0] = 0.f;
        b[1] = 0.f;
      }
      step3(big, sml, a, b);
    }
    if (g8 < 2) {
      const int j = 8 * warp + 2 * t4;
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const float d = __fadd_rn(big[e], sml[e]);
        if (j + e <= pos) {
          if (g8 == 0) sAkk[j + e] = __fmul_rn(sc[2], d);
          else sAqk[j + e] = d;
        }
      }
    }
  }
  // the {kn, G} region becomes the E staging (generic reads above, async writes below)
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
  if (tid == 0) mbx(mE, D * D * 4);
  __syncthreads();
  if (tid < 128) bulk(sE + tid * LDE, E + tid * D, D * 4, mE);
  // ---- cross-block products (v5), operands = R rows j < b0 from global; warp w: {UT|W} n-tiles 4w..4w+3, then
  //      U n-tiles 2w, 2w+1 (dsync7 order and fragments)
  if (b0 > 0) {
    const int ns = b0 / 8;
    {
      float bv[6][4][2];
#pragma unroll
      for (int s = 0; s < 6; ++s)
        if (s < ns) {
          const float* r0 = C + (8 * s + t4) * ROWF + C_UT;
#pragma unroll
          for (int q4 = 0; q4 < 4; ++q4) {
            const int c = (warp * 4 + q4) * 8 + g8;
            bv[s][q4][0] = ldc(r0 + c);
            bv[s][q4][1] = ldc(r0 + 4 * ROWF + c);
          }
        }
      float big[4][4], sml[4][4];
#pragma unroll
      for (int q = 0; q < 4; ++q)
#pragma unroll
        for (int e = 0; e < 4; ++e) big[q][e] = sml[q][e] = -0.f;
#pragma unroll
      for (int s = 0; s < 6; ++s)
        if (s < ns) {
          const int j0 = 8 * s + t4;
          const float ak[4] = {g8 == 0 ? sAkk[j0] : 0.f, 0.f, g8 == 0 ? sAkk[j0 + 4] : 0.f, 0.f};
#pragma unroll
          for (int q4 = 0; q4 < 4; ++q4) step3(big[q4], sml[q4], ak, bv[s][q4]);
        }
      if (lane < 4) {
#pragma unroll
        for (int q4 = 0; q4 < 4; ++q4) {
          const int c0 = (warp * 4 + q4) * 8;
          sM[c0 + 2 * lane] = __fadd_rn(big[q4][0], sml[q4][0]);
          sM[c0 + 2 * lane + 1] = __fadd_rn(big[q4][1], sml[q4][1]);
        }
      }
    }
    {
      float bv[6][2][2];
#pragma unroll
      for (int s = 0; s < 6; ++s)
        if (s < ns) {
          const float* r0 = C + (8 * s + t4) * ROWF + C_U;
#pragma unroll
          for (int q2 = 0; q2 < 2; ++q2) {
            const int c = (warp * 2 + q2) * 8 + g8;
            bv[s][q2][0] = ldc(r0 + c);
            bv[s][q2][1] = ldc(r0 + 4 * ROWF + c);
          }
        }
      float big[2][4], sml[2][4];
#pragma unroll
      for (int q = 0; q < 2; ++q)
#pragma unroll
        for (int e = 0; e < 4; ++e) big[q][e] = sml[q][e] = -0.f;
#pragma unroll
      for (int s = 0; s < 6; ++s)
        if (s < ns) {
          const int j0 = 8 * s + t4;
          const float aq[4] = {g8 == 0 ? sAqk[j0] : 0.f, 0.f, g8 == 0 ? sAqk[j0 + 4] : 0.f, 0.f};
#pragma unroll
          for (int q2 = 0; q2 < 2; ++q2) step3(big[q2], sml[q2], aq, bv[s][q2]);
        }
      if (lane < 4) {
#pragma unroll
        for (int q2 = 0; q2 < 2; ++q2) {
          const int c0 = (warp * 2 + q2) * 8;
          sM2[c0 + 2 * lane] = __fadd_rn(big[q2][0], sml[q2][0]);
          sM2[c0 + 2 * lane + 1] = __fadd_rn(big[q2][1], sml[q2][1]);
        }
      }
    }
    __syncthreads();
  }
  // ---- WY row (rows b0 .. pos-1 of {UT|W} from global)
  {
    float rw[SUB - 1];
#pragma unroll
    for (int jj = 0; jj < SUB - 1; ++jj)
      if (b0 + jj < pos) rw[jj] = ldc(C + (b0 + jj) * ROWF + C_UT + tid);
    const float b = sc[2];
    const int i = tid & 127;
    float x = tid < 128 ? __fmul_rn(b, sUT[i]) : __fmul_rn(b, __fmul_rn(ex2(sG[i]), sKn[i]));
    if (b0 > 0) x = __fsub_rn(x, sM[tid]);
#pragma unroll
    for (int jj = 0; jj < SUB - 1; ++jj)
      if (b0 + jj < pos) x = fmaf(-sAkk[b0 + jj], rw[jj], x);
    if (tid < 128) {
      C[pos * ROWF + C_UT + i] = x;
      C[pos * ROWF + C_KN + i] = sKn[i];
      sUT[i] = x;
    } else {
      C[pos * ROWF + C_W + i] = x;
      C[pos * ROWF + C_G + i] = sG[i];
      sW[i] = x;
      sGQ[i] = __fmul_rn(ex2(sG[i]), sQn[i]);
    }
  }
  float ru[SUB - 1];  // U rows b0 .. pos-1 for the output chain, in flight through P1/P2
  if (tid < 128) {
#pragma unroll
    for (int jj = 0; jj < SUB - 1; ++jj)
      if (b0 + jj < pos) ru[jj] = ldc(C + (b0 + jj) * ROWF + C_U + tid);
  }
  umma::mbar_wait(mE, 0);
  __syncthreads();
  // ---- P1 / P2 (dsync7, E fragments from the staging)
  {
    float big[2][4], sml[2][4];
#pragma unroll
    for (int nt = 0; nt < 2; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e) big[nt][e] = sml[nt][e] = -0.f;
#pragma unroll
    for (int s = 0; s < 16; ++s) {
      const int k0 = 8 * s;
      float a[4];
      a[0] = g8 == 0 ? sW[k0 + t4] : g8 == 1 ? sGQ[k0 + t4] : 0.f;
      a[1] = 0.f;
      a[2] = g8 == 0 ? sW[k0 + t4 + 4] : g8 == 1 ? sGQ[k0 + t4 + 4] : 0.f;
      a[3] = 0.f;
#pragma unroll
      for (int nt = 0; nt < 2; ++nt) {
        const float* er = sE + (16 * warp + 8 * nt + g8) * LDE;
        const float b[2] = {er[k0 + t4], er[k0 + t4 + 4]};
        step3(big[nt], sml[nt], a, b);
      }
    }
    if (g8 < 2) {
#pragma unroll
      for (int nt = 0; nt < 2; ++nt)
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int vv = 16 * warp + 8 * nt + 2 * t4 + e;
          (g8 == 0 ? sP1 : sP2)[vv] = __fadd_rn(big[nt][e], sml[nt][e]);
        }
    }
  }
  __syncthreads();
  if (tid < 128) {
    const int vv = tid;
    const float u = __fsub_rn(sUT[vv], sP1[vv]);
    C[pos * ROWF + C_U + vv] = u;
    if (pos == BT - 1) sU[pos * LU + vv] = u;
    float o = sP2[vv];
    if (b0 > 0) o = __fadd_rn(o, sM2[vv]);
#pragma unroll
    for (int jj = 0; jj < SUB - 1; ++jj)
      if (b0 + jj < pos) o = fmaf(sAqk[b0 + jj], ru[jj], o);
    o = fmaf(sAqk[pos], u, o);
    out[th * D + vv] = __float2bfloat16(o);
  }
  if (pos == BT - 1) {
    // ---- P3: E'[v][k] = E[v][k]*ex2(G[k]) + sum_j KD[k][j] U[j][v] (dsync7 products; {kn, G} rows reloaded into
    //      the region the E staging used, U rows into their own; row 63 from this token's vectors)
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    if (tid == 0) mbx(mP3, (BT - 1) * 3 * D * 4);
    __syncthreads();
    if (tid < 128) {
      sKG[(BT - 1) * LKG + tid] = sKn[tid];
      sKG[(BT - 1) * LKG + D + tid] = sG[tid];
    } else if (tid < 192) {
      for (int j = tid - 128; j < BT - 1; j += 64) {
        bulk(sKG + j * LKG, C + j * ROWF + C_KN, 2 * D * 4, mP3);
        bulk(sU + j * LU, C + j * ROWF + C_U, D * 4, mP3);
      }
    }
    umma::mbar_wait(mP3, 0);
    __syncthreads();
    const int kA = 16 * warp + g8, kB = kA + 8;
    const float gA = sG[kA], gB = sG[kB];
    const float geA = ex2(gA), geB = ex2(gB);
#pragma unroll 1
    for (int pass = 0; pass < 4; ++pass) {
      float big[4][4], sml[4][4], eo[4][4];
#pragma unroll
      for (int nt = 0; nt < 4; ++nt)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          big[nt][e] = sml[nt][e] = -0.f;
          eo[nt][e] = ldc(E + (32 * pass + 8 * nt + 2 * t4 + (e & 1)) * D + (e < 2 ? kA : kB));
        }
#pragma unroll 2
      for (int s = 0; s < 8; ++s) {
        const int j0 = 8 * s + t4, j1 = j0 + 4;
        const float* r0 = sKG + j0 * LKG;
        const float* r1 = sKG + j1 * LKG;
        float a[4];
        a[0] = __fmul_rn(r0[kA], ex2(__fsub_rn(gA, r0[D + kA])));
        a[1] = __fmul_rn(r0[kB], ex2(__fsub_rn(gB, r0[D + kB])));
        a[2] = __fmul_rn(r1[kA], ex2(__fsub_rn(gA, r1[D + kA])));
        a[3] = __fmul_rn(r1[kB], ex2(__fsub_rn(gB, r1[D + kB])));
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
          const int vv = 32 * pass + 8 * nt + g8;
          const float b[2] = {sU[j0 * LU + vv], sU[j1 * LU + vv]};
          step3(big[nt], sml[nt], a, b);
        }
      }
#pragma unroll
      for (int nt = 0; nt < 4; ++nt)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int kr = e < 2 ? kA : kB, vv = 32 * pass + 8 * nt + 2 * t4 + (e & 1);
          const float d = __fadd_rn(big[nt][e], sml[nt][e]);
          E[vv * D + kr] = __fadd_rn(__fmul_rn(eo[nt][e], e < 2 ? geA : geB), d);
        }
    }
  }
  if (tid == 0) posbuf[sn * sPn + h] = pos == BT - 1 ? 0 : pos + 1;
}

// =============================================================================================
// dsplit (round 8): one CTA per (sequence, head, V-tile of BV columns) for small batches.
//   Each CTA redundantly computes Akk/Aqk and the full W row (128 k columns); it owns UT/U/E/out columns
//   [v0, v0 + BV). Only V-tile 0 writes the shared cache columns (KN, G, W) and posbuf.
//   smem: sKG [64][260] (kn|G rows), sR [64][LRS] = {UT slice | W | U slice} rows, sE [BV][132] (E slice).
// =============================================================================================
template <int BV>
struct SP {
  static constexpr int LRS = 2 * BV + D + 8;
  static constexpr int R_UTs = 0, R_Ws = BV, R_Us = BV + D;
  static constexpr int OFF_KG = 0;
  static constexpr int OFF_R = OFF_KG + BT * LKG * 4;
  static constexpr int OFF_E = OFF_R + BT * LRS * 4;
  static constexpr int OFF_V = OFF_E + BV * LDE * 4;
  static constexpr int OFF_MB = OFF_V + NV * D * 4 + 2 * BT * 4 + 64;
  static constexpr int SMEM = OFF_MB + 64;
  static constexpr int NWY = (BV + D) / 8, NTT = NWY + BV / 8;  // n-tiles: {UT|W} then U
};

template <int BV>
__global__ void __launch_bounds__(256, 1) dsplit_kernel(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v, const __nv_bfloat16* __restrict__ g,
    const __nv_bfloat16* __restrict__ beta, const float* __restrict__ A_log,
    const float* __restrict__ bias, __nv_bfloat16* __restrict__ out, const int* __restrict__ cu,
    float* __restrict__ Ebuf, float* __restrict__ cache, int* __restrict__ posbuf, int H, float lower,
    float scale, const int* __restrict__ slots, long sEn, long sCn, long sPn, long sQ) {
  using S = SP<BV>;
  constexpr int LRS = S::LRS;
  extern __shared__ __align__(1024) unsigned char sm[];
  float* sKG = reinterpret_cast<float*>(sm + S::OFF_KG);
  float* sR = reinterpret_cast<float*>(sm + S::OFF_R);
  float* sE = reinterpret_cast<float*>(sm + S::OFF_E);
  float* V = reinterpret_cast<float*>(sm + S::OFF_V);
  float *sQv = V, *sKv = V + D, *sG = V + 2 * D, *sGr = V + 3 * D, *sKn = V + 4 * D, *sQn = V + 5 * D;
  float *sKs = V + 6 * D, *sQs = V + 7 * D, *sKjp = V + 8 * D, *sW = V + 9 * D, *sGQ = V + 10 * D;
  float *sUT = V + 11 * D, *sP1 = V + 12 * D, *sP2 = V + 13 * D, *sM = V + 14 * D, *sM2 = V + 16 * D;
  float* sAkk = V + NV * D;
  float* sAqk = sAkk + BT;
  float* sc = sAqk + BT;
  uint64_t* mKG = reinterpret_cast<uint64_t*>(sm + S::OFF_MB);
  uint64_t* mR = mKG + 1;
  uint64_t* mE = mKG + 2;
  const int vt = blockIdx.x, n = blockIdx.y, h = blockIdx.z, v0 = vt * BV;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g8 = lane >> 2, t4 = lane & 3;
  // kda-cfbi-slot: row n works on pool entry slots[n] in place (null slot = padding: no state access)
  const long sn = slots ? (long)slots[n] : (long)n;
  if (slots && sn == 0) {  // padding row: zero this tile of the output, no state access
    for (long t = cu[n]; t < cu[n + 1]; ++t)
      if (tid < BV) out[(t * H + h) * D + v0 + tid] = __float2bfloat16(0.f);
    return;
  }
  if (cu[n] >= cu[n + 1]) return;  // empty row: no TMA issued, state unchanged
  float* E = Ebuf + sn * sEn + (long)h * D * D;
  float* C = cache + sn * sCn + (long)h * BT * ROWF;
  const long tf = cu[n];
  float fa = 0.f, fb = 0.f, fc = 0.f, fv = 0.f, fbeta = 0.f;
  if (tf < cu[n + 1]) {
    const long th = tf * H + h, tq = tf * sQ + (long)h * D;
    if (tid < 128) {
      fa = bf(g[th * D + tid]);
      fb = bf(q[tq + tid]);
      fc = bf(k[tq + tid]);
      if (tid < BV) fv = bf(v[tq + v0 + tid]);
    }
    if (tid == 0) fbeta = bf(beta[th]);
  }
  int pos = posbuf[sn * sPn + h];
  float gprev = 0.f, grr = 0.f;
  if (tid < 128) {
    const int rr0 = (pos / SUB) * SUB;
    if (pos > 0) gprev = ldc(C + (pos - 1) * ROWF + C_G + tid);
    if (rr0 != pos) grr = ldc(C + rr0 * ROWF + C_G + tid);
  }
  if (tid == 0) {
    umma::mbar_init(mKG, 1);
    umma::mbar_init(mR, 1);
    umma::mbar_init(mE, 1);
    umma::fence_mbar_init();
    mbx(mE, BV * D * 4);
    mbx(mKG, pos * 2 * D * 4);
    mbx(mR, pos * (2 * BV + D) * 4);
  }
  __syncthreads();
  if (tid < BV) {
    bulk(sE + tid * LDE, E + (long)(v0 + tid) * D, D * 4, mE);
  } else if (tid >= 64 && tid < 128) {
    for (int j = tid - 64; j < pos; j += 64) bulk(sKG + j * LKG, C + j * ROWF + C_KN, 2 * D * 4, mKG);
  } else if (tid >= 128 && tid < 192) {
    for (int j = tid - 128; j < pos; j += 64) {
      float* d = sR + j * LRS;
      bulk(d + S::R_UTs, C + j * ROWF + C_UT + v0, BV * 4, mR);
      bulk(d + S::R_Ws, C + j * ROWF + C_W, D * 4, mR);
      bulk(d + S::R_Us, C + j * ROWF + C_U + v0, BV * 4, mR);
    }
  }
  const float nah = head_nah(A_log[h]), low2 = __fmul_rn(lower, L2E);
  const float bias_i = bias[h * D + (tid & 127)];
  bool first = true;
  for (long t = cu[n]; t < cu[n + 1]; ++t) {
    const long th = t * H + h, tq = t * sQ + (long)h * D;
    const int rr = (pos / SUB) * SUB, b0 = pos & ~(SUB - 1);
    if (tid < 128) {
      const int i = tid;
      const float a2 = gate2(first ? fa : bf(g[th * D + i]), bias_i, nah, low2);
      sQv[i] = first ? fb : bf(q[tq + i]);
      sKv[i] = first ? fc : bf(k[tq + i]);
      if (first) {
        sG[i] = pos == 0 ? a2 : __fadd_rn(gprev, a2);
        sGr[i] = rr == pos ? sG[i] : grr;
      } else {
        sG[i] = pos == 0 ? a2 : __fadd_rn(sKG[(pos - 1) * LKG + D + i], a2);
        sGr[i] = rr == pos ? sG[i] : sKG[rr * LKG + D + i];
      }
      if (i < BV) sUT[i] = first ? fv : bf(v[tq + v0 + i]);
    }
    const bool was_first = first;
    first = false;
    __syncthreads();
    if (tid == 0 || tid == 32) {
      const float* x = tid == 0 ? sQv : sKv;
      float ss = 0.f;
#pragma unroll
      for (int k4 = 0; k4 < D / 4; ++k4) {
        const float4 y = *reinterpret_cast<const float4*>(x + 4 * k4);
        ss = fmaf(y.x, y.x, ss);
        ss = fmaf(y.y, y.y, ss);
        ss = fmaf(y.z, y.z, ss);
        ss = fmaf(y.w, y.w, ss);
      }
      if (tid == 0) {
        sc[0] = __fmul_rn(rnorm(ss), scale);
        sc[2] = sigm2(was_first ? fbeta : bf(beta[th]));
      } else {
        sc[1] = rnorm(ss);
      }
    }
    __syncthreads();
    if (tid < 128) {
      const int i = tid;
      const float G = sG[i], Gr = sGr[i];
      const float qn = __fmul_rn(sQv[i], sc[0]), kn = __fmul_rn(sKv[i], sc[1]);
      const float fr = ex2(__fsub_rn(G, Gr));
      sKn[i] = kn;
      sQn[i] = qn;
      sKs[i] = __fmul_rn(kn, fr);
      sQs[i] = __fmul_rn(qn, fr);
      sKjp[i] = __fmul_rn(kn, ex2(__fsub_rn(Gr, G)));
    }
    if (was_first) umma::mbar_wait(mKG, 0);
    __syncthreads();
    // ---- Akk / Aqk (all warps, identical in every V-tile)
    const int jB = 8 * warp + g8;
    if (8 * warp <= pos) {
      float big[4] = {-0.f, -0.f, -0.f, -0.f}, sml[4] = {-0.f, -0.f, -0.f, -0.f};
      const float* kr = sKG + jB * LKG;
#pragma unroll
      for (int s = 0; s < 16; ++s) {
        const int k0 = 8 * s;
        float a[4], b[2];
        a[0] = g8 == 0 ? sKs[k0 + t4] : g8 == 1 ? sQs[k0 + t4] : 0.f;
        a[1] = 0.f;
        a[2] = g8 == 0 ? sKs[k0 + t4 + 4] : g8 == 1 ? sQs[k0 + t4 + 4] : 0.f;
        a[3] = 0.f;
        if (jB < pos) {
          b[0] = __fmul_rn(kr[k0 + t4], ex2(__fsub_rn(sGr[k0 + t4], kr[D + k0 + t4])));
          b[1] = __fmul_rn(kr[k0 + t4 + 4], ex2(__fsub_rn(sGr[k0 + t4 + 4], kr[D + k0 + t4 + 4])));
        } else if (jB == pos) {
          b[0] = sKjp[k0 + t4];
          b[1] = sKjp[k0 + t4 + 4];
        } else {
          b[0] = 0.f;
          b[1] = 0.f;
        }
        step3(big, sml, a, b);
      }
      if (g8 < 2) {
        const int j = 8 * warp + 2 * t4;
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const float d = __fadd_rn(big[e], sml[e]);
          if (j + e <= pos) {
            if (g8 == 0) sAkk[j + e] = __fmul_rn(sc[2], d);
            else sAqk[j + e] = d;
          }
        }
      }
    }
    if (was_first) umma::mbar_wait(mR, 0);
    __syncthreads();
    // ---- cross-block products (v5): n-tiles q = warp + 8i over {UT slice | W} (A = Akk) and U slice (A = Aqk)
    if (b0 > 0) {
      for (int qq = warp; qq < S::NTT; qq += 8) {
        const bool isU = qq >= S::NWY;
        const int c = qq * 8 + g8;  // column inside sR row (UT slice | W | U slice are contiguous)
        float big[4] = {-0.f, -0.f, -0.f, -0.f}, sml[4] = {-0.f, -0.f, -0.f, -0.f};
#pragma unroll
        for (int s = 0; s < 6; ++s)
          if (8 * s < b0) {
            const int j0 = 8 * s + t4;
            const float* A = isU ? sAqk : sAkk;
            const float a[4] = {g8 == 0 ? A[j0] : 0.f, 0.f, g8 == 0 ? A[j0 + 4] : 0.f, 0.f};
            const float b[2] = {sR[j0 * LRS + c], sR[(j0 + 4) * LRS + c]};
            step3(big, sml, a, b);
          }
        if (lane < 4) {
          const int c0 = qq * 8 + 2 * lane;
          float* dst = isU ? sM2 + (c0 - S::NWY * 8) : sM + c0;
          dst[0] = __fadd_rn(big[0], sml[0]);
          dst[1] = __fadd_rn(big[1], sml[1]);
        }
      }
      __syncthreads();
    }
    // ---- WY row: columns c < BV + 128 of {UT slice | W}
    if (tid < BV + D) {
      const float b = sc[2];
      const int c = tid;
      const bool isW = c >= BV;
      const int i = isW ? c - BV : c;
      float x = isW ? __fmul_rn(b, __fmul_rn(ex2(sG[i]), sKn[i])) : __fmul_rn(b, sUT[i]);
      if (b0 > 0) x = __fsub_rn(x, sM[c]);
      for (int j = b0; j < pos; ++j) x = fmaf(-sAkk[j], sR[j * LRS + c], x);
      sR[pos * LRS + c] = x;
      if (!isW) {
        C[pos * ROWF + C_UT + v0 + i] = x;
        sUT[i] = x;
      } else {
        sW[i] = x;
        sGQ[i] = __fmul_rn(ex2(sG[i]), sQn[i]);
        if (vt == 0) {
          C[pos * ROWF + C_W + i] = x;
          C[pos * ROWF + C_KN + i] = sKn[i];
          C[pos * ROWF + C_G + i] = sG[i];
        }
      }
    }
    if (tid < D) {
      sKG[pos * LKG + tid] = sKn[tid];
      sKG[pos * LKG + D + tid] = sG[tid];
    }
    if (was_first) umma::mbar_wait(mE, 0);
    __syncthreads();
    // ---- P1 / P2: warp w < BV/8: n-tile w (v rows 8w..8w+7 of the slice)
    if (warp < BV / 8) {
      float big[4] = {-0.f, -0.f, -0.f, -0.f}, sml[4] = {-0.f, -0.f, -0.f, -0.f};
      const float* er = sE + (8 * warp + g8) * LDE;
#pragma unroll
      for (int s = 0; s < 16; ++s) {
        const int k0 = 8 * s;
        float a[4];
        a[0] = g8 == 0 ? sW[k0 + t4] : g8 == 1 ? sGQ[k0 + t4] : 0.f;
        a[1] = 0.f;
        a[2] = g8 == 0 ? sW[k0 + t4 + 4] : g8 == 1 ? sGQ[k0 + t4 + 4] : 0.f;
        a[3] = 0.f;
        const float b[2] = {er[k0 + t4], er[k0 + t4 + 4]};
        step3(big, sml, a, b);
      }
      if (g8 < 2) {
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int vv = 8 * warp + 2 * t4 + e;
          (g8 == 0 ? sP1 : sP2)[vv] = __fadd_rn(big[e], sml[e]);
        }
      }
    }
    __syncthreads();
    if (tid < BV) {
      const int vv = tid;
      const float u = __fsub_rn(sUT[vv], sP1[vv]);
      C[pos * ROWF + C_U + v0 + vv] = u;
      sR[pos * LRS + S::R_Us + vv] = u;
      float o = sP2[vv];
      if (b0 > 0) o = __fadd_rn(o, sM2[vv]);
      for (int j = b0; j < pos; ++j) o = fmaf(sAqk[j], sR[j * LRS + S::R_Us + vv], o);
      o = fmaf(sAqk[pos], u, o);
      out[th * D + v0 + vv] = __float2bfloat16(o);
    }
    __syncthreads();
    if (pos == BT - 1) {
      // ---- P3: E'[v][k] = E[v][k]*ex2(G[k]) + sum_j KD[k][j] U[j][v], v in the slice; warp w: k rows 16w..
      const int kA = 16 * warp + g8, kB = kA + 8;
      const float gA = sG[kA], gB = sG[kB];
      const float geA = ex2(gA), geB = ex2(gB);
      float big[BV / 8][4], sml[BV / 8][4];
#pragma unroll
      for (int nt = 0; nt < BV / 8; ++nt)
#pragma unroll
        for (int e = 0; e < 4; ++e) big[nt][e] = sml[nt][e] = -0.f;
#pragma unroll 2
      for (int s = 0; s < 8; ++s) {
        const int j0 = 8 * s + t4, j1 = j0 + 4;
        const float* r0 = sKG + j0 * LKG;
        const float* r1 = sKG + j1 * LKG;
        float a[4];
        a[0] = __fmul_rn(r0[kA], ex2(__fsub_rn(gA, r0[D + kA])));
        a[1] = __fmul_rn(r0[kB], ex2(__fsub_rn(gB, r0[D + kB])));
        a[2] = __fmul_rn(r1[kA], ex2(__fsub_rn(gA, r1[D + kA])));
        a[3] = __fmul_rn(r1[kB], ex2(__fsub_rn(gB, r1[D + kB])));
#pragma unroll
        for (int nt = 0; nt < BV / 8; ++nt) {
          const int vv = 8 * nt + g8;
          const float b[2] = {sR[j0 * LRS + S::R_Us + vv], sR[j1 * LRS + S::R_Us + vv]};
          step3(big[nt], sml[nt], a, b);
        }
      }
#pragma unroll
      for (int nt = 0; nt < BV / 8; ++nt)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int kr = e < 2 ? kA : kB, vv = 8 * nt + 2 * t4 + (e & 1);
          float* pe = sE + vv * LDE + kr;
          const float en = __fadd_rn(__fmul_rn(*pe, e < 2 ? geA : geB), __fadd_rn(big[nt][e], sml[nt][e]));
          *pe = en;
          E[(long)(v0 + vv) * D + kr] = en;
        }
      pos = 0;
    } else {
      ++pos;
    }
    __syncthreads();
  }
  if (tid == 0 && vt == 0) posbuf[sn * sPn + h] = pos;
}

}  // namespace cf6d8

static torch::Tensor rows_fast_impl(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor g, torch::Tensor beta,
                        torch::Tensor A_log, torch::Tensor bias, torch::Tensor cu, torch::Tensor E, torch::Tensor cache,
                        torch::Tensor pos, double lower, const int* slots, long sEn, long sCn, long sPn,
                        c10::optional<torch::Tensor> out_opt, bool one_token) {
  using namespace cf6d8;
  const int N = cu.size(0) - 1, H = q.size(2);
  // beta [1, N, H]: unit head stride; dsync9 reads any token stride, dsplit / dsync7 a contiguous beta
  TORCH_CHECK(beta.dim() == 3 && beta.stride(2) == 1, "rows_fast: beta needs a unit head stride");
  static int nsms[64] = {};
  int& nsm = nsms[q.get_device()];
  if (!nsm)
    TORCH_CHECK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, q.get_device()) == cudaSuccess,
                "kda cf-bi: SM count");
  const bool d9 = one_token && N * H * 4 > 148 && N * H > nsm;  // a single wave runs faster in dsync7
  const long sB = beta.stride(1);
  if (!d9 && sB != H) beta = beta.contiguous();
  // q / k / v [1, N, H, D]: unit inner strides and a common token stride (views into the packed conv output)
  TORCH_CHECK(q.stride(3) == 1 && q.stride(2) == D && k.strides() == q.strides() && v.strides() == q.strides(),
              "rows_fast: q/k/v need strides (*, sQ, D, 1)");
  const long sQ = q.stride(1);
  // padding (null-slot) rows are written as zeros by the kernels, every other row is fully written
  auto out = out_opt.has_value() ? *out_opt : torch::empty(v.sizes(), v.options());
  TORCH_CHECK(out.is_contiguous() && out.sizes() == v.sizes() && out.scalar_type() == v.scalar_type(), "rows_fast: out");
  auto st = c10::cuda::getCurrentCUDAStream().stream();
  if (N * H * 4 <= 148) {  // small batches: split the value dim over 32-column CTAs
    constexpr int split = 32;
    const int smem = SP<split>::SMEM;
    auto fn = dsplit_kernel<split>;
    cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    fn<<<dim3(D / split, N, H), 256, smem, st>>>(
        (const __nv_bfloat16*)q.data_ptr(), (const __nv_bfloat16*)k.data_ptr(), (const __nv_bfloat16*)v.data_ptr(),
        (const __nv_bfloat16*)g.data_ptr(), (const __nv_bfloat16*)beta.data_ptr(), A_log.data_ptr<float>(),
        bias.data_ptr<float>(), (__nv_bfloat16*)out.data_ptr(), cu.data_ptr<int>(), E.data_ptr<float>(),
        cache.data_ptr<float>(), pos.data_ptr<int>(), H, (float)lower, 1.f / sqrtf((float)D), slots, sEn, sCn, sPn, sQ);
  } else if (d9) {
    cudaFuncSetAttribute(dsync9_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM9);
    cudaFuncSetAttribute(dsync9_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    dsync9_kernel<<<dim3(N, H), 256, SMEM9, st>>>(
        (const __nv_bfloat16*)q.data_ptr(), (const __nv_bfloat16*)k.data_ptr(), (const __nv_bfloat16*)v.data_ptr(),
        (const __nv_bfloat16*)g.data_ptr(), (const __nv_bfloat16*)beta.data_ptr(), A_log.data_ptr<float>(),
        bias.data_ptr<float>(), (__nv_bfloat16*)out.data_ptr(), cu.data_ptr<int>(), E.data_ptr<float>(),
        cache.data_ptr<float>(), pos.data_ptr<int>(), H, (float)lower, 1.f / sqrtf((float)D), slots, sEn, sCn, sPn, sQ,
        sB);
  } else {
    cudaFuncSetAttribute(dsync7_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
    dsync7_kernel<<<dim3(N, H), 256, SMEM, st>>>(
        (const __nv_bfloat16*)q.data_ptr(), (const __nv_bfloat16*)k.data_ptr(), (const __nv_bfloat16*)v.data_ptr(),
        (const __nv_bfloat16*)g.data_ptr(), (const __nv_bfloat16*)beta.data_ptr(), A_log.data_ptr<float>(),
        bias.data_ptr<float>(), (__nv_bfloat16*)out.data_ptr(), cu.data_ptr<int>(), E.data_ptr<float>(),
        cache.data_ptr<float>(), pos.data_ptr<int>(), H, (float)lower, 1.f / sqrtf((float)D), slots, sEn, sCn, sPn, sQ);
  }
  return out;
}

torch::Tensor rows_fast(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor g, torch::Tensor beta,
                        torch::Tensor A_log, torch::Tensor bias, torch::Tensor cu, torch::Tensor E, torch::Tensor cache,
                        torch::Tensor pos, double lower) {
  const long H = q.size(2);
  return rows_fast_impl(q, k, v, g, beta, A_log, bias, cu, E, cache, pos, lower, nullptr, H * cf6d8::D * cf6d8::D,
                        H * cf6d8::BT * cf6d8::ROWF, H, c10::nullopt, false);
}

// kda-cfbi-slot: state pools [nslot, H, ...] (inner dims contiguous, any slot stride), slots int32 [N]
torch::Tensor rows_fast_slots(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor g,
                              torch::Tensor beta, torch::Tensor A_log, torch::Tensor bias, torch::Tensor cu,
                              torch::Tensor Epool, torch::Tensor Cpool, torch::Tensor Ppool, torch::Tensor slots,
                              double lower, c10::optional<torch::Tensor> out) {
  using namespace cf6d8;
  const long H = q.size(2);
  TORCH_CHECK(slots.scalar_type() == at::kInt && slots.is_contiguous() && slots.numel() == cu.size(0) - 1);
  TORCH_CHECK(Epool.scalar_type() == at::kFloat && Epool.size(1) == H && Epool.stride(1) == D * D &&
              Epool.stride(2) == D && Epool.stride(3) == 1, "Epool layout");
  TORCH_CHECK(Cpool.scalar_type() == at::kFloat && Cpool.size(1) == H && Cpool.stride(1) == BT * ROWF &&
              Cpool.stride(2) == ROWF && Cpool.stride(3) == 1, "Cpool layout");
  TORCH_CHECK(Ppool.scalar_type() == at::kInt && Ppool.size(1) == H && Ppool.stride(1) == 1, "Ppool layout");
  return rows_fast_impl(q, k, v, g, beta, A_log, bias, cu, Epool, Cpool, Ppool, lower, slots.data_ptr<int>(),
                        Epool.stride(0), Cpool.stride(0), Ppool.stride(0), out, true);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rows_fast", &rows_fast);
  m.def("rows_fast_slots", &rows_fast_slots, pybind11::arg("q"), pybind11::arg("k"), pybind11::arg("v"),
        pybind11::arg("g"), pybind11::arg("beta"), pybind11::arg("A_log"), pybind11::arg("bias"), pybind11::arg("cu"),
        pybind11::arg("Epool"), pybind11::arg("Cpool"), pybind11::arg("Ppool"), pybind11::arg("slots"),
        pybind11::arg("lower"), pybind11::arg("out") = pybind11::none());
}
