// K31 KDA chunk-form BI, contract v6: v5 with BT = 16 (one block per chunk) and the WY solve on u itself.
//   * chunk = 16 tokens aligned to the sequence start; state = E fp32 [V][K] + rows {kjr, u} of the open chunk
//     + {G0, Glast} (row BT of the row cache).
//   * gates / decays in log2 domain: G2 = serial cumsum of a2 within the chunk; G0 = G2 of the chunk's first row.
//   * kjr_t = kn_t * ex2(G0 - G_t) (fixed for the chunk).
//   * Akk / Aqk (row t, j <= t): tf32 product of {kn_t * ex2(G_t - G0), qn_t * ex2(G_t - G0)} with kjr_j,
//     K = 128; Akk *= beta_t.
//   * P1 / P2: tf32 product of {beta_t * ex2(G_t) * kn_t, ex2(G_t) * qn_t} with E[v][:], K = 128.
//   * u_t = beta_t * v_t - P1, then fmaf(-Akk[j], u_j) over j = 0 .. t-1 ascending.
//   * o_t = P2, then fmaf(Aqk[j], u_j) over j = 0 .. t ascending.
//   * chunk end: E'[v][k] = E[v][k] * ex2(G_end[k]) + ex2(G_end[k] - G0[k]) * P3[k][v], P3 = tf32 product of
//     kjr_j[k] with u_j[v], K = j = 16 ascending.
// Every tf32 product is mm::tile3 (3xTF32 on mma.sync m16n8k8, bitwise equal to tcgen05 kind::tf32).
// One CTA (128 threads) per (sequence, head) runs the sequence's tokens in order: the same engine serves prefill,
// chunked prefill and decode, so they are bitwise equal by construction.
#include <torch/extension.h>
#include <cuda_bf16.h>
#include <c10/cuda/CUDAStream.h>
#include "kda_cfbi6_mma.cuh"

namespace cf6v {
#define DEV __device__ __forceinline__
constexpr int D = 128, BT = 16, ROWF = 2 * D, ROWS = BT + 1;
constexpr int C_KJR = 0, C_U = D, C_G0 = 0, C_GL = D;  // rows 0 .. BT-1: {kjr, u}; row BT: {G0, Glast}
constexpr float L2E = 1.4426950408889634f;

DEV float bf(const __nv_bfloat16 x) { return __bfloat162float(x); }
DEV float ex2(float x) {
  float y;
  asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
DEV float rcpa(float x) {
  float y;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
DEV float gate2(float graw, float bias, float nah, float low2) {
  const float x = __fadd_rn(graw, bias);
  return __fmul_rn(low2, rcpa(__fadd_rn(1.f, ex2(__fmul_rn(nah, x)))));
}
DEV float head_nah(float A_log) { return __fmul_rn(-ex2(__fmul_rn(A_log, L2E)), L2E); }
DEV float sigm2(float braw) { return rcpa(__fadd_rn(1.f, ex2(__fmul_rn(-L2E, braw)))); }
DEV float rnorm(float ss) { return rsqrtf(__fadd_rn(ss, 1e-6f)); }

template <class FA, class FB>
DEV void tile3(int K, FA fa, FB fb, float* o) {
  const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  float big[4] = {-0.f, -0.f, -0.f, -0.f}, sml[4] = {-0.f, -0.f, -0.f, -0.f};
  for (int k0 = 0; k0 < K; k0 += 8) {
    const float a0 = fa(g, k0 + t), a1 = fa(g + 8, k0 + t), a2 = fa(g, k0 + t + 4), a3 = fa(g + 8, k0 + t + 4);
    const float b0 = fb(g, k0 + t), b1 = fb(g, k0 + t + 4);
    const uint32_t ah[4] = {__float_as_uint(a0), __float_as_uint(a1), __float_as_uint(a2), __float_as_uint(a3)};
    const uint32_t al[4] = {mm::lob(a0), mm::lob(a1), mm::lob(a2), mm::lob(a3)};
    const uint32_t bh[2] = {__float_as_uint(b0), __float_as_uint(b1)};
    const uint32_t bl[2] = {mm::lob(b0), mm::lob(b1)};
    mm::mma8(sml, al, bh);
    mm::mma8(sml, ah, bl);
    mm::mma8(big, ah, bh);
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) o[i] = __fadd_rn(big[i], sml[i]);
}

__global__ void __launch_bounds__(128) rows_v6_kernel(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v, const __nv_bfloat16* __restrict__ g,
    const __nv_bfloat16* __restrict__ beta, const float* __restrict__ A_log,
    const float* __restrict__ bias, __nv_bfloat16* __restrict__ out, const int* __restrict__ cu,
    float* __restrict__ Ebuf, float* __restrict__ cache, int* __restrict__ posbuf, int H, float lower,
    float scale, const int* __restrict__ slots, long sEn, long sCn, long sPn) {
  __shared__ float sQv[D], sKv[D], sG[D], sGr[D], sQn[D], sKs[D], sQs[D], sKjp[D], sW0[D], sGQ[D];
  __shared__ float sP1[D], sOE[D], sAkk[BT], sAqk[BT], sc[4];
  const int n = blockIdx.x, h = blockIdx.y, i = threadIdx.x, warp = i >> 5, lane = i & 31;
  const int gq = lane >> 2, t4 = lane & 3;
  const int sn = slots ? slots[n] : n;
  if (slots && sn == 0) {  // padding row: zero output, no state access
    for (long t = cu[n]; t < cu[n + 1]; ++t) out[(t * H + h) * D + i] = __float2bfloat16(0.f);
    return;
  }
  float* E = Ebuf + sn * sEn + (long)h * D * D;
  float* C = cache + sn * sCn + (long)h * ROWS * ROWF;
  int pos = posbuf[sn * sPn + h];
  const float nah = head_nah(A_log[h]), low2 = __fmul_rn(lower, L2E);
  const float bias_i = bias[h * D + i];
  for (long t = cu[n]; t < cu[n + 1]; ++t) {
    const long th = t * H + h;
    {
      const float a2 = gate2(bf(g[th * D + i]), bias_i, nah, low2);
      const float G = pos == 0 ? a2 : __fadd_rn(C[BT * ROWF + C_GL + i], a2);
      sQv[i] = bf(q[th * D + i]);
      sKv[i] = bf(k[th * D + i]);
      sG[i] = G;
      sGr[i] = pos == 0 ? G : C[BT * ROWF + C_G0 + i];
    }
    __syncthreads();
    if (i == 0) {
      float sq = 0.f, sk = 0.f;
      for (int kk = 0; kk < D; ++kk) sq = fmaf(sQv[kk], sQv[kk], sq);
      for (int kk = 0; kk < D; ++kk) sk = fmaf(sKv[kk], sKv[kk], sk);
      sc[0] = __fmul_rn(rnorm(sq), scale);
      sc[1] = rnorm(sk);
      sc[2] = sigm2(bf(beta[th]));
    }
    __syncthreads();
    const float b = sc[2];
    {
      const float G = sG[i], Gr = sGr[i];
      const float qn = __fmul_rn(sQv[i], sc[0]), kn = __fmul_rn(sKv[i], sc[1]);
      const float fr = ex2(__fsub_rn(G, Gr));
      const float Gam = ex2(G);
      sQn[i] = qn;
      sKs[i] = __fmul_rn(kn, fr);
      sQs[i] = __fmul_rn(qn, fr);
      sKjp[i] = __fmul_rn(kn, ex2(__fsub_rn(Gr, G)));
      sW0[i] = __fmul_rn(b, __fmul_rn(Gam, kn));
      sGQ[i] = __fmul_rn(Gam, qn);
    }
    __syncthreads();
    // Akk / Aqk: A rows {0: ks, 1: qs}, B rows j (N = 16, warps 0..1 one n-tile each)
    if (warp < 2) {
      const int j0 = 8 * warp;
      float o4[4];
      tile3(
          D, [&](int r, int kk) { return r == 0 ? sKs[kk] : r == 1 ? sQs[kk] : 0.f; },
          [&](int nn, int kk) {
            const int j = j0 + nn;
            if (j < pos) return C[j * ROWF + C_KJR + kk];
            return j == pos ? sKjp[kk] : 0.f;
          },
          o4);
      const int j = j0 + 2 * t4;
      if (gq == 0) {
        sAkk[j] = __fmul_rn(b, o4[0]);
        sAkk[j + 1] = __fmul_rn(b, o4[1]);
      } else if (gq == 1) {
        sAqk[j] = o4[0];
        sAqk[j + 1] = o4[1];
      }
    }
    // P1 / P2: A rows {0: beta * Gam * kn, 1: Gam * qn}, B rows v = E[v][:] (N = 128, 4 n-tiles per warp)
    for (int nt = 4 * warp; nt < 4 * warp + 4; ++nt) {
      const int v0 = 8 * nt;
      float o4[4];
      tile3(
          D, [&](int r, int kk) { return r == 0 ? sW0[kk] : r == 1 ? sGQ[kk] : 0.f; },
          [&](int nn, int kk) { return E[(v0 + nn) * D + kk]; }, o4);
      const int vv = v0 + 2 * t4;
      if (gq == 0) {
        sP1[vv] = o4[0];
        sP1[vv + 1] = o4[1];
      } else if (gq == 1) {
        sOE[vv] = o4[0];
        sOE[vv + 1] = o4[1];
      }
    }
    __syncthreads();
    {
      float u = __fsub_rn(__fmul_rn(b, bf(v[th * D + i])), sP1[i]);
      for (int j = 0; j < pos; ++j) u = fmaf(-sAkk[j], C[j * ROWF + C_U + i], u);
      C[pos * ROWF + C_KJR + i] = sKjp[i];
      C[pos * ROWF + C_U + i] = u;
      C[BT * ROWF + C_GL + i] = sG[i];
      if (pos == 0) C[BT * ROWF + C_G0 + i] = sG[i];
      float o = sOE[i];
      for (int j = 0; j < pos; ++j) o = fmaf(sAqk[j], C[j * ROWF + C_U + i], o);
      o = fmaf(sAqk[pos], u, o);
      out[th * D + i] = __float2bfloat16(o);
    }
    if (pos == BT - 1) {
      __syncthreads();
      // P3[k][v]: A rows k (8 m-tiles), B rows v (16 n-tiles), K = j = 16; 32 tiles per warp
      for (int tt = warp; tt < 8 * 16; tt += 4) {
        const int k0 = 16 * (tt / 16), v0 = 8 * (tt % 16);
        float o4[4];
        tile3(
            BT,
            [&](int r, int j) { return C[j * ROWF + C_KJR + k0 + r]; },
            [&](int nn, int j) { return C[j * ROWF + C_U + v0 + nn]; }, o4);
        const int ka = k0 + gq, kb = ka + 8, va = v0 + 2 * t4;
        const float ga = ex2(sG[ka]), gb = ex2(sG[kb]);
        const float ma = ex2(__fsub_rn(sG[ka], sGr[ka])), mb = ex2(__fsub_rn(sG[kb], sGr[kb]));
        E[va * D + ka] = __fadd_rn(__fmul_rn(E[va * D + ka], ga), __fmul_rn(ma, o4[0]));
        E[(va + 1) * D + ka] = __fadd_rn(__fmul_rn(E[(va + 1) * D + ka], ga), __fmul_rn(ma, o4[1]));
        E[va * D + kb] = __fadd_rn(__fmul_rn(E[va * D + kb], gb), __fmul_rn(mb, o4[2]));
        E[(va + 1) * D + kb] = __fadd_rn(__fmul_rn(E[(va + 1) * D + kb], gb), __fmul_rn(mb, o4[3]));
      }
      pos = 0;
    } else {
      ++pos;
    }
    __syncthreads();
  }
  if (i == 0) posbuf[sn * sPn + h] = pos;
}

// Prefill engine for chunk-aligned rows (pos 0 at entry), two kernels:
//   K1: per (chunk, head): G, norms, the A operands, Akk / Aqk, W0GQ = {beta Gam kn; Gam qn}, beta v, KJR,
//       GEND = ex2(G_end), GAM = ex2(G_end - G0); the rows kjr and {G0, Glast} of a last partial chunk go to the
//       row cache.
//   K2: per (V-tile, sequence, head): chunk scan of P = W0GQ x E^T, the u / o recurrences, and P3 into E (full
//       chunks) or the u rows into the row cache (last partial chunk).
// Every element is computed with the row engine's operands and order, so the two engines are bitwise equal.
// chunk_tab[c] = {sequence, first token, rows, is_last}; seq_chunks[n] .. seq_chunks[n + 1] = chunks of sequence n.
__global__ void __launch_bounds__(128) k1v6_kernel(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v, const __nv_bfloat16* __restrict__ g,
    const __nv_bfloat16* __restrict__ beta, const float* __restrict__ A_log,
    const float* __restrict__ bias, const int* __restrict__ tab, float* __restrict__ W0GQ,
    float* __restrict__ BVc, float* __restrict__ AKK, float* __restrict__ AQK, float* __restrict__ KJR,
    float* __restrict__ GEND, float* __restrict__ GAM, float* __restrict__ cache, int H, float lower,
    float scale) {
  __shared__ float sQ[BT][D], sK[BT][D], sKs[BT][D], sQs[BT][D], sKjr[BT][D];
  __shared__ float sScq[BT], sSck[BT], sB[BT];
  const int c = blockIdx.x, h = blockIdx.y, i = threadIdx.x, warp = i >> 5, lane = i & 31;
  const int gq = lane >> 2, t4 = lane & 3;
  const int seq = tab[4 * c], tok0 = tab[4 * c + 1], nrow = tab[4 * c + 2], last = tab[4 * c + 3];
  const long ch = (long)c * H + h;
  const float nah = head_nah(A_log[h]), low2 = __fmul_rn(lower, L2E);
  const float bias_i = bias[h * D + i];
  float G[BT];
#pragma unroll
  for (int t = 0; t < BT; ++t) {
    if (t < nrow) {
      const long th = (long)(tok0 + t) * H + h;
      const float a2 = gate2(bf(g[th * D + i]), bias_i, nah, low2);
      G[t] = t == 0 ? a2 : __fadd_rn(G[t - 1], a2);
      sQ[t][i] = bf(q[th * D + i]);
      sK[t][i] = bf(k[th * D + i]);
    } else {
      G[t] = 0.f;
    }
  }
  __syncthreads();
  if (i < 2 * BT && (i & (BT - 1)) < nrow) {
    const int t = i & (BT - 1);
    float s = 0.f;
    if (i < BT) {
      for (int kk = 0; kk < D; ++kk) s = fmaf(sQ[t][kk], sQ[t][kk], s);
      sScq[t] = __fmul_rn(rnorm(s), scale);
      sB[t] = sigm2(bf(beta[(long)(tok0 + t) * H + h]));
    } else {
      for (int kk = 0; kk < D; ++kk) s = fmaf(sK[t][kk], sK[t][kk], s);
      sSck[t] = rnorm(s);
    }
  }
  __syncthreads();
  float* C = cache + ((long)seq * H + h) * ROWS * ROWF;
  const bool part = last && nrow < BT;
#pragma unroll
  for (int t = 0; t < BT; ++t) {
    if (t < nrow) {
      const long th = (long)(tok0 + t) * H + h;
      const float b = sB[t];
      const float knt = __fmul_rn(sK[t][i], sSck[t]), qnt = __fmul_rn(sQ[t][i], sScq[t]);
      const float fr = ex2(__fsub_rn(G[t], G[0]));
      const float Gam = ex2(G[t]);
      const float kjr = __fmul_rn(knt, ex2(__fsub_rn(G[0], G[t])));
      sKs[t][i] = __fmul_rn(knt, fr);
      sQs[t][i] = __fmul_rn(qnt, fr);
      sKjr[t][i] = kjr;
      W0GQ[(ch * 2 * BT + t) * D + i] = __fmul_rn(b, __fmul_rn(Gam, knt));
      W0GQ[(ch * 2 * BT + BT + t) * D + i] = __fmul_rn(Gam, qnt);
      BVc[(ch * BT + t) * D + i] = __fmul_rn(b, bf(v[th * D + i]));
      KJR[(ch * BT + t) * D + i] = kjr;
      if (part) C[t * ROWF + C_KJR + i] = kjr;
    } else {
      sKs[t][i] = 0.f;
      sQs[t][i] = 0.f;
      sKjr[t][i] = 0.f;
    }
  }
  if (nrow == BT) {
    GEND[ch * D + i] = ex2(G[BT - 1]);
    GAM[ch * D + i] = ex2(__fsub_rn(G[BT - 1], G[0]));
  } else if (part) {
    C[BT * ROWF + C_G0 + i] = G[0];
#pragma unroll
    for (int t = 0; t < BT; ++t)
      if (t == nrow - 1) C[BT * ROWF + C_GL + i] = G[t];
  }
  __syncthreads();
  {
    const int mt = warp >> 1, j0 = 8 * (warp & 1);
    float o4[4];
    tile3(
        D, [&](int r, int kk) { return mt == 0 ? sKs[r][kk] : sQs[r][kk]; },
        [&](int nn, int kk) { return sKjr[j0 + nn][kk]; }, o4);
    float* A = mt == 0 ? AKK : AQK;
    const int j = j0 + 2 * t4;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int r = gq + 8 * (e >> 1), jj = j + (e & 1);
      float x = jj <= r ? o4[e] : 0.f;
      if (mt == 0 && jj <= r) x = r < nrow ? __fmul_rn(sB[r], o4[e]) : 0.f;
      A[(ch * BT + r) * BT + jj] = x;
    }
  }
}

DEV uint32_t smem_addr(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
DEV void cp16(void* s, const void* g) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem_addr(s)), "l"(g));
}
DEV void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
DEV void cp_wait_all() { asm volatile("cp.async.wait_group 0;\n" ::: "memory"); }

// K2 shared memory: E tile, two buffers of chunk operands (W0GQ, KJR, beta v, Akk | Aqk, GEND, GAM), P, U.
template <int BV>
struct K2v6 {
  static constexpr int LD = D + 4;
  static constexpr int B_W = 0, B_KJ = B_W + 2 * BT * LD, B_BV = B_KJ + BT * LD, B_A = B_BV + BT * BV;
  static constexpr int B_GE = B_A + 2 * BT * BT, B_GA = B_GE + D, BUF = B_GA + D;
  static constexpr int OFF_E = 0, OFF_BUF = BV * LD, OFF_P = OFF_BUF + 2 * BUF, OFF_U = OFF_P + 2 * BT * BV;
  static constexpr int SMEM = (OFF_U + BT * BV) * 4;
};

template <int BV>
DEV void k2_load(float* buf, int c, int H, int h, int v0, const int* tab, const float* W0GQ, const float* BVc,
                 const float* AKK, const float* AQK, const float* KJR, const float* GEND, const float* GAM) {
  using L = K2v6<BV>;
  const int i = threadIdx.x;
  const long ch = (long)c * H + h;
  for (int u = i; u < 2 * BT * D / 4; u += 128) {
    const int r = u / (D / 4), c4 = u % (D / 4);
    cp16(buf + L::B_W + r * L::LD + 4 * c4, W0GQ + (ch * 2 * BT + r) * D + 4 * c4);
  }
  for (int u = i; u < BT * BV / 4; u += 128) {
    const int r = u / (BV / 4), c4 = u % (BV / 4);
    cp16(buf + L::B_BV + r * BV + 4 * c4, BVc + (ch * BT + r) * D + v0 + 4 * c4);
  }
  for (int u = i; u < 2 * BT * BT / 4; u += 128) {
    const float* src = u < BT * BT / 4 ? AKK + ch * BT * BT + 4 * u : AQK + ch * BT * BT + 4 * (u - BT * BT / 4);
    cp16(buf + L::B_A + 4 * u, src);
  }
  if (tab[4 * c + 2] == BT) {
    for (int u = i; u < BT * D / 4; u += 128) {
      const int r = u / (D / 4), c4 = u % (D / 4);
      cp16(buf + L::B_KJ + r * L::LD + 4 * c4, KJR + (ch * BT + r) * D + 4 * c4);
    }
    for (int u = i; u < D / 4; u += 128) {
      cp16(buf + L::B_GE + 4 * u, GEND + ch * D + 4 * u);
      cp16(buf + L::B_GA + 4 * u, GAM + ch * D + 4 * u);
    }
  }
  cp_commit();
}

template <int BV>
__global__ void __launch_bounds__(128) k2v6_kernel(
    const int* __restrict__ seq_chunks, const int* __restrict__ tab, const float* __restrict__ W0GQ,
    const float* __restrict__ BVc, const float* __restrict__ AKK, const float* __restrict__ AQK,
    const float* __restrict__ KJR, const float* __restrict__ GEND, const float* __restrict__ GAM,
    __nv_bfloat16* __restrict__ out,
    float* __restrict__ Ebuf, float* __restrict__ cache, int* __restrict__ posbuf, int H) {
  using L = K2v6<BV>;
  constexpr int LD = L::LD;
  extern __shared__ __align__(16) float sm2[];
  float *sE = sm2 + L::OFF_E, *sP = sm2 + L::OFF_P, *sU = sm2 + L::OFF_U;
  const int vt = blockIdx.x, n = blockIdx.y, h = blockIdx.z, i = threadIdx.x, warp = i >> 5, lane = i & 31;
  const int gq = lane >> 2, t4 = lane & 3, v0 = vt * BV;
  float* E = Ebuf + ((long)n * H + h) * D * D;
  float* C = cache + ((long)n * H + h) * ROWS * ROWF;
  for (int p = i; p < BV * D; p += 128) sE[(p / D) * LD + p % D] = E[(long)(v0 + p / D) * D + p % D];
  int pos = 0;
  const int c0 = seq_chunks[n], c1 = seq_chunks[n + 1];
  if (c0 < c1) k2_load<BV>(sm2 + L::OFF_BUF, c0, H, h, v0, tab, W0GQ, BVc, AKK, AQK, KJR, GEND, GAM);
  for (int c = c0; c < c1; ++c) {
    float* buf = sm2 + L::OFF_BUF + ((c - c0) & 1) * L::BUF;
    float *sW = buf + L::B_W, *sKJ = buf + L::B_KJ, *sBV = buf + L::B_BV, *sAkk = buf + L::B_A;
    float *sAqk = sAkk + BT * BT, *sGe = buf + L::B_GE, *sGa = buf + L::B_GA;
    const int tok0 = tab[4 * c + 1], nrow = tab[4 * c + 2];
    cp_wait_all();
    __syncthreads();
    if (c + 1 < c1)
      k2_load<BV>(sm2 + L::OFF_BUF + ((c + 1 - c0) & 1) * L::BUF, c + 1, H, h, v0, tab, W0GQ, BVc, AKK, AQK, KJR,
                  GEND, GAM);
    for (int tt = warp; tt < 2 * (BV / 8); tt += 4) {
      const int mt = tt / (BV / 8), n0 = 8 * (tt % (BV / 8));
      float o4[4];
      tile3(
          D, [&](int r, int kk) { return sW[(BT * mt + r) * LD + kk]; },
          [&](int nn, int kk) { return sE[(n0 + nn) * LD + kk]; }, o4);
      const int r = BT * mt + gq, vv = n0 + 2 * t4;
      sP[r * BV + vv] = o4[0];
      sP[r * BV + vv + 1] = o4[1];
      sP[(r + 8) * BV + vv] = o4[2];
      sP[(r + 8) * BV + vv + 1] = o4[3];
    }
    __syncthreads();
    if (i < BV) {
      float ur[BT];
#pragma unroll
      for (int t = 0; t < BT; ++t) {
        if (t < nrow) {
          float u = __fsub_rn(sBV[t * BV + i], sP[t * BV + i]);
#pragma unroll
          for (int j = 0; j < t; ++j) u = fmaf(-sAkk[t * BT + j], ur[j], u);
          ur[t] = u;
          sU[t * BV + i] = u;
        }
      }
    }
    __syncthreads();
    for (int p = i; p < BT * BV; p += 128) {
      const int t = p / BV, vv = p % BV;
      if (t < nrow) {
        float o = sP[(BT + t) * BV + vv];
        for (int j = 0; j < t; ++j) o = fmaf(sAqk[t * BT + j], sU[j * BV + vv], o);
        o = fmaf(sAqk[t * BT + t], sU[t * BV + vv], o);
        out[((long)(tok0 + t) * H + h) * D + v0 + vv] = __float2bfloat16(o);
      }
    }
    if (nrow == BT) {
      for (int tt = warp; tt < (D / 16) * (BV / 8); tt += 4) {
        const int k0 = 16 * (tt / (BV / 8)), n0 = 8 * (tt % (BV / 8));
        float o4[4];
        tile3(
            BT, [&](int r, int j) { return sKJ[j * LD + k0 + r]; },
            [&](int nn, int j) { return sU[j * BV + n0 + nn]; }, o4);
        const int ka = k0 + gq, kb = ka + 8, va = n0 + 2 * t4;
        sE[va * LD + ka] = __fadd_rn(__fmul_rn(sE[va * LD + ka], sGe[ka]), __fmul_rn(sGa[ka], o4[0]));
        sE[(va + 1) * LD + ka] = __fadd_rn(__fmul_rn(sE[(va + 1) * LD + ka], sGe[ka]), __fmul_rn(sGa[ka], o4[1]));
        sE[va * LD + kb] = __fadd_rn(__fmul_rn(sE[va * LD + kb], sGe[kb]), __fmul_rn(sGa[kb], o4[2]));
        sE[(va + 1) * LD + kb] = __fadd_rn(__fmul_rn(sE[(va + 1) * LD + kb], sGe[kb]), __fmul_rn(sGa[kb], o4[3]));
      }
      pos = 0;
    } else {
      for (int p = i; p < nrow * BV; p += 128) C[(p / BV) * ROWF + C_U + v0 + p % BV] = sU[p];
      pos = nrow;
    }
  }
  __syncthreads();
  if (i == 0 && vt == 0) posbuf[n * H + h] = pos;
  for (int p = i; p < BV * D; p += 128) E[(long)(v0 + p / D) * D + p % D] = sE[(p / D) * LD + p % D];
}
}  // namespace cf6v

using namespace cf6v;

static torch::Tensor launch(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor g, torch::Tensor beta,
                            torch::Tensor A_log, torch::Tensor bias, torch::Tensor cu, torch::Tensor E,
                            torch::Tensor cache, torch::Tensor pos, const int* slots, double lower,
                            c10::optional<torch::Tensor> out) {
  TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous() && g.is_contiguous() &&
                  beta.is_contiguous(),
              "K31 CF-BI v6: q/k/v/g/beta must be contiguous");
  TORCH_CHECK(q.size(-1) == D && E.size(-1) == D && E.size(-2) == D, "K31 CF-BI v6: head_dim must be 128");
  TORCH_CHECK(cache.size(-2) == ROWS && cache.size(-1) == ROWF && cache.stride(-1) == 1 &&
                  cache.stride(-2) == ROWF && cache.stride(-3) == ROWS * ROWF,
              "K31 CF-BI v6: row cache layout");
  TORCH_CHECK(E.stride(-1) == 1 && E.stride(-2) == D && E.stride(-3) == D * D && pos.stride(-1) == 1,
              "K31 CF-BI v6: E / pos layout");
  const int N = cu.size(0) - 1, H = q.size(2);
  torch::Tensor o = out.has_value() ? *out : torch::empty_like(v);
  if (N > 0) {
    rows_v6_kernel<<<dim3(N, H), 128, 0, c10::cuda::getCurrentCUDAStream().stream()>>>(
        (const __nv_bfloat16*)q.data_ptr(), (const __nv_bfloat16*)k.data_ptr(), (const __nv_bfloat16*)v.data_ptr(),
        (const __nv_bfloat16*)g.data_ptr(), (const __nv_bfloat16*)beta.data_ptr(), A_log.data_ptr<float>(),
        bias.data_ptr<float>(), (__nv_bfloat16*)o.data_ptr(), cu.data_ptr<int>(), E.data_ptr<float>(),
        cache.data_ptr<float>(), pos.data_ptr<int>(), H, (float)lower, 1.f / sqrtf((float)D), slots,
        E.stride(0), cache.stride(0), pos.stride(0));
  }
  return o;
}

// rows: sequence n of cu runs on state row n of E [N,H,D,D] / cache [N,H,17,256] / pos [N,H].
torch::Tensor rows(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor g, torch::Tensor beta,
                   torch::Tensor A_log, torch::Tensor bias, torch::Tensor cu, torch::Tensor E, torch::Tensor cache,
                   torch::Tensor pos, double lower) {
  return launch(q, k, v, g, beta, A_log, bias, cu, E, cache, pos, nullptr, lower, c10::nullopt);
}

// rows_slots: sequence n runs on pool slot slots[n] (slot 0 = padding: zero output, state untouched).
torch::Tensor rows_slots(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor g, torch::Tensor beta,
                         torch::Tensor A_log, torch::Tensor bias, torch::Tensor cu, torch::Tensor Epool,
                         torch::Tensor Cpool, torch::Tensor Ppool, torch::Tensor slots, double lower,
                         c10::optional<torch::Tensor> out) {
  TORCH_CHECK(slots.dtype() == torch::kInt32 && slots.is_contiguous(), "K31 CF-BI v6: int32 slots");
  return launch(q, k, v, g, beta, A_log, bias, cu, Epool, Cpool, Ppool, slots.data_ptr<int>(), lower, out);
}

// prefill: the chunk-aligned engine (same interface as the v5 prefill engine; bv is ignored).
std::vector<torch::Tensor> prefill(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor g,
                                   torch::Tensor beta, torch::Tensor A_log, torch::Tensor bias,
                                   torch::Tensor chunk_tab, torch::Tensor seq_chunks, torch::Tensor E,
                                   torch::Tensor cache, torch::Tensor pos, double lower, int64_t bv) {
  TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous() && g.is_contiguous() &&
                  beta.is_contiguous() && E.is_contiguous() && cache.is_contiguous() && pos.is_contiguous(),
              "K31 CF-BI v6 prefill: contiguous inputs and state");
  constexpr int BV = 32;
  const int NT = chunk_tab.size(0), N = seq_chunks.size(0) - 1, H = q.size(2);
  auto opt = q.options().dtype(torch::kFloat32);
  auto W0GQ = torch::empty({NT, H, 2 * BT, D}, opt), BVc = torch::empty({NT, H, BT, D}, opt);
  auto AKK = torch::empty({NT, H, BT, BT}, opt), AQK = torch::empty({NT, H, BT, BT}, opt);
  auto KJR = torch::empty({NT, H, BT, D}, opt), GEND = torch::empty({NT, H, D}, opt);
  auto GAM = torch::empty({NT, H, D}, opt);
  auto out = torch::zeros_like(v);
  if (NT == 0) return {out};
  auto st = c10::cuda::getCurrentCUDAStream().stream();
  k1v6_kernel<<<dim3(NT, H), 128, 0, st>>>(
      (const __nv_bfloat16*)q.data_ptr(), (const __nv_bfloat16*)k.data_ptr(), (const __nv_bfloat16*)v.data_ptr(),
      (const __nv_bfloat16*)g.data_ptr(), (const __nv_bfloat16*)beta.data_ptr(), A_log.data_ptr<float>(),
      bias.data_ptr<float>(), chunk_tab.data_ptr<int>(), W0GQ.data_ptr<float>(), BVc.data_ptr<float>(),
      AKK.data_ptr<float>(), AQK.data_ptr<float>(), KJR.data_ptr<float>(), GEND.data_ptr<float>(),
      GAM.data_ptr<float>(), cache.data_ptr<float>(), H, (float)lower, 1.f / sqrtf((float)D));
  constexpr int smem = K2v6<BV>::SMEM;
  cudaFuncSetAttribute(k2v6_kernel<BV>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  k2v6_kernel<BV><<<dim3(D / BV, N, H), 128, smem, st>>>(
      seq_chunks.data_ptr<int>(), chunk_tab.data_ptr<int>(), W0GQ.data_ptr<float>(), BVc.data_ptr<float>(),
      AKK.data_ptr<float>(), AQK.data_ptr<float>(), KJR.data_ptr<float>(), GEND.data_ptr<float>(),
      GAM.data_ptr<float>(), (__nv_bfloat16*)out.data_ptr(), E.data_ptr<float>(), cache.data_ptr<float>(), pos.data_ptr<int>(), H);
  return {out};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("prefill", &prefill);
  m.def("rows", &rows);
  m.def("rows_fast", &rows);
  m.def("rows_slots", &rows_slots, pybind11::arg("q"), pybind11::arg("k"), pybind11::arg("v"), pybind11::arg("g"),
        pybind11::arg("beta"), pybind11::arg("A_log"), pybind11::arg("bias"), pybind11::arg("cu"),
        pybind11::arg("Epool"), pybind11::arg("Cpool"), pybind11::arg("Ppool"), pybind11::arg("slots"),
        pybind11::arg("lower"), pybind11::arg("out") = pybind11::none());
  m.def("rows_fast_slots", &rows_slots, pybind11::arg("q"), pybind11::arg("k"), pybind11::arg("v"),
        pybind11::arg("g"), pybind11::arg("beta"), pybind11::arg("A_log"), pybind11::arg("bias"),
        pybind11::arg("cu"), pybind11::arg("Epool"), pybind11::arg("Cpool"), pybind11::arg("Ppool"),
        pybind11::arg("slots"), pybind11::arg("lower"), pybind11::arg("out") = pybind11::none());
}
