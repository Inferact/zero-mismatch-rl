// K31 KDA chunk-form BI, contract v5 = v4 + blocked WY / intra:
//   for row t in block i = t/16 (b0 = 16i): M = tf32 product over K = b0 (k-steps ascending);
//   WY: Y = R - M (i > 0), then in-block forward substitution r = b0..t-1 (fmaf ascending);
//   intra: o = P2 + M (i > 0), then fmaf over r = b0..t ascending.
// (Header of the v4 file follows.)
// K31 KDA chunk-form BI, contract v4.
//   * gates / decays in log2 domain: G2 = serial cumsum of a2 = lower*L2E*sigmoid(...), all exp -> ex2.approx
//   * every matrix product = tcgen05 UMMA kind::tf32, M=128, 3xTF32 with separate accumulators
//     (SMALL += Al*Bh, Ah*Bl; BIG += Ah*Bh; k-steps ascending; result = BIG + SMALL)
//   * serial scalar parts (norm reduction, WY solve, intra-chunk output) unchanged in order.
#include <torch/extension.h>
#include <cuda_bf16.h>
#include <c10/cuda/CUDAStream.h>
#include "kda_cfbi5_umma.cuh"
#include "kda_cfbi6_mma.cuh"

namespace cf5 {
#define DEV __device__ __forceinline__
constexpr int D = 128, BT = 64, SUB = 16, ROWF = 5 * D;
constexpr int C_KN = 0, C_G = D, C_UT = 2 * D, C_W = 3 * D, C_U = 4 * D;
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
// a2 = lower * L2E * sigmoid(exp(A_log) * (graw + bias)); nah = -exp(A_log)*L2E, low2 = lower*L2E
DEV float gate2(float graw, float bias, float nah, float low2) {
  const float x = __fadd_rn(graw, bias);
  return __fmul_rn(low2, rcpa(__fadd_rn(1.f, ex2(__fmul_rn(nah, x)))));
}
DEV float head_nah(float A_log) { return __fmul_rn(-ex2(__fmul_rn(A_log, L2E)), L2E); }
DEV float sigm2(float braw) { return rcpa(__fadd_rn(1.f, ex2(__fmul_rn(-L2E, braw)))); }
DEV float rnorm(float ss) { return rsqrtf(__fadd_rn(ss, 1e-6f)); }
DEV void fma2(float& d0, float& d1, float a0, float a1, float b0, float b1) {
  unsigned long long d, a, b;
  asm("mov.b64 %0, {%1, %2};" : "=l"(a) : "f"(a0), "f"(a1));
  asm("mov.b64 %0, {%1, %2};" : "=l"(b) : "f"(b0), "f"(b1));
  asm("mov.b64 %0, {%1, %2};" : "=l"(d) : "f"(d0), "f"(d1));
  asm("fma.rn.f32x2 %0, %1, %2, %0;" : "+l"(d) : "l"(a), "l"(b));
  asm("mov.b64 {%0, %1}, %2;" : "=f"(d0), "=f"(d1) : "l"(d));
}
DEV uint32_t sa4(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
DEV void cp16(void* s, const void* g) { asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(sa4(s)), "l"(g)); }
DEV void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
DEV void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

constexpr int LDB = 136, LDF = 132, LDT = BT + 4;
}  // namespace cf5

using namespace cf5;
