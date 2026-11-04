// K31 CF-BI contract v5: mma.sync m16n8k8 tf32 implementation of the contract tf32 product (bitwise identical to
// tcgen05 kind::tf32, kda_cfbi5_syncprobe.py). One warp computes a 16x8 tile D = A(16xK) * B(8xK)^T:
//   per K=8 step ascending: SMALL += Al*Bh, SMALL += Ah*Bl, BIG += Ah*Bh (accumulators start at -0, == no input);
//   D = BIG + SMALL. hi = raw fp32 (hardware truncates), lo = x - trunc_tf32(x).
// Output fragment: o[0] = (g, 2t), o[1] = (g, 2t+1), o[2] = (g+8, 2t), o[3] = (g+8, 2t+1), g = lane>>2, t = lane&3.
#pragma once
#include "kda_cfbi5_umma.cuh"

namespace mm {
__device__ __forceinline__ void mma8(float* d, const uint32_t* a, const uint32_t* b) {
  asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ uint32_t lob(float x) { return __float_as_uint(umma::lo_of(x)); }
}  // namespace mm
