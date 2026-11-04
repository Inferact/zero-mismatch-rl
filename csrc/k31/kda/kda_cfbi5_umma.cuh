// K31 CF-BI: 3xTF32 hi/lo split and mbarrier helpers.
// 3xTF32 contract (v4 final): hi = trunc_tf32(x) (= bits & 0xFFFFE000), lo = x - hi (exact), lo consumed as
//   trunc_tf32(lo). kind::tf32 truncates its fp32 inputs in hardware (verified bitwise, kda_cfbi5_tprobe.py), so the
//   operands are stored raw: hi tile = x itself, lo tile = x - trunc(x).
//   (rounds 1-3 used hi = rna(x), lo = rna(x - hi); replaced in round 4.)
//   per k-step of 8: SMALL += Al*Bh, SMALL += Ah*Bl, BIG += Ah*Bh (k-steps ascending);
//   result = BIG + SMALL (single __fadd_rn), then any epilogue in explicit _rn ops.
#pragma once
#include <cstdint>

#define UDEV __device__ __forceinline__

namespace umma {

UDEV uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
UDEV uint32_t tf32t(float x) { return __float_as_uint(x) & 0xFFFFE000u; }
UDEV float lo_of(float x) { return __fsub_rn(x, __uint_as_float(tf32t(x))); }

UDEV void mbar_init(uint64_t* mbar, int cnt) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(smem_u32(mbar)), "r"(cnt) : "memory");
}
UDEV void mbar_wait(uint64_t* mbar, uint32_t parity) {
  asm volatile(
      "{\n .reg .pred p;\n WAIT_%=:\n"
      " mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      " @!p bra WAIT_%=;\n}\n" ::"r"(smem_u32(mbar)),
      "r"(parity)
      : "memory");
}
UDEV void fence_mbar_init() { asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory"); }

}  // namespace umma
