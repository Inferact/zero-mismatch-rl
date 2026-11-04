// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// K31 checkpoint MXFP4 -> BF16 dequant with an expert gather, one pass.
//
// packed uint8 [E, R, K/2] (low nibble first), scale uint8 E8M0 [E, R, K/32], experts int64 [c]
//   -> out bf16 [c, R, K],  out[j, r, k] = e2m1(nibble) * 2^(scale - 127)
//
// Every value is exact in FP32 and in BF16 (two significant bits times a power of two, smallest
// magnitude 0.5 * 2^-127 = 2^-128, a BF16 subnormal), so the result equals the reference
// dequantize_k31_mxfp4(...).to(bf16) bit for bit; scale 255 is NaN as in the reference.
// One thread per 4 packed bytes (8 values): coalesced 4-byte loads and 16-byte stores.

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

namespace {

// E2M1 code -> FP32 bits, decoded arithmetically (a per-lane table lookup would serialize).
__device__ __forceinline__ float e2m1(uint32_t n) {
  const uint32_t m = n & 7u;
  uint32_t bits = m >= 2u ? (((m >> 1) + 126u) << 23) | ((m & 1u) << 22) : (m == 1u ? 0x3f000000u : 0u);
  return __uint_as_float(bits | ((n & 8u) << 28));
}

__device__ __forceinline__ float e8m0(uint32_t e) {
  if (e == 255u) return __uint_as_float(0x7fc00000u);
  if (e == 0u) return __uint_as_float(0x00400000u);  // 2^-127 (FP32 subnormal)
  return __uint_as_float(e << 23);
}

__global__ void dequant_kernel(const uint32_t* __restrict__ packed, const uint8_t* __restrict__ scale,
                               const int64_t* __restrict__ experts, uint4* __restrict__ out,
                               int64_t groups_per_expert, int64_t total_words) {
  // one thread per 4 packed bytes (8 values, one 16-byte store); 4 threads per scale group
  const int64_t q = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= total_words) return;
  const int64_t g = q >> 2;
  const int64_t j = g / groups_per_expert;
  const int64_t src = experts[j] * groups_per_expert + (g - j * groups_per_expert);
  const uint32_t word = packed[src * 4 + (q & 3)];
  const float s = e8m0(scale[src]);
  uint32_t v[4];
#pragma unroll
  for (int b = 0; b < 4; ++b) {
    const uint32_t byte = (word >> (8 * b)) & 0xffu;
    const __nv_bfloat16 lo = __float2bfloat16_rn(e2m1(byte & 0xfu) * s);
    const __nv_bfloat16 hi = __float2bfloat16_rn(e2m1(byte >> 4) * s);
    v[b] = (uint32_t)__bfloat16_as_ushort(lo) | ((uint32_t)__bfloat16_as_ushort(hi) << 16);
  }
  out[q] = make_uint4(v[0], v[1], v[2], v[3]);
}

}  // namespace

torch::Tensor dequant(torch::Tensor packed, torch::Tensor scale, torch::Tensor experts) {
  TORCH_CHECK(packed.is_cuda() && packed.dtype() == torch::kUInt8 && packed.is_contiguous() && packed.dim() == 3);
  TORCH_CHECK(scale.dtype() == torch::kUInt8 && scale.is_contiguous() && scale.dim() == 3);
  TORCH_CHECK(experts.dtype() == torch::kInt64 && experts.is_contiguous() && experts.dim() == 1);
  const int64_t rows = packed.size(1), half = packed.size(2);
  TORCH_CHECK(half % 16 == 0 && scale.size(0) == packed.size(0) && scale.size(1) == rows &&
              scale.size(2) * 16 == half);
  const c10::cuda::CUDAGuard guard(packed.device());
  const int64_t c = experts.numel();
  auto out = torch::empty({c, rows, half * 2}, packed.options().dtype(torch::kBFloat16));
  const int64_t gpe = rows * (half / 16);
  const int64_t total = c * gpe * 4;
  if (total == 0) return out;
  const int threads = 256;
  const int64_t blocks = (total + threads - 1) / threads;
  dequant_kernel<<<blocks, threads, 0, c10::cuda::getCurrentCUDAStream().stream()>>>(
      reinterpret_cast<const uint32_t*>(packed.data_ptr<uint8_t>()), scale.data_ptr<uint8_t>(),
      experts.data_ptr<int64_t>(), reinterpret_cast<uint4*>(out.data_ptr()), gpe, total);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("dequant", &dequant); }
