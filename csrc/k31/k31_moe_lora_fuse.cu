// K31 overlay moe-lora-fuse: the elementwise glue of K31FiW4A16Experts._k31_apply_lora (mxfp4.py) as three
// row-local CUDA kernels. Every value is produced by the same IEEE operations, in the same order, as the torch
// sequence it replaces (built with --fmad=false; mul and add use the explicit _rn intrinsics), so the result is
// bitwise identical and, being row-local, batch invariant.
//
//   gather_add : x32[r, j] = fp32(gu[e2p[r], j] or +0) + fp32(ly[r, j])
//                gu = [gate | up] de-interleave of the FI GEMM1 output g1 [S, 2I] (64-column blocks [up64|gate64])
//                replaces: view/cat -> index -> where -> .float() -> add_(ly.float())
//   act_cast   : act = bf16(y32) [R, I]; act_s[i] = act[clamp(p2e[i], 0, R-1)] [S, I]
//                replaces: y32.to(bf16) -> index -> contiguous
//   slot       : slot[m, k, h] = fp32(w[m, k] * fp32(o2[e2p[m, k], h] or +0)) + fp32(d2[m, k, h])   (fp32 out)
//   slot_sum   : out[m, h] = bf16(((slot[m,0,h] + slot[m,1,h]) + slot[m,2,h]) + ...)      (k order, one rounding)
//                replaces: where/index -> .float() -> mul -> add_(d2.float()) [-> clone + k-loop add_ -> to(bf16)]
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

namespace k31lf {

__device__ __forceinline__ float bf(__nv_bfloat16 x) { return __bfloat162float(x); }

// one block per flat row r, 8 columns (16 B of bf16) per thread step
__global__ void gather_add_kernel(const __nv_bfloat16* __restrict__ g1, const int* __restrict__ e2p,
                                  const __nv_bfloat16* __restrict__ ly, float* __restrict__ x32, int inter) {
  const long r = blockIdx.x;
  const int p = e2p[r];
  const int W = 2 * inter;
  for (int j0 = threadIdx.x * 8; j0 < W; j0 += blockDim.x * 8) {
    // source column of gu[:, j0 .. j0+7] inside g1 (all 8 in the same 64-block)
    const int jj = j0 < inter ? j0 : j0 - inter;
    const int src = (jj >> 6) * 128 + (j0 < inter ? 64 : 0) + (jj & 63);
    uint4 gv = make_uint4(0, 0, 0, 0);
    if (p >= 0) gv = *reinterpret_cast<const uint4*>(g1 + (long)p * W + src);
    const uint4 lv = *reinterpret_cast<const uint4*>(ly + r * W + j0);
    const __nv_bfloat16* ga = reinterpret_cast<const __nv_bfloat16*>(&gv);
    const __nv_bfloat16* la = reinterpret_cast<const __nv_bfloat16*>(&lv);
    float o[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) o[i] = __fadd_rn(p >= 0 ? bf(ga[i]) : 0.f, bf(la[i]));
    float4* dst = reinterpret_cast<float4*>(x32 + r * W + j0);
    dst[0] = make_float4(o[0], o[1], o[2], o[3]);
    dst[1] = make_float4(o[4], o[5], o[6], o[7]);
  }
}

// blocks [0, R): act rows; blocks [R, R + S): act_s rows
__global__ void act_cast_kernel(const float* __restrict__ y32, const int* __restrict__ p2e,
                                __nv_bfloat16* __restrict__ act, __nv_bfloat16* __restrict__ act_s, int R, int inter) {
  const long b = blockIdx.x;
  long src;
  __nv_bfloat16* dst;
  if (b < R) {
    src = b;
    dst = act + b * inter;
  } else {
    int q = p2e[b - R];
    q = q < 0 ? 0 : (q > R - 1 ? R - 1 : q);
    src = q;
    dst = act_s + (b - R) * inter;
  }
  const float* s = y32 + src * inter;
  for (int j = threadIdx.x * 4; j < inter; j += blockDim.x * 4) {
    const float4 v = *reinterpret_cast<const float4*>(s + j);
    __nv_bfloat162 a = __floats2bfloat162_rn(v.x, v.y), c = __floats2bfloat162_rn(v.z, v.w);
    *reinterpret_cast<__nv_bfloat162*>(dst + j) = a;
    *reinterpret_cast<__nv_bfloat162*>(dst + j + 2) = c;
  }
}

// one block per (m, k) row
__global__ void slot_kernel(const __nv_bfloat16* __restrict__ o2, const int* __restrict__ e2p,
                            const float* __restrict__ ww, const __nv_bfloat16* __restrict__ d2,
                            float* __restrict__ slot, int hdim) {
  const long r = blockIdx.x;
  const int p = e2p[r];
  const float w = ww[r];
  for (int h = threadIdx.x * 4; h < hdim; h += blockDim.x * 4) {
    float b[4] = {0.f, 0.f, 0.f, 0.f};
    if (p >= 0) {
      const __nv_bfloat16* s = o2 + (long)p * hdim + h;
#pragma unroll
      for (int i = 0; i < 4; ++i) b[i] = bf(s[i]);
    }
    const __nv_bfloat16* dd = d2 + r * hdim + h;
    float4 o;
    o.x = __fadd_rn(__fmul_rn(w, b[0]), bf(dd[0]));
    o.y = __fadd_rn(__fmul_rn(w, b[1]), bf(dd[1]));
    o.z = __fadd_rn(__fmul_rn(w, b[2]), bf(dd[2]));
    o.w = __fadd_rn(__fmul_rn(w, b[3]), bf(dd[3]));
    *reinterpret_cast<float4*>(slot + r * hdim + h) = o;
  }
}

// grid (token m, column tile): slot leaves formed in registers, summed in k order, one bf16 rounding.
// One column per thread and many CTAs per token (a CTA per token alone starves the GPU at small M).
__global__ void slot_sum_kernel(const __nv_bfloat16* __restrict__ o2, const int* __restrict__ e2p,
                                const float* __restrict__ ww, const __nv_bfloat16* __restrict__ d2,
                                __nv_bfloat16* __restrict__ out, int topk, int hdim) {
  const long m = blockIdx.x;
  const int h = blockIdx.y * blockDim.x + threadIdx.x;
  if (h < hdim) {
    float acc = 0.f;
#pragma unroll 4
    for (int k = 0; k < topk; ++k) {
      const long r = m * topk + k;
      const int p = e2p[r];
      const float b = p >= 0 ? bf(o2[(long)p * hdim + h]) : 0.f;
      const float leaf = __fadd_rn(__fmul_rn(ww[r], b), bf(d2[r * hdim + h]));
      acc = k == 0 ? leaf : __fadd_rn(acc, leaf);
    }
    out[m * hdim + h] = __float2bfloat16_rn(acc);
  }
}

}  // namespace k31lf

static void chk(const torch::Tensor& t, at::ScalarType dt, const char* n) {
  TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.scalar_type() == dt, "k31 moe-lora-fuse: bad ", n);
}

torch::Tensor gather_add(torch::Tensor g1, torch::Tensor e2p, torch::Tensor ly, int64_t inter) {
  chk(g1, at::kBFloat16, "g1");
  chk(e2p, at::kInt, "e2p");
  chk(ly, at::kBFloat16, "ly");
  TORCH_CHECK(inter % 64 == 0 && g1.size(1) == 2 * inter && ly.size(1) == 2 * inter && ly.size(0) == e2p.numel());
  const long R = ly.size(0);
  auto x32 = torch::empty({R, 2 * inter}, ly.options().dtype(at::kFloat));
  if (R)
    k31lf::gather_add_kernel<<<R, 256, 0, c10::cuda::getCurrentCUDAStream().stream()>>>(
        (const __nv_bfloat16*)g1.data_ptr(), e2p.data_ptr<int>(), (const __nv_bfloat16*)ly.data_ptr(),
        x32.data_ptr<float>(), (int)inter);
  return x32;
}

std::vector<torch::Tensor> act_cast(torch::Tensor y32, torch::Tensor p2e) {
  chk(y32, at::kFloat, "y32");
  chk(p2e, at::kInt, "p2e");
  const long R = y32.size(0), I = y32.size(1), S = p2e.numel();
  TORCH_CHECK(I % 4 == 0 && R > 0);
  auto act = torch::empty({R, I}, y32.options().dtype(at::kBFloat16));
  auto act_s = torch::empty({S, I}, y32.options().dtype(at::kBFloat16));
  k31lf::act_cast_kernel<<<R + S, 256, 0, c10::cuda::getCurrentCUDAStream().stream()>>>(
      y32.data_ptr<float>(), p2e.data_ptr<int>(), (__nv_bfloat16*)act.data_ptr(), (__nv_bfloat16*)act_s.data_ptr(),
      (int)R, (int)I);
  return {act, act_s};
}

torch::Tensor slot(torch::Tensor o2, torch::Tensor e2p, torch::Tensor ww, torch::Tensor d2) {
  chk(o2, at::kBFloat16, "o2");
  chk(e2p, at::kInt, "e2p");
  chk(ww, at::kFloat, "ww");
  chk(d2, at::kBFloat16, "d2");
  const long M = d2.size(0), K = d2.size(1), Hd = d2.size(2);
  TORCH_CHECK(Hd % 4 == 0 && o2.size(1) == Hd && e2p.numel() == M * K && ww.numel() == M * K);
  auto s = torch::empty({M, K, Hd}, d2.options().dtype(at::kFloat));
  if (M)
    k31lf::slot_kernel<<<M * K, 256, 0, c10::cuda::getCurrentCUDAStream().stream()>>>(
        (const __nv_bfloat16*)o2.data_ptr(), e2p.data_ptr<int>(), ww.data_ptr<float>(),
        (const __nv_bfloat16*)d2.data_ptr(), s.data_ptr<float>(), (int)Hd);
  return s;
}

void slot_sum(torch::Tensor o2, torch::Tensor e2p, torch::Tensor ww, torch::Tensor d2, torch::Tensor out) {
  chk(o2, at::kBFloat16, "o2");
  chk(e2p, at::kInt, "e2p");
  chk(ww, at::kFloat, "ww");
  chk(d2, at::kBFloat16, "d2");
  chk(out, at::kBFloat16, "out");
  const long M = d2.size(0), K = d2.size(1), Hd = d2.size(2);
  TORCH_CHECK(o2.size(1) == Hd && e2p.numel() == M * K && ww.numel() == M * K && out.numel() == M * Hd);
  if (M)
    k31lf::slot_sum_kernel<<<dim3(M, (Hd + 127) / 128), 128, 0, c10::cuda::getCurrentCUDAStream().stream()>>>(
        (const __nv_bfloat16*)o2.data_ptr(), e2p.data_ptr<int>(), ww.data_ptr<float>(),
        (const __nv_bfloat16*)d2.data_ptr(), (__nv_bfloat16*)out.data_ptr(), (int)K, (int)Hd);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gather_add", &gather_add);
  m.def("act_cast", &act_cast);
  m.def("slot", &slot);
  m.def("slot_sum", &slot_sum);
}
