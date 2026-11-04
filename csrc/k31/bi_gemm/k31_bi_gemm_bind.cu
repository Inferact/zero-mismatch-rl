// torch binding for k31_bi_gemm_kernel.cu.
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include "k31_bi_gemm.cuh"

namespace {

// out = x @ w.T. x [M, K] bf16, w [N, K] bf16 (nn.Linear weight layout).
// S: split count (callers take it from the per-(K, N) table).
// add (optional): bf16 [M, N] added in fp32 to the sum before the single
// rounding (fused LoRA f32add epilogue).
torch::Tensor bi_gemm(torch::Tensor x, torch::Tensor w, int64_t S,
                      bool out_fp32, std::optional<torch::Tensor> add,
                      int64_t tile_n) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda(), "k31_bi_gemm: CUDA tensors");
  TORCH_CHECK(x.dim() == 2 && w.dim() == 2, "k31_bi_gemm: 2D operands");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 &&
                  w.scalar_type() == at::kBFloat16,
              "k31_bi_gemm: bf16 operands");
  const int64_t M = x.size(0), K = x.size(1), N = w.size(0);
  TORCH_CHECK(w.size(1) == K, "k31_bi_gemm: K mismatch");
  TORCH_CHECK(N % 8 == 0 && K % 8 == 0, "k31_bi_gemm: N, K must be % 8");
  TORCH_CHECK(tile_n == 128 || tile_n == 64, "k31_bi_gemm: tile_n 128 or 64");
  const c10::cuda::OptionalCUDAGuard guard(x.device());
  auto c = torch::empty({M, N}, x.options().dtype(out_fp32 ? at::kFloat
                                                           : at::kBFloat16));
  if (M == 0) return c;
  auto ok = [](const torch::Tensor& t) {
    return t.stride(1) == 1 && t.stride(0) % 8 == 0 &&
           reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0;
  };
  torch::Tensor x2 = ok(x) ? x : x.contiguous();
  torch::Tensor w2 = ok(w) ? w : w.contiguous();
  if (M == 1 && x2.stride(0) % 8 != 0) x2 = x2.contiguous();
  k31bi::Args a{x2.data_ptr(), w2.data_ptr(), c.data_ptr(),
                x2.stride(0),  w2.stride(0),  N,
                static_cast<int>(M), static_cast<int>(N), static_cast<int>(K)};
  torch::Tensor add2;
  if (add.has_value() && add->defined()) {
    TORCH_CHECK(add->is_cuda() && add->scalar_type() == at::kBFloat16 &&
                    add->dim() == 2 && add->size(0) == M && add->size(1) == N,
                "k31_bi_gemm: add must be bf16 [M, N]");
    add2 = ok(*add) ? *add : add->contiguous();
    a.add = add2.data_ptr();
    a.ldadd = add2.stride(0);
  }
  const k31bi::Plan pl = k31bi::make_plan(static_cast<int>(K), static_cast<int>(S));
  auto stream = c10::cuda::getCurrentCUDAStream(x.device().index()).stream();
  const char* err = k31bi::gemm(a, pl, out_fp32, stream, static_cast<int>(tile_n));
  TORCH_CHECK(err == nullptr, err);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return c;
}

// r5 small-M swap-AB chunk-balanced persistent launch (execution only, bitwise equal to bi_gemm). cnt: int32
// slice of at least sk_counters(M, N) zeros that no concurrent launch uses (the kernel leaves it zero).
torch::Tensor bi_gemm_sk(torch::Tensor x, torch::Tensor w, int64_t S, bool out_fp32, torch::Tensor cnt,
                         int64_t max_ctas, std::optional<torch::Tensor> add) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda() && cnt.is_cuda(), "k31_bi_gemm_sk: CUDA tensors");
  TORCH_CHECK(x.dim() == 2 && w.dim() == 2, "k31_bi_gemm_sk: 2D operands");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && w.scalar_type() == at::kBFloat16, "k31_bi_gemm_sk: bf16");
  const int64_t M = x.size(0), K = x.size(1), N = w.size(0);
  TORCH_CHECK(w.size(1) == K && N % 8 == 0 && K % 8 == 0, "k31_bi_gemm_sk: shapes");
  TORCH_CHECK(cnt.scalar_type() == at::kInt && cnt.is_contiguous() &&
                  cnt.numel() >= k31bi::sw_counters(static_cast<int>(M), static_cast<int>(N)),
              "k31_bi_gemm_sk: counter slice too small");
  const c10::cuda::OptionalCUDAGuard guard(x.device());
  auto c = torch::empty({M, N}, x.options().dtype(out_fp32 ? at::kFloat : at::kBFloat16));
  if (M == 0) return c;
  auto ok = [](const torch::Tensor& t) {
    return t.stride(1) == 1 && t.stride(0) % 8 == 0 && reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0;
  };
  torch::Tensor x2 = ok(x) ? x : x.contiguous();
  torch::Tensor w2 = ok(w) ? w : w.contiguous();
  if (M == 1 && x2.stride(0) % 8 != 0) x2 = x2.contiguous();
  k31bi::Args a{x2.data_ptr(), w2.data_ptr(), c.data_ptr(), x2.stride(0), w2.stride(0), N,
                static_cast<int>(M), static_cast<int>(N), static_cast<int>(K)};
  torch::Tensor add2;
  if (add.has_value() && add->defined()) {
    TORCH_CHECK(add->is_cuda() && add->scalar_type() == at::kBFloat16 && add->dim() == 2 && add->size(0) == M &&
                    add->size(1) == N,
                "k31_bi_gemm_sk: add must be bf16 [M, N]");
    add2 = ok(*add) ? *add : add->contiguous();
    a.add = add2.data_ptr();
    a.ldadd = add2.stride(0);
  }
  const k31bi::Plan pl = k31bi::make_plan(static_cast<int>(K), static_cast<int>(S));
  auto ws = torch::empty({pl.S, M, N}, x.options().dtype(at::kFloat));
  auto stream = c10::cuda::getCurrentCUDAStream(x.device().index()).stream();
  const char* err = k31bi::gemm_sw(a, pl, out_fp32, ws.data_ptr<float>(), cnt.data_ptr<int>(),
                                   static_cast<int>(max_ctas), stream);
  TORCH_CHECK(err == nullptr, err);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return c;
}

int64_t sk_counters(int64_t M, int64_t N) {
  return k31bi::sw_counters(static_cast<int>(M), static_cast<int>(N));
}

std::vector<int64_t> plan(int64_t K, int64_t S) {
  const k31bi::Plan pl = k31bi::make_plan(static_cast<int>(K), static_cast<int>(S));
  return {pl.S, pl.k_chunk};
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("bi_gemm", &bi_gemm, "K31 batch-invariant split-K GEMM (x @ w.T [+ add])",
        pybind11::arg("x"), pybind11::arg("w"), pybind11::arg("S"),
        pybind11::arg("out_fp32"), pybind11::arg("add") = pybind11::none(),
        pybind11::arg("tile_n") = 128);
  m.def("plan", &plan, "(S, k_chunk) actually used for (K, S_req)");
  m.def("bi_gemm_sk", &bi_gemm_sk, "r5 chunk-balanced persistent launch of the same GEMM",
        pybind11::arg("x"), pybind11::arg("w"), pybind11::arg("S"), pybind11::arg("out_fp32"),
        pybind11::arg("cnt"), pybind11::arg("max_ctas"), pybind11::arg("add") = pybind11::none());
  m.def("sk_counters", &sk_counters, "per-tile counters the r5 persistent launch needs for (M, N)");
}
