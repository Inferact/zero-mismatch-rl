// K31 overlay fmlora-cuda: expert LoRA of K31FiW4A16Experts._k31_apply_lora as CUDA mma.sync kernels.
//
// Replaces punica's Triton `_fused_moe_lora_one_shot_kernel` (+ moe_lora_align_block_size) for the
// `--enable-moe-shared-loras` layout with one adapter slot. Arithmetic per row is the one-shot kernel's:
//
//   shrink : tmp[r]  = chain of mma.m16n8k16 (bf16 x bf16 -> fp32) over k = 0, 16, 32, .. K-16 in order,
//                      accumulated into one fp32 accumulator (Triton folds `tmp += tl.dot(x, a)` into the
//                      dot's accumulator); then bf16(tmp) round-to-nearest-even
//   expand : acc[n]  = mma(tmp_bf16[0:16], B[n, 0:16]) from 0, then mma(tmp_bf16[16:32], B[n, 16:32], acc)
//            w2 only: acc *= topk_weight (fp32)
//            out      = bf16(acc), -0 -> +0 (punica adds it to the zero-initialised BF16 output)
//
// Rows never mix (one mma row = one routed slot), the K order is fixed and nothing depends on the grid, so
// the result of a slot is independent of the batch (BI). Row grouping comes from FlashInfer's moe_sort tiles
// (the same tiles the FI W4A16 GEMMs use), so no extra align/sort kernel is needed. ldmatrix only moves the
// operands into the mma fragment layout; it does not change any value.
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

#include <cstdint>

namespace k31lm {

using bf16 = __nv_bfloat16;

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
__device__ __forceinline__ uint32_t ld32(const bf16* p) { return __ldg(reinterpret_cast<const uint32_t*>(p)); }

__device__ __forceinline__ uint32_t pack_out(float x, float y) {
  // bf16 RNE, then +0 for -0 (punica: prev(+0) + bf16(acc))
  __nv_bfloat162 v = __floats2bfloat162_rn(x, y);
  uint32_t u = *reinterpret_cast<uint32_t*>(&v);
  if ((u & 0xFFFFu) == 0x8000u) u &= 0xFFFF0000u;
  if ((u & 0xFFFF0000u) == 0x80000000u) u &= 0x0000FFFFu;
  return u;
}

__device__ __forceinline__ void cp16(void* smem, const void* gmem, bool pred) {
  const int n = pred ? 16 : 0;  // zero-fill dead rows
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(sptr(smem)), "l"(gmem), "r"(n));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

struct Tiles {
  const int* t2e;  // tile -> local expert
  const int* t2l;  // tile -> permuted row limit (exclusive)
  const int* nnt;  // [1] number of live tiles
  const int* p2e;  // permuted row -> expanded slot (token * topk + k)
  int rt;          // tile rows
};

// Row bookkeeping of one 16-row m-subtile: returns false when the whole subtile is dead.
// blockIdx.x = tile * (rt / 16) + sub; rows p = tile * rt + sub * 16 + i, live when p < t2l[tile].
__device__ __forceinline__ bool subtile(const Tiles& T, int& tile, int& p0, int& plim) {
  const int msub = T.rt >> 4;
  tile = blockIdx.x / msub;
  if (tile >= __ldg(T.nnt)) return false;
  p0 = tile * T.rt + (blockIdx.x % msub) * 16;
  plim = __ldg(T.t2l + tile);
  return p0 < plim;
}

// ---------------------------------------------------------------------------------------------------- shrink
constexpr int SKC = 128, SLD = SKC + 8;  // k chunk, padded smem row (272 B: ldmatrix rows hit distinct banks)
constexpr int RANK = 32;
constexpr int stage_bytes(int wn) { return (16 + 8 * wn) * SLD * 2; }

// CTA = WN warps = 16 rows x 8*WN rank columns over the full K; warp w owns 8 rank columns (its own
// accumulator chain); x and A stream through an ST-stage cp.async ring shared by the warps. WN trades
// per-SM operand bandwidth (x is re-read by every CTA of a row tile) against CTA count.
// SHARED_A (w13): rows are tokens 0..M-1 of X; grid.y = column group over [A1; A3] (2R columns),
//                 A = A_slice [R, K] shared by all experts, out[token][column].
// else (w2):      rows are the live permuted rows of the tiles, X row p (act_s, already permuted),
//                 A = A_e [E, R, K] of the tile's expert, out[p][n].
// One accumulator chained over all of K (the order Triton's `tmp += tl.dot` compiles to).
template <bool SHARED_A, int ST, int WN>
__global__ void __launch_bounds__(32 * WN) shrink_kernel(const bf16* __restrict__ X, int M, int K,
                                                     const bf16* __restrict__ A0, const bf16* __restrict__ A1,
                                                     long a_estride, Tiles T, bf16* __restrict__ out, int ldo) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  auto xs = reinterpret_cast<bf16(*)[16][SLD]>(smem_raw);
  auto as_ = reinterpret_cast<bf16(*)[8 * WN][SLD]>(smem_raw + ST * 16 * SLD * 2);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
  int rbase, rlim, ocol;
  const bf16* A;
  if constexpr (SHARED_A) {
    rbase = blockIdx.x * 16;
    rlim = M;
    ocol = blockIdx.y * 8 * WN;
    A = (ocol < RANK ? A0 : A1) + (long)(ocol % RANK) * K;
  } else {
    int tile, p0, plim;
    if (!subtile(T, tile, p0, plim)) return;
    rbase = p0;
    rlim = plim;
    ocol = blockIdx.y * 8 * WN;
    A = A0 + (long)__ldg(T.t2e + tile) * a_estride + (long)ocol * K;
  }
  // per stage: (16 x rows + 8*WN A rows) * 16 16-byte copies, spread over the CTA's threads
  auto load = [&](int stage, int kc) {
    const int k0 = kc * SKC;
    constexpr int NCP = (16 + 8 * WN) * 16, NT = 32 * WN;
    static_assert(NCP % NT == 0 && 256 % NT == 0, "loader split");
#pragma unroll
    for (int i = 0; i < NCP / NT; ++i) {
      const int idx = tid + NT * i;
      if (i < 256 / NT) {
        const int row = idx >> 4, c = (idx & 15) * 8;
        const int gr = rbase + row;
        const bool live = gr < rlim;
        cp16(&xs[stage][row][c], X + (long)(live ? gr : 0) * K + k0 + c, live);
      } else {
        const int j = idx - 256, row = j >> 4, c = (j & 15) * 8;
        cp16(&as_[stage][row][c], A + (long)row * K + k0 + c, true);
      }
    }
  };
  const int nk = K / SKC;
#pragma unroll
  for (int s = 0; s < ST - 1; ++s) {
    if (s < nk) load(s, s);
    cp_commit();
  }
  // ldmatrix lane addresses: x tile (m16 x k16) and this warp's 8 A rows (two k16 steps per x4)
  const int xr = lane & 15, xc = (lane >> 4) * 8;
  const int ar = warp * 8 + (lane & 7), ac = (lane >> 3) * 8;  // rows of as_ are this CTA's columns
  // fragments of one chunk: 4 x (two A-operand x4 + one B x4); the next chunk's are read from shared
  // memory before the current chunk's mma chain runs (the asm statements keep program order)
  uint32_t fa[2][4][2][4], fb[2][4][4];
  auto frag = [&](int buf, int st) {
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      ldsm_x4(fa[buf][q][0], &xs[st][xr][q * 32 + xc]);
      ldsm_x4(fa[buf][q][1], &xs[st][xr][q * 32 + 16 + xc]);
      ldsm_x4(fb[buf][q], &as_[st][ar][q * 32 + ac]);
    }
  };
  cp_wait<ST - 2>();
  if constexpr (WN == 1)
    __syncwarp();
  else
    __syncthreads();
  frag(0, 0);
  float c[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll 2
  for (int kc = 0; kc < nk; ++kc) {
    const int cur = kc & 1;
    cp_wait<ST - 3>();  // chunks <= kc + 1 have landed
    if constexpr (WN == 1)
      __syncwarp();
    else
      __syncthreads();  // and every warp is done reading the slot refilled below
    {
      const int nx = kc + ST - 1;
      if (nx < nk) load(nx % ST, nx);
      cp_commit();
    }
    if (kc + 1 < nk) frag(cur ^ 1, (kc + 1) % ST);
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      mma16816(c, fa[cur][q][0], fb[cur][q][0], fb[cur][q][1]);
      mma16816(c, fa[cur][q][1], fb[cur][q][2], fb[cur][q][3]);
    }
  }
  cp_wait<0>();
  const int col = ocol + warp * 8 + 2 * t;  // ocol already includes the CTA's column group
  const int r0 = rbase + g, r1 = r0 + 8;
  if (r0 < rlim)
    *reinterpret_cast<__nv_bfloat162*>(out + (long)r0 * ldo + col) = __floats2bfloat162_rn(c[0], c[1]);
  if (r1 < rlim)
    *reinterpret_cast<__nv_bfloat162*>(out + (long)r1 * ldo + col) = __floats2bfloat162_rn(c[2], c[3]);
}

// ---------------------------------------------------------------------------------------------------- expand
// CTA = 4 warps = 16 live permuted rows of a tile x ECOLS output columns; warp w owns EWCOLS = 128 columns
// (16 n8 tiles, K = R = 32: two k16 steps each). B (128 rows x 64 B per warp) is copied to shared memory with
// coalesced 16-byte cp.async and read back with ldmatrix.
// W13 (MUL = false): Trow = t13[token][slice * 32 ..], B = B_s[slice][e] [N, 32], out = ly[slot][slice*N + n]
// W2  (MUL = true):  Trow = t2[p][..],                  B = B2 [N, 32] shared,  out = d2[slot][n] * w[slot]
// grid.y = ECOLS-column chunk over nslices * N (N % ECOLS == 0: a chunk never straddles two slices).
constexpr int ECOLS = 512, EWCOLS = 128, BLD = RANK + 8;  // 80 B rows: conflict-free ldmatrix
template <bool MUL>
__global__ void __launch_bounds__(128) expand_kernel(const bf16* __restrict__ Tm, int ldt, int topk,
                                                     const bf16* __restrict__ B0, const bf16* __restrict__ B1,
                                                     long b_estride, int N, Tiles T,
                                                     const float* __restrict__ ww, const int* __restrict__ tlm,
                                                     const int* __restrict__ enabled, const int* __restrict__ lids,
                                                     int nlids, bf16* __restrict__ out, int ldo) {
  __shared__ __align__(16) bf16 bs[4][EWCOLS][BLD];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
  int tile, p0, plim;
  if (!subtile(T, tile, p0, plim)) return;
  const int e = __ldg(T.t2e + tile);
  const int c0 = blockIdx.y * ECOLS + warp * EWCOLS;
  const int slice = c0 / N, n0 = c0 % N;
  const bf16* Bm = (slice == 0 ? B0 : B1) + (long)e * b_estride + (long)n0 * RANK;
  // this warp's B block: 128 rows x 4 chunks of 16 B, 16 per lane, fully coalesced
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const int idx = lane + 32 * i, row = idx >> 2, q = (idx & 3) * 8;
    cp16(&bs[warp][row][q], Bm + (long)row * RANK + q, true);
  }
  cp_commit();
  const int pa = p0 + g, pb = pa + 8;
  const bool va = pa < plim, vb = pb < plim;
  const int sa = va ? __ldg(T.p2e + pa) : 0, sb = vb ? __ldg(T.p2e + pb) : 0;
  const int ta = sa / topk, tb = sb / topk;
  // adapter slot 0 only (caller guarantees max_loras == 1): skip tokens without it
  const bool en = __ldg(enabled) != 0;
  // slot 0 counts only while it is one of active_lora_ids[:nlids] (the punica / lin-kernel rule):
  // prepare_tensors does not rewrite token_lora_mapping for an all-no-LoRA batch
  auto act = [&](int tk) {
    const int l = __ldg(tlm + tk);
    if (l != 0) return false;
    bool f = false;
    for (int i = 0; i < nlids; ++i) f |= __ldg(lids + i) == l;
    return f;
  };
  const bool oa = va && en && act(ta), ob = vb && en && act(tb);
  const bf16 *ra, *rb;
  if constexpr (MUL) {
    ra = Tm + (long)(va ? pa : 0) * ldt;
    rb = Tm + (long)(vb ? pb : 0) * ldt;
  } else {
    ra = Tm + (long)ta * ldt + slice * RANK;
    rb = Tm + (long)tb * ldt + slice * RANK;
  }
  uint32_t a0[4], a1[4];
  a0[0] = va ? ld32(ra + 2 * t) : 0u;
  a0[1] = vb ? ld32(rb + 2 * t) : 0u;
  a0[2] = va ? ld32(ra + 8 + 2 * t) : 0u;
  a0[3] = vb ? ld32(rb + 8 + 2 * t) : 0u;
  a1[0] = va ? ld32(ra + 16 + 2 * t) : 0u;
  a1[1] = vb ? ld32(rb + 16 + 2 * t) : 0u;
  a1[2] = va ? ld32(ra + 24 + 2 * t) : 0u;
  a1[3] = vb ? ld32(rb + 24 + 2 * t) : 0u;
  float wa = 1.f, wb = 1.f;
  if constexpr (MUL) {
    wa = va ? __ldg(ww + sa) : 0.f;
    wb = vb ? __ldg(ww + sb) : 0.f;
  }
  bf16* oa_p = out + (long)sa * ldo + c0;
  bf16* ob_p = out + (long)sb * ldo + c0;
  cp_wait<0>();
  __syncwarp();
  const int br = lane & 7, bc = (lane >> 3) * 8;
  uint32_t b[16][4];
#pragma unroll
  for (int j = 0; j < 16; ++j) ldsm_x4(b[j], &bs[warp][j * 8 + br][bc]);
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    float c[4] = {0.f, 0.f, 0.f, 0.f};
    mma16816(c, a0, b[j][0], b[j][1]);
    mma16816(c, a1, b[j][2], b[j][3]);
    if constexpr (MUL) {
      c[0] = __fmul_rn(c[0], wa);
      c[1] = __fmul_rn(c[1], wa);
      c[2] = __fmul_rn(c[2], wb);
      c[3] = __fmul_rn(c[3], wb);
    }
    const int col = j * 8 + 2 * t;
    if (oa) *reinterpret_cast<uint32_t*>(oa_p + col) = pack_out(c[0], c[1]);
    if (ob) *reinterpret_cast<uint32_t*>(ob_p + col) = pack_out(c[2], c[3]);
  }
}

}  // namespace k31lm

using k31lm::bf16;

static void chk(const torch::Tensor& t, at::ScalarType dt, const char* n) {
  TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.scalar_type() == dt, "k31 moe-lora-mma: bad ", n);
}

static k31lm::Tiles tiles(const torch::Tensor& t2e, const torch::Tensor& t2l, const torch::Tensor& nnt,
                          const torch::Tensor& p2e, int64_t rt) {
  chk(t2e, at::kInt, "t2e");
  chk(t2l, at::kInt, "t2l");
  chk(nnt, at::kInt, "nnt");
  chk(p2e, at::kInt, "p2e");
  TORCH_CHECK(rt % 16 == 0 && t2e.numel() * rt <= p2e.numel());
  return k31lm::Tiles{t2e.data_ptr<int>(), t2l.data_ptr<int>(), nnt.data_ptr<int>(), p2e.data_ptr<int>(), (int)rt};
}

using ShrinkFn = void (*)(const bf16*, int, int, const bf16*, const bf16*, long, k31lm::Tiles, bf16*, int);
template <bool SA, int ST, int WN>
static ShrinkFn prep() {
  // > 48 KB dynamic shared memory: opt in once per device
  static int done[64] = {};
  int dev = 0;
  C10_CUDA_CHECK(cudaGetDevice(&dev));
  if (!done[dev]) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(k31lm::shrink_kernel<SA, ST, WN>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, ST * k31lm::stage_bytes(WN)));
    done[dev] = 1;
  }
  return k31lm::shrink_kernel<SA, ST, WN>;
}
template <bool SA, int WN>
static ShrinkFn pick_st(int64_t stages) {
  switch (stages) {
    case 4: return prep<SA, 4, WN>();
    case 6: return prep<SA, 6, WN>();
    case 8: return prep<SA, 8, WN>();
  }
  TORCH_CHECK(false, "stages must be 4/6/8");
  return nullptr;
}
template <bool SA>
static ShrinkFn pick(int64_t stages, int64_t wn) {
  switch (wn) {
    case 1: return pick_st<SA, 1>(stages);
    case 2: return pick_st<SA, 2>(stages);
    case 4: return pick_st<SA, 4>(stages);
  }
  TORCH_CHECK(false, "warps per CTA must be 1/2/4");
  return nullptr;
}

// t13[M, 2R] = bf16(x @ [A1; A3]^T) per token, chained K order
void shrink13(torch::Tensor x, torch::Tensor a1, torch::Tensor a3, torch::Tensor out, int64_t stages,
              int64_t wn) {
  chk(x, at::kBFloat16, "x");
  chk(a1, at::kBFloat16, "a1");
  chk(a3, at::kBFloat16, "a3");
  chk(out, at::kBFloat16, "t13");
  const int M = x.size(0), K = x.size(1), R = a1.size(0);
  TORCH_CHECK(R == k31lm::RANK && a1.size(1) == K && a3.sizes() == a1.sizes() && K % k31lm::SKC == 0);
  TORCH_CHECK(out.size(0) >= M && out.size(1) == 2 * R);
  if (M == 0) return;
  auto f = pick<true>(stages, wn);
  f<<<dim3((M + 15) / 16, 2 * R / (8 * wn)), 32 * wn, stages * k31lm::stage_bytes(wn),
      c10::cuda::getCurrentCUDAStream().stream()>>>(
      (const bf16*)x.data_ptr(), M, K, (const bf16*)a1.data_ptr(), (const bf16*)a3.data_ptr(), 0, k31lm::Tiles{},
      (bf16*)out.data_ptr(), 2 * R);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// t2[p, R] = bf16(act_s[p] @ A2[e(p)]^T) for the live permuted rows
void shrink2(torch::Tensor act_s, torch::Tensor a2, torch::Tensor t2e, torch::Tensor t2l, torch::Tensor nnt,
             torch::Tensor p2e, int64_t rt, torch::Tensor out, int64_t stages, int64_t wn) {
  chk(act_s, at::kBFloat16, "act_s");
  chk(a2, at::kBFloat16, "a2");
  chk(out, at::kBFloat16, "t2");
  const int S = act_s.size(0), K = act_s.size(1), R = a2.size(1);
  TORCH_CHECK(a2.dim() == 3 && R == k31lm::RANK && a2.size(2) == K && K % k31lm::SKC == 0 && out.size(0) == S &&
              out.size(1) == R);
  auto T = tiles(t2e, t2l, nnt, p2e, rt);
  const int ntiles = t2e.numel();
  if (ntiles == 0) return;
  auto f = pick<false>(stages, wn);
  f<<<dim3(ntiles * (rt / 16), R / (8 * wn)), 32 * wn, stages * k31lm::stage_bytes(wn),
      c10::cuda::getCurrentCUDAStream().stream()>>>(
      (const bf16*)act_s.data_ptr(), S, K, (const bf16*)a2.data_ptr(), nullptr, (long)R * K, T, (bf16*)out.data_ptr(),
      R);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ly[slot, s*N + n] = bf16(t13[token, s*32:(s+1)*32] @ B_s[e]^T) for the live slots (others untouched)
void expand13(torch::Tensor t13, torch::Tensor b1, torch::Tensor b3, torch::Tensor t2e, torch::Tensor t2l,
              torch::Tensor nnt, torch::Tensor p2e, int64_t rt, int64_t topk, torch::Tensor tlm,
              torch::Tensor enabled, torch::Tensor lids, int64_t nlids, torch::Tensor ly) {
  chk(lids, at::kInt, "lids");
  TORCH_CHECK(nlids >= 0 && nlids <= lids.numel());
  chk(t13, at::kBFloat16, "t13");
  chk(b1, at::kBFloat16, "b1");
  chk(b3, at::kBFloat16, "b3");
  chk(tlm, at::kInt, "tlm");
  chk(enabled, at::kInt, "enabled");
  chk(ly, at::kBFloat16, "ly");
  const int N = b1.size(1);
  TORCH_CHECK(b1.dim() == 3 && b1.size(2) == k31lm::RANK && b3.sizes() == b1.sizes() && N % k31lm::ECOLS == 0);
  TORCH_CHECK(t13.size(1) == 2 * k31lm::RANK && ly.size(1) == 2 * N);
  auto T = tiles(t2e, t2l, nnt, p2e, rt);
  const int ntiles = t2e.numel();
  if (ntiles == 0) return;
  k31lm::expand_kernel<false>
      <<<dim3(ntiles * (rt / 16), 2 * N / k31lm::ECOLS), 128, 0, c10::cuda::getCurrentCUDAStream().stream()>>>(
          (const bf16*)t13.data_ptr(), 2 * k31lm::RANK, (int)topk, (const bf16*)b1.data_ptr(),
          (const bf16*)b3.data_ptr(), (long)N * k31lm::RANK, N, T, nullptr, tlm.data_ptr<int>(),
          enabled.data_ptr<int>(), lids.data_ptr<int>(), (int)nlids, (bf16*)ly.data_ptr(), 2 * N);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// d2[slot, n] = bf16(fp32(t2[p] @ B2^T) * w[slot]) for the live slots (others untouched)
void expand2(torch::Tensor t2, torch::Tensor b2, torch::Tensor t2e, torch::Tensor t2l, torch::Tensor nnt,
             torch::Tensor p2e, int64_t rt, int64_t topk, torch::Tensor ww, torch::Tensor tlm,
             torch::Tensor enabled, torch::Tensor lids, int64_t nlids, torch::Tensor d2) {
  chk(lids, at::kInt, "lids");
  TORCH_CHECK(nlids >= 0 && nlids <= lids.numel());
  chk(t2, at::kBFloat16, "t2");
  chk(b2, at::kBFloat16, "b2");
  chk(ww, at::kFloat, "ww");
  chk(tlm, at::kInt, "tlm");
  chk(enabled, at::kInt, "enabled");
  chk(d2, at::kBFloat16, "d2");
  const int N = b2.size(0);
  TORCH_CHECK(b2.dim() == 2 && b2.size(1) == k31lm::RANK && N % k31lm::ECOLS == 0 && t2.size(1) == k31lm::RANK);
  TORCH_CHECK(d2.size(-1) == N && d2.numel() == ww.numel() * N);
  auto T = tiles(t2e, t2l, nnt, p2e, rt);
  const int ntiles = t2e.numel();
  if (ntiles == 0) return;
  k31lm::expand_kernel<true>
      <<<dim3(ntiles * (rt / 16), N / k31lm::ECOLS), 128, 0, c10::cuda::getCurrentCUDAStream().stream()>>>(
          (const bf16*)t2.data_ptr(), k31lm::RANK, (int)topk, (const bf16*)b2.data_ptr(), nullptr, 0, N, T,
          ww.data_ptr<float>(), tlm.data_ptr<int>(), enabled.data_ptr<int>(), lids.data_ptr<int>(), (int)nlids,
          (bf16*)d2.data_ptr(), N);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("shrink13", &shrink13);
  m.def("shrink2", &shrink2);
  m.def("expand13", &expand13);
  m.def("expand2", &expand2);
}
