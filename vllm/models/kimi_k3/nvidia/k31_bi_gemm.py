"""K31 batch-invariant split-K GEMM (CUDA, tcgen05; port of vLLM csrc/batch_invariant/bi_gemm_sm100_splitk).

out = x @ w.T with x [M, K] bf16 and w [N, K] bf16 (nn.Linear layout); bf16 out, or fp32 out for the router.

Numerics are fixed by (K, N) alone:
  * S = split_count(K, N) chunks of k_chunk = ceil(K / S) rounded up to 64 (the k-tile);
  * each chunk is one fp32 tcgen05 accumulation from zero (UMMA 64x128x16, k ascending);
  * chunk partials are summed left to right in fp32 (run = q0; run += q1; ...);
  * one rounding to bf16 (or none for fp32 out).
Whether the chunks run as a DSMEM cluster (small M), one CTA walking all of them (64/128-row tiles) or a 2-CTA
256-row tile (large M, csrc/k31/bi_gemm/k31_bi_gemm_2sm.cu) is execution only: all do the same fp32 additions in the same order,
bitwise equal (gated in bigemm_test.py). K <= 3584 runs S = 1 through cuBLASLt (see route(); r4: every bf16 dense
linear of the TP = 1 deployment is routed, nothing falls back to the Triton persistent matmul).

Inference (overlay k31-bi-gemm) and training must both call `bi_linear(x, w)` / `bi_router(x, w)` from this
file so they use the same split plan and kernel. Requires sm_100a / sm_103a (GB200 / GB300).
"""

from __future__ import annotations

import torch

VERSION = "r5"  # bumped whenever numerics or the API change; printed in the "engaged" lines

# ---------------------------------------------------------------------------------------------------------------
# dispatch rule and split plan: functions of (K, N) only, never of M
# ---------------------------------------------------------------------------------------------------------------
# r4 (deployment: attention TP = 1, DP + EP): every bf16 dense linear is taken.
#   K <= CUBLAS_MAX_K  -> "cublas": one chunk (S = 1) through cuBLASLt with split-K off (the image's BI setting on
#                         SM100), gated bitwise equal to this kernel at S = 1 for every routed shape and M;
#   K >  CUBLAS_MAX_K  -> "splitk": this kernel, S = split_count(K, N) chunks, any N (N % 8 != 0 is padded to a
#                         multiple of 8 with zero weight rows and sliced back; numerics unchanged).
CUBLAS_MAX_K = 3584

# Split count S depends on K only (column slices of one weight share one plan). Chosen so that row-parallel
# training (K split over TP 4 or 8) can reproduce it: S % 8 == 0 and K % (S * 64) == 0 whenever possible, so
# chunk boundaries fall on TP shard boundaries for TP in {1, 2, 4, 8}:
#   K = 7168 -> 8 (k_chunk 896), 12288 -> 8 (1536), 6144 -> 8 (768), 14336 -> 8 (1792), 33792 -> 16 (2112).
SPLIT_TABLE: dict[tuple[int, int], int] = {}


def default_split(K: int) -> int:
    if K < 1792:
        return 1
    if K >= 28672 and K % (16 * 64) == 0:
        return 16
    if K % (8 * 64) == 0:
        return 8
    return max(1, min(16, K // 896))


def split_count(K: int, N: int) -> int:
    """The numerics plan S of (K, N): 1 for K <= CUBLAS_MAX_K (one chunk, the cuBLAS route and the LoRA leaf
    alike), else default_split(K)."""
    if K <= CUBLAS_MAX_K:
        return 1
    return SPLIT_TABLE.get((K, N), default_split(K))


@torch._dynamo.assume_constant_result
def enabled() -> bool:
    """An sm_10x GPU. A constant to torch.compile: evaluated once while tracing, so it adds no guard or
    graph break."""
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10


def route(K: int, N: int) -> str | None:
    """Shape rule (K, N only): "splitk" (this kernel) or "cublas" (S = 1 via cuBLASLt, split-K off)."""
    if K % 8 or N < 1:
        return None
    return "cublas" if K <= CUBLAS_MAX_K else "splitk"


def eligible(K: int, N: int) -> bool:
    return route(K, N) is not None


_EXT = None


def ext():
    """csrc/k31/bi_gemm (built with vLLM for sm_100a / sm_103a, CUTLASS headers of the flashinfer the image ships)."""
    global _EXT
    if _EXT is None:
        cap = torch.cuda.get_device_capability()
        if cap[0] != 10:
            raise RuntimeError(f"k31_bi_gemm needs sm_100a/sm_103a, got sm_{cap[0]}{cap[1]}")
        from vllm import _k31_bi_gemm_C

        _EXT = _k31_bi_gemm_C
    return _EXT


# ---------------------------------------------------------------------------------------------------------------
# r5 launch policy (execution only: every launch mode gives the same bits, bigemm r5 gate)
# ---------------------------------------------------------------------------------------------------------------
# (K, N) -> [[max_M, mode, force_cpc], ...]: the first row with M <= max_M picks the launch; no row -> mode 0
# (r4 auto). mode 64 = the r4 64-row plan with 64-wide N tiles. mode 41 = small-M swap-AB chunk-balanced persistent
# kernel (128 weight rows x 16 tokens per tile);
# force_cpc = max CTAs, 0 = SMs.
R5_POLICY: dict[tuple[int, int], list[list[int]]] = {(12288, 7168): [[4, 41, 0]], (7168, 12288): [[4, 41, 0]], (6144, 7168): [[4, 41, 0]], (7168, 3584): [[4, 41, 0]], (33792, 7168): [[4, 41, 0]], (7168, 49376): [[1, 41, 0]], (1536, 18432): [[1, 41, 0]], (7168, 67584): [[1, 41, 0]]}
# Attention TP 8 decode shapes (weights cold in L2): perf P10 (t_gemm2) mode 41 up to max_M where it is faster at every
# timed M; perf4 P17 (t_gemm17) mode 64 from the M where 64-wide N tiles are faster at every timed M up to 64.
# KDA in_proj, LM head, dense gate_up, MLA q_b and q_a|kv_a, dense down chunk, shared expert, router, attention residual
R5_POLICY.update({
    (7168, 6288): [[4, 41, 0], [11, 0, 0], [64, 64, 0]], (7168, 20480): [[4, 41, 0]],
    (7168, 8448): [[2, 41, 0], [11, 0, 0], [64, 64, 0]], (1536, 2304): [[2, 41, 0], [64, 64, 0]],
    (7168, 2112): [[2, 41, 0], [64, 64, 0]], (2112, 7168): [[1, 41, 0], [12, 0, 0], [64, 64, 0]],
    (1536, 7168): [[12, 0, 0], [64, 64, 0]], (768, 7168): [[12, 0, 0], [64, 64, 0]],
    (7168, 1536): [[4, 0, 0], [64, 64, 0]], (7168, 896): [[64, 64, 0]], (7168, 128): [[64, 64, 0]],
    (7168, 32): [[64, 64, 0]], (128, 1536): [[64, 64, 0]],
})
_SK_POOL: dict = {}  # device index -> [int32 zeros, next free offset]
_SK_CNT: dict = {}  # (device index, K, N) -> int32 slice (one per caller shape; kernels leave it zero)


def _sk_pool(dev: torch.device) -> list:
    """Counter pool for the persistent launch, allocated on the first GEMM of a device (the eager profile run,
    before any CUDA-graph capture)."""
    p = _SK_POOL.get(dev.index)
    if p is None:
        p = _SK_POOL[dev.index] = [torch.zeros(1 << 16, dtype=torch.int32, device=dev), 0]
    return p


def _sk_cnt(x: torch.Tensor, K: int, N: int, max_m: int) -> torch.Tensor:
    key = (x.device.index, K, N)
    t = _SK_CNT.get(key)
    if t is None:
        pool = _sk_pool(x.device)
        need = int(ext().sk_counters(max_m, N))
        assert pool[1] + need <= pool[0].numel(), "k31 bi-gemm r5: counter pool exhausted"
        t = _SK_CNT[key] = pool[0][pool[1]:pool[1] + need]
        pool[1] += need
    return t


def _launch(x: torch.Tensor, w: torch.Tensor, S: int, out_fp32: bool, add=None) -> torch.Tensor:
    M, K, N = x.shape[0], x.shape[1], w.shape[0]
    mode, cpc, max_m = 0, 0, 0
    for mm, mo, fc in R5_POLICY.get((K, N), ()):
        if M <= mm:
            mode, cpc, max_m = mo, fc, mm
            break
    if mode == 41:
        return ext().bi_gemm_sk(x, w, S, out_fp32, _sk_cnt(x, K, N, max_m), cpc, add)
    return ext().bi_gemm(x, w, S, out_fp32, add, 64 if mode == 64 else 128)


# ---------------------------------------------------------------------------------------------------------------
# custom op (torch.compile / CUDA-graph safe)
# ---------------------------------------------------------------------------------------------------------------
_seen: set = set()


@torch.library.custom_op("k31bi::gemm", mutates_args=())
def _gemm_op(x: torch.Tensor, w: torch.Tensor, S: int, out_fp32: bool) -> torch.Tensor:
    key = (x.shape[1], w.shape[0], out_fp32)
    if key not in _seen:  # runs eagerly (opaque to torch.compile), so this logs once per shape and process
        _seen.add(key)
        s, kc = ext().plan(x.shape[1], S)
        print(f"K31 bi-gemm {VERSION} engaged: K={key[0]} N={key[1]} S={s} k_chunk={kc} out={'fp32' if out_fp32 else 'bf16'}"
              f" r5_policy={R5_POLICY.get((key[0], key[1]), [])}", flush=True)
        _sk_pool(x.device)
    N = w.shape[0]
    if N % 8:  # e.g. the attention-residual projections (N = 1): zero weight rows, same sums, sliced back
        wp = torch.cat([w, w.new_zeros(8 - N % 8, w.shape[1])])
        return _launch(x, wp, S, out_fp32)[:, :N].contiguous()
    return _launch(x, w, S, out_fp32)


@_gemm_op.register_fake
def _(x, w, S, out_fp32):
    return x.new_empty((x.shape[0], w.shape[0]), dtype=torch.float32 if out_fp32 else torch.bfloat16)


@torch.library.custom_op("k31bi::mm_s1", mutates_args=())
def _mm_s1_op(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    # opaque to torch.compile so Inductor never swaps in its own (Triton) mm
    key = ("s1", x.shape[1], w.shape[0])
    if key not in _seen:
        _seen.add(key)
        print(f"K31 bi-gemm {VERSION} engaged: K={key[1]} N={key[2]} S=1 cuBLASLt (split-K off)", flush=True)
    return torch.mm(x, w.t())


@_mm_s1_op.register_fake
def _(x, w):
    return x.new_empty((x.shape[0], w.shape[0]))


def bi_linear(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """x [..., K] bf16 @ w[N, K].T -> bf16 [..., N]."""
    K, N = x.shape[-1], w.shape[0]
    x2 = x.reshape(-1, K)
    if route(K, N) == "cublas":
        y = _mm_s1_op(x2, w)
    else:
        y = _gemm_op(x2, w, split_count(K, N), False)
    return y.reshape(*x.shape[:-1], N)


def bi_router(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Router logits: x [M, K] bf16 @ w[N, K].T -> fp32 [M, N] (same fp32 sum, no bf16 rounding)."""
    K = x.shape[-1]
    return _gemm_op(x.reshape(-1, K), w, split_count(K, w.shape[0]), True)


# ---------------------------------------------------------------------------------------------------------------
# LoRA f32add leaf (overlay lora-f32add-bi-r3): y = bf16( fp32(sum_k x*w) + fp32(delta) ), one rounding
# ---------------------------------------------------------------------------------------------------------------
@torch.library.custom_op("k31bi::gemm_add", mutates_args=())
def _gemm_add_op(x: torch.Tensor, w: torch.Tensor, add: torch.Tensor, S: int) -> torch.Tensor:
    key = ("add", x.shape[1], w.shape[0])
    if key not in _seen:
        _seen.add(key)
        s, kc = ext().plan(x.shape[1], S)
        print(f"K31 bi-gemm {VERSION} engaged: lora-f32add K={key[1]} N={key[2]} S={s} k_chunk={kc} "
              f"(fp32 sum + bf16 delta, one rounding, fused) r5_policy={R5_POLICY.get((key[1], key[2]), [])}",
              flush=True)
        _sk_pool(x.device)
    return _launch(x, w, S, False, add)


@_gemm_add_op.register_fake
def _(x, w, add, S):
    return x.new_empty((x.shape[0], w.shape[0]))


def f32add_eligible(K: int, N: int) -> bool:
    """LoRA f32add base products the kernel takes: every bf16 (K, N) with K % 8 == N % 8 == 0 (shape rule only).
    No cuBLAS route here: the leaf is the kernel's fp32 sum, so small K runs the kernel at S = 1."""
    return K % 8 == 0 and N % 8 == 0


def lora_f32add(x: torch.Tensor, w: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """bf16 x [M, K], bf16 w [N, K], bf16 delta [M, N] -> bf16 [M, N] = round(fp32(x @ w.T) + fp32(delta)).
    The fp32 sum is the split-K plan of (K, N) (S = split_count(K, N)); bitwise equal to
    (linear_f32(x, w) + delta.float()).to(torch.bfloat16)."""
    K, N = x.shape[-1], w.shape[0]
    return _gemm_add_op(x.reshape(-1, K), w, delta.reshape(-1, N), split_count(K, N))


def linear_f32(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """fp32 [M, N] sum of the same plan, no rounding (training reference / unfused form of lora_f32add)."""
    K, N = x.shape[-1], w.shape[0]
    return _gemm_op(x.reshape(-1, K), w, split_count(K, N), True)


def plan(K: int, N: int) -> tuple[int, int]:
    s, kc = ext().plan(K, split_count(K, N))
    return int(s), int(kc)


# ---------------------------------------------------------------------------------------------------------------
# training side: the chunk plan, and the K-sharded (row-parallel) reproduction
# ---------------------------------------------------------------------------------------------------------------
def chunks(K: int, N: int) -> list[tuple[int, int]]:
    """[k0, k1) of every chunk of the (K, N) plan, in summation order."""
    s, kc = plan(K, N)
    return [(j * kc, min(K, (j + 1) * kc)) for j in range(s) if j * kc < K]


def chunk_partial(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """fp32 [M, N] partial of ONE chunk: x [M, kc] and w [N, kc] are that chunk's columns (one fp32 tcgen05
    accumulation from zero, k ascending). Bitwise equal to the chunk's partial inside the S-chunk kernel."""
    return ext().bi_gemm(x.contiguous(), w.contiguous(), 1, True, None)


def ordered_sum(partials, delta: torch.Tensor | None = None, out_fp32: bool = False) -> torch.Tensor:
    """Left-to-right fp32 sum of the chunk partials in global chunk order (+ fp32(delta) for the LoRA leaf), then
    one rounding to bf16 (or none). Bitwise equal to bi_linear / bi_router / lora_f32add of the full GEMM."""
    run = partials[0].clone()
    for p in partials[1:]:
        run += p
    if delta is not None:
        run += delta.float()
    return run if out_fp32 else run.to(torch.bfloat16)
