# SPDX-License-Identifier: Apache-2.0
"""Attention TP > 1 that reproduces the TP = 1 bi-gemm bits (see csrc/k31/k31_tp_ordered.cu).

Row-parallel layer [N, K] at TP W, rank r holding K columns [r*K/W, (r+1)*K/W):
    partials = the L = S / W chunk partials of rank r (k31_bi_gemm.chunk_partial, fp32, from zero, k ascending)
    out      = bf16( ((p_0 + p_1) + ... + p_{S-1}) [+ fp32(delta)] )     -- left-to-right, global chunk order
which is exactly k31_bi_gemm.bi_linear / lora_f32add of the full weight at TP = 1 (S = split_count(K)).
The LoRA delta of a row-parallel layer is the TP = 1 punica delta: the input is gathered (gather_cols) and the
full lora_A is kept, so shrink/expand run on the same operands as at TP = 1. For large inputs (k31_lin_lora.split_wins)
each shrink split runs on one rank instead, on its own input columns (split_cols, k31_lin_lora.apply_split).

On whenever VLLM_BATCH_INVARIANT=1 and the TP group has the K31 tree (TP > 1).
Sizes: max_num_batched_tokens x N (7168); MAX_L local chunks (16 / TP); gather width = heads x (kv_lora_rank +
qk_rope_head_dim) of the model (55296 = 96 x 576);
one-shot up to ONESHOT_T tokens (2).
"""

import torch
import torch.distributed as dist

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)
_INST = None  # the TP group's instance (one per process)


N = 7168
ONESHOT_T = 2


def active() -> bool:
    return _INST is not None


@torch._dynamo.assume_constant_result
def active_c() -> bool:
    """active() as a constant to torch.compile (the instance exists before the model is built)."""
    return _INST is not None


_EXT = None


def _ext():
    """csrc/k31/k31_tp_ordered.cu, built with vLLM (sm_100a / sm_103a, --fmad=false)."""
    global _EXT
    if _EXT is None:
        from vllm import _k31_tp_ordered_C

        _EXT = _k31_tp_ordered_C
    return _EXT


def _align(x, a=4096):
    return (x + a - 1) // a * a


class K31TpOrdered:
    def __init__(self, group, me, world, device):
        import torch.distributed._symmetric_memory as symm_mem

        from vllm.config import get_current_vllm_config

        global _INST
        assert world in (2, 4, 8, 16), world
        if not envs.VLLM_DISABLE_SHARED_EXPERTS_STREAM:
            # two streams each running a Lamport kernel can wait on each other across ranks
            raise RuntimeError("K31 tp-ordered requires VLLM_DISABLE_SHARED_EXPERTS_STREAM=1")
        e = _ext()
        self.W, self.me = world, me
        cfg = get_current_vllm_config()
        hf = cfg.model_config.hf_text_config
        if hf.hidden_size != N:
            raise RuntimeError(f"K31 tp-ordered is built for hidden size {N}, the model has {hf.hidden_size}")
        self.max_t = cfg.scheduler_config.max_num_batched_tokens
        self.N = N
        self.max_l = 16 // world
        self.os_t = ONESHOT_T
        gk = hf.num_attention_heads * (hf.kv_lora_rank + hf.qk_rope_head_dim)
        n = self.max_t * self.N
        self.rs_eb = _align(self.max_l * n * 4)
        self.ag_eb = _align(n * 2)
        self.os_eb = _align(max(self.os_t, 1) * self.N * 4 * world * self.max_l)
        self.g_eb = _align(self.max_t * gk * 2)
        self.p_eb = _align(self.max_t * 8 * 32 * 4)  # fp32 partials of the 8 rank-32 lin_lora shrink splits
        self.rs_off = 0
        self.ag_off = 3 * self.rs_eb
        self.os_off = self.ag_off + 3 * self.ag_eb
        self.g_off = self.os_off + 3 * self.os_eb
        self.p_off = self.g_off + 3 * self.g_eb
        self.m_off = self.p_off + 3 * self.p_eb  # gather epoch markers, one int64 per rank (see the .cu)
        total = self.m_off + _align(world * 8)
        self.buf = symm_mem.empty(total, dtype=torch.uint8, device=device)
        self.hdl = symm_mem.rendezvous(self.buf, group.group_name)
        self.peers = torch.tensor([int(p) for p in self.hdl.buffer_ptrs], dtype=torch.int64)
        self.base = int(self.buf.data_ptr())
        mc = int(getattr(self.hdl, "multicast_ptr", 0) or 0)
        self.mc = mc
        e.fill_sentinel(self.base, total)
        torch.cuda.synchronize(device)
        self.state = torch.zeros(32, dtype=torch.int64, device=device)
        self.empty = torch.empty(0, dtype=torch.bfloat16, device=device)
        self.ranks = dist.get_process_group_ranks(group)
        dist.barrier(group=group)
        _INST = self
        logger.info("K31 tp-ordered: world=%d rank=%d max_t=%d N=%d max_l=%d oneshot_t=%d gather_k=%d "
                    "multicast=%s buffer=%.2f GiB", world, me, self.max_t, self.N, self.max_l, self.os_t, gk,
                    bool(mc), total / 2**30)

    def ordered(self, partials, delta):
        L, M, N = partials.shape
        n = M * N
        out = torch.empty((M, N), dtype=torch.bfloat16, device=partials.device)
        if n == 0:
            return out
        if L > self.max_l or n > self.max_t * self.N:
            raise RuntimeError(f"K31 tp-ordered: partials {tuple(partials.shape)} exceed max_t={self.max_t} "
                               f"x N={self.N} / max_l={self.max_l}")
        two = not (M <= self.os_t and N <= self.N)
        if n % (8 * self.W):
            raise RuntimeError(f"K31 tp-ordered: n={n} not a multiple of 8*W")
        d = self.empty if delta is None else delta.contiguous().view(-1)
        _ext().ordered_all_reduce(partials.contiguous().view(L, n), d, out.view(-1), self.peers, self.base,
                                  self.mc, self.state, self.me, two, self.rs_off, self.rs_eb, self.ag_off,
                                  self.ag_eb, self.os_off, self.os_eb)
        return out

    def gather_window(self, q, d, grp, width):
        """q [T, hw*d] (this rank's heads) -> [T, width*d]: the heads of this rank's grp-aligned head window, from
        all ranks, then zeros (gather_window in the .cu)"""
        T = q.shape[0]
        out = torch.empty((T, width * d), dtype=q.dtype, device=q.device)
        if T == 0:
            return out
        if T * width * d * 2 > self.g_eb:
            raise RuntimeError(f"K31 tp-ordered: window {T}x{width * d} exceeds max_t / gather width")
        _ext().gather_window(q.contiguous(), out, self.peers, self.base, self.state, self.me, d, grp, self.g_off,
                             self.g_eb, self.m_off)
        return out

    def gather_cols(self, x, part=False):
        """part: x holds the fp32 shrink partials of k31_lin_lora.apply_split, gathered on their own region and
        epoch (split_cols before them pushes to some ranks only, see the .cu)"""
        T, kl = x.shape
        out = torch.empty((T, self.W * kl), dtype=x.dtype, device=x.device)
        if T == 0:
            return out
        off, eb = (self.p_off, self.p_eb) if part else (self.g_off, self.g_eb)
        if out.numel() * out.element_size() > eb:
            raise RuntimeError(f"K31 tp-ordered: gather {T}x{self.W * kl} exceeds max_t / gather width")
        _ext().gather_cols(x.contiguous(), out, self.peers, self.base, self.mc, self.state, self.me, off, eb, part)
        return out

    def split_cols(self, x):
        """x [T, K/W] (this rank's columns) -> [T, (8/W) * kc]: the input columns of this rank's k31_lin_lora
        shrink splits in chain order (split_cols in the .cu)"""
        T, kl = x.shape
        kc = -(-self.W * kl // 2048) * 256
        out = torch.empty((T, 8 // self.W * kc), dtype=x.dtype, device=x.device)
        if T == 0:
            return out
        if out.numel() * 2 > self.g_eb:
            raise RuntimeError(f"K31 tp-ordered: split {T}x{out.shape[1]} exceeds max_t / gather width")
        _ext().split_cols(x.contiguous(), out, self.peers, self.base, self.state, self.me, self.g_off, self.g_eb,
                          self.m_off)
        return out


# ------------------------------------------------------------------ custom ops (torch.compile / graph safe)
@torch.library.custom_op("k31tpo::ordered", mutates_args=())
def _ordered_op(partials: torch.Tensor, delta: torch.Tensor | None) -> torch.Tensor:
    return _INST.ordered(partials, delta)


@_ordered_op.register_fake
def _(partials, delta):
    return partials.new_empty((partials.shape[1], partials.shape[2]), dtype=torch.bfloat16)


@torch.library.custom_op("k31tpo::gather_cols", mutates_args=())
def _gather_op(x: torch.Tensor) -> torch.Tensor:
    return _INST.gather_cols(x)


@_gather_op.register_fake
def _(x):
    return x.new_empty((x.shape[0], x.shape[1] * _fake_world()))


def _fake_world():
    if _INST is not None:
        return _INST.W
    from vllm.distributed.parallel_state import get_tensor_model_parallel_world_size

    return get_tensor_model_parallel_world_size()


# ------------------------------------------------------------------ row-parallel helpers
_SEEN = set()


def row_partials(x: torch.Tensor, w: torch.Tensor, k_full: int, tp: int) -> torch.Tensor:
    """fp32 [L, M, N]: this rank's chunk partials of the TP = 1 plan of (k_full, N)."""
    from vllm.models.kimi_k3.nvidia import k31_bi_gemm as bg

    N, kl = w.shape
    if bg.route(k_full, N) != "splitk":
        raise RuntimeError(f"K31 tp-ordered: row-parallel K={k_full} N={N} is a one-chunk (cuBLAS) shape at TP 1")
    s, kc = bg.plan(k_full, N)
    L = kl // kc
    if kl % kc or L * tp != s:
        raise RuntimeError(f"K31 tp-ordered: K={k_full} plan S={s} k_chunk={kc} does not split over TP {tp}")
    key = (k_full, N, kl)
    if key not in _SEEN:
        _SEEN.add(key)
        logger.info("K31 tp-ordered engaged: row-parallel K=%d N=%d S=%d k_chunk=%d local_chunks=%d tp=%d",
                    k_full, N, s, kc, L, tp)
    x2 = x.reshape(-1, kl)
    if L == 1:
        return torch.ops.k31bi.gemm(x2, w, 1, True).unsqueeze(0)
    return torch.stack([torch.ops.k31bi.gemm(x2[:, j * kc:(j + 1) * kc].contiguous(),
                                             w[:, j * kc:(j + 1) * kc].contiguous(), 1, True) for j in range(L)])


@torch.library.custom_op("k31tpo::row", mutates_args=())
def _row_op(x: torch.Tensor, w: torch.Tensor, delta: torch.Tensor | None, k_full: int, tp: int) -> torch.Tensor:
    """Whole row-parallel layer (opaque to torch.compile): x [..., K/tp] bf16, w [N, K/tp] bf16, delta [M, N]
    bf16 or None -> bf16 [..., N] = TP 1 bi_linear / lora_f32add of the full layer."""
    assert w.dtype == torch.bfloat16 and x.dtype == torch.bfloat16, (x.dtype, w.dtype)
    p = row_partials(x, w, k_full, tp)
    out = _INST.ordered(p, delta)
    return out.reshape(*x.shape[:-1], w.shape[0])


@_row_op.register_fake
def _(x, w, delta, k_full, tp):
    return x.new_empty((*x.shape[:-1], w.shape[0]))
