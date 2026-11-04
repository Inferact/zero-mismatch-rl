# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""vLLM FlashInfer W4A16 expert forward (with expert LoRA) and an fp32 backward.

Role of :mod:`megatron.lite.primitive.alignment.vllm_grouped_moe` for K31:
the forward value is the deployed kernel sequence (``_k31_apply_lora`` of the
``fmlora-policy`` overlay, ported in ``vllm_k31_moe.fi_w4a16_lora_leaves``);
the backward recomputes the same function per expert in fp32 from the exact
MXFP4 dequantization of the checkpoint (``dequantize_mxfp4``) and returns only
``dX``, ``d(route weight)`` and the LoRA ``dA`` / ``dB``.  Base expert weights
are frozen (no ``dW``).

LoRA layout (vLLM ``--enable-moe-shared-loras``):

* shared (replicated on every EP rank, *non-expert* gradient class):
  ``w1_a``, ``w3_a`` ``[r, H]`` and ``w2_b`` ``[H, r]``;
* per expert (EP-local, expert class): ``w1_b``, ``w3_b`` ``[E_loc, I, r]``
  and ``w2_a`` ``[E_loc, r, I]``.

The forward uses exactly the tensors vLLM serves: ``A`` as BF16 and ``B`` as
the stored ``bf16(B * scaling)`` (``LoRALayerWeights.optimize``).  The master
``B`` receives ``scaling * dL/d(stored B)``.
"""

from __future__ import annotations


from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from megatron.lite.primitive.kernels.vllm_k31_moe import (
    ExpertLoRAContext,
    FiW4A16Experts,
    PunicaShim,
    fi_w4a16_lora_leaves,
)
from megatron.lite.primitive.quantization.mxfp4 import dequantize_mxfp4


def vllm_stored_lora_b(b: Tensor, scaling: float) -> Tensor:
    """``LoRALayerWeights.optimize``: ``lora_b *= scaling`` on the BF16 tensor."""

    b16 = b.detach().to(torch.bfloat16)
    if scaling == 1:
        return b16.contiguous()
    return (b16 * scaling).contiguous()


def dequantize_k31_mxfp4(packed: Tensor, scale: Tensor) -> Tensor:
    """Checkpoint MXFP4 (uint8 nibbles, low nibble first; uint8 E8M0) -> fp32."""

    return dequantize_mxfp4(
        packed.contiguous().view(torch.int8), scale.contiguous().view(torch.float8_e8m0fnu)
    )


@dataclass(frozen=True)
class ExpertLoRASpec:
    scaling: float
    adapter_enabled: tuple[int, ...] = (1, 0)
    max_loras: int = 1
    specialize_active_lora: bool = False


def build_expert_lora_context(
    num_tokens: int,
    device: torch.device,
    spec: ExpertLoRASpec,
    *,
    top_k: int,
    w1_a: Tensor,
    w3_a: Tensor,
    w1_b: Tensor,
    w3_b: Tensor,
    w2_a: Tensor,
    w2_b: Tensor,
) -> ExpertLoRAContext:
    """vLLM ``FusedMoEWithLoRA`` stacked buffers for one adapter in slot 0."""

    e_loc = w1_b.shape[0]
    r, h = w1_a.shape
    inter = w1_b.shape[1]
    if spec.max_loras != 1:
        raise NotImplementedError("training owns exactly one adapter slot")

    def a_shared(a: Tensor) -> Tensor:
        return a.detach().to(torch.bfloat16).contiguous().view(1, 1, r, h)

    w13_a = (a_shared(w1_a), a_shared(w3_a))
    w13_b = (
        vllm_stored_lora_b(w1_b, spec.scaling).view(1, e_loc, inter, r),
        vllm_stored_lora_b(w3_b, spec.scaling).view(1, e_loc, inter, r),
    )
    w2_a_s = (w2_a.detach().to(torch.bfloat16).contiguous().view(1, e_loc, r, inter),)
    w2_b_s = (vllm_stored_lora_b(w2_b, spec.scaling).view(1, 1, h, r),)
    return ExpertLoRAContext(
        w13_lora_a_stacked=w13_a,
        w13_lora_b_stacked=w13_b,
        w2_lora_a_stacked=w2_a_s,
        w2_lora_b_stacked=w2_b_s,
        adapter_enabled=torch.tensor(spec.adapter_enabled, dtype=torch.int32, device=device),
        max_loras=spec.max_loras,
        top_k=top_k,
        local_num_experts=e_loc,
        punica_wrapper=PunicaShim(
            num_tokens,
            device,
            max_loras=spec.max_loras,
            specialize_active_lora=spec.specialize_active_lora,
        ),
        enable_moe_shared_loras=True,
    )


def _situ_reference(gate: Tensor, up: Tensor, beta: float, linear_beta: float | None) -> Tensor:
    """``SituAndMul.forward_native`` in fp32 (the function the CUDA op computes)."""

    act = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    if linear_beta is not None:
        up = linear_beta * torch.tanh(up / linear_beta)
    return act * up


def expert_leaves_reference(
    x: Tensor,
    topk_ids: Tensor,
    topk_weights: Tensor,
    experts: FiW4A16Experts,
    *,
    scaling: float,
    w1_a: Tensor,
    w3_a: Tensor,
    w1_b: Tensor,
    w3_b: Tensor,
    w2_a: Tensor,
    w2_b: Tensor,
    stored_lora: bool = True,
) -> Tensor:
    """Differentiable fp32 reference of the expert leaves ``[M, top_k, H]``.

    Same function as the visible forward, evaluated in fp32 with no
    intermediate BF16 rounding.  ``stored_lora`` uses ``bf16(B * s)`` values for
    ``B`` (as served); otherwise ``B * s`` in fp32.
    """

    m, topk = topk_ids.shape
    inter = experts.intermediate
    out = x.new_zeros((m, topk, experts.hidden), dtype=torch.float32)
    local = topk_ids.long() - experts.local_offset
    valid = (local >= 0) & (local < experts.num_local_experts)
    if not bool(valid.any()):
        return out + 0.0 * (x.float().sum() + topk_weights.float().sum())

    def b_used(b: Tensor) -> Tensor:
        if stored_lora:
            return vllm_stored_lora_b(b, scaling).float()
        return b.float() * scaling

    a1, a3 = w1_a.to(torch.bfloat16).float(), w3_a.to(torch.bfloat16).float()
    b2 = b_used(w2_b)
    flat_e = torch.where(valid, local, torch.full_like(local, -1)).reshape(-1)
    order = torch.argsort(flat_e, stable=True)
    counts = torch.bincount(flat_e[valid.reshape(-1)], minlength=experts.num_local_experts).tolist()
    start = int((~valid).sum().item())
    pieces = []
    for e, c in enumerate(counts):
        if c == 0:
            continue
        sel = order[start : start + c]
        start += c
        n_idx, k_idx = sel // topk, sel % topk
        xe = x[n_idx].float()
        we = topk_weights[n_idx, k_idx].float().unsqueeze(-1)
        w13 = dequantize_k31_mxfp4(experts.ckpt_w13[e], experts.ckpt_w13_scale[e])
        w2 = dequantize_k31_mxfp4(experts.ckpt_w2[e], experts.ckpt_w2_scale[e])
        g = xe @ w13.t()
        gate = g[:, :inter] + (xe @ a1.t()) @ b_used(w1_b[e]).t()
        up = g[:, inter:] + (xe @ a3.t()) @ b_used(w3_b[e]).t()
        act = _situ_reference(gate, up, experts.situ_beta, experts.situ_linear_beta)
        y = act @ w2.t()
        d2 = we * ((act @ w2_a[e].to(torch.bfloat16).float().t()) @ b2.t())
        pieces.append((n_idx, k_idx, we * y + d2))
    for n_idx, k_idx, leaf in pieces:
        out = out.index_put((n_idx, k_idx), leaf)
    return out


# Rows per FI W4A16 call (the full-model RL value); the kernels are row-local, so chunking is exact.
FI_MAX_ROWS = 16384
# Expert backward: "split" (production); "fp32" / "fp64" run the same graph as a reference for gradient validation.
EXPERT_BWD_MODE = "split"


class _SavedExperts:
    """``ctx`` stand-in for the expert VJPs: the nine saved tensors plus experts / spec."""

    def __init__(self, saved, experts, spec) -> None:
        self.saved_tensors = saved
        self.experts = experts
        self.spec = spec


def expert_backward_rows(ctx: Any, grad_rows: Tensor, rows_index: Tensor):
    """Expert VJP from the grads of the routed leaf rows ``leaves.view(-1, H)[rows_index]`` only.

    ``leaves`` are the per-slot leaves ``[n, k, H]`` of the ``n`` received tokens; the split
    backward reads exactly the local routed slots, which are the listed ones (each listed once).
    Holds ``[L, H]`` instead of ``[n, k, H]`` fp32 (``L ~ n * k / EP`` rows).  The ``fp32`` /
    ``fp64`` reference modes scatter the rows into the dense ``[n, k, H]`` grad they differentiate.
    """

    x, topk_ids = ctx.saved_tensors[0], ctx.saved_tensors[1]
    n, k = topk_ids.shape
    h = grad_rows.shape[1]
    mode = EXPERT_BWD_MODE
    if mode == "split":
        L = grad_rows.shape[0]
        inv = torch.full((n * k,), L, dtype=torch.long, device=x.device)
        inv[rows_index] = torch.arange(L, device=x.device)
        compact = torch.cat((grad_rows.float(), grad_rows.new_zeros((1, h), dtype=torch.float32)))
        return _expert_backward_split(ctx, compact, inv)
    # "fp32": reference; "fp64": the same graph in float64 (accuracy yardstick only)
    dense = torch.zeros((n * k, h), dtype=torch.float32, device=x.device)
    dense.index_add_(0, rows_index, grad_rows.float())
    return _expert_backward_fp32(
        ctx, dense.view(n, k, h), torch.float64 if mode == "fp64" else torch.float32
    )


class VLLMFiW4A16LeafRowsFunction(torch.autograd.Function):
    """Visible FI W4A16 + expert-LoRA leaves of the rows ``rows_index`` (flat ``n * k + slot``).

    Returns ``leaves.view(-1, H)[rows_index]`` fp32 without the dense ``[n, k, H]`` slot tensor;
    backward: :func:`expert_backward_rows`.  The FI W4A16 / LoRA kernels are row-local (each
    received token's leaves depend on that row only) but their workspaces and int32 offsets are
    sized for bounded row counts, so they run in ``FI_MAX_ROWS`` chunks; each chunk's leaves are
    copied into the output rows it owns and the chunk is freed.
    """

    @staticmethod
    def forward(ctx, experts, spec, rows_index, x, topk_ids, topk_weights, w1_a, w3_a, w1_b, w3_b, w2_a, w2_b):
        rows = x.shape[0]
        k = topk_ids.shape[1]
        step = FI_MAX_ROWS
        # Lean experts carry no FI w13 copy: build it for this call only (same bytes); the
        # backward reads the checkpoint bytes, so ``ctx`` keeps the lean object.
        full = experts.materialize()
        h = None
        out = None
        chunk_of = rows_index // (step * k)
        for ci, lo in enumerate(range(0, max(rows, 1), step)):
            hi = min(rows, lo + step)
            lora = build_expert_lora_context(
                hi - lo,
                x.device,
                spec,
                top_k=k,
                w1_a=w1_a,
                w3_a=w3_a,
                w1_b=w1_b,
                w3_b=w3_b,
                w2_a=w2_a,
                w2_b=w2_b,
            )
            piece = fi_w4a16_lora_leaves(
                x[lo:hi].detach(), topk_ids[lo:hi], topk_weights[lo:hi].detach(), full, lora
            )
            if out is None:
                h = piece.shape[-1]
                out = torch.empty((rows_index.numel(), h), dtype=piece.dtype, device=x.device)
            sel = (chunk_of == ci).nonzero().squeeze(1)
            if sel.numel():
                out.index_copy_(0, sel, piece.reshape(-1, h).index_select(0, rows_index.index_select(0, sel) - lo * k))
            del piece
        del full
        ctx.experts = experts.lean()
        ctx.spec = spec
        ctx.save_for_backward(x, topk_ids, topk_weights, w1_a, w3_a, w1_b, w3_b, w2_a, w2_b, rows_index)
        ctx.mark_non_differentiable(topk_ids)
        return out

    @staticmethod
    def backward(ctx, grad_rows):
        saved = ctx.saved_tensors
        sub_ctx = _SavedExperts(saved[:9], ctx.experts, ctx.spec)
        grads = expert_backward_rows(sub_ctx, grad_rows, saved[9])
        # (experts, spec, rows_index, x, topk_ids, topk_weights, w1_a, w3_a, w1_b, w3_b, w2_a, w2_b)
        return (None, None, None, *grads[2:])


_E2M1_PAIR_BF16: dict[torch.device, Tensor] = {}
_E8M0_BF16: dict[torch.device, Tensor] = {}


def dequantize_k31_mxfp4_bf16(packed: Tensor, scale: Tensor) -> Tensor:
    """Checkpoint MXFP4 -> BF16, equal to ``dequantize_k31_mxfp4(...).to(bf16)``.

    E2M1 levels have at most two significant bits and E8M0 scales are powers of two, so
    every dequantized value (and the BF16 product forming it) is exact in BF16, including
    the 2^-127 scale (BF16 subnormal).  One 256-entry byte -> (low, high) nibble table gather
    and one scale multiply.
    """

    dev = packed.device
    lut = _E2M1_PAIR_BF16.get(dev)
    if lut is None:
        pos = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
        e2m1 = torch.tensor((*pos, *(-v for v in pos)), dtype=torch.float32)
        byte = torch.arange(256)
        lut = torch.stack((e2m1[byte & 0x0F], e2m1[byte >> 4]), dim=-1).to(torch.bfloat16).to(dev)
        _E2M1_PAIR_BF16[dev] = lut
        exps = torch.arange(256, dtype=torch.float64)
        sc = torch.where(exps == 255, torch.nan, torch.exp2(exps - 127))
        _E8M0_BF16[dev] = sc.to(torch.float32).to(torch.bfloat16).to(dev)
    raw = packed.contiguous().view(torch.uint8)
    vals = torch.nn.functional.embedding(raw.int(), lut)  # [..., K/2, 2]
    s = _E8M0_BF16[dev][scale.contiguous().view(torch.uint8).int()]  # [..., K/32]
    shape = (*raw.shape[:-1], raw.shape[-1] * 2)
    return (vals.view(*shape[:-1], -1, 32) * s.unsqueeze(-1)).view(shape)


def _dequant_experts(packed: Tensor, scale: Tensor, ex_t: Tensor) -> Tensor:
    """BF16 dequant of the local experts ``ex_t`` (CUDA one-pass kernel)."""

    from megatron.lite.primitive.kernels import k31_mxfp4_dequant as dq

    return dq.dequant_experts(packed, scale, ex_t)


# Local experts per batched step of the split backward (BF16 w13 + w2 = 66 MB each) and the
# padded-row budget of one step.
EXPERT_BWD_CHUNK = 32
EXPERT_BWD_ROWS = 16384


def _bmm_f32(a: Tensor, b: Tensor) -> Tensor:
    """Batched FP32 product through cuBLAS (``aten::baddbmm`` is not overridden by BI mode)."""

    out = torch.empty((a.shape[0], a.shape[1], b.shape[2]), dtype=torch.float32, device=a.device)
    return torch.baddbmm(out, a.float(), b.float(), beta=0.0)


def _expert_backward_split(ctx: Any, grad_rows: Tensor, row_of_slot: Tensor):
    """Same VJP as ``_expert_backward_fp32`` on BF16 tensor cores, batched over local experts.

    ``grad_rows [L + 1, H]`` holds the routed leaf grads plus a zero row; ``row_of_slot [m * k]``
    maps each flat slot to its row (``L`` for slots not routed here).  A constant number of host
    syncs per call, independent of the chunk count: the per-expert counts and their prefix in one
    copy, one global BF16-exactness check of the grads, the per-slot counts of the token
    reduction; the chunk tables go to the device in one batched H2D copy.

    Explicit formulas, no inner autograd graph except the elementwise SiTU derivative:

    * ``G = X W13^T`` and ``U = GS W2`` have BF16-exact operands (BF16 tokens, dequantized MXFP4
      weights, BF16 grad of the BF16-visible combine), so a BF16 GEMM with FP32 accumulate is the
      FP32 product; ``dX = dGU W13`` splits the FP32 ``dGU`` into BF16 terms; an FP32 grad that
      is not BF16-exact is split the same way; every BF16 GEMM accumulates K in 512-wide blocks
      through the FP32 epilogue (``split_bf16_mm``: FP32 SIMT accuracy);
    * ``d(route weight) = act . (GS W2 + (GS B2) A2)`` (no recompute of ``act W2^T``);
    * rank-r LoRA products stay FP32 (cuBLAS batched).

    Per chunk of experts (descending row count) the routed rows are gathered straight into a
    ``[c, n_max, .]`` layout; padding rows repeat a real row with route weight 0, so every
    gradient they feed is exactly 0, and their outputs land in a discarded extra row.
    """

    from megatron.lite.primitive.alignment.split_bf16_mm import bmm_bf16, mm_bf16, split_bf16_terms

    x, topk_ids, topk_weights, w1_a, w3_a, w1_b, w3_b, w2_a, w2_b = ctx.saved_tensors
    experts: FiW4A16Experts = ctx.experts
    scaling = ctx.spec.scaling
    beta, lbeta = experts.situ_beta, experts.situ_linear_beta
    m, topk = topk_ids.shape
    inter = experts.intermediate
    h = x.shape[1]
    dev = x.device
    f32, bf16 = torch.float32, torch.bfloat16
    e_loc = experts.num_local_experts
    dx = torch.zeros((m, h), dtype=f32, device=dev)
    dw = torch.zeros((m * topk + 1,), dtype=f32, device=dev)
    da1 = torch.zeros(w1_a.shape, dtype=f32, device=dev)
    da3 = torch.zeros_like(da1)
    db2 = torch.zeros(w2_b.shape, dtype=f32, device=dev)
    db1 = torch.zeros(w1_b.shape, dtype=f32, device=dev)
    db3 = torch.zeros_like(db1)
    da2 = torch.zeros(w2_a.shape, dtype=f32, device=dev)
    local = topk_ids.long() - experts.local_offset
    valid = (local >= 0) & (local < e_loc)
    flat_e = torch.where(valid, local, torch.full_like(local, -1)).reshape(-1)
    order = torch.argsort(flat_e, stable=True)
    counts = torch.bincount(flat_e[valid.reshape(-1)], minlength=e_loc)
    # counts and their exclusive prefix in one device->host copy
    starts_t = torch.cumsum(counts, 0) - counts
    both = torch.cat((counts, starts_t)).tolist()
    counts_l, starts_l = both[:e_loc], both[e_loc:]
    rows = sum(counts_l)
    if rows:
        sel = order[order.numel() - rows :]  # flat (token, slot) of local routes, grouped by expert
        # per-route dX rows (+ a padding sink row): reduced into tokens in a fixed slot order below
        dx_rows = torch.empty((rows + 1, h), dtype=f32, device=dev)
        x_d = x.detach()
        we_flat = topk_weights.detach().reshape(-1).float()
        a1b = w1_a.detach().to(bf16)
        a3b = w3_a.detach().to(bf16)
        a13f = torch.cat((a1b, a3b), 0).float()  # [2r, H]
        b2b = vllm_stored_lora_b(w2_b, scaling)  # [H, r]
        b1v = vllm_stored_lora_b(w1_b, scaling).float()  # [E, I, r]
        b3v = vllm_stored_lora_b(w3_b, scaling).float()
        a2v = w2_a.detach().to(bf16).float()  # [E, r, I]
        a13b_t = torch.cat((a1b, a3b), 0).t()  # [H, 2r]
        r = a1b.shape[0]
        # work items (expert, first row, rows): experts split into blocks of <= budget rows,
        # largest first; a chunk holds <= EXPERT_BWD_CHUNK distinct experts and <= budget
        # padded rows (bounds the FP32 temporaries to ~200 KB per row)
        budget = EXPERT_BWD_ROWS
        max_experts = EXPERT_BWD_CHUNK
        items = []
        for e in range(e_loc):
            for o in range(0, counts_l[e], budget):
                items.append((e, o, min(budget, counts_l[e] - o)))
        items.sort(key=lambda it: -it[2])
        chunks, cur = [], []
        for it in items:
            n_first = -(-(cur[0][2] if cur else it[2]) // 8) * 8
            if cur and (
                len(cur) == max_experts
                or (len(cur) + 1) * n_first > budget
                or any(it[0] == jt[0] for jt in cur)
            ):
                chunks.append(cur)
                cur = []
            cur.append(it)
        if cur:
            chunks.append(cur)
        # the (expert, rows, first row) tables of all chunks in one H2D copy, not three per chunk
        flat_tab = []
        for chunk in chunks:
            flat_tab += [it[0] for it in chunk]
            flat_tab += [it[2] for it in chunk]
            flat_tab += [starts_l[it[0]] + it[1] for it in chunk]
        tables = torch.tensor(flat_tab, dtype=torch.long).pin_memory().to(dev, non_blocking=True)
        # split_bf16_terms is elementwise: every chunk residual is a row subset of this one
        rest_all = split_bf16_terms(grad_rows)[1:]
        rest_zero = not rest_all or not bool(rest_all[0].any())
        del rest_all
        toff = 0
        for chunk in chunks:
            c = len(chunk)
            # padded rows per item: a multiple of 8 (cuBLASLt rejects some 1-row batched
            # layouts under the BI preferences)
            n = -(-chunk[0][2] // 8) * 8
            ex_t = tables[toff : toff + c]
            cnt = tables[toff + c : toff + 2 * c]
            first_row = tables[toff + 2 * c : toff + 3 * c]
            toff += 3 * c
            col = torch.arange(n, device=dev)
            real = col.unsqueeze(0) < cnt.unsqueeze(1)  # [c, n]
            pos = first_row.unsqueeze(1) + torch.minimum(col.unsqueeze(0), cnt.unsqueeze(1) - 1)
            flat = sel[pos.reshape(-1)]  # [c*n] flat route index (pads repeat a real route)
            tok = flat // topk
            we_p = torch.where(real.reshape(-1), we_flat[flat], 0.0).view(c, n, 1)
            sink_row = torch.where(real.reshape(-1), pos.reshape(-1), rows)
            sink_flat = torch.where(real.reshape(-1), flat, m * topk)
            xp = x_d.index_select(0, tok).view(c, n, h)  # BF16
            gs_p = grad_rows.index_select(0, row_of_slot.index_select(0, flat)).float()  # [c*n, H]
            gs_hi, *gs_rest = split_bf16_terms(gs_p)
            if rest_zero:
                gs_rest = []  # globally checked above: no residual term anywhere
            elif gs_rest and not bool(gs_rest[0].any()):
                gs_rest = []  # the combine backward delivers BF16-exact grads
            gs_hi = gs_hi.view(c, n, h)
            xa = mm_bf16(xp.view(c * n, h), a13b_t).view(c, n, 2 * r)
            xa1p, xa3p = xa[..., :r], xa[..., r:]
            gb2p = mm_bf16([gs_hi.view(c * n, h), *gs_rest], b2b).view(c, n, r)
            w13c = _dequant_experts(experts.ckpt_w13, experts.ckpt_w13_scale, ex_t)
            w2c = _dequant_experts(experts.ckpt_w2, experts.ckpt_w2_scale, ex_t)
            gu = bmm_bf16(xp, w13c.transpose(1, 2))  # [c, n, 2I]
            gu[..., :inter] += _bmm_f32(xa1p, b1v[ex_t].transpose(1, 2))
            gu[..., inter:] += _bmm_f32(xa3p, b3v[ex_t].transpose(1, 2))
            uu = bmm_bf16([gs_hi, *(t.view(c, n, h) for t in gs_rest)], w2c)  # [c, n, I] = GS W2
            del w2c, gs_hi, gs_rest
            a2c = a2v[ex_t]  # [c, r, I]
            uu += _bmm_f32(gb2p, a2c)
            with torch.enable_grad():
                gate = gu[..., :inter].detach().requires_grad_(True)
                up = gu[..., inter:].detach().requires_grad_(True)
                act = _situ_reference(gate, up, beta, lbeta)
                dgate, dup = torch.autograd.grad(act, (gate, up), uu * we_p)
            act = act.detach()
            del gu, gate, up
            dw.index_copy_(0, sink_flat, (act * uu).sum(-1).view(-1))
            del uu
            dgu = torch.cat((dgate, dup), dim=-1)
            dgu_terms = split_bf16_terms(dgu)
            dxp = bmm_bf16(dgu_terms, w13c)  # [c, n, H]
            del dgu_terms, w13c
            # LoRA through the shared outer A factors: d(XA1) = dgate B1, d(XA3) = dup B3
            dgb = torch.cat((_bmm_f32(dgate, b1v[ex_t]), _bmm_f32(dup, b3v[ex_t])), dim=-1)  # [c,n,2r]
            dxp += (dgb.view(c * n, 2 * r) @ a13f).view(c, n, h)
            da13 = dgb.view(c * n, 2 * r).t() @ xp.view(c * n, h).float()  # [2r, H]
            da1 += da13[:r]
            da3 += da13[r:]
            dx_rows.index_copy_(0, sink_row, dxp.view(c * n, h))
            del dxp, dgb, da13
            aa2 = _bmm_f32(act, a2c.transpose(1, 2))  # [c, n, r]
            db2 += gs_p.t() @ (aa2 * we_p).view(c * n, r)
            # experts are distinct within a chunk: index_add_ is deterministic
            da2.index_add_(0, ex_t, _bmm_f32((gb2p * we_p).transpose(1, 2), act))
            db1.index_add_(0, ex_t, _bmm_f32(dgate.transpose(1, 2), xa1p))
            db3.index_add_(0, ex_t, _bmm_f32(dup.transpose(1, 2), xa3p))
            del act, dgate, dup, dgu, aa2, gs_p, xp
        # deterministic token reduction: one index_add per slot k (unique tokens per call)
        k_of = sel % topk
        t_of = sel // topk
        # ascending row lists per slot from one stable sort
        korder = torch.argsort(k_of, stable=True)
        kcnt = torch.bincount(k_of, minlength=topk).tolist()
        ko = 0
        for k in range(topk):
            if kcnt[k]:
                rk = korder[ko : ko + kcnt[k]]
                dx.index_add_(0, t_of[rk], dx_rows[rk])
            ko += kcnt[k]
        del dx_rows
    # stored B = bf16(B * s): straight-through the rounding, chain rule on s
    return (
        None,
        None,
        dx.to(x.dtype),
        None,
        dw[: m * topk].view(m, topk).to(topk_weights.dtype),
        da1.to(w1_a.dtype),
        da3.to(w3_a.dtype),
        (db1 * scaling).to(w1_b.dtype),
        (db3 * scaling).to(w3_b.dtype),
        da2.to(w2_a.dtype),
        (db2 * scaling).to(w2_b.dtype),
    )


def _expert_backward_fp32(ctx: Any, grad_slot: Tensor, dt: torch.dtype = torch.float32):
    """Reference backward: per-expert fp32 recompute through autograd."""
    x, topk_ids, topk_weights, w1_a, w3_a, w1_b, w3_b, w2_a, w2_b = ctx.saved_tensors
    experts: FiW4A16Experts = ctx.experts
    scaling = ctx.spec.scaling
    m, topk = topk_ids.shape
    inter = experts.intermediate
    gs = grad_slot.to(dt)
    dx = torch.zeros(x.shape, dtype=dt, device=x.device)
    dw = torch.zeros((m, topk), dtype=dt, device=x.device)
    da1 = torch.zeros(w1_a.shape, dtype=dt, device=x.device)
    da3 = torch.zeros_like(da1)
    db2 = torch.zeros(w2_b.shape, dtype=dt, device=x.device)
    db1 = torch.zeros(w1_b.shape, dtype=dt, device=x.device)
    db3 = torch.zeros_like(db1)
    da2 = torch.zeros(w2_a.shape, dtype=dt, device=x.device)
    local = topk_ids.long() - experts.local_offset
    valid = (local >= 0) & (local < experts.num_local_experts)
    flat_e = torch.where(valid, local, torch.full_like(local, -1)).reshape(-1)
    order = torch.argsort(flat_e, stable=True)
    counts = torch.bincount(
        flat_e[valid.reshape(-1)], minlength=experts.num_local_experts
    ).tolist()
    start = int((~valid).sum().item())
    a1s = w1_a.detach().to(torch.bfloat16).to(dt)
    a3s = w3_a.detach().to(torch.bfloat16).to(dt)
    b2s = vllm_stored_lora_b(w2_b, scaling).to(dt)
    for e, c in enumerate(counts):
        if c == 0:
            continue
        sel = order[start : start + c]
        start += c
        n_idx, k_idx = sel // topk, sel % topk
        w13 = dequantize_k31_mxfp4(experts.ckpt_w13[e], experts.ckpt_w13_scale[e]).to(dt)
        w2 = dequantize_k31_mxfp4(experts.ckpt_w2[e], experts.ckpt_w2_scale[e]).to(dt)
        with torch.enable_grad():
            xe = x[n_idx].detach().to(dt).requires_grad_(True)
            we = topk_weights[n_idx, k_idx].detach().to(dt).unsqueeze(-1).requires_grad_(True)
            a1 = a1s.clone().requires_grad_(True)
            a3 = a3s.clone().requires_grad_(True)
            b1 = vllm_stored_lora_b(w1_b[e], scaling).to(dt).requires_grad_(True)
            b3 = vllm_stored_lora_b(w3_b[e], scaling).to(dt).requires_grad_(True)
            a2 = w2_a[e].detach().to(torch.bfloat16).to(dt).requires_grad_(True)
            b2 = b2s.clone().requires_grad_(True)
            g = xe @ w13.t()
            gate = g[:, :inter] + (xe @ a1.t()) @ b1.t()
            up = g[:, inter:] + (xe @ a3.t()) @ b3.t()
            act = _situ_reference(gate, up, experts.situ_beta, experts.situ_linear_beta)
            leaf = we * (act @ w2.t()) + we * ((act @ a2.t()) @ b2.t())
            grads = torch.autograd.grad(
                leaf, (xe, we, a1, a3, b1, b3, a2, b2), gs[n_idx, k_idx]
            )
        gxe, gwe, ga1, ga3, gb1, gb3, ga2, gb2 = grads
        dx.index_add_(0, n_idx, gxe)
        dw[n_idx, k_idx] = gwe.squeeze(-1)
        da1 += ga1
        da3 += ga3
        db2 += gb2
        db1[e] = gb1
        db3[e] = gb3
        da2[e] = ga2
    # stored B = bf16(B * s): straight-through the rounding, chain rule on s
    return (
        None,
        None,
        dx.to(x.dtype),
        None,
        dw.to(topk_weights.dtype),
        da1.to(w1_a.dtype),
        da3.to(w3_a.dtype),
        (db1 * scaling).to(w1_b.dtype),
        (db3 * scaling).to(w3_b.dtype),
        da2.to(w2_a.dtype),
        (db2 * scaling).to(w2_b.dtype),
    )


__all__ = [
    "ExpertLoRASpec",
    "VLLMFiW4A16LeafRowsFunction",
    "expert_backward_rows",
    "build_expert_lora_context",
    "dequantize_k31_mxfp4",
    "expert_leaves_reference",
    "vllm_stored_lora_b",
]
