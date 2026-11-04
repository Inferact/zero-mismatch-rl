"""K31 routed-expert transport and the canonical-slot combine owner.

Contract (deployed by the rollout's ``tp-invariant`` / ``moe-tree-ep2`` /
``moe-tree-dp`` overlays)::

    leaf[t, k]  = fl32(fl32(w_tk * base_tk) + fp32(d2_tk))      # FI + expert LoRA
    out[t]      = bf16(tree16_c(leaf[t, slot_c]))   c = expert-id ascending
    out[t]      = +0 where out[t] == 0

``tree16`` is the adjacent-pair balanced fp32 tree of ``local_subtree``.  The
result depends only on the expert ids, never on the EP size or placement.

Training transport: tokens reach the expert ranks by normal DeepEP dispatch
(``transport="deepep"``; a torch all-to-all ``"nccl"`` transport moves the
same bytes when DeepEP is unavailable).  The fp32 leaves of every routed slot
return to the token's rank with ``all_to_all_single`` (DeepEP combine sums in
its own order and is BF16-only, so it cannot carry the contract), are placed
in canonical slot order and reduced by the inference kernel's own subtree.

Backward: the tree is a sum, so every leaf receives ``grad_out[t]`` (straight
through the single BF16 rounding); those rows travel back by the inverse
all-to-all.  Dispatch gradients use DeepEP combine / the inverse all-to-all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor

from megatron.lite.primitive.kernels.vllm_k31_moe import tree16_root

_TOKEN_SHIFT = 5  # slot k < 32
_RANK_SHIFT = 40


def _group_size(group) -> int:
    if group is None or not dist.is_available() or not dist.is_initialized():
        return 1
    return dist.get_world_size(group)


def _group_rank(group) -> int:
    if group is None or not dist.is_available() or not dist.is_initialized():
        return 0
    return dist.get_rank(group)


def _exchange_counts(counts: Tensor, group) -> Tensor:
    out = torch.empty_like(counts)
    dist.all_to_all_single(out, counts.contiguous(), group=group)
    return out


def _a2a(tensor: Tensor, out_rows: list[int], in_rows: list[int], group) -> Tensor:
    out = tensor.new_empty((sum(out_rows),) + tuple(tensor.shape[1:]))
    dist.all_to_all_single(
        out,
        tensor.contiguous(),
        output_split_sizes=out_rows,
        input_split_sizes=in_rows,
        group=group,
    )
    return out


# --------------------------------------------------------------------------- dispatch
@dataclass
class DispatchedTokens:
    """Expert-side view of one dispatch."""

    hidden: Tensor  # [N, H] BF16
    topk_ids: Tensor  # [N, K] global expert ids; non-local slots hold ``sentinel``
    topk_weights: Tensor  # [N, K] fp32
    local_mask: Tensor  # [N, K] bool, slot expert lives on this rank
    src_rank: Tensor  # [N] int64
    src_token: Tensor  # [N] int64


class _NCCLTokenDispatch(torch.autograd.Function):
    """Token-level all-to-all: a token goes once to every rank owning one of its experts."""

    @staticmethod
    def forward(ctx: Any, hidden: Tensor, weights: Tensor, ids: Tensor, group, local_experts: int):
        ep = _group_size(group)
        t, k = ids.shape
        dest = ids.long() // local_experts
        hits = torch.zeros((t, ep), dtype=torch.bool, device=ids.device)
        hits.scatter_(1, dest, True)
        rank_idx, tok_idx = torch.nonzero(hits.t(), as_tuple=True)  # sorted by (rank, token)
        send = torch.bincount(rank_idx, minlength=ep)
        recv = _exchange_counts(send, group)
        send_l, recv_l = send.tolist(), recv.tolist()
        recv_hidden = _a2a(hidden.index_select(0, tok_idx), recv_l, send_l, group)
        recv_weights = _a2a(weights.float().index_select(0, tok_idx), recv_l, send_l, group)
        recv_ids = _a2a(ids.long().index_select(0, tok_idx), recv_l, send_l, group)
        recv_tok = _a2a(tok_idx, recv_l, send_l, group)
        src_rank = torch.repeat_interleave(
            torch.arange(ep, device=ids.device), recv, output_size=sum(recv_l)
        )
        ctx.group, ctx.send_l, ctx.recv_l = group, send_l, recv_l
        ctx.save_for_backward(tok_idx)
        ctx.shape = (t, hidden.shape[1], k)
        ctx.mark_non_differentiable(recv_ids, recv_tok, src_rank)
        return recv_hidden, recv_weights, recv_ids, recv_tok, src_rank

    @staticmethod
    def backward(ctx: Any, g_hidden, g_weights, _g_ids, _g_tok, _g_rank):
        (tok_idx,) = ctx.saved_tensors
        t, h, k = ctx.shape
        dev = tok_idx.device
        if g_hidden is None:
            g_hidden = torch.zeros((sum(ctx.recv_l), h), device=dev)
        if g_weights is None:
            g_weights = torch.zeros((sum(ctx.recv_l), k), device=dev)
        # autograd hands the grad of the BF16 recv_hidden in BF16: send it as is (half the bytes),
        # the upcast after the exchange is exact
        g_send = g_hidden if g_hidden.dtype == torch.bfloat16 else g_hidden.float()
        back_h = _a2a(g_send, ctx.send_l, ctx.recv_l, ctx.group).float()
        back_w = _a2a(g_weights.float(), ctx.send_l, ctx.recv_l, ctx.group)
        dh = torch.zeros((t, h), dtype=torch.float32, device=dev).index_add_(0, tok_idx, back_h)
        dw = torch.zeros((t, k), dtype=torch.float32, device=dev).index_add_(0, tok_idx, back_w)
        return dh.to(torch.bfloat16), dw, None, None, None


def _deepep_dispatch(hidden: Tensor, weights: Tensor, ids: Tensor, group, num_experts: int):
    """Normal DeepEP dispatch (mLite's ``_DeepEPDispatch`` owner) + exact source tags."""

    from megatron.lite.primitive.modules.dispatcher import (
        _DeepEPDispatch,
        _get_deepep_buffer,
    )

    rank = _group_rank(group)
    t = hidden.shape[0]
    recv_hidden, recv_idx, recv_w, _, handle = _DeepEPDispatch.apply(
        group,
        hidden,
        ids.long().contiguous(),
        weights.float().contiguous(),
        num_experts,
        False,
        False,
    )
    # Source tags through the cached layout: small integers are exact in BF16 and
    # the dispatch is a byte copy.
    tok = torch.arange(t, device=hidden.device)
    tag = torch.zeros((t, 16), dtype=torch.bfloat16, device=hidden.device)
    tag[:, 0] = float(rank)
    tag[:, 1] = (tok // 256).to(torch.bfloat16)
    tag[:, 2] = (tok % 256).to(torch.bfloat16)
    tag[:, 3] = 1.0
    buffer = _get_deepep_buffer(group, 32)
    recv_tag = buffer.dispatch(tag, handle=handle)[0]
    if recv_tag.shape[0] != recv_hidden.shape[0] or not bool((recv_tag[:, 3] == 1).all()):
        raise RuntimeError("DeepEP tag dispatch does not match the primary dispatch layout")
    src_rank = recv_tag[:, 0].long()
    src_tok = recv_tag[:, 1].long() * 256 + recv_tag[:, 2].long()
    return recv_hidden, recv_idx, recv_w, src_rank, src_tok


def dispatch_tokens(
    hidden: Tensor,
    topk_weights: Tensor,
    topk_ids: Tensor,
    *,
    group,
    num_experts: int,
    transport: str,
) -> DispatchedTokens:
    """Move every token to the EP ranks that own its experts.

    ``topk_ids`` are global expert ids ``[T, K]``; experts are placed in
    contiguous blocks of ``num_experts // ep`` per rank (vLLM's EP placement).
    """

    ep = _group_size(group)
    rank = _group_rank(group)
    e_loc = num_experts // ep
    if num_experts % ep:
        raise ValueError("routed experts must divide evenly across EP ranks")
    offset = rank * e_loc
    sentinel = (offset + e_loc) % num_experts
    if ep == 1 or transport == "local":
        if ep != 1:
            raise ValueError("local transport requires EP=1")
        t = hidden.shape[0]
        return DispatchedTokens(
            hidden=hidden,
            topk_ids=topk_ids.long(),
            topk_weights=topk_weights.float(),
            local_mask=torch.ones_like(topk_ids, dtype=torch.bool),
            src_rank=torch.zeros(t, dtype=torch.long, device=hidden.device),
            src_token=torch.arange(t, device=hidden.device),
        )
    if transport == "deepep":
        recv_hidden, recv_idx, recv_w, src_rank, src_tok = _deepep_dispatch(
            hidden, topk_weights, topk_ids, group, num_experts
        )
        local = recv_idx >= 0
        ids = torch.where(
            local, recv_idx.long() + offset, torch.full_like(recv_idx.long(), sentinel)
        )
        weights = recv_w.float()
    elif transport == "nccl":
        recv_hidden, weights, recv_ids, src_tok, src_rank = _NCCLTokenDispatch.apply(
            hidden, topk_weights.float(), topk_ids, group, e_loc
        )
        local = (recv_ids >= offset) & (recv_ids < offset + e_loc)
        ids = torch.where(local, recv_ids, torch.full_like(recv_ids, sentinel))
    else:
        raise ValueError(f"unknown MoE transport {transport!r}")
    return DispatchedTokens(
        hidden=recv_hidden,
        topk_ids=ids,
        topk_weights=weights,
        local_mask=local,
        src_rank=src_rank,
        src_token=src_tok,
    )


# --------------------------------------------------------------------------- combine
@dataclass
class CanonicalCombinePlan:
    group: Any
    ep: int
    num_tokens: int
    send_counts: list[int]  # token side: routes whose expert lives on rank r
    recv_counts: list[int]  # expert side: routes received from rank r
    expert_rows: Tensor  # expert side: flat (n * K + k) of local slots, send order
    canon_pos: Tensor  # token side: c * T + t for each arriving leaf
    src_token: Tensor  # token side: t for each arriving leaf


def plan_canonical_combine(
    topk_ids: Tensor,
    dispatched: DispatchedTokens,
    *,
    group,
    num_experts: int,
) -> CanonicalCombinePlan:
    """Route bookkeeping shared by forward and backward (no data movement but counts)."""

    ep = _group_size(group)
    t, k = topk_ids.shape
    if k != 16:
        raise ValueError("the K31 canonical tree has exactly 16 leaves")
    dev = topk_ids.device
    ids = topk_ids.long()
    e_loc = num_experts // ep
    # token side: leaves arrive grouped by expert rank, each block sorted by (t, k)
    flat = torch.arange(t * k, device=dev)
    dest = (ids // e_loc).reshape(-1)
    order_src = torch.argsort(dest * (t * k) + flat, stable=True)
    send = torch.bincount(dest, minlength=ep)
    perm = torch.argsort(ids, dim=1, stable=True)
    inv = torch.empty_like(perm)
    inv.scatter_(1, perm, torch.arange(k, device=dev).expand(t, k))
    src_token = order_src // k
    canon_pos = inv.reshape(-1)[order_src] * t + src_token
    # expert side: local slots sorted by (source rank, source token, k)
    n = dispatched.hidden.shape[0]
    local = dispatched.local_mask.reshape(-1)
    flat_e = torch.arange(n * k, device=dev)[local]
    nn_, kk = flat_e // k, flat_e % k
    key = (
        (dispatched.src_rank[nn_] << _RANK_SHIFT) | (dispatched.src_token[nn_] << _TOKEN_SHIFT) | kk
    )
    expert_rows = flat_e[torch.argsort(key)]
    recv = torch.bincount(dispatched.src_rank[nn_], minlength=ep)
    if ep > 1:
        expected = _exchange_counts(send, group)
        if not torch.equal(expected, recv):
            raise RuntimeError(
                f"MoE combine route counts disagree: token side sends {expected.tolist()}, "
                f"expert side holds {recv.tolist()}"
            )
    elif int(recv.sum()) != t * k:
        raise RuntimeError("EP=1 combine must own every routed slot")
    return CanonicalCombinePlan(
        group=group,
        ep=ep,
        num_tokens=t,
        send_counts=send.tolist(),
        recv_counts=recv.tolist(),
        expert_rows=expert_rows,
        canon_pos=canon_pos,
        src_token=src_token,
    )


def canonical_tree_reduce(leaves16: Tensor, num_tokens: int, hidden: int) -> Tensor:
    """``[16 * T, H]`` fp32 leaves in canonical order -> contract BF16 ``[T, H]``."""

    root = tree16_root(leaves16.view(16, num_tokens * hidden)).view(num_tokens, hidden)
    out = root.to(torch.bfloat16)
    return out.masked_fill_(out == 0, 0.0)


class _CanonicalCombine(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, leaf_rows: Tensor, plan: CanonicalCombinePlan) -> Tensor:
        if leaf_rows.dtype != torch.float32:
            raise TypeError("MoE combine leaves must be fp32")
        h = leaf_rows.shape[1]
        if plan.ep > 1:
            arrived = _a2a(leaf_rows, plan.send_counts, plan.recv_counts, plan.group)
        else:
            arrived = leaf_rows
        t = plan.num_tokens
        if arrived.shape[0] != 16 * t:
            raise RuntimeError(f"combine received {arrived.shape[0]} leaves for {t} tokens")
        leaves16 = torch.empty((16 * t, h), dtype=torch.float32, device=leaf_rows.device)
        leaves16.index_copy_(0, plan.canon_pos, arrived)
        ctx.plan = plan
        return canonical_tree_reduce(leaves16, t, h)

    @staticmethod
    def backward(ctx: Any, grad_out: Tensor):
        plan: CanonicalCombinePlan = ctx.plan
        rows = grad_out.index_select(0, plan.src_token)
        if plan.ep > 1:
            rows = _a2a(rows, plan.recv_counts, plan.send_counts, plan.group)
        return rows.float(), None


def canonical_combine(slot: Tensor, plan: CanonicalCombinePlan) -> Tensor:
    """Expert-side ``slot[N, 16, H]`` fp32 leaves -> token-side BF16 latent."""

    n, k, h = slot.shape
    leaf_rows = slot.reshape(n * k, h).index_select(0, plan.expert_rows)
    return _CanonicalCombine.apply(leaf_rows, plan)


def combine_reference(leaves: Tensor, topk_ids: Tensor) -> Tensor:
    """Single-GPU torch statement of the contract (``k31_tree_ref.moe_combine_leaves_ref``)."""

    order = topk_ids.long().argsort(dim=1, stable=True)
    lc = leaves.float().gather(1, order[..., None].expand_as(leaves))
    values = [lc[:, c] for c in range(lc.shape[1])]
    while len(values) > 1:
        values = [values[i] + values[i + 1] for i in range(0, len(values), 2)]
    out = values[0].to(torch.bfloat16)
    return torch.where(out == 0, torch.zeros_like(out), out)


__all__ = [
    "CanonicalCombinePlan",
    "DispatchedTokens",
    "canonical_combine",
    "canonical_tree_reduce",
    "combine_reference",
    "dispatch_tokens",
    "plan_canonical_combine",
]
