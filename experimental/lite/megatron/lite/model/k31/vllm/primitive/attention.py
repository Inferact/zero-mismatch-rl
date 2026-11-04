"""K31 DSA attention-core owner: indexer selection, fp8_ds_mla KV, absorbed MLA, gate.

Forward values come from the exact vLLM calls of the K31 BI=1 rollout
(``primitive/kernels/vllm_k31.py``); gradients come from separate owners:

* indexer top-k: frozen, discrete, inference-only (no gradient, no aux loss);
* W_UK^T absorb / W_UV up-projection: the vLLM batch-invariant ``bmm`` forward
  with an FP32 VJP for the activation (and for W only if it is trainable);
* sparse attention: the pinned FlashMLA fp8 kernel forward; the vendor cuDNN
  DSA backward runs on the indices and natural-log LSE saved from that same
  visible call, with the KV taken as the official bf16 upconvert of the cache
  (STE through the fp8_ds_mla quantization: identity w.r.t. the dequantized
  value, the bf16 tail is exact) and ``attn_sink=-inf`` (K31 has no sink);
* output gate ``attn * sigmoid(g)``: vLLM's eager bf16 ops with an FP32 VJP.

Port of ``deepseek_v4/vllm/primitive/attention.py``: the
``_VisibleSparseAttentionFunction`` / ``attention_core`` pattern is kept;
compressor, SWA, inverse RoPE and the indexer auxiliary loss are dropped.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch

from ._contract import own_visible_tensor as _own

# K31 has no attention sink: -inf adds exp(-inf) = 0 to the softmax
# denominator.  A zero sink would add exp(0) and is wrong (measured:
# cos 0.98 vs 0.999997 against the FP32 reference).
_NO_SINK = float("-inf")


@dataclass
class DSAAdapters:
    """The vLLM entries one K31 DSA layer calls (``primitive/kernels/vllm_k31.py``)."""

    indexer: Any
    kv_insert: Any
    sparse: Any
    upconvert: Any
    bmm: Any


def default_dsa_adapters() -> DSAAdapters:
    from megatron.lite.primitive.kernels.vllm_k31 import (
        DSAAbsorbBmmAdapter,
        DSAFP8KVUpconvertAdapter,
        DSAIndexerAdapter,
        DSAMLAKVInsertAdapter,
        DSASparseMLAAdapter,
    )

    return DSAAdapters(
        indexer=DSAIndexerAdapter(),
        kv_insert=DSAMLAKVInsertAdapter(),
        sparse=DSASparseMLAAdapter(),
        upconvert=DSAFP8KVUpconvertAdapter(),
        bmm=DSAAbsorbBmmAdapter(),
    )


def _ensure_dsa_backward():
    from megatron.lite.primitive.kernels import dsa_kernels

    dsa_kernels._ensure_dsa_namespace()
    return dsa_kernels._DSA.sparse_attention_backward_wrapper


def default_sparse_backward(q, kv, out, grad_out, lse, indices, scale, *, deterministic=True):
    """cuDNN DSA sparse-attention VJP at K31 geometry (d_qk 576, d_v 512, no sink).

    ``q`` (T, H, 576) bf16, ``kv`` (S, 576) bf16 workspace, ``out``/``grad_out``
    (T, H, 512) bf16, ``lse`` (T, H) fp32 natural log, ``indices`` (T, K) int32
    workspace rows with a -1 tail.
    """

    backward = _ensure_dsa_backward()
    sink = torch.full((q.shape[1],), _NO_SINK, dtype=torch.float32, device=q.device)
    result = backward(
        q.contiguous(),
        kv.contiguous(),
        out.contiguous(),
        grad_out.contiguous(),
        lse.contiguous(),
        sink,
        indices.contiguous(),
        softmax_scale=scale,
        topk_length=None,
        deterministic=deterministic,
    )
    return result["dq"], result["dkv"]


class _AbsorbFunction(torch.autograd.Function):
    """``ql_nope = q_nope @ W_UK^T`` per head, vLLM BI bmm forward, FP32 VJP."""

    @staticmethod
    def forward(ctx, adapter, q_nope, w_uk_t):
        ctx.save_for_backward(q_nope, w_uk_t)
        return _own(adapter.absorb(q_nope.detach(), w_uk_t.detach()))

    @staticmethod
    def backward(ctx, grad):
        q_nope, w_uk_t = ctx.saved_tensors
        dq = dw = None
        g = grad.float()
        if ctx.needs_input_grad[1]:
            dq = torch.einsum("bnl,npl->bnp", g, w_uk_t.float()).to(q_nope.dtype)
        if ctx.needs_input_grad[2]:
            dw = torch.einsum("bnp,bnl->npl", q_nope.float(), g).to(w_uk_t.dtype)
        return None, dq, dw


class _VUpFunction(torch.autograd.Function):
    """``o[b, n] = attn[b, n] @ W_UV[n]`` written as (T, N*V), BI bmm forward, FP32 VJP."""

    @staticmethod
    def forward(ctx, adapter, attn, w_uv):
        ctx.save_for_backward(attn, w_uv)
        n, _lr, v = w_uv.shape
        out = attn.new_empty((attn.shape[0], n * v))
        return _own(adapter.v_up(attn.detach(), w_uv.detach(), out))

    @staticmethod
    def backward(ctx, grad):
        attn, w_uv = ctx.saved_tensors
        n, _lr, v = w_uv.shape
        g = grad.float().view(-1, n, v)
        dattn = dw = None
        if ctx.needs_input_grad[1]:
            dattn = torch.einsum("bnv,nlv->bnl", g, w_uv.float()).to(attn.dtype)
        if ctx.needs_input_grad[2]:
            dw = torch.einsum("bnl,bnv->nlv", attn.float(), g).to(w_uv.dtype)
        return None, dattn, dw


class _OutputGateFunction(torch.autograd.Function):
    """vLLM ``attn_out * g_proj(h).sigmoid()`` (two bf16 eager ops), FP32 VJP."""

    @staticmethod
    def forward(ctx, attn, g):
        ctx.save_for_backward(attn, g)
        return attn * g.sigmoid()

    @staticmethod
    def backward(ctx, grad):
        attn, g = ctx.saved_tensors
        sig = torch.sigmoid(g.float())
        dy = grad.float()
        dattn = (dy * sig).to(attn.dtype) if ctx.needs_input_grad[0] else None
        dg = (
            (dy * attn.float() * sig * (1.0 - sig)).to(g.dtype) if ctx.needs_input_grad[1] else None
        )
        return dattn, dg


class _VisibleSparseAttentionFunction(torch.autograd.Function):
    """Differentiate the official fp8 sparse-attention visible invocation.

    ``visible_op(q, kv_rows)`` inserts the token rows into the fp8_ds_mla
    cache, runs the pinned FlashMLA kernel, and returns
    ``(out, lse, workspace, workspace_indices)`` -- the tensors actually
    consumed or produced inside that call.  Backward runs the vendor VJP on
    them; ``dkv`` on workspace rows is the token-row gradient (identity STE).
    """

    @staticmethod
    def forward(ctx, visible_op, backward_op, scale, q, kv_rows):
        result = visible_op(q.detach(), kv_rows.detach())
        if not isinstance(result, (tuple, list)) or len(result) != 4:
            raise RuntimeError("K31 sparse MLA visible op must return out, lse, workspace, indices")
        out, lse, workspace, indices = result
        ctx.save_for_backward(q.detach(), workspace, out, lse, indices)
        ctx.backward_op = backward_op or default_sparse_backward
        ctx.scale = scale
        ctx.kv_rows = kv_rows.shape[0]
        return _own(out)

    @staticmethod
    def backward(ctx, grad_out):
        q, workspace, out, lse, indices = ctx.saved_tensors
        dq, dkv = ctx.backward_op(q, workspace, out, grad_out, lse, indices, ctx.scale)
        dkv = dkv.reshape(workspace.shape)
        if not bool(torch.isfinite(dq).all()) or not bool(torch.isfinite(dkv).all()):
            raise FloatingPointError(
                "K31 FlashMLA sparse backward produced non-finite gradients: "
                f"dq_finite={bool(torch.isfinite(dq).all())}, "
                f"dkv_finite={bool(torch.isfinite(dkv).all())}"
            )
        if dkv.shape[0] != ctx.kv_rows:
            raise RuntimeError("K31 DSA workspace rows must equal packed token rows")
        return None, None, None, dq, dkv.to(q.dtype)


def visible_sparse_attention(
    visible_op: Callable[..., Any],
    q: torch.Tensor,
    kv_rows: torch.Tensor,
    *,
    softmax_scale: float,
    backward_op: Callable[..., Any] | None = None,
) -> torch.Tensor:
    return _VisibleSparseAttentionFunction.apply(visible_op, backward_op, softmax_scale, q, kv_rows)


def indexer_topk(
    adapter,
    hidden_states: torch.Tensor,
    q_c: torch.Tensor,
    *,
    weights: dict[str, Any],
    num_heads: int,
    head_dim: int,
    topk: int,
    metadata: Any,
) -> torch.Tensor:
    """Frozen indexer: logical (request-local) top-k, -1 tail; no gradient."""

    topk_indices = adapter(
        hidden_states,
        q_c,
        wq=weights["wq"],
        wk=weights["wk"],
        wv=weights["wv"],
        k_norm_weight=weights["k_norm_weight"],
        k_norm_bias=weights["k_norm_bias"],
        k_norm_eps=weights["k_norm_eps"],
        num_heads=num_heads,
        head_dim=head_dim,
        topk=topk,
        metadata=metadata,
    )
    return topk_indices.detach()


def attention_core(
    adapters,
    q: torch.Tensor,
    kv_c_normed: torch.Tensor,
    k_pe: torch.Tensor,
    logical_topk: torch.Tensor,
    metadata: Any,
    *,
    w_uk_t: torch.Tensor,
    w_uv: torch.Tensor,
    softmax_scale: float,
    qk_nope_head_dim: int,
    padded_heads: int,
    backward_op: Callable[..., Any] | None = None,
) -> torch.Tensor:
    """K31 absorbed sparse MLA: returns the pre-gate attention output (T, H*V).

    ``q`` (T, H, nope+rope) is the q_b_proj output; ``kv_c_normed`` (T, 512)
    and ``k_pe`` (T, 64) the fused_qkv_a outputs after kv_a_layernorm (K31 is
    NoPE: k_pe / q_pe are not rotated).  ``logical_topk`` (T, 2048) holds
    request-local token indices from :func:`indexer_topk`.
    """

    if q.ndim != 3:
        raise ValueError("q must be (T, H, qk_head_dim)")
    q_nope, q_pe = q.split([qk_nope_head_dim, q.shape[-1] - qk_nope_head_dim], dim=-1)
    ql_nope = _AbsorbFunction.apply(adapters.bmm, q_nope, w_uk_t)
    q_cat = torch.cat((ql_nope, q_pe), dim=-1)
    kv_rows = torch.cat((kv_c_normed, k_pe), dim=-1)
    logical = logical_topk.detach()
    global_topk = adapters.sparse.convert_indices(
        metadata.req_id_per_token, metadata.block_table, logical, block_size=metadata.block_size
    )
    workspace_topk = torch.where(
        logical >= 0,
        logical
        + metadata.workspace_starts.index_select(0, metadata.req_id_per_token.long())[:, None],
        logical,
    )

    def visible(q_value: torch.Tensor, kv_value: torch.Tensor):
        # kv_value is the detached autograd edge; insert the original rows exactly as vLLM does.
        del kv_value
        adapters.kv_insert(
            kv_c_normed.detach(),
            k_pe.detach(),
            metadata.mla_cache,
            metadata.slot_mapping,
            metadata.k_scale,
        )
        out, lse = adapters.sparse(
            q_value,
            metadata.mla_cache,
            global_topk,
            softmax_scale=softmax_scale,
            padded_heads=padded_heads,
            cache_lens=metadata.cache_lens,
            dummy_block_table=metadata.dummy_block_table,
        )
        workspace = q_value.new_empty((q_value.shape[0], q_value.shape[-1]))
        adapters.upconvert(
            metadata.mla_cache, workspace, metadata.block_table, metadata.workspace_starts
        )
        return out, lse, workspace, workspace_topk

    attn = visible_sparse_attention(
        visible, q_cat, kv_rows, softmax_scale=softmax_scale, backward_op=backward_op
    )
    return _VUpFunction.apply(adapters.bmm, attn, w_uv)


def absorbed_weights(
    kv_b_proj_weight: torch.Tensor,
    *,
    kv_lora_rank: int,
    num_heads: int,
    qk_nope_head_dim: int,
    v_head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``W_UK_T`` (N, P, L) and ``W_UV`` (N, L, V) exactly as vLLM lays them out.

    Mirrors ``split_kv_b_proj`` + ``MLAAttention.process_weights_after_loading``
    (unquantized bf16 kv_b_proj): the same views, hence the same strides.
    Differentiable in ``kv_b_proj_weight`` (K31 freezes it: not a LoRA target).
    """

    weight = kv_b_proj_weight.T
    if weight.shape != (kv_lora_rank, num_heads * (qk_nope_head_dim + v_head_dim)):
        raise ValueError(
            f"kv_b_proj weight {tuple(kv_b_proj_weight.shape)} does not match MLA dims"
        )
    weight = weight.view(kv_lora_rank, num_heads, qk_nope_head_dim + v_head_dim)
    w_uk, w_uv = weight.split([qk_nope_head_dim, v_head_dim], dim=-1)
    return w_uk.permute(1, 2, 0), w_uv.transpose(0, 1)


def output_gate(attn: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """``attn * sigmoid(g)`` with vLLM's eager bf16 values and an FP32 VJP."""

    return _OutputGateFunction.apply(attn, g)


__all__ = [
    "DSAAdapters",
    "absorbed_weights",
    "default_dsa_adapters",
    "attention_core",
    "default_sparse_backward",
    "indexer_topk",
    "output_gate",
    "visible_sparse_attention",
]
