# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Strict adapters for the K31 MoE-path inference kernels.

Same rules as :mod:`megatron.lite.primitive.kernels.vllm_k31` (router logits,
dense / LoRA linears, norms and SiTU come from there): no vLLM model
objects that own weights, no vLLM forward context, no numerical fallbacks.
Every entry resolves to the function the K31 BI=1 rollout executes with the
deploy overlay chain:

* routing: the router object vLLM's ``create_fused_moe_router`` builds for the
  K31 arguments (sigmoid, ``e_score_correction_bias`` for selection only,
  renormalize, ``routed_scaling_factor`` 1.0).
* routed experts: FlashInfer CuTe-DSL W4A16 *private* entries ``moe_sort``,
  ``moe_permute_w4a16``, ``_run_grouped_gemm``, ``_get_workspace`` and
  ``get_max_num_permuted_tokens`` (flashinfer 0.7.3), the
  ``k31_moe_lora_fuse`` glue and punica ``fused_moe_lora`` with the pinned
  configs of the ``fmlora-policy`` overlay.  These private entries are pinned
  here and nowhere else; a FlashInfer upgrade must re-validate this file.
* latent up-projection of ``_k31_bi_tail``: plain ``torch.mm`` (cuBLASLt pins).
* canonical MoE combine: ``k31_tree_allreduce._ext().local_subtree``.

The pieces that are pure host-side policy (tactic choice, LoRA launch configs,
expert-LoRA launch sequence) are called from the overlay module itself so that
training and rollout share one copy.
"""

from __future__ import annotations

import importlib
import types
from dataclasses import dataclass, replace
from typing import Any

import torch
from torch import Tensor

from megatron.lite.primitive.kernels.vllm_k31 import _op, _symbol

_RUNTIME = "megatron.lite.model.k31.vllm.runtime_metadata"
_MX = "vllm.model_executor.layers.quantization.mxfp4"
_LF = "vllm.model_executor.layers.quantization.k31_moe_lora_fuse"
_TREE = "vllm.distributed.device_communicators.k31_tree_allreduce"
_FI = "flashinfer.fused_moe.cute_dsl.blackwell.moe_w4a16"
_FI_PREP = "flashinfer.fused_moe.prepare"
_FI_ENUMS = "flashinfer.tllm_enums"
_PUNICA = "vllm.lora.punica_wrapper.punica_gpu"
_LORA_META = "vllm.lora.ops.triton_ops.lora_kernel_metadata"
_ROUTER_FACTORY = "vllm.model_executor.layers.fused_moe.router.router_factory"


def _module(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise NotImplementedError(
            f"K31 MoE kernel module {name} is unavailable; run inside the K31 BI "
            "image with the full overlay chain mounted."
        ) from exc


# --------------------------------------------------------------------------- runtime
def initialize_k31_moe_batch_invariance() -> None:
    """The skeleton's process-wide BI init plus the MoE-path switches.

    ``initialize_k31_vllm_batch_invariance`` is idempotent and checks the
    cuBLAS pins on the process-start values (vLLM's
    ``override_envs_for_invariance`` rewrites ``CUBLAS_WORKSPACE_CONFIG`` on
    every call).
    """

    _symbol(_RUNTIME, "initialize_k31_vllm_batch_invariance")()


def _require_bi_initialized() -> None:
    if not getattr(_module(_RUNTIME), "_INITIALIZED", False):
        raise RuntimeError("K31 MoE visible calls need initialize_k31_moe_batch_invariance() first")


class MoELatentUpAdapter:
    """``_k31_bi_tail``: ``torch.mm(normed_latent, routed_expert_up_proj.weight.t())``.

    The rollout calls plain ``torch.mm`` here (not ``linear_batch_invariant``);
    under SM100 BI mode it is cuBLASLt with the split-K-free workspace pins
    that the BI init applies before the first cuBLAS handle exists.
    """

    def __call__(self, latent: Tensor, weight: Tensor) -> Tensor:
        if latent.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
            raise TypeError("latent up-projection is BF16 x BF16")
        if latent.ndim != 2 or weight.ndim != 2 or latent.shape[1] != weight.shape[1]:
            raise ValueError("latent [T, L] and weight [H, L] expected")
        _require_bi_initialized()
        return torch.mm(latent, weight.t())


# --------------------------------------------------------------------------- punica
class PunicaShim:
    """Stand-in for ``PunicaWrapperGPU`` with one adapter in slot 0.

    Only the token-mapping metadata is replaced; every method is vLLM's own
    ``PunicaWrapperGPU`` function, so launch shapes and kernels are identical.
    """

    def __init__(
        self,
        num_tokens: int,
        device: torch.device,
        *,
        max_loras: int = 1,
        specialize_active_lora: bool = False,
    ) -> None:
        meta_cls = _symbol(_LORA_META, "LoRAKernelMeta")
        meta = meta_cls.make(max_loras=max_loras, max_num_tokens=max(num_tokens, 1), device=device)
        meta.prepare_tensors(torch.zeros(num_tokens, dtype=torch.int32, device=device))
        self.token_mapping_meta = meta
        self.lora_config = types.SimpleNamespace(specialize_active_lora=specialize_active_lora)
        self.num_tokens = num_tokens
        wrapper = _symbol(_PUNICA, "PunicaWrapperGPU")
        for name in (
            "add_shrink",
            "add_expand",
            "add_lora_linear",
            "moe_lora_align_block_size",
            "add_lora_fused_moe",
        ):
            setattr(self, name, types.MethodType(getattr(wrapper, name), self))

    def token_lora_mapping(self) -> Tensor:
        return self.token_mapping_meta.token_lora_mapping[: self.num_tokens]


# --------------------------------------------------------------------------- router
def build_k31_router(
    *,
    top_k: int,
    num_experts: int,
    e_score_correction_bias: Tensor,
    renormalize: bool = True,
    num_expert_group: int = 1,
    topk_group: int = 1,
    scoring_func: str = "sigmoid",
    routed_scaling_factor: float = 1.0,
):
    """The routing object vLLM builds for ``FusedMoEFactory`` with K31 args.

    It owns no weights (the bias is passed by reference); only
    ``select_experts`` is used.
    """

    factory = _symbol(_ROUTER_FACTORY, "create_fused_moe_router")
    return factory(
        top_k=top_k,
        global_num_experts=num_experts,
        renormalize=renormalize,
        use_grouped_topk=True,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        scoring_func=scoring_func,
        routed_scaling_factor=routed_scaling_factor,
        e_score_correction_bias=e_score_correction_bias,
        skip_padding=True,
    )


# --------------------------------------------------------------------------- canonical tree
def tree16_root(leaves: Tensor) -> Tensor:
    """fp32 root of the K31 16-leaf adjacent-pair tree over ``leaves[16, n]``."""

    if leaves.dtype != torch.float32 or leaves.ndim != 2 or leaves.shape[0] != 16:
        raise ValueError("tree16_root expects fp32 [16, n] leaves")
    return _symbol(_TREE, "_ext")().local_subtree(leaves.contiguous())


# --------------------------------------------------------------------------- FI W4A16 experts
def _fi_w13_rows(w13: Tensor) -> Tensor:
    """``_k31_prepare_fi_w4a16``'s w13 rewrite: rows ``[up; gate]``, 64-row interleave."""

    interleave = _symbol(_FI_PREP, "_interleave_linear_and_gate")
    inter = w13.shape[1] // 2
    w13o = torch.cat([w13[:, inter:], w13[:, :inter]], 1)
    return interleave(w13o, group_size=64, dim=1).contiguous()


_FI_W13_PERM: dict[tuple[int, torch.device], Tensor] = {}


def _fi_w13_perm(w13: Tensor) -> Tensor:
    """Row index map of :func:`_fi_w13_rows`, so the rewrite is one gather instead of three copies.

    Built by running the rewrite itself on the row ids, hence the same permutation by construction.
    """

    key = (w13.shape[1], w13.device)
    perm = _FI_W13_PERM.get(key)
    if perm is None:
        ids = torch.arange(w13.shape[1], device=w13.device).view(1, -1, 1)
        perm = _fi_w13_rows(ids).view(-1)
        _FI_W13_PERM[key] = perm
    return perm


def _gather_rows(w: Tensor, perm: Tensor) -> Tensor:
    """``w.index_select(1, perm)`` with each row moved as 8-byte words (byte gathers are slow)."""

    row = w.shape[-1] * w.element_size()
    word = next(t for t in (torch.int64, torch.int32, torch.int16, torch.uint8) if row % t.itemsize == 0)
    return w.contiguous().view(word).index_select(1, perm).view(w.dtype)


@dataclass
class FiW4A16Experts:
    """EP-local routed experts in the FlashInfer W4A16 layout of the overlay.

    ``ckpt_*`` keep the checkpoint MXFP4 bytes (rows ``[gate; up]``) for the
    BF16/fp32 backward; the FI tensors are the lossless rewrite done by
    ``_k31_prepare_fi_w4a16`` (vLLM process_weights_after_loading).
    """

    w13: Tensor
    w13_sf: Tensor
    w1_alpha: Tensor
    w2: Tensor
    w2_sf: Tensor
    w2_alpha: Tensor
    ckpt_w13: Tensor
    ckpt_w13_scale: Tensor
    ckpt_w2: Tensor
    ckpt_w2_scale: Tensor
    num_experts: int
    local_offset: int
    situ_beta: float
    situ_linear_beta: float | None

    @property
    def num_local_experts(self) -> int:
        return int(self.w13.shape[0])

    @property
    def intermediate(self) -> int:
        return int(self.w13.shape[1]) // 2

    @property
    def hidden(self) -> int:
        return int(self.w2.shape[1])

    def expert_map(self) -> Tensor | None:
        """vLLM's EP expert_map (global -> local, -1 elsewhere); None at EP=1."""

        if self.num_local_experts == self.num_experts:
            return None
        dev = self.ckpt_w13.device
        em = torch.full((self.num_experts,), -1, dtype=torch.int32, device=dev)
        em[self.local_offset : self.local_offset + self.num_local_experts] = torch.arange(
            self.num_local_experts, dtype=torch.int32, device=dev
        )
        return em

    @property
    def w13_cached(self) -> bool:
        return not self.w13.is_meta

    def materialize(self) -> "FiW4A16Experts":
        """This object with the FI ``w13`` rows built from the checkpoint bytes.

        Lean objects (``cache_w13=False``) hold a meta placeholder; the FI rewrite of w13 is a
        pure row permutation of ``ckpt_w13`` (``_interleave_linear_and_gate`` of ``[up; gate]``),
        so building it per call gives the same bytes without a resident second copy.
        """

        if self.w13_cached:
            return self
        return replace(self, w13=_gather_rows(self.ckpt_w13, _fi_w13_perm(self.ckpt_w13)))

    def lean(self) -> "FiW4A16Experts":
        """This object without the FI ``w13`` bytes (backward reads ``ckpt_*`` only)."""

        if not self.w13_cached:
            return self
        return replace(self, w13=torch.empty(self.w13.shape, dtype=self.w13.dtype, device="meta"))

    @classmethod
    def from_checkpoint(
        cls,
        w13: Tensor,
        w13_scale: Tensor,
        w2: Tensor,
        w2_scale: Tensor,
        *,
        num_experts: int,
        local_offset: int,
        situ_beta: float,
        situ_linear_beta: float | None,
        cache_w13: bool = True,
    ) -> "FiW4A16Experts":
        """Port of ``_k31_prepare_fi_w4a16`` (overlay moe-fi-w4a16-bi).

        ``cache_w13=False`` keeps only a meta placeholder for the FI w13 rows (see
        :meth:`materialize`); scales, alphas and w2 (no rewrite) are kept as usual.

        ``w13``: uint8 ``[E, 2I, H/2]`` rows ``[gate; up]``; scales uint8 E8M0
        ``[E, 2I, H/32]``; ``w2``: uint8 ``[E, H, I/2]``.
        """

        mx = _module(_MX)
        interleave = _symbol(_FI_PREP, "_interleave_linear_and_gate")
        for name, value in (
            ("w13", w13),
            ("w13_scale", w13_scale),
            ("w2", w2),
            ("w2_scale", w2_scale),
        ):
            if value.dtype != torch.uint8:
                raise TypeError(f"{name} must be uint8 checkpoint bytes, got {value.dtype}")
        e, n2, kh = w13.shape
        inter = n2 // 2
        if n2 % 128 or (kh * 2) % 256 or (w2.shape[2] * 2) % 256:
            raise ValueError(
                f"FI W4A16 shape constraints violated: {tuple(w13.shape)}, {tuple(w2.shape)}"
            )
        b13, a13 = mx._k31_e8m0_to_e4m3_pow2(w13_scale, "w13")
        b2, a2 = mx._k31_e8m0_to_e4m3_pow2(w2_scale, "w2")
        b13o = torch.cat([b13[:, inter:], b13[:, :inter]], 1)
        if cache_w13:
            w13i = _fi_w13_rows(w13)
        else:
            w13i = torch.empty(w13.shape, dtype=w13.dtype, device="meta")
        sf13 = mx._k31_swizzle_mma(interleave(b13o, group_size=64, dim=1).contiguous())
        sf2 = mx._k31_swizzle_mma(b2.contiguous())
        return cls(
            w13=w13i,
            w13_sf=sf13,
            w1_alpha=a13,
            w2=w2.contiguous(),
            w2_sf=sf2,
            w2_alpha=a2,
            ckpt_w13=w13,
            ckpt_w13_scale=w13_scale,
            ckpt_w2=w2,
            ckpt_w2_scale=w2_scale,
            num_experts=num_experts,
            local_offset=local_offset,
            situ_beta=float(situ_beta),
            situ_linear_beta=None if situ_linear_beta is None else float(situ_linear_beta),
        )


@dataclass
class ExpertLoRAContext:
    """The fields of vLLM's ``MoELoRAContext`` that the FI LoRA path reads."""

    w13_lora_a_stacked: tuple[Tensor, ...]
    w13_lora_b_stacked: tuple[Tensor, ...]
    w2_lora_a_stacked: tuple[Tensor, ...]
    w2_lora_b_stacked: tuple[Tensor, ...]
    adapter_enabled: Tensor
    max_loras: int
    top_k: int
    local_num_experts: int
    punica_wrapper: PunicaShim
    enable_moe_shared_loras: bool = True
    fully_sharded: bool = False
    tp_rank: int = 0


def _fi_lora_policy() -> Any:
    """Bound-method host for the overlay's static LoRA launch policy."""

    cls = _symbol(_MX, "K31FiW4A16Experts")
    return types.SimpleNamespace(_k31_lora_cfgs=cls._k31_lora_cfgs), cls


def fi_w4a16_lora_leaves(
    x: Tensor,
    topk_ids: Tensor,
    topk_weights: Tensor,
    experts: FiW4A16Experts,
    lora: ExpertLoRAContext,
) -> Tensor:
    """Port of ``K31FiW4A16Experts._k31_apply_lora`` up to the per-slot leaves.

    Returns ``slot[M, top_k, H]`` fp32 in router-slot order:
    ``fp32(w_k * base_k) + fp32(d2_k)`` for slots whose expert is local, and
    exactly zero for the other slots.  ``topk_ids`` are global expert ids.
    """

    fi = _module(_FI)
    lf = _module(_LF)
    mx = _module(_MX)
    enums = _module(_FI_ENUMS)
    host, cls = _fi_lora_policy()
    if x.dtype != torch.bfloat16 or x.ndim != 2:
        raise TypeError("FI W4A16 expects BF16 [tokens, hidden]")
    m, topk = topk_ids.shape
    hdim = x.shape[1]
    e_loc = experts.num_local_experts
    inter = experts.intermediate
    dev = x.device
    ids32 = topk_ids.to(torch.int32)
    ww = topk_weights.to(torch.float32)
    tactic = mx._k31_pick_tactic(m, topk, experts.num_experts)
    rt = tactic[0][0][1]
    ws = fi._get_workspace(x, topk, experts.num_experts, e_loc, inter, rt)
    t2e, t2l, e2p, p2e, _, nnt = fi.moe_sort(
        token_selected_experts=ids32,
        token_final_scales=ww,
        num_experts=experts.num_experts,
        top_k=topk,
        local_expert_offset=experts.local_offset,
        num_local_experts=e_loc,
        tile_tokens_dim=rt,
        enable_pdl=False,
        **ws.moe_sort_buffers,
    )
    s = fi.get_max_num_permuted_tokens(m, topk, e_loc, rt)
    t2e, t2l, e2p, p2e = t2e[: s // rt], t2l[: s // rt], e2p[:m], p2e[:s]
    hid = ws.hidden_workspace[:s]
    fi.moe_permute_w4a16(
        input=x,
        permuted_output=hid,
        tile_idx_to_mn_limit=t2l,
        permuted_idx_to_expanded_idx=p2e,
        num_non_exiting_tiles=nnt,
        max_num_permuted_tokens=s,
        top_k=topk,
        tile_size=rt,
        enable_pdl=False,
    )
    common = dict(
        tile_idx_to_expert_idx=t2e,
        tile_idx_to_mn_limit=t2l,
        num_non_exiting_tiles=nnt,
        num_local_experts=e_loc,
        activation_type=None,
        swiglu_alpha=enums.DEFAULT_SWIGLU_ALPHA,
        swiglu_beta=enums.DEFAULT_SWIGLU_BETA,
        swiglu_limit=enums.DEFAULT_SWIGLU_LIMIT,
        situ_beta=None,
        situ_linear_beta=None,
        use_fused_finalize=False,
        permuted_idx_to_expanded_idx=None,
        token_final_scales=None,
        enable_pdl=False,
    )
    g1 = torch.empty(s, 2 * inter, device=dev, dtype=torch.bfloat16)
    fi._run_grouped_gemm(
        weight=experts.w13,
        weight_sf=experts.w13_sf,
        activations=hid,
        alpha=experts.w1_alpha,
        output=g1,
        tactic=tactic[0],
        **common,
    )
    e2p32 = lf.i32(e2p)
    ly = torch.zeros(m * topk, 2 * inter, device=dev, dtype=torch.bfloat16)
    tlm = lora.punica_wrapper.token_lora_mapping()
    expert_map = experts.expert_map()
    sorted_l, eids_l, npad_l, tlm_l = cls._k31_lora_w13(
        host, lora, ly, x, topk_ids, topk_weights, expert_map, m, topk, tlm
    )
    x32 = lf.gather_add(g1, e2p32, ly, inter)
    del ly, g1
    y32 = torch.empty(m * topk, inter, device=dev, dtype=torch.float32)
    _op("_C", "situ_and_mul")(
        y32,
        x32,
        experts.situ_beta,
        -1.0 if experts.situ_linear_beta is None else experts.situ_linear_beta,
    )
    del x32
    act, act_s = lf.act_cast(y32, lf.i32(p2e))
    del y32
    o2 = torch.empty(s, hdim, device=dev, dtype=torch.bfloat16)
    fi._run_grouped_gemm(
        weight=experts.w2,
        weight_sf=experts.w2_sf,
        activations=act_s,
        alpha=experts.w2_alpha,
        output=o2,
        tactic=tactic[1],
        **common,
    )
    d2 = torch.zeros(m, topk, hdim, device=dev, dtype=torch.bfloat16)
    cls._k31_lora_w2(host, lora, d2, act, topk_weights, sorted_l, eids_l, npad_l, tlm_l, topk)
    slot = lf.slot(o2, e2p32, ww, d2)
    return slot.view(m, topk, hdim)


__all__ = [
    "ExpertLoRAContext",
    "FiW4A16Experts",
    "MoELatentUpAdapter",
    "PunicaShim",
    "build_k31_router",
    "fi_w4a16_lora_leaves",
    "initialize_k31_moe_batch_invariance",
    "tree16_root",
]
