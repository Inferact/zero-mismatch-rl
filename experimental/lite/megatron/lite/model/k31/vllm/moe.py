"""Latent MoE slot: forward bitwise equal to the K31 BI=1 rollout.

Rollout statement (``KimiMoE.forward`` + ``LatentMoERunner._k31_bi_tail`` under
``VLLM_BATCH_INVARIANT=1``, TP=1, EP)::

    logits  = bi_router(x, gate.weight)                               fp32 [T, 896]
    w, ids  = vLLM router(logits; sigmoid, bias for selection only)   [T, 16]
    z       = linear_batch_invariant(x, routed_expert_down_proj)      bf16 [T, 3584]
    leaves  = FI W4A16 experts + expert LoRA on z                      fp32 [T, 16, 3584]
    latent  = canonical-slot tree16(leaves) -> bf16, -0 -> +0
    routed  = torch.mm(rms_norm(latent), routed_expert_up_proj.weight.T)
    shared  = shared_experts(x)        (K31DenseMLP, LoRA on gate/up/down)
    out     = routed + shared          (bf16 add, ``result.add_(shared)``)

Backward: dX and LoRA dA / dB only (base frozen).  Owners: fp32 linear / router
VJP and the analytic RMSNorm VJP (skeleton primitives), the fixed-route sigmoid
VJP (``primitive.router``), the FI-expert fp32 recompute
(``vllm_fi_w4a16_moe``) and the canonical-combine all-to-all
(``primitive.moe``).

Parameter names/shapes (TP=1, EP-local experts ``E = num_experts // ep``):

* ``gate.weight`` bf16 [896, 7168], ``gate.e_score_correction_bias`` fp32 [896];
* ``routed_expert_down_proj``: ``K31Linear`` 7168 -> 3584,
  ``routed_expert_norm.weight`` bf16 [3584],
  ``routed_expert_up_proj``: ``K31Linear`` 3584 -> 7168 (no LoRA);
* ``shared_experts``: ``K31DenseMLP`` (gate_up 7168 -> 2 x 6144, down 6144 -> 7168,
  LoRA targets gate_proj / up_proj / down_proj);
* ``experts``: packed MXFP4 buffers loaded as-is (uint8 nibbles + uint8 E8M0
  scales, group 32, w13 = [w1; w3] along rows):
  ``w13_weight_packed`` [E, 6144, 1792], ``w13_weight_scale`` [E, 6144, 112],
  ``w2_weight_packed`` [E, 3584, 1536], ``w2_weight_scale`` [E, 3584, 96];
* expert LoRA (when ``experts`` is a target), peft layout of
  ``--enable-moe-shared-loras``: shared outer factors
  ``experts.lora_w1_A`` / ``experts.lora_w3_A`` [1, r, 3584] and
  ``experts.lora_w2_B`` [1, 3584, r] (replicated on every EP rank -> NOT expert
  params for gradient reduction, they reduce over every EP rank), per-expert
  ``experts.lora_w1_B`` / ``experts.lora_w3_B`` [E, 3072, r] and
  ``experts.lora_w2_A`` [E, r, 3072].
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from megatron.lite.model.k31.config import K31Config
from megatron.lite.model.k31.vllm.checkpoint import EXPERT_CLASSIFIER as is_expert_param
from megatron.lite.model.k31.vllm.model import K31DenseMLP, K31Linear, K31LoRASpec
from megatron.lite.model.k31.vllm.primitive import gate_linear, rms_norm, visible_linear
from megatron.lite.model.k31.vllm.primitive.moe import (
    _CanonicalCombine,
    dispatch_tokens,
    plan_canonical_combine,
)
from megatron.lite.model.k31.vllm.primitive.router import (
    fixed_route_vjp,
    sigmoid_route_weights,
)
from megatron.lite.model.k31.vllm.runtime_metadata import K31Runtime
from megatron.lite.model.k31.vllm.primitive.moe import _a2a
from megatron.lite.primitive.alignment.vllm_fi_w4a16_moe import (
    ExpertLoRASpec,
    VLLMFiW4A16LeafRowsFunction,
    expert_backward_rows,
)
from megatron.lite.primitive.kernels import vllm_k31_moe as K
from megatron.lite.primitive.kernels.vllm_k31 import BIRouterAdapter, RMSNormAdapter
from megatron.lite.primitive.parallel import ParallelState
from megatron.lite.primitive.recompute import active_recompute_tape


class _StoredExpertsLatent(torch.autograd.Function):
    """Recompute stand-in for ``_CanonicalCombine(VLLMFiW4A16LeafRowsFunction(...), plan)``.

    Under the structured recompute each MoE layer keeps its routed latent (the canonical-combine output,
    BF16 ``[T, latent]``, 7 KiB per token) on the tape, and the recompute skips the routed-expert forward
    (FI W4A16 + expert LoRA leaves + combine).  Forward returns the latent the original forward produced.
    Backward replays the two backward steps of the full path: ``_CanonicalCombine.backward`` (gather to
    arriving leaves, a2a back to the expert side), then the unchanged expert VJP on the routed leaf rows of
    the saved dispatch, so gradients are bit for bit those of the full recompute.
    """

    @staticmethod
    def forward(ctx, latent, experts, spec, plan, x, topk_ids, topk_weights, w1_a, w3_a, w1_b, w3_b, w2_a, w2_b):
        ctx.experts = experts.lean()
        ctx.spec = spec
        ctx.plan = plan
        ctx.save_for_backward(x, topk_ids, topk_weights, w1_a, w3_a, w1_b, w3_b, w2_a, w2_b)
        ctx.mark_non_differentiable(topk_ids)
        return latent.clone()

    @staticmethod
    def backward(ctx, grad_latent):
        plan = ctx.plan
        rows = grad_latent.index_select(0, plan.src_token)
        if plan.ep > 1:
            rows = _a2a(rows, plan.recv_counts, plan.send_counts, plan.group)
        grads = expert_backward_rows(ctx, rows.float(), plan.expert_rows)
        # (latent, experts, spec, plan, x, topk_ids, topk_weights, w1_a, w3_a, w1_b, w3_b, w2_a, w2_b)
        return (None, None, None, None, *grads[2:])


# Expert LoRA factors shared by all experts (outer A of w1/w3, outer B of w2).
SHARED_EXPERT_LORA = ("lora_w1_A", "lora_w3_A", "lora_w2_B")
PER_EXPERT_LORA = ("lora_w1_B", "lora_w3_B", "lora_w2_A")

# DeepEP normal-mode kernels (DeepEP 2.0 legacy intranode.cu) hold one thread
# per local expert: ``num_experts / num_ranks <= kNumThreads``.
DEEPEP_MAX_LOCAL_EXPERTS = 128


def default_ep_transport(num_experts: int, ep_size: int) -> str:
    """DeepEP where its kernels apply, else NCCL."""

    if ep_size <= 1:
        return "local"
    if num_experts // ep_size > DEEPEP_MAX_LOCAL_EXPERTS:
        return "nccl"
    nvl = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if ep_size > nvl:
        return "nccl"
    return "deepep"


class _Gate(nn.Module):
    def __init__(self, config: K31Config) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(config.num_experts, config.hidden_size, dtype=torch.bfloat16)
        )
        self.e_score_correction_bias = nn.Parameter(
            torch.zeros(config.num_experts, dtype=torch.float32)
        )


class _Norm(nn.Module):
    def __init__(self, size: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size, dtype=torch.bfloat16))


class K31RoutedExperts(nn.Module):
    """EP-local FI W4A16 experts (frozen MXFP4 buffers) + expert LoRA parameters."""

    def __init__(
        self,
        config: K31Config,
        *,
        local_experts: int,
        local_offset: int = 0,
        lora: K31LoRASpec | None,
    ) -> None:
        super().__init__()
        latent = config.routed_expert_hidden_size
        inter = config.moe_intermediate_size
        group = 32
        e = local_experts
        self.num_experts = config.num_experts
        self.local_experts = e
        self.local_offset = local_offset
        self.situ_beta = float(config.activation_situ_beta)
        self.situ_linear_beta = (
            None
            if config.activation_situ_linear_beta is None
            else float(config.activation_situ_linear_beta)
        )
        self.register_buffer(
            "w13_weight_packed", torch.empty(e, 2 * inter, latent // 2, dtype=torch.uint8)
        )
        self.register_buffer(
            "w13_weight_scale", torch.empty(e, 2 * inter, latent // group, dtype=torch.uint8)
        )
        self.register_buffer(
            "w2_weight_packed", torch.empty(e, latent, inter // 2, dtype=torch.uint8)
        )
        self.register_buffer(
            "w2_weight_scale", torch.empty(e, latent, inter // group, dtype=torch.uint8)
        )
        self.has_lora = lora is not None and "experts" in lora.targets
        self.lora_scaling = 1.0
        if self.has_lora:
            r = lora.rank
            self.lora_scaling = float(lora.alpha) / (r**0.5 if lora.use_rslora else float(r))
            bf16 = torch.bfloat16
            self.lora_w1_A = nn.Parameter(torch.empty(1, r, latent, dtype=bf16))
            self.lora_w3_A = nn.Parameter(torch.empty(1, r, latent, dtype=bf16))
            self.lora_w2_B = nn.Parameter(torch.zeros(1, latent, r, dtype=bf16))
            self.lora_w1_B = nn.Parameter(torch.zeros(e, inter, r, dtype=bf16))
            self.lora_w3_B = nn.Parameter(torch.zeros(e, inter, r, dtype=bf16))
            self.lora_w2_A = nn.Parameter(torch.empty(e, r, inter, dtype=bf16))
        self._fi: K.FiW4A16Experts | None = None
        self._fi_key: tuple | None = None

    def _apply(self, fn, recurse=True):
        # ``Module.to``/``cpu``/``cuda`` move the checkpoint buffers, but the FI-layout copy built
        # from them is a cached object, not a buffer: it would stay on the GPU through a param
        # offload (a second copy of the expert bytes held while the colocated rollout is awake).
        # Drop it; the next forward rebuilds it from the moved buffers, as it already does because
        # their data_ptr changes.
        self._fi = None
        self._fi_key = None
        return super()._apply(fn, recurse)

    def _buffers_key(self) -> tuple:
        return tuple(
            (b.data_ptr(), b._version)
            for b in (
                self.w13_weight_packed,
                self.w13_weight_scale,
                self.w2_weight_packed,
                self.w2_weight_scale,
            )
        )

    @property
    def fi(self) -> K.FiW4A16Experts:
        """The ``_k31_prepare_fi_w4a16`` rewrite of the loaded buffers (cached)."""

        key = self._buffers_key()
        if self._fi is None or self._fi_key != key:
            self._fi = K.FiW4A16Experts.from_checkpoint(
                self.w13_weight_packed,
                self.w13_weight_scale,
                self.w2_weight_packed,
                self.w2_weight_scale,
                num_experts=self.num_experts,
                local_offset=self.local_offset,
                situ_beta=self.situ_beta,
                situ_linear_beta=self.situ_linear_beta,
                # No resident FI copy of w13 (11 MB per expert, 57 GB per rank on the
                # full model at EP16); it is rebuilt per forward call from these buffers.
                cache_w13=False,
            )
            self._fi_key = key
        return self._fi

    def stored(self, latent: Tensor, sent, plan) -> Tensor:
        """The forward's routed latent with the autograd graph of experts + combine (recompute only)."""

        return _StoredExpertsLatent.apply(
            latent,
            self.fi,
            ExpertLoRASpec(scaling=self.lora_scaling),
            plan,
            sent.hidden,
            sent.topk_ids,
            sent.topk_weights,
            self.lora_w1_A[0],
            self.lora_w3_A[0],
            self.lora_w1_B,
            self.lora_w3_B,
            self.lora_w2_A,
            self.lora_w2_B[0],
        )

    def leaf_rows(self, sent, plan) -> Tensor:
        """FP32 leaves ``[L, H]`` of the local routed slots, in ``plan.expert_rows`` order.

        Only the routed rows are written (``L`` = local routes), not a dense ``[n, top_k, H]`` slot
        tensor of the received tokens (26 GB at EP8 with a 16k-token micro-batch)."""

        if not self.has_lora:
            # RL always serves the adapter; the no-LoRA FI branch is a different
            # rollout pipeline (fused finalize) and is deliberately not ported.
            raise NotImplementedError("K31 MoE training path requires expert LoRA")
        return VLLMFiW4A16LeafRowsFunction.apply(
            self.fi,
            ExpertLoRASpec(scaling=self.lora_scaling),
            plan.expert_rows,
            sent.hidden,
            sent.topk_ids,
            sent.topk_weights,
            self.lora_w1_A[0],
            self.lora_w3_A[0],
            self.lora_w1_B,
            self.lora_w3_B,
            self.lora_w2_A,
            self.lora_w2_B[0],
        )


class K31MoE(nn.Module):
    def __init__(
        self,
        config: K31Config,
        *,
        layer_idx: int,
        ps: ParallelState | None,
        selected_stages: frozenset[str],
        lora: K31LoRASpec | None,
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.ps = ps or ParallelState()
        self.selected_stages = selected_stages
        h = config.hidden_size
        latent = config.routed_expert_hidden_size
        if config.num_experts % self.ps.ep_size:
            raise ValueError("EP must divide the routed experts")
        if (
            config.moe_router_activation_func != "sigmoid"
            or config.num_expert_group != 1
            or config.topk_group != 1
        ):
            raise NotImplementedError("K31 MoE: sigmoid router without expert groups only")
        self.gate = _Gate(config)
        self.routed_expert_down_proj = K31Linear(h, (latent,), ("routed_expert_down_proj",))
        self.routed_expert_norm = _Norm(latent)
        self.routed_expert_up_proj = K31Linear(latent, (h,), ("routed_expert_up_proj",))
        self.shared_experts = K31DenseMLP(
            h,
            config.num_shared_experts * config.moe_intermediate_size,
            beta=config.activation_situ_beta,
            linear_beta=config.activation_situ_linear_beta,
            lora=lora,
        )
        local = config.num_experts // self.ps.ep_size
        self.experts = K31RoutedExperts(
            config, local_experts=local, local_offset=self.ps.ep_rank * local, lora=lora
        )
        self.ep_group = self.ps.ep_group if self.ps.ep_size > 1 else None
        self.transport = default_ep_transport(config.num_experts, self.ps.ep_size)
        self.router_replay = None
        self._router = None

    # ------------------------------------------------------------------ router
    def _selection(self):
        if self._router is None:
            cfg = self.config
            self._router = K.build_k31_router(
                top_k=cfg.num_experts_per_token,
                num_experts=cfg.num_experts,
                e_score_correction_bias=self.gate.e_score_correction_bias.detach(),
                renormalize=cfg.moe_renormalize,
                num_expert_group=cfg.num_expert_group,
                topk_group=cfg.topk_group,
                routed_scaling_factor=cfg.routed_scaling_factor,
            )
        return self._router

    def _replay_route(self, logits: Tensor, weights: Tensor, ids: Tensor) -> tuple[Tensor, Tensor]:
        """R3: rows whose recorded expert set equals the native one keep vLLM's
        weights (and order) bit for bit; only diverged rows are recomputed."""

        replay = self.router_replay
        if replay is None:
            return weights, ids
        selected = replay.select_indices(ids)
        if selected is ids:
            return weights, ids
        selected = selected.to(device=ids.device, dtype=ids.dtype)
        same = (selected.sort(-1).values == ids.sort(-1).values).all(-1, keepdim=True)
        if bool(same.all()):
            return weights, ids
        replayed = sigmoid_route_weights(
            logits,
            selected,
            renormalize=self.config.moe_renormalize,
            route_scale=self.config.routed_scaling_factor,
        ).to(weights.dtype)
        return torch.where(same, weights, replayed), torch.where(same, ids, selected)

    def route(self, hidden: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        logits = gate_linear(BIRouterAdapter(), hidden, self.gate.weight)
        router = self._selection()
        tape = active_recompute_tape()

        def visible(lg: Tensor) -> tuple[Tensor, Tensor]:
            if tape is not None and tape.replaying:
                # Activation recompute: the routes this layer actually used in its
                # forward (after R3 replay) and their vLLM weights, not a new top-k and
                # not a second pull from the R3 queue.
                w, ids = tape.replay()
                if ids.shape[0] != lg.shape[0]:
                    raise RuntimeError("recompute route tape does not match the layer's tokens")
                return w.clone(), ids.clone()
            w, ids = router.select_experts(hidden_states=hidden, router_logits=lg)
            w, ids = self._replay_route(lg, w, ids)
            if tape is not None:
                # Copies: dispatch / permutation may reuse the storage.
                tape.record((w.detach().clone(), ids.detach().clone()))
            return w, ids

        weights, ids = fixed_route_vjp(
            visible,
            logits,
            renormalize=self.config.moe_renormalize,
            route_scale=self.config.routed_scaling_factor,
        )
        return logits, weights, ids

    # ------------------------------------------------------------------ experts
    def routed_latent(self, latent_in: Tensor, weights: Tensor, ids: Tensor) -> Tensor:
        """Experts + canonical combine on the down-projected latent -> BF16 latent."""

        sent = dispatch_tokens(
            latent_in,
            weights,
            ids,
            group=self.ep_group,
            num_experts=self.config.num_experts,
            transport=self.transport if self.ep_group is not None else "local",
        )
        plan = plan_canonical_combine(
            ids, sent, group=self.ep_group, num_experts=self.config.num_experts
        )
        tape = active_recompute_tape()
        if tape is not None and tape.replaying:
            latent = tape.replay()
            if not isinstance(latent, torch.Tensor) or latent.shape[0] != latent_in.shape[0]:
                raise RuntimeError("recompute tape does not hold this layer's routed latent")
            return self.experts.stored(latent, sent, plan)
        out = _CanonicalCombine.apply(self.experts.leaf_rows(sent, plan), plan)
        if tape is not None:
            tape.record(out.detach())
        return out

    def forward_parts(
        self, hidden_states: Tensor, *, runtime: K31Runtime | None
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if hidden_states.ndim != 2 or hidden_states.dtype != torch.bfloat16:
            raise ValueError("K31 MoE expects BF16 [tokens, hidden]")
        logits, weights, ids = self.route(hidden_states)
        latent_in = self.routed_expert_down_proj(hidden_states, runtime)
        latent = self.routed_latent(latent_in, weights, ids)
        normed = rms_norm(
            RMSNormAdapter(), latent, self.routed_expert_norm.weight, self.config.rms_norm_eps
        )
        routed = visible_linear(K.MoELatentUpAdapter(), normed, self.routed_expert_up_proj.weight)
        shared = self.shared_experts(hidden_states, runtime)
        out = routed + shared
        return out, {
            "logits": logits,
            "weights": weights,
            "ids": ids,
            "down": latent_in,
            "latent": latent,
            "normed": normed,
            "upmm": routed,
            "shared": shared,
        }

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        metadata: Any,
        runtime: K31Runtime,
    ) -> torch.Tensor:
        del metadata  # routing replay is carried by ``router_replay``
        return self.forward_parts(hidden_states, runtime=runtime)[0]


__all__ = [
    "DEEPEP_MAX_LOCAL_EXPERTS",
    "K31MoE",
    "K31RoutedExperts",
    "PER_EXPERT_LORA",
    "SHARED_EXPERT_LORA",
    "default_ep_transport",
    "is_expert_param",
]
