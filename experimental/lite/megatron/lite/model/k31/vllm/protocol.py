"""Training-capable protocol for the K31 vLLM implementation (DS4 vLLM skeleton)."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from megatron.lite.model.k31.config import K31Config
from megatron.lite.model.k31.vllm.checkpoint import (
    EXPERT_CLASSIFIER,
    PLACEMENT_FN,
    SUPPORTS_LORA_EXPERT_SLICE,
    export_hf_weights,
    load_hf_weights,
    load_lora_adapter,
    lora_adapter_config,
    save_hf_weights,
)
from megatron.lite.model.k31.vllm.runtime_metadata import (
    build_k31_runtime,
    initialize_k31_vllm_batch_invariance,
    k31_vllm_forward_context,
)
from megatron.lite.model.protocol_utils import (
    add_loss_context_kwargs,
    nested_from_packed,
    pack_r3_replay_mask as _pack_r3_replay_mask,
    pack_routed_experts as _pack_routed_experts,
    router_replay_roots as router_replay_roots,
)
from megatron.lite.primitive.bundle import ModelBundle
from megatron.lite.primitive.modules.lora import LoraConfig, freeze_non_lora_params
from megatron.lite.primitive.parallel import init_parallel
from megatron.lite.primitive.recompute import apply_recompute, parse_recompute_spec
from megatron.lite.runtime.contracts import OptimizerConfig, ParallelConfig

_CANONICAL_STAGES = frozenset({"attn_res", "linear", "kda", "dsa", "router_moe", "deepep"})
_ALIASES = {
    "attn": ("attn_res", "linear", "kda", "dsa"),
    "moe": ("router_moe", "deepep"),
}
_SELECTABLE_MODULES = _CANONICAL_STAGES | _ALIASES.keys()


def _attn_of(kind: str):
    def pick(layer):
        is_kda = layer.config.is_kda_layer(layer.layer_idx)
        return layer.self_attn if kind == "attn" or (kind == "kda") == is_kda else None

    return pick


# DS4 vLLM pattern (``primitive.recompute`` + module map), on the structured
# checkpoint: "full" recomputes the whole decoder layer and keeps only its input
# stream (prefix, pending delta, bank references).  The other entries checkpoint
# one sub-module.  The router is not separately recomputable: its decision is part
# of the MoE checkpoint's replay tape.
_RECOMPUTE_MODULE_MAP = {
    "attn": _attn_of("attn"),
    "kda": _attn_of("kda"),
    "dsa": _attn_of("dsa"),
    "moe": lambda layer: layer.mlp,
    "mlp": lambda layer: layer.mlp,
}
_RECOMPUTE_NO_RNG = frozenset({"full", *_RECOMPUTE_MODULE_MAP})


def _recompute_layers(layers: list, ps, counts: tuple[int, ...]) -> list:
    """``ImplConfig.recompute_num_layers``: Megatron's ``recompute_method=block`` on this
    stage -- only the first N local decoder layers are checkpointed, the rest keep
    their activations.  ``(N,)`` (every stage) or ``(n0, n1, ...)`` (one count per pipeline
    rank: in 1F1B stage ``s`` holds ``pp - s`` micro-batches, so later stages can keep
    more).  Empty: every layer (the default).  The kept layers run the same forward
    and backward kernels on the same operands; only the recompute pass is skipped."""

    if not counts:
        return layers
    pp_rank = int(getattr(ps, "pp_rank", 0) or 0)
    if len(counts) == 1:
        n = counts[0]
    else:
        pp_size = int(getattr(ps, "pp_size", 1) or 1)
        if len(counts) != pp_size:
            raise ValueError(
                f"recompute_num_layers={list(counts)} has {len(counts)} counts for pp={pp_size}"
            )
        n = counts[pp_rank]
    n = max(0, min(n, len(layers)))
    return layers[:n]


# The LoRA target modules the rollout serves.
DEPLOY_LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "g_proj",
    "f_a_proj",
    "b_proj",
    "f_b_proj",
    "o_proj",
    "q_a_proj",
    "kv_a_proj_with_mqa",
    "q_b_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "experts",
)


@dataclass(frozen=True)
class SelectorConfig:
    """Stable selector for global decoder layer IDs and module kinds."""

    global_layer_ids: tuple[int, ...] = ()
    module_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        layer_ids = tuple(self.global_layer_ids)
        requested = tuple(self.module_names)
        if any(type(i) is not int or i < 0 for i in layer_ids):
            raise ValueError("global_layer_ids must contain non-negative integers.")
        if len(set(layer_ids)) != len(layer_ids):
            raise ValueError("global_layer_ids must not contain duplicates.")
        if any(type(name) is not str for name in requested):
            raise TypeError("module_names must contain strings.")
        unknown = set(requested) - _SELECTABLE_MODULES
        if unknown:
            raise ValueError(f"Unknown selector module names: {sorted(unknown)}")
        if len(set(requested)) != len(requested):
            raise ValueError("module_names must not contain duplicates.")
        names: list[str] = []
        for name in requested:
            for stage in _ALIASES.get(name, (name,)):
                if stage not in names:
                    names.append(stage)
        object.__setattr__(self, "global_layer_ids", layer_ids)
        object.__setattr__(self, "module_names", tuple(names))

    def selects(self, global_layer_id: int, module_name: str) -> bool:
        return global_layer_id in self.global_layer_ids and module_name in self.module_names


@dataclass(frozen=True)
class ImplConfig:
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    optimizer: Any | None = None
    optimizer_config: OptimizerConfig | None = None
    hf_path: str = ""
    selector: SelectorConfig = field(default_factory=SelectorConfig)
    recompute: tuple[str, ...] = ()
    # Block recompute: checkpoint only the first N local decoder layers (see ``_recompute_layers``).
    recompute_num_layers: tuple[int, ...] = ()
    offload: tuple[str, ...] = ()
    use_thd: bool = False
    use_deepep: bool = False
    attention_backend_override: str | None = None
    deterministic: bool = True
    qat: Any | None = None
    mtp_enable: bool = False
    max_tokens_per_rank: int = 8192
    # K31 trains LoRA only; must match the rollout's --max-lora-rank and adapter.
    lora: LoraConfig = field(default_factory=LoraConfig)
    max_lora_rank: int = 0
    lora_use_rslora: bool = False
    lora_adapter_path: str = ""
    # The rollout's ``--max-model-len``: DSA builds its FlashMLA ``cache_seqlens``
    # and indexer chunk limits from it, so it is part of the bitwise contract.
    rollout_max_model_len: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.selector, Mapping):
            object.__setattr__(self, "selector", SelectorConfig(**dict(self.selector)))
        counts = self.recompute_num_layers
        counts = (counts,) if isinstance(counts, int) else tuple(counts)
        if any(type(n) is not int or n < 0 for n in counts):
            raise ValueError(f"recompute_num_layers must be non-negative integers, got {counts}")
        object.__setattr__(self, "recompute_num_layers", counts)
        if isinstance(self.lora, Mapping):
            from megatron.lite.primitive.modules.lora import normalize_lora_config

            object.__setattr__(self, "lora", normalize_lora_config(dict(self.lora)))


def is_expert_param(name: str) -> bool:
    return EXPERT_CLASSIFIER(name)


def _finalize_replica_grads(model: nn.Module, parallel_state) -> None:
    """Sum gradients over each parameter's replicas, then divide by the DP size.

    Every rank's loss is its share of the global objective ``(1/DP) sum_r L_r``.
    A dense (replicated) parameter sees only ``dL_r`` locally; an expert
    parameter already sees ``sum`` over every EP rank's tokens (the combine
    backward returns them), so after summing over its expert-DP replicas it
    holds ``sum_r dL_r`` exactly like the dense sum over DP.  Both therefore
    divide by the full DP size -- the Megatron-Core DDP convention
    (``expert_gradient_scaling_factor = 1 / dp``) and what mLite FSDP2 applies
    (``expert_dp / dp_cp`` on top of its expert-DP average).  Dividing experts
    by ``expert_dp_size`` instead would scale their effective LR by EP.
    The shared outer expert-LoRA factors are dense-class (replicated on every
    EP rank, each holding a partial sum over its local experts), so their sum
    over the DP group is the full gradient.
    """
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        return
    divisor = parallel_state.dp_cp_size
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        expert = is_expert_param(name)
        group = parallel_state.ep_dp_group if expert else parallel_state.dp_group
        size = parallel_state.expert_dp_size if expert else parallel_state.dp_size
        if group is not None and size > 1:
            dist.all_reduce(parameter.grad, group=group)
        if divisor > 1:
            parameter.grad.div_(divisor)


class _K31FP32AdamW:
    """mLite ``FP32AdamW`` (FP32 masters) plus global-norm clipping for K31's layout.

    K31's trainable parameters are BF16 LoRA factors; stepping them in BF16 drops
    updates below BF16 resolution (lr 1e-6 on O(1e-2) factors).  The masters live
    in FP32 and are rounded once into the BF16 parameter, exactly as the FSDP2
    path does.  Clipping uses the global L2 norm after ``_finalize_replica_grads``:
    dense-class gradients are identical on every rank and count once; expert
    gradients are sharded over the EP group and their squared sums add over it;
    with PP each stage's total (its own layers) then adds over the PP group.
    The clip coefficient multiplies the FP32 copy of each gradient inside the
    update (as Megatron clips its FP32 main grads); scaling the BF16 ``.grad`` in
    place would round every gradient once more before AdamW.
    ``step`` returns ``(ok, grad_norm, None)``, the Megatron optimizer contract the
    runtime already reads; a non-finite norm skips the update.
    """

    def __init__(self, named_params, config, parallel_state):
        from megatron.lite.primitive.optimizers.fsdp2.adamw import FP32AdamW

        class _ScaledFP32AdamW(FP32AdamW):
            grad_scale = 1.0

            def _prepare_grad(self, grad, master):
                grad = super()._prepare_grad(grad, master)
                return grad if self.grad_scale == 1.0 else grad.mul(self.grad_scale)

        self._named = list(named_params)
        self._expert = [p for n, p in self._named if is_expert_param(n)]
        self._dense = [p for n, p in self._named if not is_expert_param(n)]
        self._clip = float(config.clip_grad or 0.0)
        self._ep_group = parallel_state.ep_group if parallel_state.ep_size > 1 else None
        # Pipeline stages own disjoint layers: their squared sums add over the PP group.
        self._pp_group = parallel_state.pp_group if parallel_state.pp_size > 1 else None
        self._inner = _ScaledFP32AdamW(
            [p for _, p in self._named],
            lr=config.lr,
            weight_decay=config.weight_decay,
            betas=(config.adam_beta1 or 0.9, config.adam_beta2 or 0.999),
            eps=config.adam_eps or 1e-8,
        )

    @property
    def param_groups(self):
        return self._inner.param_groups

    @property
    def state(self):
        return self._inner.state

    def zero_grad(self, *args, **kwargs) -> None:
        self._inner.zero_grad(*args, **kwargs)

    def reload_model_params(self) -> None:
        self._inner.reload_model_params()

    def state_dict(self):
        return self._inner.state_dict()

    def load_state_dict(self, state_dict) -> None:
        self._inner.load_state_dict(state_dict)

    @torch.no_grad()
    def grad_norm(self) -> torch.Tensor:
        from megatron.lite.primitive.optimizers.fsdp2.grad_clip import (
            all_reduce_scalar_,
            sharded_grad_sq_sum,
        )

        device = torch.device("cuda", torch.cuda.current_device())
        dense = sharded_grad_sq_sum(self._dense, accum_dtype=torch.float64, default_device=device)
        expert = sharded_grad_sq_sum(self._expert, accum_dtype=torch.float64, default_device=device)
        expert = expert.to(device)
        if self._ep_group is not None:
            all_reduce_scalar_(expert, op=torch.distributed.ReduceOp.SUM, group=self._ep_group)
        total = dense.to(device) + expert
        if self._pp_group is not None:
            all_reduce_scalar_(total, op=torch.distributed.ReduceOp.SUM, group=self._pp_group)
        return total.sqrt()

    @torch.no_grad()
    def step(self):
        norm = float(self.grad_norm().item())
        if not math.isfinite(norm):
            return False, norm, None
        # Same coefficient as Megatron / mLite clip_grads_with_sharded_norm_.
        coef = self._clip / (norm + 1.0e-6) if self._clip > 0 else 1.0
        self._inner.grad_scale = min(coef, 1.0)
        self._inner.step()
        return True, norm, None


def select_router_layers(model_cfg: K31Config, routed: torch.Tensor) -> torch.Tensor:
    """Keep the MoE-layer columns of rollout routes indexed by decoder layer.

    vLLM's routed-experts capturer allocates one column per decoder layer
    (``get_total_num_hidden_layers``); the dense first layer(s) have no router
    and their column is filler.  The trainer has one ``RouterReplay`` per MoE
    layer, in layer order.
    """

    width = (
        int(routed.values().size(1))
        if getattr(routed, "is_nested", False)
        else int(routed.size(-2))
    )
    moe_layers = [i for i in range(model_cfg.num_hidden_layers) if model_cfg.is_moe_layer(i)]
    if width == len(moe_layers):
        return routed
    if width != model_cfg.num_hidden_layers:
        raise ValueError(
            f"R3 routes have {width} layer columns; expected {model_cfg.num_hidden_layers} "
            f"(per decoder layer) or {len(moe_layers)} (per MoE layer)"
        )
    index = torch.tensor(moe_layers, dtype=torch.long)
    if getattr(routed, "is_nested", False):
        rows = [row.index_select(1, index.to(row.device)) for row in routed.unbind(0)]
        return torch.nested.as_nested_tensor(rows, layout=torch.jagged)
    return routed.index_select(-2, index.to(routed.device))


def build_model_config(source: str | Path | dict, **overrides) -> K31Config:
    if isinstance(source, dict):
        config = K31Config._from_hf_dict(source)
    else:
        config = K31Config.from_hf(str(source))
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config


def lora_spec_from_impl(impl_cfg: ImplConfig):
    from megatron.lite.model.k31.vllm.model import K31LoRASpec

    lora = impl_cfg.lora
    if not lora.enabled:
        return None
    return K31LoRASpec(
        rank=lora.rank,
        alpha=float(lora.rank if lora.alpha is None else lora.alpha),
        max_rank=int(impl_cfg.max_lora_rank or lora.rank),
        targets=frozenset(lora.target_modules),
        use_rslora=impl_cfg.lora_use_rslora,
    )


def _validate_contract(model_cfg: K31Config, impl_cfg: ImplConfig) -> None:
    if impl_cfg.max_tokens_per_rank <= 0:
        raise ValueError("max_tokens_per_rank must be positive")
    parallel = impl_cfg.parallel
    dimensions = {
        "tp": parallel.tp,
        "etp": 1 if parallel.etp is None else parallel.etp,
        "vpp": parallel.vpp,
        "cp": parallel.cp,
    }
    unsupported = {name: size for name, size in dimensions.items() if size > 1}
    if unsupported:
        values = ", ".join(f"{name}={size}" for name, size in unsupported.items())
        raise NotImplementedError(
            f"K31 vLLM keeps TP/ETP/VPP/CP at one (KDA CF-BI has no CP); got {values}."
        )
    if parallel.ep <= 0 or model_cfg.num_experts % parallel.ep:
        raise ValueError(f"EP={parallel.ep} must divide {model_cfg.num_experts} routed experts.")
    if not impl_cfg.use_deepep:
        raise NotImplementedError("K31 vLLM requires normal DeepEP training transport.")
    if impl_cfg.mtp_enable:
        raise NotImplementedError("K31 vLLM does not support MTP.")
    has_dsa = any(
        not model_cfg.is_kda_layer(i) for i in range(model_cfg.num_hidden_layers)
    )
    if has_dsa and impl_cfg.rollout_max_model_len <= 0:
        raise ValueError(
            "K31 DSA metadata needs the rollout's --max-model-len (impl_cfg.rollout_max_model_len)"
        )
    optimizer_name = (
        impl_cfg.optimizer
        if isinstance(impl_cfg.optimizer, str)
        else getattr(impl_cfg.optimizer_config, "optimizer", None)
    )
    if impl_cfg.optimizer is not None and optimizer_name == "fsdp2" and parallel.ep > 1:
        # FSDP2 shards whole ``experts`` modules on the expert-DP mesh; K31's
        # experts module also holds the shared outer expert-LoRA factors, whose
        # gradient must be summed over every EP rank.  Use adamw/sgd, whose
        # finalize step reduces per parameter class.
        raise NotImplementedError("K31 vLLM with EP>1 does not support the fsdp2 optimizer")
    if not impl_cfg.lora.enabled:
        raise NotImplementedError("K31 vLLM trains LoRA only; a LoRA config is required.")
    if impl_cfg.lora.dropout:
        raise NotImplementedError("LoRA dropout is not part of the rollout contract.")
    if impl_cfg.max_lora_rank and impl_cfg.max_lora_rank < impl_cfg.lora.rank:
        raise ValueError("max_lora_rank must be >= the adapter rank")
    disabled = {
        "offload": impl_cfg.offload,
        "use_thd": impl_cfg.use_thd,
        "attention_backend_override": impl_cfg.attention_backend_override,
        "qat": impl_cfg.qat,
    }
    enabled = [name for name, value in disabled.items() if value]
    if enabled:
        raise NotImplementedError(
            "K31 vLLM does not install training/runtime features: " + ", ".join(enabled)
        )
    invalid = [i for i in impl_cfg.selector.global_layer_ids if i >= model_cfg.num_hidden_layers]
    if invalid:
        raise ValueError(f"Selector layer IDs are outside the model: {invalid}")
    selected = impl_cfg.selector.global_layer_ids
    if selected and selected != tuple(range(max(selected) + 1)):
        raise ValueError(
            "K31 vLLM selectors must form a contiguous global-layer prefix starting at zero; "
            f"got {selected}."
        )
    if 0 in selected:
        required = {"attn_res", "linear"} | {
            "kda" if model_cfg.is_kda_layer(0) else "dsa"
        }
        missing = required - set(impl_cfg.selector.module_names)
        if missing:
            raise ValueError(
                "layer-0 candidate audit requires every execution stage; missing "
                + ", ".join(sorted(missing))
            )


def _roll_packed_targets(
    values: torch.Tensor | None,
    seq_lens: torch.Tensor | None,
) -> torch.Tensor | None:
    if values is None or seq_lens is None:
        return values
    rolled = values.clone()
    offset = 0
    for length_value in seq_lens.detach().cpu().tolist():
        length = int(length_value)
        if length > 0:
            rolled[offset : offset + length - 1] = values[offset + 1 : offset + length]
            rolled[offset + length - 1] = 0
        offset += length
    if offset != values.numel():
        raise ValueError(
            f"packed sequence lengths sum to {offset}, but targets contain {values.numel()} values"
        )
    return rolled


def _forward_step(
    model: nn.Module,
    batch,
    *,
    attention_metadata=None,
    moe_metadata=None,
    runtime=None,
    forward_inputs: Mapping[str, Any] | None = None,
) -> dict[str, torch.Tensor]:
    seq_lens = getattr(batch, "seq_lens", None)
    kwargs = {
        "input_ids": batch.input_ids,
        "position_ids": getattr(batch, "position_ids", None),
        "attention_metadata": (
            getattr(batch, "attention_metadata", None)
            if attention_metadata is None
            else attention_metadata
        ),
        "moe_metadata": (
            getattr(batch, "moe_metadata", None) if moe_metadata is None else moe_metadata
        ),
        "runtime": runtime,
        "labels": _roll_packed_targets(getattr(batch, "labels", None), seq_lens),
        "loss_mask": _roll_packed_targets(getattr(batch, "loss_mask", None), seq_lens),
        "temperature": getattr(batch, "temperature", 1.0),
    }
    if forward_inputs is not None:
        kwargs.update(forward_inputs)
    add_loss_context_kwargs(kwargs)
    return model(**kwargs)


def build_attention_metadata(
    model_cfg: K31Config,
    runtime,
    *,
    layer_ids,
    rollout_max_model_len: int,
) -> dict[int, Any]:
    """Per-microbatch attention metadata keyed by global layer id.

    DSA: one ``build_dsa_metadata`` object shared by every DSA layer (each call
    rewrites its private caches before reading them).  KDA: ``build_kda_metadata``
    (``None``; sequence boundaries come from ``runtime.cu_seqlens`` and the CF-BI
    state pool, slots 1..R in the rollout conv-state layout, is per call).
    MoE needs none: R3 routes travel through ``K31MoE.router_replay``.
    """

    from megatron.lite.model.k31.vllm.dsa import build_dsa_metadata
    from megatron.lite.model.k31.vllm.kda import build_kda_metadata

    layer_ids = tuple(layer_ids) or tuple(range(model_cfg.num_hidden_layers))
    kda = build_kda_metadata(runtime)
    dsa = None
    metadata: dict[int, Any] = {}
    for layer_idx in layer_ids:
        if model_cfg.is_kda_layer(layer_idx):
            metadata[layer_idx] = kda
            continue
        if dsa is None:
            dsa = build_dsa_metadata(runtime, max_model_len=rollout_max_model_len)
        metadata[layer_idx] = dsa
    return metadata


_ROLLOUT_SAMPLING_DEFAULTS = {
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": -1,
    "min_p": 0.0,
    "repetition_penalty": 1.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
}


def validate_rollout_sampling(rollout: Any) -> None:
    """Refuse rollout sampling settings whose logprobs the trainer does not reproduce.

    The trainer recomputes the rollout logprob from the raw logits (temperature 1) or, in vLLM's
    processed-logprobs mode, from the FP32 logits after the sampler's temperature kernel (vLLM
    raises a temperature below 0.01 to 0.01).  vLLM's sampler applies min_p, top-k/top-p
    and penalties before it computes processed logprobs, and moves the whole batch to FP32
    as soon as one request needs any of them.  Greedy training rollouts (``do_sample=False``)
    and ``logit_bias`` change the sampled distribution the same way.  Validation sampling
    (``val_kwargs``, e.g. greedy) is not checked: validation logprobs never enter the loss.
    """

    def raw(key: str) -> Any:
        if isinstance(rollout, Mapping):
            return rollout.get(key)
        return getattr(rollout, key, None)

    def value(key: str) -> Any:
        got = raw(key)
        return _ROLLOUT_SAMPLING_DEFAULTS[key] if got is None else got

    bad = {
        key: value(key)
        for key, default in _ROLLOUT_SAMPLING_DEFAULTS.items()
        if key not in ("temperature", "top_k") and float(value(key)) != default
    }
    if float(value("temperature")) < 0.01:
        bad["temperature"] = value("temperature")
    if float(value("temperature")) != 1.0 and raw("logprobs_mode") not in (None, "processed_logprobs"):
        bad["logprobs_mode"] = raw("logprobs_mode")
    if int(value("top_k")) not in (-1, 0):
        bad["top_k"] = value("top_k")
    if raw("do_sample") is not None and not bool(raw("do_sample")):
        bad["do_sample"] = raw("do_sample")
    if raw("logit_bias"):
        bad["logit_bias"] = raw("logit_bias")
    if bad:
        raise ValueError(
            "K31 training reproduces only temperature-scaled rollout logprobs; unsupported "
            "rollout sampling settings: "
            + ", ".join(f"{k}={v!r}" for k, v in sorted(bad.items()))
        )


def unpack_forward_output(model: nn.Module, batch, output) -> Any:
    return nested_from_packed(output, batch.seq_lens)


def pack_routed_experts(model: nn.Module, batch, routed_experts):
    """Pack rollout routes in the contiguous token order used by K31 vLLM."""

    return _pack_routed_experts(model, batch, routed_experts, contiguous=True)


def pack_r3_replay_mask(model: nn.Module, batch) -> torch.Tensor:
    return _pack_r3_replay_mask(model, batch, contiguous=True)


def build_model(model_cfg: K31Config, *, impl_cfg: ImplConfig) -> ModelBundle:
    _validate_contract(model_cfg, impl_cfg)
    initialize_k31_vllm_batch_invariance()
    from megatron.lite.model.k31.vllm.model import K31Layer, K31Model

    parallel_state = init_parallel(impl_cfg.parallel)
    lora = lora_spec_from_impl(impl_cfg)
    model = K31Model(
        model_cfg,
        ps=parallel_state,
        selected_layer_ids=impl_cfg.selector.global_layer_ids,
        selected_module_names=impl_cfg.selector.module_names,
        lora=lora,
    )
    freeze_non_lora_params(model)
    recompute_spec = parse_recompute_spec(impl_cfg.recompute)
    if recompute_spec:
        from megatron.lite.model.k31.vllm.model import capture_forward_context

        # K31 has no randomness in its forward; the structured checkpoint replays the
        # MoE routes recorded in the forward (never a new top-k or a second R3 pull)
        # and re-installs the forward's vLLM forward context.
        apply_recompute(
            _recompute_layers(
                list(model.layers.values()), parallel_state, impl_cfg.recompute_num_layers
            ),
            recompute_spec,
            _RECOMPUTE_MODULE_MAP,
            no_rng_modules=set(_RECOMPUTE_NO_RNG),
            structured=True,
            context_fn=capture_forward_context,
            strict=True,
        )
    if torch.cuda.is_available():
        model = model.cuda()
        from vllm.v1.worker.workspace import (
            init_workspace_manager,
            is_workspace_manager_initialized,
        )

        if not is_workspace_manager_initialized():
            init_workspace_manager(next(model.parameters()).device, num_ubatches=1)
    from vllm.config import VllmConfig

    vllm_config = VllmConfig()

    def forward_step(model: nn.Module, batch) -> dict[str, torch.Tensor]:
        seq_lens = getattr(batch, "seq_lens", None)
        lengths = (
            [int(batch.input_ids.numel())]
            if seq_lens is None
            else [int(n) for n in seq_lens.detach().cpu().tolist()]
        )
        if sum(lengths) > impl_cfg.max_tokens_per_rank:
            raise ValueError(
                f"packed batch has {sum(lengths)} tokens, exceeding "
                f"max_tokens_per_rank={impl_cfg.max_tokens_per_rank}"
            )
        runtime = build_k31_runtime(
            lengths, batch.input_ids.device, with_lora=lora is not None
        )
        # This pipeline stage's layers only (all selected layers when PP=1).
        attention_metadata = build_attention_metadata(
            model_cfg,
            runtime,
            layer_ids=getattr(model, "selected_layer_ids", None)
            or impl_cfg.selector.global_layer_ids,
            rollout_max_model_len=impl_cfg.rollout_max_model_len,
        )
        with k31_vllm_forward_context(batch, parallel_state, vllm_config=vllm_config):
            return _forward_step(
                model, batch, runtime=runtime, attention_metadata=attention_metadata
            )

    optimizer = None
    finalize_grads = None
    optimizer_backend = "none"
    post_model_load_hook = None
    if impl_cfg.optimizer is not None:
        config = impl_cfg.optimizer_config or OptimizerConfig()
        name = impl_cfg.optimizer if isinstance(impl_cfg.optimizer, str) else config.optimizer
        if name not in ("adam", "adamw", "sgd", "fsdp2"):
            raise ValueError(f"K31 vLLM training supports adam/adamw/sgd/fsdp2, got {name!r}")
        trainable = [p for p in model.parameters() if p.requires_grad]
        if name == "fsdp2":
            optimizer_backend = "fsdp2"

            def _post_model_load_hook():
                from megatron.lite.primitive.optimizers.fsdp2 import (
                    build_fsdp2_training_optimizer,
                )

                return {
                    "optimizer": build_fsdp2_training_optimizer(
                        [model],
                        config,
                        parallel_state,
                        unit_modules=(K31Layer,),
                        expert_classifier=is_expert_param,
                        deterministic=impl_cfg.deterministic,
                        vpp=impl_cfg.parallel.vpp,
                        leaf_module_names=(),
                        use_fp32_shards=True,
                        cast_forward_inputs=False,
                    )
                }

            post_model_load_hook = _post_model_load_hook
        elif name == "sgd":
            optimizer = torch.optim.SGD(
                trainable, lr=config.lr, weight_decay=config.weight_decay, foreach=False
            )
            optimizer_backend = "sgd"
        else:
            optimizer = _K31FP32AdamW(
                [(n, p) for n, p in model.named_parameters() if p.requires_grad],
                config,
                parallel_state,
            )
            optimizer_backend = "adamw"
        if name != "fsdp2":

            def finalize_grads() -> None:
                _finalize_replica_grads(model, parallel_state)

    extras = {
        "model_cfg": model_cfg,
        "optimizer_backend": optimizer_backend,
        "selector": impl_cfg.selector,
        "lora_spec": lora,
    }
    if impl_cfg.lora_adapter_path:
        adapter = impl_cfg.lora_adapter_path
        previous = post_model_load_hook

        def _load_adapter_hook():
            load_lora_adapter(model, adapter, model_cfg, parallel_state)
            # FP32 masters were taken at build time; refresh them from the adapter.
            reload = getattr(optimizer, "reload_model_params", None)
            if callable(reload):
                reload()
            return previous() if previous is not None else {}

        post_model_load_hook = _load_adapter_hook
    if torch.cuda.is_available():
        from vllm.v1.worker.workspace import reset_workspace_manager

        extras["close_hook"] = reset_workspace_manager
    if post_model_load_hook is not None:
        extras["post_model_load_hook"] = post_model_load_hook
    return ModelBundle(
        chunks=[model],
        parallel_state=parallel_state,
        optimizer=optimizer,
        finalize_grads=finalize_grads,
        forward_step=forward_step,
        extras=extras,
    )


def export_peft_config(chunks) -> dict | None:
    """peft ``adapter_config`` of the weight-sync payload.

    ``export_hf_weights`` yields only the LoRA adapter (the base is frozen and the
    rollout loads it from the checkpoint), so the RL trainer must hand the payload to
    the rollout as an adapter, not as base weights.
    """

    first = chunks[0] if isinstance(chunks, (list, tuple)) else chunks
    lora = getattr(first, "lora_spec", None)
    if lora is None:
        return None
    return lora_adapter_config(lora, lora.targets)


def vocab_size(model_cfg: K31Config) -> int | None:
    return model_cfg.vocab_size


__all__ = [
    "DEPLOY_LORA_TARGETS",
    "EXPERT_CLASSIFIER",
    "ImplConfig",
    "PLACEMENT_FN",
    "SUPPORTS_LORA_EXPERT_SLICE",
    "SelectorConfig",
    "build_attention_metadata",
    "build_model",
    "build_model_config",
    "export_hf_weights",
    "export_peft_config",
    "is_expert_param",
    "load_hf_weights",
    "save_hf_weights",
    "select_router_layers",
    "unpack_forward_output",
    "validate_rollout_sampling",
    "vocab_size",
]
