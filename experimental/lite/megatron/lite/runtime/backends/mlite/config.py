# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Megatron Lite backend configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from megatron.lite.runtime.contracts.config import (
    OptimizerConfig,
    ParallelConfig,
    pick_fields,
    refuse_retired_env,
)

# Environment switches Megatron Lite no longer reads (any model), and what replaced each one.
RETIRED_ENV = {
    "MLITE_OFFLOAD_PERSIST": "MegatronLiteConfig.offload_persist "
    "(verl: actor_rollout_ref.actor.engine.offload_persist)",
    "MLITE_OFFLOAD_PERSIST_PIN": "MegatronLiteConfig.offload_persist_pin "
    "(verl: actor_rollout_ref.actor.engine.offload_persist_pin)",
    "MLITE_OFFLOAD_PERSIST_VERIFY": "MegatronLiteConfig.offload_persist_verify "
    "(verl: actor_rollout_ref.actor.engine.offload_persist_verify)",
    "MLITE_CKPT_TRAINABLE_ONLY": "MegatronLiteConfig.ckpt_trainable_only "
    "(verl: actor_rollout_ref.actor.engine.ckpt_trainable_only)",
    "MLITE_LORA_EXPERT_SLICE": "always on where the model protocol sets SUPPORTS_LORA_EXPERT_SLICE",
    "MLITE_ACTOR_TIMING": "removed (offload_persist_verify still reports digest mismatches)",
}


@dataclass(slots=True)
class DebugConfig:
    """Megatron Lite backend debug flags. Not exposed to end users."""

    param_update: bool = False
    optimizer_state: bool = False
    grad_phases: bool = False
    router_summary: bool = False
    moe_io: bool = False
    attn_io: bool = False


@dataclass
class MegatronLiteConfig:
    """Config for MegatronLiteRuntime (Megatron Lite's default 5D parallel runtime).

    Megatron Lite-specific training features live in ``impl_cfg`` (a plain dict —
    each impl reads the keys it needs via its own typed ImplConfig).
    """

    # ── identity ──
    model_name: str = "auto"
    impl: str = "lite"
    hf_path: str = ""

    # ── parallelism and optimizer ──
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)

    # ── common runtime/model fields ──
    attention_backend_override: str | None = "flash"
    router_aux_loss_coef: float | None = None
    load_hf_weights: bool = True

    # ── offload and checkpoint ──
    # Module offload keeps one host copy per parameter / buffer / gradient across offloads
    # (frozen ones copied once) instead of fresh host memory per offload; byte-exact either way.
    offload_persist: bool = False
    # Pin the host copies of trainable tensors (``offload_persist``).
    offload_persist_pin: bool = True
    # Re-digest skipped frozen tensors on every offload (diagnostic, ``offload_persist``).
    offload_persist_verify: bool = False
    # Rank-local (node-local disk) checkpoints hold only parameters that require grad; frozen
    # ones are rebuilt from the base checkpoint, so this needs ``load_hf_weights``.  Recorded in
    # the layout: a checkpoint loads only with the setting it was saved with.
    ckpt_trainable_only: bool = False

    # ── impl-specific (each impl reads its own keys) ──
    impl_cfg: dict[str, Any] = field(default_factory=dict)

    # ── debug ──
    debug: DebugConfig = field(default_factory=DebugConfig)

    # ── bench-only hook: mutate model_cfg after build (e.g. expert truncation) ──
    model_config_hook: Any = None

    def __post_init__(self) -> None:
        refuse_retired_env(RETIRED_ENV, "Megatron Lite")
        if self.ckpt_trainable_only and not self.load_hf_weights:
            raise ValueError(
                "ckpt_trainable_only=True needs load_hf_weights=True: a resume restores only the "
                "trainable parameters, the frozen ones would stay at their random init"
            )

    @classmethod
    def from_dict(cls, hf_path: str, cfg: dict[str, Any]) -> MegatronLiteConfig:
        """Construct MegatronLiteConfig from a flat dict (legacy / OmegaConf path)."""
        if "num_microbatches" in cfg:
            raise ValueError(
                "MegatronLiteConfig no longer accepts `num_microbatches`; "
                "pass it to Runtime.forward_backward(..., num_microbatches=...) instead"
            )
        parallel = ParallelConfig(**pick_fields(ParallelConfig, cfg))

        opt_d = cfg.get("optimizer", {})
        optimizer = (
            OptimizerConfig(**pick_fields(OptimizerConfig, opt_d))
            if isinstance(opt_d, dict)
            else OptimizerConfig()
        )
        if isinstance(opt_d, dict) and isinstance(opt_d.get("override_optimizer_config"), dict):
            optimizer.override_optimizer_config = dict(opt_d["override_optimizer_config"])

        # impl_cfg: merge nested dict + top-level overrides
        impl_cfg: dict[str, Any] = {}
        nested = cfg.get("impl_cfg")
        if isinstance(nested, dict):
            impl_cfg.update(nested)
        for k in list(impl_cfg):
            if k in cfg:
                impl_cfg[k] = cfg[k]
        for k in ("recompute", "use_thd", "use_deepep", "precision_aware_opt"):
            if k in cfg and k not in impl_cfg:
                impl_cfg[k] = cfg[k]

        skip = {"parallel", "optimizer", "impl_cfg", "debug"}
        return cls(
            **{k: v for k, v in pick_fields(cls, cfg).items() if k not in skip},
            hf_path=hf_path,
            parallel=parallel,
            optimizer=optimizer,
            impl_cfg=impl_cfg,
        )


__all__ = ["MegatronLiteConfig", "DebugConfig", "RETIRED_ENV"]
