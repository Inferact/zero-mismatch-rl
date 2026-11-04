# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""K31 model geometry (code name K31).

The release config nests the text model under ``text_config``.  Layer lists in
``linear_attn_config.kda_layers`` / ``dsa_config.dsa_layers`` are **one-based**
there; this dataclass stores them zero-based, which is what every consumer in
this package uses.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any

from megatron.lite.primitive.config import load_hf_config_dict


@dataclass
class K31Config:
    vocab_size: int = 163840
    hidden_size: int = 7168
    num_hidden_layers: int = 93
    rms_norm_eps: float = 1e-5
    initializer_range: float = 0.02
    tie_word_embeddings: bool = False

    # AttnRes residual bank (replaces the plain residual stream).
    attn_res_block_size: int = 12

    # Dense first layer(s) and SiTU activation.
    first_k_dense_replace: int = 1
    intermediate_size: int = 33792
    activation_situ_beta: float = 4.0
    activation_situ_linear_beta: float = 25.0

    # Latent MoE.
    num_experts: int = 896
    num_experts_per_token: int = 16
    num_shared_experts: int = 2
    moe_intermediate_size: int = 3072
    routed_expert_hidden_size: int = 3584
    latent_moe_use_norm: bool = True
    moe_layer_freq: int = 1
    moe_renormalize: bool = True
    moe_router_activation_func: str = "sigmoid"
    routed_scaling_factor: float = 1.0
    num_expert_group: int = 1
    topk_group: int = 1
    topk_method: str = "noaux_tc"

    # KDA (zero-based layer ids).
    kda_layers: list[int] = field(default_factory=list)
    kda_num_heads: int = 96
    kda_head_dim: int = 128
    kda_short_conv_kernel_size: int = 4
    kda_gate_lower_bound: float = -5.0
    kda_use_full_rank_gate: bool = True

    # DSA / MLA (zero-based layer ids).
    dsa_layers: list[int] = field(default_factory=list)
    num_attention_heads: int = 96
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    mla_use_nope: bool = True
    mla_use_output_gate: bool = True
    index_topk: int = 2048
    index_head_dim: int = 128
    index_num_heads: int = 32
    index_rope_dim: int = 0
    index_k_norm_type: str = "layernorm"
    indexer_use_fp8: bool = False

    # MTP layers of the checkpoint: neither built nor loaded (the protocol refuses mtp_enable).
    num_nextn_predict_layers: int = 0
    quantization_config: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ layout
    def is_kda_layer(self, layer_idx: int) -> bool:
        return layer_idx in self.kda_layers

    def is_dsa_layer(self, layer_idx: int) -> bool:
        return layer_idx in self.dsa_layers

    def is_moe_layer(self, layer_idx: int) -> bool:
        return (
            self.num_experts > 0
            and layer_idx >= self.first_k_dense_replace
            and layer_idx % self.moe_layer_freq == 0
        )

    def is_block_write_layer(self, layer_idx: int) -> bool:
        return layer_idx % self.attn_res_block_size == 0

    def block_write_idx(self, layer_idx: int) -> int:
        return layer_idx // self.attn_res_block_size

    def prev_valid_blocks(self, layer_idx: int) -> int:
        # vLLM: cdiv(layer_idx, attn_res_block_size)
        return -(-layer_idx // self.attn_res_block_size)

    @property
    def num_attn_res_blocks(self) -> int:
        # vLLM: cdiv(end_layer, attn_res_block_size) on the last PP rank.
        return -(-self.num_hidden_layers // self.attn_res_block_size)

    @property
    def kda_projection_size(self) -> int:
        return self.kda_num_heads * self.kda_head_dim

    @property
    def kda_in_proj_sizes(self) -> tuple[int, ...]:
        """vLLM ``in_proj_qkvgfab`` slice order: q, k, v, g, f_a, b (TP=1, no pad)."""
        p = self.kda_projection_size
        return (p, p, p, p, self.kda_head_dim, self.kda_num_heads)

    # ------------------------------------------------------------------ io
    @classmethod
    def from_hf(cls, path: str, **overrides) -> K31Config:
        return cls._from_hf_dict(load_hf_config_dict(path), **overrides)

    @classmethod
    def _from_hf_dict(cls, hf: dict[str, Any], **overrides) -> K31Config:
        text = hf.get("text_config", hf)
        for scale in (text.get("logit_scale"), hf.get("logit_scale")):
            if scale not in (None, 1, 1.0):
                raise ValueError(f"K31 training has no logit_scale; the checkpoint sets {scale}")
        names = {item.name for item in fields(cls)}
        kwargs = {k: v for k, v in text.items() if k in names and v is not None}
        if "tie_word_embeddings" in hf and "tie_word_embeddings" not in kwargs:
            kwargs["tie_word_embeddings"] = bool(hf["tie_word_embeddings"])
        linear = text.get("linear_attn_config") or {}
        if linear:
            kwargs["kda_layers"] = [int(i) - 1 for i in linear.get("kda_layers", [])]
            kwargs["kda_num_heads"] = int(linear.get("num_heads", cls.kda_num_heads))
            kwargs["kda_head_dim"] = int(linear.get("head_dim", cls.kda_head_dim))
            kwargs["kda_short_conv_kernel_size"] = int(
                linear.get("short_conv_kernel_size", cls.kda_short_conv_kernel_size)
            )
            if linear.get("gate_lower_bound") is not None:
                kwargs["kda_gate_lower_bound"] = float(linear["gate_lower_bound"])
            kwargs["kda_use_full_rank_gate"] = bool(
                linear.get("use_full_rank_gate", False)
            )
        dsa = text.get("dsa_config") or {}
        if dsa:
            kwargs["dsa_layers"] = [int(i) - 1 for i in dsa.get("dsa_layers", [])]
            for src, dst in (
                ("index_topk", "index_topk"),
                ("index_head_dim", "index_head_dim"),
                ("index_num_heads", "index_num_heads"),
                ("index_rope_dim", "index_rope_dim"),
                ("index_k_norm_type", "index_k_norm_type"),
                ("indexer_use_fp8", "indexer_use_fp8"),
            ):
                if dsa.get(src) is not None:
                    kwargs[dst] = dsa[src]
        kwargs.update(overrides)
        config = cls(**kwargs)
        config.validate()
        return config

    def validate(self) -> None:
        overlap = set(self.kda_layers) & set(self.dsa_layers)
        if overlap:
            raise ValueError(f"layers are both KDA and DSA: {sorted(overlap)}")
        covered = set(self.kda_layers) | set(self.dsa_layers)
        missing = [i for i in range(self.num_hidden_layers) if i not in covered]
        if missing:
            raise ValueError(f"layers are neither KDA nor DSA: {missing}")
        if any(i < 0 or i >= self.num_hidden_layers for i in covered):
            raise ValueError("KDA/DSA layer ids must be one-based in the HF config")


__all__ = ["K31Config"]
