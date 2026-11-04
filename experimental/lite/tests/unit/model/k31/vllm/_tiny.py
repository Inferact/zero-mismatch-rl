"""Tiny K31 geometry shared by the static/skeleton suites (never release sizes)."""

from __future__ import annotations

from megatron.lite.model.k31.config import K31Config
from megatron.lite.model.k31.vllm.model import K31LoRASpec


def tiny_config(**overrides) -> K31Config:
    values = dict(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=3,
        attn_res_block_size=2,
        first_k_dense_replace=1,
        intermediate_size=8,
        num_experts=4,
        num_experts_per_token=2,
        num_shared_experts=1,
        moe_intermediate_size=64,
        routed_expert_hidden_size=64,
        kda_layers=[0, 1],
        kda_num_heads=2,
        kda_head_dim=4,
        kda_short_conv_kernel_size=4,
        dsa_layers=[2],
        num_attention_heads=2,
        q_lora_rank=8,
        kv_lora_rank=8,
        qk_nope_head_dim=4,
        qk_rope_head_dim=2,
        v_head_dim=4,
        index_head_dim=4,
        index_num_heads=2,
    )
    values.update(overrides)
    config = K31Config(**values)
    config.validate()
    return config


TINY_TARGETS = frozenset(
    {
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
    }
)


def tiny_lora(rank: int = 2, alpha: float = 4.0, max_rank: int = 4) -> K31LoRASpec:
    return K31LoRASpec(rank=rank, alpha=alpha, max_rank=max_rank, targets=TINY_TARGETS)
