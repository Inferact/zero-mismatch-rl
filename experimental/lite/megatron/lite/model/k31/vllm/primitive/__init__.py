"""Autograd owners for the K31 vLLM-visible training path."""

from .activation import situ_and_mul, situ_and_mul_reference
from .attn_res import attn_res, attn_res_reference
from .linear import (
    LoRAProjection,
    gate_linear,
    visible_linear,
    visible_lora_linear,
    vllm_fold_lora_b,
    vllm_lora_scaling,
)
from .norm import rms_norm

__all__ = [
    "LoRAProjection",
    "attn_res",
    "attn_res_reference",
    "gate_linear",
    "rms_norm",
    "situ_and_mul",
    "situ_and_mul_reference",
    "visible_linear",
    "visible_lora_linear",
    "vllm_fold_lora_b",
    "vllm_lora_scaling",
]
