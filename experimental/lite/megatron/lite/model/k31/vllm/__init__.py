"""K31 training implementation on the rollout's vLLM / BI-overlay kernels."""

from megatron.lite.model.k31.vllm.model import K31Model
from megatron.lite.model.k31.vllm.protocol import ImplConfig, SelectorConfig
from megatron.lite.model.k31.vllm.runtime_metadata import (
    K31Runtime,
    build_k31_runtime,
    initialize_k31_vllm_batch_invariance,
)

__all__ = [
    "ImplConfig",
    "K31Model",
    "K31Runtime",
    "SelectorConfig",
    "build_k31_runtime",
    "initialize_k31_vllm_batch_invariance",
]
