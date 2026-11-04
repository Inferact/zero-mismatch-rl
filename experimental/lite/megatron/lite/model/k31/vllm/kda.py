"""KDA attention slot.

Parameter names and shapes mirror the rollout's fused TP=1 layout so that
``checkpoint.py`` loads them and every GEMM has the rollout's (K, N):

* ``in_proj_qkvgfab``: ``K31Linear`` 7168 -> q, k, v, g (4 x 12288), f_a (128),
  b (96) = 49376 rows, LoRA slices named after the peft targets;
* ``f_b_proj``: 128 -> 12288;  ``o_proj``: 12288 -> 7168;
* ``conv1d.weight`` fp32 ``[3 * 12288, 1, 4]`` (q, k, v conv kernels stacked);
* ``A_log`` fp32 ``[96]``, ``dt_bias`` fp32 ``[12288]``, ``o_norm.weight`` fp32 ``[128]``
  (rounded to bf16 at the kernel boundary, as vLLM holds it).

``forward`` follows ``KimiK3DeltaAttention.forward`` of the rollout: packed in_proj
-> split (q|k|v, g, f_a, b) -> ``f_b_proj(f_a)`` -> :func:`primitive.kda.kda_core`
(conv -> CF-BI -> gated norm, FLA backward) -> ``o_proj``. KDA needs no per-layer
metadata beyond the packed sequence boundaries in ``runtime.cu_seqlens``; every
training sequence starts at position 0 (``ctx = 0``).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from megatron.lite.model.k31.config import K31Config
from megatron.lite.model.k31.vllm.model import K31Linear, K31LoRASpec
from megatron.lite.model.k31.vllm.primitive.kda import KDAKernelConfig, kda_core
from megatron.lite.model.k31.vllm.runtime_metadata import K31Runtime
from megatron.lite.primitive.parallel import ParallelState


class _Weight(nn.Module):
    def __init__(self, *shape: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(*shape, dtype=dtype))


def build_kda_metadata(runtime: K31Runtime, *_: Any, **__: Any) -> None:
    """KDA has no metadata of its own: sequence boundaries come from ``runtime.cu_seqlens`` and the CF-BI
    state pool is allocated per call."""

    del runtime
    return None


class K31KDAAttention(nn.Module):
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
        self.ps = ps
        self.selected_stages = selected_stages
        h = config.hidden_size
        p = config.kda_projection_size
        d = config.kda_head_dim
        self.in_proj_qkvgfab = K31Linear(
            h,
            config.kda_in_proj_sizes,
            ("q_proj", "k_proj", "v_proj", "g_proj", "f_a_proj", "b_proj"),
            lora=lora,
        )
        self.f_b_proj = K31Linear(d, (p,), ("f_b_proj",), lora=lora)
        self.o_proj = K31Linear(p, (h,), ("o_proj",), lora=lora)
        self.conv1d = _Weight(3 * p, 1, config.kda_short_conv_kernel_size, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.empty(config.kda_num_heads, dtype=torch.float32))
        self.dt_bias = nn.Parameter(torch.empty(p, dtype=torch.float32))
        self.o_norm = _Weight(d, dtype=torch.float32)

    def kernel_config(self) -> KDAKernelConfig:
        c = self.config
        return KDAKernelConfig(
            num_heads=c.kda_num_heads,
            head_dim=c.kda_head_dim,
            conv_width=c.kda_short_conv_kernel_size,
            gate_lower_bound=c.kda_gate_lower_bound,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        metadata: Any,
        runtime: K31Runtime,
    ) -> torch.Tensor:
        del metadata
        c = self.config
        p, d, nh = c.kda_projection_size, c.kda_head_dim, c.kda_num_heads
        projected = self.in_proj_qkvgfab(hidden_states, runtime)
        mixed_qkv, g_proj, f_a, beta = projected.split([3 * p, p, d, nh], dim=-1)
        g1 = self.f_b_proj(f_a, runtime)
        core = kda_core(
            self.kernel_config(),
            mixed_qkv,
            g1,
            beta,
            g_proj,
            self.conv1d.weight.view(3 * p, c.kda_short_conv_kernel_size),
            self.A_log,
            self.dt_bias,
            self.o_norm.weight,
            cu_seqlens=runtime.cu_seqlens,
        )
        return self.o_proj(core, runtime)


__all__ = ["K31KDAAttention", "build_kda_metadata"]
