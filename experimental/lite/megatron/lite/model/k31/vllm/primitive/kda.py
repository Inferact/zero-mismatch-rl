"""K31 KDA autograd owner: CF-BI visible forward, FLA ``chunk_kda`` recompute VJP.

Boundary (``KimiK3DeltaAttention.forward`` around ``_forward``): inputs are the packed q|k|v projection before
the short conv, the raw gate ``g1`` (``f_b_proj`` output), raw ``beta`` and the norm gate ``g2``; the output is
the normed core ``[T, H*D]`` bf16, i.e. the ``o_proj`` input.

Forward value: the rollout's prefill kernels through the ``KDA*`` adapters of ``vllm_k31`` (conv -> CF-BI
prefill engine -> gated norm). Bitwise equal to the BI=1 rollout, including tokens the rollout produced one at
a time with the decode engine under CUDA graphs (tests/unit/model/k31/vllm/test_kda_bitwise.py).

Backward: CF-BI has no backward kernel. ``visible_functional_vjp`` recomputes a mathematically equivalent graph
and differentiates it:

* short conv: per-sequence causal depthwise conv + SiLU in FP32 torch ops;
* core: FLA ``chunk_kda`` (gate fused in kernel, ``safe_gate`` lower bound, in-kernel q/k L2 norm), all Triton
  autotuners pinned by :func:`pin_fla_autotune`;
* gated RMSNorm: FP32 torch ops on the forward's weight (rounded to BF16 like the rollout's).

The recomputed graph is an adjacent function (FLA and CF-BI forwards differ by ~4e-3 relative L2 after the
norm), so the gradient is straight-through. Against an FP64 sequential recurrence it is within 0.4-0.9 %
relative L2 (cos > 0.99996) for the activations, i.e. FLA's own BF16 accuracy.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from megatron.lite.primitive.kernels.vllm_k31 import (
    KDA_CONV_WIDTH,
    KDA_HEAD_DIM,
    KDA_NORM_METHODS,
    KDACausalConvAdapter,
    KDACFBIPrefillAdapter,
    KDAGatedNormAdapter,
)

from ._recompute import visible_functional_vjp


@dataclass(frozen=True)
class KDAKernelConfig:
    """One KDA layer as the BI=1 rollout runs it (TP=1: all heads local)."""

    num_heads: int
    head_dim: int = KDA_HEAD_DIM
    conv_width: int = KDA_CONV_WIDTH
    gate_lower_bound: float = -5.0
    # vLLM builds ``FusedRMSNormGated(head_dim, activation="sigmoid")``: default eps, bf16 weight.
    norm_eps: float = 1e-5
    norm_method: str = "forward_cuda"

    def __post_init__(self) -> None:
        if self.head_dim != KDA_HEAD_DIM:
            raise ValueError(f"K31 CF-BI needs head_dim {KDA_HEAD_DIM}, got {self.head_dim}")
        if self.conv_width != KDA_CONV_WIDTH:
            raise ValueError(f"K31 KDA short conv width is {KDA_CONV_WIDTH}, got {self.conv_width}")
        if not self.gate_lower_bound < 0:
            raise ValueError("K31 CF-BI needs a negative gate_lower_bound")
        if self.norm_method not in KDA_NORM_METHODS:
            raise ValueError(f"norm_method must be one of {KDA_NORM_METHODS}")

    @property
    def projection_size(self) -> int:
        return self.num_heads * self.head_dim


def _check(cfg: KDAKernelConfig, mixed_qkv, g1, beta, g2, conv_weight, A_log, dt_bias, norm_weight) -> None:
    hd, h = cfg.projection_size, cfg.num_heads
    t = mixed_qkv.shape[0]
    want = {
        "mixed_qkv": (mixed_qkv, (t, 3 * hd), torch.bfloat16),
        "g1": (g1, (t, hd), torch.bfloat16),
        "beta": (beta, (t, h), torch.bfloat16),
        "g2": (g2, (t, hd), torch.bfloat16),
        "conv_weight": (conv_weight, (3 * hd, cfg.conv_width), torch.float32),
        "A_log": (A_log, (h,), torch.float32),
        "dt_bias": (dt_bias, (hd,), torch.float32),
        "norm_weight": (norm_weight, (cfg.head_dim,), None),
    }
    for name, (value, shape, dtype) in want.items():
        if tuple(value.shape) != shape:
            raise ValueError(f"KDA {name} shape {tuple(value.shape)} != {shape}")
        if dtype is not None and value.dtype != dtype:
            raise TypeError(f"KDA {name} dtype {value.dtype} != {dtype}")


def kda_visible(
    cfg: KDAKernelConfig,
    mixed_qkv: Tensor,
    g1: Tensor,
    beta: Tensor,
    g2: Tensor,
    conv_weight: Tensor,
    A_log: Tensor,
    dt_bias: Tensor,
    norm_weight: Tensor,
    cu_seqlens: Tensor,
) -> Tensor:
    """Rollout prefill branch (``_k31_cfbi_forward``, ``num_prefills > 0``) with ``ctx = 0`` per sequence."""

    _check(cfg, mixed_qkv, g1, beta, g2, conv_weight, A_log, dt_bias, norm_weight)
    H, D, HD = cfg.num_heads, cfg.head_dim, cfg.projection_size
    T = mixed_qkv.shape[0]
    R = int(cu_seqlens.numel()) - 1
    conv = KDACausalConvAdapter()
    # rollout mamba conv-state layout: storage [slot, W-1, 3HD] viewed as (slot, 3HD, W-1); slot 0 unused
    conv_state = torch.empty(
        R + 1, cfg.conv_width - 1, 3 * HD, dtype=mixed_qkv.dtype, device=mixed_qkv.device
    ).transpose(1, 2)
    q, k, v = (
        conv(x, w, s, cu_seqlens).reshape(1, T, H, D)
        for x, w, s in zip(
            mixed_qkv.split(HD, dim=-1), conv_weight.split(HD, dim=0), conv_state.split(HD, dim=-2)
        )
    )
    core = KDACFBIPrefillAdapter()(
        q, k, v, g1.reshape(1, T, H, D), beta.reshape(1, T, H), A_log, dt_bias, cu_seqlens,
        cfg.gate_lower_bound,
    )
    # vLLM holds o_norm.weight in the model dtype (bf16); a wider master is rounded like the loader does
    out = KDAGatedNormAdapter(cfg.norm_method)(
        core, g2.reshape(T, H, D), norm_weight.to(torch.bfloat16), cfg.norm_eps
    )
    return out.reshape(T, HD)


# ---------------------------------------------------------------------------------------------------------------
# backward graph
# ---------------------------------------------------------------------------------------------------------------
_FLA_PACKAGES = ("fla.ops.kda", "fla.ops.common", "fla.ops.gla", "fla.ops.utils", "fla.modules.l2norm")
_PINNED: dict[str, str] = {}


def _autotuner_of(obj):
    from triton.runtime.autotuner import Autotuner

    for _ in range(8):
        if obj is None or isinstance(obj, Autotuner):
            return obj
        obj = getattr(obj, "fn", None)
    return None


def pin_fla_autotune() -> dict[str, str]:
    """Pin every Triton autotuner on the chunk_kda path to its first config.

    Autotune keys of these kernels do not include T, so the selected config would otherwise depend on the
    timing of the first call in each process. Pinning makes the backward reproducible across
    processes. Idempotent; returns {kernel: config}."""

    importlib.import_module("fla.ops.kda")
    importlib.import_module("fla.ops.kda.chunk_bwd")
    for name, mod in list(sys.modules.items()):
        if mod is None or not name.startswith(_FLA_PACKAGES):
            continue
        for attr, obj in list(vars(mod).items()):
            tuner = _autotuner_of(obj)
            if tuner is None:
                continue
            if not getattr(tuner, "_k31_pinned", False):
                tuner.configs = [tuner.configs[0]]
                tuner._k31_pinned = True
            _PINNED[f"{name}.{attr}"] = str(tuner.configs[0])
    return dict(_PINNED)


def _token_positions(cu_seqlens: Tensor, total: int, device) -> Tensor:
    cu = cu_seqlens.to(device=device, dtype=torch.long)
    seq = torch.repeat_interleave(torch.arange(cu.numel() - 1, device=device), cu.diff())
    return torch.arange(total, device=device) - cu[seq]


def causal_conv_silu(x: Tensor, weight: Tensor, cu_seqlens: Tensor) -> Tensor:
    """Packed causal depthwise conv1d + SiLU, FP32. x [T, C], weight [C, W]; weight[:, W-1] hits x[t]."""

    T, W = x.shape[0], weight.shape[1]
    xf = x.float()
    wf = weight.float()
    pos = _token_positions(cu_seqlens, T, x.device)
    acc = xf * wf[:, W - 1]
    for shift in range(1, W):
        shifted = F.pad(xf[: T - shift], (0, 0, shift, 0)) if shift < T else torch.zeros_like(xf)
        valid = (pos >= shift).unsqueeze(-1)
        acc = acc + torch.where(valid, shifted, torch.zeros_like(shifted)) * wf[:, W - 1 - shift]
    return F.silu(acc)


def gated_rms_norm(x: Tensor, gate: Tensor, weight: Tensor, eps: float) -> Tensor:
    # sum / D instead of mean: VLLM_BATCH_INVARIANT replaces aten::mean.dim in this process
    xf = x.float()
    y = xf * torch.rsqrt(xf.square().sum(dim=-1, keepdim=True) / xf.shape[-1] + eps)
    return y * weight.float() * torch.sigmoid(gate.float())


def fla_functional(cfg: KDAKernelConfig, cu_seqlens: Tensor) -> Callable[..., Tensor]:
    """Differentiable recompute graph with the visible op's signature."""

    H, D, HD = cfg.num_heads, cfg.head_dim, cfg.projection_size

    def functional(mixed_qkv, g1, beta, g2, conv_weight, A_log, dt_bias, norm_weight):
        from fla.ops.kda import chunk_kda

        T = mixed_qkv.shape[0]
        cu = cu_seqlens.to(device=mixed_qkv.device, dtype=torch.long)
        qkv = causal_conv_silu(mixed_qkv, conv_weight, cu).to(mixed_qkv.dtype)
        q, k, v = (t.reshape(1, T, H, D) for t in qkv.split(HD, dim=-1))
        o, _ = chunk_kda(
            q=q,
            k=k,
            v=v,
            g=g1.reshape(1, T, H, D),
            beta=beta.float().sigmoid().reshape(1, T, H),
            A_log=A_log,
            dt_bias=dt_bias,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            safe_gate=True,
            lower_bound=cfg.gate_lower_bound,
            cu_seqlens=cu,
        )
        # the weight kda_visible normalizes with: rounded to bf16 (gated_rms_norm widens it to fp32)
        out = gated_rms_norm(
            o.reshape(T, H, D), g2.reshape(T, H, D), norm_weight.to(torch.bfloat16), cfg.norm_eps
        )
        return out.to(mixed_qkv.dtype).reshape(T, HD)

    return functional


def kda_core(
    cfg: KDAKernelConfig,
    mixed_qkv: Tensor,
    g1: Tensor,
    beta: Tensor,
    g2: Tensor,
    conv_weight: Tensor,
    A_log: Tensor,
    dt_bias: Tensor,
    norm_weight: Tensor,
    *,
    cu_seqlens: Tensor,
) -> Tensor:
    """KDA layer between ``in_proj`` / ``f_b_proj`` and ``o_proj``; returns ``[T, H*D]`` bf16.

    Gradients flow to the four activations; the base parameters (conv, A_log, dt_bias, norm) receive
    gradients only if they require grad (K31 RL freezes them; LoRA lives in the projections)."""

    pin_fla_autotune()

    def visible(*inputs):
        return kda_visible(cfg, *inputs, cu_seqlens)

    return visible_functional_vjp(
        visible,
        fla_functional(cfg, cu_seqlens),
        (mixed_qkv, g1, beta, g2, conv_weight, A_log, dt_bias, norm_weight),
        version_indices=(4, 5, 6, 7),
    )


__all__ = [
    "KDAKernelConfig",
    "causal_conv_silu",
    "fla_functional",
    "gated_rms_norm",
    "kda_core",
    "kda_visible",
    "pin_fla_autotune",
]
