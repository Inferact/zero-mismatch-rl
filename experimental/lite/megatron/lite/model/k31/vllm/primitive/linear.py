"""Training bridges for K31 vLLM-visible dense, router and LoRA-leaf linears.

Forward values come from the rollout kernels (overlay bi-gemm-r5 and
lora-f32add-bi-r3); backward is an FP32 VJP.  The K31 base is frozen, so the
base-weight gradient is only formed when the weight actually requires grad.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch import nn

from ._contract import (
    check_parameter_versions,
    own_visible_tensor,
    parameter_versions,
)


def _mm_dx(dy32: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if weight.dtype != torch.bfloat16:
        return torch.mm(dy32, weight.float())
    from megatron.lite.primitive.alignment.split_bf16_mm import mm_f32

    return mm_f32(dy32, weight.detach())


def _fp32_linear_grads(
    grad_output: torch.Tensor,
    value: torch.Tensor,
    weight: torch.Tensor,
    *,
    need_dx: bool,
    need_dw: bool,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if grad_output.shape[:-1] != value.shape[:-1]:
        raise RuntimeError("linear bridge received incompatible grad_output shape")
    dy2d = grad_output.reshape(-1, grad_output.shape[-1]).float()
    dx = dw = None
    if need_dx:
        # BF16 weights are exact in BF16: FP32-accurate dx on tensor cores (split_bf16_mm).
        dx = _mm_dx(dy2d, weight).to(value.dtype).reshape(value.shape)
    if need_dw:
        x2d = value.reshape(-1, value.shape[-1]).float()
        dw = torch.mm(dy2d.T, x2d).to(weight.dtype)
    return dx, dw


class _VLLMLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, value, master_weight, visible_op):
        output = visible_op(value, master_weight)
        if isinstance(output, (tuple, list)):
            output = output[0]
        output = own_visible_tensor(output)
        ctx.save_for_backward(value)
        ctx.master_weight = master_weight
        ctx.versions = parameter_versions((master_weight,))
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor):
        master_weight = ctx.master_weight
        check_parameter_versions((master_weight,), ctx.versions)
        (value,) = ctx.saved_tensors
        dx, dw = _fp32_linear_grads(
            grad_output,
            value,
            master_weight,
            need_dx=ctx.needs_input_grad[0],
            need_dw=ctx.needs_input_grad[1],
        )
        return dx, dw, None


def visible_linear(
    visible_op: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    value: torch.Tensor,
    master_weight: torch.Tensor,
) -> torch.Tensor:
    """Exact rollout dense linear forward (``bi_linear``) with an FP32 VJP."""

    return _VLLMLinearFunction.apply(value, master_weight, visible_op)


def gate_linear(
    visible_op: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    value: torch.Tensor,
    master_weight: torch.Tensor,
) -> torch.Tensor:
    """Exact rollout router logits (``bi_router``, FP32 output) with an FP32 VJP."""

    return _VLLMLinearFunction.apply(value, master_weight, visible_op)


# ---------------------------------------------------------------------------
# LoRA leaf
# ---------------------------------------------------------------------------
def vllm_lora_scaling(rank: int, alpha: float, *, use_rslora: bool = False) -> float:
    """``PEFTHelper.vllm_lora_scaling_factor``: alpha/r, or alpha/sqrt(r) for rsLoRA."""

    return float(alpha) / (math.sqrt(rank) if use_rslora else float(rank))


def vllm_fold_lora_b(lora_b: torch.Tensor, scaling: float) -> torch.Tensor:
    """The B vLLM holds after loading: ``b = b.to(bf16); b *= scaling`` (LoRALayerWeights.optimize)."""

    folded = lora_b.detach().to(torch.bfloat16).clone()
    if scaling != 1:
        folded *= scaling
    return folded


class LoRAProjection(nn.Module):
    """Adapter factors of one (possibly packed) vLLM projection.

    ``slice_names`` are the HF/peft target names in vLLM's packed slice order;
    ``slice_sizes`` their output widths.  Parameters are the raw peft tensors:
    ``lora_A[name]`` is ``[rank, in]`` and ``lora_B[name]`` is ``[out_slice,
    rank]``; the forward folds ``scaling`` into B exactly like vLLM.
    """

    def __init__(
        self,
        in_features: int,
        slice_names: Sequence[str],
        slice_sizes: Sequence[int],
        *,
        rank: int,
        alpha: float,
        max_rank: int | None = None,
        use_rslora: bool = False,
    ) -> None:
        super().__init__()
        if len(slice_names) != len(slice_sizes) or not slice_names:
            raise ValueError("LoRA slices need one name per output slice")
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.in_features = int(in_features)
        self.slice_names = tuple(slice_names)
        self.slice_sizes = tuple(int(size) for size in slice_sizes)
        self.rank = int(rank)
        self.max_rank = int(max_rank or rank)
        if self.max_rank < self.rank:
            raise ValueError("max_rank must be >= rank")
        self.scaling = vllm_lora_scaling(rank, alpha, use_rslora=use_rslora)
        self.lora_A = nn.ParameterDict(
            {
                name: nn.Parameter(torch.empty(rank, in_features, dtype=torch.bfloat16))
                for name in self.slice_names
            }
        )
        self.lora_B = nn.ParameterDict(
            {
                name: nn.Parameter(torch.zeros(size, rank, dtype=torch.bfloat16))
                for name, size in zip(self.slice_names, self.slice_sizes, strict=True)
            }
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for weight in self.lora_A.values():
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        for weight in self.lora_B.values():
            nn.init.zeros_(weight)

    @property
    def out_features(self) -> int:
        return sum(self.slice_sizes)

    def a_params(self) -> tuple[torch.Tensor, ...]:
        return tuple(self.lora_A[name] for name in self.slice_names)

    def b_params(self) -> tuple[torch.Tensor, ...]:
        return tuple(self.lora_B[name] for name in self.slice_names)


def _stack_vllm(
    a_params: Sequence[torch.Tensor],
    b_params: Sequence[torch.Tensor],
    scaling: float,
    max_rank: int,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    """vLLM ``lora_{a,b}_stacked``: ``[max_loras=1, 1, max_rank, in]`` / ``[1, 1, out, max_rank]``."""

    a_stacked = []
    b_stacked = []
    for a, b in zip(a_params, b_params, strict=True):
        rank = a.shape[0]
        sa = a.new_zeros((1, 1, max_rank, a.shape[1]), dtype=torch.bfloat16)
        sa[0, 0, :rank].copy_(a.detach().to(torch.bfloat16))
        sb = b.new_zeros((1, 1, b.shape[0], max_rank), dtype=torch.bfloat16)
        sb[0, 0, :, :rank].copy_(vllm_fold_lora_b(b, scaling))
        a_stacked.append(sa)
        b_stacked.append(sb)
    return tuple(a_stacked), tuple(b_stacked)


class _VLLMLoRALinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, value, master_weight, spec, *factors):
        (
            delta_op,
            leaf_op,
            meta,
            scaling,
            max_rank,
            slice_sizes,
        ) = spec
        n = len(slice_sizes)
        a_params, b_params = factors[:n], factors[n:]
        a_stacked, b_stacked = _stack_vllm(a_params, b_params, scaling, max_rank)
        x2d = value.reshape(-1, value.shape[-1])
        if not x2d.is_contiguous():
            x2d = x2d.contiguous()
        delta = delta_op(x2d, a_stacked, b_stacked, tuple(slice_sizes), meta)
        output = own_visible_tensor(leaf_op(x2d, master_weight, delta))
        output = output.reshape(*value.shape[:-1], output.shape[-1])
        ctx.save_for_backward(value, *factors)
        ctx.master_weight = master_weight
        ctx.versions = parameter_versions((master_weight, *factors))
        ctx.scaling = scaling
        ctx.slice_sizes = tuple(slice_sizes)
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor):
        saved = ctx.saved_tensors
        value, factors = saved[0], saved[1:]
        n = len(ctx.slice_sizes)
        a_params, b_params = factors[:n], factors[n:]
        weight = ctx.master_weight
        check_parameter_versions((weight, *factors), ctx.versions)
        x32 = value.reshape(-1, value.shape[-1]).float()
        dy32 = grad_output.reshape(-1, grad_output.shape[-1]).float()
        need_dx = ctx.needs_input_grad[0]
        need_dw = ctx.needs_input_grad[1]
        dx32 = _mm_dx(dy32, weight) if need_dx else None
        dw = torch.mm(dy32.T, x32).to(weight.dtype) if need_dw else None
        grads_a: list[torch.Tensor | None] = []
        grads_b: list[torch.Tensor | None] = []
        offset = 0
        for i, size in enumerate(ctx.slice_sizes):
            a32 = a_params[i].float()
            b_visible32 = vllm_fold_lora_b(b_params[i], ctx.scaling).float()
            dy_s = dy32[:, offset : offset + size]
            offset += size
            dyb = torch.mm(dy_s, b_visible32)  # [T, r]
            if dx32 is not None:
                dx32 = dx32 + torch.mm(dyb, a32)
            grads_a.append(
                torch.mm(dyb.T, x32).to(a_params[i].dtype)
                if ctx.needs_input_grad[3 + i]
                else None
            )
            if ctx.needs_input_grad[3 + n + i]:
                xa = torch.mm(x32, a32.T)  # [T, r]
                # STE through vLLM's bf16 fold: d b_raw = scaling * d b_visible.
                grads_b.append((torch.mm(dy_s.T, xa) * ctx.scaling).to(b_params[i].dtype))
            else:
                grads_b.append(None)
        dx = dx32.to(value.dtype).reshape(value.shape) if dx32 is not None else None
        return (dx, dw, None, *grads_a, *grads_b)


def visible_lora_linear(
    delta_op: Callable[..., torch.Tensor],
    leaf_op: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    value: torch.Tensor,
    master_weight: torch.Tensor,
    lora: LoRAProjection,
    meta: Any,
) -> torch.Tensor:
    """Rollout LoRA leaf ``bf16(fp32(x W^T) + fp32(punica delta))`` with an FP32 VJP.

    ``delta_op`` is ``PunicaLoRADeltaAdapter`` (same shrink/expand launch as
    the rollout), ``leaf_op`` is ``LoRALeafAdapter``; ``meta`` is the
    batch's ``LoRAKernelMeta`` (all tokens on adapter slot 0).
    """

    if master_weight.shape[0] != lora.out_features:
        raise ValueError(
            f"LoRA slices cover {lora.out_features} rows, base weight has {master_weight.shape[0]}"
        )
    spec = (delta_op, leaf_op, meta, lora.scaling, lora.max_rank, lora.slice_sizes)
    return _VLLMLoRALinearFunction.apply(
        value, master_weight, spec, *lora.a_params(), *lora.b_params()
    )


__all__ = [
    "LoRAProjection",
    "gate_linear",
    "visible_linear",
    "visible_lora_linear",
    "vllm_fold_lora_b",
    "vllm_lora_scaling",
]
