"""Analytic backward bridge for the vLLM-visible batch-invariant RMSNorm."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from ._contract import (
    check_parameter_versions,
    own_visible_tensor,
    parameter_versions,
)


def _rms_norm_vjp(
    grad_output: torch.Tensor,
    value: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    need_dw: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    x = value.float()
    dy = grad_output.float()
    w = weight.float()
    rstd = torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps)
    weighted_dy = dy * w
    projection = (weighted_dy * x).mean(dim=-1, keepdim=True)
    dx = rstd * weighted_dy - x * rstd.pow(3) * projection
    dw = None
    if need_dw:
        reduce_dims = tuple(range(value.ndim - 1))
        dw = (dy * x * rstd).sum(dim=reduce_dims).to(weight.dtype)
    return dx.to(value.dtype), dw


class _VLLMRMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        value: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        visible_op: Callable[..., torch.Tensor],
    ) -> torch.Tensor:
        output = own_visible_tensor(visible_op(value, weight, eps))
        ctx.save_for_backward(value)
        ctx.weights = (weight,)
        ctx.eps = eps
        ctx.versions = parameter_versions((weight,))
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor):
        (weight,) = ctx.weights
        check_parameter_versions((weight,), ctx.versions)
        (value,) = ctx.saved_tensors
        grad_value, grad_weight = _rms_norm_vjp(
            grad_output, value, weight, ctx.eps, need_dw=ctx.needs_input_grad[1]
        )
        return grad_value, grad_weight, None, None


def rms_norm(
    visible_op: Callable[..., torch.Tensor],
    value: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """``rms_norm_batch_invariant`` forward, analytic FP32 VJP."""

    return _VLLMRMSNormFunction.apply(value, weight, eps, visible_op)


__all__ = ["_VLLMRMSNormFunction", "rms_norm"]
