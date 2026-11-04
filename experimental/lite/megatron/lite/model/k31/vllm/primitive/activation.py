"""SiTU gated activation: vLLM ``_C.situ_and_mul`` forward, functional FP32 VJP."""

from __future__ import annotations

from collections.abc import Callable

import torch

from ._recompute import visible_functional_vjp


def situ_and_mul_reference(
    value: torch.Tensor, beta: float, linear_beta: float | None
) -> torch.Tensor:
    """Torch form of ``SituAndMul`` (FP32 math, one rounding)."""

    d = value.shape[-1] // 2
    gate = value[..., :d].float()
    up = value[..., d:].float()
    gate = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    if linear_beta is not None:
        up = linear_beta * torch.tanh(up / linear_beta)
    return (gate * up).to(value.dtype)


def situ_and_mul(
    visible_op: Callable[[torch.Tensor, float, float | None], torch.Tensor],
    value: torch.Tensor,
    *,
    beta: float,
    linear_beta: float | None,
) -> torch.Tensor:
    return visible_functional_vjp(
        lambda x: visible_op(x, beta, linear_beta),
        lambda x: situ_and_mul_reference(x, beta, linear_beta),
        (value,),
    )


__all__ = ["situ_and_mul", "situ_and_mul_reference"]
