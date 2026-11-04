"""Fixed-active-set VJP for the K31 vLLM router (sigmoid, bias for selection only).

K31 routing (``FusedMoEFactory`` arguments of ``KimiMoE``): scores are
``sigmoid(logits)``; ``e_score_correction_bias`` only enters the top-k
*selection*; the returned weights are the unbiased scores of the selected
experts, renormalized over the 16 slots, times ``routed_scaling_factor`` (1.0).
The selection is discrete, so the VJP keeps the visible ids fixed and
differentiates only the selected weights with respect to the fp32 logits.
"""

from __future__ import annotations

import os
from typing import Any

import torch

from ._contract import own_visible_tensor


def sigmoid_route_weights(
    logits: torch.Tensor, ids: torch.Tensor, *, renormalize: bool, route_scale: float
) -> torch.Tensor:
    """Differentiable torch form of the selected-weight map (VJP and R3 replay)."""

    selected = torch.sigmoid(logits.float()).gather(-1, ids.long())
    if renormalize:
        selected = selected / selected.sum(dim=-1, keepdim=True)
    if route_scale != 1.0:
        selected = selected * route_scale
    return selected


class _VLLMSigmoidRouteFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, logits, visible_op, renormalize, route_scale):
        weights, ids = visible_op(logits)
        # Consumers (dispatch/permutation) may reuse the id storage; keep a copy.
        ctx.save_for_backward(logits, ids.clone())
        ctx.renormalize = renormalize
        ctx.route_scale = route_scale
        ctx.mark_non_differentiable(ids)
        return own_visible_tensor(weights), ids

    @staticmethod
    def backward(ctx: Any, grad_weights, _grad_ids):
        logits, ids = ctx.saved_tensors
        if os.getenv("MLITE_VALIDATE_INDICES") == "1" and ids.numel():
            low, high = torch.aminmax(ids)
            if int(low.item()) < 0 or int(high.item()) >= logits.shape[-1]:
                raise ValueError("router VJP expert ids are outside the logits")
        with torch.enable_grad():
            replay = logits.detach().float().requires_grad_(True)
            selected = sigmoid_route_weights(
                replay, ids, renormalize=ctx.renormalize, route_scale=ctx.route_scale
            )
            (dlogits,) = torch.autograd.grad(selected, replay, grad_weights.float())
        return dlogits.to(logits.dtype), None, None, None


def fixed_route_vjp(visible_op, logits, *, renormalize: bool, route_scale: float):
    return _VLLMSigmoidRouteFunction.apply(logits, visible_op, renormalize, route_scale)


__all__ = ["fixed_route_vjp", "sigmoid_route_weights"]
