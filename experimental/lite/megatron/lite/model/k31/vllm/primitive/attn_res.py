"""AttnRes (K31 layer-to-layer residual mixing; DS4's mHC slot).

The rollout op ``attn_res(prefix, delta, blocks, ...)`` mutates two buffers in
place: ``prefix += delta`` (rounded to bf16) and, on block-write layers,
``blocks[:, block_write_idx] = updated prefix``.  Training keeps *value*
semantics instead:

* ``prefix`` is an ordinary tensor; a call with ``delta`` returns the updated
  prefix as a new output;
* the bank is a tuple of per-block ``[T, H]`` tensors, each the updated prefix
  of its block-write layer (written once, never rewritten);
* the visible kernel runs on scratch copies (a prefix clone and a freshly
  assembled ``[T, slots, H]`` bank), so neither autograd nor activation
  recompute (or another microbatch) ever sees a buffer the kernel mutated.

Backward recomputes an FP32 functional graph of the same math
(``attn_res_reference``) and differentiates it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch

from ._recompute import visible_functional_vjp


def attn_res_reference(
    prefix: torch.Tensor,
    delta: torch.Tensor | None,
    blocks: Sequence[torch.Tensor],
    norm_weight: torch.Tensor,
    qk_weight: torch.Tensor,
    output_norm_weight: torch.Tensor | None,
    *,
    num_blocks: int,
    eps: float,
    output_norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FP32 math of the rollout kernel; returns ``(output, updated_prefix)``."""

    if delta is not None:
        # bf16 prefix-add (one rounding); its gradient is the identity (STE).
        updated = (prefix.float() + delta.float()).to(prefix.dtype)
    else:
        updated = prefix
    if num_blocks == 0:
        mixed = updated.float()
    else:
        rows = torch.stack([*blocks[:num_blocks], updated], dim=-2).float()
        qk = norm_weight.float() * qk_weight.float()
        rstd = torch.rsqrt(rows.square().mean(dim=-1, keepdim=True) + eps)
        logits = (rows * qk).sum(dim=-1, keepdim=True) * rstd
        probs = torch.softmax(logits, dim=-2)
        mixed = (probs * rows).sum(dim=-2)
    if output_norm_weight is not None:
        rstd_out = torch.rsqrt(mixed.square().mean(dim=-1, keepdim=True) + output_norm_eps)
        mixed = mixed * rstd_out * output_norm_weight.float()
    return mixed.to(prefix.dtype), updated


def _bank(blocks: Sequence[torch.Tensor], slots: int, like: torch.Tensor) -> torch.Tensor:
    bank = like.new_empty((like.shape[0], max(1, slots), like.shape[1]))
    for index, block in enumerate(blocks):
        bank[:, index].copy_(block)
    return bank


def attn_res(
    visible_op: Callable[..., torch.Tensor],
    prefix: torch.Tensor,
    delta: torch.Tensor | None,
    blocks: Sequence[torch.Tensor],
    norm_weight: torch.Tensor,
    qk_weight: torch.Tensor,
    output_norm_weight: torch.Tensor | None,
    *,
    num_blocks: int,
    block_write_idx: int,
    eps: float,
    output_norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One rollout ``attn_res`` call with value semantics.

    Returns ``(output, updated_prefix)``.  ``updated_prefix`` is ``prefix``
    itself when ``delta`` is None.  On a block-write call
    (``block_write_idx >= 0``) the block the rollout writes equals
    ``updated_prefix``; the caller appends it to its bank tuple.
    """

    blocks = tuple(blocks)
    if num_blocks > len(blocks):
        raise ValueError(f"num_blocks={num_blocks} but only {len(blocks)} blocks exist")
    if block_write_idx >= 0 and block_write_idx != len(blocks):
        raise ValueError(
            f"block_write_idx={block_write_idx} must append to a bank of {len(blocks)} blocks"
        )
    used = blocks[:num_blocks]
    slots = max(num_blocks, block_write_idx + 1)
    has_delta = delta is not None
    has_out_norm = output_norm_weight is not None

    def unpack(args):
        it = iter(args)
        prefix_ = next(it)
        delta_ = next(it) if has_delta else None
        norm_w = next(it)
        qk_w = next(it)
        out_w = next(it) if has_out_norm else None
        rest = tuple(it)
        return prefix_, delta_, norm_w, qk_w, out_w, rest

    def visible(*args):
        prefix_, delta_, norm_w, qk_w, out_w, rest = unpack(args)
        scratch_prefix = prefix_.clone() if has_delta else prefix_
        bank = _bank(rest, slots, prefix_)
        output = visible_op(
            scratch_prefix,
            delta_,
            bank,
            norm_w,
            qk_w,
            out_w,
            num_blocks,
            block_write_idx,
            eps,
            output_norm_eps,
        )
        return (output, scratch_prefix) if has_delta else output

    def functional(*args):
        prefix_, delta_, norm_w, qk_w, out_w, rest = unpack(args)
        output, updated = attn_res_reference(
            prefix_,
            delta_,
            rest,
            norm_w,
            qk_w,
            out_w,
            num_blocks=num_blocks,
            eps=eps,
            output_norm_eps=output_norm_eps,
        )
        return (output, updated) if has_delta else output

    inputs: list[torch.Tensor] = [prefix]
    if has_delta:
        inputs.append(delta)
    inputs += [norm_weight, qk_weight]
    if has_out_norm:
        inputs.append(output_norm_weight)
    first_weight = 2 if has_delta else 1
    version_indices = tuple(range(first_weight, first_weight + 2 + int(has_out_norm)))
    inputs += list(used)
    result = visible_functional_vjp(
        visible, functional, tuple(inputs), version_indices=version_indices
    )
    if has_delta:
        output, updated = result
        return output, updated
    return result, prefix


__all__ = ["attn_res", "attn_res_reference"]
