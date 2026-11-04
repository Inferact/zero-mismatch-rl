"""Structured activation checkpoint (``primitive.recompute``): pytree stream, replay tape, context."""

from __future__ import annotations

import contextlib
import copy
import itertools
from dataclasses import dataclass

import pytest
import torch
import torch.nn as nn
from torch.utils import _pytree

from megatron.lite.primitive.recompute import (
    active_recompute_tape,
    apply_recompute,
    recompute_stats,
    reset_recompute_stats,
)


@dataclass
class _Stream:
    prefix: torch.Tensor
    delta: torch.Tensor | None
    blocks: tuple[torch.Tensor, ...]


_pytree.register_pytree_node(
    _Stream,
    lambda s: ([s.prefix, s.delta, s.blocks], None),
    lambda c, _: _Stream(c[0], c[1], tuple(c[2])),
    serialized_type_name="tests.recompute._Stream",
)


class _Entry(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *tensors):
        ctx.set_materialize_grads(False)
        return tuple(t.view_as(t) for t in tensors)

    @staticmethod
    def backward(ctx, *grads):
        return grads


class _Runtime:
    """Non-tensor runtime object; ``decide`` changes its answer on every call."""

    def __init__(self) -> None:
        self.counter = itertools.count(1)
        self.context_depth = 0

    def decide(self) -> int:
        return next(self.counter)


class _Layer(nn.Module):
    def __init__(self, hidden: int, write_block: bool, seed: int) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.weight = nn.Parameter((torch.randn(hidden, hidden, generator=g) * 0.2).bfloat16())
        self.write_block = write_block

    def forward(self, stream: _Stream, *, runtime: _Runtime) -> _Stream:
        incoming_blocks = stream.blocks
        tensors = [stream.prefix, *([] if stream.delta is None else [stream.delta]), *stream.blocks]
        if torch.is_grad_enabled() and any(t.requires_grad for t in tensors):
            out = list(_Entry.apply(*tensors))
            prefix = out.pop(0)
            delta = None if stream.delta is None else out.pop(0)
            stream = _Stream(prefix, delta, tuple(out))
        x = stream.prefix if stream.delta is None else stream.prefix + stream.delta
        tape = active_recompute_tape()
        if tape is not None and tape.replaying:
            choice = tape.replay()
        else:
            choice = runtime.decide()
            if tape is not None:
                tape.record(choice)
        assert runtime.context_depth in (0, 1)
        mixed = x
        for block in stream.blocks:
            mixed = mixed + block * 0.5
        h = torch.tanh(mixed @ self.weight) * (1.0 + 0.01 * choice)
        blocks = (*stream.blocks, x) if self.write_block else stream.blocks
        for block in blocks:
            h = h + block * 0.25
        # Pass the incoming bank on as received (not its entry aliases), so a block's
        # gradient is one sum per consuming layer in both modes.  The new block also
        # stays the prefix: one tensor at two output positions.
        out_blocks = (*incoming_blocks, x) if self.write_block else incoming_blocks
        return _Stream(prefix=x, delta=h, blocks=out_blocks)


def _model(num_layers: int = 6, hidden: int = 32) -> nn.ModuleList:
    return nn.ModuleList(
        _Layer(hidden, write_block=(i % 2 == 0), seed=i) for i in range(num_layers)
    )


def _run(layers: nn.ModuleList, x: torch.Tensor, runtime: _Runtime):
    stream = _Stream(prefix=x, delta=None, blocks=())
    for layer in layers:
        stream = layer(stream, runtime=runtime)
    head = stream.prefix + stream.delta
    for block in stream.blocks:
        head = head + block
    loss = (head.float() ** 2).sum()
    loss.backward()
    return head.detach(), {n: p.grad.clone() for n, p in layers.named_parameters()}


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int16)


@contextlib.contextmanager
def _enter_context(runtime: _Runtime):
    runtime.context_depth += 1
    try:
        yield
    finally:
        runtime.context_depth -= 1


class TestStructuredCheckpoint:
    @pytest.mark.parametrize("spec", [["full"], ["sub"]], ids=["full", "module-map"])
    def test_values_and_grads_bitwise(self, spec, monkeypatch):
        monkeypatch.setenv("MLITE_RECOMPUTE_VERIFY", "1")
        x = torch.randn(17, 32).bfloat16()
        plain = _model()
        ckpt = copy.deepcopy(plain)
        captured: list[_Runtime] = []
        rt_ckpt = _Runtime()

        def context_fn():
            captured.append(rt_ckpt)
            return lambda: _enter_context(rt_ckpt)

        module_map = {"sub": lambda layer: layer}
        apply_recompute(
            ckpt,
            spec,
            module_map,
            no_rng_modules={"full", "sub"},
            structured=True,
            context_fn=context_fn,
            strict=True,
        )
        reset_recompute_stats()
        head_ref, grads_ref = _run(plain, x, _Runtime())
        head, grads = _run(ckpt, x, rt_ckpt)
        assert torch.equal(_bits(head), _bits(head_ref))
        for name, grad in grads_ref.items():
            assert torch.equal(_bits(grads[name]), _bits(grad)), name
        stats = recompute_stats()
        assert stats["checkpoints"] == stats["recomputes"] == len(plain)
        assert stats["tape_entries"] == len(plain)
        assert stats["verified"] > 0 and stats["mismatch"] == 0
        assert len(captured) == len(plain)
        # The runtime decided once per layer: the recompute replayed the tape.
        assert next(rt_ckpt.counter) == len(plain) + 1

    def test_input_requires_no_grad(self):
        """Layer 0 input from a frozen embedding: the LoRA-style parameters still get grads."""
        x = torch.randn(9, 32).bfloat16()
        layers = _model(2)
        apply_recompute(layers, ["full"], {}, structured=True)
        _, grads = _run(layers, x, _Runtime())
        assert all(bool(g.abs().sum() > 0) for g in grads.values())

    def test_pass_through_blocks_keep_identity(self):
        x = torch.randn(5, 32).bfloat16().requires_grad_(True)
        layer = _Layer(32, write_block=False, seed=0)
        apply_recompute(nn.ModuleList([layer]), ["full"], {}, structured=True)
        block = torch.randn(5, 32).bfloat16().requires_grad_(True)
        out = layer(_Stream(prefix=x, delta=None, blocks=(block,)), runtime=_Runtime())
        assert out.blocks[0] is block
        assert out.delta.grad_fn is not None

    def test_no_grad_runs_plain_forward(self):
        layers = _model(2)
        apply_recompute(layers, ["full"], {}, structured=True)
        reset_recompute_stats()
        with torch.no_grad():
            stream = _Stream(prefix=torch.randn(4, 32).bfloat16(), delta=None, blocks=())
            for layer in layers:
                stream = layer(stream, runtime=_Runtime())
        assert recompute_stats()["checkpoints"] == 0

    def test_strict_rejects_unknown_module(self):
        with pytest.raises(ValueError, match="unknown recompute modules"):
            apply_recompute(
                _model(1), ["router"], {"moe": lambda layer: layer}, structured=True, strict=True
            )
