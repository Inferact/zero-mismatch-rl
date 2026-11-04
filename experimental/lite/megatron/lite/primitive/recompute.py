# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Model-agnostic activation recompute and offload wrappers."""

from __future__ import annotations

import contextlib
import os
from collections.abc import Callable, Iterator
from typing import Any

import torch  # pyright: ignore[reportMissingImports]
import torch.nn as nn  # pyright: ignore[reportMissingImports]
from torch.utils import _pytree as pytree  # pyright: ignore[reportMissingImports]

# ── CheckpointWithoutOutput ───────────────────────────────────────────────────
# Zero-copy C++ extension: makes dst's UntypedStorage point to src's data.
# Operates below TensorImpl level → ALL views/reshapes that share dst's StorageImpl
# (e.g. TE GroupedLinear's inp.reshape() saved for backward) see the restored data.
# Equivalent to MC's share_storage in megatron/core/tensor_parallel/random.py.
_SHARE_STORAGE_SRC = r"""
#include <torch/extension.h>

void share_storage(at::Tensor dst, at::Tensor src) {
    auto* dst_impl = dst.storage().unsafeGetStorageImpl();
    auto* src_ref  = new c10::Storage(src.storage());
    void*       data   = src_ref->data_ptr().get();
    size_t      nbytes = src_ref->nbytes();
    c10::Device device = src_ref->device();
    c10::DataPtr shared(
        data,
        static_cast<void*>(src_ref),
        [](void* ctx) { delete static_cast<c10::Storage*>(ctx); },
        device);
    dst_impl->set_data_ptr(std::move(shared));
    dst_impl->set_nbytes(nbytes);
}
"""

_share_storage_ext = None


def _get_share_storage() -> Callable:
    global _share_storage_ext
    if _share_storage_ext is None:
        from torch.utils.cpp_extension import load_inline

        _share_storage_ext = load_inline(
            name="share_storage_ext",
            cpp_sources=_SHARE_STORAGE_SRC,
            functions=["share_storage"],
            verbose=False,
        )
    return _share_storage_ext.share_storage


class _CheckpointWithoutOutputFn(torch.autograd.Function):
    """Autograd Function for CheckpointWithoutOutput.

    Forward: runs function with no_grad, saves inputs.
    Backward: uses outputs/inputs set by CheckpointWithoutOutput._recompute().
    """

    @staticmethod
    def forward(ctx, run_function, ckpt_obj, *args):
        ctx.run_function = run_function
        ctx.preserve_rng_state = ckpt_obj.preserve_rng_state
        if ckpt_obj.preserve_rng_state:
            ctx.cpu_rng_state = torch.get_rng_state()
            ctx.cuda_rng_state = torch.cuda.get_rng_state()

        ctx.tensor_indices = [i for i, a in enumerate(args) if isinstance(a, torch.Tensor)]
        ctx.non_tensor_args = [
            (i, a) for i, a in enumerate(args) if not isinstance(a, torch.Tensor)
        ]
        ctx.num_args = len(args)
        ctx.save_for_backward(*[a for a in args if isinstance(a, torch.Tensor)])

        with torch.no_grad():
            outputs = run_function(*args)

        ckpt_obj.ctx = ctx
        return outputs

    @staticmethod
    def backward(ctx, *grad_outputs):
        # inputs and outputs are set by CheckpointWithoutOutput._recompute()
        # before this backward runs (via the hook registered on the downstream tensor).
        inputs = ctx.inputs
        outputs = ctx.outputs
        torch.autograd.backward(outputs, grad_outputs)
        ctx.outputs = None
        ctx.inputs = None
        grads = tuple(inp.grad if isinstance(inp, torch.Tensor) else None for inp in inputs)
        return (None, None) + grads


class CheckpointWithoutOutput:
    """Checkpoint a function and discard its output to save memory.

    Equivalent to MC's CheckpointWithoutOutput from megatron/core/tensor_parallel/random.py.
    The output tensor's storage is freed immediately after downstream computation;
    it is recomputed just-in-time during backward via a hook on the downstream output.

    The C++ share_storage extension restores the output at the StorageImpl level so
    that ALL aliases (including views saved by TE GroupedLinear's backward) see the data.

    Usage (mirrors MC's moe_act pattern)::

        ckpt = CheckpointWithoutOutput()
        h = ckpt.checkpoint(activation_func, fc1_out, probs)
        fc2_out = fc2(h, m_splits)
        ckpt.discard_output_and_register_recompute(fc2_out)
    """

    def __init__(self, preserve_rng_state: bool = True):
        self.preserve_rng_state = preserve_rng_state
        self.run_function: Callable | None = None
        self._cpu_rng: torch.Tensor | None = None
        self._cuda_rng: torch.Tensor | None = None
        self.ctx: Any | None = None
        self.outputs: tuple[torch.Tensor, ...] | None = None

    def checkpoint(self, run_function: Callable, *args) -> Any:
        self.run_function = run_function
        if self.preserve_rng_state:
            self._cpu_rng = torch.get_rng_state()
            self._cuda_rng = torch.cuda.get_rng_state()

        outputs = _CheckpointWithoutOutputFn.apply(run_function, self, *args)
        self.outputs = (outputs,) if isinstance(outputs, torch.Tensor) else tuple(outputs)
        return outputs

    def _recompute(self, _) -> None:
        if self.ctx is None:
            return

        # Reconstruct args from saved context.
        tensors = list(self.ctx.saved_tensors)
        args: list = [None] * self.ctx.num_args
        t_it = iter(tensors)
        for i in self.ctx.tensor_indices:
            t = next(t_it)
            args[i] = t.detach().requires_grad_(t.requires_grad)
        for i, val in self.ctx.non_tensor_args:
            args[i] = val

        # Recompute with forward-time RNG states.
        if self.preserve_rng_state:
            saved_cpu = torch.get_rng_state()
            saved_cuda = torch.cuda.get_rng_state()
            torch.set_rng_state(self._cpu_rng)
            torch.cuda.set_rng_state(self._cuda_rng)
        with torch.enable_grad():
            outputs = self.run_function(*args)
        if self.preserve_rng_state:
            torch.set_rng_state(saved_cpu)
            torch.cuda.set_rng_state(saved_cuda)

        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)

        # Zero-copy: make original output's StorageImpl point to recomputed data.
        share_storage = _get_share_storage()
        for orig, new in zip(self.outputs, outputs, strict=False):
            share_storage(orig, new)

        self.ctx.outputs = list(outputs)
        self.ctx.inputs = args
        self.run_function = None
        self._cpu_rng = None
        self._cuda_rng = None
        self.outputs = None
        self.ctx = None

    def reset(self) -> None:
        """Reset state so the instance can be reused across forward passes."""
        self.run_function = None
        self._cpu_rng = None
        self._cuda_rng = None
        self.ctx = None
        self.outputs = None

    def discard_output_and_register_recompute(self, hook_tensor: torch.Tensor) -> None:
        """Free output tensor storage; recompute when hook_tensor's grad is computed."""
        for out in self.outputs:
            out.untyped_storage().resize_(0)
        if hook_tensor.requires_grad:
            hook_tensor.register_hook(self._recompute)


ModuleMap = dict[str, Callable[[nn.Module], nn.Module | None]]
"""Maps module name → lambda that extracts a sub-module from a layer."""


def apply_recompute(
    layers: nn.ModuleList,
    module_names: list[str],
    module_map: ModuleMap,
    no_rng_modules: set[str] | None = None,
    *,
    structured: bool = False,
    context_fn: Callable[[], Callable[[], Any] | None] | None = None,
    strict: bool = False,
) -> None:
    """Wrap specified sub-modules with activation checkpointing for recomputation.

    ``structured=True`` selects :class:`StructuredCheckpointFunction`: arguments and
    outputs may be arbitrary pytrees (dataclasses registered with ``torch.utils._pytree``,
    runtime objects, ``None``), input tensors returned unchanged keep their own autograd
    history, decisions recorded on the :class:`RecomputeTape` are replayed instead of
    re-derived, and ``context_fn`` re-installs the forward's process context during the
    recompute.  RNG state is preserved only for modules not in ``no_rng_modules``
    (``"full"`` names the whole layer).  ``strict`` rejects unknown module names.
    """
    if not module_names:
        return
    no_rng = no_rng_modules or set()
    if strict:
        unknown = [n for n in module_names if n != "full" and n not in module_map]
        if unknown:
            raise ValueError(f"unknown recompute modules {unknown}; known: {sorted(module_map)}")
        if "full" in module_names and len(module_names) > 1:
            raise ValueError("recompute 'full' already covers every module of the layer")
    for layer in layers:
        if "full" in module_names:
            wrap_checkpoint(
                layer,
                preserve_rng_state="full" not in no_rng,
                structured=structured,
                context_fn=context_fn,
            )
        else:
            wrapped: set[int] = set()
            for mod_name in module_names:
                if mod_name in module_map:
                    submod = module_map[mod_name](layer)
                    if submod is not None and (not structured or id(submod) not in wrapped):
                        wrapped.add(id(submod))
                        wrap_checkpoint(
                            submod,
                            preserve_rng_state=mod_name not in no_rng,
                            structured=structured,
                            context_fn=context_fn,
                        )


def apply_offload(layers: nn.ModuleList, module_names: list[str], module_map: ModuleMap) -> None:
    """Wrap specified sub-modules with activation offloading to CPU."""
    if not module_names:
        return
    try:
        from torch.utils.checkpoint import CheckpointPolicy  # noqa: F401
    except ImportError:
        log_rank0("WARNING: torch.utils.checkpoint policy_fn not available, skipping offload")
        return
    for layer in layers:
        for mod_name in module_names:
            if mod_name in module_map:
                submod = module_map[mod_name](layer)
                if submod is not None:
                    wrap_offload(submod)


class CheckpointFunction(torch.autograd.Function):
    """Reentrant activation checkpoint using custom autograd.Function.

    Adapted from Megatron-Core's CheckpointFunction. Key differences:
    - No distribute_saved_activations (not needed for our TP implementation)
    - No model-parallel RNG tracker (we use TE's built-in RNG management)
    - Handles expert_bias restore for MoE router determinism
    """

    @staticmethod
    def forward(
        ctx: Any, run_function: Callable, preserve_rng_state: bool, *args: torch.Tensor
    ) -> Any:
        ctx.run_function = run_function
        ctx.preserve_rng_state = preserve_rng_state

        # Save RNG states for deterministic recomputation.
        if preserve_rng_state:
            ctx.cpu_rng_state = torch.get_rng_state()
            ctx.cuda_rng_state = torch.cuda.get_rng_state()

        # Run forward without gradient tracking — discard intermediate activations.
        with torch.no_grad():
            outputs = run_function(*args)

        # Save inputs for recomputation in backward.
        ctx.save_for_backward(*args)
        return outputs

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor) -> tuple:
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError(
                "Checkpointing is not compatible with .grad(), use .backward() instead"
            )

        inputs = ctx.saved_tensors

        # Fork RNG: restore forward-time states, then reset to current after recompute.
        if ctx.preserve_rng_state:
            current_cpu_rng = torch.get_rng_state()
            current_cuda_rng = torch.cuda.get_rng_state()
            torch.set_rng_state(ctx.cpu_rng_state)
            torch.cuda.set_rng_state(ctx.cuda_rng_state)

        # Recompute forward pass with gradients enabled.
        detached = tuple(
            t.detach().requires_grad_(t.requires_grad) if isinstance(t, torch.Tensor) else t
            for t in inputs
        )
        with torch.enable_grad():
            outputs = ctx.run_function(*detached)

        # Restore RNG states.
        if ctx.preserve_rng_state:
            torch.set_rng_state(current_cpu_rng)
            torch.cuda.set_rng_state(current_cuda_rng)

        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)

        # Filter to outputs that need gradients.
        outputs_with_grad = []
        grad_for_outputs = []
        for out, grad in zip(outputs, grad_outputs, strict=False):
            if torch.is_tensor(out) and out.requires_grad:
                outputs_with_grad.append(out)
                grad_for_outputs.append(grad)

        if outputs_with_grad:
            torch.autograd.backward(outputs_with_grad, grad_for_outputs)

        grads = tuple(inp.grad if isinstance(inp, torch.Tensor) else None for inp in detached)
        # None for run_function, None for preserve_rng_state, then grads for each input.
        return (None, None) + grads


def wrap_checkpoint(
    module: nn.Module,
    *,
    preserve_rng_state: bool = True,
    structured: bool = False,
    context_fn: Callable[[], Callable[[], Any] | None] | None = None,
) -> None:
    """Wrap a module's forward with reentrant activation checkpointing."""
    if structured:
        _wrap_structured_checkpoint(
            module, preserve_rng_state=preserve_rng_state, context_fn=context_fn
        )
        return
    original_forward = module.forward
    _routers = [m for m in module.modules() if hasattr(m, "expert_bias")]

    def _checkpointed_forward(*args, **kwargs):
        # expert_bias is modified in-place by the router during forward.
        # Save and restore it so the recomputation in backward sees the same values.
        if _routers:
            saved = [r.expert_bias.clone() for r in _routers]
            call_count = [0]

            def _fwd(*a, **kw):
                call_count[0] += 1
                if call_count[0] > 1:
                    for r, s in zip(_routers, saved, strict=False):
                        r.expert_bias.copy_(s)
                return original_forward(*a, **kw)

        else:
            _fwd = original_forward

        # CheckpointFunction.apply only accepts positional tensor args.
        # Wrap kwargs into the function closure.
        if kwargs:

            def _fn(*a):
                return _fwd(*a, **kwargs)

        else:
            _fn = _fwd

        return CheckpointFunction.apply(_fn, preserve_rng_state, *args)

    module.forward = _checkpointed_forward


# ── Structured checkpoint ─────────────────────────────────────────────────────
# Models whose layers pass non-tensor state (a decoder-stream dataclass, runtime
# objects, metadata) or whose forward consumes decisions made outside its inputs
# (R3 routes replayed from the rollout) cannot use CheckpointFunction: it saves
# only positional tensors, requires tensor outputs, and re-derives everything in
# backward.  The structured variant flattens arguments and outputs with
# ``torch.utils._pytree`` and adds a replay tape and a captured forward context.


class RecomputeTape:
    """Decisions a checkpointed forward records and its recompute replays verbatim.

    A module calls :func:`active_recompute_tape`; during the original (no-grad)
    forward the tape is recording and the module stores what it decided (for
    example the MoE routes actually used, after R3 replay), during the recompute
    the tape is replaying and the module takes the stored value instead of
    deciding again.  Every recorded entry must be consumed exactly once, in order.
    """

    def __init__(self) -> None:
        self._entries: list[Any] = []
        self._cursor = 0
        self.replaying = False

    def record(self, value: Any) -> Any:
        if self.replaying:
            raise RuntimeError("recompute tape is replaying; record() is a forward-only call")
        self._entries.append(value)
        return value

    def replay(self) -> Any:
        if not self.replaying:
            raise RuntimeError("recompute tape is recording; replay() is a recompute-only call")
        if self._cursor >= len(self._entries):
            raise RuntimeError(
                f"recompute consumed more tape entries than its forward recorded ({len(self._entries)})"
            )
        value = self._entries[self._cursor]
        self._cursor += 1
        return value

    def finish(self) -> None:
        if self._cursor != len(self._entries):
            raise RuntimeError(
                f"recompute replayed {self._cursor} of {len(self._entries)} recorded tape entries"
            )

    def __len__(self) -> int:
        return len(self._entries)


_TAPES: list[RecomputeTape] = []
_RECOMPUTE_STATS = {
    "checkpoints": 0,
    "recomputes": 0,
    "tape_entries": 0,
    "verified": 0,
    "mismatch": 0,
}


def active_recompute_tape() -> RecomputeTape | None:
    """The tape of the innermost structured checkpoint being run, if any."""

    return _TAPES[-1] if _TAPES else None


def recompute_stats() -> dict[str, int]:
    """Counters since :func:`reset_recompute_stats`: checkpointed forwards, recomputes,
    replayed tape entries, and (``MLITE_RECOMPUTE_VERIFY=1``) recomputed outputs compared
    bit for bit with the forward's / found different."""

    return dict(_RECOMPUTE_STATS)


def reset_recompute_stats() -> None:
    for key in _RECOMPUTE_STATS:
        _RECOMPUTE_STATS[key] = 0


@contextlib.contextmanager
def _tape_scope(tape: RecomputeTape) -> Iterator[None]:
    _TAPES.append(tape)
    try:
        yield
    finally:
        popped = _TAPES.pop()
        assert popped is tape


def _bitwise_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.is_floating_point():
        width = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[a.element_size()]
        return bool(torch.equal(a.contiguous().view(width), b.contiguous().view(width)))
    return bool(torch.equal(a, b))


def _same_memory(a: Any, b: torch.Tensor) -> bool:
    return a is b or (
        isinstance(a, torch.Tensor)
        and a.data_ptr() == b.data_ptr()
        and a.shape == b.shape
        and a.stride() == b.stride()
        and a.dtype == b.dtype
    )


class _StructuredRun:
    """Everything a structured checkpoint needs besides the saved input tensors."""

    __slots__ = (
        "context",
        "cpu_rng",
        "cuda_rng",
        "expected",
        "in_leaves",
        "in_spec",
        "out_kinds",
        "out_spec",
        "preserve_rng_state",
        "run_function",
        "tape",
        "tensor_slots",
    )

    def __init__(self, run_function, in_spec, in_leaves, tensor_slots, preserve_rng_state, context):
        self.run_function = run_function
        self.in_spec = in_spec
        self.in_leaves = in_leaves
        self.tensor_slots = tensor_slots
        self.preserve_rng_state = preserve_rng_state
        self.cpu_rng = None
        self.cuda_rng = None
        self.context = context
        self.tape = RecomputeTape()
        self.out_spec = None
        self.out_kinds: list[tuple[str, Any]] = []
        self.expected: list[torch.Tensor] | None = None

    def call(self, tensors) -> Any:
        leaves = list(self.in_leaves)
        for slot, tensor in zip(self.tensor_slots, tensors, strict=True):
            leaves[slot] = tensor
        args, kwargs = pytree.tree_unflatten(leaves, self.in_spec)
        return self.run_function(*args, **kwargs)


class StructuredCheckpointFunction(torch.autograd.Function):
    """Reentrant checkpoint over pytree arguments/outputs with a replay tape.

    Forward (no grad) runs the module once with a recording :class:`RecomputeTape`
    and saves only the input tensors.  Output leaves that *are* input tensors are
    passed through outside this function (they keep their own autograd history and
    their gradient is not routed through the checkpoint); every other tensor leaf is
    an output of this function.  Backward re-runs the module on detached inputs with
    the tape replaying and the forward's captured context, then back-propagates.
    """

    @staticmethod
    def forward(ctx: Any, run: _StructuredRun, anchor: torch.Tensor, *tensors: torch.Tensor):
        del anchor  # makes the outputs differentiable when no input tensor requires grad
        ctx.set_materialize_grads(False)
        if run.preserve_rng_state:
            run.cpu_rng = torch.get_rng_state()
            run.cuda_rng = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
        with torch.no_grad(), _tape_scope(run.tape):
            out = run.call(tensors)
        out_leaves, run.out_spec = pytree.tree_flatten(out)
        by_id = {id(t): i for i, t in enumerate(tensors)}
        new: list[torch.Tensor] = []
        new_ids: dict[int, int] = {}
        for leaf in out_leaves:
            if isinstance(leaf, torch.Tensor):
                j = by_id.get(id(leaf))
                if j is not None:
                    run.out_kinds.append(("input", j))
                elif id(leaf) in new_ids:
                    # One tensor at several output positions stays one output, so its
                    # consumers' gradients accumulate in one buffer, as without recompute.
                    run.out_kinds.append(("new", new_ids[id(leaf)]))
                else:
                    new_ids[id(leaf)] = len(new)
                    run.out_kinds.append(("new", len(new)))
                    new.append(leaf)
            else:
                run.out_kinds.append(("const", leaf))
        if not new:
            raise RuntimeError("structured checkpoint: the module produced no new tensor output")
        if os.environ.get("MLITE_RECOMPUTE_VERIFY") == "1":
            run.expected = [t.detach().clone() for t in new]
        non_diff = [t for t in new if not t.is_floating_point()]
        if non_diff:
            ctx.mark_non_differentiable(*non_diff)
        ctx.run = run
        ctx.save_for_backward(*tensors)
        _RECOMPUTE_STATS["checkpoints"] += 1
        return tuple(new)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor | None):
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError(
                "Checkpointing is not compatible with .grad(), use .backward() instead"
            )
        run: _StructuredRun = ctx.run
        inputs = ctx.saved_tensors
        detached = tuple(t.detach().requires_grad_(t.requires_grad) for t in inputs)
        if run.preserve_rng_state:
            current_cpu = torch.get_rng_state()
            current_cuda = torch.cuda.get_rng_state() if run.cuda_rng is not None else None
            torch.set_rng_state(run.cpu_rng)
            if run.cuda_rng is not None:
                torch.cuda.set_rng_state(run.cuda_rng)
        run.tape.replaying = True
        context = run.context() if run.context is not None else contextlib.nullcontext()
        try:
            with torch.enable_grad(), _tape_scope(run.tape), context:
                out = run.call(detached)
        finally:
            if run.preserve_rng_state:
                torch.set_rng_state(current_cpu)
                if current_cuda is not None:
                    torch.cuda.set_rng_state(current_cuda)
        run.tape.finish()
        out_leaves, out_spec = pytree.tree_flatten(out)
        if out_spec != run.out_spec or len(out_leaves) != len(run.out_kinds):
            raise RuntimeError(
                "structured checkpoint: recompute output structure differs from forward"
            )
        slots: dict[int, torch.Tensor] = {}
        for leaf, (kind, ref) in zip(out_leaves, run.out_kinds, strict=True):
            if kind == "input":
                # With grad enabled the module may hand the input back through an
                # autograd alias (an identity node); it must still be the same memory.
                if not _same_memory(leaf, detached[ref]):
                    raise RuntimeError(
                        "structured checkpoint: pass-through output changed in recompute"
                    )
            elif kind == "new":
                if not isinstance(leaf, torch.Tensor):
                    raise RuntimeError("structured checkpoint: recompute lost a tensor output")
                if ref in slots and slots[ref] is not leaf:
                    raise RuntimeError("structured checkpoint: shared output split in recompute")
                slots[ref] = leaf
        new = [slots[i] for i in range(len(slots))]
        if run.expected is not None:
            for got, want in zip(new, run.expected, strict=True):
                _RECOMPUTE_STATS["verified"] += 1
                if not _bitwise_equal(got.detach(), want):
                    _RECOMPUTE_STATS["mismatch"] += 1
        _RECOMPUTE_STATS["recomputes"] += 1
        _RECOMPUTE_STATS["tape_entries"] += len(run.tape)
        roots, root_grads = [], []
        for tensor, grad in zip(new, grad_outputs, strict=True):
            if grad is not None and tensor.requires_grad:
                roots.append(tensor)
                root_grads.append(grad)
        if roots:
            torch.autograd.backward(roots, root_grads)
        grads = tuple(t.grad if t.requires_grad else None for t in detached)
        ctx.run = None
        return (None, None, *grads)


def _wrap_structured_checkpoint(
    module: nn.Module,
    *,
    preserve_rng_state: bool,
    context_fn: Callable[[], Callable[[], Any] | None] | None,
) -> None:
    original_forward = module.forward

    def _checkpointed_forward(*args, **kwargs):
        # No graph to save (old-logprob / eval), or already inside a structured
        # checkpoint (its recompute covers this module): plain forward.
        if not torch.is_grad_enabled() or _TAPES:
            return original_forward(*args, **kwargs)
        leaves, spec = pytree.tree_flatten((args, kwargs))
        slots = [i for i, leaf in enumerate(leaves) if isinstance(leaf, torch.Tensor)]
        tensors = [leaves[i] for i in slots]
        in_leaves = [None if isinstance(leaf, torch.Tensor) else leaf for leaf in leaves]
        context = context_fn() if context_fn is not None else None
        run = _StructuredRun(original_forward, spec, in_leaves, slots, preserve_rng_state, context)
        anchor = torch.empty(0, requires_grad=True)
        new = StructuredCheckpointFunction.apply(run, anchor, *tensors)
        new = new if isinstance(new, tuple) else (new,)
        out_leaves = []
        for kind, ref in run.out_kinds:
            if kind == "input":
                out_leaves.append(tensors[ref])
            elif kind == "new":
                out_leaves.append(new[ref])
            else:
                out_leaves.append(ref)
        return pytree.tree_unflatten(out_leaves, run.out_spec)

    module.forward = _checkpointed_forward


def wrap_offload(module: nn.Module) -> None:
    """Wrap a module's forward with activation offloading."""
    original_forward = module.forward

    def _offloaded_forward(*args, **kwargs):
        return torch.utils.checkpoint.checkpoint(
            original_forward, *args, use_reentrant=False, **kwargs
        )

    module.forward = _offloaded_forward


def log_rank0(msg: str) -> None:
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print(f"[megatron.lite] {msg}", flush=True)


def parse_recompute_spec(
    recompute: str | list[str] | tuple[str, ...] | None,
) -> list[str]:
    """Parse a recompute spec into a list of module names."""
    if recompute is None or recompute == "none":
        return []
    if recompute == "full":
        return ["full"]
    if isinstance(recompute, (list, tuple)):
        return list(recompute)
    return recompute.split(",")


__all__ = [
    "CheckpointFunction",
    "CheckpointWithoutOutput",
    "ModuleMap",
    "RecomputeTape",
    "StructuredCheckpointFunction",
    "active_recompute_tape",
    "apply_offload",
    "apply_recompute",
    "parse_recompute_spec",
    "recompute_stats",
    "reset_recompute_stats",
    "wrap_checkpoint",
    "wrap_offload",
]
