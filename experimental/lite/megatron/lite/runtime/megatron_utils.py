# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Megatron-Core utilities aligned with VERL's megatron_utils.

All functions here are ported from VERL and could theoretically be replaced
by ``from verl.utils.megatron_utils import ...`` if Megatron Lite ever depends on verl.

Sources:
  - verl/utils/megatron_utils.py  (offload/load, register_megatron_training_hooks)
  - verl/utils/megatron/optimizer.py  (get_megatron_last_lr)
"""

from __future__ import annotations

import gc
from typing import Any

import torch

# ======================================================================
# Rank utilities
# ======================================================================


def is_mp_src_rank_with_outputs() -> bool:
    """True on the rank that holds the final model output (loss).

    Only last PP stage, first TP rank, first CP rank has the output.
    VERL: MegatronEngine.is_mp_src_rank_with_outputs
    """
    from megatron.core import parallel_state as mpu

    return (
        mpu.get_tensor_model_parallel_rank() == 0
        and mpu.get_pipeline_model_parallel_rank()
        == mpu.get_pipeline_model_parallel_world_size() - 1
        and mpu.get_context_parallel_rank() == 0
    )


# ======================================================================
# Training hooks — register_megatron_training_hooks
# ======================================================================


def register_training_hooks(model_list: list, optimizer) -> None:
    """Register megatron training callbacks on model config.

    Ref: megatron/training/training.py (core_v0.15.0rc7, L2039-L2057)
    """
    from megatron.core.distributed import DistributedDataParallel as DDP
    from megatron.core.distributed import finalize_model_grads
    from megatron.core.utils import get_model_config

    for one_model in model_list:
        config = get_model_config(one_model)
        if optimizer is not None:
            config.grad_scale_func = optimizer.scale_loss
        config.finalize_model_grads_func = finalize_model_grads

        optimizer_config = getattr(optimizer, "config", None)
        overlap_param_gather = getattr(optimizer_config, "overlap_param_gather", False)
        overlap_grad_reduce = getattr(one_model.ddp_config, "overlap_grad_reduce", False)
        align_grad_reduce = True
        align_param_gather = getattr(one_model.ddp_config, "align_param_gather", False)

        if isinstance(model_list[0], DDP) and overlap_grad_reduce:
            config.no_sync_func = [m.no_sync for m in model_list]
            if len(model_list) == 1:
                config.no_sync_func = config.no_sync_func[0]
            if align_grad_reduce:
                config.grad_sync_func = [m.start_grad_sync for m in model_list]
                if len(model_list) == 1:
                    config.grad_sync_func = config.grad_sync_func[0]
        if overlap_param_gather and align_param_gather:
            config.param_sync_func = [m.start_param_sync for m in model_list]
            if len(model_list) == 1:
                config.param_sync_func = config.param_sync_func[0]


# ======================================================================
# Model offload / load — offload_megatron_model_to_cpu / load_megatron_model_to_gpu
# ======================================================================


def _is_megatron_ddp(model_chunk: Any) -> bool:
    try:
        from megatron.core.distributed import DistributedDataParallel as DDP
    except Exception:
        return False

    return (
        isinstance(model_chunk, DDP)
        and hasattr(model_chunk, "buffers")
        and hasattr(model_chunk, "expert_parallel_buffers")
        and hasattr(model_chunk, "module")
    )


def _has_fsdp2(model_chunk) -> bool:
    try:
        from torch.distributed.fsdp import FSDPModule
    except ImportError:
        return False
    modules = getattr(model_chunk, "modules", None)
    return callable(modules) and any(isinstance(m, FSDPModule) for m in modules())


def _reshard_fsdp2_modules(model_chunk: Any) -> None:
    """Return nested FSDP2 modules to their sharded state before offload.

    ``reshard_after_forward`` controls the normal forward/backward schedule,
    but it is not a context-switch barrier: prefetch, export, or an exception
    can leave one or more units materialized when VERL asks us to offload the
    model.  Calling ``Module.to("cpu")`` in that state moves full parameters
    and can also make FSDP2 reset a partially materialized plain Parameter as
    though it were a sharded DTensor.  Reshard children before parents so only
    local shards cross the device boundary.
    """
    try:
        from torch.distributed.fsdp import FSDPModule
    except ImportError:
        return

    modules = getattr(model_chunk, "modules", None)
    if not callable(modules):
        return
    for module in reversed(list(modules())):
        if isinstance(module, FSDPModule):
            module.reshard()


# ======================================================================
# Persistent host copies for module offload (``persist=True``)
# ======================================================================
# A colocated RL actor offloads its whole model to the host after every phase.  ``Module.to("cpu")``
# allocates fresh pageable host memory each time (page faults over ~100+ GiB per GPU on a large MoE)
# and copies every byte back, although the frozen base never changes.  With ``persist=True`` each
# parameter / buffer / gradient keeps one host tensor across offloads: frozen tensors whose GPU
# copy is unchanged since the last onload (same storage, same version counter) are not copied again,
# trainable ones are copied into their existing host tensor (pinned with ``persist_pin``).
# ``persist_verify`` re-digests the skipped frozen tensors (diagnostic).
# Every byte that comes back to the GPU is the byte that left it; nothing numerical changes.


def _persist_names(model_chunk) -> tuple[dict[int, str], set[str]]:
    names: dict[int, str] = {}
    frozen: set[str] = set()
    for n, p in model_chunk.named_parameters(remove_duplicate=False):
        names.setdefault(id(p), "p:" + n)
        if not p.requires_grad:
            frozen.add("p:" + n)
        if p.grad is not None:
            names.setdefault(id(p.grad), "g:" + n)
    for n, b in model_chunk.named_buffers(remove_duplicate=False):
        names.setdefault(id(b), "b:" + n)
        frozen.add("b:" + n)
    return names, frozen


def _bytes_digest(t: torch.Tensor) -> int:
    flat = t.detach().contiguous().view(-1).view(torch.uint8)
    w = flat[: flat.numel() // 8 * 8].view(torch.int64)
    return int(w.sum()) ^ int(flat[flat.numel() // 8 * 8 :].long().sum()) if flat.numel() else 0


def _offload_module_persistent(model_chunk, *, pin: bool, verify: bool) -> dict[str, int]:
    cache = model_chunk.__dict__.setdefault("_mlite_host_cache", {})
    names, frozen = _persist_names(model_chunk)
    stats = {"skipped_bytes": 0, "copied_bytes": 0, "verify_fail": 0}

    def fn(t: torch.Tensor) -> torch.Tensor:
        if t.device.type != "cuda":
            return t
        key = names.get(id(t))
        if key is None:
            stats["copied_bytes"] += t.numel() * t.element_size()
            return t.to("cpu")
        e = cache.get(key)
        same_layout = e is not None and e["host"].shape == t.shape and e["host"].dtype == t.dtype
        if (
            key in frozen
            and same_layout
            and e.get("ptr") == t.data_ptr()
            and e.get("version") == t._version
        ):
            if verify and _bytes_digest(t) != e.get("digest"):
                stats["verify_fail"] += 1
                e["host"].copy_(t)
            stats["skipped_bytes"] += t.numel() * t.element_size()
            return e["host"]
        if not same_layout:
            host = torch.empty(t.shape, dtype=t.dtype, device="cpu", pin_memory=pin and key not in frozen)
            e = cache[key] = {"host": host}
        e["host"].copy_(t, non_blocking=key not in frozen)
        stats["copied_bytes"] += t.numel() * t.element_size()
        return e["host"]

    model_chunk._apply(fn)
    torch.cuda.synchronize()
    if stats["verify_fail"]:
        skipped, copied, bad = stats["skipped_bytes"] / 2**30, stats["copied_bytes"] / 2**30, stats["verify_fail"]
        print(
            f"mlite offload_persist skipped_gib={skipped:.2f} copied_gib={copied:.2f} verify_fail={bad}",
            flush=True,
        )
    return stats


def _load_module_persistent(model_chunk, *, verify: bool) -> None:
    cache = model_chunk.__dict__.setdefault("_mlite_host_cache", {})
    model_chunk.to("cuda", non_blocking=True)
    torch.cuda.synchronize()
    names, frozen = _persist_names(model_chunk)
    for n, p in model_chunk.named_parameters(remove_duplicate=False):
        key = "p:" + n
        if key in cache:
            cache[key].update(ptr=p.data_ptr(), version=p._version)
            if verify and key in frozen:
                cache[key]["digest"] = _bytes_digest(p)
    for n, b in model_chunk.named_buffers(remove_duplicate=False):
        key = "b:" + n
        if key in cache:
            cache[key].update(ptr=b.data_ptr(), version=b._version)
            if verify:
                cache[key]["digest"] = _bytes_digest(b)


def offload_model_to_cpu(
    model_list: list,
    *,
    persist: bool = False,
    persist_pin: bool = True,
    persist_verify: bool = False,
) -> None:
    """Offload DDP model to CPU via buffer-resize (zero-copy on GPU side).

    Other modules (not FSDP2) keep persistent host copies with ``persist`` (see above)."""
    for model_chunk in model_list:
        if _is_megatron_ddp(model_chunk):
            all_buffers = [model_chunk.buffers, model_chunk.expert_parallel_buffers]
            for buffers in all_buffers:
                for buffer in buffers:
                    if buffer.param_data.storage().size() > 0:
                        buffer.param_data.cpu_data = buffer.param_data.data.cpu().pin_memory()
                        buffer.param_data_size = buffer.param_data.storage().size()
                        buffer.param_data.storage().resize_(0)

                    if buffer.grad_data.storage().size() > 0:
                        buffer.grad_data_size = buffer.grad_data.storage().size()
                        buffer.grad_data.storage().resize_(0)

            for param in model_chunk.module.parameters():
                if not param.requires_grad and param.device.type != "cpu":
                    param.data = param.data.to("cpu", non_blocking=True)
        else:
            _reshard_fsdp2_modules(model_chunk)
            if persist and not _has_fsdp2(model_chunk):
                _offload_module_persistent(model_chunk, pin=persist_pin, verify=persist_verify)
            else:
                model_chunk.to("cpu")


def load_model_to_gpu(
    model_list: list,
    load_grad: bool = True,
    *,
    persist: bool = False,
    persist_verify: bool = False,
) -> None:
    """Load DDP model back to GPU from pinned CPU copy (``persist``: as offloaded)."""
    for model_chunk in model_list:
        if _is_megatron_ddp(model_chunk):
            all_buffers = [model_chunk.buffers, model_chunk.expert_parallel_buffers]
            for buffers in all_buffers:
                for buffer in buffers:
                    if load_grad and hasattr(buffer, "grad_data_size"):
                        current_size = buffer.grad_data.storage().size()
                        if current_size == 0 or current_size == buffer.grad_data_size:
                            buffer.grad_data.storage().resize_(buffer.grad_data_size)
                            buffer.grad_data.zero_()
                        else:
                            buffer.grad_data.zero_()

                    if buffer.param_data.storage().size() == 0:
                        buffer.param_data.storage().resize_(buffer.param_data_size)
                        buffer.param_data.copy_(buffer.param_data.cpu_data, non_blocking=True)

            for param in model_chunk.module.parameters():
                if not param.requires_grad and param.device.type == "cpu":
                    param.data = param.data.to("cuda", non_blocking=True)
        else:
            if persist and not _has_fsdp2(model_chunk):
                _load_module_persistent(model_chunk, verify=persist_verify)
            else:
                model_chunk.to("cuda")


# ======================================================================
# Optimizer offload / load — offload_megatron_optimizer / load_megatron_optimizer
# ======================================================================


def offload_optimizer(optimizer, *, persist: bool = False) -> None:
    """Offload optimizer states to CPU (``persist``: pinned host tensors kept across offloads)."""
    try:
        from megatron.core.optimizer import ChainedOptimizer
    except ImportError:
        ChainedOptimizer = ()

    for _opt in _iter_opts(optimizer, ChainedOptimizer):
        state_owner = getattr(_opt, "optimizer", _opt)
        if state_owner is not None:
            hdo = state_owner
            if all(
                hasattr(hdo, a) for a in ("sub_optimizers", "inner_param_to_orig_param", "state")
            ):
                for sub_opt in hdo.sub_optimizers:
                    for param, state in sub_opt.state.items():
                        for k, v in state.items():
                            if not isinstance(v, torch.Tensor):
                                continue
                            orig_param = hdo.inner_param_to_orig_param.get(param, param)
                            hdo.state[orig_param][k] = state[k] = v.to("cpu")
            elif persist:
                host = hdo.__dict__.setdefault("_mlite_host_state", {})
                for pid, state in enumerate(hdo.state.values()):
                    for key, value in state.items():
                        if isinstance(value, torch.Tensor) and value.device.type == "cuda":
                            h = host.get((pid, key))
                            if h is None or h.shape != value.shape or h.dtype != value.dtype:
                                h = host[(pid, key)] = torch.empty(
                                    value.shape, dtype=value.dtype, device="cpu", pin_memory=True
                                )
                            h.copy_(value, non_blocking=True)
                            state[key] = h
                torch.cuda.synchronize()
            else:
                for state in hdo.state.values():
                    for key, value in state.items():
                        if isinstance(value, torch.Tensor):
                            state[key] = value.to("cpu", non_blocking=True)

    gc.collect()
    torch.cuda.empty_cache()


def load_optimizer(optimizer) -> None:
    """Load optimizer states back to GPU."""
    try:
        from megatron.core.optimizer import ChainedOptimizer
    except ImportError:
        ChainedOptimizer = ()

    for _opt in _iter_opts(optimizer, ChainedOptimizer):
        state_owner = getattr(_opt, "optimizer", _opt)
        if state_owner is not None:
            if hasattr(state_owner, "_move_new_state_to_right_device"):
                state_owner._move_new_state_to_right_device()
            else:
                for state in state_owner.state.values():
                    for key, value in state.items():
                        if isinstance(value, torch.Tensor):
                            state[key] = value.to("cuda", non_blocking=True)

    gc.collect()
    torch.cuda.empty_cache()


def _iter_opts(optimizer, chained_cls):
    if isinstance(optimizer, chained_cls):
        return optimizer.chained_optimizers
    return [optimizer]


# ======================================================================
# Checkpoint helpers
# ======================================================================


def build_sharded_state_dict(
    model_list: list, optimizer: Any = None, lr_scheduler: Any = None
) -> dict[str, Any]:
    """Build sharded state dict for model + optimizer + lr_scheduler.

    Uses ``model0.`` / ``model1.`` prefix for VPP (multiple model chunks).
    """
    sharded_state_dict: dict[str, Any] = {}

    for i, model_chunk in enumerate(model_list):
        prefix = f"model{i}." if len(model_list) > 1 else "model."
        chunk_sd = model_chunk.sharded_state_dict(prefix=prefix)
        sharded_state_dict.update(chunk_sd)

    if optimizer is not None:
        opt_sd = optimizer.sharded_state_dict(model_sharded_state_dict=sharded_state_dict)
        sharded_state_dict.update(opt_sd)

    if lr_scheduler is not None:
        sharded_state_dict["lr_scheduler"] = lr_scheduler.state_dict()

    return sharded_state_dict
