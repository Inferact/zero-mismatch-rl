# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""
DCP (Distributed Checkpoint) framework for training checkpoints.

Model-agnostic: takes a placement function to describe how each parameter is sharded.
HF weight loading/saving is model-specific and lives in models/<name>/checkpoint.py.
"""

from __future__ import annotations

import contextlib
import os
import random
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch  # pyright: ignore[reportMissingImports]
import torch.distributed as dist  # pyright: ignore[reportMissingImports]
import torch.distributed.checkpoint as dcp  # pyright: ignore[reportMissingImports]
import torch.nn as nn  # pyright: ignore[reportMissingImports]
from torch.distributed.device_mesh import DeviceMesh  # pyright: ignore[reportMissingImports]
from torch.distributed.tensor import DTensor  # pyright: ignore[reportMissingImports]

from megatron.lite.primitive.parallel import ParallelState
from megatron.lite.primitive.protocols import (
    ExpertClassifierFn,
    PlacementFn,
    default_expert_classifier,
    default_placement_fn,
)


def save_training_checkpoint(
    model: nn.Module | Iterable[nn.Module],
    optimizer,
    step: int | str,
    path: str | None = None,
    config=None,
    ps: ParallelState | None = None,
    get_placements: PlacementFn = default_placement_fn,
    is_expert: ExpertClassifierFn = default_expert_classifier,
    *,
    use_dcp: bool | None = True,
    save_rng: bool = True,
    save_model: bool = True,
    save_optimizer: bool = True,
    extra_state: Mapping[str, Any] | None = None,
    trainable_only: bool = False,
) -> None:
    """Save training checkpoint using DTensor + DCP for automatic resharding.

    ``extra_state`` maps file names to picklable states of the same step (an LR scheduler, for
    example).  They are written into the step directory on every node before it is published, so a
    published checkpoint always carries them; ``checkpoint_extra_state_path`` finds them again.
    ``trainable_only``: a rank-local checkpoint (node-local disks) holds only the parameters that
    require grad (see ``_rank_local_params``).
    """
    if path is None and isinstance(step, str):
        path = step
        step = 0
    if path is None:
        raise ValueError("checkpoint path is required")
    step = int(step)
    if use_dcp is None:
        use_dcp = True
    if not use_dcp:
        if extra_state is not None:
            raise ValueError("extra_state needs a DCP checkpoint (use_dcp=True)")
        _save_local_training_checkpoint(model, optimizer, step, path, save_rng=save_rng)
        return
    if _supports_dist_opt_distckpt(model, optimizer):
        ckpt_path = os.path.join(path, f"step_{step}")
        os.makedirs(ckpt_path, exist_ok=True)
        _save_dist_opt_checkpoint(
            model, optimizer, step, ckpt_path, save_model=save_model, save_optimizer=save_optimizer
        )
        if save_rng:
            _save_rng_sidecar(ckpt_path)
        _save_extra_state(ckpt_path, extra_state)
        log_rank0(f"Saved dist_opt checkpoint at step {step} to {ckpt_path}")
        return
    if config is None or ps is None:
        raise ValueError("DCP checkpointing requires config and ParallelState.")
    if not isinstance(model, nn.Module):
        raise TypeError("DCP checkpointing currently expects a single nn.Module.")
    dense_mesh, expert_mesh = _build_meshes(config)
    state_dict: dict = {"step": step}
    # Pipeline stages own DIFFERENT parameters but their local layers re-index
    # to 0..N, so without a per-stage prefix the DCP FQNs collide across pp ranks
    # (stage0 layer0 and stage1 layer1 both -> "model.0.layers.0..."), corrupting
    # the round-trip. Mirror distckpt's pp-aware keying: disjoint keyspace per stage.
    model_prefix = f"model_pp{ps.pp_rank}" if ps.pp_size > 1 else "model"

    if save_model:
        for name, param in model.named_parameters():
            placements = get_placements(name)
            mesh = expert_mesh if is_expert(name) else dense_mesh
            state_dict[f"{model_prefix}.{name}"] = _dcp_tensor_from_param(param, mesh, placements)

    ckpt_path = os.path.join(path, f"step_{step}")
    # Written into step_N.partial, every shard read back and verified, then renamed (see
    # _save_dcp_verified / _publish_partial below).
    stage = _begin_partial(ckpt_path)
    if _is_shared_dir(stage):
        _save_dcp_verified(state_dict, stage)
    else:
        # Node-local checkpoint disks: a global DCP save stores each replicated
        # tensor once, in one rank's file on one node, so the other nodes cannot load it.  Each
        # rank saves its own local tensors instead (exact-layout resume only, no resharding).
        local = {"step": step}
        if save_model:
            for name, param in _rank_local_params(model, trainable_only):
                local[f"{model_prefix}.{name}"] = _to_local_tensor(param).detach()
        _save_rank_local_dcp(local, stage, ps, trainable_only=trainable_only)

    def write_states() -> None:
        if save_optimizer:
            _save_optimizer_checkpoint(optimizer, stage)
        if save_rng:
            _save_rng_sidecar(stage)

    _on_every_rank(write_states, f"{stage}: optimizer / RNG state not written on every rank")
    _save_extra_state(stage, _with_generation(extra_state))
    _publish_partial(stage, ckpt_path)
    log_rank0(f"Saved training checkpoint at step {step} to {ckpt_path}")


def load_training_checkpoint(
    model: nn.Module | Iterable[nn.Module],
    optimizer,
    path: str,
    config=None,
    ps: ParallelState | None = None,
    get_placements: PlacementFn = default_placement_fn,
    is_expert: ExpertClassifierFn = default_expert_classifier,
    *,
    use_dcp: bool | None = True,
    load_rng: bool = True,
    load_parameter_state_update_legacy_format: bool = False,
    load_model: bool = True,
    load_optimizer: bool = True,
    trainable_only: bool = False,
) -> int:
    """Load training checkpoint with automatic resharding across different parallel configs.

    ``trainable_only`` must be the value a rank-local checkpoint was saved with."""
    if use_dcp is None:
        use_dcp = True
    if not use_dcp:
        return _load_local_training_checkpoint(
            model,
            optimizer,
            path,
            load_rng=load_rng,
            load_parameter_state_update_legacy_format=load_parameter_state_update_legacy_format,
        )
    ckpt_path = _on_every_rank(
        lambda: _resolve_step_checkpoint_path(path), f"checkpoint {path} not resolvable on every rank"
    )
    _check_same_checkpoint(ckpt_path)
    if _supports_dist_opt_distckpt(model, optimizer):
        step = _load_dist_opt_checkpoint(
            model, optimizer, ckpt_path, load_model=load_model, load_optimizer=load_optimizer
        )
        if load_rng:
            _load_rng_sidecar(ckpt_path)
        log_rank0(f"Loaded dist_opt checkpoint from {path} at step {step}")
        return step
    if config is None or ps is None:
        raise ValueError("DCP checkpointing requires config and ParallelState.")
    if not isinstance(model, nn.Module):
        raise TypeError("DCP checkpointing currently expects a single nn.Module.")
    dense_mesh, expert_mesh = _build_meshes(config)

    state_dict: dict = {"step": 0}
    # Same pp-aware keying as save (see save_training_checkpoint): per-stage
    # disjoint keyspace so pp ranks don't read each other's colliding FQNs.
    model_prefix = f"model_pp{ps.pp_rank}" if ps.pp_size > 1 else "model"

    rank_local = _on_every_rank(
        lambda: os.path.isdir(ckpt_path) and any(d.startswith("rank_") for d in os.listdir(ckpt_path)),
        f"{ckpt_path} not readable on every rank",
    )
    if _any_rank(rank_local):

        def load_rank_local() -> int:
            step = _load_rank_local_dcp(
                model, model_prefix, ckpt_path, ps, load_model=load_model, trainable_only=trainable_only
            )
            if load_optimizer:
                _load_optimizer_checkpoint(optimizer, ckpt_path)
            if load_rng:
                _load_rng_sidecar(ckpt_path)
            return step

        step = _on_every_rank(load_rank_local, f"rank-local checkpoint {ckpt_path} not loadable on every rank")
        log_rank0(f"Loaded rank-local training checkpoint from {path} at step {step}")
        return step

    if load_model:
        for name, param in model.named_parameters():
            placements = get_placements(name)
            mesh = expert_mesh if is_expert(name) else dense_mesh
            state_dict[f"{model_prefix}.{name}"] = _empty_dcp_tensor_like_param(
                param, mesh, placements
            )

    dcp.load(state_dict, checkpoint_id=ckpt_path)

    if load_model:
        for name, param in model.named_parameters():
            key = f"{model_prefix}.{name}"
            if key in state_dict:
                t = state_dict[key]
                with torch.no_grad():
                    _copy_tensor_(param, t)

    def load_states() -> None:
        if load_optimizer:
            _load_optimizer_checkpoint(optimizer, ckpt_path)
        if load_rng:
            _load_rng_sidecar(ckpt_path)

    _on_every_rank(load_states, f"checkpoint {ckpt_path}: optimizer / RNG state not loadable on every rank")
    step = state_dict.get("step", 0)
    log_rank0(f"Loaded training checkpoint from {path} at step {step}")
    return step


def _resolve_step_checkpoint_path(path: str) -> str:
    name = os.path.basename(os.path.normpath(path))
    if name.startswith("step_"):
        if not name[5:].isdigit():
            raise ValueError(
                f"{path} is not a published checkpoint: only step_<N> directories are "
                "(an interrupted save leaves step_<N>.partial)"
            )
        return path

    # step_N only: a step_N.partial left by an interrupted save is not a checkpoint.
    step_dirs = sorted(
        [d for d in os.listdir(path) if d.startswith("step_") and d[5:].isdigit()],
        key=lambda d: int(d.split("_")[1]),
    )
    if step_dirs:
        return os.path.join(path, step_dirs[-1])
    return path


def checkpoint_extra_state_path(path: str, name: str) -> str:
    """Where ``save_training_checkpoint(..., extra_state={name: ...})`` put ``name`` for the checkpoint
    that ``load_training_checkpoint(path)`` loads."""
    return os.path.join(_resolve_step_checkpoint_path(path), name)


def _supports_dist_opt_distckpt(model: nn.Module | Iterable[nn.Module], optimizer) -> bool:
    try:
        from megatron.lite.primitive.ckpt.distckpt import supports_dist_opt_distckpt
    except ModuleNotFoundError as exc:
        if exc.name != "megatron.core":
            raise
        return False

    return supports_dist_opt_distckpt(model, optimizer)


def _save_dist_opt_checkpoint(
    model: nn.Module | Iterable[nn.Module],
    optimizer,
    step: int,
    path: str,
    *,
    save_model: bool,
    save_optimizer: bool,
) -> None:
    from megatron.lite.primitive.ckpt.distckpt import save_dist_opt_checkpoint

    save_dist_opt_checkpoint(
        model, optimizer, step, path, save_model=save_model, save_optimizer=save_optimizer
    )


def _load_dist_opt_checkpoint(
    model: nn.Module | Iterable[nn.Module],
    optimizer,
    path: str,
    *,
    load_model: bool,
    load_optimizer: bool,
) -> int:
    from megatron.lite.primitive.ckpt.distckpt import load_dist_opt_checkpoint

    return load_dist_opt_checkpoint(
        model, optimizer, path, load_model=load_model, load_optimizer=load_optimizer
    )


def _optimizer_checkpoint_path(path: str) -> str:
    rank = dist.get_rank() if dist.is_initialized() else 0
    return os.path.join(path, f"optimizer_rank_{rank}.pt")


def _save_optimizer_checkpoint(optimizer, path: str) -> None:
    if optimizer is None:
        log_rank0("Skipping optimizer checkpoint save because optimizer is None")
        return
    state_dict_fn = getattr(optimizer, "state_dict", None)
    if not callable(state_dict_fn):
        raise TypeError(f"Optimizer {type(optimizer).__name__} does not provide state_dict().")
    target = _optimizer_checkpoint_path(path)
    state = state_dict_fn()
    for attempt in range(3):
        tmp = f"{target}.tmp{attempt}"
        torch.save(state, tmp)
        bad = _zip_checkpoint_error(tmp)
        if bad is None:
            os.replace(tmp, target)
            return
        log_rank0(f"optimizer checkpoint {tmp} failed verification ({bad}); rewriting")
        os.remove(tmp)
    raise RuntimeError(f"optimizer checkpoint {target} failed verification three times")


def _zip_checkpoint_error(file_path: str) -> str | None:
    """Return why a torch.save zip file is unreadable, or None when every member's CRC checks out.

    A rank's optimizer file on Lustre once came back with one 84 MiB record written twice, which shifted
    the central directory and made torch.load fail on resume.
    """
    import zipfile

    try:
        with zipfile.ZipFile(file_path) as zf:
            if any(info.header_offset != 0 for info in zf.infolist()[:1]):
                return "first member not at offset 0"
            bad = zf.testzip()
            return None if bad is None else f"crc mismatch in {bad}"
    except (zipfile.BadZipFile, OSError) as exc:
        return repr(exc)


def _load_optimizer_checkpoint(optimizer, path: str) -> None:
    if optimizer is None:
        log_rank0("Skipping optimizer checkpoint load because optimizer is None")
        return
    ckpt_path = _optimizer_checkpoint_path(path)
    if not os.path.exists(ckpt_path):
        log_rank0(f"No optimizer checkpoint found at {ckpt_path}; loading model state only")
        return
    load_state_dict_fn = getattr(optimizer, "load_state_dict", None)
    if not callable(load_state_dict_fn):
        raise TypeError(f"Optimizer {type(optimizer).__name__} does not provide load_state_dict().")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    load_state_dict_fn(state)


def _model_chunks(model: nn.Module | Iterable[nn.Module]) -> list[nn.Module]:
    if isinstance(model, nn.Module):
        return [model]
    chunks = list(model)
    if not all(isinstance(chunk, nn.Module) for chunk in chunks):
        raise TypeError("checkpoint model chunks must be nn.Module instances.")
    return chunks


def _to_local_tensor(tensor: Any) -> torch.Tensor:
    local_tensor = getattr(tensor, "_local_tensor", None)
    if isinstance(local_tensor, torch.Tensor):
        return local_tensor
    to_local = getattr(tensor, "to_local", None)
    if callable(to_local):
        return to_local()
    return tensor


def _is_dtensor_like(tensor: Any) -> bool:
    return (
        callable(getattr(tensor, "to_local", None))
        and hasattr(tensor, "device_mesh")
        and hasattr(tensor, "placements")
    )


def _dcp_tensor_from_param(param: torch.Tensor, mesh: DeviceMesh, placements: list) -> DTensor:
    if _is_dtensor_like(param):
        return _dtensor_from_dtensor_like_param(param, _to_local_tensor(param).detach())
    return DTensor.from_local(_to_local_tensor(param).detach(), mesh, placements)


def _empty_dcp_tensor_like_param(
    param: torch.Tensor, mesh: DeviceMesh, placements: list
) -> DTensor:
    if _is_dtensor_like(param):
        return _dtensor_from_dtensor_like_param(param, torch.empty_like(_to_local_tensor(param)))
    return DTensor.from_local(torch.empty_like(_to_local_tensor(param)), mesh, placements)


def _dtensor_from_dtensor_like_param(param: torch.Tensor, local_tensor: torch.Tensor) -> DTensor:
    return DTensor.from_local(
        local_tensor,
        param.device_mesh,
        param.placements,
        shape=tuple(param.shape),
        stride=tuple(param.stride()),
    )


def _copy_tensor_(target: torch.Tensor, src: torch.Tensor) -> None:
    local_target = _to_local_tensor(target)
    local_src = _to_local_tensor(src).to(device=local_target.device, dtype=local_target.dtype)
    if isinstance(local_target, torch.Tensor) and local_target is not target:
        local_target.copy_(local_src)
    else:
        target.copy_(local_src)


def _chunk_tensor_state(module: nn.Module) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for name, param in module.named_parameters():
        state[f"param.{name}"] = _to_local_tensor(param.detach()).cpu().clone()
    for name, buffer in module.named_buffers():
        state[f"buffer.{name}"] = _to_local_tensor(buffer.detach()).cpu().clone()
    return state


def _load_chunk_tensor_state(module: nn.Module, state: dict[str, torch.Tensor]) -> None:
    params = dict(module.named_parameters())
    buffers = dict(module.named_buffers())
    missing: list[str] = []
    for key, src in state.items():
        kind, name = key.split(".", 1)
        if kind == "param" and name in params:
            with torch.no_grad():
                _copy_tensor_(params[name], src)
        elif kind == "buffer" and name in buffers:
            with torch.no_grad():
                _copy_tensor_(buffers[name], src)
        else:
            missing.append(key)
    if missing:
        raise RuntimeError(f"checkpoint contains unknown tensor keys: {missing}")


def _local_checkpoint_file(path: str | os.PathLike[str]) -> Path:
    ckpt_path = Path(path)
    if ckpt_path.is_dir() or ckpt_path.suffix == "":
        if _is_distributed_checkpoint_ranked():
            return ckpt_path / f"training_state_{_rank_suffix()}.pt"
        return ckpt_path / "training_state.pt"
    return ckpt_path


def _local_optimizer_parameter_state_file(ckpt_file: Path) -> Path:
    return ckpt_file.with_name(f"{ckpt_file.stem}.optimizer_parameter_state{ckpt_file.suffix}")


def _rank_suffix() -> str:
    if dist.is_available() and dist.is_initialized():
        return f"rank_{dist.get_rank():05d}"
    return "rank_00000"


def _is_distributed_checkpoint_ranked() -> bool:
    return dist.is_available() and dist.is_initialized()


def _rng_sidecar_file(path: str | os.PathLike[str]) -> Path:
    return Path(path) / f"rng_state_{_rank_suffix()}.pt"


def _cpu_clone(tensor: torch.Tensor | None) -> torch.Tensor | None:
    if tensor is None:
        return None
    return tensor.detach().cpu().clone()


def _get_cuda_rng_state() -> torch.Tensor | None:
    if not torch.cuda.is_initialized():
        return None
    return _cpu_clone(torch.cuda.get_rng_state())


def _get_cuda_rng_tracker_states() -> dict[str, torch.Tensor]:
    if not torch.cuda.is_initialized():
        return {}

    from megatron.core import tensor_parallel

    states = tensor_parallel.get_cuda_rng_tracker().get_states()
    return {name: _cpu_clone(state) for name, state in states.items() if state is not None}


def _get_rng_state() -> dict[str, Any]:
    return {
        "random_rng_state": random.getstate(),
        "np_rng_state": np.random.get_state(),
        "torch_rng_state": _cpu_clone(torch.get_rng_state()),
        "cuda_rng_state": _get_cuda_rng_state(),
        "rng_tracker_states": _get_cuda_rng_tracker_states(),
    }


def _restore_cuda_rng_tracker_states(states: dict[str, torch.Tensor]) -> None:
    if not states or not torch.cuda.is_initialized():
        return
    try:
        from megatron.core import tensor_parallel

        tracker = tensor_parallel.get_cuda_rng_tracker()
        graph_safe = tensor_parallel.is_graph_safe_cuda_rng_tracker(tracker)
        restored = {
            name: tensor_parallel.convert_cuda_rng_state(state, to_graphable=graph_safe)
            for name, state in states.items()
        }
        tracker.set_states(restored)
    except Exception as exc:
        raise RuntimeError("Failed to restore Megatron tensor-parallel RNG tracker state.") from exc


def _restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    random.setstate(state["random_rng_state"])
    np.random.set_state(state["np_rng_state"])
    torch.set_rng_state(state["torch_rng_state"])
    cuda_rng_state = state.get("cuda_rng_state")
    if cuda_rng_state is not None and torch.cuda.is_initialized():
        torch.cuda.set_rng_state(cuda_rng_state)
    _restore_cuda_rng_tracker_states(state.get("rng_tracker_states", {}))


def _save_rng_sidecar(path: str | os.PathLike[str]) -> None:
    rng_file = _rng_sidecar_file(path)
    rng_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_get_rng_state(), rng_file)


def _load_rng_sidecar(path: str | os.PathLike[str]) -> None:
    rng_file = _rng_sidecar_file(path)
    if not rng_file.exists():
        log_rank0(f"RNG sidecar not found at {rng_file}; skipping RNG restore.")
        return
    _restore_rng_state(torch.load(rng_file, map_location="cpu", weights_only=False))


def _save_local_training_checkpoint(
    model: nn.Module | Iterable[nn.Module],
    optimizer,
    step: int,
    path: str,
    *,
    save_rng: bool = True,
) -> None:
    chunks = _model_chunks(model)
    ckpt_file = _local_checkpoint_file(path)
    ckpt_file.parent.mkdir(parents=True, exist_ok=True)
    save_parameter_state = getattr(optimizer, "save_parameter_state", None)
    optimizer_parameter_state_file = (
        _local_optimizer_parameter_state_file(ckpt_file) if callable(save_parameter_state) else None
    )
    state = {
        "format": "megatron_lite.local_training.v1",
        "step": int(step),
        "model": [_chunk_tensor_state(chunk) for chunk in chunks],
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "optimizer_parameter_state": (
            optimizer_parameter_state_file.name
            if optimizer_parameter_state_file is not None
            else None
        ),
        "rng_state": _get_rng_state() if save_rng else None,
    }
    torch.save(state, ckpt_file)
    if optimizer_parameter_state_file is not None:
        save_parameter_state(str(optimizer_parameter_state_file))
    log_rank0(f"Saved local training checkpoint at step {step} to {ckpt_file}")


def _load_local_training_checkpoint(
    model: nn.Module | Iterable[nn.Module],
    optimizer,
    path: str,
    *,
    load_rng: bool = True,
    load_parameter_state_update_legacy_format: bool = False,
) -> int:
    ckpt_file = _local_checkpoint_file(path)
    state = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    if state.get("format") != "megatron_lite.local_training.v1":
        raise RuntimeError(f"Unsupported local checkpoint format in {ckpt_file}")
    chunks = _model_chunks(model)
    chunk_states = state.get("model")
    if not isinstance(chunk_states, list) or len(chunk_states) != len(chunks):
        raise RuntimeError("Checkpoint model chunk count does not match target model.")
    for chunk, chunk_state in zip(chunks, chunk_states, strict=True):
        _load_chunk_tensor_state(chunk, chunk_state)
    if optimizer is not None and state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])
        parameter_state_name = state.get("optimizer_parameter_state")
        load_parameter_state = getattr(optimizer, "load_parameter_state", None)
        if parameter_state_name is not None and callable(load_parameter_state):
            load_parameter_state(
                str(ckpt_file.with_name(parameter_state_name)),
                update_legacy_format=load_parameter_state_update_legacy_format,
            )
        else:
            reload_model_params = getattr(optimizer, "reload_model_params", None)
            if callable(reload_model_params):
                reload_model_params()
    if load_rng:
        _restore_rng_state(state.get("rng_state"))
    step = int(state.get("step", 0))
    log_rank0(f"Loaded local training checkpoint from {ckpt_file} at step {step}")
    return step


# ----------------------------------------------------------------------
# Verified, atomically published DCP training checkpoints
# ----------------------------------------------------------------------
#
# A DCP save writes every rank's model shard straight into the step directory, so a short or
# corrupt shard (Lustre once returned a model shard that torch.save had cut short)
# fails the whole job or, worse, is only found on resume.  The step is therefore written into
# ``step_N.partial``: each rank reads its own shard entries back and compares them bitwise with
# the tensors it saved (rewriting the checkpoint up to three times on any failure), the optimizer
# file is CRC-checked as before, and only then is ``step_N.partial`` renamed to ``step_N``.
# ``.metadata`` is written by the coordinator only.  A repeated save of a published step moves the
# old ``step_N`` aside (``step_N.replaced``) and deletes it only once the new one is in place.

_PARTIAL_SUFFIX = ".partial"
_REPLACED_SUFFIX = ".replaced"
_GENERATION = "generation.pt"


def _with_generation(extra_state: Mapping[str, Any] | None) -> dict:
    """``extra_state`` plus a token unique to this save, the same on every rank."""
    token = [os.urandom(8).hex()]
    if _dist_on():
        dist.broadcast_object_list(token, src=0)
    return {**(extra_state or {}), _GENERATION: token[0]}


def _check_same_checkpoint(ckpt_path: str) -> None:
    """Every rank resolved the same step directory, written by the same save: a re-save that failed to
    publish on one node leaves that node with the older copy of the step."""
    if not _dist_on():
        return

    def read() -> tuple:
        token = os.path.join(ckpt_path, _GENERATION)
        return os.path.basename(os.path.normpath(ckpt_path)), torch.load(token) if os.path.exists(token) else None

    mine = _on_every_rank(read, f"checkpoint {ckpt_path} not readable on every rank")
    seen = [None] * dist.get_world_size()
    dist.all_gather_object(seen, mine)
    if len(set(seen)) > 1:
        raise RuntimeError(
            f"ranks resolved different checkpoints (step directory, save token): {sorted(set(seen), key=str)}"
        )


def _dist_on() -> bool:
    return dist.is_available() and dist.is_initialized()


def _any_rank(flag: bool) -> bool:
    if not _dist_on():
        return flag
    device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else None
    t = torch.tensor([1 if flag else 0], dtype=torch.int32, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return bool(t.item())


def _on_every_rank(fn, what: str):
    result, error = None, None
    try:
        result = fn()
    except (Exception, dcp.CheckpointException) as exc:
        error = exc
    if _any_rank(error is not None):
        raise RuntimeError(what) from error
    return result


def _bitwise_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.dtype != b.dtype or tuple(a.shape) != tuple(b.shape):
        return False
    a = a.detach().contiguous().cpu()
    b = b.detach().contiguous().cpu()
    if a.numel() == 0:
        return True
    return torch.equal(a.view(torch.uint8), b.view(torch.uint8))


def _dcp_own_shard_error(metadata, state_dict: dict, ckpt_path: str, rank: int | None = None) -> str | None:
    """Read back every storage entry this rank wrote and compare it with the saved value.

    Returns why the shard is bad, or None.  Entries are found through the save's own metadata
    (``__<rank>_*.distcp`` files, offset, length); tensors are compared byte for byte with the
    local shard that was handed to ``dcp.save``.
    """
    import io

    if rank is None:
        rank = dist.get_rank() if _dist_on() else 0
    prefix = f"__{rank}_"
    storage = getattr(metadata, "storage_data", None)
    if not isinstance(storage, dict):
        return "save returned no storage metadata"
    handles: dict[str, Any] = {}
    try:
        for index, info in storage.items():
            rel = getattr(info, "relative_path", "")
            if not os.path.basename(rel).startswith(prefix):
                continue
            if getattr(info, "transform_descriptors", None):
                return f"{index.fqn}: unexpected storage transforms {info.transform_descriptors}"
            fh = handles.get(rel)
            if fh is None:
                fh = handles[rel] = open(os.path.join(ckpt_path, rel), "rb")  # noqa: SIM115
            fh.seek(info.offset)
            blob = fh.read(info.length)
            if len(blob) != info.length:
                return f"{rel}: {index.fqn} short read {len(blob)} of {info.length} bytes"
            expected = state_dict.get(index.fqn)
            if not isinstance(expected, torch.Tensor):
                continue
            try:
                got = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=True)
            except Exception as exc:  # noqa: BLE001
                return f"{rel}: {index.fqn} unreadable ({exc!r})"[:500]
            if not _bitwise_equal(got, _to_local_tensor(expected)):
                return f"{rel}: {index.fqn} differs from the saved tensor"
    finally:
        for fh in handles.values():
            fh.close()
    return None


# Rank-local checkpoints: step_N/rank_<global rank>/ holds that rank's own DCP save (no_dist),
# step_N/layout_rank_<n>.json the parallel layout it was written with.
_RANK_LOCAL_LAYOUT = "layout.json"


def _rank_local_dir(ckpt_path: str) -> str:
    return os.path.join(ckpt_path, f"rank_{_rank_suffix()[5:]}")


def _rank_local_params(model: nn.Module, trainable_only: bool):
    """Parameters of a rank-local checkpoint; ``trainable_only``: only those that require grad.

    Frozen parameters (the base model under a LoRA adapter) are rebuilt bit for bit from the base
    checkpoint at every start, so saving them again only multiplies the node-local footprint (K31 full
    model: ~106 GiB of replicated frozen dense weights per node per checkpoint).  The flag is part of
    the recorded layout, so a checkpoint loads only with the setting it was saved with.
    """
    for name, param in model.named_parameters():
        if trainable_only and not param.requires_grad:
            continue
        yield name, param


def _layout_record(ps: ParallelState | None, trainable_only: bool) -> dict:
    import json

    rec = {"world_size": dist.get_world_size() if _dist_on() else 1, "rank": dist.get_rank() if _dist_on() else 0}
    if trainable_only:
        rec["params"] = "trainable"
    if ps is not None:
        for key in ("tp_size", "cp_size", "pp_size", "ep_size", "etp_size", "dp_size", "pp_rank", "ep_rank"):
            if hasattr(ps, key):
                rec[key] = int(getattr(ps, key))
        pp_layout = getattr(ps, "pp_layout", None)
        if pp_layout is not None:
            rec["pp_layout"] = json.loads(json.dumps(pp_layout))
    return rec


def _is_shared_dir(path: str) -> bool:
    """Whether every rank sees the same ``path`` (shared filesystem) or not (node-local disk)."""
    if not _dist_on() or dist.get_world_size() == 1:
        return True
    nonce = [None]
    if dist.get_rank() == 0:
        nonce = [f".shared_probe_{os.getpid()}_{random.getrandbits(32):08x}"]
    dist.broadcast_object_list(nonce, src=0)
    probe = os.path.join(path, nonce[0])

    def write_probe() -> None:
        if dist.get_rank() == 0:
            with open(probe, "w", encoding="utf-8") as f:
                f.write("probe\n")

    _on_every_rank(write_probe, f"{path}: shared-filesystem probe not written")
    shared = not _any_rank(not os.path.exists(probe))
    dist.barrier()
    if dist.get_rank() == 0:
        with contextlib.suppress(OSError):
            os.remove(probe)
    return shared


def _save_rank_local_dcp(local: dict, stage: str, ps: ParallelState | None, *, trainable_only: bool) -> None:
    import json

    rank_dir = _rank_local_dir(stage)
    for attempt in range(3):
        try:
            os.makedirs(rank_dir, exist_ok=True)
            metadata = dcp.save(local, checkpoint_id=rank_dir, no_dist=True)
            bad = _dcp_own_shard_error(metadata, local, rank_dir, rank=0)
        except (Exception, dcp.CheckpointException) as exc:
            bad = repr(exc)
        if bad is not None:
            rank = dist.get_rank() if _dist_on() else 0
            print(f"[megatron.lite] rank {rank}: rank-local shard verification failed ({bad}); attempt {attempt}", flush=True)
        if not _any_rank(bad is not None):
            break
    else:
        raise RuntimeError(f"rank-local checkpoint {rank_dir} failed shard verification three times")

    def write_layout() -> None:
        tmp = os.path.join(rank_dir, f"{_RANK_LOCAL_LAYOUT}.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(_layout_record(ps, trainable_only), f)
        os.replace(tmp, os.path.join(rank_dir, _RANK_LOCAL_LAYOUT))

    _on_every_rank(write_layout, f"{stage}: rank-local layout not written on every rank")


def _load_rank_local_dcp(
    model: nn.Module, model_prefix: str, ckpt_path: str, ps, *, load_model: bool, trainable_only: bool
) -> int:
    import json

    rank_dir = _rank_local_dir(ckpt_path)
    layout_file = os.path.join(rank_dir, _RANK_LOCAL_LAYOUT)
    if not os.path.exists(layout_file):
        raise FileNotFoundError(
            f"{rank_dir}: no rank-local checkpoint for this rank; rank-local checkpoints resume only on "
            "the same nodes in the same layout"
        )
    with open(layout_file, encoding="utf-8") as f:
        saved = json.load(f)
    now = _layout_record(ps, trainable_only)
    if saved != now:
        raise RuntimeError(f"rank-local checkpoint layout {saved} != current {now} (exact-layout resume only)")
    local: dict = {"step": 0}
    if load_model:
        for name, param in _rank_local_params(model, trainable_only):
            local[f"{model_prefix}.{name}"] = torch.empty_like(_to_local_tensor(param))
    dcp.load(local, checkpoint_id=rank_dir, no_dist=True)
    if load_model:
        for name, param in _rank_local_params(model, trainable_only):
            with torch.no_grad():
                _copy_tensor_(param, local[f"{model_prefix}.{name}"])
    return int(local.get("step", 0))


def _save_dcp_verified(state_dict: dict, ckpt_path: str) -> None:
    for attempt in range(3):
        metadata = dcp.save(state_dict, checkpoint_id=ckpt_path)
        try:
            bad = _dcp_own_shard_error(metadata, state_dict, ckpt_path)
        except Exception as exc:
            bad = repr(exc)
        if bad is not None:
            rank = dist.get_rank() if _dist_on() else 0
            print(f"[megatron.lite] rank {rank}: DCP shard verification failed ({bad}); attempt {attempt}", flush=True)
        if not _any_rank(bad is not None):
            return
    raise RuntimeError(f"DCP checkpoint {ckpt_path} failed shard verification three times")


def _save_extra_state(path: str, extra_state: Mapping[str, Any] | None) -> None:
    """Write ``extra_state`` into ``path`` by the first rank of every node (node-local disks each get a
    copy), each file replaced atomically: a repeated save never leaves a node with the older state.
    A failed write on any node fails the save on every rank, before anything is published."""
    if extra_state is None:
        return
    rank = dist.get_rank() if _dist_on() else 0
    writer = True
    if _dist_on():
        host = os.uname().nodename
        hosts = [None] * dist.get_world_size()
        dist.all_gather_object(hosts, host)
        writer = hosts.index(host) == rank

    def write() -> None:
        if writer:
            for name, state in extra_state.items():
                target = os.path.join(path, name)
                tmp = f"{target}.tmp{rank}"
                torch.save(state, tmp)
                os.replace(tmp, target)

    _on_every_rank(write, f"{path}: extra state not written on every node")


def _begin_partial(ckpt_path: str) -> str:
    stage = ckpt_path + _PARTIAL_SUFFIX
    _on_every_rank(lambda: os.makedirs(stage, exist_ok=True), f"{stage} not created on every rank")
    return stage


def _publish_partial(stage: str, ckpt_path: str) -> None:
    """Rename ``stage`` to ``ckpt_path`` once per filesystem (shared or node-local)."""
    import shutil

    def publish() -> None:
        if not os.path.isdir(stage):
            return
        replaced = ckpt_path + _REPLACED_SUFFIX
        if os.path.isdir(ckpt_path):
            # the published copy is deleted only after the new one has taken its place
            if os.path.isdir(replaced):
                shutil.rmtree(replaced)
            os.replace(ckpt_path, replaced)
            try:
                os.replace(stage, ckpt_path)
            except BaseException:
                os.replace(replaced, ckpt_path)
                raise
        else:
            os.replace(stage, ckpt_path)
        if os.path.isdir(replaced):
            shutil.rmtree(replaced)

    if not _dist_on():
        publish()
        return
    failed = f"checkpoint {ckpt_path} was not published from {stage}"
    dist.barrier()
    _on_every_rank(publish if dist.get_rank() == 0 else lambda: None, failed)
    host = os.uname().nodename
    hosts = [None] * dist.get_world_size()
    dist.all_gather_object(hosts, host)
    _on_every_rank(publish if dist.get_rank() == hosts.index(host) and dist.get_rank() != 0 else lambda: None, failed)
    if _any_rank(not os.path.isdir(ckpt_path) or os.path.isdir(stage)):
        raise RuntimeError(failed)


def _build_meshes(config):
    """Build separate meshes for dense and expert parameters.

    Dense mesh  [PP, DP, CP, TP]  — matches init_parallel dense decomposition.
    Expert mesh [PP, EDP, EP, ETP] — matches init_parallel expert decomposition.

    Both meshes use C-order layout so the innermost (rightmost) dimension
    corresponds to the fastest-changing rank index, consistent with
    init_parallel's rank = (...) * inner_size + inner_rank formula.
    """
    ws = dist.get_world_size()
    tp = int(config.tp or 1)
    ep = int(config.ep or 1)
    etp = max(int(config.etp or 1), 1)
    cp = max(int(config.cp or 1), 1)
    pp = max(int(config.pp or 1), 1)

    dense_dp = ws // (tp * cp * pp)
    expert_dp = ws // (etp * ep * pp)

    ranks = torch.arange(ws)
    dense_mesh = DeviceMesh("cuda", ranks.reshape(pp, dense_dp, cp, tp))
    expert_mesh = DeviceMesh("cuda", ranks.reshape(pp, expert_dp, ep, etp))
    return dense_mesh, expert_mesh


def log_rank0(msg: str) -> None:
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print(f"[megatron.lite] {msg}", flush=True)


# ======================================================================
# QKV / FC1 canonicalize for DCP (interleaved-TP ↔ canonical layout)
# ======================================================================


def _ag(data, size, group, dim=0):
    from megatron.lite.primitive.ckpt.hf_weights import allgather_concat

    return allgather_concat(data, size, group, dim)


def canonicalize_qkv_for_dcp(model, num_attention_heads, num_key_value_heads, head_dim, ps):
    """Rearrange fused QKV from interleaved-TP to canonical (Q|K|V) for DCP save."""
    if ps.tp_size <= 1:
        return
    from megatron.lite.primitive.utils import ensure_divisible

    nq = ensure_divisible(num_attention_heads, ps.tp_size) * head_dim
    nkv = ensure_divisible(num_key_value_heads, ps.tp_size) * head_dim
    for name, param in model.named_parameters():
        if "qkv" not in name or "layer_norm" in name:
            continue
        full = _ag(param.data, ps.tp_size, ps.tp_group)
        cs = param.data.shape[0]
        q, k, v = [], [], []
        for r in range(ps.tp_size):
            s = full[r * cs : (r + 1) * cs]
            q.append(s[:nq])
            k.append(s[nq : nq + nkv])
            v.append(s[nq + nkv :])
        canon = torch.cat([torch.cat(q), torch.cat(k), torch.cat(v)], dim=0)
        param.data.copy_(canon.chunk(ps.tp_size, dim=0)[ps.tp_rank])


def decanon_qkv_after_dcp(model, num_attention_heads, num_key_value_heads, head_dim, ps):
    """Reverse of canonicalize_qkv_for_dcp."""
    if ps.tp_size <= 1:
        return
    qs = num_attention_heads * head_dim
    kvs = num_key_value_heads * head_dim
    for name, param in model.named_parameters():
        if "qkv" not in name or "layer_norm" in name:
            continue
        full = _ag(param.data, ps.tp_size, ps.tp_group)
        ql = full[:qs].chunk(ps.tp_size)[ps.tp_rank]
        kl = full[qs : qs + kvs].chunk(ps.tp_size)[ps.tp_rank]
        vl = full[qs + kvs :].chunk(ps.tp_size)[ps.tp_rank]
        param.data.copy_(torch.cat([ql, kl, vl], dim=0))


def canonicalize_fc1_for_dcp(model, ps):
    """Rearrange fused gate-up FC1 from interleaved-ETP to canonical for DCP save."""
    if ps.etp_size <= 1:
        return
    for name, param in model.named_parameters():
        if "experts" not in name or "fc1" not in name:
            continue
        full = _ag(param.data, ps.etp_size, ps.etp_group)
        cs = param.data.shape[0]
        ffn = cs // 2
        g, u = [], []
        for r in range(ps.etp_size):
            s = full[r * cs : (r + 1) * cs]
            g.append(s[:ffn])
            u.append(s[ffn:])
        canon = torch.cat([torch.cat(g), torch.cat(u)], dim=0)
        param.data.copy_(canon.chunk(ps.etp_size, dim=0)[ps.etp_rank])


def decanon_fc1_after_dcp(model, ps):
    """Reverse of canonicalize_fc1_for_dcp."""
    if ps.etp_size <= 1:
        return
    for name, param in model.named_parameters():
        if "experts" not in name or "fc1" not in name:
            continue
        full = _ag(param.data, ps.etp_size, ps.etp_group)
        ffn = full.shape[0] // 2
        gl = full[:ffn].chunk(ps.etp_size)[ps.etp_rank]
        ul = full[ffn:].chunk(ps.etp_size)[ps.etp_rank]
        param.data.copy_(torch.cat([gl, ul], dim=0))


__all__ = [
    "canonicalize_fc1_for_dcp",
    "canonicalize_qkv_for_dcp",
    "checkpoint_extra_state_path",
    "decanon_fc1_after_dcp",
    "decanon_qkv_after_dcp",
    "load_training_checkpoint",
    "save_training_checkpoint",
]
