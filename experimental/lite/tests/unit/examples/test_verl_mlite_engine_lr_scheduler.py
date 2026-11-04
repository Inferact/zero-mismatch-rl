# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""LR scheduler state of verl checkpoints: saved with the step, resumed on every node or refused."""

import os
import shutil
import sys
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.checkpoint import filesystem as dcp_fs

VERL_EXAMPLE_ROOT = Path(__file__).resolve().parents[3] / "examples" / "verl"
if str(VERL_EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(VERL_EXAMPLE_ROOT))

pytestmark = pytest.mark.optional


class _Schedule:
    def __init__(self, steps=0, fail=False):
        self.num_steps = steps
        self.fail = fail

    def state_dict(self):
        if self.fail:
            raise RuntimeError("scheduler state unavailable")
        return {"num_steps": self.num_steps}

    def load_state_dict(self, state):
        self.num_steps = int(state["num_steps"])


def _engine(rank, schedule, module, optimizer=None):
    from megatron.lite.runtime.backends.mlite.config import MegatronLiteConfig

    ps = SimpleNamespace(pp_size=1, pp_rank=0)
    handle = SimpleNamespace(
        _optimizer=optimizer,
        _lr_scheduler=schedule,
        _config=MegatronLiteConfig(),
        _parallel_state=ps,
    )
    return SimpleNamespace(
        _require_initialized=lambda: None,
        checkpoint_config={"save_contents": ["model", "optimizer"]},
        _checkpoint_hooks=lambda: (lambda name: None, lambda name: False),
        is_param_offload_enabled=False,
        module=module,
        handle=handle,
        _rank=rank,
    )


def _cpu_dcp():
    """The real DCP checkpoint code with CPU tensors in place of DTensors."""
    from megatron.lite.primitive.ckpt import dcp as ckpt_dcp

    torch.cuda.is_available = lambda: False
    ckpt_dcp._build_meshes = lambda config: (None, None)
    ckpt_dcp._dcp_tensor_from_param = lambda param, mesh, placements: param.detach()
    ckpt_dcp._empty_dcp_tensor_like_param = (
        lambda param, mesh, placements: torch.empty_like(param)
    )


def _remove_scheduler_files(root):
    for dirpath, _, files in os.walk(root):
        for name in files:
            if name == "lr_scheduler.pt":
                os.remove(os.path.join(dirpath, name))


def _resume(mlite_engine, rank, node_dir):
    resumed = _Schedule()
    try:
        mlite_engine.MegatronLiteEngine.load_checkpoint(
            _engine(rank, resumed, torch.nn.Linear(4, 2)), node_dir
        )
    except (RuntimeError, ValueError, FileNotFoundError):
        return "refused"
    return str(resumed.num_steps)


def _run(rank, world, root, scenario):
    """Every rank has its own checkpoint directory and node name: node-local checkpoint disks."""
    sys.path.insert(0, str(VERL_EXAMPLE_ROOT))
    from verl_mlite.engine import mlite_engine

    _cpu_dcp()
    os.uname = lambda: SimpleNamespace(nodename=f"node{rank}")
    dist.init_process_group(
        "gloo", init_method=f"file://{root}/rendezvous", rank=rank, world_size=world
    )
    try:
        node_dir = os.path.join(root, f"node{rank}", "global_step_5", "actor")
        steps = (5, 7) if scenario == "resave-same-step" else (5,)
        for steps_done in steps:
            mlite_engine.MegatronLiteEngine.save_checkpoint(
                _engine(rank, _Schedule(steps_done), torch.nn.Linear(4, 2)),
                node_dir,
                global_step=5,
            )
        if scenario == "missing-on-one-node" and rank == 1:
            _remove_scheduler_files(node_dir)
        result = _resume(mlite_engine, rank, node_dir)
        with open(os.path.join(root, f"result{rank}"), "w", encoding="utf-8") as f:
            f.write(result)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "scenario,expected",
    [
        ("every-node-resumes", ["5", "5"]),
        ("missing-on-one-node", ["refused", "refused"]),
        ("resave-same-step", ["7", "7"]),
    ],
    ids=["every-node-resumes", "missing-on-one-node", "resave-same-step"],
)
def test_lr_scheduler_on_node_local_disks(tmp_path, scenario, expected):
    pytest.importorskip(
        "verl", reason="VERL is required for this optional example test."
    )
    mp.spawn(_run, args=(2, str(tmp_path), scenario), nprocs=2, join=True)
    assert [(tmp_path / f"result{rank}").read_text() for rank in range(2)] == expected


def _run_one_node_fails(rank, world, root, scenario):
    """One rank (its own node) fails; the process stays alive, as a Ray worker does after an exception."""
    sys.path.insert(0, str(VERL_EXAMPLE_ROOT))
    from megatron.lite.primitive.ckpt import dcp as ckpt_dcp
    from verl_mlite.engine import mlite_engine

    _cpu_dcp()
    os.uname = lambda: SimpleNamespace(nodename=f"node{rank}")
    dist.init_process_group(
        "gloo",
        init_method=f"file://{root}/rendezvous",
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=20),
    )
    disk = "shared" if scenario.startswith("shared") else f"node{rank}"
    node_dir = os.path.join(root, disk, "global_step_5", "actor")
    real_save = torch.save
    real_write = dcp_fs.FileSystemWriter.write_data
    real_read = dcp_fs.FileSystemReader.read_data
    real_readback = ckpt_dcp._dcp_own_shard_error
    real_replace = os.replace
    failing_rank = 0 if scenario == "publish-fails" else 1
    full_disk = OSError(28, "No space left on device")

    def failing_save(obj, f, *args, **kwargs):
        name = (
            "optimizer_rank" if scenario == "optimizer-write-fails" else "lr_scheduler"
        )
        if name in str(f):
            raise full_disk
        return real_save(obj, f, *args, **kwargs)

    def failing_io(*args, **kwargs):
        raise full_disk

    def failing_replace(src, dst, *args, **kwargs):
        if str(src).endswith(".partial"):
            raise full_disk
        return real_replace(src, dst, *args, **kwargs)

    module = torch.nn.Linear(4, 2)
    optimizer = None
    if scenario == "optimizer-write-fails":
        optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    outcome = "ok"
    try:
        if rank == failing_rank:
            if scenario in ("write-fails", "optimizer-write-fails"):
                torch.save = failing_save
            elif scenario == "shard-write-fails":
                dcp_fs.FileSystemWriter.write_data = failing_io
            elif scenario == "shared-readback-fails":
                ckpt_dcp._dcp_own_shard_error = failing_io
            elif scenario == "publish-fails":
                os.replace = failing_replace
        mlite_engine.MegatronLiteEngine.save_checkpoint(
            _engine(rank, _Schedule(5), module, optimizer), node_dir, global_step=5
        )
        torch.save = real_save
        if rank == 1 and scenario == "unreadable":
            for dirpath, _, files in os.walk(node_dir):
                if "lr_scheduler.pt" in files:
                    with open(os.path.join(dirpath, "lr_scheduler.pt"), "wb") as f:
                        f.write(b"truncated")
        if rank == 1 and scenario == "directory-missing":
            shutil.rmtree(os.path.join(root, "node1"))
        if rank == 1 and scenario == "layout-missing":
            for dirpath, _, files in os.walk(node_dir):
                if "layout.json" in files:
                    os.remove(os.path.join(dirpath, "layout.json"))
        if rank == 1 and scenario == "shard-read-fails":
            dcp_fs.FileSystemReader.read_data = failing_io
        mlite_engine.MegatronLiteEngine.load_checkpoint(
            _engine(rank, _Schedule(), torch.nn.Linear(4, 2)), node_dir
        )
    except Exception as exc:
        outcome = f"{type(exc).__name__}: {exc}"
        if rank == failing_rank:
            time.sleep(30)
    torch.save, os.replace = real_save, real_replace
    dcp_fs.FileSystemWriter.write_data, dcp_fs.FileSystemReader.read_data = real_write, real_read
    ckpt_dcp._dcp_own_shard_error = real_readback
    published = os.path.isdir(os.path.join(node_dir, "step_5"))
    with open(os.path.join(root, f"result{rank}"), "w", encoding="utf-8") as f:
        f.write(f"{published}|{outcome}")
    try:
        dist.destroy_process_group()
    except Exception:
        pass


@pytest.mark.parametrize(
    "scenario,reported,published",
    [
        ("unreadable", "LR scheduler state unreadable on ranks [1]", True),
        ("directory-missing", "LR scheduler state unreadable on ranks [1]", None),
        ("write-fails", "extra state not written on every node", False),
        ("shard-write-fails", "failed shard verification three times", False),
        (
            "optimizer-write-fails",
            "optimizer / RNG state not written on every rank",
            False,
        ),
        ("publish-fails", "was not published", False),
        ("layout-missing", "not loadable on every rank", True),
        ("shard-read-fails", "not loadable on every rank", True),
        ("shared-readback-fails", "failed shard verification three times", False),
    ],
    ids=[
        "unreadable",
        "directory-missing",
        "write-fails",
        "shard-write-fails",
        "optimizer-write-fails",
        "publish-fails",
        "layout-missing",
        "shard-read-fails",
        "shared-readback-fails",
    ],
)
def test_one_node_failure_fails_every_rank(tmp_path, scenario, reported, published):
    pytest.importorskip(
        "verl", reason="VERL is required for this optional example test."
    )
    mp.spawn(
        _run_one_node_fails, args=(2, str(tmp_path), scenario), nprocs=2, join=True
    )
    for rank in range(2):
        was_published, outcome = (tmp_path / f"result{rank}").read_text().split("|", 1)
        assert outcome.startswith("RuntimeError") and reported in outcome, (
            rank,
            outcome,
        )
        if published is not None:
            assert was_published == str(published), (rank, was_published)


def _run_resave_publish_fails(rank, world, root):
    """Step 5 is saved, then saved again with the same scheduler state; the second publish succeeds on
    node 0 and fails on node 1, which keeps its older copy of step_5."""
    sys.path.insert(0, str(VERL_EXAMPLE_ROOT))
    from verl_mlite.engine import mlite_engine

    _cpu_dcp()
    os.uname = lambda: SimpleNamespace(nodename=f"node{rank}")
    dist.init_process_group(
        "gloo",
        init_method=f"file://{root}/rendezvous",
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=20),
    )
    node_dir = os.path.join(root, f"node{rank}", "global_step_5", "actor")
    real_replace = os.replace

    def failing_replace(src, dst, *args, **kwargs):
        if str(src).endswith(".partial"):
            raise OSError(28, "No space left on device")
        return real_replace(src, dst, *args, **kwargs)

    outcomes = []
    for attempt in range(2):
        if rank == 1 and attempt == 1:
            os.replace = failing_replace
        try:
            mlite_engine.MegatronLiteEngine.save_checkpoint(
                _engine(rank, _Schedule(5), torch.nn.Linear(4, 2)), node_dir, global_step=5
            )
            outcomes.append("saved")
        except RuntimeError as exc:
            outcomes.append(f"save failed: {exc}")
        os.replace = real_replace
    try:
        mlite_engine.MegatronLiteEngine.load_checkpoint(
            _engine(rank, _Schedule(), torch.nn.Linear(4, 2)), node_dir
        )
        outcomes.append("loaded")
    except Exception as exc:
        outcomes.append(f"{type(exc).__name__}: {exc}")
    with open(os.path.join(root, f"result{rank}"), "w", encoding="utf-8") as f:
        f.write("|".join(outcomes))
    dist.destroy_process_group()


def test_resume_refuses_a_step_published_on_some_nodes_only(tmp_path):
    pytest.importorskip(
        "verl", reason="VERL is required for this optional example test."
    )
    mp.spawn(_run_resave_publish_fails, args=(2, str(tmp_path)), nprocs=2, join=True)
    for rank in range(2):
        saved, resaved, loaded = (tmp_path / f"result{rank}").read_text().split("|")
        assert saved == "saved" and "was not published" in resaved, (rank, saved, resaved)
        assert loaded.startswith("RuntimeError") and "different checkpoints" in loaded, (rank, loaded)


@pytest.mark.parametrize(
    "scenario",
    ["missing-everywhere", "save-fails-before-scheduler"],
    ids=["missing-everywhere", "save-fails"],
)
def test_resume_without_scheduler_state_is_refused(tmp_path, monkeypatch, scenario):
    pytest.importorskip(
        "verl", reason="VERL is required for this optional example test."
    )
    from megatron.lite.primitive.ckpt import dcp as ckpt_dcp
    from verl_mlite.engine import mlite_engine

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(ckpt_dcp, "_build_meshes", lambda config: (None, None))
    monkeypatch.setattr(
        ckpt_dcp,
        "_dcp_tensor_from_param",
        lambda param, mesh, placements: param.detach(),
    )
    monkeypatch.setattr(
        ckpt_dcp,
        "_empty_dcp_tensor_like_param",
        lambda param, mesh, placements: torch.empty_like(param),
    )
    node_dir = str(tmp_path / "global_step_5" / "actor")
    schedule = _Schedule(5, fail=scenario == "save-fails-before-scheduler")
    try:
        mlite_engine.MegatronLiteEngine.save_checkpoint(
            _engine(0, schedule, torch.nn.Linear(4, 2)), node_dir, global_step=5
        )
    except RuntimeError:
        pass
    if scenario == "missing-everywhere":
        _remove_scheduler_files(node_dir)
    assert _resume(mlite_engine, 0, node_dir) == "refused"
