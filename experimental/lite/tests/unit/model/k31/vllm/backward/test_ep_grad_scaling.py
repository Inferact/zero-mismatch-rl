"""Replica-gradient finalize under EP (torchrun, WORLD_SIZE >= 2, EP = world).

Each rank's loss is its local token mean and the training objective is
``(1/DP) sum_r L_r`` (verl's global-token normalisation only multiplies every
rank's loss by the same constant).  After ``_finalize_replica_grads`` every
trainable class -- a dense parameter, the shared outer expert-LoRA factors
(dense class, replicated on every EP rank) and the per-expert LoRA factors --
must hold the gradient of that objective, so all of them get the same
effective learning rate.  The single-GPU reference evaluates the objective on
every rank's tokens at once.  Dividing per-expert gradients by the expert-DP
size (the DS4 non-FSDP convention) is EP times too large.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from k31_moe_fixtures import (  # noqa: E402
    load_random_experts,
    make_moe,
    random_expert_lora,
    random_mxfp4_experts,
    random_routing,
)
from megatron.lite.model.k31.config import K31Config  # noqa: E402
from megatron.lite.model.k31.vllm import protocol  # noqa: E402
from megatron.lite.model.k31.vllm.moe import is_expert_param  # noqa: E402
from megatron.lite.primitive.parallel import ParallelState  # noqa: E402


def _rel(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-300))


class _Probe(nn.Module):
    """A dense trainable parameter on the MoE input (stands in for dense LoRA)."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.lora_probe = nn.Parameter(torch.ones(hidden, dtype=torch.float32))


def _rooted(moe, probe) -> nn.Module:
    layer = nn.Module()
    layer.mlp = moe
    layer.self_attn = probe
    root = nn.Module()
    root.layers = nn.ModuleDict({"1": layer})
    return root


@pytest.mark.gpus(2)
def test_finalize_gives_every_lora_class_the_global_objective():
    if not torch.cuda.is_available() or importlib.util.find_spec("vllm") is None:
        pytest.skip("requires the K31 BI image")
    if int(os.environ.get("WORLD_SIZE", "1")) < 2:
        pytest.skip("requires torchrun")
    from megatron.lite.primitive.kernels.vllm_k31_moe import initialize_k31_moe_batch_invariance

    initialize_k31_moe_batch_invariance()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    if not dist.is_initialized():
        dist.init_process_group(
            "nccl", device_id=torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        )
    world, rank = dist.get_world_size(), dist.get_rank()
    cfg = K31Config()
    h = cfg.routed_expert_hidden_size
    full = random_mxfp4_experts(cfg.num_experts, h, cfg.moe_intermediate_size, seed=41)
    e_loc = cfg.num_experts // world
    sl = slice(rank * e_loc, (rank + 1) * e_loc)
    moe = make_moe(cfg, group=dist.group.WORLD, transport="nccl")
    load_random_experts(moe, full)
    random_expert_lora(moe, seed=42)
    ref = make_moe(cfg)
    load_random_experts(ref, full)
    random_expert_lora(ref, seed=42)
    del full
    probe, ref_probe = _Probe(h).cuda(), _Probe(h).cuda()

    def inputs(r):
        g = torch.Generator(device="cuda").manual_seed(700 + r)
        t = 40 + 7 * r  # unequal token counts: the per-rank means differ in weight
        x = (torch.randn(t, h, device="cuda", generator=g) * 0.5).to(torch.bfloat16)
        ids, w = random_routing(t, cfg.num_experts, cfg.num_experts_per_token, seed=800 + r)
        gout = torch.randn(t, h, device="cuda", generator=g)
        return x, ids, w, gout

    def local_loss(model, prb, x, ids, w, gout):
        xin = (x.float() * prb.lora_probe).to(torch.bfloat16)
        lat = model.routed_latent(xin, w, ids)
        return (lat.float() * gout).sum() / x.shape[0]

    x, ids, w, gout = inputs(rank)
    local_loss(moe, probe, x, ids, w, gout).backward()
    ps = ParallelState()
    ps.ep_group, ps.ep_size, ps.ep_rank = dist.group.WORLD, world, rank
    ps.dp_group, ps.dp_size, ps.dp_cp_size = dist.group.WORLD, world, world
    ps.ep_dp_group, ps.expert_dp_size = None, 1
    root = _rooted(moe, probe)
    raw = {n: p.grad.detach().clone() for n, p in root.named_parameters() if p.grad is not None}
    protocol._finalize_replica_grads(root, ps)

    # reference objective: (1/DP) * sum_r mean_r, all ranks' tokens on one GPU
    total = sum(
        local_loss(ref, ref_probe, *inputs(r)) for r in range(world)
    ) / world
    total.backward()
    ref_root = _rooted(ref, ref_probe)
    ref_named = dict(ref_root.named_parameters())

    report = {}
    for name, p in root.named_parameters():
        if p.grad is None:
            continue
        rg = ref_named[name].grad
        if is_expert_param(name):
            report[name] = _rel(p.grad.float(), rg[sl].float())
            # DS4 non-FSDP convention: all-reduce over expert-DP (size 1) / expert_dp_size
            report[name + ":expert_dp_divisor"] = _rel(raw[name].float(), rg[sl].float())
        else:
            report[name] = _rel(p.grad.float(), rg.float())
    if rank == 0:
        print("K31_EP_GRAD_SCALING", world, {k: round(v, 7) for k, v in report.items()}, flush=True)
    for name, err in report.items():
        if name.endswith(":expert_dp_divisor"):
            assert abs(err - (world - 1)) < 0.05 * world, (name, err)
        else:
            assert err < 1e-2, (name, err)
    dist.barrier()
