"""Shared-outer expert LoRA gradients under EP (torchrun, WORLD_SIZE >= 2).

w1/w3 ``lora_A`` and w2 ``lora_B`` are replicated on every EP rank but each
rank only differentiates through its own experts, so the per-rank gradient is
a partial sum over experts.  The correct gradient is the reduction over *all*
EP ranks (dense class).  Reducing only over the expert-data-parallel group
(what the default ``.experts.`` classifier would do) silently keeps the
partial.  Per-expert factors must equal the single-GPU rows of their experts.
"""

from __future__ import annotations

import importlib.util
import os

import pytest
import torch
import torch.distributed as dist

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from k31_moe_fixtures import (  # noqa: E402
    load_random_experts,
    make_moe,
    random_expert_lora,
    random_mxfp4_experts,
    random_routing,
)
from megatron.lite.model.k31.config import K31Config  # noqa: E402
from megatron.lite.model.k31.vllm.moe import DEEPEP_MAX_LOCAL_EXPERTS, is_expert_param  # noqa: E402


def _rel(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-300))


@pytest.mark.gpus(2)
@pytest.mark.parametrize("transport", ["deepep", "nccl"])
def test_shared_outer_lora_grads_need_all_ep_reduction(transport):
    if not torch.cuda.is_available() or importlib.util.find_spec("vllm") is None:
        pytest.skip("requires the K31 BI image")
    if int(os.environ.get("WORLD_SIZE", "1")) < 2:
        pytest.skip("requires torchrun")
    if transport not in os.environ.get("K31_EP_TRANSPORTS", "deepep,nccl").split(","):
        pytest.skip(f"{transport} transport disabled")
    from megatron.lite.primitive.kernels.vllm_k31_moe import initialize_k31_moe_batch_invariance

    initialize_k31_moe_batch_invariance()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    if not dist.is_initialized():
        dist.init_process_group(
            "nccl", device_id=torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        )
    ep, rank = dist.get_world_size(), dist.get_rank()
    cfg = K31Config()
    if transport == "deepep" and cfg.num_experts // ep > DEEPEP_MAX_LOCAL_EXPERTS:
        pytest.skip(
            f"DeepEP normal mode needs <= {DEEPEP_MAX_LOCAL_EXPERTS} local experts (EP={ep})"
        )
    full = random_mxfp4_experts(
        cfg.num_experts, cfg.routed_expert_hidden_size, cfg.moe_intermediate_size, seed=31
    )
    e_loc = cfg.num_experts // ep
    sl = slice(rank * e_loc, (rank + 1) * e_loc)
    moe = make_moe(cfg, group=dist.group.WORLD, transport=transport)
    load_random_experts(moe, full)
    random_expert_lora(moe, seed=32)
    ref = make_moe(cfg)
    load_random_experts(ref, full)
    random_expert_lora(ref, seed=32)
    del full

    def inputs(r):
        g = torch.Generator(device="cuda").manual_seed(500 + r)
        t = 48 + 3 * r
        h = cfg.routed_expert_hidden_size
        x = (torch.randn(t, h, device="cuda", generator=g) * 0.5).to(torch.bfloat16)
        ids, w = random_routing(t, cfg.num_experts, cfg.num_experts_per_token, seed=600 + r)
        gout = torch.randn(t, h, device="cuda", generator=g).to(torch.bfloat16)
        return x, ids, w, gout

    x, ids, w, gout = inputs(rank)
    x.requires_grad_(True)
    lat = moe.routed_latent(x, w, ids)
    lat.backward(gout)
    # single GPU reference over every rank's tokens (one loss = sum of the per-rank losses)
    xs, ids_s, ws, gs = zip(*(inputs(r) for r in range(ep)))
    xr = torch.cat(xs).requires_grad_(True)
    lat_r = ref.routed_latent(xr, torch.cat(ws), torch.cat(ids_s))
    lat_r.backward(torch.cat(gs))
    off = sum(48 + 3 * r for r in range(rank))
    assert _rel(x.grad.float(), xr.grad[off : off + x.shape[0]].float()) < 1e-2

    named = dict(moe.experts.named_parameters(prefix="model.layers.1.mlp.experts"))
    ref_named = dict(ref.experts.named_parameters(prefix="model.layers.1.mlp.experts"))
    report = {}
    for name, p in named.items():
        rp = ref_named[name]
        if is_expert_param(name):
            report[name] = _rel(p.grad.float(), rp.grad[sl].float())
            continue
        partial = p.grad.float().clone()
        total = partial.clone()
        dist.all_reduce(total)  # dense class: reduce over every EP rank
        report[name] = _rel(total, rp.grad.float())
        # the expert-class reduction (expert-DP group of size 1 here) keeps the partial: wrong
        report[name + ":ep_dp_only"] = _rel(partial, rp.grad.float())
    if rank == 0:
        print(
            "K31_SHARED_LORA_GRADS",
            transport,
            {k: round(v, 7) for k, v in report.items()},
            flush=True,
        )
    for name, err in report.items():
        if name.endswith(":ep_dp_only"):
            assert err > 0.05, (name, err)
        else:
            assert err < 1e-2, (name, err)
    dist.barrier()
