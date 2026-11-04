"""EP invariance of the K31 canonical-slot combine (torchrun, any EP size).

Every rank owns ``896 / EP`` FI W4A16 experts (same random MXFP4 bytes and
expert LoRA on all ranks, sliced by global expert id) and also a single-GPU
EP=1 copy of all 896 experts.  The EP latent of each rank's tokens must equal,
bit for bit, the EP=1 latent of the same tokens (the single-GPU canonical
tree), for normal DeepEP dispatch and for the torch all-to-all transport.

Run::

    torchrun --nproc-per-node=4 -m pytest -q tests/unit/model/k31/vllm/test_deepep_ep2.py
    # EP16: torchrun --nnodes=4 --nproc-per-node=4 ... (K31_EP_TRANSPORTS=nccl,deepep)
"""

from __future__ import annotations

import importlib.util
import json
import os

import pytest
import torch
import torch.distributed as dist

from k31_moe_fixtures import (
    bits_equal,
    load_random_experts,
    make_moe,
    mismatch,
    random_expert_lora,
    random_mxfp4_experts,
    random_routing,
)
from megatron.lite.model.k31.config import K31Config
from megatron.lite.model.k31.vllm.moe import DEEPEP_MAX_LOCAL_EXPERTS


def _skip_reason() -> str | None:
    if not torch.cuda.is_available() or importlib.util.find_spec("vllm") is None:
        return "requires the K31 BI image"
    if int(os.environ.get("WORLD_SIZE", "1")) < 2:
        return "requires torchrun with WORLD_SIZE >= 2"
    return None


_STATE: dict = {}


def _setup():
    if "group" in _STATE:
        return _STATE
    from megatron.lite.primitive.kernels.vllm_k31_moe import initialize_k31_moe_batch_invariance

    initialize_k31_moe_batch_invariance()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    if not dist.is_initialized():
        dist.init_process_group(
            "nccl", device_id=torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        )
    cfg = K31Config()
    full = random_mxfp4_experts(
        cfg.num_experts, cfg.routed_expert_hidden_size, cfg.moe_intermediate_size, seed=21
    )
    ep, rank = dist.get_world_size(), dist.get_rank()
    ref = make_moe(cfg)
    load_random_experts(ref, full)
    random_expert_lora(ref, seed=22)
    mods = {}
    for transport in os.environ.get("K31_EP_TRANSPORTS", "deepep,nccl").split(","):
        if deepep_skip_reason(transport, cfg.num_experts, ep):
            continue
        m = make_moe(cfg, group=dist.group.WORLD, transport=transport)
        load_random_experts(m, full)
        random_expert_lora(m, seed=22)
        mods[transport] = m
    del full
    torch.cuda.empty_cache()
    _STATE.update(group=dist.group.WORLD, ep=ep, rank=rank, cfg=cfg, ref=ref, mods=mods)
    return _STATE


def deepep_skip_reason(transport: str, num_experts: int, ep: int) -> str | None:
    if transport == "deepep" and num_experts // ep > DEEPEP_MAX_LOCAL_EXPERTS:
        return f"DeepEP normal mode needs <= {DEEPEP_MAX_LOCAL_EXPERTS} local experts (EP={ep})"
    return None


def _inputs(t: int, rank: int, cfg):
    g = torch.Generator(device="cuda").manual_seed(1000 + 7 * rank + t)
    x = (torch.randn(t, cfg.routed_expert_hidden_size, device="cuda", generator=g) * 0.5).to(
        torch.bfloat16
    )
    ids, w = random_routing(
        t, cfg.num_experts, cfg.num_experts_per_token, seed=2000 + 13 * rank + t
    )
    return x, ids, w


@pytest.mark.gpus(2)
@pytest.mark.parametrize("transport", ["deepep", "nccl"])
@pytest.mark.parametrize("t", [1, 9, 333, 2048], ids=lambda t: f"T{t}")
def test_ep_canonical_combine_equals_single_gpu_tree(t, transport):
    reason = _skip_reason()
    if reason:
        pytest.skip(reason)
    st = _setup()
    if transport not in st["mods"]:
        pytest.skip(
            deepep_skip_reason(transport, st["cfg"].num_experts, st["ep"])
            or f"{transport} disabled"
        )
    rank, cfg = st["rank"], st["cfg"]
    tokens = t + 5 * rank  # uneven token counts across ranks
    x, ids, w = _inputs(tokens, rank, cfg)
    with torch.no_grad():
        ref = st["ref"].routed_latent(x, w, ids)
        got = st["mods"][transport].routed_latent(x, w, ids)
        bad = 0 if bits_equal(got, ref) else mismatch(got, ref)
    counts = torch.tensor([bad], device="cuda", dtype=torch.int64)
    dist.all_reduce(counts)
    if rank == 0:
        print(
            f"K31EP ep={st['ep']} T={t} transport={transport} mismatches={int(counts)}", flush=True
        )
    assert int(counts.sum()) == 0


@pytest.mark.gpus(2)
def test_combine_all_to_all_cost():
    """fp32 leaves (16 x 3584 x 4 B = 229 KB / token) vs one BF16 hidden copy per token."""

    reason = _skip_reason()
    if reason:
        pytest.skip(reason)
    st = _setup()
    ep, group = st["ep"], st["group"]
    res = {}
    for t in (256, 1024, 4096):
        rows = t * 16
        leaves = torch.randn(rows, 3584, device="cuda")
        out = torch.empty_like(leaves)
        hid = torch.randn(t, 7168, device="cuda").to(torch.bfloat16)
        hout = torch.empty_like(hid)
        times = {}
        for name, (a, b) in {"leaves_fp32": (leaves, out), "hidden_bf16": (hid, hout)}.items():
            for _ in range(3):
                dist.all_to_all_single(b, a, group=group)
            torch.cuda.synchronize()
            dist.barrier(group=group)
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(10):
                dist.all_to_all_single(b, a, group=group)
            e.record()
            torch.cuda.synchronize()
            times[name] = s.elapsed_time(e) / 10 * 1000.0
        mx = torch.tensor([times["leaves_fp32"], times["hidden_bf16"]], device="cuda")
        dist.all_reduce(mx, op=dist.ReduceOp.MAX)
        res[t] = {
            "leaves_fp32_us": round(float(mx[0]), 1),
            "hidden_bf16_us": round(float(mx[1]), 1),
            "leaves_MB_per_rank": round(rows * 3584 * 4 / 1e6, 1),
        }
    if st["rank"] == 0:
        print("K31EP_COST", json.dumps({"ep": ep, "per_rank_tokens": res}), flush=True)
