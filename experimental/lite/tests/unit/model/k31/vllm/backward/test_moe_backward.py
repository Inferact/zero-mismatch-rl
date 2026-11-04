"""Backward of the K31 MoE owners vs FP32 references (dX and LoRA dA / dB only)."""

from __future__ import annotations

import importlib.util
import os

import pytest
import torch

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from k31_moe_fixtures import (  # noqa: E402
    load_random_experts,
    make_moe,
    random_expert_lora,
    random_mxfp4_experts,
    random_routing,
    runtime_for,
)
from megatron.lite.model.k31.config import K31Config  # noqa: E402
from megatron.lite.model.k31.vllm.moe import K31MoE  # noqa: E402
from megatron.lite.model.k31.vllm.primitive.moe import (  # noqa: E402
    canonical_combine,
    dispatch_tokens,
    plan_canonical_combine,
)

gpu = pytest.mark.skipif(
    not (
        torch.cuda.is_available()
        and importlib.util.find_spec("vllm") is not None
        and os.environ.get("VLLM_BATCH_INVARIANT") == "1"
    ),
    reason="requires the K31 BI image",
)


@pytest.fixture(scope="module", autouse=True)
def _bi_mode():
    if torch.cuda.is_available() and os.environ.get("VLLM_BATCH_INVARIANT") == "1":
        from megatron.lite.primitive.kernels.vllm_k31_moe import initialize_k31_moe_batch_invariance

        initialize_k31_moe_batch_invariance()
    yield


def _cos(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-300))


def _rel(a, b):
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm().clamp_min(1e-300))


@gpu
def test_combine_backward_broadcasts_grad_to_every_leaf():
    t = 37
    ids, _ = random_routing(t, 896, 16, seed=3)
    leaves = torch.randn(t, 16, 3584, device="cuda", requires_grad=True)
    sent = dispatch_tokens(
        torch.zeros(t, 3584, dtype=torch.bfloat16, device="cuda"),
        torch.ones(t, 16, device="cuda"),
        ids,
        group=None,
        num_experts=896,
        transport="local",
    )
    out = canonical_combine(leaves, plan_canonical_combine(ids, sent, group=None, num_experts=896))
    g = torch.randn_like(out)
    out.backward(g)
    assert torch.equal(leaves.grad, g.float().unsqueeze(1).expand_as(leaves))


@gpu
def test_expert_owner_grads_match_fp32_reference():
    """dX, d(route weight), shared dA (w1/w3) / dB (w2) and per-expert dB (w1/w3) / dA (w2)."""

    from megatron.lite.primitive.alignment.vllm_fi_w4a16_moe import (
        ExpertLoRASpec,
        VLLMFiW4A16LeafRowsFunction,
        expert_leaves_reference,
    )
    from megatron.lite.primitive.kernels.vllm_k31_moe import FiW4A16Experts

    e, h, i, r, m = 64, 3584, 3072, 32, 96
    fi = FiW4A16Experts.from_checkpoint(
        *random_mxfp4_experts(e, h, i, seed=4),
        num_experts=e,
        local_offset=0,
        situ_beta=4.0,
        situ_linear_beta=25.0,
    )
    g = torch.Generator(device="cuda").manual_seed(5)
    lw = {
        "w1_a": torch.randn(r, h, device="cuda", generator=g) / h**0.5,
        "w3_a": torch.randn(r, h, device="cuda", generator=g) / h**0.5,
        "w1_b": torch.randn(e, i, r, device="cuda", generator=g) * 0.02,
        "w3_b": torch.randn(e, i, r, device="cuda", generator=g) * 0.02,
        "w2_a": torch.randn(e, r, i, device="cuda", generator=g) / i**0.5,
        "w2_b": torch.randn(h, r, device="cuda", generator=g) * 0.02,
    }
    lw = {k: v.to(torch.bfloat16).requires_grad_(True) for k, v in lw.items()}
    x = (
        (torch.randn(m, h, device="cuda", generator=g) * 0.5)
        .to(torch.bfloat16)
        .requires_grad_(True)
    )
    ids, w = random_routing(m, e, 16, seed=6)
    w = w.requires_grad_(True)
    gout = torch.randn(m, 16, h, device="cuda", generator=g)
    # every slot is local (EP 1); a scrambled row order exercises the leaf-row index map
    rows_index = torch.randperm(m * 16, device="cuda", generator=g)
    leaf_rows = VLLMFiW4A16LeafRowsFunction.apply(
        fi, ExpertLoRASpec(scaling=2.0), rows_index, x, ids, w, *lw.values()
    )
    grads = torch.autograd.grad(leaf_rows, [x, w, *lw.values()], gout.view(m * 16, h)[rows_index])

    ref_in = {k: v.detach().float().requires_grad_(True) for k, v in lw.items()}
    xr = x.detach().float().requires_grad_(True)
    wr = w.detach().float().requires_grad_(True)
    ref = expert_leaves_reference(xr, ids, wr, fi, scaling=2.0, stored_lora=False, **ref_in)
    ref_grads = torch.autograd.grad(ref, [xr, wr, *ref_in.values()], gout)
    # visible forward vs fp32 reference: BF16 rounding of the GEMM outputs only
    assert _cos(leaf_rows.detach(), ref.detach().view(m * 16, h)[rows_index]) > 0.9999
    names = ["x", "w", *lw]
    report = {}
    for name, a, b in zip(names, grads, ref_grads, strict=True):
        report[name] = (_cos(a.float(), b.float()), _rel(a.float(), b.float()))
    print("K31MOE_EXPERT_BWD", report, flush=True)
    for name, (cos, rel) in report.items():
        assert cos > 0.9999 and rel < 1e-2, (name, cos, rel)


def _block_reference(moe: K31MoE, x, ids, P):
    """FP32 statement of the whole block at the visible routing (ids fixed)."""

    from megatron.lite.primitive.alignment.vllm_fi_w4a16_moe import expert_leaves_reference

    cfg = moe.config
    s = moe.experts.lora_scaling
    xf = x.float()
    logits = xf @ moe.gate.weight.float().t()
    sel = torch.sigmoid(logits).gather(1, ids.long())
    w = sel / sel.sum(-1, keepdim=True)
    z = xf @ moe.routed_expert_down_proj.weight.float().t()
    ex = {k: P[f"experts.lora_{k}"] for k in ("w1_A", "w3_A", "w1_B", "w3_B", "w2_A", "w2_B")}
    leaves = expert_leaves_reference(
        z,
        ids,
        w,
        moe.experts.fi,
        scaling=s,
        stored_lora=False,
        w1_a=ex["w1_A"][0],
        w3_a=ex["w3_A"][0],
        w1_b=ex["w1_B"],
        w3_b=ex["w3_B"],
        w2_a=ex["w2_A"],
        w2_b=ex["w2_B"][0],
    )
    latent = leaves.sum(1)
    rstd = torch.rsqrt(latent.square().mean(-1, keepdim=True) + cfg.rms_norm_eps)
    normed = latent * rstd * moe.routed_expert_norm.weight.float()
    routed = normed @ moe.routed_expert_up_proj.weight.float().t()
    se = moe.shared_experts

    def lin(v, proj, prefix, names):
        y = v @ proj.weight.float().t()
        outs = [
            (v @ P[f"{prefix}.lora.lora_A.{n}"].t()) @ (P[f"{prefix}.lora.lora_B.{n}"] * s).t()
            for n in names
        ]
        return y + torch.cat(outs, -1)

    gu = lin(xf, se.gate_up_proj, "shared_experts.gate_up_proj", ("gate_proj", "up_proj"))
    d = gu.shape[-1] // 2
    gate, up = gu[:, :d], gu[:, d:]
    beta, lbeta = cfg.activation_situ_beta, cfg.activation_situ_linear_beta
    act = beta * torch.tanh(gate / beta) * torch.sigmoid(gate) * (lbeta * torch.tanh(up / lbeta))
    shared = lin(act, se.down_proj, "shared_experts.down_proj", ("down_proj",))
    return routed + shared


@gpu
def test_moe_block_dx_and_lora_grads_match_fp32_reference():
    cfg = K31Config()
    moe = make_moe(cfg)
    g = torch.Generator(device="cuda").manual_seed(7)
    se = moe.shared_experts
    with torch.no_grad():
        for p, std in (
            (moe.gate.weight, 0.02),
            (moe.routed_expert_down_proj.weight, 0.02),
            (moe.routed_expert_up_proj.weight, 0.02),
            (se.gate_up_proj.weight, 0.02),
            (se.down_proj.weight, 0.02),
        ):
            p.copy_(torch.randn(p.shape, device="cuda", generator=g) * std)
        moe.gate.e_score_correction_bias.copy_(
            torch.randn(cfg.num_experts, device="cuda", generator=g) * 0.01
        )
        moe.routed_expert_norm.weight.copy_(
            1 + 0.1 * torch.randn(cfg.routed_expert_hidden_size, device="cuda", generator=g)
        )
        for proj in (se.gate_up_proj, se.down_proj):
            for a in proj.lora.lora_A.values():
                a.copy_(torch.randn(a.shape, device="cuda", generator=g) / a.shape[1] ** 0.5)
            for b in proj.lora.lora_B.values():
                b.copy_(torch.randn(b.shape, device="cuda", generator=g) * 0.02)
    load_random_experts(
        moe,
        random_mxfp4_experts(
            cfg.num_experts, cfg.routed_expert_hidden_size, cfg.moe_intermediate_size, seed=8
        ),
    )
    random_expert_lora(moe, seed=9)
    t = 64
    x = (
        (torch.randn(t, cfg.hidden_size, device="cuda", generator=g))
        .to(torch.bfloat16)
        .requires_grad_(True)
    )
    out, parts = moe.forward_parts(x, runtime=runtime_for(t))
    gout = torch.randn(out.shape, device="cuda", generator=g)
    names = [n for n, p in moe.named_parameters() if p.requires_grad]
    params = [p for p in moe.parameters() if p.requires_grad]
    grads = torch.autograd.grad(out, [x, *params], gout.to(out.dtype))
    xr = x.detach().float().requires_grad_(True)
    master = {
        n: p.detach().float().requires_grad_(True) for n, p in zip(names, params, strict=True)
    }
    ref = _block_reference(moe, xr, parts["ids"], master)
    ref_grads = torch.autograd.grad(ref, [xr, *master.values()], gout)
    assert _cos(out.detach().float(), ref.detach()) > 0.999
    report = {"x": (_cos(grads[0].float(), ref_grads[0]), _rel(grads[0].float(), ref_grads[0]))}
    for name, a, b in zip(names, grads[1:], ref_grads[1:], strict=True):
        report[name] = (_cos(a.float(), b), _rel(a.float(), b))
    print(
        "K31MOE_BLOCK_BWD",
        {k: (round(c, 6), round(r, 5)) for k, (c, r) in report.items()},
        flush=True,
    )
    assert len(names) == 12  # 6 expert LoRA + 6 shared-expert LoRA factors
    for name, (cos, rel) in report.items():
        assert cos > 0.999 and rel < 5e-2, (name, cos, rel)
