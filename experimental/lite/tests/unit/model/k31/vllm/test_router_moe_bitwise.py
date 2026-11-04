"""K31 router + FI W4A16 experts + expert LoRA + canonical combine + latent tail.

GPU tests run inside the K31 BI image with the full overlay chain and
``VLLM_BATCH_INVARIANT=1``.  Rollout gates need
``K31_MODEL_DIR``, ``K31_ADAPTER_DIR`` and ``K31_ROLLOUT_DUMP`` (output of the
MoE dump hook on a real BI=1 DP4/EP4 server with CUDA graph FULL).
"""

from __future__ import annotations

import glob
import importlib.util
import os
import types

import pytest
import torch

from megatron.lite.model.k31.vllm.moe import is_expert_param
from megatron.lite.model.k31.vllm.primitive.moe import (
    canonical_combine,
    combine_reference,
    dispatch_tokens,
    plan_canonical_combine,
)

from k31_moe_fixtures import bits_equal, env_path, mismatch, random_mxfp4_experts, random_routing


def _gpu_ready() -> bool:
    return (
        torch.cuda.is_available()
        and importlib.util.find_spec("vllm") is not None
        and os.environ.get("VLLM_BATCH_INVARIANT") == "1"
    )


gpu = pytest.mark.skipif(not _gpu_ready(), reason="requires the K31 BI image (CUDA, vLLM, BI env)")


@pytest.fixture(scope="module", autouse=True)
def _bi_mode():
    if _gpu_ready():
        from megatron.lite.primitive.kernels.vllm_k31_moe import initialize_k31_moe_batch_invariance

        initialize_k31_moe_batch_invariance()
    yield


# --------------------------------------------------------------------------- CPU
@pytest.mark.parametrize(
    "name,expected",
    [
        ("model.layers.1.mlp.experts.lora_w1_B", True),
        ("model.layers.1.mlp.experts.lora_w3_B", True),
        ("model.layers.1.mlp.experts.lora_w2_A", True),
        ("model.layers.1.mlp.experts.lora_w1_A", False),
        ("model.layers.1.mlp.experts.lora_w3_A", False),
        ("model.layers.1.mlp.experts.lora_w2_B", False),
        ("model.layers.1.mlp.shared_experts.gate_up_proj.lora.lora_A.gate_proj", False),
        ("model.layers.0.mlp.down_proj.lora.lora_B.down_proj", False),
    ],
    ids=[
        "w1_b",
        "w3_b",
        "w2_a",
        "shared-w1_a",
        "shared-w3_a",
        "shared-w2_b",
        "shared-expert",
        "dense",
    ],
)
def test_expert_param_classifier(name, expected):
    assert is_expert_param(name) is expected


def test_combine_plan_ep1_canonical_positions():
    ids = torch.tensor([[5, 1, 9, 3] * 4], dtype=torch.long)
    ids[0] = torch.tensor([15, 2, 7, 0, 11, 4, 9, 13, 1, 6, 14, 3, 8, 12, 5, 10])
    sent = dispatch_tokens(
        torch.zeros(1, 16, dtype=torch.bfloat16),
        torch.ones(1, 16),
        ids,
        group=None,
        num_experts=16,
        transport="local",
    )
    plan = plan_canonical_combine(ids, sent, group=None, num_experts=16)
    # leaf of slot k lands at canonical position rank(ids[k])
    assert plan.canon_pos.tolist() == ids[0].argsort().argsort().tolist()


# --------------------------------------------------------------------------- GPU: combine
@gpu
@pytest.mark.parametrize("t", [1, 7, 64, 333], ids=lambda t: f"T{t}")
def test_canonical_tree_matches_torch_contract(t):
    torch.manual_seed(t)
    ids, _ = random_routing(t, 896, 16, seed=t)
    # wide-magnitude leaves make the summation order observable
    leaves = torch.randn(t, 16, 3584, device="cuda") * torch.exp2(
        torch.randint(-12, 12, (t, 16, 1), device="cuda").float()
    )
    sent = dispatch_tokens(
        torch.zeros(t, 3584, dtype=torch.bfloat16, device="cuda"),
        torch.ones(t, 16, device="cuda"),
        ids,
        group=None,
        num_experts=896,
        transport="local",
    )
    plan = plan_canonical_combine(ids, sent, group=None, num_experts=896)
    out = canonical_combine(leaves, plan)
    ref = combine_reference(leaves, ids)
    assert bits_equal(out, ref), mismatch(out, ref)
    # sensitivity: router-slot order differs from the canonical order
    k_order = leaves.sum(1).to(torch.bfloat16)
    if t >= 64:
        assert mismatch(k_order, ref) > 0


# --------------------------------------------------------------------------- GPU: FI port
def _fake_overlay_experts(fi, lora):
    from vllm.model_executor.layers.quantization.mxfp4 import K31FiW4A16Experts as cls

    fake = types.SimpleNamespace(
        _lora_context=lora,
        _k31_nglob=fi.num_experts,
        _k31_off=fi.local_offset,
        _k31_w1a=fi.w1_alpha,
        _k31_w2a=fi.w2_alpha,
        w1_scale=fi.w13_sf,
        w2_scale=fi.w2_sf,
        _k31_lora_cfgs=cls._k31_lora_cfgs,
    )
    fake._k31_lora_w13 = types.MethodType(cls._k31_lora_w13, fake)
    fake._k31_lora_w2 = types.MethodType(cls._k31_lora_w2, fake)
    return cls, fake


@gpu
@pytest.mark.parametrize("m", [1, 9, 200], ids=lambda m: f"M{m}")
def test_fi_port_matches_overlay_apply_lora(m, monkeypatch):
    """The ported sequence equals the overlay's own ``_k31_apply_lora`` (k-order sum, tree off)."""

    from megatron.lite.primitive.alignment.vllm_fi_w4a16_moe import (
        ExpertLoRASpec,
        build_expert_lora_context,
    )
    from megatron.lite.primitive.kernels.vllm_k31_moe import FiW4A16Experts, fi_w4a16_lora_leaves

    e, h, i, r = 64, 3584, 3072, 32
    fi = FiW4A16Experts.from_checkpoint(
        *random_mxfp4_experts(e, h, i, seed=3),
        num_experts=e,
        local_offset=0,
        situ_beta=4.0,
        situ_linear_beta=25.0,
    )
    g = torch.Generator(device="cuda").manual_seed(m)
    lora_w = dict(
        w1_a=torch.randn(r, h, device="cuda", generator=g) / h**0.5,
        w3_a=torch.randn(r, h, device="cuda", generator=g) / h**0.5,
        w1_b=torch.randn(e, i, r, device="cuda", generator=g) * 0.02,
        w3_b=torch.randn(e, i, r, device="cuda", generator=g) * 0.02,
        w2_a=torch.randn(e, r, i, device="cuda", generator=g) / i**0.5,
        w2_b=torch.randn(h, r, device="cuda", generator=g) * 0.02,
    )
    x = (torch.randn(m, h, device="cuda", generator=g) * 0.5).to(torch.bfloat16)
    ids, w = random_routing(m, e, 16, seed=m)
    lora = build_expert_lora_context(m, x.device, ExpertLoRASpec(scaling=2.0), top_k=16, **lora_w)
    slot = fi_w4a16_lora_leaves(x, ids, w, fi, lora)
    acc = slot[:, 0].clone()
    for k in range(1, 16):
        acc.add_(slot[:, k])
    mine = acc.to(torch.bfloat16)
    cls, fake = _fake_overlay_experts(
        fi, build_expert_lora_context(m, x.device, ExpertLoRASpec(scaling=2.0), top_k=16, **lora_w)
    )
    out = torch.empty(m, h, dtype=torch.bfloat16, device="cuda")
    cls._k31_apply_lora(
        fake,
        out,
        x,
        fi.w13,
        fi.w2,
        w,
        ids,
        None,
        4.0,
        25.0,
        tlm=torch.zeros(m, dtype=torch.int32, device="cuda"),
    )
    assert bits_equal(mine, out), mismatch(mine, out)
    assert bool((slot.abs().sum(-1) > 0).all())


@gpu
@pytest.mark.parametrize("m", [1, 17, 300], ids=lambda m: f"M{m}")
def test_fi_leaves_are_row_local(m):
    """Leaves of a row do not depend on the other rows (training batches differ from rollout)."""

    from megatron.lite.primitive.alignment.vllm_fi_w4a16_moe import (
        ExpertLoRASpec,
        build_expert_lora_context,
    )
    from megatron.lite.primitive.kernels.vllm_k31_moe import FiW4A16Experts, fi_w4a16_lora_leaves

    e, h, i, r = 896, 3584, 3072, 32
    fi = FiW4A16Experts.from_checkpoint(
        *random_mxfp4_experts(e, h, i, seed=5),
        num_experts=e,
        local_offset=0,
        situ_beta=4.0,
        situ_linear_beta=25.0,
    )
    g = torch.Generator(device="cuda").manual_seed(11)
    lw = dict(
        w1_a=torch.randn(r, h, device="cuda", generator=g) / h**0.5,
        w3_a=torch.randn(r, h, device="cuda", generator=g) / h**0.5,
        w1_b=torch.randn(e, i, r, device="cuda", generator=g) * 0.02,
        w3_b=torch.randn(e, i, r, device="cuda", generator=g) * 0.02,
        w2_a=torch.randn(e, r, i, device="cuda", generator=g) / i**0.5,
        w2_b=torch.randn(h, r, device="cuda", generator=g) * 0.02,
    )
    x = (torch.randn(m + 40, h, device="cuda", generator=g) * 0.5).to(torch.bfloat16)
    ids, w = random_routing(m + 40, e, 16, seed=m)
    big = fi_w4a16_lora_leaves(
        x,
        ids,
        w,
        fi,
        build_expert_lora_context(m + 40, x.device, ExpertLoRASpec(2.0), top_k=16, **lw),
    )
    small = fi_w4a16_lora_leaves(
        x[:m].contiguous(),
        ids[:m].contiguous(),
        w[:m].contiguous(),
        fi,
        build_expert_lora_context(m, x.device, ExpertLoRASpec(2.0), top_k=16, **lw),
    )
    assert bits_equal(big[:m].contiguous(), small)


# --------------------------------------------------------------------------- GPU: MXFP4 interpretation
def _fi_gemm_rows(fi, expert: int, act: torch.Tensor, which: int) -> torch.Tensor:
    """Run one FI grouped GEMM (1 = w13, 2 = w2) on rows all routed to ``expert`` (top_k 1)."""

    import flashinfer.fused_moe.cute_dsl.blackwell.moe_w4a16 as F
    from flashinfer.tllm_enums import (
        DEFAULT_SWIGLU_ALPHA,
        DEFAULT_SWIGLU_BETA,
        DEFAULT_SWIGLU_LIMIT,
    )
    from vllm.model_executor.layers.quantization.mxfp4 import _k31_pick_tactic

    m = act.shape[0]
    ids = torch.full((m, 1), fi.local_offset + expert, dtype=torch.int32, device=act.device)
    ww = torch.ones(m, 1, dtype=torch.float32, device=act.device)
    tactic = _k31_pick_tactic(m, 1, fi.num_experts)
    rt = tactic[0][0][1]
    x_like = torch.empty(m, fi.hidden, dtype=torch.bfloat16, device=act.device)
    ws = F._get_workspace(x_like, 1, fi.num_experts, fi.num_local_experts, fi.intermediate, rt)
    t2e, t2l, e2p, p2e, _, nnt = F.moe_sort(
        token_selected_experts=ids,
        token_final_scales=ww,
        num_experts=fi.num_experts,
        top_k=1,
        local_expert_offset=fi.local_offset,
        num_local_experts=fi.num_local_experts,
        tile_tokens_dim=rt,
        enable_pdl=False,
        **ws.moe_sort_buffers,
    )
    s = F.get_max_num_permuted_tokens(m, 1, fi.num_local_experts, rt)
    t2e, t2l, e2p, p2e = t2e[: s // rt], t2l[: s // rt], e2p[:m], p2e[:s]
    perm = torch.zeros(s, act.shape[1], dtype=torch.bfloat16, device=act.device)
    F.moe_permute(
        input=act,
        permuted_output=perm,
        tile_idx_to_mn_limit=t2l,
        permuted_idx_to_expanded_idx=p2e,
        num_non_exiting_tiles=nnt,
        max_num_permuted_tokens=s,
        top_k=1,
        tile_size=rt,
        enable_pdl=False,
    )
    common = dict(
        tile_idx_to_expert_idx=t2e,
        tile_idx_to_mn_limit=t2l,
        num_non_exiting_tiles=nnt,
        num_local_experts=fi.num_local_experts,
        activation_type=None,
        swiglu_alpha=DEFAULT_SWIGLU_ALPHA,
        swiglu_beta=DEFAULT_SWIGLU_BETA,
        swiglu_limit=DEFAULT_SWIGLU_LIMIT,
        situ_beta=None,
        situ_linear_beta=None,
        use_fused_finalize=False,
        permuted_idx_to_expanded_idx=None,
        token_final_scales=None,
        enable_pdl=False,
    )
    if which == 1:
        out = torch.empty(s, 2 * fi.intermediate, dtype=torch.bfloat16, device=act.device)
        F._run_grouped_gemm(
            weight=fi.w13,
            weight_sf=fi.w13_sf,
            activations=perm,
            alpha=fi.w1_alpha,
            output=out,
            tactic=tactic[0],
            **common,
        )
        v = out.view(s, fi.intermediate // 64, 2, 64)
        out = torch.cat([v[:, :, 1].reshape(s, -1), v[:, :, 0].reshape(s, -1)], 1)  # -> [gate | up]
    else:
        out = torch.empty(s, fi.hidden, dtype=torch.bfloat16, device=act.device)
        F._run_grouped_gemm(
            weight=fi.w2,
            weight_sf=fi.w2_sf,
            activations=perm,
            alpha=fi.w2_alpha,
            output=out,
            tactic=tactic[1],
            **common,
        )
    flat = e2p.reshape(-1).long()
    return out[flat]


@gpu
@pytest.mark.parametrize("source", ["checkpoint", "random"])
def test_dequantize_mxfp4_matches_fi_w4a16_interpretation(source):
    """One-hot activations make each FI output row one weight column exactly:
    E8M0 scale, nibble order, w1/w3 order and the 64-row interleave all checked bitwise."""

    from megatron.lite.primitive.alignment.vllm_fi_w4a16_moe import dequantize_k31_mxfp4
    from megatron.lite.primitive.kernels.vllm_k31_moe import FiW4A16Experts

    if source == "checkpoint":
        model_dir = env_path("K31_MODEL_DIR")
        if model_dir is None:
            pytest.skip("K31_MODEL_DIR not set")
        from k31_moe_fixtures import load_moe_layer

        moe = load_moe_layer(model_dir, 1, adapter_dir=None)
        # The module keeps a lean FI object (no resident w13 rows); the one-hot GEMM needs them.
        fi = moe.experts.fi.materialize()
        experts = (0, 437, 895)
    else:
        fi = FiW4A16Experts.from_checkpoint(
            *random_mxfp4_experts(8, 3584, 3072, seed=9),
            num_experts=8,
            local_offset=0,
            situ_beta=4.0,
            situ_linear_beta=25.0,
        )
        experts = (0, 7)
    for e in experts:
        w13 = dequantize_k31_mxfp4(fi.ckpt_w13[e], fi.ckpt_w13_scale[e])  # [2I, H]
        w2 = dequantize_k31_mxfp4(fi.ckpt_w2[e], fi.ckpt_w2_scale[e])  # [H, I]
        eye_h = torch.eye(fi.hidden, dtype=torch.bfloat16, device="cuda")
        got13 = _fi_gemm_rows(fi, e, eye_h, 1)  # row j = W13[:, j]
        # a one-hot GEMM sums +0 products, so a -0 weight comes out as +0: compare with -0 folded
        ref13 = w13.t().contiguous().to(torch.bfloat16) + 0.0
        assert bits_equal(got13, ref13), (e, mismatch(got13, ref13))
        eye_i = torch.eye(fi.intermediate, dtype=torch.bfloat16, device="cuda")
        got2 = _fi_gemm_rows(fi, e, eye_i, 2)
        ref2 = w2.t().contiguous().to(torch.bfloat16) + 0.0
        assert bits_equal(got2, ref2), (e, mismatch(got2, ref2))
        assert bool(
            (w13.to(torch.bfloat16).float() == w13).all()
        )  # dequant values are exact in BF16


# --------------------------------------------------------------------------- GPU: router
@gpu
def test_router_adapter_is_vllm_selection_and_r3_noop():
    from megatron.lite.primitive.kernels.vllm_k31_moe import build_k31_router
    from megatron.lite.primitive.modules.router_replay import RouterReplay, RouterReplayAction
    from k31_moe_fixtures import make_moe
    from megatron.lite.model.k31.config import K31Config

    torch.manual_seed(0)
    logits = torch.randn(257, 896, device="cuda") * 3
    bias = torch.randn(896, device="cuda") * 0.1
    router = build_k31_router(top_k=16, num_experts=896, e_score_correction_bias=bias)
    w, ids = router.select_experts(
        hidden_states=torch.empty(257, 1, device="cuda"), router_logits=logits
    )
    scores = torch.sigmoid(logits)
    ref_ids = torch.topk(scores + bias, 16, dim=-1, sorted=True).indices
    assert torch.equal(ids.long().sort(-1).values, ref_ids.sort(-1).values)
    sel = scores.gather(1, ids.long())
    torch.testing.assert_close(w, sel / sel.sum(-1, keepdim=True), rtol=1e-6, atol=1e-7)
    # row-locality of the selection kernel
    w1, ids1 = router.select_experts(
        hidden_states=torch.empty(3, 1, device="cuda"), router_logits=logits[:3].contiguous()
    )
    assert bits_equal(w1, w[:3].contiguous()) and torch.equal(ids1, ids[:3])
    # R3 replay of the native ids keeps the visible weights bit for bit
    moe = make_moe(K31Config())
    RouterReplay.clear_global_router_replay_instances()
    moe.router_replay = RouterReplay()
    RouterReplay.set_replay_data([ids.long()])
    RouterReplay.set_global_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
    w2, ids2 = moe._replay_route(logits, w, ids)
    assert bits_equal(w2, w) and torch.equal(ids2.long(), ids.long())
    RouterReplay.clear_global_router_replay_instances()


# --------------------------------------------------------------------------- GPU: rollout gates
def _records(dump_dir, key_match):
    out = []
    for f in sorted(glob.glob(os.path.join(dump_dir, "rec_dp*.pt"))):
        r = torch.load(f)
        for key, st in r["rec"].items():
            if key_match(key):
                out.append((r["dp"], key, st, r["maxt"]))
    return out


def _valid_rows(st, maxt):
    n = min(st["cnt"], st["lens"].numel())
    assert st["cnt"] <= st["lens"].numel(), "dump ring wrapped"
    rows = []
    for i in range(n):
        ln = int(st["lens"][i])
        sl = slice(i * maxt, i * maxt + ln)
        keep = st["bufs"]["pad"][sl] == 0
        rows.append({k: v[sl][keep] for k, v in st["bufs"].items()})
    return {k: torch.cat([r[k] for r in rows]) for k in rows[0]} if rows else None


def _rollout_env():
    model_dir, adapter_dir, dump = (
        env_path("K31_MODEL_DIR"),
        env_path("K31_ADAPTER_DIR"),
        env_path("K31_ROLLOUT_DUMP"),
    )
    if None in (model_dir, adapter_dir, dump):
        pytest.skip("K31_MODEL_DIR / K31_ADAPTER_DIR / K31_ROLLOUT_DUMP not set")
    return model_dir, adapter_dir, dump


@gpu
@pytest.mark.parametrize("layer", [1, 2, 3])
def test_moe_block_matches_rollout_dump(layer):
    """Training forward of the whole MoE block (EP1, one batch of every recorded row of every DP rank)
    is bitwise the BI=1 DP4/EP4 CUDA-graph rollout."""

    from k31_moe_fixtures import load_moe_layer, runtime_for

    model_dir, adapter_dir, dump = _rollout_env()
    moe_rows, tail_rows = [], []
    for dp, _, st, maxt in _records(dump, lambda k: k == f"moe:{layer}"):
        moe_rows.append(_valid_rows(st, maxt))
    for dp, _, st, maxt in _records(dump, lambda k: k == f"tail:{layer}"):
        tail_rows.append(_valid_rows(st, maxt))
    assert moe_rows and len(moe_rows) == len(tail_rows)
    rec = {k: torch.cat([r[k] for r in moe_rows]).cuda() for k in moe_rows[0]}
    tail = {k: torch.cat([r[k] for r in tail_rows]).cuda() for k in tail_rows[0]}
    moe = load_moe_layer(model_dir, layer, adapter_dir=adapter_dir)
    with torch.no_grad():
        out, parts = moe.forward_parts(rec["x"], runtime=runtime_for(rec["x"].shape[0]))
    report = {
        "rows": rec["x"].shape[0],
        "logits": mismatch(parts["logits"], rec["logits"]),
        "down": mismatch(parts["down"], rec["down"]),
        "latent": mismatch(parts["latent"], tail["latent"]),
        "normed": mismatch(parts["normed"], tail["normed"]),
        "upmm": mismatch(parts["upmm"], tail["upmm"]),
        "shared": mismatch(parts["shared"], tail["shared"]),
        "out": mismatch(out, rec["out"]),
    }
    print("K31MOE_ROLLOUT", layer, report, flush=True)
    assert all(v == 0 for k, v in report.items() if k != "rows"), report


@gpu
def test_dense_mlp_matches_rollout_dump():
    from k31_moe_fixtures import load_dense_mlp, runtime_for

    model_dir, adapter_dir, dump = _rollout_env()
    rows = [
        _valid_rows(st, maxt)
        for _, _, st, maxt in _records(dump, lambda k: k.endswith("layers.0.mlp"))
    ]
    assert rows
    rec = {k: torch.cat([r[k] for r in rows]).cuda() for k in rows[0]}
    mlp = load_dense_mlp(model_dir, adapter_dir=adapter_dir)
    with torch.no_grad():
        out = mlp(rec["x"], runtime_for(rec["x"].shape[0]))
    print("K31MLP0_ROLLOUT", rec["x"].shape[0], mismatch(out, rec["out"]), flush=True)
    assert bits_equal(out, rec["out"])


@gpu
def test_lora_tensors_are_vllm_stored_values():
    """Training builds exactly vLLM's stored LoRA tensors (A as BF16, B as bf16(B * scaling))."""

    from megatron.lite.primitive.alignment.vllm_fi_w4a16_moe import vllm_stored_lora_b

    from k31_moe_fixtures import _adapter_tensors, adapter_config

    _, adapter_dir, dump = _rollout_env()
    ad = adapter_config(adapter_dir)
    s = ad["lora_alpha"] / ad["r"]
    t = _adapter_tensors(adapter_dir)
    checked = 0
    for f in sorted(glob.glob(os.path.join(dump, "weights_dp*.pt"))):
        w = torch.load(f)
        for name, d in w["mods"].items():
            if "w13_lora_a_stacked" in d:
                layer = int(name.split("layers.")[1].split(".")[0])
                p = f"base_model.model.language_model.model.layers.{layer}.block_sparse_moe.experts."
                e_loc = d["w13_lora_b_stacked"][0].shape[1]
                off = w["ep"] * e_loc
                sl = slice(off, off + e_loc)
                mine = {
                    "w13_lora_a_stacked": (
                        t[p + "w1.lora_A.weight"].reshape(
                            1, 1, *t[p + "w1.lora_A.weight"].shape[1:]
                        ),
                        t[p + "w3.lora_A.weight"].reshape(
                            1, 1, *t[p + "w3.lora_A.weight"].shape[1:]
                        ),
                    ),
                    "w13_lora_b_stacked": (
                        vllm_stored_lora_b(t[p + "w1.lora_B.weight"][sl], s)[None],
                        vllm_stored_lora_b(t[p + "w3.lora_B.weight"][sl], s)[None],
                    ),
                    "w2_lora_a_stacked": (t[p + "w2.lora_A.weight"][sl].to(torch.bfloat16)[None],),
                    "w2_lora_b_stacked": (
                        vllm_stored_lora_b(t[p + "w2.lora_B.weight"][0], s)[None, None],
                    ),
                }
                for key, tup in mine.items():
                    got = d[key]
                    assert len(got) == len(tup), (name, key, len(got))
                    for a, b in zip(got, tup, strict=True):
                        a0 = a[:1]
                        assert a0.shape == b.shape, (name, key, a0.shape, b.shape)
                        assert bits_equal(a0.to(torch.bfloat16), b.to(torch.bfloat16)), (name, key)
                checked += 1
            elif "lora_a_stacked" in d:
                checked += 1
    assert checked > 0
