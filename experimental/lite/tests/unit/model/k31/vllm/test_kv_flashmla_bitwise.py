"""K31 DSA forward: adapters are the rollout's exact calls; training == rollout bitwise.

GPU tests need the K31 image with the deployed overlay chain (ds4-indexer-topk,
ds4-sparse-mla, ds4-sparse-mla-allow, sparse-mla-tp1, bi-gemm-r5, ...) and
``VLLM_BATCH_INVARIANT=1``.  The rollout gate additionally needs a dump from a
real BI=1 vLLM run (``K31_DSA_ROLLOUT_DUMP=<dir>``; produced by the dsa-dump
overlay, CUDA graph FULL, force_mqa, MBT 1024).
"""

from __future__ import annotations

import glob
import importlib.util
import os
from unittest.mock import Mock

import pytest
import torch

from megatron.lite.primitive.kernels import vllm_k31
from megatron.lite.primitive.kernels.vllm_k31 import (
    DSAAbsorbBmmAdapter,
    DSAMLAKVInsertAdapter,
    DSASparseMLAAdapter,
)

H, NOPE, ROPE, V, KV_RANK, TOPK = 96, 128, 64, 128, 512, 2048
IDX_HEADS, IDX_DIM, HIDDEN, Q_RANK = 32, 128, 7168, 1536


def test_kv_insert_cpu_contract_calls_fp8_ds_mla(monkeypatch) -> None:
    kernel = Mock()
    monkeypatch.setattr(vllm_k31, "_symbol", lambda module, name: kernel)
    kv = torch.zeros(3, 512, dtype=torch.bfloat16)
    pe = torch.zeros(3, 64, dtype=torch.bfloat16)
    cache = torch.zeros(1, 64, 656, dtype=torch.uint8)
    slots = torch.arange(3, dtype=torch.int64)
    scale = torch.ones(())
    assert DSAMLAKVInsertAdapter()(kv, pe, cache, slots, scale) is cache
    kernel.assert_called_once()
    assert kernel.call_args.kwargs == {"kv_cache_dtype": "fp8_ds_mla", "scale": scale}
    with pytest.raises(ValueError):
        DSAMLAKVInsertAdapter()(kv, pe, torch.zeros(1, 64, 584, dtype=torch.uint8), slots, scale)


def test_sparse_cpu_contract_pins_plan_and_pads_heads(monkeypatch) -> None:
    import types

    calls = {}
    sched = types.SimpleNamespace()

    def lookup(_module, name):
        if name == "get_mla_metadata":
            return lambda: (sched, None)
        if name == "_k31_pin_sched":
            return lambda s, q, idx: calls.setdefault("pin", (s, tuple(q.shape), tuple(idx.shape)))

        def kernel(**kwargs):
            calls["kernel"] = kwargs
            q = kwargs["q"]
            return torch.zeros(1, q.shape[1], q.shape[2], 512), torch.zeros(
                1, q.shape[2], q.shape[1]
            )

        return kernel

    monkeypatch.setattr(vllm_k31, "_symbol", lookup)
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    if importlib.util.find_spec("vllm") is None:
        pytest.skip("vllm.envs is required to read VLLM_BATCH_INVARIANT")
    q = torch.zeros(5, H, 576, dtype=torch.bfloat16)
    idx = torch.zeros(5, TOPK, dtype=torch.int32)
    out, lse = DSASparseMLAAdapter()(
        q,
        torch.zeros(2, 64, 656, dtype=torch.uint8),
        idx,
        softmax_scale=0.1,
        padded_heads=128,
        cache_lens=torch.ones(1, dtype=torch.int32),
        dummy_block_table=torch.zeros(1, 1, dtype=torch.int32),
    )
    assert calls["pin"][1] == (1, 5, 128, 576) and calls["pin"][2] == (1, 5, TOPK)
    assert calls["kernel"]["is_fp8_kvcache"] and calls["kernel"]["head_dim_v"] == 512
    assert out.shape == (5, H, 512) and lse.shape == (5, H)


# ---------------------------------------------------------------------------
# GPU: K31 image + overlays
# ---------------------------------------------------------------------------


def _k31_overlays_ready() -> bool:
    if not torch.cuda.is_available() or importlib.util.find_spec("vllm") is None:
        return False
    try:
        from vllm import envs
        from vllm.v1.attention.backends.mla import flashmla_sparse

        return bool(envs.VLLM_BATCH_INVARIANT) and hasattr(flashmla_sparse, "_k31_pin_sched")
    except Exception:  # noqa: BLE001
        return False


requires_k31 = pytest.mark.skipif(
    not _k31_overlays_ready(),
    reason="requires CUDA, the K31 vLLM image with the BI overlay chain and VLLM_BATCH_INVARIANT=1",
)


def _runtime():
    from megatron.lite.model.k31.vllm.dsa import initialize_k31_dsa_runtime

    initialize_k31_dsa_runtime(torch.device("cuda"))


def _synthetic_layer(seed: int = 0):
    g = torch.Generator(device="cpu").manual_seed(seed)

    def rnd(*shape, std):
        return (torch.randn(*shape, generator=g) * std).to(torch.bfloat16).cuda()

    weights = {
        "wq": rnd(IDX_HEADS * IDX_DIM, Q_RANK, std=Q_RANK**-0.5),
        "wk": rnd(IDX_DIM, HIDDEN, std=HIDDEN**-0.5),
        "wv": rnd(IDX_HEADS, HIDDEN, std=HIDDEN**-0.5),
        "k_norm_weight": torch.ones(IDX_DIM, device="cuda")
        + 0.1 * torch.randn(IDX_DIM, generator=g).cuda(),
        "k_norm_bias": 0.1 * torch.randn(IDX_DIM, generator=g).cuda(),
        "k_norm_eps": 1e-6,
    }
    kv_b = rnd(H * (NOPE + V), KV_RANK, std=KV_RANK**-0.5)
    return weights, kv_b


def _synthetic_inputs(counts, seed: int = 1):
    g = torch.Generator(device="cpu").manual_seed(seed)
    t = sum(counts)

    def rnd(*shape, std=1.0):
        return (torch.randn(*shape, generator=g) * std).to(torch.bfloat16).cuda()

    return {
        "hidden": rnd(t, HIDDEN),
        "q_c": rnd(t, Q_RANK),
        "q": rnd(t, H, NOPE + ROPE, std=0.5),
        "kv_c_normed": rnd(t, KV_RANK),
        "k_pe": rnd(t, ROPE),
        "g": rnd(t, H * V),
    }


def _forward(counts, inputs, weights, kv_b, scale, *, max_model_len=5120):
    from megatron.lite.model.k31.vllm.dsa import K31DSAMetadataBuilder
    from megatron.lite.model.k31.vllm.primitive.attention import (
        absorbed_weights,
        attention_core,
        default_dsa_adapters,
        indexer_topk,
        output_gate,
    )

    adapters = default_dsa_adapters()
    meta = K31DSAMetadataBuilder(device="cuda", max_model_len=max_model_len).build(list(counts))
    topk = indexer_topk(
        adapters.indexer,
        inputs["hidden"],
        inputs["q_c"],
        weights=weights,
        num_heads=IDX_HEADS,
        head_dim=IDX_DIM,
        topk=TOPK,
        metadata=meta.indexer,
    ).clone()
    w_uk_t, w_uv = absorbed_weights(
        kv_b, kv_lora_rank=KV_RANK, num_heads=H, qk_nope_head_dim=NOPE, v_head_dim=V
    )
    attn = attention_core(
        adapters,
        inputs["q"],
        inputs["kv_c_normed"],
        inputs["k_pe"],
        topk,
        meta,
        w_uk_t=w_uk_t,
        w_uv=w_uv,
        softmax_scale=scale,
        qk_nope_head_dim=NOPE,
        padded_heads=128,
    )
    return topk, attn, output_gate(attn, inputs["g"])


def _split(x, counts):
    return list(torch.split(x, list(counts)))


@pytest.mark.gpus(1)
@requires_k31
def test_fp8_ds_mla_insert_and_upconvert_bitwise() -> None:
    from vllm import _custom_ops as ops

    from megatron.lite.primitive.kernels.vllm_k31 import DSAFP8KVUpconvertAdapter

    torch.manual_seed(3)
    t = 130
    kv = torch.randn(t, 512, device="cuda").bfloat16()
    pe = torch.randn(t, 64, device="cuda").bfloat16()
    slots = torch.randperm(4 * 64, device="cuda")[:t].sort().values
    ref = torch.zeros(4, 64, 656, dtype=torch.uint8, device="cuda")
    cand = ref.clone()
    ops.concat_and_cache_mla(
        kv, pe, ref, slots, kv_cache_dtype="fp8_ds_mla", scale=torch.ones((), device="cuda")
    )
    DSAMLAKVInsertAdapter()(kv, pe, cand, slots, torch.ones((), device="cuda"))
    assert torch.equal(ref, cand)
    rows = cand.view(-1, 656)[slots]
    nope = rows[:, :512].contiguous().view(torch.float8_e4m3fn).float()
    scales = rows[:, 512:528].contiguous().view(torch.float32).repeat_interleave(128, dim=1)
    manual = torch.cat(
        (nope * scales, rows[:, 528:].contiguous().view(torch.bfloat16).float()), 1
    ).bfloat16()
    bt = (slots // 64).unique_consecutive().to(torch.int32)
    # one request whose rows are exactly the inserted slots (contiguous positions per block)
    contiguous = torch.arange(t, device="cuda")
    ref2 = torch.zeros_like(ref)
    DSAMLAKVInsertAdapter()(kv, pe, ref2, contiguous, torch.ones((), device="cuda"))
    ws = torch.empty(t, 576, dtype=torch.bfloat16, device="cuda")
    DSAFP8KVUpconvertAdapter()(
        ref2,
        ws,
        torch.arange(3, dtype=torch.int32, device="cuda")[None],
        torch.zeros(1, dtype=torch.int32, device="cuda"),
    )
    rows2 = ref2.view(-1, 656)[:t]
    nope2 = rows2[:, :512].contiguous().view(torch.float8_e4m3fn).float()
    sc2 = rows2[:, 512:528].contiguous().view(torch.float32).repeat_interleave(128, dim=1)
    manual2 = torch.cat(
        (nope2 * sc2, rows2[:, 528:].contiguous().view(torch.bfloat16).float()), 1
    ).bfloat16()
    assert torch.equal(ws, manual2)
    assert torch.equal(manual[:, 512:], pe) and bt.numel() >= 1


@pytest.mark.gpus(1)
@requires_k31
def test_absorb_bmm_adapter_matches_vllm_call_layout() -> None:
    from vllm.model_executor.determinism.batch_invariant import bmm_batch_invariant

    from megatron.lite.model.k31.vllm.primitive.attention import absorbed_weights

    _runtime()
    _w, kv_b = _synthetic_layer()
    w_uk_t, w_uv = absorbed_weights(
        kv_b, kv_lora_rank=KV_RANK, num_heads=H, qk_nope_head_dim=NOPE, v_head_dim=V
    )
    q = torch.randn(37, H, NOPE + ROPE, device="cuda").bfloat16()
    q_nope = q[..., :NOPE].transpose(0, 1)
    ref = q.new_empty((37, H, KV_RANK))
    bmm_batch_invariant(q_nope, w_uk_t, out=ref.transpose(0, 1))
    assert torch.equal(DSAAbsorbBmmAdapter().absorb(q[..., :NOPE], w_uk_t), ref)
    attn = torch.randn(37, H, KV_RANK, device="cuda").bfloat16()
    out_ref = attn.new_empty((37, H * V))
    bmm_batch_invariant(attn.transpose(0, 1), w_uv, out=out_ref.view(-1, H, V).transpose(0, 1))
    out = attn.new_empty((37, H * V))
    assert torch.equal(DSAAbsorbBmmAdapter().v_up(attn, w_uv, out), out_ref)
    # row-local: a single row equals its slice of the batch
    assert torch.equal(DSAAbsorbBmmAdapter().absorb(q[5:6, :, :NOPE], w_uk_t), ref[5:6])


@pytest.mark.gpus(1)
@requires_k31
@pytest.mark.parametrize("counts", [(300, 2600), (1500,), (2049, 77, 2600)])
def test_training_forward_is_packing_invariant(counts) -> None:
    """Each request's indexer top-k, attention and gated output are bitwise equal
    alone, packed, and packed in reverse order (lengths below and above 2048)."""

    _runtime()
    weights, kv_b = _synthetic_layer()
    inputs = _synthetic_inputs(counts)
    scale = (NOPE + ROPE) ** -0.5
    packed = _forward(counts, inputs, weights, kv_b, scale)
    per_req = [_split(x, counts) for x in packed]
    for r, n in enumerate(counts):
        sl = slice(sum(counts[:r]), sum(counts[:r]) + n)
        alone = _forward((n,), {k: v[sl] for k, v in inputs.items()}, weights, kv_b, scale)
        for name, got, want in zip(
            ("topk", "attn", "gated"), alone, (p[r] for p in per_req), strict=True
        ):
            assert torch.equal(got, want), f"request {r} ({n} tokens): {name} depends on packing"
    rev = tuple(reversed(counts))
    rev_inputs = {k: torch.cat(list(reversed(_split(v, counts)))) for k, v in inputs.items()}
    reversed_out = [
        list(reversed(_split(x, rev))) for x in _forward(rev, rev_inputs, weights, kv_b, scale)
    ]
    for pk, rv in zip(per_req, reversed_out, strict=True):
        for a, b in zip(pk, rv, strict=True):
            assert torch.equal(a, b)
    topk = per_req[0]
    for r, n in enumerate(counts):
        valid = (topk[r] >= 0).sum(-1)
        expect = torch.clamp(torch.arange(1, n + 1, device="cuda"), max=TOPK)
        assert torch.equal(valid, expect), "row t must keep min(t+1, 2048) causal keys"
        assert bool((topk[r] < torch.arange(1, n + 1, device="cuda")[:, None]).all())


def _rollout_dump_dir() -> str | None:
    path = os.environ.get("K31_DSA_ROLLOUT_DUMP")
    return path if path and os.path.isdir(path) else None


@pytest.mark.gpus(1)
@requires_k31
@pytest.mark.skipif(
    _rollout_dump_dir() is None, reason="set K31_DSA_ROLLOUT_DUMP to a dsa-dump directory"
)
def test_training_forward_matches_rollout_dump_bitwise() -> None:
    """Gate: training primitive == real BI=1 rollout (CUDA graph, force_mqa, chunked
    prefill + decode), per case alone and all cases packed, forward and reverse."""

    from megatron.lite.model.k31.vllm.primitive.attention import absorbed_weights

    _runtime()
    root = _rollout_dump_dir()
    wts = torch.load(os.path.join(root, "weights.pt"))
    files = sorted(glob.glob(os.path.join(root, "case*.pt")))
    assert files, "empty dump"
    report = []
    for prefix, w in wts.items():
        weights = {
            "wq": w["wq_index"].cuda(),
            "wk": w["wk_index"].cuda(),
            "wv": w["wv_index"].cuda(),
            "k_norm_weight": w["k_norm_weight"].cuda(),
            "k_norm_bias": w["k_norm_bias"].cuda(),
            "k_norm_eps": w["k_norm_eps"],
        }
        kv_b = w["kv_b_proj"].cuda()
        w_uk_t, w_uv = absorbed_weights(
            kv_b, kv_lora_rank=KV_RANK, num_heads=H, qk_nope_head_dim=NOPE, v_head_dim=V
        )
        assert torch.equal(w_uk_t.cpu(), w["W_UK_T"]) and torch.equal(w_uv.cpu(), w["W_UV"])
        assert tuple(w_uk_t.stride()) == tuple(w["W_UK_T_stride"]) and tuple(
            w_uv.stride()
        ) == tuple(w["W_UV_stride"])
        cases = []
        for f in files:
            rec = torch.load(f)["layers"][prefix]
            valid = rec["slot"] >= 0
            pos = rec["pos"][valid]
            order = torch.argsort(pos, stable=True)
            pos = pos[order]
            assert torch.equal(
                pos, torch.arange(pos.numel())
            ), f"{f}: dump rows are not one request 0..L-1"
            cases.append({k: v[valid][order].cuda() for k, v in rec.items()})
        counts = tuple(c["pos"].numel() for c in cases)

        def inputs_of(cs):
            return {
                "hidden": torch.cat([c["hidden"] for c in cs]),
                "q_c": torch.cat([c["q_c"] for c in cs]),
                "q": torch.cat([c["q"] for c in cs]).view(-1, H, NOPE + ROPE),
                "kv_c_normed": torch.cat([c["kv_c_normed"] for c in cs]),
                "k_pe": torch.cat([c["k_pe"] for c in cs]),
                "g": torch.cat([c["g"] for c in cs]),
            }

        layouts = [[i] for i in range(len(cases))] + [
            list(range(len(cases))),
            list(reversed(range(len(cases)))),
        ]
        for layout in layouts:
            cs = [cases[i] for i in layout]
            got = _forward(
                tuple(counts[i] for i in layout), inputs_of(cs), weights, kv_b, w["scale"]
            )
            for name, key, tensor in zip(
                ("topk", "attn", "gated"), ("topk", "attn", "gated"), got, strict=True
            ):
                parts = _split(tensor, [counts[i] for i in layout])
                for i, part in zip(layout, parts, strict=True):
                    want = cases[i][key]
                    same = part == want
                    rows_bad = int((~same.reshape(part.shape[0], -1).all(-1)).sum())
                    report.append((prefix, tuple(layout), i, name, rows_bad))
    bad = [r for r in report if r[-1]]
    print("rollout gate rows checked:", len(report), "mismatching entries:", bad)
    assert not bad, f"training != rollout: {bad}"


def _load_dump_cases(root: str, prefix: str):
    cases = []
    for f in sorted(glob.glob(os.path.join(root, "case*.pt"))):
        rec = torch.load(f)["layers"][prefix]
        valid = rec["slot"] >= 0
        pos = rec["pos"][valid]
        order = torch.argsort(pos, stable=True)
        assert torch.equal(pos[order], torch.arange(pos.numel())), f"{f}: not one request"
        cases.append({k: v[valid][order].cuda() for k, v in rec.items()})
    return cases


@pytest.mark.gpus(1)
@requires_k31
@pytest.mark.skipif(
    _rollout_dump_dir() is None or not os.path.isdir(os.environ.get("K31_DSA_CKPT", "")),
    reason="set K31_DSA_ROLLOUT_DUMP and K31_DSA_CKPT (the checkpoint the dump was taken from)",
)
def test_dsa_layer_forward_matches_rollout_dump_bitwise() -> None:
    """Gate: the whole ``K31DSAAttention.forward`` slot (hidden -> o_proj output),
    loaded through the skeleton checkpoint loader, equals the BI=1 rollout layer
    output bitwise, per case alone and all cases packed forward / reverse."""

    from torch import nn

    from megatron.lite.model.k31.config import K31Config
    from megatron.lite.model.k31.vllm.checkpoint import load_hf_weights
    from megatron.lite.model.k31.vllm.dsa import K31DSAAttention, build_dsa_metadata
    from megatron.lite.model.k31.vllm.primitive.norm import rms_norm
    from megatron.lite.model.k31.vllm.runtime_metadata import build_k31_runtime
    from megatron.lite.primitive.kernels.vllm_k31 import RMSNormAdapter
    from megatron.lite.primitive.parallel import ParallelState

    _runtime()
    root = _rollout_dump_dir()
    ckpt = os.environ["K31_DSA_CKPT"]
    cfg = K31Config.from_hf(ckpt)
    ((prefix, w),) = torch.load(os.path.join(root, "weights.pt")).items()
    layer_idx = int(prefix.split("layers.")[1].split(".")[0])
    assert cfg.is_dsa_layer(layer_idx)
    attn = K31DSAAttention(
        cfg, layer_idx=layer_idx, ps=None, selected_stages=frozenset(), lora=None
    )
    holder = nn.Module()
    holder.self_attn = attn
    container = nn.Module()
    container.layers = nn.ModuleDict({str(layer_idx): holder})
    load_hf_weights(container, ckpt, cfg, ParallelState())
    attn.cuda()
    # Same frozen tensors as the rollout engine held.
    assert torch.equal(attn.kv_b_proj.weight.cpu(), w["kv_b_proj"])
    assert torch.equal(attn.indexer.wq_b.weight.cpu(), w["wq_index"])
    assert torch.equal(attn.indexer.wk.weight.cpu(), w["wk_index"])
    assert torch.equal(attn.indexer.weights_proj.weight.cpu(), w["wv_index"])
    assert torch.equal(attn.indexer.k_norm.weight.float().cpu(), w["k_norm_weight"].float())
    assert w["k_norm_eps"] == cfg.rms_norm_eps

    cases = _load_dump_cases(root, prefix)
    counts = [c["pos"].numel() for c in cases]
    layouts = [[i] for i in range(len(cases))] + [
        list(range(len(cases))),
        list(reversed(range(len(cases)))),
    ]
    report = []
    eps = cfg.rms_norm_eps
    with torch.no_grad():
        for layout in layouts:
            lens = [counts[i] for i in layout]
            hidden = torch.cat([cases[i]["hidden"] for i in layout])
            runtime = build_k31_runtime(lens, "cuda", with_lora=False)
            meta = build_dsa_metadata(runtime, max_model_len=5120, layer_name=prefix)
            # Diagnostics for the projection front-end (same calls as forward).
            qkv = attn.fused_qkv_a_proj(hidden, runtime)
            q_c, kv_lora = qkv.split([cfg.q_lora_rank, cfg.kv_lora_rank + ROPE], dim=-1)
            q_c = rms_norm(RMSNormAdapter(), q_c, attn.q_a_layernorm.weight, eps)
            kv_c = rms_norm(
                RMSNormAdapter(), kv_lora[:, : cfg.kv_lora_rank], attn.kv_a_layernorm.weight, eps
            )
            stages = {
                "q_c": q_c,
                "kv_c_normed": kv_c,
                "q": attn.q_b_proj(q_c, runtime),
                "g": attn.g_proj(hidden, runtime),
                "o": attn(hidden, metadata=meta, runtime=runtime),
            }
            for name, tensor in stages.items():
                for i, part in zip(layout, torch.split(tensor, lens), strict=True):
                    want = cases[i][name].reshape(part.shape)
                    rows_bad = int((~(part == want).reshape(part.shape[0], -1).all(-1)).sum())
                    report.append((tuple(layout), i, name, rows_bad))
    bad = [r for r in report if r[-1]]
    print("layer gate entries checked:", len(report), "mismatching entries:", bad)
    assert not bad, f"training DSA layer != rollout: {bad}"
