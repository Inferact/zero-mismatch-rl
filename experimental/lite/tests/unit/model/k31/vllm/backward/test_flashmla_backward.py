"""K31 DSA backward: vendor cuDNN DSA VJP at K31 geometry and the full owner chain.

Geometry: 96 heads, d_qk 576 (absorbed 512 + un-rotated 64), d_v 512, top-k
2048, 656-byte fp8_ds_mla KV, no attention sink.  References are FP32 autograd
graphs on the same visible indices and the dequantized KV (STE contract:
identity through the fp8_ds_mla quantization).  Tolerances fixed before the
first run: cosine >= 0.9999 and relative L2 <= 1e-2 for the vendor VJP;
cosine >= 0.999 and relative L2 <= 3e-2 for the composed chain (bf16
intermediates on the visible path).
"""

from __future__ import annotations

import importlib.util

import pytest
import torch
import torch.nn.functional as F

H, NOPE, ROPE, V, KV_RANK, TOPK = 96, 128, 64, 128, 512, 2048
D, DV = KV_RANK + ROPE, KV_RANK


def _ready() -> bool:
    if not torch.cuda.is_available() or importlib.util.find_spec("cudnn") is None:
        return False
    try:
        from vllm import envs
        from vllm.v1.attention.backends.mla import flashmla_sparse

        return bool(envs.VLLM_BATCH_INVARIANT) and hasattr(flashmla_sparse, "_k31_pin_sched")
    except Exception:  # noqa: BLE001
        return False


requires_k31 = pytest.mark.skipif(
    not _ready(),
    reason="requires CUDA, cuDNN DSA, the K31 vLLM image + overlays, VLLM_BATCH_INVARIANT=1",
)


def _metrics(got: torch.Tensor, ref: torch.Tensor) -> tuple[float, float]:
    got, ref = got.float().flatten(), ref.float().flatten()
    return float(F.cosine_similarity(got, ref, dim=0)), float((got - ref).norm() / ref.norm())


def _random_topk(counts, seed=5):
    """Request-local ascending top-k: full causal prefix below 2048, random subset above."""

    g = torch.Generator().manual_seed(seed)
    rows = []
    for n in counts:
        for t in range(n):
            row = torch.full((TOPK,), -1, dtype=torch.int32)
            if t + 1 <= TOPK:
                row[: t + 1] = torch.arange(t + 1, dtype=torch.int32)
            else:
                row[:] = torch.randperm(t + 1, generator=g)[:TOPK].sort().values.to(torch.int32)
            rows.append(row)
    return torch.stack(rows).cuda()


def _row_chunks(t, size=256):
    """FP32 references in row chunks: keeps every reference tensor below 2^31
    elements (BI mode routes aten bmm to vLLM's Triton kernel, int32-indexed)."""

    return [slice(a, min(a + size, t)) for a in range(0, t, size)]


def _sparse_reference(q, kv, idx, scale):
    sel = kv[idx.clamp_min(0).long()]
    s = (torch.einsum("thd,tkd->thk", q, sel) * scale).masked_fill(
        ~(idx >= 0)[:, None, :], float("-inf")
    )
    return torch.einsum("thk,tkd->thd", torch.softmax(s, -1), sel[..., :DV]), torch.logsumexp(s, -1)


def _setup(counts, seed=0):
    from megatron.lite.model.k31.vllm.dsa import K31DSAMetadataBuilder, initialize_k31_dsa_runtime

    initialize_k31_dsa_runtime(torch.device("cuda"))
    torch.manual_seed(seed)
    t = sum(counts)
    meta = K31DSAMetadataBuilder(device="cuda", max_model_len=5120).build(list(counts))
    q = (torch.randn(t, H, NOPE + ROPE, device="cuda") * 0.5).bfloat16()
    kv_c = torch.randn(t, KV_RANK, device="cuda").bfloat16()
    k_pe = torch.randn(t, ROPE, device="cuda").bfloat16()
    g = torch.randn(t, H * V, device="cuda").bfloat16()
    kv_b = (torch.randn(H * (NOPE + V), KV_RANK, device="cuda") * KV_RANK**-0.5).bfloat16()
    return meta, q, kv_c, k_pe, g, kv_b, _random_topk(counts)


def _dequant_rows(meta, t):
    rows = meta.mla_cache.view(-1, 656)[meta.slot_mapping]
    nope = rows[:, :512].contiguous().view(torch.float8_e4m3fn).float()
    scales = rows[:, 512:528].contiguous().view(torch.float32).repeat_interleave(128, dim=1)
    rope = rows[:, 528:].contiguous().view(torch.bfloat16).float()
    return torch.cat((nope * scales, rope), dim=1)[:t]


@pytest.mark.gpus(1)
@requires_k31
@pytest.mark.parametrize("deterministic", [True, False], ids=["deterministic", "atomic"])
def test_vendor_sparse_backward_matches_fp32_reference(deterministic) -> None:
    from megatron.lite.model.k31.vllm.primitive.attention import (
        default_dsa_adapters,
        default_sparse_backward,
    )

    counts = (300, 2600)
    meta, q, kv_c, k_pe, _g, _kvb, topk = _setup(counts)
    t = sum(counts)
    adapters = default_dsa_adapters()
    adapters.kv_insert(kv_c, k_pe, meta.mla_cache, meta.slot_mapping, meta.k_scale)
    q_cat = torch.cat(
        (torch.randn(t, H, KV_RANK, device="cuda").bfloat16() * 0.3, q[..., NOPE:]), -1
    )
    glob = adapters.sparse.convert_indices(
        meta.req_id_per_token, meta.block_table, topk, block_size=64
    )
    scale = (NOPE + ROPE) ** -0.5
    out, lse = adapters.sparse(
        q_cat,
        meta.mla_cache,
        glob,
        softmax_scale=scale,
        padded_heads=128,
        cache_lens=meta.cache_lens,
        dummy_block_table=meta.dummy_block_table,
    )
    ws = torch.empty(t, D, dtype=torch.bfloat16, device="cuda")
    adapters.upconvert(meta.mla_cache, ws, meta.block_table, meta.workspace_starts)
    assert torch.equal(ws, _dequant_rows(meta, t).bfloat16())
    widx = torch.where(
        topk >= 0, topk + meta.workspace_starts[meta.req_id_per_token.long()][:, None], topk
    )
    grad = torch.randn(t, H, DV, device="cuda").bfloat16()
    dq, dkv = default_sparse_backward(
        q_cat, ws, out, grad, lse, widx, scale, deterministic=deterministic
    )
    assert torch.isfinite(dq).all() and torch.isfinite(dkv).all()
    dq_ref = torch.empty(t, H, D, device="cuda")
    dkv_ref = torch.zeros(t, D, device="cuda")
    for rows in _row_chunks(t):
        with torch.enable_grad():
            qr = q_cat[rows].float().requires_grad_(True)
            kr = ws.float().requires_grad_(True)
            ref, lse_ref = _sparse_reference(qr, kr, widx[rows], scale)
            torch.testing.assert_close(lse[rows], lse_ref.detach(), rtol=0, atol=1e-4)
            dq_c, dkv_c = torch.autograd.grad(ref, (qr, kr), grad[rows].float())
        dq_ref[rows] = dq_c
        dkv_ref += dkv_c
    arch = torch.cuda.get_device_capability()
    for name, got, want in (("dq", dq, dq_ref), ("dkv", dkv.reshape(t, D), dkv_ref)):
        cos, rel = _metrics(got, want)
        print(
            f"K31_DSA_BWD sm_{arch[0]}{arch[1]} vendor det={deterministic} {name} cos={cos:.6f} rel={rel:.3e}"
        )
        assert cos >= 0.9999 and rel <= 1e-2, (cos, rel)
    if deterministic:
        dq2, dkv2 = default_sparse_backward(
            q_cat, ws, out, grad, lse, widx, scale, deterministic=True
        )
        assert torch.equal(dq, dq2) and torch.equal(
            dkv, dkv2
        ), "deterministic VJP must be run-to-run bitwise"


@pytest.mark.gpus(1)
@requires_k31
def test_attention_core_and_gate_gradients_match_fp32_ste_reference() -> None:
    from megatron.lite.model.k31.vllm.primitive.attention import (
        absorbed_weights,
        attention_core,
        default_dsa_adapters,
        output_gate,
    )

    counts = (300, 2600)
    meta, q, kv_c, k_pe, g, kv_b, topk = _setup(counts, seed=1)
    t = sum(counts)
    scale = (NOPE + ROPE) ** -0.5
    w_uk_t, w_uv = absorbed_weights(
        kv_b, kv_lora_rank=KV_RANK, num_heads=H, qk_nope_head_dim=NOPE, v_head_dim=V
    )
    leaves = [x.detach().clone().requires_grad_(True) for x in (q, kv_c, k_pe, g)]
    out = output_gate(
        attention_core(
            default_dsa_adapters(),
            leaves[0],
            leaves[1],
            leaves[2],
            topk,
            meta,
            w_uk_t=w_uk_t,
            w_uv=w_uv,
            softmax_scale=scale,
            qk_nope_head_dim=NOPE,
            padded_heads=128,
        ),
        leaves[3],
    )
    cot = torch.randn_like(out)
    out.backward(cot)
    grads = [x.grad for x in leaves]
    assert all(gr is not None and torch.isfinite(gr).all() for gr in grads)

    deq = _dequant_rows(meta, t)  # STE: the reference sees the dequantized KV value
    ste = deq - torch.cat((kv_c, k_pe), -1).float()
    widx = torch.where(
        topk >= 0, topk + meta.workspace_starts[meta.req_id_per_token.long()][:, None], topk
    )
    y_ref = torch.empty(t, H * V, device="cuda")
    refs = [torch.zeros(x.shape, device="cuda") for x in (q, kv_c, k_pe, g)]
    for chunk in _row_chunks(t):
        with torch.enable_grad():
            qr, gr = (x[chunk].detach().float().requires_grad_(True) for x in (q, g))
            kvr, per = (x.detach().float().requires_grad_(True) for x in (kv_c, k_pe))
            rows = torch.cat((kvr, per), -1) + ste
            ql = torch.einsum("thp,hpl->thl", qr[..., :NOPE], w_uk_t.float())
            o, _ = _sparse_reference(torch.cat((ql, qr[..., NOPE:]), -1), rows, widx[chunk], scale)
            y = torch.einsum("thl,hlv->thv", o, w_uv.float()).reshape(-1, H * V) * torch.sigmoid(gr)
            dq_c, dkv_c, dpe_c, dg_c = torch.autograd.grad(
                y, (qr, kvr, per, gr), cot[chunk].float()
            )
        y_ref[chunk] = y.detach()
        refs[0][chunk] = dq_c
        refs[1] += dkv_c
        refs[2] += dpe_c
        refs[3][chunk] = dg_c
    arch = torch.cuda.get_device_capability()
    cos, rel = _metrics(out.detach(), y_ref)
    print(f"K31_DSA_BWD sm_{arch[0]}{arch[1]} chain forward cos={cos:.6f} rel={rel:.3e}")
    assert cos >= 0.999 and rel <= 3e-2, ("forward", cos, rel)
    for name, got, want in zip(("q", "kv_c", "k_pe", "g"), grads, refs, strict=True):
        cos, rel = _metrics(got, want)
        print(f"K31_DSA_BWD sm_{arch[0]}{arch[1]} chain d{name} cos={cos:.6f} rel={rel:.3e}")
        assert cos >= 0.999 and rel <= 3e-2, (name, cos, rel)


@pytest.mark.gpus(1)
@requires_k31
def test_backward_rejects_nonfinite_vendor_gradients() -> None:
    from megatron.lite.model.k31.vllm.primitive.attention import (
        absorbed_weights,
        attention_core,
        default_dsa_adapters,
    )

    counts = (64,)
    meta, q, kv_c, k_pe, _g, kv_b, topk = _setup(counts, seed=2)
    w_uk_t, w_uv = absorbed_weights(
        kv_b, kv_lora_rank=KV_RANK, num_heads=H, qk_nope_head_dim=NOPE, v_head_dim=V
    )

    def poisoned(q_, kv_, out_, grad_, lse_, idx_, scale_):
        return torch.full_like(q_, float("nan")), torch.zeros_like(kv_)

    leaf = q.detach().clone().requires_grad_(True)
    out = attention_core(
        default_dsa_adapters(),
        leaf,
        kv_c,
        k_pe,
        topk,
        meta,
        w_uk_t=w_uk_t,
        w_uv=w_uv,
        softmax_scale=0.1,
        qk_nope_head_dim=NOPE,
        padded_heads=128,
        backward_op=poisoned,
    )
    with pytest.raises(FloatingPointError):
        out.sum().backward()
