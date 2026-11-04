"""K31 dense / router / LoRA-leaf forwards are the rollout kernels, bit for bit."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from megatron.lite.model.k31.vllm.model import K31Linear, K31LoRASpec
from megatron.lite.model.k31.vllm.primitive import (
    gate_linear,
    visible_linear,
    visible_lora_linear,
    vllm_fold_lora_b,
)
from megatron.lite.model.k31.vllm.runtime_metadata import build_k31_runtime
from megatron.lite.primitive.kernels import vllm_k31
from megatron.lite.primitive.kernels.vllm_k31 import (
    BILinearAdapter,
    BIRouterAdapter,
    LoRALeafAdapter,
    PunicaLoRADeltaAdapter,
)

# Every bf16 dense linear of the TP=1 deployment: (K, N).
TP1_SHAPES = {
    "kda_in_proj_qkvgfab": (7168, 49376),
    "kda_f_b": (128, 12288),
    "kda_dsa_o_proj": (12288, 7168),
    "dsa_fused_qkv_a": (7168, 2112),
    "dsa_fused_qkv_a_g": (7168, 14400),
    "dsa_q_b": (1536, 18432),
    "dsa_kv_b": (512, 24576),
    "dsa_g_shared_gate_up": (7168, 12288),
    "indexer_wq_b": (1536, 4096),
    "indexer_wk": (7168, 128),
    "indexer_weights_proj": (7168, 32),
    "shared_down": (6144, 7168),
    "latent_down": (7168, 3584),
    "latent_up": (3584, 7168),
    "dense_gate_up": (7168, 67584),
    "dense_down": (33792, 7168),
    "lm_head": (7168, 163840),
}
# Former TP4 shards (column / row) kept as plan regressions.
TP4_SHAPES = {
    "tp4_kda_in": (7168, 12448),
    "tp4_o_proj": (3072, 7168),
    "tp4_q_b": (1536, 4608),
}
ROUTER_SHAPE = (7168, 896)

# LoRA-wrapped projections: (K, slice sizes).
LORA_SHAPES = {
    "kda_in_proj_qkvgfab": (7168, (12288, 12288, 12288, 12288, 128, 96)),
    "kda_f_b": (128, (12288,)),
    "kda_dsa_o_proj": (12288, (7168,)),
    "dsa_fused_qkv_a": (7168, (1536, 576)),
    "dsa_q_b": (1536, (18432,)),
    "dsa_g": (7168, (12288,)),
    "shared_gate_up": (7168, (6144, 6144)),
    "shared_down": (6144, (7168,)),
    "dense_gate_up": (7168, (33792, 33792)),
    "dense_down": (33792, (7168,)),
}
M_VALUES = (1, 3, 64, 513)
PROBE_M = 513


def _x(m: int, k: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(m, k, device="cuda", dtype=torch.bfloat16, generator=g)


def _w(n: int, k: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(n, k, device="cuda", generator=g) * k**-0.5).to(torch.bfloat16)


# ----------------------------------------------------------------------------- CPU contract
def test_adapter_rejects_bad_dtype_before_kernel_lookup(monkeypatch) -> None:
    lookup = Mock()
    monkeypatch.setattr(vllm_k31, "_symbol", lookup)
    with pytest.raises(TypeError, match="dtype"):
        BILinearAdapter()(torch.zeros(2, 8), torch.zeros(4, 8, dtype=torch.bfloat16))
    lookup.assert_not_called()


def test_partially_targeted_packed_projection_is_rejected() -> None:
    spec = K31LoRASpec(rank=2, alpha=2.0, max_rank=2, targets=frozenset({"q_a_proj"}))
    with pytest.raises(ValueError, match="partially"):
        K31Linear(8, (4, 4), ("q_a_proj", "kv_a_proj_with_mqa"), lora=spec)


def test_fold_matches_vllm_optimize_arithmetic() -> None:
    b = torch.randn(64, 8).to(torch.bfloat16)
    expected = b.clone()
    expected *= 2.0 / 3.0  # LoRALayerWeights.optimize: lora_b *= scaling (bf16 in place)
    assert torch.equal(vllm_fold_lora_b(b, 2.0 / 3.0), expected)


# ----------------------------------------------------------------------------- GPU gates
@pytest.mark.gpus(1, min_architecture="blackwell")
@pytest.mark.parametrize("name", sorted(TP1_SHAPES) + sorted(TP4_SHAPES))
def test_dense_linear_is_rollout_kernel_and_batch_invariant(k31_bi, name) -> None:
    from vllm.model_executor.determinism.batch_invariant import linear_batch_invariant

    k, n = {**TP1_SHAPES, **TP4_SHAPES}[name]
    w = _w(n, k, 1)
    big = _x(PROBE_M, k, 2)
    ref_big = linear_batch_invariant(big, w, None)
    assert torch.equal(ref_big, k31_bi.bi_linear(big, w))
    for m in M_VALUES:
        x = big[:m].clone()
        out = visible_linear(BILinearAdapter(), x, w)
        assert out.dtype == torch.bfloat16
        assert torch.equal(out, linear_batch_invariant(x, w, None)), m
        assert torch.equal(out, ref_big[:m]), f"batch variance at M={m}"


@pytest.mark.gpus(1, min_architecture="blackwell")
def test_router_is_rollout_bi_router_fp32(k31_bi) -> None:
    k, n = ROUTER_SHAPE
    w = _w(n, k, 3)
    big = _x(PROBE_M, k, 4)
    ref = k31_bi.bi_router(big, w)
    assert ref.dtype == torch.float32
    for m in M_VALUES:
        out = gate_linear(BIRouterAdapter(), big[:m].clone(), w)
        assert torch.equal(out, ref[:m]), m


def _lora_factors(k, sizes, rank, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    a = tuple(
        (torch.randn(rank, k, device="cuda", generator=g) * k**-0.5).to(torch.bfloat16)
        for _ in sizes
    )
    b = tuple(
        (torch.randn(n, rank, device="cuda", generator=g) * 0.02).to(torch.bfloat16)
        for n in sizes
    )
    return a, b


@pytest.mark.gpus(1, min_architecture="blackwell")
@pytest.mark.parametrize("name", sorted(LORA_SHAPES))
def test_lora_leaf_is_rollout_f32add_with_punica_delta(k31_bi, name) -> None:
    from megatron.lite.model.k31.vllm.primitive.linear import LoRAProjection

    k, sizes = LORA_SHAPES[name]
    rank, alpha, max_rank = 32, 64.0, 32
    lora = LoRAProjection(
        k, [f"s{i}" for i in range(len(sizes))], sizes, rank=rank, alpha=alpha, max_rank=max_rank
    ).cuda()
    a, b = _lora_factors(k, sizes, rank, 5)
    with torch.no_grad():
        for i, name_ in enumerate(lora.slice_names):
            lora.lora_A[name_].copy_(a[i])
            lora.lora_B[name_].copy_(b[i])
    w = _w(sum(sizes), k, 6)
    big = _x(PROBE_M, k, 7)
    runtime_big = build_k31_runtime([PROBE_M], "cuda", with_lora=True)
    out_big = visible_lora_linear(
        PunicaLoRADeltaAdapter(), LoRALeafAdapter(), big, w, lora, runtime_big.lora_meta
    )
    # Leaf identity (overlay contract): fused leaf == fp32 plan sum + fp32(delta), one rounding.
    a_st = tuple(t.view(1, 1, rank, k) for t in a)
    b_st = tuple(vllm_fold_lora_b(t, lora.scaling).view(1, 1, t.shape[0], rank) for t in b)
    delta = PunicaLoRADeltaAdapter()(big, a_st, b_st, sizes, runtime_big.lora_meta)
    unfused = (k31_bi.linear_f32(big, w) + delta.float()).to(torch.bfloat16)
    assert torch.equal(out_big, unfused)
    assert torch.equal(out_big, k31_bi.lora_f32add(big, w, delta))
    # delta is the bf16 LoRA branch (sanity against fp32 math, not a bitwise contract).
    ref = torch.cat(
        [(big.float() @ a[i].float().T) @ b_st[i][0, 0].float().T for i in range(len(sizes))], 1
    )
    assert (delta.float() - ref).abs().max() <= 2e-2 * ref.abs().max() + 1e-3
    # Batch invariance of the whole leaf (punica shrink/expand + f32add).
    for m in M_VALUES:
        runtime = build_k31_runtime([m], "cuda", with_lora=True)
        out = visible_lora_linear(
            PunicaLoRADeltaAdapter(), LoRALeafAdapter(), big[:m].clone(), w, lora, runtime.lora_meta
        )
        assert torch.equal(out, out_big[:m]), f"batch variance at M={m}"


class TestPunicaPointerCache:
    """Training stacks fresh bf16 LoRA factors every forward; the vLLM pointer cache must not keep them.

    Without the fix, the cache pinned one stack copy per forward (~9.5 GiB per GPU after one step) and the
    second rollout ran out of memory.
    """

    @pytest.mark.gpus(1, min_architecture="blackwell")
    @pytest.mark.parametrize("name", ["kda_f_b", "shared_gate_up"], ids=["one-slice", "two-slices"])
    def test_repeated_forwards_leave_no_cache_entries(self, k31_bi, name) -> None:
        from vllm.lora.ops.triton_ops import utils as punica_utils

        from megatron.lite.model.k31.vllm.primitive.linear import LoRAProjection

        k, sizes = LORA_SHAPES[name]
        lora = LoRAProjection(
            k, [f"s{i}" for i in range(len(sizes))], sizes, rank=32, alpha=64.0, max_rank=32
        ).cuda()
        a, b = _lora_factors(k, sizes, 32, 5)
        with torch.no_grad():
            for i, name_ in enumerate(lora.slice_names):
                lora.lora_A[name_].copy_(a[i])
                lora.lora_B[name_].copy_(b[i])
        w = _w(sum(sizes), k, 6)
        x = _x(64, k, 7)
        runtime = build_k31_runtime([64], "cuda", with_lora=True)
        n_a, n_b = len(punica_utils._LORA_A_PTR_DICT), len(punica_utils._LORA_B_PTR_DICT)
        first = visible_lora_linear(
            PunicaLoRADeltaAdapter(), LoRALeafAdapter(), x, w, lora, runtime.lora_meta
        )
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        for _ in range(20):
            out = visible_lora_linear(
                PunicaLoRADeltaAdapter(), LoRALeafAdapter(), x, w, lora, runtime.lora_meta
            )
            assert torch.equal(out, first)
            del out
        torch.cuda.synchronize()
        assert len(punica_utils._LORA_A_PTR_DICT) == n_a
        assert len(punica_utils._LORA_B_PTR_DICT) == n_b
        assert torch.cuda.memory_allocated() <= base


@pytest.mark.gpus(1, min_architecture="blackwell")
def test_k31_linear_dispatches_lora_only_when_targeted(k31_bi) -> None:
    spec = K31LoRASpec(rank=4, alpha=8.0, max_rank=8, targets=frozenset({"o_proj"}))
    plain = K31Linear(1536, (512,), ("kv_b_proj",), lora=spec).cuda()
    wrapped = K31Linear(1536, (512,), ("o_proj",), lora=spec).cuda()
    assert plain.lora is None and wrapped.lora is not None
    x = _x(5, 1536, 8)
    with torch.no_grad():
        wrapped.weight.copy_(_w(512, 1536, 9))
        plain.weight.copy_(wrapped.weight)
    runtime = build_k31_runtime([5], "cuda", with_lora=True)
    # B = 0 -> delta = 0 -> leaf == bf16(fp32 sum) == bi_linear at K <= 3584 (S = 1).
    assert torch.equal(wrapped(x, runtime), plain(x))
    with pytest.raises(RuntimeError, match="lora_meta"):
        wrapped(x, SimpleNamespace(lora_meta=None))
