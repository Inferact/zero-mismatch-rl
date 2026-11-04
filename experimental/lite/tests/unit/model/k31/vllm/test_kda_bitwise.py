"""K31 KDA G1 gate: the training forward (whole prefill through the CF-BI path) is bitwise equal to what the BI=1
rollout produced token by token (decode engine under CUDA graphs) for the same tokens.

The rollout reference is a real vLLM run (deploy overlay chain + dump hook, FULL graphs,
LoRA adapter, slice4 KDA layers 1-3). Point K31_KDA_G1_DUMP at the dump directory and K31_KDA_G1_CASES at its
cases.json; optional K31_KDA_G1_REPORT receives a JSON summary.
"""

from __future__ import annotations

import importlib.util
import json
import os

import pytest
import torch
import torch.nn.functional as F

from megatron.lite.model.k31.vllm.primitive.kda import (
    KDAKernelConfig,
    causal_conv_silu,
    kda_core,
    kda_visible,
)
from megatron.lite.primitive.kernels.vllm_k31 import _kda_cu_seqlens

import _kda_dump as dump  # noqa: E402


# ---------------------------------------------------------------------------------------------------------------
# CPU contract tests
# ---------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"num_heads": 4, "head_dim": 64}, "head_dim"),
        ({"num_heads": 4, "conv_width": 3}, "conv width"),
        ({"num_heads": 4, "gate_lower_bound": 0.0}, "gate_lower_bound"),
        ({"num_heads": 4, "norm_method": "forward_hip"}, "norm_method"),
    ],
    ids=["head-dim", "conv-width", "lower-bound", "norm-method"],
)
def test_kernel_config_rejects_non_deploy_geometry(kwargs, error) -> None:
    with pytest.raises(ValueError, match=error):
        KDAKernelConfig(**kwargs)


@pytest.mark.parametrize(
    "cu",
    [[1, 4], [0], [0, 3, 3], [0, 5, 2]],
    ids=["nonzero-start", "empty", "empty-seq", "decreasing"],
)
def test_cu_seqlens_validation(cu) -> None:
    with pytest.raises(ValueError, match="cu_seqlens"):
        _kda_cu_seqlens(torch.tensor(cu))


@pytest.mark.parametrize("lens", [[1], [3, 1, 7], [4, 4]], ids=["one-token", "ragged", "equal"])
def test_conv_graph_restarts_at_sequence_boundaries(lens) -> None:
    torch.manual_seed(0)
    C, W = 6, 4
    x = torch.randn(sum(lens), C, dtype=torch.float64)
    w = torch.randn(C, W, dtype=torch.float64)
    cu = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()])
    got = causal_conv_silu(x, w, cu)
    want = []
    for s, e in zip(cu[:-1].tolist(), cu[1:].tolist()):
        seq = x[s:e].T.unsqueeze(0)  # [1, C, L]
        y = F.conv1d(F.pad(seq, (W - 1, 0)), w.unsqueeze(1), groups=C)
        want.append(F.silu(y)[0].T)
    torch.testing.assert_close(got.double(), torch.cat(want), rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------------------------------------------
# GPU G1 gate against the rollout dump
# ---------------------------------------------------------------------------------------------------------------
_needs_dump = pytest.mark.skipif(
    not (torch.cuda.is_available() and importlib.util.find_spec("vllm") is not None and dump.available()),
    reason="requires a CUDA GPU, the K31 BI vLLM image with overlays, and K31_KDA_G1_DUMP",
)
_REPORT: dict = {}


@pytest.fixture(scope="module")
def g1(k31_bi):
    params, seqs, cases = dump.load_sequences(os.environ[dump.DUMP_ENV], os.environ[dump.CASES_ENV])
    yield params, seqs, cases
    path = os.environ.get("K31_KDA_G1_REPORT")
    if path:
        with open(path, "w") as f:
            json.dump(_REPORT, f, indent=1, sort_keys=True)


def _cfg(p: dump.LayerParams) -> KDAKernelConfig:
    return KDAKernelConfig(
        num_heads=p.num_heads,
        head_dim=p.head_dim,
        gate_lower_bound=p.lower,
        norm_eps=p.norm_eps,
        norm_method=p.norm_method,
    )


def _inputs(p: dump.LayerParams, rows: list[dict[str, torch.Tensor]], dev="cuda"):
    T = sum(r["qkv"].shape[0] for r in rows)
    cat = {k: torch.cat([r[k] for r in rows]).to(dev) for k in dump.FIELDS}
    x = (
        cat["qkv"],
        cat["g1"].reshape(T, -1),
        cat["beta"],
        cat["g2"].reshape(T, -1),
        p.conv_weight.to(dev),
        p.A_log.to(dev),
        p.dt_bias.to(dev),
        p.norm_weight.to(dev),
    )
    lens = [r["qkv"].shape[0] for r in rows]
    cu = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.int32)
    return x, cu, cat["o"].reshape(T, -1)


def _run(p, rows, *, grad: bool = False):
    x, cu, want = _inputs(p, rows)
    cfg = _cfg(p)
    if grad:
        x = tuple(t.detach().requires_grad_(i < 4) for i, t in enumerate(x))
        got = kda_core(cfg, *x, cu_seqlens=cu)
    else:
        got = kda_visible(cfg, *x, cu)
    return got.detach(), want, cu


def _mismatch(got: torch.Tensor, want: torch.Tensor) -> dict:
    bad = (got.view(torch.int16) != want.view(torch.int16)).any(dim=-1)
    diff = (got.float() - want.float()).abs()
    return {"rows": int(want.shape[0]), "bad_rows": int(bad.sum()), "max_abs": float(diff.max())}


def _layers_cases(seqs, kind):
    return sorted({(layer, tag) for (layer, tag) in seqs if tag.startswith(kind)})


@pytest.mark.gpus(1, min_architecture="blackwell")
@_needs_dump
def test_g1_prefill_equals_rollout_decode_per_sequence(g1) -> None:
    params, seqs, _ = g1
    keys = _layers_cases(seqs, "gen")
    assert keys, "dump has no gen<i> sequences"
    failures = []
    for layer, tag in keys:
        s = seqs[(layer, tag)]
        got, want, _ = _run(params[layer], [s.rows])
        m = _mismatch(got, want)
        m.update(decode_rows=int(s.graph_rows.sum()), prefill_rows=int((~s.graph_rows).sum()))
        _REPORT[f"single/{layer}/{tag}"] = m
        if m["bad_rows"]:
            failures.append((layer, tag, m))
    assert not failures, failures


@pytest.mark.gpus(1, min_architecture="blackwell")
@_needs_dump
@pytest.mark.parametrize("order", ["forward", "reverse"], ids=["packed", "packed-reversed"])
def test_g1_packed_sequences_equal_rollout(g1, order) -> None:
    params, seqs, _ = g1
    for layer in params:
        keys = [k for k in _layers_cases(seqs, "gen") if k[0] == layer]
        if order == "reverse":
            keys = keys[::-1]
        got, want, cu = _run(params[layer], [seqs[k].rows for k in keys])
        m = _mismatch(got, want)
        m["sequences"] = len(keys)
        _REPORT[f"{order}/{layer}"] = m
        assert m["bad_rows"] == 0, (layer, m)


@pytest.mark.gpus(1, min_architecture="blackwell")
@_needs_dump
def test_g1_prefill_equals_engine_scoring_prefill(g1) -> None:
    params, seqs, _ = g1
    for layer, tag in _layers_cases(seqs, "score"):
        got, want, _ = _run(params[layer], [seqs[(layer, tag)].rows])
        m = _mismatch(got, want)
        _REPORT[f"score/{layer}/{tag}"] = m
        assert m["bad_rows"] == 0, (layer, tag, m)


@pytest.mark.gpus(1, min_architecture="blackwell")
@_needs_dump
def test_rollout_inputs_agree_between_decode_and_scoring(g1) -> None:
    """Upstream sanity (not a KDA property): the scoring prefill fed the KDA layer the same inputs as the
    generation, so the decode-vs-prefill comparison above is apples to apples."""
    _, seqs, _ = g1
    for layer, tag in _layers_cases(seqs, "gen"):
        score = seqs.get((layer, "score" + tag[3:]))
        if score is None:
            continue
        gen = seqs[(layer, tag)]
        L = gen.rows["qkv"].shape[0]
        same = {k: bool(torch.equal(gen.rows[k], score.rows[k][:L])) for k in ("qkv", "g1", "beta", "g2")}
        same["o"] = bool(torch.equal(gen.rows["o"], score.rows["o"][:L]))
        _REPORT[f"inputs/{layer}/{tag}"] = same
        assert all(same.values()), (layer, tag, same)


@pytest.mark.gpus(1, min_architecture="blackwell")
@_needs_dump
def test_autograd_owner_returns_the_visible_value(g1) -> None:
    params, seqs, _ = g1
    layer = sorted(params)[0]
    keys = [k for k in _layers_cases(seqs, "gen") if k[0] == layer]
    got, want, _ = _run(params[layer], [seqs[k].rows for k in keys], grad=True)
    m = _mismatch(got, want)
    _REPORT[f"autograd/{layer}"] = m
    assert m["bad_rows"] == 0, m
