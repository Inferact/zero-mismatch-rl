"""K31 KDA backward gate: the straight-through VJP (CF-BI forward value, FLA chunk_kda recompute gradient)
against a sequential-recurrence reference in FP64 (truth) and FP32.

Reported per input (mixed_qkv, g1, beta, g2 and the frozen base parameters): relative L2, cosine, max abs, next
to two floors: FP32 recurrence vs FP64, and BF16 rounding of the FP64 gradient. Real inputs come from the G1
rollout dump (K31_KDA_G1_DUMP / K31_KDA_G1_CASES); without it a synthetic layer is used.
K31_KDA_GRAD_REPORT receives the JSON metrics.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
import time

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _kda_dump as dump  # noqa: E402
from _kda_reference import kda_reference  # noqa: E402

from megatron.lite.model.k31.vllm.primitive._contract import parameter_versions
from megatron.lite.model.k31.vllm.primitive.kda import (
    KDAKernelConfig,
    fla_functional,
    kda_core,
    kda_visible,
    pin_fla_autotune,
)

NAMES = ("mixed_qkv", "g1", "beta", "g2", "conv_weight", "A_log", "dt_bias", "norm_weight")
ACTIVATIONS = NAMES[:4]
_needs_gpu = pytest.mark.skipif(
    not (torch.cuda.is_available() and importlib.util.find_spec("vllm") is not None
         and importlib.util.find_spec("fla") is not None),
    reason="requires a CUDA GPU, the K31 BI vLLM image with overlays, and fla",
)
_REPORT: dict = {}


def _metrics(got: torch.Tensor, ref: torch.Tensor) -> dict:
    g, r = got.double().flatten(), ref.double().flatten()
    rn = float(r.norm())
    return {
        "rel_l2": float((g - r).norm()) / rn if rn else float((g - r).norm()),
        "cos": float(torch.dot(g, r) / (g.norm() * r.norm())) if rn and float(g.norm()) else float("nan"),
        "max_abs": float((g - r).abs().max()),
        "ref_rms": rn / math.sqrt(r.numel()),
    }


def _synthetic(T_lens=None, H=96, D=128, seed=7):
    if T_lens is None:
        env = os.environ.get("K31_KDA_GRAD_LENS", "1,63,64,65,700,1200")
        T_lens = tuple(int(v) for v in env.split(","))
    g = torch.Generator(device="cpu").manual_seed(seed)
    T = sum(T_lens)
    HD = H * D

    def rn(*s, std=1.0):
        return torch.randn(*s, generator=g) * std

    x = (
        rn(T, 3 * HD).bfloat16(),
        rn(T, HD, std=2.0).bfloat16(),
        rn(T, H).bfloat16(),
        rn(T, HD).bfloat16(),
        rn(3 * HD, 4, std=0.3),
        torch.log(torch.empty(H).uniform_(1, 16, generator=g)),
        rn(HD, std=0.5),
        (1 + rn(D, std=0.1)).bfloat16(),
    )
    cu = torch.tensor([0, *torch.tensor(T_lens).cumsum(0).tolist()], dtype=torch.int32)
    cfg = KDAKernelConfig(num_heads=H, head_dim=D, gate_lower_bound=-5.0, norm_eps=1e-5)
    return "synthetic", cfg, tuple(t.cuda() for t in x), cu


def _from_dump():
    params, seqs, _ = dump.load_sequences(os.environ[dump.DUMP_ENV], os.environ[dump.CASES_ENV])
    layer = sorted(params)[0]
    p = params[layer]
    rows = [seqs[k].rows for k in sorted(seqs) if k[0] == layer and k[1].startswith("gen")]
    T = sum(r["qkv"].shape[0] for r in rows)
    cat = {k: torch.cat([r[k] for r in rows]) for k in ("qkv", "g1", "beta", "g2")}
    x = (cat["qkv"], cat["g1"].reshape(T, -1), cat["beta"], cat["g2"].reshape(T, -1),
         p.conv_weight, p.A_log, p.dt_bias, p.norm_weight)
    lens = [r["qkv"].shape[0] for r in rows]
    cu = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.int32)
    cfg = KDAKernelConfig(num_heads=p.num_heads, head_dim=p.head_dim, gate_lower_bound=p.lower,
                          norm_eps=p.norm_eps, norm_method=p.norm_method)
    return f"dump:{layer}", cfg, tuple(t.cuda() for t in x), cu


def _reference_grads(cfg, x, cu, grad_out, dtype, group=16):
    """FP64/FP32 recurrence value and VJP for every input, head group by head group."""
    H, D = cfg.num_heads, cfg.head_dim
    leaves = tuple(t.detach().to(dtype).requires_grad_(True) for t in x)
    grads = [torch.zeros_like(t) for t in leaves]
    outs = []
    cul = cu.tolist()
    for h0 in range(0, H, group):
        hs = slice(h0, min(H, h0 + group))
        y = kda_reference(*leaves, cul, num_heads=H, head_dim=D, lower=cfg.gate_lower_bound,
                          eps=cfg.norm_eps, heads=hs)
        go = grad_out[:, hs.start * D:hs.stop * D].to(dtype)
        gs = torch.autograd.grad(y, leaves, go, allow_unused=True)
        for i, gi in enumerate(gs):
            if gi is not None:
                grads[i] += gi
        outs.append(y.detach())
    return torch.cat(outs, dim=1), grads


def _ste(cfg, x, cu, grad_out):
    leaves = tuple(t.detach().clone().requires_grad_(True) for t in x)
    out = kda_core(cfg, *leaves, cu_seqlens=cu)
    out.backward(grad_out)
    return out.detach(), [t.grad for t in leaves]


@pytest.fixture(scope="module")
def case(k31_bi):
    pin_fla_autotune()
    built = [_synthetic()]
    if dump.available():
        built.append(_from_dump())
    yield built
    path = os.environ.get("K31_KDA_GRAD_REPORT")
    if path:
        with open(path, "w") as f:
            json.dump(_REPORT, f, indent=1, sort_keys=True)


@pytest.mark.gpus(1, min_architecture="blackwell")
@pytest.mark.timeout(seconds=3600)
@_needs_gpu
def test_ste_gradient_against_sequential_reference(case) -> None:
    for name, cfg, x, cu in case:
        torch.manual_seed(11)
        grad_out = torch.randn(x[0].shape[0], cfg.projection_size, device="cuda").bfloat16()
        out, ste = _ste(cfg, x, cu, grad_out)
        y64, g64 = _reference_grads(cfg, x, cu, grad_out, torch.float64)
        y32, g32 = _reference_grads(cfg, x, cu, grad_out, torch.float32)
        with torch.no_grad():
            fla_out = fla_functional(cfg, cu)(*x)
        rep = {
            "tokens": int(cu[-1]),
            "sequences": int(cu.numel() - 1),
            "forward": {
                "cfbi_vs_fp64": _metrics(out, y64),
                "fla_vs_fp64": _metrics(fla_out, y64),
                "fp32_vs_fp64": _metrics(y32, y64),
                "bf16_round_fp64": _metrics(y64.bfloat16(), y64),
                "cfbi_vs_fla": _metrics(out, fla_out),
            },
            "grad": {},
        }
        for i, n in enumerate(NAMES):
            rep["grad"][n] = {
                "ste_vs_fp64": _metrics(ste[i], g64[i]),
                "fp32_vs_fp64": _metrics(g32[i], g64[i]),
                "bf16_round_fp64": _metrics(g64[i].to(x[i].dtype), g64[i]),
            }
            assert torch.isfinite(ste[i]).all(), (name, n)
        _REPORT[name] = rep
        for n in ACTIVATIONS:
            m = rep["grad"][n]["ste_vs_fp64"]
            assert m["cos"] > 0.999 and m["rel_l2"] < 0.05, (name, n, m)


@pytest.mark.gpus(1, min_architecture="blackwell")
@_needs_gpu
def test_ste_backward_is_deterministic(case) -> None:
    name, cfg, x, cu = case[0]
    torch.manual_seed(12)
    grad_out = torch.randn(x[0].shape[0], cfg.projection_size, device="cuda").bfloat16()
    _, a = _ste(cfg, x, cu, grad_out)
    _, b = _ste(cfg, x, cu, grad_out)
    same = {n: bool(torch.equal(ga, gb)) for n, ga, gb in zip(NAMES, a, b)}
    _REPORT.setdefault("determinism", {})[name] = same
    assert all(same.values()), same


@pytest.mark.gpus(1, min_architecture="blackwell")
@_needs_gpu
def test_backward_rejects_mutated_base_parameter(case) -> None:
    _, cfg, x, cu = case[0]
    leaves = list(t.detach().clone().requires_grad_(i < 4) for i, t in enumerate(x))
    out = kda_core(cfg, *leaves, cu_seqlens=cu)
    before = parameter_versions(leaves[4:])
    leaves[5].add_(0)
    assert parameter_versions(leaves[4:]) != before
    with pytest.raises(RuntimeError, match="master parameter changed"):
        out.sum().backward()


@pytest.mark.gpus(1, min_architecture="blackwell")
@_needs_gpu
def test_timing(case) -> None:
    name, cfg, x, cu = case[0]
    grad_out = torch.ones(x[0].shape[0], cfg.projection_size, device="cuda").bfloat16()

    def t(fn, n=5):
        fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n * 1e3

    leaves = tuple(v.detach().clone().requires_grad_(i < 4) for i, v in enumerate(x))
    fwd = t(lambda: kda_visible(cfg, *x, cu))
    fla = fla_functional(cfg, cu)
    fla_fwd = t(lambda: fla(*x))
    step = t(lambda: kda_core(cfg, *leaves, cu_seqlens=cu).backward(grad_out))
    fla_step = t(lambda: fla(*leaves).backward(grad_out))
    _REPORT.setdefault("timing_ms", {})[name] = {
        "tokens": int(cu[-1]), "cfbi_forward": fwd, "fla_forward": fla_fwd,
        "ste_forward_backward": step, "fla_forward_backward": fla_step,
    }
