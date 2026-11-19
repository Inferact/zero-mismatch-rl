"""Public API graph cache coverage for the FP32-state contract."""

import importlib
import pytest
import torch
from vllm.models.kimi_k3.nvidia.kda_fp32_state import prepare_decode
from .helpers import bits, case, prefill, decode

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.parametrize("dtype", [torch.float32]),
]


def test_dynamic_graph_and_workspace(dtype):
    lengths = [0, 17, 129, 258]
    x = case(lengths, dtype, h=12)
    _, _, plan = prefill(x, workspace_limit_bytes=8 * 1024 * 1024)
    assert plan.workspace_bytes <= 8 * 1024 * 1024
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        plan.run()
    for ls in [[101, 101, 101, 101], [404, 0, 0, 0], [0, 0, 0, 0]]:
        y = case(ls, dtype, h=12, padding=404 - sum(ls))
        for key in x:
            x[key].copy_(y[key])
        graph.replay()
        expected, state, _ = prefill(y)
        bits(plan.out[:, : sum(ls)], expected[:, : sum(ls)])
        bits(plan.state_out, state)


@pytest.mark.parametrize(
    "scale,eps,lower_bound", [(0.125, 0.01, -1.0), (0.0, 1e-06, 0.0)]
)
def test_scalars_q8_graph_and_cache(dtype, scale, eps, lower_bound):
    from vllm.models.kimi_k3.nvidia.kda_fp32_state.kernels.decode_multi import (
        compiled_multi_kernel as compiled_q8_kernel,
    )

    kwargs = dict(scale=scale, eps=eps, lower_bound=lower_bound)
    for b, h in [(3, 3), (5, 7)]:
        x = case([8] * b, dtype, h)
        dense = {
            k: x[k].view(b, 8, *x[k].shape[2:]) for k in ("q", "k", "v", "g", "beta")
        }
        dense.update({k: x[k] for k in ("A_log", "dt_bias", "state_in")})
        checkpoints = torch.empty(b, 8, h, 128, 128, device="cuda", dtype=dtype)
        out = torch.empty_like(dense["q"])
        plan = prepare_decode(
            state_dtype=dtype, **dense, out=out, checkpoint_states=checkpoints, **kwargs
        )
        plan.run()
        before = compiled_q8_kernel.cache_info().misses
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            plan.run()
        x["g"].fill_(-20)
        graph.replay()
        expected, state, _ = prefill(x, **kwargs)
        bits(out.view_as(expected), expected)
        bits(checkpoints[:, -1], state)
        order = list(reversed(range(b)))
        y = {
            k: dense[k][order].contiguous()
            for k in ("q", "k", "v", "g", "beta", "state_in")
        }
        y.update(A_log=x["A_log"], dt_bias=x["dt_bias"])
        oo, cc = (torch.empty_like(out), torch.empty_like(checkpoints))
        prepare_decode(
            state_dtype=dtype, **y, out=oo, checkpoint_states=cc, **kwargs
        ).run()
        bits(oo, out[order])
        bits(cc, checkpoints[order])
        assert compiled_q8_kernel.cache_info().misses == before


def test_prefill_runtime_cache(dtype):
    prefix = "vllm.models.kimi_k3.nvidia.kda_fp32_state.kernels.prefill"
    kernels = [
        getattr(importlib.import_module(prefix + "." + module), name)
        for module, name in [
            ("scan", "compiled_scan_kernel"),
            ("coefficients", "compiled_coeff_kernel"),
            ("schedule", "compiled_schedule"),
        ]
    ]
    expected = None
    for lengths, h in [([257], 3), ([1, 8, 129], 12), ([0, 17], 16), ([128] * 4, 32)]:
        x = case(lengths, dtype, h)
        prefill(x)
        misses = tuple((k.cache_info().misses for k in kernels))
        if expected is not None:
            assert misses == expected
        expected = misses


@pytest.mark.parametrize("q", [1, 8])
def test_decode_cache_reuse(dtype, q):
    module = "decode_q1" if q == 1 else "decode_multi"
    name = "compiled_decode_kernel" if q == 1 else "compiled_multi_kernel"
    kernel = getattr(
        importlib.import_module(
            "vllm.models.kimi_k3.nvidia.kda_fp32_state.kernels." + module
        ),
        name,
    )
    for batches in ([1, 2, 3], [64, 65, 66]):
        expected = None
        for b in batches:
            decode(case([q] * b, dtype, h=12), q)
            misses = kernel.cache_info().misses
            if expected is not None:
                assert misses == expected
            expected = misses
