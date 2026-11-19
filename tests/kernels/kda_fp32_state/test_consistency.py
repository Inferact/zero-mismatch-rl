"""Public API consistency coverage for the FP32-state contract."""

import pytest
import torch
from .helpers import bits, case, select, prefill, decode

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.parametrize("dtype", [torch.float32]),
]


@pytest.mark.parametrize("kind", ["normal", "open", "saturated", "zero_qk"])
@pytest.mark.parametrize("lower_bound", [-5.0, -2.5, 0.0])
def test_all_pd_cuts_and_gates(dtype, kind, lower_bound):
    x = case([25], dtype, h=3)
    if kind == "zero_qk":
        x["q"].zero_()
        x["k"].zero_()
    if kind in ("open", "saturated"):
        x["g"].fill_(-1000)
        x["A_log"].zero_()
        x["dt_bias"].zero_()
        if kind == "saturated":
            x["g"][..., ::2] = 1000
            x["beta"][:, ::2] = 1000
            x["beta"][:, 1::2] = -1000
    expected, terminal, _ = prefill(x, lower_bound=lower_bound)
    for cut in range(18):
        prefix, _ = select(x, [25], [(0, cut)])
        first, state, _ = prefill(prefix, lower_bound=lower_bound)
        middle, _ = select(x, [25], [(cut, cut + 8)], state=state)
        second, state, _ = decode(middle, 8, lower_bound=lower_bound)
        pieces = [first, second]
        for token in range(cut + 8, 25):
            one, _ = select(x, [25], [(token, token + 1)], state=state)
            part, state, _ = decode(one, 1, lower_bound=lower_bound)
            pieces.append(part)
        bits(torch.cat(pieces, 1), expected)
        bits(state, terminal)


@pytest.mark.parametrize("length", [4096, 32768])
@pytest.mark.parametrize("h", [12, 16, 32])
def test_long_windows_and_decode_suffix(dtype, length, h):
    x = case([length, 0, 129], dtype, h)
    expected, terminal, _ = prefill(x)
    bounded, bounded_state, plan = prefill(x, workspace_limit_bytes=8 * 1024**2)
    assert plan.workspace_bytes <= 8 * 1024**2
    bits(bounded, expected)
    bits(bounded_state, terminal)
    prefix, _ = select(x, [length, 0, 129], [(0, length - 9), (0, 0), (0, 129)])
    first, state, _ = prefill(prefix)
    bits(first[:, : length - 9], expected[:, : length - 9])
    suffix, _ = select(
        x,
        [length, 0, 129],
        [(length - 9, length - 1), (0, 0), (0, 0)],
        order=[0],
        state=state[:1].clone(),
    )
    out, state, _ = decode(suffix, 8)
    bits(out, expected[:, length - 9 : length - 1])
    last, _ = select(
        x,
        [length, 0, 129],
        [(length - 1, length), (0, 0), (0, 0)],
        order=[0],
        state=state.contiguous(),
    )
    out, state, _ = decode(last, 1)
    bits(out, expected[:, length - 1 : length])
    bits(state, terminal[:1])
