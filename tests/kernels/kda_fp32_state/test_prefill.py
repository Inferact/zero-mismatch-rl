"""Public API prefill coverage for the FP32-state contract."""

import itertools
import pytest
import torch
from vllm.models.kimi_k3.nvidia.kda_fp32_state import prepare_prefill
from .helpers import bits, case, select, prefill, segment

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.parametrize("dtype", [torch.float32]),
]


@pytest.mark.parametrize("h", [1, 3, 6, 12, 16, 32])
def test_arbitrary_splits(dtype, h):
    lengths = [0, 1, 7, 8, 16, 17, 127, 128, 129, 257]
    x = case(lengths, dtype, h)
    expected, terminal, _ = prefill(x)
    state = x["state_in"].clone()
    pieces = [[] for _ in lengths]
    offsets = [0] * len(lengths)
    for step in itertools.cycle([1, 8, 31, 128]):
        ends = [min(l, s + step) for l, s in zip(lengths, offsets)]
        y, ls = segment(x, lengths, offsets, ends, state)
        out, state, _ = prefill(y)
        cursor = 0
        for i, length in enumerate(ls):
            pieces[i].append(out[:, cursor : cursor + length].clone())
            cursor += length
        offsets = ends
        if offsets == lengths:
            break
    bits(torch.cat([torch.cat(p, 1) for p in pieces], 1), expected)
    bits(state, terminal)


@pytest.mark.parametrize("lengths", [[], [0], [0, 0, 0], [0, 1, 8, 129]])
def test_empty_padding_and_inplace(dtype, lengths):
    x = case(lengths, dtype, padding=13)
    out, terminal, _ = prefill(x)
    assert (out[:, sum(lengths) :] == 31).all()
    y = dict(x, state_in=x["state_in"].clone())
    oo = torch.full_like(out, 31)
    prepare_prefill(state_dtype=dtype, **y, out=oo, state_out=y["state_in"]).run()
    bits(oo, out)
    bits(y["state_in"], terminal)


@pytest.mark.parametrize("h", [1, 3, 6, 12, 16, 32])
def test_varlen_batch_reorder(dtype, h):
    lengths = [0, 1, 7, 8, 17, 127, 128, 129, 257]
    x = case(lengths, dtype, h)
    expected, state, _ = prefill(x)
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    for order in ([8, 0, 3, 1], list(reversed(range(len(lengths)))), [4]):
        y, _ = select(x, lengths, [(0, n) for n in lengths], order=order)
        out, final, _ = prefill(y)
        bits(
            out, torch.cat([expected[:, offsets[i] : offsets[i + 1]] for i in order], 1)
        )
        bits(final, state[order])


@pytest.mark.parametrize(
    "b,h", [(12, 11), (12, 12), (12, 13), (8, 16), (5, 32), (4, 48)]
)
def test_prefill_batch_vs_individual_requests(dtype, b, h):
    """Changing the batch must preserve all states across head counts."""
    lengths = [257 + 31 * (i % 3) for i in range(b)]
    lengths[-1] = 0
    x = case(lengths, dtype, h)
    expected, terminal, _ = prefill(x)
    outputs, states = ([], [])
    for i in range(b):
        one, _ = select(x, lengths, [(0, n) for n in lengths], order=[i])
        out, state, _ = prefill(one)
        outputs.append(out)
        states.append(state)
    bits(torch.cat(outputs, 1), expected)
    bits(torch.cat(states, 0), terminal)
