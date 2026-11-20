"""Complete-head sharding at the K3.1 head count: 96 heads on one rank against the same heads split into shards.

K3.1 runs 96 KDA heads per rank with TP1, and a TP8 run holds 12 per rank. The kernels treat heads independently, so the
outputs, states and checkpoints of any complete-head shard must equal the matching heads of the full run, bytewise. The shards
run one after another on one GPU and, when two GPUs are present, alternately on both. The PD handoff test prefills on one GPU
and decodes on the other.
"""

import pytest
import torch

from .helpers import NAMES, bits, case, decode, prefill, select

H = 96


def run(mode, x):
    if mode == "prefill":
        out, state, _ = prefill(x)
        return out, state, None
    return decode(x, int(mode[1:]))


@pytest.mark.parametrize("mode", ["prefill", "q1", "q4", "q8"])
@pytest.mark.parametrize("shards", [2, 4, 8, 16])
@pytest.mark.parametrize("devices", [1, 2])
def test_head_shards_equal_the_full_run(devices, shards, mode):
    if torch.cuda.device_count() < devices:
        pytest.skip("two GPUs required")
    if devices == 2:
        assert torch.cuda.get_device_name(0) == torch.cuda.get_device_name(1)
    lengths = [0, 17, 129, 513] if mode == "prefill" else [int(mode[1:])] * 8
    x = case(lengths, torch.float32, H)
    expected, state, checkpoints = run(mode, x)
    width = H // shards
    for shard in range(shards):
        start, end = shard * width, (shard + 1) * width
        device = torch.device("cuda", shard % devices)
        with torch.cuda.device(device):
            y = {
                k: x[k][:, :, start:end]
                .to(device)
                .clone(memory_format=torch.contiguous_format)
                for k in NAMES
            }
            y.update(
                A_log=x["A_log"][start:end].to(device).clone(),
                dt_bias=x["dt_bias"][start:end].to(device).clone(),
                state_in=x["state_in"][:, start:end]
                .to(device)
                .clone(memory_format=torch.contiguous_format),
                cu_seqlens=x["cu_seqlens"].to(device),
            )
            out, final, cp = run(mode, y)
            bits(out.to("cuda:0"), expected[:, :, start:end])
            bits(final.to("cuda:0"), state[:, start:end])
            if cp is not None:
                bits(cp.to("cuda:0"), checkpoints[:, :, start:end])


def test_pd_handoff_at_deployed_heads():
    if torch.cuda.device_count() < 2:
        pytest.skip("two GPUs required")
    first_length, second_length = 1025, 513
    lengths = [first_length, second_length]
    x = case(lengths, torch.float32, H)
    expected, terminal, _ = prefill(x)
    head, _ = select(x, lengths, [(0, first_length - 8), (0, second_length - 8)])
    prefix, state, _ = prefill(head)
    tail, _ = select(
        x,
        lengths,
        [(first_length - 8, first_length), (second_length - 8, second_length)],
        order=[1, 0],
        state=state[[1, 0]].contiguous(),
    )
    tail = {k: v.to("cuda:1") for k, v in tail.items()}
    with torch.cuda.device(1):
        out, state, _ = decode(tail, 8)
    out, state = out.to("cuda:0"), state.to("cuda:0")
    pieces = [
        prefix[:, : first_length - 8],
        out[:, 8:],
        prefix[:, first_length - 8 :],
        out[:, :8],
    ]
    bits(torch.cat(pieces, 1), expected)
    bits(state[[1, 0]], terminal)
