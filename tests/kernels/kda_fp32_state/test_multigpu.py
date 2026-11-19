"""Public API multigpu coverage for the FP32-state contract."""

import pytest
import torch
from .helpers import bits, case, select, prefill, decode, NAMES

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.parametrize("dtype", [torch.float32]),
]


@pytest.mark.parametrize("h", [12, 16, 32])
@pytest.mark.parametrize("mode", ["prefill", "q1", "q8"])
def test_two_gpu_head_shards(dtype, h, mode):
    if torch.cuda.device_count() < 2:
        pytest.skip("two GPUs required")
    assert torch.cuda.get_device_name(0) == torch.cuda.get_device_name(1)
    lengths = [0, 17, 129] if mode == "prefill" else [int(mode[1:])] * 3
    x = case(lengths, dtype, h)

    def run(y):
        if mode == "prefill":
            o, s, _ = prefill(y)
            return (o, s, None)
        return decode(y, int(mode[1:]))

    expected, state, checkpoints = run(x)
    for shard in range(4):
        start, end = (shard * h // 4, (shard + 1) * h // 4)
        device = torch.device("cuda", shard % 2)
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
            out, final, cp = run(y)
            bits(out.to("cuda:0"), expected[:, :, start:end])
            bits(final.to("cuda:0"), state[:, start:end])
            if cp is not None:
                bits(cp.to("cuda:0"), checkpoints[:, :, start:end])


def test_two_gpu_pd_handoff(dtype):
    if torch.cuda.device_count() < 2:
        pytest.skip("two GPUs required")
    x = case([25, 17], dtype, h=12)
    expected, terminal, _ = prefill(x)
    first, _ = select(x, [25, 17], [(0, 17), (0, 9)])
    prefix, state, _ = prefill(first)
    second, _ = select(
        x, [25, 17], [(17, 25), (9, 17)], order=[1, 0], state=state[[1, 0]].contiguous()
    )
    second = {k: v.to("cuda:1") for k, v in second.items()}
    with torch.cuda.device(1):
        out, state, _ = decode(second, 8)
    out, state = (out.to("cuda:0"), state.to("cuda:0"))
    bits(
        torch.cat([prefix[:, :17], out[:, 8:], prefix[:, 17:], out[:, :8]], 1), expected
    )
    bits(state[[1, 0]], terminal)
