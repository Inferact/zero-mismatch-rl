"""Public API decode coverage for the FP32-state contract."""

import pytest
import torch
from vllm.models.kimi_k3.nvidia.kda_fp32_state import prepare_decode
from .helpers import bits, case, select, prefill, decode, segment

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.parametrize("dtype", [torch.float32]),
]


@pytest.mark.parametrize("b,h", [(1, 1), (3, 3), (5, 6), (64, 12), (64, 16), (64, 32)])
def test_q8_all_checkpoints_and_acceptance(dtype, b, h):
    x = case([8] * b, dtype, h)
    dense = {k: x[k].view(b, 8, *x[k].shape[2:]) for k in ("q", "k", "v", "g", "beta")}
    dense.update({k: x[k] for k in ("A_log", "dt_bias", "state_in")})
    out = torch.empty_like(dense["q"])
    checkpoints = torch.empty(b, 8, h, 128, 128, device="cuda", dtype=dtype)
    plan = prepare_decode(
        state_dtype=dtype, **dense, out=out, checkpoint_states=checkpoints
    )
    got, final = plan.run()
    expected, terminal, _ = prefill(x)
    bits(got.view_as(expected), expected)
    bits(final, terminal)
    state = x["state_in"]
    for t in range(8):
        token = {
            k: dense[k][:, t : t + 1].clone(memory_format=torch.contiguous_format)
            for k in ("q", "k", "v", "g", "beta")
        }
        token.update(A_log=x["A_log"], dt_bias=x["dt_bias"], state_in=state)
        output, next_state = (torch.empty_like(token["q"]), torch.empty_like(state))
        prepare_decode(
            state_dtype=dtype, **token, out=output, state_out=next_state
        ).run()
        bits(output[:, 0], out[:, t])
        bits(next_state, checkpoints[:, t])
        state = next_state
    accepted = [i % 9 for i in range(b)]
    indices = torch.tensor(accepted, device="cuda")
    selected = torch.where(
        (indices == 0)[:, None, None, None],
        x["state_in"],
        checkpoints[torch.arange(b, device="cuda"), (indices - 1).clamp_min(0)],
    ).contiguous()
    prefix, _ = segment(x, [8] * b, [0] * b, accepted)
    _, expected_state, _ = prefill(prefix)
    bits(selected, expected_state)
    suffix, _ = segment(x, [8] * b, accepted, [8] * b, selected)
    _, continued, _ = prefill(suffix)
    bits(continued, terminal)


@pytest.mark.parametrize("q", [1, 8])
@pytest.mark.parametrize("h", [12, 16, 32])
def test_decode_batch_reorder(dtype, q, h):
    x = case([q] * 64, dtype, h)
    out, state, cp = decode(x, q)
    for order in ([63], [3, 17, 0, 63], list(reversed(range(64)))):
        y, _ = select(x, [q] * 64, [(0, q)] * 64, order=order)
        oo, ss, cc = decode(y, q)
        bits(oo.view(len(order), q, h, 128), out.view(64, q, h, 128)[order])
        bits(ss, state[order])
        if cp is not None:
            bits(cc, cp[order])


def test_q8_workspace_windows(dtype):
    x = case([8] * 3, dtype, h=3)
    eo, es, ec = decode(x, 8)
    row_bytes = 3 * 3 * 392 * 4
    for tokens in [1, 3, 8]:
        out, state, cp = decode(
            x, 8, workspace_limit_bytes=tokens * row_bytes + 3 * 8 * 4
        )
        bits(out, eo)
        bits(state, es)
        bits(cp, ec)
