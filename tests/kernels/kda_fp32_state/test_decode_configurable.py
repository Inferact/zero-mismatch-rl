"""Configurable-Q FP32 decode against independent public execution paths."""

import pytest
import torch
from vllm.models.kimi_k3.nvidia.kda_fp32_state import prepare_decode
from .helpers import NAMES, bits, case, decode, prefill, select
from .test_state_pool import dense

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
QS = [2, 3, 4, 6, 8, 9, 16]


def fixture(b, q, h, gate="normal"):
    x = case([q] * b, torch.float32, h)
    x["state_in"].add_(0.000123)
    if gate != "normal":
        x["A_log"].zero_()
        x["dt_bias"].zero_()
        x["g"].fill_(-1000)
        if gate == "saturated":
            x["g"][..., ::2] = 1000
            x["beta"][..., ::2] = 1000
            x["beta"][..., 1::2] = -1000
    return x


def q1_reference(x, b, q):
    state = x["state_in"]
    outputs, states = ([], [])
    for t in range(q):
        part, _ = select(x, [q] * b, [(t, t + 1)] * b, state=state)
        out, state, _ = decode(part, 1)
        outputs.append(out.view(b, 1, *out.shape[2:]))
        states.append(state)
    return (torch.cat(outputs, 1).view_as(x["q"]), torch.stack(states, 1))


@pytest.mark.parametrize("q", QS)
@pytest.mark.parametrize("b,h", [(1, 1), (3, 3), (17, 12), (64, 16), (65, 32)])
@pytest.mark.parametrize("gate", ["normal", "open", "saturated"])
def test_q_all_outputs_checkpoints(q, b, h, gate):
    x = fixture(b, q, h, gate)
    expected, terminal, _ = prefill(x)
    ref, checkpoints = q1_reference(x, b, q)
    out, state, cp = decode(x, q)
    bits(out, expected)
    bits(out, ref)
    bits(state, terminal)
    bits(cp, checkpoints)


@pytest.mark.parametrize("q", QS)
@pytest.mark.parametrize("indexed", [False, True])
@pytest.mark.parametrize("inplace", [False, True])
def test_q_graph_slots_acceptance_and_continuation(q, indexed, inplace):
    b, h = (q + 1, 3)
    x = fixture(b, q, h)
    pool = (
        torch.cat([x["state_in"], x["state_in"].flip(0)], 0)
        if indexed
        else x["state_in"].clone()
    )
    initial = pool.clone()
    slots = torch.arange(b, device="cuda", dtype=torch.int32) if indexed else None
    destination = pool if inplace else pool.clone()
    counts = torch.arange(b, device="cuda", dtype=torch.int32)
    cp = torch.empty(b, q, h, 128, 128, device="cuda")
    out = torch.empty(b, q, h, 128, device="cuda", dtype=torch.bfloat16)
    plan = prepare_decode(
        **dense(x, b, q),
        state_in=pool,
        out=out,
        state_out=destination,
        checkpoint_states=cp,
        state_indices=slots,
        num_accepted_tokens=counts,
        state_dtype=torch.float32
    )
    plan.run()
    plan.commit()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        plan.run()
        plan.commit()
    for iteration in range(3):
        pool.copy_(initial)
        destination.copy_(initial)
        if indexed:
            slots.copy_(
                torch.arange(b, device="cuda", dtype=torch.int32).flip(0)
                + iteration % 2 * b
            )
            ids = slots.long()
        else:
            ids = torch.arange(b, device="cuda")
        counts.copy_(
            (torch.arange(b, device="cuda", dtype=torch.int32) + iteration) % (q + 1)
        )
        reference_input = dict(x, state_in=initial[ids].contiguous())
        yy, states = q1_reference(reference_input, b, q)
        graph.replay()
        bits(out.view_as(yy), yy)
        bits(cp, states)
        chosen = initial[ids].clone()
        for i in range(b):
            a = (i + iteration) % (q + 1)
            if a:
                chosen[i].copy_(states[i, a - 1])
        wanted = initial.clone()
        wanted[ids] = chosen
        bits(destination, wanted)
        next_q = 6 if q == 4 else 4
        nxt = fixture(b, next_q, h)
        ref, final, cps = decode(dict(nxt, state_in=chosen), next_q)
        oo = torch.empty(b, next_q, h, 128, device="cuda", dtype=torch.bfloat16)
        cc = torch.empty(b, next_q, h, 128, 128, device="cuda")
        prepare_decode(
            **dense(nxt, b, next_q),
            state_in=destination,
            out=oo,
            checkpoint_states=cc,
            state_indices=slots,
            state_dtype=torch.float32
        ).run()
        bits(oo.view_as(ref), ref)
        bits(cc, cps)
        bits(cc[:, -1], final)


@pytest.mark.parametrize("h", [3, 12, 16, 32])
def test_mixed_q_chain_matches_prefill(h):
    lengths = [53] * 3
    x = fixture(3, 53, h, "open")
    expected, terminal, _ = prefill(x)
    state, parts, start = (x["state_in"], [], 0)
    for q in [4, 6, 8, 3, 16, 16]:
        part, _ = select(x, lengths, [(start, start + q)] * 3, state=state.contiguous())
        out, state, _ = decode(part, q)
        parts.append(out.view(3, q, h, 128))
        start += q
    bits(torch.cat(parts, 1).view_as(expected), expected)
    bits(state, terminal)


@pytest.mark.parametrize("q", [4, 6])
@pytest.mark.parametrize("h", [12, 16, 32])
def test_configurable_q_two_gpu_head_shards(q, h):
    if torch.cuda.device_count() < 2:
        pytest.skip("two GPUs required")
    x = fixture(3, q, h)
    expected, final, checkpoints = decode(x, q)
    for device in range(2):
        lo, hi = (device * (h // 2), (device + 1) * (h // 2))
        with torch.cuda.device(device):
            dev = torch.device("cuda", device)
            part = {k: x[k][:, :, lo:hi].contiguous().to(dev) for k in NAMES}
            part.update(
                A_log=x["A_log"][lo:hi].contiguous().to(dev),
                dt_bias=x["dt_bias"][lo:hi].contiguous().to(dev),
                state_in=x["state_in"][:, lo:hi].contiguous().to(dev),
                cu_seqlens=x["cu_seqlens"].to(dev),
            )
            out, state, cp = decode(part, q)
            bits(out.to("cuda:0"), expected[:, :, lo:hi])
            bits(state.to("cuda:0"), final[:, lo:hi])
            bits(cp.to("cuda:0"), checkpoints[:, :, lo:hi])


def test_q_cache_and_empty():
    from vllm.models.kimi_k3.nvidia.kda_fp32_state.kernels.decode_multi import (
        compiled_multi_kernel,
    )
    from vllm.models.kimi_k3.nvidia.kda_fp32_state._checkpoint import compiled_commit

    for q in [4, 6, 5, 7, 17, 31, 64]:
        before = compiled_multi_kernel.cache_info().misses
        x = fixture(1, q, 3)
        decode(x, q)
        first = compiled_multi_kernel.cache_info().misses
        assert before <= first <= before + 1
        for b, h in [(3, 7), (65, 12)]:
            decode(fixture(b, q, h), q)
            assert compiled_multi_kernel.cache_info().misses == first
        kernel = compiled_commit(
            0, torch.cuda.get_device_capability(), True, False, False
        )
        b, h = (2, 3)
        x = fixture(b, q, h)
        cp = torch.empty(b, q, h, 128, 128, device="cuda")
        p = prepare_decode(
            **dense(x, b, q),
            state_in=x["state_in"],
            out=torch.empty_like(dense(x, b, q)["q"]),
            state_out=torch.empty_like(x["state_in"]),
            checkpoint_states=cp,
            num_accepted_tokens=torch.full((b,), q, device="cuda", dtype=torch.int32),
            state_dtype=torch.float32
        )
        p.run()
        p.commit()
        assert p._commit_call[0] is kernel
        bits(p.accepted_state, cp[:, -1])
        empty = fixture(0, q, 3)
        cp0 = torch.empty(0, q, 3, 128, 128, device="cuda")
        plan = prepare_decode(
            **dense(empty, 0, q),
            state_in=empty["state_in"],
            out=torch.empty_like(dense(empty, 0, q)["q"]),
            state_out=empty["state_in"],
            checkpoint_states=cp0,
            num_accepted_tokens=torch.empty(0, device="cuda", dtype=torch.int32),
            state_dtype=torch.float32
        )
        plan.run()
        plan.commit()


def test_q_validation():
    for q in [0, 4]:
        x = fixture(2, q, 3)
        z = dense(x, 2, q)
        with pytest.raises(ValueError, match="shape|checkpoints"):
            prepare_decode(
                **z,
                state_in=x["state_in"],
                out=torch.empty_like(z["q"]),
                checkpoint_states=torch.empty(2, 8, 3, 128, 128, device="cuda"),
                state_dtype=torch.float32
            )
    x = fixture(2, 4, 3)
    with pytest.raises(ValueError, match="state_dtype must be torch.float32"):
        prepare_decode(
            **dense(x, 2, 4),
            state_in=x["state_in"].bfloat16(),
            out=torch.empty_like(dense(x, 2, 4)["q"]),
            state_dtype=torch.bfloat16
        )


@pytest.mark.parametrize("q", QS)
def test_q_batch_reorder(q):
    b, h = (17, 3)
    x = fixture(b, q, h)
    out, state, cp = decode(x, q)
    for order in ([16], [3, 7, 0, 16], list(reversed(range(b)))):
        part, _ = select(x, [q] * b, [(0, q)] * b, order=order)
        got, terminal, checkpoints = decode(part, q)
        bits(got.view(len(order), q, h, 128), out.view(b, q, h, 128)[order])
        bits(terminal, state[order])
        bits(checkpoints, cp[order])
