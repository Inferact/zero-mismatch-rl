"""Terminal-only decode preserves the public recurrence and indexed state semantics."""

import pytest
import torch
from vllm.models.kimi_k3.nvidia.kda_fp32_state import prepare_decode
from .helpers import bits, prefill, select, decode
from .test_decode_configurable import fixture, q1_reference
from .test_state_pool import dense

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def terminal(x, b, q, state=None, slots=None, inplace=False):
    z = dense(x, b, q)
    state = x["state_in"] if state is None else state
    dest = state if inplace else state.clone()
    plan = prepare_decode(
        **z,
        state_in=state,
        state_out=dest,
        out=torch.empty_like(z["q"]),
        state_indices=slots,
    )
    return plan


@pytest.mark.parametrize("q", [2, 3, 4, 6, 8, 9, 16])
@pytest.mark.parametrize("b,h", [(1, 1), (3, 3), (17, 12), (64, 16), (65, 32)])
@pytest.mark.parametrize("gate", ["normal", "open", "saturated"])
def test_terminal_against_prefill_q1_checkpoints(q, b, h, gate):
    x = fixture(b, q, h, gate)
    yy, ss, _ = prefill(x)
    q1, cp = q1_reference(x, b, q)
    oo, tt, cc = decode(x, q)
    p = terminal(x, b, q)
    out, state = p.run()
    bits(out.view_as(yy), yy)
    bits(out.view_as(q1), q1)
    bits(out.view_as(oo), oo)
    bits(state, ss)
    bits(state, cp[:, -1])
    bits(state, tt)
    assert p.checkpoint_states is None


@pytest.mark.parametrize("q", [4, 6, 8])
@pytest.mark.parametrize("inplace", [False, True])
def test_terminal_graph_dynamic_slots_and_continuation(q, inplace):
    b, h = 7, 3
    x = fixture(b, q, h)
    initial = torch.cat([x["state_in"], x["state_in"].flip(0)], 0)
    pool = initial.clone()
    ids = torch.arange(b, device="cuda", dtype=torch.int32)
    p = terminal(x, b, q, pool, ids, inplace)
    p.run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        p.run()
    for it in range(3):
        ids.copy_(
            torch.arange(b, device="cuda", dtype=torch.int32).flip(0) + (it % 2) * b
        )
        pool.copy_(initial)
        p.state_out.copy_(initial)
        expected, ss, _ = prefill(dict(x, state_in=initial[ids.long()].contiguous()))
        graph.replay()
        bits(p.out.view_as(expected), expected)
        want = initial.clone()
        want[ids.long()] = ss
        bits(p.state_out, want)
        if not inplace:
            bits(pool, initial)
        nxt = fixture(b, 6, h)
        ref, final, _ = prefill(dict(nxt, state_in=ss))
        pp = terminal(nxt, b, 6, p.state_out, ids)
        oo, st = pp.run()
        bits(oo.view_as(ref), ref)
        bits(st[ids.long()], final)


def test_terminal_splits_reorder_cache_empty_and_errors():
    from vllm.models.kimi_k3.nvidia.kda_fp32_state.kernels.decode_multi import (
        compiled_multi_kernel,
    )

    x = fixture(5, 37, 3, "open")
    ref, ss, _ = prefill(x)
    st = x["state_in"]
    parts = []
    start = 0
    for q in [4, 6, 8, 3, 16]:
        part, _ = select(x, [37] * 5, [(start, start + q)] * 5, state=st)
        p = terminal(part, 5, q)
        oo, st = p.run()
        parts.append(oo)
        start += q
    bits(torch.cat(parts, 1).view_as(ref), ref)
    bits(st, ss)
    for order in [[4], [3, 0, 2], list(reversed(range(5)))]:
        part, _ = select(x, [37] * 5, [(0, 37)] * 5, order=order)
        p = terminal(part, len(order), 37)
        oo, st = p.run()
        bits(oo, ref.view(5, 37, 3, 128)[order])
        bits(st, ss[order])
    terminal(fixture(1, 4, 3), 1, 4).run()
    miss = compiled_multi_kernel.cache_info().misses
    for b, h in [(3, 7), (65, 12)]:
        terminal(fixture(b, 4, h), b, 4).run()
    assert compiled_multi_kernel.cache_info().misses == miss
    terminal(fixture(0, 4, 3), 0, 4).run()
    xx = fixture(2, 4, 3)
    z = dense(xx, 2, 4)
    with pytest.raises(ValueError, match="checkpoint"):
        prepare_decode(
            **z,
            state_in=xx["state_in"],
            state_out=xx["state_in"],
            out=torch.empty_like(z["q"]),
            num_accepted_tokens=torch.ones(2, device="cuda", dtype=torch.int32),
        )


@pytest.mark.parametrize("q", [4, 6, 8])
def test_terminal_two_gpu_head_shards(q):
    if torch.cuda.device_count() < 2:
        pytest.skip("two GPUs required")
    b, h = 3, 12
    x = fixture(b, q, h)
    yy, ss = terminal(x, b, q).run()
    for dev in range(2):
        lo, hi = dev * 6, (dev + 1) * 6
        with torch.cuda.device(dev):
            part = {
                k: x[k][:, :, lo:hi].contiguous().to(f"cuda:{dev}")
                for k in ["q", "k", "v", "g", "beta"]
            }
            part.update(
                A_log=x["A_log"][lo:hi].contiguous().to(f"cuda:{dev}"),
                dt_bias=x["dt_bias"][lo:hi].contiguous().to(f"cuda:{dev}"),
                state_in=x["state_in"][:, lo:hi].contiguous().to(f"cuda:{dev}"),
            )
            oo, st = terminal(part, b, q).run()
            bits(oo.to("cuda:0"), yy[:, :, lo:hi])
            bits(st.to("cuda:0"), ss[:, lo:hi])
