"""Bytewise invariance of the public APIs at the K3.1 head count of 96 per rank.

K3.1 runs complete KDA heads on every rank (rollout TP1, trainer TP/CP 1), so a
layer sees H=96. At this head count one 128-token coefficient segment needs
18.4 MiB, so the default 256 MiB prefill workspace holds only a few segments and
a prefill is split into many serial windows. Every invariant must hold for each
window count, so the workspace limit is varied between 24 MiB (one segment) and
the default.

Cases that other modules already run at smaller head counts are called with
h=96; cases that hard-code a head count or a workspace too small for H=96 are
restated here (prefill windows, pooled prefill, long sequences, Graph replay with
dynamic prefill metadata, acceptance commit at several batch sizes). Set
KDA_LONG_CASES=1 for the 20K- and 196K-token shapes.
"""

import inspect
import os

import pytest
import torch
from vllm.models.kimi_k3.nvidia.kda_fp32_state import prepare_decode, prepare_prefill

from . import test_api, test_decode, test_decode_configurable, test_prefill
from . import test_state_pool
from .helpers import bits, case, decode, prefill, select
from .test_decode_configurable import fixture as decode_fixture
from .test_decode_configurable import q1_reference
from .test_state_pool import dense

H = 96
MIB = 1024**2
WORKSPACES = (24 * MIB, 40 * MIB, 64 * MIB)
LONG = pytest.mark.skipif(
    os.environ.get("KDA_LONG_CASES") != "1", reason="set KDA_LONG_CASES=1"
)
ENVELOPE = [(1, 1024), (2, 512), (4, 256), (8, 128), (16, 64), (32, 32)]
SHAPES = [
    *ENVELOPE,
    (1, 5120),
    pytest.param(4, 5120, marks=LONG),
    pytest.param(1, 196608, marks=LONG),
]
MIXED = [0, 1, 31, 32, 127, 128, 129, 255, 256, 257, 511, 513, 0, 1023]

EXISTING = [
    ("arbitrary_splits", test_prefill.test_arbitrary_splits, {"h": H}),
    ("varlen_batch_reorder", test_prefill.test_varlen_batch_reorder, {"h": H}),
    (
        "batch_vs_individual_12",
        test_prefill.test_prefill_batch_vs_individual_requests,
        {"b": 12, "h": H},
    ),
    (
        "batch_vs_individual_4",
        test_prefill.test_prefill_batch_vs_individual_requests,
        {"b": 4, "h": H},
    ),
    (
        "q8_checkpoints_vs_q1_and_prefill",
        test_api.test_q8_checkpoints_match_q1_and_prefill,
        {"h": H},
    ),
    (
        "q8_checkpoints_acceptance_9",
        test_decode.test_q8_all_checkpoints_and_acceptance,
        {"b": 9, "h": H},
    ),
    (
        "q8_checkpoints_acceptance_32",
        test_decode.test_q8_all_checkpoints_and_acceptance,
        {"b": 32, "h": H},
    ),
    (
        "decode_batch_reorder_q1",
        test_decode.test_decode_batch_reorder,
        {"q": 1, "h": H},
    ),
    (
        "decode_batch_reorder_q8",
        test_decode.test_decode_batch_reorder,
        {"q": 8, "h": H},
    ),
    *[
        (
            f"q{q}_outputs_checkpoints_{gate}",
            test_decode_configurable.test_q_all_outputs_checkpoints,
            {"q": q, "b": 17, "h": H, "gate": gate},
        )
        for q in (4, 6, 8)
        for gate in ("normal", "saturated")
    ],
    (
        "mixed_q_chain_matches_prefill",
        test_decode_configurable.test_mixed_q_chain_matches_prefill,
        {"h": H},
    ),
    ("pool_decode_graph_q1", test_state_pool.test_pool_decode_graph, {"q": 1, "h": H}),
    ("pool_decode_graph_q8", test_state_pool.test_pool_decode_graph, {"q": 8, "h": H}),
]


@pytest.mark.parametrize(
    "function,kwargs",
    [pytest.param(function, kwargs, id=name) for name, function, kwargs in EXISTING],
)
def test_existing_invariants_at_deployed_heads(function, kwargs):
    if "dtype" in inspect.signature(function).parameters:
        kwargs = dict(kwargs, dtype=torch.float32)
    function(**kwargs)


@pytest.mark.parametrize("requests,length", SHAPES)
def test_prefill_batch_order_and_workspace_windows(requests, length):
    lengths = [length] * requests
    spans = [(0, length)] * requests
    x = case(lengths, torch.float32, H)
    expected, terminal, default = prefill(x)
    for limit in WORKSPACES:
        out, state, plan = prefill(x, workspace_limit_bytes=limit)
        assert plan.workspace_bytes <= limit
        bits(out, expected)
        bits(state, terminal)
    smallest = prefill(x, workspace_limit_bytes=WORKSPACES[0])[2]
    assert smallest.workspace_bytes < default.workspace_bytes
    if requests == 1:
        return
    outputs, states = [], []
    for i in range(requests):
        one, _ = select(x, lengths, spans, order=[i])
        out, state, _ = prefill(one)
        outputs.append(out)
        states.append(state)
    bits(torch.cat(outputs, 1), expected)
    bits(torch.cat(states, 0), terminal)
    order = list(reversed(range(requests)))
    reordered, _ = select(x, lengths, spans, order=order)
    out, state, _ = prefill(reordered)
    bits(
        out,
        torch.cat([expected[:, i * length : (i + 1) * length] for i in order], 1),
    )
    bits(state, terminal[order])


def test_prefill_mixed_lengths_padding_and_inplace_state():
    x = case(MIXED, torch.float32, H, padding=13)
    expected, terminal, _ = prefill(x)
    assert (expected[:, sum(MIXED) :] == 31).all()
    out, state, plan = prefill(x, workspace_limit_bytes=WORKSPACES[0])
    assert plan.workspace_bytes <= WORKSPACES[0]
    bits(out, expected)
    bits(state, terminal)
    y = dict(x, state_in=x["state_in"].clone())
    inplace = torch.full_like(expected, 31)
    prepare_prefill(
        state_dtype=torch.float32, **y, out=inplace, state_out=y["state_in"]
    ).run()
    bits(inplace, expected)
    bits(y["state_in"], terminal)


def test_prefill_matches_q1_decode_chain():
    requests, length = 3, 129
    x = decode_fixture(requests, length, H)
    reference, checkpoints = q1_reference(x, requests, length)
    for limit in (None, WORKSPACES[0]):
        kwargs = {} if limit is None else {"workspace_limit_bytes": limit}
        out, state, _ = prefill(x, **kwargs)
        bits(out, reference)
        bits(state, checkpoints[:, -1])


def test_long_sequence_windows_and_decode_suffix():
    length = 5120
    lengths = [length, 0, 129]
    x = case(lengths, torch.float32, H)
    expected, terminal, _ = prefill(x)
    for limit in WORKSPACES:
        bounded, state, plan = prefill(x, workspace_limit_bytes=limit)
        assert plan.workspace_bytes <= limit
        bits(bounded, expected)
        bits(state, terminal)
    prefix, _ = select(x, lengths, [(0, length - 9), (0, 0), (0, 129)])
    first, state, _ = prefill(prefix)
    bits(first[:, : length - 9], expected[:, : length - 9])
    suffix, _ = select(
        x,
        lengths,
        [(length - 9, length - 1), (0, 0), (0, 0)],
        order=[0],
        state=state[:1].clone(),
    )
    out, state, _ = decode(suffix, 8)
    bits(out, expected[:, length - 9 : length - 1])
    last, _ = select(
        x,
        lengths,
        [(length - 1, length), (0, 0), (0, 0)],
        order=[0],
        state=state.contiguous(),
    )
    out, state, _ = decode(last, 1)
    bits(out, expected[:, length - 1 : length])
    bits(state, terminal[:1])


@pytest.mark.parametrize("inplace", [False, True])
def test_pool_prefill_with_bounded_workspace(inplace):
    lengths = [0, 1, 8, 129, 513]
    x = case(lengths, torch.float32, H)
    expected, final, _ = prefill(x)
    pool = torch.full((13, H, 128, 128), 7, device="cuda", dtype=torch.float32)
    slots = torch.tensor([11, 0, 7, 4, 9], device="cuda", dtype=torch.int32)
    pool[slots.long()] = x["state_in"]
    initial = pool.clone()
    destination = pool if inplace else pool.clone()
    out = torch.empty_like(x["q"])
    prepare_prefill(
        **dict(x, state_in=pool),
        out=out,
        state_out=destination,
        state_dtype=torch.float32,
        state_indices=slots,
        workspace_limit_bytes=WORKSPACES[0],
    ).run()
    bits(out, expected)
    initial[slots.long()] = final
    bits(destination, initial)


@pytest.mark.parametrize("q", [1, 4, 6, 8])
@pytest.mark.parametrize("b", [1, 8, 32])
def test_decode_batch_slots_and_single_requests(b, q):
    x = decode_fixture(b, q, H)
    out, state, checkpoints = decode(x, q)
    slots = torch.arange(b - 1, -1, -1, device="cuda", dtype=torch.int32)
    pool = torch.full((b + 2, H, 128, 128), 0.5, device="cuda", dtype=torch.float32)
    pool[slots.long()] = x["state_in"]
    untouched = pool.clone()
    pooled = torch.empty(b, q, H, 128, device="cuda", dtype=torch.bfloat16)
    if q == 1:
        destination = pool.clone()
        prepare_decode(
            **dense(x, b, q),
            state_in=pool,
            out=pooled,
            state_out=destination,
            state_indices=slots,
            state_dtype=torch.float32,
        ).run()
        untouched[slots.long()] = state
        bits(destination, untouched)
    else:
        pooled_checkpoints = torch.empty(
            b, q, H, 128, 128, device="cuda", dtype=torch.float32
        )
        prepare_decode(
            **dense(x, b, q),
            state_in=pool,
            out=pooled,
            checkpoint_states=pooled_checkpoints,
            state_indices=slots,
            state_dtype=torch.float32,
        ).run()
        bits(pooled_checkpoints, checkpoints)
        bits(pool, untouched)
    bits(pooled.view_as(out), out)
    for i in range(b):
        one, _ = select(x, [q] * b, [(0, q)] * b, order=[i])
        single_out, single_state, single_checkpoints = decode(one, q)
        bits(single_out, out[:, i * q : (i + 1) * q])
        bits(single_state, state[i : i + 1])
        if single_checkpoints is not None:
            bits(single_checkpoints, checkpoints[i : i + 1])


@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("q", [4, 6, 8])
def test_acceptance_commit_and_graph_replay(q, inplace):
    b = q + 1
    x = decode_fixture(b, q, H)
    pool = torch.cat([x["state_in"], x["state_in"].flip(0)], 0)
    initial = pool.clone()
    slots = torch.arange(b, device="cuda", dtype=torch.int32)
    destination = pool if inplace else pool.clone()
    counts = torch.arange(b, device="cuda", dtype=torch.int32)
    cp = torch.empty(b, q, H, 128, 128, device="cuda")
    out = torch.empty(b, q, H, 128, device="cuda", dtype=torch.bfloat16)
    plan = prepare_decode(
        **dense(x, b, q),
        state_in=pool,
        out=out,
        state_out=destination,
        checkpoint_states=cp,
        state_indices=slots,
        num_accepted_tokens=counts,
        state_dtype=torch.float32,
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
        slots.copy_(
            torch.arange(b, device="cuda", dtype=torch.int32).flip(0)
            + iteration % 2 * b
        )
        ids = slots.long()
        counts.copy_(
            (torch.arange(b, device="cuda", dtype=torch.int32) + iteration) % (q + 1)
        )
        reference, states = q1_reference(
            dict(x, state_in=initial[ids].contiguous()), b, q
        )
        graph.replay()
        bits(out.view_as(reference), reference)
        bits(cp, states)
        chosen = initial[ids].clone()
        for i in range(b):
            accepted = (i + iteration) % (q + 1)
            if accepted:
                chosen[i].copy_(states[i, accepted - 1])
        wanted = initial.clone()
        wanted[ids] = chosen
        bits(destination, wanted)
        next_q = 6 if q == 4 else 4
        following = decode_fixture(b, next_q, H)
        expected, final, expected_checkpoints = decode(
            dict(following, state_in=chosen), next_q
        )
        next_out = torch.empty(b, next_q, H, 128, device="cuda", dtype=torch.bfloat16)
        next_cp = torch.empty(b, next_q, H, 128, 128, device="cuda")
        prepare_decode(
            **dense(following, b, next_q),
            state_in=destination,
            out=next_out,
            checkpoint_states=next_cp,
            state_indices=slots,
            state_dtype=torch.float32,
        ).run()
        bits(next_out.view_as(expected), expected)
        bits(next_cp, expected_checkpoints)
        bits(next_cp[:, -1], final)


def test_prefill_graph_dynamic_metadata_with_bounded_workspace():
    lengths = [0, 17, 129, 258]
    x = case(lengths, torch.float32, H)
    _, _, plan = prefill(x, workspace_limit_bytes=WORKSPACES[0])
    assert plan.workspace_bytes <= WORKSPACES[0]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        plan.run()
    for replayed in ([101, 101, 101, 101], [404, 0, 0, 0], [0, 0, 0, 0]):
        y = case(replayed, torch.float32, H, padding=404 - sum(replayed))
        for key in x:
            x[key].copy_(y[key])
        graph.replay()
        expected, state, _ = prefill(y)
        bits(plan.out[:, : sum(replayed)], expected[:, : sum(replayed)])
        bits(plan.state_out, state)


@pytest.mark.parametrize("b", [1, 65])
def test_commit_runtime_batch_sizes(b):
    x = case([8] * b, torch.float32, H)
    pool = torch.cat([x["state_in"], x["state_in"]], 0)
    indices = torch.arange(b, device="cuda", dtype=torch.int32) + b
    counts = torch.arange(b, device="cuda", dtype=torch.int32) % 9
    cp = torch.empty(b, 8, H, 128, 128, device="cuda", dtype=torch.float32)
    out = torch.empty(b, 8, H, 128, device="cuda", dtype=torch.bfloat16)
    destination = pool.clone()
    plan = prepare_decode(
        **dense(x, b, 8),
        state_in=pool,
        out=out,
        state_out=destination,
        checkpoint_states=cp,
        state_indices=indices,
        num_accepted_tokens=counts,
        state_dtype=torch.float32,
    )
    plan.run()
    plan.commit()
    expected = x["state_in"].clone()
    for i in range(b):
        accepted = i % 9
        if accepted:
            expected[i].copy_(cp[i, accepted - 1])
    bits(destination[indices.long()], expected)
    bits(destination[:b], pool[:b])
    counts.zero_()
    plan.run()
    plan.commit()
    bits(destination, pool)
