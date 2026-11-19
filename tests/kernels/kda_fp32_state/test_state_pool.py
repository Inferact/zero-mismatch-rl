"""Indexed public APIs, speculative commit, and dynamic Graph metadata."""

import importlib
import pytest
import torch
from vllm.models.kimi_k3.nvidia.kda_fp32_state import prepare_decode, prepare_prefill
from .helpers import bits, case, prefill, decode, NAMES

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.parametrize("dtype", [torch.float32]),
]


def dense(x, b, q):
    z = {k: x[k].view(b, q, *x[k].shape[2:]) for k in NAMES}
    z.update({k: x[k] for k in ("A_log", "dt_bias")})
    return z


@pytest.mark.parametrize("h", [3, 12, 16, 32])
@pytest.mark.parametrize("inplace", [False, True])
def test_pool_prefill(dtype, h, inplace):
    lengths = [0, 1, 8, 129, 513]
    x = case(lengths, dtype, h)
    expected, final, _ = prefill(x)
    pool = torch.full((13, h, 128, 128), 7, device="cuda", dtype=dtype)
    slots = torch.tensor([11, 0, 7, 4, 9], device="cuda", dtype=torch.int32)
    pool[slots.long()] = x["state_in"]
    initial = pool.clone()
    destination = pool if inplace else pool.clone()
    out = torch.empty_like(x["q"])
    z = dict(x, state_in=pool)
    p = prepare_prefill(
        **z,
        out=out,
        state_out=destination,
        state_dtype=dtype,
        state_indices=slots,
        workspace_limit_bytes=16 * 1024**2
    )
    p.run()
    bits(out, expected)
    initial[slots.long()] = final
    bits(destination, initial)


@pytest.mark.parametrize("q", [1, 8])
@pytest.mark.parametrize("h", [3, 12, 16, 32])
def test_pool_decode_graph(dtype, q, h):
    b = 9
    x = case([q] * b, dtype, h)
    pool = torch.full((23, h, 128, 128), 0.25, device="cuda", dtype=dtype)
    destination = pool.clone()
    slots = torch.tensor(
        [19, 0, 12, 7, 4, 22, 11, 9, 6], device="cuda", dtype=torch.int32
    )
    pool[slots.long()] = x["state_in"]
    out = torch.empty(b, q, h, 128, device="cuda", dtype=torch.bfloat16)
    cp = torch.empty(b, 8, h, 128, 128, device="cuda", dtype=dtype) if q == 8 else None
    p = prepare_decode(
        **dense(x, b, q),
        state_in=pool,
        out=out,
        state_out=destination if q == 1 else None,
        checkpoint_states=cp,
        state_indices=slots,
        state_dtype=dtype
    )
    p.run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        p.run()
    for order in [
        list(range(b)),
        list(reversed(range(b))),
        [3, 0, 8, 4, 1, 7, 2, 5, 6],
    ]:
        old_slots = slots.clone()
        slots.copy_(old_slots[order])
        y = dict(x, state_in=pool[slots.long()].contiguous())
        expected, final, checkpoints = decode(y, q)
        untouched = destination.clone()
        graph.replay()
        bits(out.view_as(expected), expected)
        if q == 1:
            untouched[slots.long()] = final
            bits(destination, untouched)
        else:
            bits(cp, checkpoints)
    prefix = "kernels"
    name = "compiled_decode_kernel" if q == 1 else "compiled_multi_kernel"
    module_name = (
        "decode_multi" if dtype == torch.float32 and q > 1 else "decode_q" + str(q)
    )
    module = importlib.import_module(
        "vllm.models.kimi_k3.nvidia.kda_fp32_state." + prefix + "." + module_name
    )
    compiler = getattr(module, name)
    misses = compiler.cache_info().misses
    bigger = torch.cat([pool, pool], dim=0)
    pp = prepare_decode(
        **dense(x, b, q),
        state_in=bigger,
        out=out,
        state_out=torch.empty_like(bigger) if q == 1 else None,
        checkpoint_states=cp,
        state_indices=slots,
        state_dtype=dtype
    )
    pp.run()
    assert compiler.cache_info().misses == misses


@pytest.mark.parametrize("indexed", [False, True])
@pytest.mark.parametrize("inplace", [False, True])
def test_acceptance_and_continuation(dtype, indexed, inplace):
    b, h = (9, 3)
    x = case([8] * b, dtype, h)
    slots = (
        torch.tensor([19, 0, 12, 7, 4, 22, 11, 9, 6], device="cuda", dtype=torch.int32)
        if indexed
        else None
    )
    pool = torch.full(
        (23 if indexed else b, h, 128, 128), 0.5, device="cuda", dtype=dtype
    )
    ids = slots.long() if indexed else torch.arange(b, device="cuda")
    pool[ids] = x["state_in"]
    initial = pool.clone()
    dst = pool if inplace else pool.clone()
    cp = torch.empty(b, 8, h, 128, 128, device="cuda", dtype=dtype)
    out = torch.empty(b, 8, h, 128, device="cuda", dtype=torch.bfloat16)
    accepted = torch.arange(b, device="cuda", dtype=torch.int32)
    p = prepare_decode(
        **dense(x, b, 8),
        state_in=pool,
        out=out,
        state_out=dst,
        checkpoint_states=cp,
        state_indices=slots,
        num_accepted_tokens=accepted,
        state_dtype=dtype
    )
    p.run()
    for t in range(1, 9):
        y = {
            k: x[k]
            .view(b, 8, *x[k].shape[2:])[:, :t]
            .contiguous()
            .view(1, b * t, *x[k].shape[2:])
            for k in NAMES
        }
        y.update(
            A_log=x["A_log"],
            dt_bias=x["dt_bias"],
            state_in=x["state_in"],
            cu_seqlens=torch.arange(b + 1, device="cuda", dtype=torch.int32) * t,
        )
        yy, ss, _ = prefill(y)
        bits(cp[:, t - 1], ss)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        p.run()
        p.commit()
    for counts in [list(range(9)), [0] * 9, [8] * 9, [4, 0, 7, 1, 8, 3, 2, 6, 5]]:
        pool.copy_(initial)
        dst.copy_(initial)
        accepted.copy_(torch.tensor(counts, device="cuda", dtype=torch.int32))
        graph.replay()
        chosen = initial[ids].clone()
        for i, a in enumerate(counts):
            if a:
                chosen[i].copy_(cp[i, a - 1])
        expected_pool = initial.clone()
        expected_pool[ids] = chosen
        bits(dst, expected_pool)
        nxt = case([1] * b, dtype, h)
        ref, terminal, _ = decode(dict(nxt, state_in=chosen), 1)
        oo = torch.empty(b, 1, h, 128, device="cuda", dtype=torch.bfloat16)
        output_pool = dst.clone()
        prepare_decode(
            **dense(nxt, b, 1),
            state_in=dst,
            out=oo,
            state_out=output_pool,
            state_indices=slots,
            state_dtype=dtype
        ).run()
        bits(oo.view_as(ref), ref)
        bits(output_pool[ids], terminal)


def test_empty_indexed(dtype):
    x = case([], dtype, h=3)
    pool = torch.empty(0, 3, 128, 128, device="cuda", dtype=dtype)
    idx = torch.empty(0, device="cuda", dtype=torch.int32)
    prepare_prefill(
        **dict(x, state_in=pool),
        out=torch.empty_like(x["q"]),
        state_out=pool,
        state_indices=idx,
        state_dtype=dtype
    ).run()
    for q in [1, 8]:
        z = {
            k: torch.empty(
                (0, q, 3) if k == "beta" else (0, q, 3, 128),
                device="cuda",
                dtype=torch.bfloat16,
            )
            for k in NAMES
        }
        z.update(A_log=x["A_log"], dt_bias=x["dt_bias"])
        cp = (
            torch.empty(0, 8, 3, 128, 128, device="cuda", dtype=dtype)
            if q == 8
            else None
        )
        p = prepare_decode(
            **z,
            state_in=pool,
            out=torch.empty_like(z["q"]),
            state_out=pool,
            checkpoint_states=cp,
            state_indices=idx,
            state_dtype=dtype,
            num_accepted_tokens=idx if q == 8 else None
        )
        p.run()
        if q == 8:
            p.commit()


def test_prefill_dynamic_slots_and_lengths(dtype):
    lengths = [0, 17, 129, 258]
    x = case(lengths, dtype, h=3)
    pool = torch.randn(17, 3, 128, 128, device="cuda").to(dtype)
    dst = pool.clone()
    indices = torch.tensor([11, 0, 9, 3], device="cuda", dtype=torch.int32)
    out = torch.full_like(x["q"], 31)
    p = prepare_prefill(
        **dict(x, state_in=pool),
        out=out,
        state_out=dst,
        state_indices=indices,
        state_dtype=dtype,
        workspace_limit_bytes=2 * 1024**2
    )
    p.run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        p.run()
    for lengths, slots in [
        ([101] * 4, [2, 8, 12, 16]),
        ([404, 0, 0, 0], [9, 7, 3, 0]),
        ([0] * 4, [3, 0, 1, 8]),
    ]:
        cu = [0]
        for n in lengths:
            cu.append(cu[-1] + n)
        x["cu_seqlens"].copy_(torch.tensor(cu, device="cuda", dtype=torch.int32))
        indices.copy_(torch.tensor(slots, device="cuda", dtype=torch.int32))
        before = dst.clone()
        graph.replay()
        y = dict(x, state_in=pool[indices.long()].contiguous())
        expected, final, _ = prefill(y)
        bits(out[:, : sum(lengths)], expected[:, : sum(lengths)])
        before[indices.long()] = final
        bits(dst, before)


@pytest.mark.parametrize("b", [1, 64, 65])
def test_acceptance_runtime_batch(dtype, b):
    h = 3
    x = case([8] * b, dtype, h)
    pool = torch.cat([x["state_in"], x["state_in"]], 0)
    indices = torch.arange(b, device="cuda", dtype=torch.int32) + b
    counts = torch.arange(b, device="cuda", dtype=torch.int32) % 9
    cp = torch.empty(b, 8, h, 128, 128, device="cuda", dtype=dtype)
    out = torch.empty(b, 8, h, 128, device="cuda", dtype=torch.bfloat16)
    dst = pool.clone()
    from vllm.models.kimi_k3.nvidia.kda_fp32_state._checkpoint import compiled_commit

    p = prepare_decode(
        **dense(x, b, 8),
        state_in=pool,
        out=out,
        state_out=dst,
        checkpoint_states=cp,
        state_indices=indices,
        num_accepted_tokens=counts,
        state_dtype=dtype
    )
    p.run()
    p.commit()
    expected = x["state_in"].clone()
    for i in range(b):
        a = i % 9
        if a:
            expected[i].copy_(cp[i, a - 1])
    bits(dst[indices.long()], expected)
    bits(dst[:b], pool[:b])
    before = compiled_commit.cache_info().misses
    counts.zero_()
    p.run()
    p.commit()
    bits(dst, pool)
    assert compiled_commit.cache_info().misses == before


@pytest.mark.parametrize("h", [12, 16, 32])
def test_pool_multigpu_head_shards(dtype, h):
    if torch.cuda.device_count() < 2:
        pytest.skip("two GPUs required")
    b = 9
    x = case([8] * b, dtype, h)
    full, _, cp = decode(x, 8)
    counts = torch.arange(b, device="cuda", dtype=torch.int32)
    accepted = x["state_in"].clone()
    for i in range(1, b):
        accepted[i].copy_(cp[i, i - 1])
    for device in range(2):
        lo, hi = (device * (h // 2), (device + 1) * (h // 2))
        with torch.cuda.device(device):
            dstdev = torch.device("cuda", device)
            z = {
                k: x[k][..., lo:hi, :].contiguous().to(dstdev)
                for k in ("q", "k", "v", "g")
            }
            z["beta"] = x["beta"][..., lo:hi].contiguous().to(dstdev)
            z["A_log"] = x["A_log"][lo:hi].contiguous().to(dstdev)
            z["dt_bias"] = x["dt_bias"][lo:hi].contiguous().to(dstdev)
            state = x["state_in"][:, lo:hi].contiguous().to(dstdev)
            pool = torch.zeros(2 * b + 1, hi - lo, 128, 128, device=dstdev, dtype=dtype)
            slots = torch.arange(b, device=dstdev, dtype=torch.int32) * 2 + 1
            pool[slots.long()] = state
            cc = torch.empty(b, 8, hi - lo, 128, 128, device=dstdev, dtype=dtype)
            oo = torch.empty(b, 8, hi - lo, 128, device=dstdev, dtype=torch.bfloat16)
            p = prepare_decode(
                **dense(z, b, 8),
                state_in=pool,
                out=oo,
                state_out=pool,
                checkpoint_states=cc,
                state_indices=slots,
                num_accepted_tokens=counts.to(dstdev),
                state_dtype=dtype
            )
            p.run()
            p.commit()
            bits(oo.to("cuda:0").view(1, b * 8, hi - lo, 128), full[..., lo:hi, :])
            bits(cc.to("cuda:0"), cp[:, :, lo:hi])
            bits(pool[slots.long()].to("cuda:0"), accepted[:, lo:hi])


def test_pool_metadata_validation(dtype):
    x = case([8] * 2, dtype, 3)
    z = dense(x, 2, 8)
    cp = torch.empty(2, 8, 3, 128, 128, device="cuda", dtype=dtype)
    out = torch.empty_like(z["q"])
    state = x["state_in"]
    kw = dict(state_in=state, out=out, checkpoint_states=cp, state_dtype=dtype)
    for slots in [
        torch.zeros(2, device="cuda", dtype=torch.int64),
        torch.zeros(3, device="cuda", dtype=torch.int32),
        torch.zeros(2, dtype=torch.int32),
    ]:
        with pytest.raises(ValueError, match="state_indices"):
            prepare_decode(**z, **kw, state_indices=slots)
    counts = torch.zeros(2, device="cuda", dtype=torch.int32)
    with pytest.raises(ValueError, match="state_out"):
        prepare_decode(**z, **kw, num_accepted_tokens=counts)
    with pytest.raises(ValueError, match="num_accepted_tokens"):
        prepare_decode(
            **z,
            **kw,
            state_out=torch.empty_like(state),
            num_accepted_tokens=counts.long()
        )
    with pytest.raises(ValueError, match="overlapping"):
        prepare_decode(
            **z,
            **kw,
            state_out=cp.view(-1, 3, 128, 128)[:2],
            num_accepted_tokens=counts
        )
