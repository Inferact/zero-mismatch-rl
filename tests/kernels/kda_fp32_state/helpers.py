"""Input construction and bitwise checks; no kernel arithmetic."""

import torch
from vllm.models.kimi_k3.nvidia.kda_fp32_state import prepare_decode, prepare_prefill


def bits(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    assert torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
    )


def make_case(lengths, h=12, *, seed=123, padding=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    t, n = sum(lengths) + padding, len(lengths)

    def rand(shape, dtype=torch.bfloat16, scale=0.2):
        return (torch.randn(shape, device="cuda", generator=gen) * scale).to(dtype)

    xs = [rand((1, t, h, 128)) for _ in range(4)]
    beta = rand((1, t, h))
    a = rand((h,), torch.float32)
    dt = rand((h, 128), torch.float32)
    state = rand((n, h, 128, 128), torch.float32)
    return dict(
        q=xs[0],
        k=xs[1],
        v=xs[2],
        g=xs[3],
        beta=beta,
        A_log=a,
        dt_bias=dt,
        cu_seqlens=torch.tensor(
            [0] + list(torch.tensor(lengths).cumsum(0).tolist()),
            device="cuda",
            dtype=torch.int32,
        ),
        state_in=state,
    )


def select(case, lengths, spans, order=None, state=None):
    """Pack arbitrary per-request intervals using host-known metadata."""
    order = list(range(len(lengths))) if order is None else list(order)
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    result = {}
    for name in ("q", "k", "v", "g", "beta"):
        x = case[name]
        result[name] = torch.cat(
            [x[:, offsets[r] + spans[r][0] : offsets[r] + spans[r][1]] for r in order],
            dim=1,
        ).contiguous()
    result.update(A_log=case["A_log"], dt_bias=case["dt_bias"])
    lens = [spans[r][1] - spans[r][0] for r in order]
    offsets = [0]
    for length in lens:
        offsets.append(offsets[-1] + length)
    result["cu_seqlens"] = torch.tensor(offsets, device="cuda", dtype=torch.int32)
    result["state_in"] = (
        case["state_in"][order].contiguous() if state is None else state
    )
    return result, lens


def case(lengths, dtype=torch.float32, h=3, padding=0):
    x = make_case(lengths, h=h, padding=padding)
    x["state_in"] = x["state_in"].to(dtype)
    return x


def prefill(x, **kwargs):
    out, state = torch.full_like(x["q"], 31), torch.empty_like(x["state_in"])
    plan = prepare_prefill(
        **x, out=out, state_out=state, state_dtype=x["state_in"].dtype, **kwargs
    )
    plan.run()
    return out, state, plan


NAMES = ("q", "k", "v", "g", "beta")


def segment(x, lengths, starts, ends, state=None):
    y, ls = select(x, lengths, list(zip(starts, ends)), state=state)
    y["cu_seqlens"] = y["cu_seqlens"].int()
    return y, ls


def decode(x, q, **kwargs):
    b = x["state_in"].shape[0]
    dense = {k: x[k].view(b, q, *x[k].shape[2:]) for k in NAMES}
    dense.update({k: x[k] for k in ("A_log", "dt_bias", "state_in")})
    out = torch.empty_like(dense["q"])
    dtype = x["state_in"].dtype
    cp = (
        torch.empty(b, q, x["q"].shape[2], 128, 128, device=out.device, dtype=dtype)
        if q > 1
        else None
    )
    state = None if q > 1 else torch.empty_like(x["state_in"])
    plan = prepare_decode(
        **dense,
        out=out,
        state_out=state,
        checkpoint_states=cp,
        state_dtype=dtype,
        **kwargs
    )
    out, state = plan.run()
    return out.view_as(x["q"]), state, cp
