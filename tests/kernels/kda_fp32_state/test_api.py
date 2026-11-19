"""Exercise the public facade, including checkpoint continuation and dynamic Graphs."""

import pytest
import torch
from vllm.models.kimi_k3.nvidia.kda_fp32_state import prepare_decode, prepare_prefill
from .helpers import bits
from .helpers import make_case

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
DTYPES = [torch.float32]
NAMES = ("q", "k", "v", "g", "beta")


def case(lengths, dtype, h=3):
    x = make_case(lengths, h=h)
    x["state_in"] = x["state_in"].to(dtype)
    x["cu_seqlens"] = x["cu_seqlens"].int()
    return x


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("h", [1, 3, 6, 12, 16, 32])
def test_q8_checkpoints_match_q1_and_prefill(dtype, h):
    b = 9
    x = case([8] * b, dtype, h)
    original = x["state_in"].clone()
    dense = {k: x[k].view(b, 8, *x[k].shape[2:]) for k in NAMES}
    dense.update({k: x[k] for k in ("A_log", "dt_bias", "state_in")})
    out = torch.empty_like(dense["q"])
    cp = torch.empty(b, 8, h, 128, 128, device="cuda", dtype=dtype)
    p = prepare_decode(**dense, out=out, checkpoint_states=cp, state_dtype=dtype)
    _, terminal = p.run()
    eo, es = (torch.empty_like(x["q"]), torch.empty_like(original))
    prepare_prefill(**x, out=eo, state_out=es, state_dtype=dtype).run()
    bits(out.view_as(eo), eo)
    bits(terminal, es)
    state = original
    for t in range(8):
        token = {
            k: dense[k][:, t : t + 1].clone(memory_format=torch.contiguous_format)
            for k in NAMES
        }
        o, s = (torch.empty_like(token["q"]), torch.empty_like(state))
        prepare_decode(
            **token,
            A_log=x["A_log"],
            dt_bias=x["dt_bias"],
            state_in=state,
            out=o,
            state_out=s,
            state_dtype=dtype
        ).run()
        bits(o[:, 0], out[:, t])
        bits(s, cp[:, t])
        state = s
    for i in range(9):
        prefix = {
            k: dense[k][i : i + 1, :i].clone(memory_format=torch.contiguous_format)
            for k in NAMES
        }
        s = torch.empty_like(original[i : i + 1])
        prepare_prefill(
            **prefix,
            A_log=x["A_log"],
            dt_bias=x["dt_bias"],
            state_in=original[i : i + 1],
            out=torch.empty_like(prefix["q"]),
            state_out=s,
            state_dtype=dtype
        ).run()
        accepted = original[i : i + 1] if i == 0 else cp[i : i + 1, i - 1].contiguous()
        bits(accepted, s)
        suffix = {
            k: dense[k][i : i + 1, i:].clone(memory_format=torch.contiguous_format)
            for k in NAMES
        }
        prepare_prefill(
            **suffix,
            A_log=x["A_log"],
            dt_bias=x["dt_bias"],
            state_in=accepted,
            out=torch.empty_like(suffix["q"]),
            state_out=s,
            state_dtype=dtype
        ).run()
        bits(s, es[i : i + 1])
    bits(x["state_in"], original)


@pytest.mark.parametrize("dtype", DTYPES)
def test_varlen_graph_dynamic_metadata(dtype):
    x = case([0, 8, 129], dtype)
    o, s = (torch.empty_like(x["q"]), torch.empty_like(x["state_in"]))
    plan = prepare_prefill(**x, out=o, state_out=s, state_dtype=dtype)
    plan.run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        plan.run()
    for lengths in ([0, 8, 129], [17, 0, 120]):
        x["cu_seqlens"].copy_(
            torch.tensor(
                [0, lengths[0], sum(lengths[:2]), sum(lengths)],
                device="cuda",
                dtype=torch.int32,
            )
        )
        x["q"].normal_()
        graph.replay()
        expected_o, expected_s = (torch.empty_like(o), torch.empty_like(s))
        prepare_prefill(
            **x, out=expected_o, state_out=expected_s, state_dtype=dtype
        ).run()
        bits(o, expected_o)
        bits(s, expected_s)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("q", [1, 8])
def test_decode_graph_and_empty(dtype, q):
    for b in [0, 3]:
        x = case([q] * b, dtype)
        dense = {k: x[k].view(b, q, *x[k].shape[2:]) for k in NAMES}
        dense.update({k: x[k] for k in ("A_log", "dt_bias", "state_in")})
        o = torch.empty_like(dense["q"])
        cp = (
            torch.empty(b, 8, 3, 128, 128, device="cuda", dtype=dtype)
            if q == 8
            else None
        )
        state = None if q == 8 else torch.empty_like(x["state_in"])
        p = prepare_decode(
            **dense, out=o, state_out=state, checkpoint_states=cp, state_dtype=dtype
        )
        p.run()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            p.run()
        dense["q"].normal_()
        graph.replay()
        saved_o, saved_s = (o.clone(), p.state_out.clone())
        saved_cp = None if cp is None else cp.clone()
        p.run()
        bits(o, saved_o)
        bits(p.state_out, saved_s)
        if cp is not None:
            bits(cp, saved_cp)


@pytest.mark.parametrize("dtype", DTYPES)
def test_contract_errors(dtype):
    x = case([8], dtype)
    x.pop("cu_seqlens")
    cp = torch.empty(1, 8, 3, 128, 128, device="cuda", dtype=dtype)
    out = torch.empty_like(x["q"])
    with pytest.raises(ValueError, match="state_dtype must be torch.float32"):
        prepare_decode(**x, out=out, checkpoint_states=cp, state_dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="without checkpoints requires state_out"):
        prepare_decode(**x, out=out, state_dtype=dtype)
    with pytest.raises(ValueError):
        prepare_decode(**x, out=out, checkpoint_states=cp[..., :64], state_dtype=dtype)
    with pytest.raises(ValueError):
        prepare_decode(
            **dict(x, state_in=cp[:, 0]),
            out=out,
            checkpoint_states=cp,
            state_dtype=dtype
        )
    torch.cuda.synchronize()
    with torch.cuda.graph(torch.cuda.CUDAGraph()):
        with pytest.raises(RuntimeError, match="outside CUDA Graph"):
            prepare_decode(**x, out=out, checkpoint_states=cp, state_dtype=dtype)
