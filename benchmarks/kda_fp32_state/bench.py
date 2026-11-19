"""Time the complete prepared KDA API with fixed-address CUDA Graph replay."""

import argparse
import itertools
import json
import statistics

import torch
from vllm.models.kimi_k3.nvidia.kda_fp32_state import prepare_decode, prepare_prefill


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["decode", "prefill"], default="prefill")
    parser.add_argument("--lengths", default="0,128,1024,4096")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--query-length", type=int, default=1)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--workspace-mib", type=int, default=256)
    parser.add_argument(
        "--state-pool",
        action="store_true",
        help="use device slot indices into a larger pool",
    )
    parser.add_argument(
        "--accept-counts",
        default=None,
        help="Accepted counts 0..Q, repeated across B",
    )
    args = parser.parse_args()
    if args.query_length < 1:
        parser.error("--query-length must be positive")
    counts = None
    if args.accept_counts is not None:
        counts = [int(a) for a in args.accept_counts.split(",")]
        if (
            args.mode != "decode"
            or args.query_length <= 1
            or not counts
            or min(counts) < 0
            or max(counts) > args.query_length
        ):
            parser.error("--accept-counts requires Q>1 and values 0..Q")
    dtype = torch.float32
    lengths = (
        [int(s) for s in args.lengths.split(",")]
        if args.mode == "prefill"
        else [args.query_length] * args.batch
    )
    if (
        not lengths
        or min(lengths) < 0
        or sum(lengths) == 0
        or args.heads < 1
        or args.samples < 1
        or args.workspace_mib < 1
    ):
        parser.error(
            "require nonnegative lengths, positive total/heads/samples/workspace"
        )
    torch.manual_seed(7)
    b, h, total = len(lengths), args.heads, sum(lengths)
    shape = (
        (1, total, h, 128) if args.mode == "prefill" else (b, args.query_length, h, 128)
    )
    xs = [torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(4)]
    beta = torch.randn(shape[:-1], device="cuda", dtype=torch.bfloat16)
    a, bias = torch.zeros(h, device="cuda"), torch.zeros(h, 128, device="cuda")
    initial = torch.randn(b, h, 128, 128, device="cuda", dtype=dtype) * 0.1
    indices = None
    if args.state_pool:
        indices = (
            torch.arange(b, device="cuda", dtype=torch.int32).flip(0).contiguous() * 2
            + 1
        )
        pool = torch.zeros(2 * b + 1, h, 128, 128, device="cuda", dtype=dtype)
        pool[indices.long()] = initial
        initial = pool
    accepted = (
        None
        if counts is None
        else torch.tensor(
            [counts[i % len(counts)] for i in range(b)],
            device="cuda",
            dtype=torch.int32,
        )
    )
    output = torch.empty_like(xs[0])
    saved = initial.clone()
    if args.mode == "prefill":
        terminal = torch.zeros_like(initial)
        cu = torch.tensor(
            [0] + list(itertools.accumulate(lengths)), device="cuda", dtype=torch.int32
        )
        plan = prepare_prefill(
            *xs,
            beta,
            a,
            bias,
            initial,
            output,
            terminal,
            cu_seqlens=cu,
            state_indices=indices,
            state_dtype=dtype,
            workspace_limit_bytes=args.workspace_mib * 1024**2,
        )
        written = [output, terminal]
    elif args.query_length == 1:
        terminal = torch.zeros_like(initial)
        plan = prepare_decode(
            *xs,
            beta,
            a,
            bias,
            initial,
            output,
            terminal,
            state_dtype=dtype,
            state_indices=indices,
            workspace_limit_bytes=args.workspace_mib * 1024**2,
        )
        written = [output, terminal]
    else:
        checkpoints = torch.empty(b, args.query_length, h, 128, 128, device="cuda", dtype=dtype)
        terminal = torch.zeros_like(initial) if accepted is not None else None
        plan = prepare_decode(
            *xs,
            beta,
            a,
            bias,
            initial,
            output,
            terminal,
            checkpoint_states=checkpoints,
            state_indices=indices,
            num_accepted_tokens=accepted,
            state_dtype=dtype,
            workspace_limit_bytes=args.workspace_mib * 1024**2,
        )
        written = [output, checkpoints] + ([terminal] if accepted is not None else [])

    def run():
        plan.run()
        if accepted is not None:
            plan.commit()

    for _ in range(3):
        run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    samples = []
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    for i in range(args.samples + 3):
        start.record()
        for _ in range(10):
            graph.replay()
        end.record()
        end.synchronize()
        if i >= 3:
            samples.append(start.elapsed_time(end) * 100)
    assert torch.equal(initial.view(torch.uint8), saved.view(torch.uint8))
    assert all(torch.isfinite(t).all() for t in written)
    import cutlass

    print(
        json.dumps(
            dict(
                gpu=torch.cuda.get_device_name(),
                torch=torch.__version__,
                cuda=torch.version.cuda,
                cute_dsl=cutlass.__version__,
                mode=args.mode,
                lengths=lengths,
                heads=h,
                state_dtype="fp32",
                workspace_bytes=plan.workspace_bytes,
                state_pool=args.state_pool,
                accept_counts=counts,
                scope="complete prepared GPU API"
                + (" including commit" if accepted is not None else "")
                + "; fixed-address Graph; prepare/prefix/acceptance decision excluded",
                median_us=statistics.median(samples),
                samples_us=samples,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
