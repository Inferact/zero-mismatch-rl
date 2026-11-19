"""Every reachable prefill engine variant, compared bytewise with differently planned runs.

The prefill scan is one compiled kernel, but the host picks among several
register engines, CTA layouts and window schedules per call (see variants.py).
A bytewise result at the shapes a test happens to use says nothing about the
variants it never reaches. This module plans a grid of (requests, heads, packed
tokens, workspace limit) points plus the K3.1 rollout and trainer shapes,
groups them by the variant they select, and runs the cheapest point of each
variant, plus the cheapest point with at least 256 tokens per request on
average (several 128-token segments, partial tails, window boundaries), against
runs that select other variants: a smaller workspace, each request alone,
reversed request order, two consecutive calls that continue the state, extra
requests appended so the engine changes, a pooled indexed state, and the Q1
decode chain over the last tokens of each request.

The test fails if a variant was only ever compared with runs that selected the
same variant, if the planner rejected a point, or if the variant planned for a
point is not the one the real launch selected.

Set KDA_VARIANT_REPORT=path to write the variant table as JSON.
"""

import itertools
import json
import os

import torch
from vllm.models.kimi_k3.nvidia.kda_fp32_state import prepare_prefill

from . import variants
from .helpers import case, decode, prefill, select

TAIL = 3
NAMES = ("q", "k", "v", "g", "beta")


def same(label, a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        raise AssertionError(
            f"{label}: {tuple(a.shape)} {a.dtype} vs {tuple(b.shape)} {b.dtype}"
        )
    a, b = a.contiguous(), b.contiguous()
    if not (torch.isfinite(a).all() and torch.isfinite(b).all()):
        raise AssertionError(f"{label}: non-finite values")
    differing = (a.view(torch.uint8) != b.view(torch.uint8)).flatten().nonzero()
    if differing.numel():
        first = int(differing[0]) // a.element_size()
        coordinates = torch.unravel_index(torch.tensor(first), a.shape)
        where = tuple(int(i) for i in coordinates)
        raise AssertionError(
            f"{label}: {differing.numel()} differing bytes, first element {where}"
        )


def join(first, first_lengths, second, second_lengths):
    pieces, a, b = [], 0, 0
    for n, m in zip(first_lengths, second_lengths):
        pieces += [first[:, a : a + n], second[:, b : b + m]]
        a, b = a + n, b + m
    return torch.cat(pieces, 1)


def append_requests(x, fillers, heads):
    extra = case(fillers, torch.float32, heads)
    y = {name: torch.cat([x[name], extra[name]], 1) for name in NAMES}
    y.update(A_log=x["A_log"], dt_bias=x["dt_bias"])
    y["cu_seqlens"] = torch.cat(
        [x["cu_seqlens"], x["cu_seqlens"][-1] + extra["cu_seqlens"][1:]]
    )
    y["state_in"] = torch.cat([x["state_in"], extra["state_in"]], 0)
    return y


def check_point(variant, point):
    requests, heads, tokens, limit = point
    sms = variants.sm_count()
    lengths = variants.ragged_lengths(requests, tokens)
    offsets = [0, *itertools.accumulate(lengths)]
    spans = [(0, n) for n in lengths]
    x = case(lengths, torch.float32, heads)
    selected = []

    def run(label, y, **kwargs):
        with variants.recorded_binds() as seen:
            out, state, plan = prefill(y, **kwargs)
        count = y["state_in"].shape[0]
        selected.append((label, variants.describe(seen[-1], count, heads, sms)))
        return out, state, plan

    main, final, plan = run("main", x, workspace_limit_bytes=limit)
    launched = selected[0][1]
    assert (
        launched.variant == variant
    ), f"planning selected [{variant}] but the launch selected [{launched.variant}]"
    assert plan.workspace_bytes <= limit

    for small in (variants.small_workspace(heads), variants.smallest_workspace(heads)):
        out, state, _ = run("small workspace", x, workspace_limit_bytes=small)
        same("small workspace", out, main)
        same("small workspace state", state, final)
        if selected[-1][1].windows != launched.windows:
            break

    alone = range(requests) if requests <= 4 else {0, requests // 2, requests - 1}
    for i in sorted(alone):
        one, _ = select(x, lengths, spans, order=[i])
        out, state, _ = run(f"request {i} alone", one)
        same(f"request {i} alone", out, main[:, offsets[i] : offsets[i + 1]])
        same(f"request {i} alone state", state, final[i : i + 1])

    order = list(reversed(range(requests)))
    reordered, _ = select(x, lengths, spans, order=order)
    out, state, _ = run("reversed order", reordered, workspace_limit_bytes=limit)
    same(
        "reversed order",
        out,
        torch.cat([main[:, offsets[i] : offsets[i + 1]] for i in order], 1),
    )
    same("reversed order state", state, final[order])

    cuts = [min(n, 1 + 5 * n // 11) for n in lengths]
    first, first_lengths = select(x, lengths, [(0, c) for c in cuts])
    out1, middle, _ = run("two calls", first, workspace_limit_bytes=limit)
    second, second_lengths = select(
        x, lengths, [(c, n) for c, n in zip(cuts, lengths)], state=middle
    )
    out2, state, _ = run("two calls", second, workspace_limit_bytes=limit)
    same("two calls", join(out1, first_lengths, out2, second_lengths), main)
    same("two calls state", state, final)

    fillers = variants.crossing_fillers(point, variant)
    if fillers is not None:
        grown = append_requests(x, fillers, heads)
        out, state, _ = run("extra requests", grown, workspace_limit_bytes=limit)
        same("extra requests", out[:, :tokens], main)
        same("extra requests state", state[:requests], final)

    slots = torch.arange(requests - 1, -1, -1, device="cuda", dtype=torch.int32)
    pool = torch.full(
        (requests + 2, heads, 128, 128), 3.0, device="cuda", dtype=torch.float32
    )
    pool[slots.long()] = x["state_in"]
    destination = pool.clone()
    pooled = torch.empty_like(x["q"])
    prepare_prefill(
        **dict(x, state_in=pool),
        out=pooled,
        state_out=destination,
        state_dtype=torch.float32,
        state_indices=slots,
        workspace_limit_bytes=limit,
    ).run()
    pool[slots.long()] = final
    same("pooled state", pooled, main)
    same("pooled state pool", destination, pool)

    tail = TAIL if any(n >= TAIL for n in lengths) else 1
    chain = [i for i, n in enumerate(lengths) if n >= tail][:4]
    head, _ = select(x, lengths, [(0, max(0, n - tail)) for n in lengths], order=chain)
    _, state, _ = prefill(head)
    for step in range(tail):
        token_spans = [
            (n - tail + step, n - tail + step + 1) if n >= tail else (0, 0)
            for n in lengths
        ]
        token, _ = select(x, lengths, token_spans, order=chain, state=state)
        out, state, _ = decode(token, 1)
        expected = torch.stack(
            [main[0, offsets[i] + lengths[i] - tail + step] for i in chain]
        )[None]
        same(f"q1 decode chain step {step}", out, expected)
    same("q1 decode chain state", state, final[chain])

    return [label for label, planned in selected if planned.variant != variant]


def test_every_reachable_variant_matches_differently_planned_runs():
    dropped = []
    table = variants.reachable(dropped=dropped)
    assert table, "no variant was planned"
    assert not dropped, f"the planner rejected {[tuple(p) for p in dropped[:5]]}"
    failures, rows = [], []
    for variant, chosen in table.items():
        crossed, failed = [], False
        for point in chosen:
            try:
                crossed += check_point(variant, point)
            except AssertionError as error:
                failed = True
                failures.append(f"[{variant}] at {tuple(point)}: {error}")
        if not crossed and not failed:
            failures.append(
                f"[{variant}] was only compared with runs that selected the same variant"
            )
        rows.append(
            dict(
                variant=str(variant),
                **variant._asdict(),
                points=[point._asdict() for point in chosen],
                cross_variant_runs=sorted(set(crossed)),
            )
        )
    report = os.environ.get("KDA_VARIANT_REPORT")
    if report:
        rows.sort(key=lambda row: row["variant"])
        with open(report, "w") as stream:
            json.dump(dict(sms=variants.sm_count(), variants=rows), stream, indent=2)
    assert not failures, "\n".join(failures)
