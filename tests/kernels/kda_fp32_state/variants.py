"""Planner introspection for the FP32-state prefill scan.

The scan is one compiled kernel, but plan.bind() chooses from the request count,
head count, packed token capacity, workspace limit and SM count which row
engine, CTA layout, radix flags, shared-memory size and number of coefficient
windows a call uses. Request lengths are device-side runtime data and take no
part in that choice. describe() reads the choice back from the launches that
bind() returns, so no copy of the policy lives here.
"""

import contextlib
import importlib
import itertools
from typing import NamedTuple

import torch
from vllm.models.kimi_k3.nvidia.kda_fp32_state import prepare_prefill

PREFILL = "vllm.models.kimi_k3.nvidia.kda_fp32_state.kernels.prefill"
MIB = 1024**2
DEFAULT_LIMIT = 256 * MIB
SEGMENT_ROW_BYTES = 128 * 392 * 4
HEADS = (1, 3, 6, 12, 16, 32, 48, 96)
REQUESTS = (1, 2, 3, 4, 6, 8, 12, 16, 32, 64)
TOKENS_PER_REQUEST = (1, 64, 128, 512, 2048, 5120, 8192)
WORKSPACE_LIMITS = (DEFAULT_LIMIT, 64 * MIB, 24 * MIB)
MAX_TOKEN_HEADS = 4_000_000
LONG_PER_REQUEST = 256
SCAN_ARGUMENTS = {"rows": 12, "long_pairs": 13, "radix4": 14, "smem": 18}


class Variant(NamedTuple):
    rows: int
    long_split: bool
    radix_low: int
    two_ahead: bool
    kpack: bool
    pair: bool
    smem80: bool
    many_waves: bool
    windows: int
    overlap: bool

    def __str__(self):
        names = (
            "long_split",
            "two_ahead",
            "kpack",
            "pair",
            "smem80",
            "many_waves",
            "overlap",
        )
        windows = f"win{self.windows}" + ("+" if self.windows == 3 else "")
        flags = [name for name in names if getattr(self, name)]
        return " ".join([f"rows{self.rows}", f"low{self.radix_low}", windows, *flags])


class Point(NamedTuple):
    requests: int
    heads: int
    tokens: int
    workspace_limit: int

    @property
    def cost(self):
        return self.tokens * self.heads


class Planned(NamedTuple):
    variant: Variant
    windows: int
    workspace_bytes: int
    radix4: int
    smem: int


@contextlib.contextmanager
def recorded_binds():
    plan = importlib.import_module(PREFILL + ".plan")
    original, seen = plan.bind, []

    def spy(*args, **kwargs):
        calls = original(*args, **kwargs)
        seen.append(calls)
        return calls

    plan.bind = spy
    try:
        yield seen
    finally:
        plan.bind = original


def sm_count():
    return torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count


def describe(calls, requests, heads, sms):
    scan = importlib.import_module(PREFILL + ".scan")
    try:
        if len(calls) % 2 == 0:
            raise ValueError(len(calls))
        scans = calls[2::2]
        rows, long_pairs, radix4, smem = (
            scans[0][1][SCAN_ARGUMENTS[name]] for name in SCAN_ARGUMENTS
        )
    except (IndexError, TypeError, ValueError) as error:
        raise AssertionError(
            "plan.bind() no longer returns (schedule, (coeff, scan) * windows) "
            "with the scan arguments at the indices in SCAN_ARGUMENTS"
        ) from error
    ctas = long_pairs * 8 + (requests * heads - long_pairs) * (16 // rows)
    variant = Variant(
        rows=rows,
        long_split=bool(long_pairs),
        radix_low=radix4 & 7,
        two_ahead=bool(radix4 & scan.TWO_AHEAD),
        kpack=bool(radix4 & scan.KPACK),
        pair=bool(radix4 & scan.PAIR_ENGINE),
        smem80=smem == scan.SMEM_TWO_PER_SM,
        many_waves=ctas > 3 * sms,
        windows=min(len(scans), 3),
        overlap=bool(radix4 & scan.OVERLAP),
    )
    return Planned(variant, len(scans), 0, radix4, smem)


def plan_point(point):
    requests, heads, tokens, limit = point
    device = torch.device("cuda", torch.cuda.current_device())
    bf16 = dict(device=device, dtype=torch.bfloat16)
    q, k, v, g, out = (torch.empty(1, tokens, heads, 128, **bf16) for _ in range(5))
    beta = torch.empty(1, tokens, heads, **bf16)
    a_log = torch.empty(heads, device=device, dtype=torch.float32)
    dt_bias = torch.empty(heads, 128, device=device, dtype=torch.float32)
    state = torch.empty(requests, heads, 128, 128, device=device, dtype=torch.float32)
    cu_seqlens = torch.empty(requests + 1, device=device, dtype=torch.int32)
    with recorded_binds() as seen:
        try:
            plan = prepare_prefill(
                q,
                k,
                v,
                g,
                beta,
                a_log,
                dt_bias,
                state,
                out,
                state,
                cu_seqlens=cu_seqlens,
                workspace_limit_bytes=limit,
            )
        except ValueError as error:
            if "workspace" not in str(error):
                raise
            return None
    planned = describe(seen[-1], requests, heads, sm_count())
    return planned._replace(workspace_bytes=plan.workspace_bytes)


def grid():
    for heads, requests, per_request, limit in itertools.product(
        HEADS, REQUESTS, TOKENS_PER_REQUEST, WORKSPACE_LIMITS
    ):
        yield Point(requests, heads, requests * per_request, limit)


def envelope(heads):
    for requests in range(1, 33):
        sizes = {requests * m for m in (1, 2, 8, 32, 64, 128, 256, 512)} | {1024}
        for tokens in sorted(sizes):
            if requests <= tokens <= 1024:
                yield Point(requests, heads, tokens, DEFAULT_LIMIT)
    for requests, length in (
        (1, 1024),
        (1, 2048),
        (1, 5120),
        (2, 5120),
        (4, 5120),
        (8, 5120),
        (1, 32768),
        (1, 131072),
        (1, 196608),
    ):
        yield Point(requests, heads, requests * length, DEFAULT_LIMIT)


def points():
    return itertools.chain(grid(), envelope(12), envelope(96))


def plan_all(candidates=None, dropped=None):
    for index, point in enumerate(points() if candidates is None else candidates):
        if point.cost > MAX_TOKEN_HEADS:
            continue
        planned = plan_point(point)
        if planned is not None:
            yield point, planned
        elif dropped is not None:
            dropped.append(point)
        if index % 128 == 127:
            torch.cuda.empty_cache()
    torch.cuda.empty_cache()


def reachable(candidates=None, dropped=None):
    best = {}
    for point, planned in plan_all(candidates, dropped):
        entry = best.setdefault(planned.variant, [None, None])
        if entry[0] is None or point.cost < entry[0].cost:
            entry[0] = point
        long = point.tokens >= LONG_PER_REQUEST * point.requests
        if long and (entry[1] is None or point.cost < entry[1].cost):
            entry[1] = point
    return {
        variant: [point for point in dict.fromkeys(entry) if point is not None]
        for variant, entry in best.items()
    }


def ragged_lengths(requests, tokens):
    base, extra = divmod(tokens, requests)
    lengths = [base + (i < extra) for i in range(requests)]
    for i in range(0, requests - 1, 2):
        move = min(lengths[i], (7 * i + 5) % 13)
        lengths[i] -= move
        lengths[i + 1] += move
    if requests >= 4:
        lengths[-1] += lengths[requests // 2]
        lengths[requests // 2] = 0
    return lengths


def small_workspace(heads):
    return 2 * heads * SEGMENT_ROW_BYTES + MIB


def smallest_workspace(heads):
    return heads * SEGMENT_ROW_BYTES + MIB


def filler_lengths(count):
    return [1 + (29 * i + 7) % 61 for i in range(count)]


def crossing_fillers(point, variant):
    """Lengths of extra requests that make the planner select another variant.

    Appending requests changes the request count and the packed capacity, so the
    requests under test run on a different engine variant. None if no tried
    count changes the variant.
    """
    for count in (3, 9, 33, 130):
        lengths = filler_lengths(count)
        grown = Point(
            point.requests + count,
            point.heads,
            point.tokens + sum(lengths),
            point.workspace_limit,
        )
        planned = plan_point(grown)
        if planned is not None and planned.variant != variant:
            return lengths
    return None
