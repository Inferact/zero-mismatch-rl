"""Runtime placement and bounded windows; no input metadata is read on host."""

import functools
from .schedule import SEGMENT


# (latency floor, fixed cost, per request-head/SM cost), microseconds/token.
# Fit on independent varlen shapes N=1/3/5/7/11/17/29/31,
# H=7/11/13/19/23/29/17/37, lengths 1408..1536. These are global
# engine costs; no benchmark N/H/L value selects a production branch.
_ENGINE_COST = {
    2: (0.109, 0.000, 0.570),
    4: (0.137, 0.028, 0.433),
    8: (0.240, 0.102, 0.375),
}


def _placement(n, h, sms, max_rows=4):
    """Choose the lowest predicted latency using work normalized by SM count.

    A latency floor handles underfilled grids; the linear term models steady
    throughput. All choices share the same compiled scan and arithmetic.
    """
    # When a four-row grid leaves unused SM positions, split whole longest
    # requests without increasing the number of one-CTA-per-SM waves.
    base = 4 * n * h
    for capacity in (sms, 2 * sms, 3 * sms):
        if capacity - sms < base <= capacity < 2 * base:
            split = max(0, min(n - 1, (capacity - base) // (4 * h)))
            if split:
                return 4, split * h
    work = n * h / sms

    def cost(rows):
        floor, fixed, slope = _ENGINE_COST[rows]
        return max(floor, fixed + slope * work)

    rows = min((r for r in _ENGINE_COST if r <= max_rows), key=lambda r: (cost(r), -r))
    return rows, 0


def _window_cost(slots, segments, n):
    """Sum over windows of the busiest request's segment count (round-robin
    ranks, equal-length requests) plus a relaunch overhead per window."""
    cost = 0.0
    for start in range(0, segments, slots):
        cost += (min(slots, segments - start) + n - 1) // n + 0.5
    return cost


@functools.lru_cache(maxsize=256)
def _plan_windows(limit, capacity, n, total, heads, many_waves=False):
    """Partition the bounded workspace into ordered, nonoverlapping adjacent windows."""
    bound = max(1, min(capacity, (total + (SEGMENT - 1) * n) // SEGMENT))
    # Keep a fully resident scan in one window when its coefficients fit.
    # Multi-wave scans can hide the next coefficient window behind the
    # current recurrence; otherwise avoid extra state I/O and scan startup.
    if bound <= limit and (not many_waves or bound <= 2 * n):
        return [(0, bound, 0, 1)]
    if limit < 2:
        return [(start, 1, 0, 1) for start in range(bound)]
    half = limit // 2
    first = max(1, min(bound, 2 * n, half - half % n if half >= n else half))
    rest = bound - first
    if many_waves and rest - (limit - first) > min(n, first):
        # Grids beyond one resident wave whose rest needs several windows of
        # at most half the workspace: each of those windows' coefficients
        # hides behind a one-round scan, so the first window (exposed
        # coefficients) keeps one segment per request.
        first = max(1, min(bound, n, half))
        rest = bound - first
    sizes = [first]
    if 0 < rest <= limit - first:
        sizes.append(rest)
    elif 0 < rest - (limit - first) <= min(n, first):
        # The bound overshoots the real segment count by up to N - 1, so a
        # small overflow gets its own (usually empty) last window instead of
        # splitting the run into more full windows.
        sizes += [limit - first, rest - (limit - first)]
    elif rest > 0:
        # Windows of at most half the workspace. Windows take segments
        # round-robin across requests, so the size minimizing the modelled
        # cost (_window_cost: busiest request's segments plus a relaunch per
        # window) is chosen among even splits and multiples of N.
        least = (rest + half - 1) // half
        candidates = {half} | {(rest + k - 1) // k for k in range(least, least + 3)}
        candidates |= {n * j for j in range(max(1, half // n - 2), half // n + 1)}
        size = min(
            (c for c in candidates if 0 < c <= half),
            key=lambda c: (_window_cost(c, rest, n), -c),
        )
        windows = (rest + size - 1) // size
        sizes += [size] * (windows - 1) + [rest - size * (windows - 1)]
        if len(sizes) >= 3 and sizes[-1] + sizes[-2] <= half:
            sizes[-2:] = [sizes[-2] + sizes[-1]]
    plan, start = [], 0
    for index, count in enumerate(sizes):
        offset = 0 if index % 2 == 0 else limit - count
        wait = 1 if index == 0 or sizes[index - 1] + count > limit else 0
        plan.append((start, count, offset, wait))
        start += count
    return plan
