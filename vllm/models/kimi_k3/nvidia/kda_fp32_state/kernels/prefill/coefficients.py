"""Coefficient preparation for the windowed recurrent scan.

One warp per head computes, for two consecutive tokens of a segment, the
normalized q and k (FP32 sum of squares, butterfly sum, rsqrt), the decay
gates and the sigmoid beta, and writes them as 392-float workspace rows in
the swizzled layout the scan's consumer lanes read with 16-byte loads (both
tokens' loads issued first; each row's operations unchanged).
"""

import functools
import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
from ..math import decay_gate, LOG2_E
from ... import _primitives as prims
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm
from cutlass._mlir.dialects import arith as mlir_arith
from cutlass._mlir.dialects import math as mlir_math


from .schedule import SEGMENT

COUNTERS = 32  # block counters per window (overlapped windows)


FULL_MASK = 4294967295


TOKENS_PER_WARP = 4  # tokens of one head per warp (SEGMENT is a multiple)


TOKEN_BLOCKS = (
    SEGMENT // TOKENS_PER_WARP
)  # coefficient blocks per segment and head group


FAST_DIVISOR_LIMIT = (
    18446744073709551616.0  # 2^64: the Newton quotient below is div.rn's
)


@dsl_user_op
def _rcp_approx(x, *, loc=None, ip=None):
    """MUFU reciprocal estimate (the first step of div.rn's fast path)."""
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [x.ir_value(loc=loc, ip=ip)],
            "rcp.approx.ftz.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def _ex2(x):
    # The gate's fast-math exp2 (ex2.approx.ftz), as in decay_gate.
    return cutlass.Float32(
        mlir_math.exp2(x.ir_value(), fastmath=mlir_arith.FastMathFlags.fast)
    )


@cute.jit
def _gate_pair(x0, x1, slope, floor, one, zero, minus_one):
    """decay_gate of two biased raw gates with div.rn's own fast path:
    exp2(floor / (1 + exp2(slope x))). The quotient is formed by the same
    MUFU estimate and five FMAs (paired) that div.rn executes when its range
    check passes; callers fall back to decay_gate when a divisor leaves
    [1, 2^64), so every result has div.rn's bits. Returns (gate0, gate1, den
    maximum)."""
    s0, s1 = prims.mul_packed_f32x2((x0, x1), (slope, slope))
    e0 = _ex2(s0)
    e1 = _ex2(s1)
    d0, d1 = prims.add_packed_f32x2((e0, e1), (one, one))
    # -(1 + e) exactly (RN is symmetric): the negated divisor operand.
    n0, n1 = prims.fma_packed_f32x2(
        (e0, e1), (minus_one, minus_one), (minus_one, minus_one)
    )
    r0 = _rcp_approx(d0)
    r1 = _rcp_approx(d1)
    t0, t1 = prims.fma_packed_f32x2((n0, n1), (r0, r1), (one, one))
    r0, r1 = prims.fma_packed_f32x2((r0, r1), (t0, t1), (r0, r1))
    q0, q1 = prims.fma_packed_f32x2((r0, r1), (floor, floor), (zero, zero))
    t0, t1 = prims.fma_packed_f32x2((n0, n1), (q0, q1), (floor, floor))
    q0, q1 = prims.fma_packed_f32x2((r0, r1), (t0, t1), (q0, q1))
    return _ex2(q0), _ex2(q1), cute.arch.fmax(d0, d1, nan=True)


@cute.jit
def _load_pair(
    q_in,
    k_in,
    g_in,
    q_bf16,
    k_bf16,
    gate_bf16,
    pair: cutlass.Constexpr[int],
    lane_offset,
    token_stride,
    valid,
):
    """Raw q, k, g of tokens 2 pair and 2 pair + 1 into stage pair % 2. Past
    the segment's end a token reloads the last valid one (a valid address;
    its row is not stored)."""
    PER_LANE = 4
    src_layout = cute.make_layout((PER_LANE,), stride=(1,))
    for u in cutlass.range_constexpr(2):
        step = cute.min(cutlass.Int32(2 * pair + u), valid - 1) * token_stride
        for tensor, source in ((q_bf16, q_in), (k_bf16, k_in), (gate_bf16, g_in)):
            loaded = cute.make_rmem_tensor(src_layout, cutlass.BFloat16)
            cute.autovec_copy(
                cute.make_tensor(
                    _known_aligned(source.iterator + lane_offset + step, 8), src_layout
                ),
                loaded,
            )
            for i in cutlass.range_constexpr(PER_LANE):
                tensor[pair % 2, u, i] = loaded[i]


@cute.jit
def _coeff_group(
    q_in,
    k_in,
    g_in,
    beta_in,
    alog_in,
    bias_in,
    coeff,
    scale,
    eps,
    lower_bound,
    cu_seqlens,
    prefix,
    tasks,
    start,
    thread,
    block,
    head_block,
):
    """Workspace rows of window tokens TOKENS_PER_WARP block + 0..7 (one
    segment) for one four-head group, one warp per head: all eight tokens'
    loads are issued first. Every row is formed with the same FP32
    operations, lane mapping and reduction pairings as before (same bits):
    the gate's quotient uses div.rn's fast-path instructions (exact fallback
    to the division when its range check could fail), and the q/k
    sum-of-squares butterfly runs as a reduce-scatter over token pairs (each
    sum pairs the same lanes in the same order) with the scale factors
    gathered back by index shuffles."""
    tidx = thread % 32
    n_heads = cutlass.Int32(q_in.shape[2])
    head_idx = head_block * 4 + thread // 32
    local = TOKENS_PER_WARP * block
    task_idx = local // SEGMENT + start
    token_in_segment = local % SEGMENT
    PER_LANE = 4
    cute.make_layout((128,), stride=(1,))
    src_layout = cute.make_layout((PER_LANE,), stride=(1,))
    safe_head = cute.min(head_idx, n_heads - 1)
    total_segments = prefix[cu_seqlens.shape[0] - 1]
    segment_first = tasks[task_idx, 1]
    segment_tokens = tasks[task_idx, 2]
    a_log = alog_in[safe_head].to(cutlass.Float32)
    dt_bias = cute.make_rmem_tensor(src_layout, cutlass.Float32)
    cute.autovec_copy(
        cute.make_tensor(
            _known_aligned(bias_in.iterator + safe_head * 128 + tidx * PER_LANE, 16),
            src_layout,
        ),
        dt_bias,
    )
    active = (
        (head_idx < n_heads)
        & (task_idx < total_segments)
        & (token_in_segment < segment_tokens)
    )
    if active:
        valid = cute.min(segment_tokens - token_in_segment, TOKENS_PER_WARP)
        token0 = segment_first + token_in_segment
        pair4 = cute.make_layout((2, 2, PER_LANE), stride=(2 * PER_LANE, PER_LANE, 1))
        # Two token pairs of raw rows in flight: pair p + 1 loads while pair p
        # is formed (stage p % 2).
        q_bf16 = cute.make_rmem_tensor(pair4, cutlass.BFloat16)
        k_bf16 = cute.make_rmem_tensor(pair4, cutlass.BFloat16)
        gate_bf16 = cute.make_rmem_tensor(pair4, cutlass.BFloat16)
        # This lane's four elements of token0's head row (64-bit once); the
        # other tokens are 32-bit offsets of whole token rows.
        lane_offset = (
            cutlass.Int64(token0) * n_heads + head_idx
        ) * 128 + tidx * PER_LANE
        token_stride = n_heads * 128
        _load_pair(
            q_in,
            k_in,
            g_in,
            q_bf16,
            k_bf16,
            gate_bf16,
            0,
            lane_offset,
            token_stride,
            valid,
        )
        # Lane t holds token t's beta (one row each, formed lane-parallel).
        beta_step = cute.min(tidx % TOKENS_PER_WARP, valid - 1) * n_heads
        raw_beta = cute.make_tensor(
            beta_in.iterator + (cutlass.Int64(token0) * n_heads + head_idx) + beta_step,
            cute.make_layout((1,)),
        )[0].to(cutlass.Float32)
        policy = prims.evict_last_policy()
        # Row of window token local (64-bit once) plus whole-row 32-bit steps.
        row_base = (cutlass.Int64(local) * n_heads + head_idx) * 392
        row_stride = n_heads * 392
        decay_rate = cute.exp(a_log, fastmath=True)
        log2e = cutlass.Float32(LOG2_E)
        slope = -decay_rate * log2e
        floor = lower_bound * log2e
        one = cutlass.Float32(1.0)
        zero = cutlass.Float32(0.0)
        minus_one = cutlass.Float32(-1.0)
        limit = cutlass.Float32(FAST_DIVISOR_LIMIT)
        slot4 = (tidx ^ ((tidx >> 3) & 1)) * 4
        upper = (tidx >> 4) & 1
        quarter = (tidx >> 3) & 1
        source8 = tidx & 7
        for pair in cutlass.range_constexpr(TOKENS_PER_WARP // 2):
            if cutlass.const_expr(pair + 1 < TOKENS_PER_WARP // 2):
                _load_pair(
                    q_in,
                    k_in,
                    g_in,
                    q_bf16,
                    k_bf16,
                    gate_bf16,
                    pair + 1,
                    lane_offset,
                    token_stride,
                    valid,
                )
            stage = pair % 2
            q_src = cute.make_rmem_tensor(
                cute.make_layout((2, PER_LANE), stride=(PER_LANE, 1)), cutlass.Float32
            )
            k_src = cute.make_rmem_tensor(
                cute.make_layout((2, PER_LANE), stride=(PER_LANE, 1)), cutlass.Float32
            )
            gate_src = cute.make_rmem_tensor(
                cute.make_layout((2, PER_LANE), stride=(PER_LANE, 1)), cutlass.Float32
            )
            sums = cute.make_rmem_tensor(
                cute.make_layout((4,), stride=(1,)), cutlass.Float32
            )
            for u in cutlass.range_constexpr(2):
                t = 2 * pair + u
                biased = cute.make_rmem_tensor(src_layout, cutlass.Float32)
                for i in cutlass.range_constexpr(PER_LANE):
                    q_src[u, i] = q_bf16[stage, u, i].to(cutlass.Float32)
                    k_src[u, i] = k_bf16[stage, u, i].to(cutlass.Float32)
                    biased[i] = gate_bf16[stage, u, i].to(cutlass.Float32) + dt_bias[i]
                gate_src[u, 0], gate_src[u, 1], high01 = _gate_pair(
                    biased[0], biased[1], slope, floor, one, zero, minus_one
                )
                gate_src[u, 2], gate_src[u, 3], high23 = _gate_pair(
                    biased[2], biased[3], slope, floor, one, zero, minus_one
                )
                # A divisor outside [1, 2^64) (or NaN) anywhere in the warp:
                # this token's gates through decay_gate's div.rn instead.
                high = cute.arch.fmax(high01, high23, nan=True)
                exact = cute.arch.vote_any_sync((high >= limit) | (high != high))
                if exact:
                    for i in cutlass.range_constexpr(PER_LANE):
                        gate_src[u, i] = decay_gate(biased[i], decay_rate, lower_bound)
                # Even/odd squares (product, then fused term), as before.
                q_sum_even, q_sum_odd = prims.mul_packed_f32x2(
                    (q_src[u, 0], q_src[u, 1]), (q_src[u, 0], q_src[u, 1])
                )
                q_sum_even, q_sum_odd = prims.fma_packed_f32x2(
                    (q_src[u, 2], q_src[u, 3]),
                    (q_src[u, 2], q_src[u, 3]),
                    (q_sum_even, q_sum_odd),
                )
                k_sum_even, k_sum_odd = prims.mul_packed_f32x2(
                    (k_src[u, 0], k_src[u, 1]), (k_src[u, 0], k_src[u, 1])
                )
                k_sum_even, k_sum_odd = prims.fma_packed_f32x2(
                    (k_src[u, 2], k_src[u, 3]),
                    (k_src[u, 2], k_src[u, 3]),
                    (k_sum_even, k_sum_odd),
                )
                sums[2 * u], sums[2 * u + 1] = prims.add_packed_f32x2(
                    (q_sum_even, k_sum_even), (q_sum_odd, k_sum_odd)
                )
            # Butterfly levels 16 and 8 as a reduce-scatter of the four sums
            # (q, k of both tokens), then levels 4, 2, 1 on one sum per lane.
            # Each lane adds the same two operands as the butterfly's lanes.
            keep0 = sums[0] if upper == 0 else sums[2]
            keep1 = sums[1] if upper == 0 else sums[3]
            send0 = sums[2] if upper == 0 else sums[0]
            send1 = sums[3] if upper == 0 else sums[1]
            got0 = cute.arch.shuffle_sync_bfly(send0, offset=16, mask=FULL_MASK)
            got1 = cute.arch.shuffle_sync_bfly(send1, offset=16, mask=FULL_MASK)
            level0, level1 = prims.add_packed_f32x2((keep0, keep1), (got0, got1))
            mine = level0 if quarter == 0 else level1
            other = level1 if quarter == 0 else level0
            total = mine + cute.arch.shuffle_sync_bfly(other, offset=8, mask=FULL_MASK)
            for offset in [4, 2, 1]:
                total = total + cute.arch.shuffle_sync_bfly(
                    total, offset=offset, mask=FULL_MASK
                )
            # Lane bits (4, 3): 0 q of token 0, 1 k of token 0, 2 q of token 1,
            # 3 k of token 1. q factors carry the scale (scale * rsqrt).
            factor = cute.rsqrt(total + eps, fastmath=True)
            if quarter == 0:
                factor = factor * scale
            factors = cute.make_rmem_tensor(
                cute.make_layout((4,), stride=(1,)), cutlass.Float32
            )
            for f in cutlass.range_constexpr(4):
                factors[f] = cute.arch.shuffle_sync(factor, offset=source8 | (8 * f))
            for u in cutlass.range_constexpr(2):
                t = 2 * pair + u
                q_norm = cute.make_rmem_tensor(src_layout, cutlass.Float32)
                k_norm = cute.make_rmem_tensor(src_layout, cutlass.Float32)
                for i in cutlass.range_constexpr(2):
                    q_norm[2 * i], q_norm[2 * i + 1] = prims.mul_packed_f32x2(
                        (q_src[u, 2 * i], q_src[u, 2 * i + 1]),
                        (factors[2 * u], factors[2 * u]),
                    )
                    k_norm[2 * i], k_norm[2 * i + 1] = prims.mul_packed_f32x2(
                        (k_src[u, 2 * i], k_src[u, 2 * i + 1]),
                        (factors[2 * u + 1], factors[2 * u + 1]),
                    )
                if t < valid:
                    # Swizzled natural layout: the float4 holding K = 8 v + 4 h +
                    # (0..3) sits in slot 2 v + (h ^ bit 2 of v), i.e. this lane's
                    # K = 4 tidx + i go to slot tidx ^ bit 3 of tidx in one 16-byte
                    # store (under an L2 evict_last policy: the scan reads it back).
                    address = cutlass.Int64(
                        (coeff.iterator + row_base + (t * row_stride + slot4)).toint()
                    )
                    prims.store4_policy(
                        address, (q_norm[0], q_norm[1], q_norm[2], q_norm[3]), policy
                    )
                    prims.store4_policy(
                        address + 512,
                        (k_norm[0], k_norm[1], k_norm[2], k_norm[3]),
                        policy,
                    )
                    prims.store4_policy(
                        address + 1024,
                        (
                            gate_src[u, 0],
                            gate_src[u, 1],
                            gate_src[u, 2],
                            gate_src[u, 3],
                        ),
                        policy,
                    )
        # beta plus three zero pad floats (the scan copies this 16-byte chunk
        # whole): lane t stores token t's.
        beta = cutlass.Float32(1.0) / (
            cutlass.Float32(1.0) + cute.exp(-raw_beta, fastmath=True)
        )
        if tidx < valid:
            prims.store4_policy(
                cutlass.Int64(
                    (coeff.iterator + row_base + (tidx * row_stride + 384)).toint()
                ),
                (beta, zero, zero, zero),
                policy,
            )


@cute.kernel
def _coeff_kernel(
    q_in: cute.Tensor,
    k_in: cute.Tensor,
    g_in: cute.Tensor,
    beta_in: cute.Tensor,
    alog_in: cute.Tensor,
    bias_in: cute.Tensor,
    coeff: cute.Tensor,
    scale: cutlass.Float32,
    eps: cutlass.Float32,
    lower_bound: cutlass.Float32,
    cu_seqlens: cute.Tensor,
    prefix: cute.Tensor,
    tasks: cute.Tensor,
    start: cutlass.Int32,
    count: cutlass.Int32,
    wait_first: cutlass.Int32,
    counters: cute.Tensor,
    window_index: cutlass.Int32,
    NATURAL: cutlass.Constexpr[bool],
):
    # Launched with PDL. The first window waits for its predecessor (the
    # schedule kernel); later windows follow the previous window's scan,
    # which triggers after its own wait, and write the other workspace
    # region, so they overlap that scan. The next scan (PDL secondary of this
    # kernel) waits for this kernel's completion before reading anything it
    # or the previous scan wrote.
    thread, _, _ = cute.arch.thread_idx()
    block, head_block, _ = cute.arch.block_idx()
    # Block (0, 0) always waits for the primary (the previous window's scan,
    # at its end below), so this kernel cannot complete before it; the next scan, whose
    # griddepcontrol.wait covers this kernel, therefore also reads after the previous
    # scan finished writing its continuation state. The other blocks wait
    # only when their region may still be read.
    # No explicit trigger: this window's scan (PDL secondary) launches as
    # the last block exits, so its CTAs are placed on SMs the previous scan
    # has left (an early trigger crowded unequal-length layouts).
    # Overlapped windows (wait_first 2, or 3 when this region overlaps the
    # previous window's): this kernel launched after every CTA of the
    # previous scan passed its flag wait, so the scan that last read this
    # region has finished reading it. The next scan (PDL secondary) waits
    # for this kernel's block counters (below) instead of its grid
    # completion, which the hardware orders after the previous scan's.
    if wait_first == 1 or wait_first == 3:
        cute.arch.griddepcontrol_wait()
    _coeff_group(
        q_in,
        k_in,
        g_in,
        beta_in,
        alog_in,
        bias_in,
        coeff,
        scale,
        eps,
        lower_bound,
        cu_seqlens,
        prefix,
        tasks,
        start,
        thread,
        block,
        head_block,
    )
    if wait_first == 0 and block == 0 and head_block == 0:
        # Block (0, 0) writes its rows (the region the previous scan does not
        # read) first and only then waits for that scan, so the kernel still
        # cannot complete before it, without a post-scan work tail.
        cute.arch.griddepcontrol_wait()
    if wait_first >= 2:
        # Publish this block's rows: after the block barrier, one release
        # reduction (no returned value) on one of the window's 32 block
        # counters (spread to avoid one contended address).
        cute.arch.sync_threads()
        if thread == 0:
            linear = head_block * (count * TOKEN_BLOCKS) + block
            cute.arch.red(
                counters.iterator + window_index * COUNTERS + linear % COUNTERS,
                cutlass.Int32(1),
                op="add",
                dtype="s32",
                sem="release",
                scope="gpu",
            )
        # Trigger at the block's end: the next scan launches once every
        # block has written its rows (not at grid completion), and its CTAs
        # do not occupy SMs while this kernel still runs.
        cute.arch.griddepcontrol_launch_dependents()


@cute.jit
def _known_aligned(pointer, alignment: cutlass.Constexpr[int]):
    return cute.make_ptr(
        pointer.dtype, pointer.toint(), pointer.memspace, assumed_align=alignment
    )


@cute.jit
def _launch_coeff(
    q: cute.Tensor,
    k: cute.Tensor,
    g: cute.Tensor,
    beta: cute.Tensor,
    a: cute.Tensor,
    bias: cute.Tensor,
    coeff: cute.Tensor,
    scale: cutlass.Float32,
    eps: cutlass.Float32,
    lower_bound: cutlass.Float32,
    cu_seqlens: cute.Tensor,
    prefix: cute.Tensor,
    tasks: cute.Tensor,
    start: cutlass.Int32,
    count: cutlass.Int32,
    wait_first: cutlass.Int32,
    counters: cute.Tensor,
    window_index: cutlass.Int32,
    stream: cuda.CUstream,
    PDL: cutlass.Constexpr[bool],
    NATURAL: cutlass.Constexpr[bool],
):
    _coeff_kernel(
        q,
        k,
        g,
        beta,
        a,
        bias,
        coeff,
        scale,
        eps,
        lower_bound,
        cu_seqlens,
        prefix,
        tasks,
        start,
        count,
        wait_first,
        counters,
        window_index,
        NATURAL,
    ).launch(
        grid=[count * TOKEN_BLOCKS, cute.ceil_div(q.shape[2], 4), 1],
        block=[128, 1, 1],
        stream=stream,
        use_pdl=PDL,
    )


def _fake(shape, dtype=cute.BFloat16):
    return cute.runtime.make_fake_compact_tensor(
        dtype, shape, assumed_align=32, stride_order=tuple(reversed(range(len(shape))))
    )


@functools.cache
def compiled_coeff_kernel(device_index, capability):
    pdl, natural = True, True
    b, length, h, n, m, r, w = (cute.sym_int() for _ in range(7))
    x = _fake((b, length, h, 128))
    return cute.compile(
        _launch_coeff,
        x,
        x,
        x,
        _fake((b, length, h)),
        _fake((h,), cute.Float32),
        _fake((h * 128,), cute.Float32),
        _fake((n,), cute.Float32),
        cutlass.Float32(0.0),
        cutlass.Float32(0.0),
        cutlass.Float32(0.0),
        _fake((m,), cute.Int32),
        _fake((m,), cute.Int32),
        _fake((r, 3), cute.Int32),
        cutlass.Int32(0),
        cutlass.Int32(0),
        cutlass.Int32(0),
        _fake((w,), cute.Int32),
        cutlass.Int32(0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        pdl,
        natural,
        options="--enable-tvm-ffi",
    )
