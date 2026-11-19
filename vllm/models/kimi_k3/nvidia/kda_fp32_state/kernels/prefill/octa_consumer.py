"""Eight-row recursive-halving consumer.

Each lane owns eight V rows of one K group (eight K values). Prediction and
output use a radix-8 reduce-scatter over K-group levels 8, 4 and 2 (each lane
ends with its own row's level-2 sum), one level-1 exchange, and a delta
all-gather. Every FP32 operation and the 8/4/2/1 pairing match the four-row
consumer, so the bits are identical.
"""

import cutlass
import cutlass.cute as cute
from ... import _primitives as prims
from ..._primitives import _known_aligned
from .consumer import (
    wait_test,
    finish_wait,
    _gmem_ptr,
    _load_natural8,
    PIPE_STAGES,
    VALUE_FLOAT_OFFSET,
    FULL_MASK,
    STEADY_TOKENS,
)

ROWS = 8


@cute.jit
def _octa_tree(partials):
    """partials[s]: dual-chain sum of slot s (row s ^ x) for this K lane.
    Returns the 8/4/2/1-tree sum of row x (x = K-lane bits 3..1)."""
    # K lane k ^ m (m even) holds row x in slot m >> 1; lane offset 2 m.
    v2 = cute.arch.shuffle_sync_bfly(partials[1], offset=4, mask=FULL_MASK)
    v4 = cute.arch.shuffle_sync_bfly(partials[2], offset=8, mask=FULL_MASK)
    v6 = cute.arch.shuffle_sync_bfly(partials[3], offset=12, mask=FULL_MASK)
    v8 = cute.arch.shuffle_sync_bfly(partials[4], offset=16, mask=FULL_MASK)
    v10 = cute.arch.shuffle_sync_bfly(partials[5], offset=20, mask=FULL_MASK)
    v12 = cute.arch.shuffle_sync_bfly(partials[6], offset=24, mask=FULL_MASK)
    v14 = cute.arch.shuffle_sync_bfly(partials[7], offset=28, mask=FULL_MASK)
    # Level 8: (k, k^8), (k^2, k^10), (k^4, k^12), (k^6, k^14).
    a0, a2 = prims.add_packed_f32x2((partials[0], v2), (v8, v10))
    a4, a6 = prims.add_packed_f32x2((v4, v6), (v12, v14))
    # Level 4: (k, k^4), (k^2, k^6).
    b0, b2 = prims.add_packed_f32x2((a0, a2), (a4, a6))
    # Level 2, then level 1 with K lane k ^ 1 (same row).
    c = b0 + b2
    return c + cute.arch.shuffle_sync_bfly(c, offset=2, mask=FULL_MASK)


@cute.jit
def _octa_token(
    parity: cutlass.Constexpr[int],
    staged,
    staged_values,
    full_bars,
    empty_bars,
    k_lane,
    value_index,
    h,
    gates,
    keys,
    q_reg,
    output_address,
    memspace: cutlass.Constexpr,
    store_lane,
    slot,
    next_slot,
    next_phase,
):
    other = 1 - parity
    if cutlass.const_expr(parity == 1):
        ready = cute.arch.mbarrier_test_wait(full_bars + (next_slot >> 1), next_phase)
    beta = staged[slot, 384]
    v_own = staged_values[slot, 2 * VALUE_FLOAT_OFFSET + value_index]
    for i in [2, 3, 0, 1, 4, 5, 6, 7]:
        for jp in cutlass.range_constexpr(ROWS // 2):
            h[2 * jp, i], h[2 * jp + 1, i] = prims.mul_packed_f32x2(
                (h[2 * jp, i], h[2 * jp + 1, i]), (gates[parity, i], gates[parity, i])
            )
    accum = cute.make_rmem_tensor(
        cute.make_layout((ROWS, 2), stride=(2, 1)), cutlass.Float32
    )
    for c in cutlass.range_constexpr(2):
        for jp in cutlass.range_constexpr(ROWS // 2):
            accum[2 * jp, c], accum[2 * jp + 1, c] = prims.mul_packed_f32x2(
                (h[2 * jp, 2 + c], h[2 * jp + 1, 2 + c]),
                (keys[parity, 2 + c], keys[parity, 2 + c]),
            )
    for i in [0, 4, 6]:
        for c in cutlass.range_constexpr(2):
            for jp in cutlass.range_constexpr(ROWS // 2):
                accum[2 * jp, c], accum[2 * jp + 1, c] = prims.fma_packed_f32x2(
                    (h[2 * jp, i + c], h[2 * jp + 1, i + c]),
                    (keys[parity, i + c], keys[parity, i + c]),
                    (accum[2 * jp, c], accum[2 * jp + 1, c]),
                )
    partials = cute.make_rmem_tensor(
        cute.make_layout((ROWS,), stride=(1,)), cutlass.Float32
    )
    for jp in cutlass.range_constexpr(ROWS // 2):
        partials[2 * jp], partials[2 * jp + 1] = prims.add_packed_f32x2(
            (accum[2 * jp, 0], accum[2 * jp + 1, 0]),
            (accum[2 * jp, 1], accum[2 * jp + 1, 1]),
        )
    prediction = _octa_tree(partials)
    delta_own = (v_own.to(cutlass.Float32) - prediction) * beta
    deltas = cute.make_rmem_tensor(
        cute.make_layout((ROWS,), stride=(1,)), cutlass.Float32
    )
    deltas[0] = delta_own
    # Slot s holds row s ^ x, owned by K lane k ^ 2s (lane offset 4 s).
    for s in cutlass.range_constexpr(1, ROWS):
        deltas[s] = cute.arch.shuffle_sync_bfly(delta_own, offset=4 * s, mask=FULL_MASK)
    _load_natural8(staged, slot, 0, k_lane, q_reg, 0)
    for i in cutlass.range_constexpr(8):
        for jp in cutlass.range_constexpr(ROWS // 2):
            h[2 * jp, i], h[2 * jp + 1, i] = prims.fma_packed_f32x2(
                (keys[parity, i], keys[parity, i]),
                (deltas[2 * jp], deltas[2 * jp + 1]),
                (h[2 * jp, i], h[2 * jp + 1, i]),
            )
    if cutlass.const_expr(parity == 1):
        finish_wait(full_bars + (next_slot >> 1), next_phase, ready)
    _load_natural8(staged, next_slot, 2, k_lane, gates, other)
    _load_natural8(staged, next_slot, 1, k_lane, keys, other)
    for c in cutlass.range_constexpr(2):
        for jp in cutlass.range_constexpr(ROWS // 2):
            accum[2 * jp, c], accum[2 * jp + 1, c] = prims.mul_packed_f32x2(
                (h[2 * jp, c], h[2 * jp + 1, c]), (q_reg[0, c], q_reg[0, c])
            )
    for p in cutlass.range_constexpr(1, 4):
        for c in cutlass.range_constexpr(2):
            for jp in cutlass.range_constexpr(ROWS // 2):
                accum[2 * jp, c], accum[2 * jp + 1, c] = prims.fma_packed_f32x2(
                    (h[2 * jp, 2 * p + c], h[2 * jp + 1, 2 * p + c]),
                    (q_reg[0, 2 * p + c], q_reg[0, 2 * p + c]),
                    (accum[2 * jp, c], accum[2 * jp + 1, c]),
                )
    for jp in cutlass.range_constexpr(ROWS // 2):
        partials[2 * jp], partials[2 * jp + 1] = prims.add_packed_f32x2(
            (accum[2 * jp, 0], accum[2 * jp + 1, 0]),
            (accum[2 * jp, 1], accum[2 * jp + 1, 1]),
        )
    if cutlass.const_expr(parity == 1):
        cute.arch.mbarrier_arrive(empty_bars + (slot >> 1))
    output = _octa_tree(partials)
    if store_lane:
        target = cute.make_tensor(
            _gmem_ptr(output_address, cutlass.BFloat16, memspace, 2),
            cute.make_layout((1,)),
        )
        target[0] = output.to(cutlass.BFloat16)


@cute.jit
def run_octa_consumer(
    lane,
    warp,
    cta_v_offset,
    bh,
    start,
    count,
    state_src,
    state_dst,
    staged,
    staged_values,
    full_bars,
    empty_bars,
    output_row0,
    head_stride,
    memspace: cutlass.Constexpr,
):
    k_lane = lane >> 1
    v_lane = lane & 1
    x = (k_lane >> 1) & 7
    block_row0 = cta_v_offset + warp * (2 * ROWS) + v_lane * ROWS
    value_index = warp * (2 * ROWS) + v_lane * ROWS + x
    store_lane = (k_lane & 1) == 0
    state_offset = cutlass.Int64(bh) * 128 * 128
    state_layout = cute.make_layout((128, 128), stride=(128, 1))
    h_read = cute.make_tensor(
        _known_aligned(state_src.iterator + state_offset, 32), state_layout
    )
    if start > 0:
        h_read = cute.make_tensor(
            _known_aligned(state_dst.iterator + state_offset, 32), state_layout
        )
    h = cute.make_rmem_tensor(
        cute.make_layout((ROWS, 8), stride=(8, 1)), cutlass.Float32
    )
    h_io = cute.make_rmem_tensor(cute.make_layout((8,), stride=(1,)), cutlass.Float32)
    for s in cutlass.range_constexpr(ROWS):
        h_tile = cute.local_tile(h_read, (1, 8), (block_row0 + (s ^ x), k_lane))
        cute.autovec_copy(h_tile, h_io)
        for i in cutlass.range_constexpr(8):
            h[s, i] = h_io[i]
    pair8 = cute.make_layout((2, 8), stride=(8, 1))
    gates = cute.make_rmem_tensor(pair8, cutlass.Float32)
    keys = cute.make_rmem_tensor(pair8, cutlass.Float32)
    q_reg = cute.make_rmem_tensor(
        cute.make_layout((1, 8), stride=(8, 1)), cutlass.Float32
    )
    output_address = (
        output_row0 + start * head_stride + cutlass.Int64((block_row0 + x) * 2)
    )
    wait_test(full_bars + 0, 0)
    _load_natural8(staged, 0, 2, k_lane, gates, 0)
    _load_natural8(staged, 0, 1, k_lane, keys, 0)
    groups = count // STEADY_TOKENS
    for group_index in cutlass.range(groups):
        base = (group_index % (PIPE_STAGES // STEADY_TOKENS)) * STEADY_TOKENS
        phase = (group_index // (PIPE_STAGES // STEADY_TOKENS)) & 1
        for i in cutlass.range_constexpr(STEADY_TOKENS):
            if cutlass.const_expr(i + 1 < STEADY_TOKENS):
                next_slot, next_phase = base + i + 1, phase
            else:
                next_slot = (base + STEADY_TOKENS) & (PIPE_STAGES - 1)
                next_phase = ((group_index + 1) // (PIPE_STAGES // STEADY_TOKENS)) & 1
            _octa_token(
                i % 2,
                staged,
                staged_values,
                full_bars,
                empty_bars,
                k_lane,
                value_index,
                h,
                gates,
                keys,
                q_reg,
                output_address,
                memspace,
                store_lane,
                base + i,
                next_slot,
                next_phase,
            )
            output_address = output_address + head_stride
    tail_start = STEADY_TOKENS * groups
    tail_count = count - tail_start
    for pair in cutlass.range(tail_count // 2):
        for i in cutlass.range_constexpr(2):
            t = tail_start + 2 * pair + i
            _octa_token(
                i,
                staged,
                staged_values,
                full_bars,
                empty_bars,
                k_lane,
                value_index,
                h,
                gates,
                keys,
                q_reg,
                output_address,
                memspace,
                store_lane,
                t & (PIPE_STAGES - 1),
                (t + 1) & (PIPE_STAGES - 1),
                ((t + 1) // PIPE_STAGES) & 1,
            )
            output_address = output_address + head_stride
    if (tail_count & 1) == 1:
        t = count - 1
        _octa_token(
            0,
            staged,
            staged_values,
            full_bars,
            empty_bars,
            k_lane,
            value_index,
            h,
            gates,
            keys,
            q_reg,
            output_address,
            memspace,
            store_lane,
            t & (PIPE_STAGES - 1),
            (t + 1) & (PIPE_STAGES - 1),
            ((t + 1) // PIPE_STAGES) & 1,
        )
    h_out = cute.make_tensor(
        _known_aligned(state_dst.iterator + state_offset, 32), state_layout
    )
    words = cute.make_rmem_tensor(cute.make_layout((8,), stride=(1,)), cutlass.Float32)
    for s in cutlass.range_constexpr(ROWS):
        for i in cutlass.range_constexpr(8):
            words[i] = h[s, i]
        tile = cute.make_tensor(
            _known_aligned(
                h_out.iterator + (block_row0 + (s ^ x)) * 128 + k_lane * 8, 16
            ),
            cute.make_layout((8,), stride=(1,)),
        )
        cute.autovec_copy(words, tile)
