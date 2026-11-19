"""Two-K-lane consumer: eight lanes per V row, all-reduce trees.

Lane (r, b) of a warp owns V row 4 warp + r of the CTA's sixteen rows and the
two K lanes kA = (b & 3) + 4 (b >> 2) and kB = kA + 8 (16 K values). The
fixed tree over the sixteen K-lane partials,
    T_a = (P_a + P_{a+8}) + (P_{a+4} + P_{a+12}),
    sum = (T_0 + T_2) + (T_1 + T_3),
is formed as: lane-local P_kA + P_kB, one exchange with lane b ^ 4 (the other
half of T_a), then one radix-4 round over b ^ 1, b ^ 2, b ^ 3. Every lane ends
with the full sum (same operands, commutative pairings), so the delta needs no
broadcast. Gate, prediction, delta, update and output use the same FP32
operations, chain orders and rounding as the other engines; only the lane
placement differs.

The output dot of token t - 1 is issued after token t's prediction dot, so it
fills that token's shuffle latency instead of sitting between update and gate.
"""

import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm
from ... import _primitives as prims
from ..._primitives import _known_aligned
from ..state_io import _store_bf16_if
from .consumer import (
    wait_test,
    finish_wait,
    _load_natural8,
    PIPE_STAGES,
    VALUE_FLOAT_OFFSET,
    FULL_MASK,
)


PAIR_STEADY = 16  # tokens per straight-line steady group


@dsl_user_op
def _arrive_if(mbar, pred, *, loc=None, ip=None):
    """mbarrier arrive (one count per thread), predicated on pred != 0."""
    mbar_i32 = llvm.ptrtoint(T.i32(), mbar.to_llvm_ptr(loc=loc, ip=ip), loc=loc, ip=ip)
    llvm.inline_asm(
        None,
        [mbar_i32, pred.ir_value(loc=loc, ip=ip)],
        "{ .reg .pred p; setp.ne.s32 p, $1, 0; "
        "@p mbarrier.arrive.shared::cta.b64 _, [$0]; }",
        "r,r,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@cute.jit
def _pair_tree(u):
    """u = P_kA + P_kB of this lane. Level 8/4 half from lane b ^ 4, then the
    level 2/1 radix-4 round; every lane returns the full fixed-tree sum."""
    total = u + cute.arch.shuffle_sync_bfly(u, offset=16, mask=FULL_MASK)
    near = cute.arch.shuffle_sync_bfly(total, offset=8, mask=FULL_MASK)
    far = cute.arch.shuffle_sync_bfly(total, offset=4, mask=FULL_MASK)
    both = cute.arch.shuffle_sync_bfly(total, offset=12, mask=FULL_MASK)
    return (total + near) + (far + both)


@cute.jit
def _pair_tree8(u):
    """The same fixed-tree sum as _pair_tree in one exchange round: u of the
    seven other lanes b ^ 1 .. b ^ 7 of the row, then T_j = u_j + u_{j ^ 4}
    for j = b, b ^ 1, b ^ 2, b ^ 3 and (T_b + T_{b^2}) + (T_{b^1} + T_{b^3})
    locally (the operands each lane of _pair_tree pairs, in the same tree)."""
    u1 = cute.arch.shuffle_sync_bfly(u, offset=4, mask=FULL_MASK)
    u2 = cute.arch.shuffle_sync_bfly(u, offset=8, mask=FULL_MASK)
    u3 = cute.arch.shuffle_sync_bfly(u, offset=12, mask=FULL_MASK)
    u4 = cute.arch.shuffle_sync_bfly(u, offset=16, mask=FULL_MASK)
    u5 = cute.arch.shuffle_sync_bfly(u, offset=20, mask=FULL_MASK)
    u6 = cute.arch.shuffle_sync_bfly(u, offset=24, mask=FULL_MASK)
    u7 = cute.arch.shuffle_sync_bfly(u, offset=28, mask=FULL_MASK)
    t0, t1 = prims.add_packed_f32x2((u, u1), (u4, u5))
    t2, t3 = prims.add_packed_f32x2((u2, u3), (u6, u7))
    low, high = prims.add_packed_f32x2((t0, t1), (t2, t3))
    return low + high


@cute.jit
def _output_partial(h, q_reg):
    """P_kA + P_kB of the output dot: per K lane the even chain 0 then 2, 4,
    6 and the odd chain 1, 3, 5, 7 run as the two halves of one packed chain
    (element pairs 2m, 2m + 1 are adjacent registers)."""
    sums = []
    for s in cutlass.range_constexpr(2):
        chain = prims.mul_packed_f32x2((h[s, 0], h[s, 1]), (q_reg[s, 0], q_reg[s, 1]))
        for m in cutlass.range_constexpr(1, 4):
            chain = prims.fma_packed_f32x2(
                (h[s, 2 * m], h[s, 2 * m + 1]),
                (q_reg[s, 2 * m], q_reg[s, 2 * m + 1]),
                chain,
            )
        sums.append(chain[0] + chain[1])
    return sums[0] + sums[1]


@cute.jit
def _pair_step(
    slot,
    next_slot,
    prev_slot,
    arrive_ok,
    store_ok,
    ARRIVE: cutlass.Constexpr[bool],
    staged,
    staged_values,
    empty_bars,
    k_a,
    k_b,
    value_index,
    h,
    gates,
    keys,
    q_reg,
    output_address,
):
    """Token t's recurrence and token t - 1's output.

    On entry h holds the state after token t - 1, gates/keys token t and
    q_reg token t - 1. On exit h holds the state after token t, gates/keys
    token t + 1 and q_reg token t. The caller has waited for slot next_slot.
    """
    beta = staged[slot, 384]
    v_own = staged_values[slot, 2 * VALUE_FLOAT_OFFSET + value_index].to(
        cutlass.Float32
    )
    pair8 = cute.make_layout((2, 8), stride=(8, 1))
    gated = cute.make_rmem_tensor(pair8, cutlass.Float32)
    for m in [1, 0, 2, 3]:
        for s in cutlass.range_constexpr(2):
            gated[s, 2 * m], gated[s, 2 * m + 1] = prims.mul_packed_f32x2(
                (h[s, 2 * m], h[s, 2 * m + 1]), (gates[s, 2 * m], gates[s, 2 * m + 1])
            )
    # Prediction per K lane: even chain term 2 rounds first, then 0, 4, 6
    # fused; odd chain 3, 1, 5, 7 (the two halves of one packed chain).
    sums = []
    for s in cutlass.range_constexpr(2):
        chain = prims.mul_packed_f32x2(
            (gated[s, 2], gated[s, 3]), (keys[s, 2], keys[s, 3])
        )
        for m in [0, 2, 3]:
            chain = prims.fma_packed_f32x2(
                (gated[s, 2 * m], gated[s, 2 * m + 1]),
                (keys[s, 2 * m], keys[s, 2 * m + 1]),
                chain,
            )
        sums.append(chain[0] + chain[1])
    # Prediction (on the recurrence): one exchange round.
    prediction = _pair_tree8(sums[0] + sums[1])
    # Output dot of token t - 1 on its updated state (independent of the
    # prediction tree above, so it fills the shuffle latency).
    output_partial = _output_partial(h, q_reg)
    if cutlass.const_expr(ARRIVE):
        # Token t - 1 closed its slot pair; its q values are consumed above.
        _arrive_if(empty_bars + (prev_slot >> 1), arrive_ok)
    # Next token's gates and keys (slot already waited for), this token's q.
    next_keys = cute.make_rmem_tensor(pair8, cutlass.Float32)
    _load_natural8(staged, next_slot, 2, k_a, gates, 0)
    _load_natural8(staged, next_slot, 2, k_b, gates, 1)
    _load_natural8(staged, next_slot, 1, k_a, next_keys, 0)
    _load_natural8(staged, next_slot, 1, k_b, next_keys, 1)
    delta = (v_own - prediction) * beta
    for m in [1, 0, 2, 3]:
        for s in cutlass.range_constexpr(2):
            h[s, 2 * m], h[s, 2 * m + 1] = prims.fma_packed_f32x2(
                (keys[s, 2 * m], keys[s, 2 * m + 1]),
                (delta, delta),
                (gated[s, 2 * m], gated[s, 2 * m + 1]),
            )
    output = _pair_tree(output_partial)
    _store_bf16_if(output_address, output, store_ok)
    _load_natural8(staged, slot, 0, k_a, q_reg, 0)
    _load_natural8(staged, slot, 0, k_b, q_reg, 1)
    for i in cutlass.range_constexpr(8):
        keys[0, i] = next_keys[0, i]
        keys[1, i] = next_keys[1, i]


@cute.jit
def run_pair_consumer(
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
    # lane = 4 b + r: the eight lanes of one quarter-warp read only two
    # coefficient chunks (b = 2 q, 2 q + 1), and the four quarter-warps' chunks
    # sit in disjoint banks, so one 16-byte coefficient load is one wavefront.
    b = lane >> 2
    is_store = 1 - cute.min(b, 1)
    k_a = (b & 3) + 4 * (b >> 2)
    k_b = k_a + 8
    local_row = warp * 4 + (lane & 3)
    row = cta_v_offset + local_row
    value_index = local_row
    state_offset = cutlass.Int64(bh) * 128 * 128
    state_layout = cute.make_layout((128, 128), stride=(128, 1))
    h_read = cute.make_tensor(
        _known_aligned(state_src.iterator + state_offset, 32), state_layout
    )
    if start > 0:
        h_read = cute.make_tensor(
            _known_aligned(state_dst.iterator + state_offset, 32), state_layout
        )
    pair8 = cute.make_layout((2, 8), stride=(8, 1))
    h = cute.make_rmem_tensor(pair8, cutlass.Float32)
    h_io = cute.make_rmem_tensor(cute.make_layout((8,), stride=(1,)), cutlass.Float32)
    cute.autovec_copy(cute.local_tile(h_read, (1, 8), (row, k_a)), h_io)
    for i in cutlass.range_constexpr(8):
        h[0, i] = h_io[i]
    cute.autovec_copy(cute.local_tile(h_read, (1, 8), (row, k_b)), h_io)
    for i in cutlass.range_constexpr(8):
        h[1, i] = h_io[i]
    gates = cute.make_rmem_tensor(pair8, cutlass.Float32)
    keys = cute.make_rmem_tensor(pair8, cutlass.Float32)
    q_reg = cute.make_rmem_tensor(pair8, cutlass.Float32)
    q_reg.fill(0.0)
    # Address of token t - 1's output for this row (token -1 is never stored).
    output_address = (
        output_row0 + cutlass.Int64(start - 1) * head_stride + cutlass.Int64(row * 2)
    )
    one = cutlass.Int32(1)
    wait_test(full_bars + 0, 0)
    _load_natural8(staged, 0, 2, k_a, gates, 0)
    _load_natural8(staged, 0, 2, k_b, gates, 1)
    _load_natural8(staged, 0, 1, k_a, keys, 0)
    _load_natural8(staged, 0, 1, k_b, keys, 1)
    groups = count // PAIR_STEADY
    for group_index in cutlass.range(groups):
        # PAIR_STEADY tokens per group: slots base + i, every shared offset
        # static (one straight-line schedule per group).
        base = (group_index % (PIPE_STAGES // PAIR_STEADY)) * PAIR_STEADY
        phase = (group_index // (PIPE_STAGES // PAIR_STEADY)) & 1
        next_phase = ((group_index + 1) // (PIPE_STAGES // PAIR_STEADY)) & 1
        # Every pair the group looks ahead into (tokens g + 2 .. g + steady + 1;
        # the token after the group exists or is the producer's placeholder)
        # up front: independent probes first, then spin only where not complete.
        ahead = full_bars + (((base + PAIR_STEADY) & (PIPE_STAGES - 1)) >> 1)
        ready = cute.make_rmem_tensor(
            cute.make_layout((PAIR_STEADY // 2,), stride=(1,)), cutlass.Boolean
        )
        for j in cutlass.range_constexpr(1, PAIR_STEADY // 2):
            ready[j] = cute.arch.mbarrier_test_wait(
                full_bars + ((base >> 1) + j), phase
            )
        ready_ahead = cute.arch.mbarrier_test_wait(ahead, next_phase)
        for j in cutlass.range_constexpr(1, PAIR_STEADY // 2):
            finish_wait(full_bars + ((base >> 1) + j), phase, ready[j])
        finish_wait(ahead, next_phase, ready_ahead)
        first = cute.min(group_index, 1)
        for i in cutlass.range_constexpr(PAIR_STEADY):
            slot = base + i
            next_slot = base + i + 1
            if cutlass.const_expr(i + 1 == PAIR_STEADY):
                next_slot = (base + PAIR_STEADY) & (PIPE_STAGES - 1)
            prev_slot = (base + i + PIPE_STAGES - 1) & (PIPE_STAGES - 1)
            ok = one
            if cutlass.const_expr(i == 0):
                ok = first
            _pair_step(
                slot,
                next_slot,
                prev_slot,
                ok,
                ok * is_store,
                i % 2 == 0,
                staged,
                staged_values,
                empty_bars,
                k_a,
                k_b,
                value_index,
                h,
                gates,
                keys,
                q_reg,
                output_address,
            )
            output_address = output_address + head_stride
    for t in cutlass.range(PAIR_STEADY * groups, count):
        slot = t & (PIPE_STAGES - 1)
        next_slot = (t + 1) & (PIPE_STAGES - 1)
        prev_slot = (t + PIPE_STAGES - 1) & (PIPE_STAGES - 1)
        if (t & 1) == 1:
            # Token t + 1 opens a new pair (real or the placeholder).
            wait_test(full_bars + (next_slot >> 1), ((t + 1) // PIPE_STAGES) & 1)
        ok = cute.min(t, 1)
        arrive = ok * (1 - (t & 1))
        _pair_step(
            slot,
            next_slot,
            prev_slot,
            arrive,
            ok * is_store,
            True,
            staged,
            staged_values,
            empty_bars,
            k_a,
            k_b,
            value_index,
            h,
            gates,
            keys,
            q_reg,
            output_address,
        )
        output_address = output_address + head_stride
    # Output of the last token.
    output = _pair_tree(_output_partial(h, q_reg))
    _store_bf16_if(output_address, output, is_store)
    h_out = cute.make_tensor(
        _known_aligned(state_dst.iterator + state_offset, 32), state_layout
    )
    for s in cutlass.range_constexpr(2):
        for i in cutlass.range_constexpr(8):
            h_io[i] = h[s, i]
        k_lane = k_a if s == 0 else k_b
        tile = cute.make_tensor(
            _known_aligned(h_out.iterator + row * 128 + k_lane * 8, 16),
            cute.make_layout((8,), stride=(1,)),
        )
        cute.autovec_copy(h_io, tile)
