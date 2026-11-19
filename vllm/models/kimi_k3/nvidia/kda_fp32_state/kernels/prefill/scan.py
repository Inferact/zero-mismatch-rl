"""Warp-specialized recurrent scan over staged FP32 coefficients.

One producer warp stages coefficient rows and V values into a 32-slot
shared ring with TMA bulk copies and mbarriers; four consumer warps keep
their V rows of the FP32 state in registers and run one of several
register engines, chosen at runtime per CTA. All engines evaluate the same
FP32 operations in the same order.
"""

import functools
import cutlass
from ..._primitives import _known_aligned
import cutlass.cute as cute
import cuda.bindings.driver as cuda
from cutlass.utils import SmemAllocator
from ... import _primitives as prims
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm

from .schedule import SEGMENT, segment_range
from .coefficients import COUNTERS, TOKEN_BLOCKS
from .halving_consumer import run_halving_consumer
from .octa_consumer import run_octa_consumer
from .halving2_consumer import run_halving2_consumer
from .pair_consumer import run_pair_consumer
from .kpack_consumer import run_kpack_consumer
from .consumer import (
    PIPE_STAGES,
    PAIRS,
    CONSUMER_WARPS,
    STAGE_FLOATS,
    VALUE_FLOAT_OFFSET,
)

MIN_BLOCKS = 3

PAIR_ENGINE = 128  # radix4 bit: sixteen-row CTAs run the eight-lane pair engine

LONG_STEADY = 512  # radix4 bit: four-row CTAs run sixteen-token steady groups

LONG_LAST = (
    4096  # radix4 bit: the two-row (long-pair) CTAs follow the runtime_rows CTAs
)

OVERLAP = 8192  # radix4 bit: windows chain per CTA through flags, not grid completion

TWO_AHEAD = (
    16384  # radix4 bit: two-row CTAs probe each pair's full barrier one pair ahead
)

KPACK = 32768  # radix4 bit: four-row CTAs run the K-packed halving engine (same bits)

COEFF_BYTES = 388 * 4  # q, k, gates, beta (+3 pad) of one coefficient row

SMEM_BYTES = (
    53 * 1024
)  # dynamic shared memory of a scan CTA (ring + barriers; three per SM fit the 164 KB carveout)

SMEM_TWO_PER_SM = 80 * 1024  # requested to keep at most two scan CTAs per SM


@dsl_user_op
def _nanosleep(ns, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [cutlass.Int32(ns).ir_value(loc=loc, ip=ip)],
        "nanosleep.u32 $0;",
        "r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _special_register(name, *, loc=None, ip=None):
    """Read a special register at this point (side effects: not merged with
    earlier reads, so no value stays live across the consumer engines)."""
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [],
            f"mov.u32 $0, %{name};",
            "=r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def _produce(
    staged,
    full_bars,
    empty_bars,
    lane,
    count,
    coeff_row0,
    coeff_stride,
    value_address0,
    head_stride,
    ranks,
    rank_offset,
    first_segment,
    first_slot,
    head,
    n_heads,
    memspace: cutlass.Constexpr,
    value_rows,
    ring_shift,
):
    """Producer warp (one elected lane): stage tokens 0..count, two per pair,
    with TMA bulk copies whose bytes complete the pair's full barrier.

    Token `count` is a placeholder (no copies) so the consumers' look-ahead
    after their last token never waits on data that does not exist.
    """
    value_bytes = cutlass.Int32(value_rows * 2)
    value_address = value_address0
    if lane >= 0:
        # The whole warp runs the loop (converged); one elected lane issues.
        # Outer loop over the window's 128-token segments: the segment's
        # workspace slot (one rank lookup) is resolved once per segment, so
        # the per-pair body carries no predicated lookup code. SEGMENT is
        # even, so pairs never straddle segments.
        for segment in cutlass.range((count + SEGMENT - 1) // SEGMENT):
            slot = ranks[rank_offset + first_segment + segment] - first_slot
            coeff_address = coeff_row0 + (
                cutlass.Int64(slot) * SEGMENT * n_heads + head
            ) * (392 * 4)
            segment_tokens = cute.min(count - segment * SEGMENT, SEGMENT)
            for local_pair in cutlass.range((segment_tokens + 1) // 2):
                k = segment * (SEGMENT // 2) + local_pair
                # The ring holds 1 << ring_shift pairs for this CTA's engine.
                pair = k & ((1 << ring_shift) - 1)
                cute.arch.mbarrier_wait(empty_bars + pair, ((k >> ring_shift) & 1) ^ 1)
                # The consumers' generic-proxy reads of this pair (ordered
                # before their arrive, acquired by the wait above) precede the
                # async-proxy (TMA) refill below: without this proxy fence the
                # refill may overtake them (rare, schedule-dependent output
                # corruption).
                cute.arch.fence_view_async_shared()
                valid = cute.min(count - 2 * k, 2)
                prims.elected_arrive_expect_tx(
                    full_bars + pair, valid * (COEFF_BYTES + value_bytes)
                )
                for i in cutlass.range_constexpr(2):
                    if 2 * k + i < count:
                        destination = staged.iterator + (2 * pair + i) * STAGE_FLOATS
                        prims.elected_bulk_copy_g2s(
                            destination,
                            coeff_address,
                            cutlass.Int32(COEFF_BYTES),
                            full_bars + pair,
                        )
                        prims.elected_bulk_copy_g2s(
                            destination + VALUE_FLOAT_OFFSET,
                            value_address,
                            value_bytes,
                            full_bars + pair,
                        )
                    coeff_address = coeff_address + coeff_stride
                    value_address = value_address + head_stride
        if (count & 1) == 0:
            # Token `count` opens a pair of its own: a placeholder (no bytes).
            k = count // 2
            pair = k & ((1 << ring_shift) - 1)
            cute.arch.mbarrier_wait(empty_bars + pair, ((k >> ring_shift) & 1) ^ 1)
            prims.elected_arrive_expect_tx(full_bars + pair, cutlass.Int32(0))


@cute.kernel
def _scan_kernel(
    v_in: cute.Tensor,
    coeff: cute.Tensor,
    state_dst: cute.Tensor,
    state_src: cute.Tensor,
    o_out: cute.Tensor,
    cu_seqlens: cute.Tensor,
    prefix: cute.Tensor,
    ranks: cute.Tensor,
    order: cute.Tensor,
    state_indices: cute.Tensor,
    window_start: cutlass.Int32,
    window_count: cutlass.Int32,
    runtime_rows: cutlass.Int32,
    long_pairs: cutlass.Int32,
    radix4: cutlass.Int32,
    checkpoints: cute.Tensor,
    checkpoint_indices: cute.Tensor,
    flags: cute.Tensor,
    counters: cute.Tensor,
    window_index: cutlass.Int32,
    CHECKPOINTS: cutlass.Constexpr[bool],
    WRITE_FINAL: cutlass.Constexpr[bool],
    RADIX4: cutlass.Constexpr[bool],
    TRIGGER: cutlass.Constexpr[bool],
    LEAN: cutlass.Constexpr[bool],
):
    """Four consumer warps own 16 or 32 V rows; warp 4 stages coefficients.

    A uniform runtime branch chooses the row engine inside this one kernel.
    Both engines use the same arithmetic and BF16 continuation-state bits.
    """
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    warp = tidx // 32
    lane = tidx % 32
    n_heads = cutlass.Int32(v_in.shape[2])
    # The first long_pairs request-head pairs (requests longest-first from
    # the schedule kernel's order, heads in order within a request) use the
    # two-row engine (eight CTAs per pair); the rest use runtime_rows.
    # Placement only: both engines run the identical arithmetic.
    long_blocks = long_pairs * 8
    is_long = bidx < long_blocks
    local_bidx = bidx if is_long else bidx - long_blocks
    if (radix4 & LONG_LAST) != 0:
        # Placement only: the runtime_rows CTAs take the first block indices
        # (one per SM in the first dispatch round), the two-row CTAs follow.
        short_blocks = (cutlass.Int32(state_dst.shape[0]) * n_heads - long_pairs) * (
            128 // (CONSUMER_WARPS * 2 * runtime_rows)
        )
        is_long = bidx >= short_blocks
        local_bidx = bidx - short_blocks if is_long else bidx
    rows = runtime_rows
    if is_long:
        rows = cutlass.Int32(2)
    pair_base = cutlass.Int32(0) if is_long else long_pairs
    row_shift = 2
    if rows == 2:
        row_shift = 3
    if rows == 8:
        row_shift = 1
    row_block = local_bidx & ((1 << row_shift) - 1)
    slot_bh = local_bidx >> row_shift
    skip = cutlass.Int32(0)
    if (radix4 & 16) != 0:
        # Spare layout (long_pairs == n_heads sizes the grid): the first 4 n_heads blocks
        # are spares for the longest request, then the usual four-row blocks of
        # every request in longest-first order. When the longest request is
        # strictly longer than the runner-up (device metadata), its four-row
        # blocks and the spares become eight two-row blocks per head (rows 0-63
        # and 64-127, all in the first placement round); otherwise the spares
        # exit at once. Placement and engine only: same arithmetic.
        dominant = cutlass.Int32(0)
        # Longest-first records: lengths of the leading two requests.
        if order[2] > order[6]:
            dominant = cutlass.Int32(1)
        rows = runtime_rows
        pair_base = cutlass.Int32(0)
        spares = n_heads * 4
        row_block = (bidx - spares) & 3
        slot_bh = (bidx - spares) >> 2
        if bidx < spares:
            slot_bh = bidx >> 2
            row_block = 4 + (bidx & 3)
            if dominant == 0:
                skip = cutlass.Int32(1)
        if slot_bh < n_heads and dominant == 1:
            rows = cutlass.Int32(2)
    value_rows = CONSUMER_WARPS * 2 * rows
    pair_index = pair_base + slot_bh
    v_head_idx = pair_index % n_heads
    # One 16-byte record (index, first token, length, first rank) per request
    # in longest-first order, written by the schedule kernel.
    record = cute.make_rmem_tensor(cute.make_layout((4,), stride=(1,)), cutlass.Int32)
    cute.autovec_copy(
        cute.make_tensor(
            _known_aligned(order.iterator + (pair_index // n_heads) * 4, 16),
            cute.make_layout((4,), stride=(1,)),
        ),
        record,
    )
    batch_idx = record[0]
    bh = batch_idx * n_heads + v_head_idx
    if cutlass.const_expr(state_indices is not None):
        bh = cutlass.Int64(state_indices[batch_idx]) * n_heads + bh % n_heads
    cta_v_offset = row_block * value_rows
    seq_start = record[1]
    length = record[2]
    rank_offset = record[3]
    segments = (length + SEGMENT - 1) // SEGMENT
    # Ranks are a permutation of [0, total segments). When this window covers
    # all of them (the usual single window) the searches below are known to
    # return [0, segments); skip their dependent global loads.
    first_segment = cutlass.Int32(0)
    end_segment = segments
    if window_start != 0 or window_count < prefix[cu_seqlens.shape[0] - 1]:
        first_segment, end_segment = segment_range(
            ranks, rank_offset, segments, window_start, window_start + window_count
        )
    start = first_segment * SEGMENT
    count = cute.max(cute.min(length, end_segment * SEGMENT) - start, 0)
    # PDL secondary of this window's coefficient kernel: everything above
    # reads only schedule outputs (complete before that kernel triggered).
    # Wait here for the coefficients and (through that kernel's ordered
    # completion) the previous window's continuation state.
    overlap = (radix4 & OVERLAP) != 0
    if overlap and window_index > 0:
        # Overlapped windows: this scan launched once every block of its
        # coefficient kernel triggered (at its end), while the previous
        # window's scan may still run. Wait for (a) all of this window's coefficient
        # blocks (release counter) and (b) the previous window's CTA of the
        # same index, whose last memory operation publishes its flag after
        # its state stores.
        if warp == 0:
            # Lane i waits for counter i: the blocks whose linear index is i
            # modulo COUNTERS.
            total_blocks = window_count * TOKEN_BLOCKS * ((n_heads + 3) // 4)
            expected = total_blocks // COUNTERS
            if lane < total_blocks % COUNTERS:
                expected = expected + 1
            counter = counters.iterator + window_index * COUNTERS + lane
            blocks = cute.arch.load(counter, cutlass.Int32, sem="acquire", scope="gpu")
            while blocks < expected:
                _nanosleep(128)
                blocks = cute.arch.load(
                    counter, cutlass.Int32, sem="acquire", scope="gpu"
                )
        if tidx == 32:
            # (Polled by a second warp, in parallel with the counters.)
            flag = flags.iterator + bidx
            ready = cute.arch.load(flag, cutlass.Int32, sem="acquire", scope="gpu")
            while ready < window_index:
                _nanosleep(128)
                ready = cute.arch.load(flag, cutlass.Int32, sem="acquire", scope="gpu")
        cute.arch.sync_threads()
        if warp == CONSUMER_WARPS:
            # The producer's TMA (async proxy) reads the coefficient rows
            # other SMs wrote through the generic proxy.
            cute.arch.fence_proxy("async.global")
    elif window_index > 0:
        cute.arch.griddepcontrol_wait()
    # The next window's coefficient kernel (PDL, other workspace region)
    # may start once every CTA of this scan has passed its waits, i.e.
    # after the previous scan, which read that region, has completed (or,
    # overlapped, every CTA of it has published its flag). In the first
    # window (no previous scan; the state comes from state_src) only the
    # producer warp waits for the coefficients, below.
    if window_index > 0:
        cute.arch.griddepcontrol_launch_dependents()
    if skip == 0 and length == 0 and window_start == 0:
        for i in cutlass.range(tidx, value_rows * 128, (CONSUMER_WARPS + 1) * 32):
            offset = cutlass.Int64(bh) * 16384 + cta_v_offset * 128 + i
            target = cute.make_tensor(
                state_dst.iterator + offset, cute.make_layout((1,))
            )
            source = cute.make_tensor(
                state_src.iterator + offset, cute.make_layout((1,))
            )
            target[0] = source[0]
    if skip == 0 and count > 0:
        memspace = v_in.iterator.memspace
        value_base = (cutlass.Int64(seq_start) * n_heads + v_head_idx) * 128
        coeff_base = cutlass.Int64(0)
        coeff_row0 = cutlass.Int64((coeff.iterator + coeff_base).toint())
        value_row0 = cutlass.Int64((v_in.iterator + value_base).toint())
        output_row0 = cutlass.Int64((o_out.iterator + value_base).toint())
        coeff_stride = cutlass.Int64(n_heads) * (392 * 4)
        head_stride = cutlass.Int64(n_heads) * (128 * 2)

        allocator = SmemAllocator()
        staged = allocator.allocate_tensor(
            cutlass.Float32,
            cute.make_layout((PIPE_STAGES, STAGE_FLOATS), stride=(STAGE_FLOATS, 1)),
            byte_alignment=128,
        )
        staged_values = cute.make_tensor(
            cute.recast_ptr(staged.iterator, dtype=cutlass.BFloat16),
            cute.make_layout(
                (PIPE_STAGES, 2 * STAGE_FLOATS), stride=(2 * STAGE_FLOATS, 1)
            ),
        )
        full_bars = allocator.allocate_array(
            cutlass.Int64, num_elems=PAIRS, byte_alignment=8
        )
        empty_bars = allocator.allocate_array(
            cutlass.Int64, num_elems=PAIRS, byte_alignment=8
        )
        # Every engine reads one slot past its last token (the look-ahead
        # after token count - 1): slot count of the 32-slot ring, or of the
        # sixteen-slot ring of two-row halving and K-packed CTAs. Zero both
        # so that read is defined; every other slot is staged by TMA before
        # it is read.
        zeros = cute.make_rmem_tensor(
            cute.make_layout((4,), stride=(1,)), cutlass.Float32
        )
        zeros.fill(0.0)
        for chunk in cutlass.range_constexpr(
            (2 * STAGE_FLOATS + 4 * (CONSUMER_WARPS + 1) * 32 - 1)
            // (4 * (CONSUMER_WARPS + 1) * 32)
        ):
            index = chunk * (CONSUMER_WARPS + 1) * 32 + tidx
            # (The two slots coincide when bit 4 of count is clear: once.)
            limit = 2 * STAGE_FLOATS if (count & 16) != 0 else STAGE_FLOATS
            if index * 4 < limit:
                zero_slot = count & (PIPE_STAGES - 1)
                if index * 4 >= STAGE_FLOATS:
                    zero_slot = count & 15
                cute.autovec_copy(
                    zeros,
                    cute.make_tensor(
                        _known_aligned(
                            staged.iterator
                            + zero_slot * STAGE_FLOATS
                            + (index * 4) % STAGE_FLOATS,
                            16,
                        ),
                        cute.make_layout((4,), stride=(1,)),
                    ),
                )
        # The zero-filled slots are later overwritten by the async (TMA) proxy.
        cute.arch.fence_view_async_shared()
        if tidx == 0:
            for s in cutlass.range_constexpr(PAIRS):
                cute.arch.mbarrier_init(full_bars + s, 1)
                cute.arch.mbarrier_init(empty_bars + s, CONSUMER_WARPS * 32)
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()

        if warp == CONSUMER_WARPS:
            if window_index == 0:
                # First window: the coefficient rows are complete once the
                # coefficient kernel (PDL primary) is.
                cute.arch.griddepcontrol_wait()
                cute.arch.griddepcontrol_launch_dependents()
            # Two-row halving CTAs use a sixteen-slot ring (static slot offsets
            # in their sixteen-token loop); every other engine the whole ring.
            ring_shift = cutlass.Int32(PAIRS.bit_length() - 1)
            if rows == 2 and (radix4 & 64) != 0 and (radix4 & PAIR_ENGINE) == 0:
                ring_shift = cutlass.Int32(3)
            if rows == 4 and (radix4 & KPACK) != 0:
                # K-packed four-row CTAs: the sixteen-slot ring (RING there).
                ring_shift = cutlass.Int32(3)
            _produce(
                staged,
                full_bars,
                empty_bars,
                lane,
                count,
                coeff_row0,
                coeff_stride,
                value_row0 + start * head_stride + cutlass.Int64(cta_v_offset * 2),
                head_stride,
                ranks,
                rank_offset,
                first_segment,
                window_start,
                v_head_idx,
                n_heads,
                memspace,
                value_rows,
                ring_shift,
            )
        else:
            if rows == 4 and (radix4 & KPACK) != 0:
                run_kpack_consumer(
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
                    memspace,
                    radix4 & LONG_STEADY,
                )
            elif rows == 4:
                run_halving_consumer(
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
                    memspace,
                    radix4 & LONG_STEADY,
                )
            elif rows == 8:
                # Eight-row engine (64 rows per CTA): same bits.
                run_octa_consumer(
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
                    memspace,
                )
            elif (radix4 & PAIR_ENGINE) != 0:
                run_pair_consumer(
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
                    memspace,
                )
            else:
                # Dispatch the two runtime consumer configurations (every
                # two-row CTA has bit 64, every four-row CTA the halving
                # engine), so the two hot loops sit next to each other in a
                # kernel less than half the size. Same engines, same bits.
                run_halving2_consumer(
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
                    memspace,
                    radix4 & TWO_AHEAD,
                )

    if (radix4 & OVERLAP) != 0:
        # Publish this CTA's window: after the barrier (every thread's state
        # and output stores), one release store. (Thread and block indices
        # are read again here.)
        cute.arch.sync_threads()
        if _special_register("tid.x") == 0:
            cute.arch.store(
                flags.iterator + _special_register("ctaid.x"),
                window_index + 1,
                sem="release",
                scope="gpu",
            )


@cute.jit
def _launch_scan(
    v: cute.Tensor,
    coeff: cute.Tensor,
    state_out: cute.Tensor,
    state_in: cute.Tensor,
    out: cute.Tensor,
    cu_seqlens: cute.Tensor,
    prefix: cute.Tensor,
    ranks: cute.Tensor,
    order: cute.Tensor,
    state_indices: cute.Tensor,
    start: cutlass.Int32,
    count: cutlass.Int32,
    runtime_rows: cutlass.Int32,
    long_pairs: cutlass.Int32,
    radix4: cutlass.Int32,
    flags: cute.Tensor,
    counters: cute.Tensor,
    window_index: cutlass.Int32,
    smem_bytes: cutlass.Int32,
    RADIX4: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
    TRIGGER: cutlass.Constexpr[bool],
    LEAN: cutlass.Constexpr[bool],
):
    # Three resident CTAs per SM (128 registers per thread): the latency-bound
    # engines use the registers to overlap consecutive tokens; the 4096 registers left
    # still holds one coefficient block of the next window (PDL overlap).
    _scan_kernel(
        v,
        coeff,
        state_out,
        state_in,
        out,
        cu_seqlens,
        prefix,
        ranks,
        order,
        state_indices,
        start,
        count,
        runtime_rows,
        long_pairs,
        radix4,
        state_out,
        coeff,
        flags,
        counters,
        window_index,
        False,
        True,
        RADIX4,
        TRIGGER,
        LEAN,
    ).launch(
        grid=[
            long_pairs * 8
            + ((cu_seqlens.shape[0] - 1) * v.shape[2] - long_pairs)
            * (128 // (CONSUMER_WARPS * 2 * runtime_rows)),
            1,
            1,
        ],
        block=[32 * (CONSUMER_WARPS + 1), 1, 1],
        # Runtime dynamic shared memory (at least the kernel's usage): host
        # policy may request more to cap resident CTAs per SM (placement only).
        smem=smem_bytes,
        stream=stream,
        min_blocks_per_mp=MIN_BLOCKS,
        use_pdl=True,
    )


def _fake(shape, dtype=cute.BFloat16):
    return cute.runtime.make_fake_compact_tensor(
        dtype, shape, assumed_align=32, stride_order=tuple(reversed(range(len(shape))))
    )


@functools.cache
def compiled_scan_kernel(device_index, capability, indexed=False):
    radix4, trigger, lean = False, True, True
    b, length, h, n, m, r, o, f, c = (cute.sym_int() for _ in range(9))
    x = _fake((1, length, h, 128))
    state = _fake((b, h, 128, 128), cute.Float32)
    return cute.compile(
        _launch_scan,
        x,
        _fake((n,), cute.Float32),
        state,
        state,
        x,
        _fake((m,), cute.Int32),
        _fake((m,), cute.Int32),
        _fake((r,), cute.Int32),
        _fake((o,), cute.Int32),
        _fake((cute.sym_int(),), cute.Int32) if indexed else None,
        cutlass.Int32(0),
        cutlass.Int32(0),
        cutlass.Int32(4),
        cutlass.Int32(0),
        cutlass.Int32(0),
        _fake((f,), cute.Int32),
        _fake((c,), cute.Int32),
        cutlass.Int32(0),
        cutlass.Int32(SMEM_BYTES),
        radix4,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        trigger,
        lean,
        options="--enable-tvm-ffi",
    )
