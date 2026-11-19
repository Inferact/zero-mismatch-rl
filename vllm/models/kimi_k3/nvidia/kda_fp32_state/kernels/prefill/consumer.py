"""Shared coefficient-slot layout constants and barrier helpers of the scan."""

import cutlass
import cutlass.cute as cute
from ..._primitives import _known_aligned

PIPE_STAGES = 32  # token slots; pairs of slots share barriers


PAIRS = PIPE_STAGES // 2


CONSUMER_WARPS = 4


STAGE_FLOATS = 420  # q, k, gates, beta (+3 pad), then up to 64 BF16 values


VALUE_FLOAT_OFFSET = 388


FULL_MASK = 4294967295


STEADY_TOKENS = 8


LANE_STRIDE = 2


@cute.jit
def wait_test(bar, phase):
    complete = cute.arch.mbarrier_test_wait(bar, phase)
    while not complete:
        complete = cute.arch.mbarrier_test_wait(bar, phase)


@cute.jit
def finish_wait(bar, phase, ready):
    """Complete a full-barrier wait whose first probe (`ready`) was issued
    earlier, so the probe's latency is off the token's critical path."""
    complete = ready
    while not complete:
        complete = cute.arch.mbarrier_test_wait(bar, phase)


@cute.jit
def _gmem_ptr(
    address,
    dtype: cutlass.Constexpr,
    memspace: cutlass.Constexpr,
    alignment: cutlass.Constexpr[int],
):
    return cute.make_ptr(dtype, address, memspace, assumed_align=alignment)


@cute.jit
def _load_natural8(
    staged,
    slot,
    component: cutlass.Constexpr[int],
    k_lane,
    destination,
    parity: cutlass.Constexpr[int],
):
    # Swizzled natural coefficient layout: the float4 holding
    # K = 8 v + 4 h + (0..3) sits in slot 2 v + (h ^ bit 2 of v); values are
    # placed by their K index, conflict-free across the sixteen K lanes.
    for half in cutlass.range_constexpr(2):
        tile4 = cute.make_tensor(
            _known_aligned(
                staged.iterator
                + slot * STAGE_FLOATS
                + component * 128
                + (2 * k_lane + (half ^ ((k_lane >> 2) & 1))) * 4,
                16,
            ),
            cute.make_layout((4,), stride=(1,)),
        )
        fragment4 = cute.make_rmem_tensor(
            cute.make_layout((4,), stride=(1,)), cutlass.Float32
        )
        cute.autovec_copy(tile4, fragment4)
        for i in cutlass.range_constexpr(4):
            destination[parity, 4 * half + i] = fragment4[i]
