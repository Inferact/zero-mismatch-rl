"""Lower-bounded sigmoid decay gate of the fixed FP32 program.

gate(x) = exp2(lb * log2(e) * sigmoid(exp(a_log) * x)), evaluated as
exp2(lb_log2e / (1 + exp2(-exp(a_log) * log2(e) * x))) with the program's
fast-math exp2 and reciprocal, so every caller produces the same bits.

Provenance: the formula and its evaluation order follow the recurrent contract's
gate helper (adapted from FlashInfer, Apache License 2.0), so the bits match
the fixed program.
"""

import cutlass
from .. import _primitives as prims
import cutlass.cute as cute
from cutlass._mlir.dialects import arith as mlir_arith
from cutlass._mlir.dialects import math as mlir_math

LOG2_E = 1.4426950408889634


@cute.jit
def decay_gate(x, rate, lower_bound):
    """x: biased raw gate (FP32); rate: exp(a_log) of the head (FP32)."""
    log2e = cutlass.Float32(LOG2_E)
    slope = -rate * log2e
    floor = lower_bound * log2e
    fast = mlir_arith.FastMathFlags.fast
    one = cutlass.Float32(1.0).ir_value()
    scaled = mlir_arith.mulf(slope.ir_value(), x.ir_value(), fastmath=fast)
    denominator = mlir_arith.addf(
        one, mlir_math.exp2(scaled, fastmath=fast), fastmath=fast
    )
    sigmoid = mlir_arith.divf(one, denominator, fastmath=fast)
    exponent = mlir_arith.mulf(floor.ir_value(), sigmoid, fastmath=fast)
    return cutlass.Float32(mlir_math.exp2(exponent, fastmath=fast))


@cute.jit
def _octa_reduce(partials):
    """partials[s]: dual-chain sum of slot s (row s ^ x) for this K lane
    (K lane = lane % 16). Returns the 8/4/2/1 K-group tree sum of row x."""
    FULL = 4294967295
    v2 = cute.arch.shuffle_sync_bfly(partials[1], offset=2, mask=FULL)
    v4 = cute.arch.shuffle_sync_bfly(partials[2], offset=4, mask=FULL)
    v6 = cute.arch.shuffle_sync_bfly(partials[3], offset=6, mask=FULL)
    v8 = cute.arch.shuffle_sync_bfly(partials[4], offset=8, mask=FULL)
    v10 = cute.arch.shuffle_sync_bfly(partials[5], offset=10, mask=FULL)
    v12 = cute.arch.shuffle_sync_bfly(partials[6], offset=12, mask=FULL)
    v14 = cute.arch.shuffle_sync_bfly(partials[7], offset=14, mask=FULL)
    # Level 8: (k, k^8), (k^2, k^10), (k^4, k^12), (k^6, k^14); level 4.
    a0, a2 = prims.add_packed_f32x2((partials[0], v2), (v8, v10))
    a4, a6 = prims.add_packed_f32x2((v4, v6), (v12, v14))
    b0, b2 = prims.add_packed_f32x2((a0, a2), (a4, a6))
    # Level 2, then level 1 with K lane k ^ 1 (same row).
    c = b0 + b2
    return c + cute.arch.shuffle_sync_bfly(c, offset=1, mask=FULL)
