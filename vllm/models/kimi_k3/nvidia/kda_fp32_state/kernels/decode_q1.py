"""Q1 decode: one FP32 recurrence step per request-head.

Each warp keeps a 16-row by 128-column FP32 state tile in registers (eight
rows by eight K values per lane), applies the fixed program's gate,
prediction, delta, update and output dot in packed f32x2 arithmetic, and
reduces K with a radix-8 reduce-scatter plus a level-1 exchange in the
program's 8/4/2/1 pairing. State I/O uses 256-bit transfers.

Provenance: adapted from FlashInfer (Copyright (c) 2025 FlashInfer team,
Apache License 2.0, commit c3af0c13328fd23b256c53b9b4f94d9e160d118c). It is
rewritten here (packed f32x2 arithmetic, radix-8 reduce-scatter, 256-bit
state I/O); all intermediate arithmetic uses FP32.
"""

from .state_io import load256, store256
from .. import _primitives as prims
from .math import decay_gate, _octa_reduce
import functools
import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
import tvm_ffi  # noqa: F401

WARPS = 2  # 16-row V tiles per CTA (8 tiles per request-head)


@cute.kernel
def _decode_kernel(
    q_in: cute.Tensor,
    k_in: cute.Tensor,
    v_in: cute.Tensor,
    g_in: cute.Tensor,
    beta_in: cute.Tensor,
    state_dst: cute.Tensor,
    state_src: cute.Tensor,
    o_out: cute.Tensor,
    alog_in: cute.Tensor,
    bias_in: cute.Tensor,
    scale: cutlass.Float32,
    eps: cutlass.Float32,
    lower_bound: cutlass.Float32,
    state_indices: cute.Tensor,
):
    """One warp owns a [16 V, 128 K] tile in registers.

    Lanes partition K into eight-element vectors and partition V in two groups.
    State I/O uses explicit 256-bit transfers. All requests are active.
    """
    thread, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    # WARPS warps per CTA, each owning one 16-row V tile; same per-warp math.
    tidx = thread % 32
    bidx = block * WARPS + thread // 32
    n_heads = cutlass.Int32(v_in.shape[2])
    TILE_COUNT = 128 // 16
    v_tile_idx = bidx % TILE_COUNT
    bh = bidx // TILE_COUNT
    v_head_idx = bh % n_heads
    head_idx = v_head_idx
    v_offset = v_tile_idx * 16
    K_GROUPS = 128 // 8
    ROW_SETS = 32 // K_GROUPS
    PER_LANE = 128 // 32
    k_lane = tidx % K_GROUPS
    v_lane = tidx // K_GROUPS
    # Dense contiguous buffers share one linear batch-head index. Use 64-bit
    # offsets so a large state allocation cannot overflow an Int32 product.
    token_offset = cutlass.Int64(bh) * 128
    state_bh = cutlass.Int64(bh)
    if cutlass.const_expr(state_indices is not None):
        state_bh = cutlass.Int64(state_indices[bh // n_heads]) * n_heads + bh % n_heads
    state_offset = state_bh * 128 * 128
    state_layout = cute.make_layout((128, 128), stride=(128, 1))
    head_layout = cute.make_layout((128,), stride=(1,))
    h_read = cute.make_tensor(
        (state_src.iterator + state_offset).align(32), state_layout
    )
    h_layout = cute.make_layout((16 // ROW_SETS, 8), stride=(8, 1))
    vec8_layout = cute.make_layout((8,), stride=(1,))
    src_layout = cute.make_layout((PER_LANE,), stride=(1,))
    h_reg = cute.make_rmem_tensor(h_layout, cutlass.Float32)
    h_io = cute.make_rmem_tensor(vec8_layout, cutlass.Float32)
    q_src = cute.make_rmem_tensor(src_layout, cutlass.Float32)
    k_src = cute.make_rmem_tensor(src_layout, cutlass.Float32)
    gate_src = cute.make_rmem_tensor(src_layout, cutlass.Float32)
    q_bf16 = cute.make_rmem_tensor(src_layout, cutlass.BFloat16)
    k_bf16 = cute.make_rmem_tensor(src_layout, cutlass.BFloat16)
    gate_bf16 = cute.make_rmem_tensor(src_layout, cutlass.BFloat16)
    q_reg = cute.make_rmem_tensor(vec8_layout, cutlass.Float32)
    k_reg = cute.make_rmem_tensor(vec8_layout, cutlass.Float32)
    gate_reg = cute.make_rmem_tensor(vec8_layout, cutlass.Float32)
    # Slot s of this lane holds tile row v_lane + 2 (s ^ x), x = K-lane bits
    # 3..1, so the radix-8 reduce-scatter leaves each lane with row x's sum.
    x = (k_lane >> 1) & 7
    for j in cutlass.range_constexpr(16 // ROW_SETS):
        v_idx = v_offset + v_lane + ROW_SETS * (j ^ x)
        h_tile = cute.local_tile(h_read, (1, 8), (v_idx, k_lane))
        values = load256(h_tile.iterator)
        for i in cutlass.range_constexpr(8):
            h_reg[j, i] = values[i]
    bias_base = head_idx * 128
    decay_rate = cute.exp(alog_in[head_idx].to(cutlass.Float32), fastmath=True)
    q_head = cute.make_tensor((q_in.iterator + token_offset).align(32), head_layout)
    k_head = cute.make_tensor((k_in.iterator + token_offset).align(32), head_layout)
    gate_head = cute.make_tensor((g_in.iterator + token_offset).align(32), head_layout)
    v_head = cute.make_tensor((v_in.iterator + token_offset).align(32), head_layout)
    o_head = cute.make_tensor((o_out.iterator + token_offset).align(32), head_layout)
    beta_head = cute.make_tensor(
        beta_in.iterator + cutlass.Int64(bh), cute.make_layout((1,))
    )
    beta = beta_head[0].to(cutlass.Float32)
    beta = cutlass.Float32(1.0) / (
        cutlass.Float32(1.0) + cute.exp(-beta, fastmath=True)
    )
    q_tile = cute.local_tile(q_head, (PER_LANE,), (tidx,))
    k_tile = cute.local_tile(k_head, (PER_LANE,), (tidx,))
    gate_tile = cute.local_tile(gate_head, (PER_LANE,), (tidx,))
    cute.autovec_copy(q_tile, q_bf16)
    cute.autovec_copy(k_tile, k_bf16)
    cute.autovec_copy(gate_tile, gate_bf16)
    v_loaded = cutlass.Float32(0.0)
    if tidx < 16:
        v_loaded = v_head[v_offset + tidx].to(cutlass.Float32)
    for i in cutlass.range_constexpr(PER_LANE):
        k_idx = tidx * PER_LANE + i
        q_src[i] = q_bf16[i].to(cutlass.Float32)
        k_src[i] = k_bf16[i].to(cutlass.Float32)
        gate_src[i] = decay_gate(
            gate_bf16[i].to(cutlass.Float32)
            + bias_in[bias_base + k_idx].to(cutlass.Float32),
            decay_rate,
            lower_bound,
        )
    # Norms as in the coefficient kernel: packed even/odd squares, packed
    # (q, k) butterfly sums, then the normalizing products.
    q_sum_even, q_sum_odd = prims.mul_packed_f32x2(
        (q_src[0], q_src[1]), (q_src[0], q_src[1])
    )
    q_sum_even, q_sum_odd = prims.fma_packed_f32x2(
        (q_src[2], q_src[3]), (q_src[2], q_src[3]), (q_sum_even, q_sum_odd)
    )
    k_sum_even, k_sum_odd = prims.mul_packed_f32x2(
        (k_src[0], k_src[1]), (k_src[0], k_src[1])
    )
    k_sum_even, k_sum_odd = prims.fma_packed_f32x2(
        (k_src[2], k_src[3]), (k_src[2], k_src[3]), (k_sum_even, k_sum_odd)
    )
    q_sum_sq, k_sum_sq = prims.add_packed_f32x2(
        (q_sum_even, k_sum_even), (q_sum_odd, k_sum_odd)
    )
    for offset in [16, 8, 4, 2, 1]:
        q_sum_sq, k_sum_sq = prims.add_packed_f32x2(
            (q_sum_sq, k_sum_sq),
            (
                cute.arch.shuffle_sync_bfly(q_sum_sq, offset=offset, mask=4294967295),
                cute.arch.shuffle_sync_bfly(k_sum_sq, offset=offset, mask=4294967295),
            ),
        )
    q_shifted, k_shifted = prims.add_packed_f32x2((q_sum_sq, k_sum_sq), (eps, eps))
    q_scale_factor = cute.rsqrt(q_shifted, fastmath=True) * scale
    k_scale_factor = cute.rsqrt(k_shifted, fastmath=True)
    for p in cutlass.range_constexpr(4):
        q_pair = []
        k_pair = []
        for i in (2 * p, 2 * p + 1):
            source_lane = ROW_SETS * k_lane + i // PER_LANE
            source_value = i % PER_LANE
            q_pair.append(
                cute.arch.shuffle_sync(
                    q_src[source_value], offset=source_lane, mask=4294967295
                )
            )
            k_pair.append(
                cute.arch.shuffle_sync(
                    k_src[source_value], offset=source_lane, mask=4294967295
                )
            )
            gate_reg[i] = cute.arch.shuffle_sync(
                gate_src[source_value], offset=source_lane, mask=4294967295
            )
        q_reg[2 * p], q_reg[2 * p + 1] = prims.mul_packed_f32x2(
            (q_pair[0], q_pair[1]), (q_scale_factor, q_scale_factor)
        )
        k_reg[2 * p], k_reg[2 * p + 1] = prims.mul_packed_f32x2(
            (k_pair[0], k_pair[1]), (k_scale_factor, k_scale_factor)
        )
    partials = cute.make_rmem_tensor(
        cute.make_layout((8,), stride=(1,)), cutlass.Float32
    )
    # Gate and prediction partials of every slot (the fixed program's packed
    # chains 2, 0, 4, 6 and 3, 1, 5, 7 along K).
    for j in cutlass.range_constexpr(16 // ROW_SETS):
        for p in cutlass.range_constexpr(4):
            h_reg[j, 2 * p], h_reg[j, 2 * p + 1] = prims.mul_packed_f32x2(
                (h_reg[j, 2 * p], h_reg[j, 2 * p + 1]),
                (gate_reg[2 * p], gate_reg[2 * p + 1]),
            )
        even, odd = prims.mul_packed_f32x2(
            (h_reg[j, 2], h_reg[j, 3]), (k_reg[2], k_reg[3])
        )
        for i in [0, 4, 6]:
            even, odd = prims.fma_packed_f32x2(
                (h_reg[j, i], h_reg[j, i + 1]), (k_reg[i], k_reg[i + 1]), (even, odd)
            )
        partials[j] = even + odd
    prediction = _octa_reduce(partials)
    v_own = cute.arch.shuffle_sync(
        v_loaded, offset=v_lane + ROW_SETS * x, mask=4294967295
    )
    delta_own = (v_own - prediction) * beta
    for j in cutlass.range_constexpr(16 // ROW_SETS):
        delta = delta_own
        if cutlass.const_expr(j > 0):
            # Slot j's row is owned by K lane k ^ 2 j.
            delta = cute.arch.shuffle_sync_bfly(
                delta_own, offset=2 * j, mask=4294967295
            )
        for p in cutlass.range_constexpr(4):
            h_reg[j, 2 * p], h_reg[j, 2 * p + 1] = prims.fma_packed_f32x2(
                (k_reg[2 * p], k_reg[2 * p + 1]),
                (delta, delta),
                (h_reg[j, 2 * p], h_reg[j, 2 * p + 1]),
            )
        even, odd = prims.mul_packed_f32x2(
            (h_reg[j, 0], h_reg[j, 1]), (q_reg[0], q_reg[1])
        )
        for p in cutlass.range_constexpr(1, 4):
            even, odd = prims.fma_packed_f32x2(
                (h_reg[j, 2 * p], h_reg[j, 2 * p + 1]),
                (q_reg[2 * p], q_reg[2 * p + 1]),
                (even, odd),
            )
        partials[j] = even + odd
    out_for_lane = _octa_reduce(partials)
    # Lanes with K-lane bit 0 clear store their row x.
    if (k_lane & 1) == 0:
        o_head[v_offset + v_lane + ROW_SETS * x] = out_for_lane.to(cutlass.BFloat16)
    h_out = cute.make_tensor(
        (state_dst.iterator + state_offset).align(32), state_layout
    )
    for j in cutlass.range_constexpr(16 // ROW_SETS):
        v_idx = v_offset + v_lane + ROW_SETS * (j ^ x)
        for i in cutlass.range_constexpr(8):
            h_io[i] = h_reg[j, i]
        h_tile = cute.local_tile(h_out, (1, 8), (v_idx, k_lane))
        store256(
            h_tile.iterator,
            (h_io[0], h_io[1], h_io[2], h_io[3], h_io[4], h_io[5], h_io[6], h_io[7]),
        )


@cute.jit
def _launch(
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    g: cute.Tensor,
    beta: cute.Tensor,
    state_out: cute.Tensor,
    state_in: cute.Tensor,
    out: cute.Tensor,
    a_log: cute.Tensor,
    dt_bias: cute.Tensor,
    scale: cutlass.Float32,
    eps: cutlass.Float32,
    lower_bound: cutlass.Float32,
    state_indices: cute.Tensor,
    stream: cuda.CUstream,
):
    _decode_kernel(
        q,
        k,
        v,
        g,
        beta,
        state_out,
        state_in,
        out,
        a_log,
        dt_bias,
        scale,
        eps,
        lower_bound,
        state_indices,
    ).launch(
        grid=[q.shape[1] * q.shape[2] * (8 // WARPS), 1, 1],
        block=[32 * WARPS, 1, 1],
        stream=stream,
    )


@functools.cache
def compiled_decode_kernel(device_index, capability, indexed=False):
    import torch

    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("prepare FP32 decode outside CUDA Graph capture")
    b, h = (cute.sym_int(), cute.sym_int())

    def fake(shape, dtype=cute.BFloat16):
        return cute.runtime.make_fake_compact_tensor(
            dtype,
            shape,
            assumed_align=32,
            stride_order=tuple(reversed(range(len(shape)))),
        )

    # Packed Q1 layout as passed by the API: [1, T=B, H, D], no host reshapes.
    x = fake((1, b, h, 128))
    state = fake((cute.sym_int() if indexed else b, h, 128, 128), cute.Float32)
    return cute.compile(
        _launch,
        x,
        x,
        x,
        x,
        fake((1, b, h)),
        state,
        state,
        x,
        fake((h,), cute.Float32),
        fake((h * 128,), cute.Float32),
        cutlass.Float32(0.0),
        cutlass.Float32(0.0),
        cutlass.Float32(0.0),
        fake((b,), cute.Int32) if indexed else None,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi --generate-line-info",
    )
