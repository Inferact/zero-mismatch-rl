"""Device-side checkpoint selection; copies storage bits without arithmetic."""

import functools
import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda


@cute.kernel
def _commit_kernel(
    initial: cute.Tensor,
    checkpoints: cute.Tensor,
    destination: cute.Tensor,
    accepted: cute.Tensor,
    indices: cute.Tensor,
    INPLACE: cutlass.Constexpr[bool],
):
    tid, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    tile, bh = block % 8, block // 8
    h = cutlass.Int32(initial.shape[1])
    request = bh // h
    head = bh % h
    slot = request
    if cutlass.const_expr(indices is not None):
        slot = indices[request]
    count = accepted[request]
    # Metadata values are caller-owned. Zero never dereferences checkpoint -1.
    if count != 0 or not INPLACE:
        offset = (cutlass.Int64(slot) * h + head) * 16384
        source = initial.iterator + offset
        if count > 0:
            source = (
                checkpoints.iterator
                + ((cutlass.Int64(request) * checkpoints.shape[1] + count - 1) * h + head) * 16384
            )
        lane_offset = tile * 2048 + tid * 8
        layout = cute.make_layout((8,), stride=(1,))
        src = cute.make_tensor((source + lane_offset).align(16), layout)
        dst = cute.make_tensor(
            (destination.iterator + offset + lane_offset).align(16), layout
        )
        values = cute.make_rmem_tensor(layout, initial.element_type)
        cute.autovec_copy(src, values)
        cute.autovec_copy(values, dst)


@cute.jit
def _launch(
    initial: cute.Tensor,
    checkpoints: cute.Tensor,
    destination: cute.Tensor,
    accepted: cute.Tensor,
    indices: cute.Tensor,
    INPLACE: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
):
    _commit_kernel(
        initial, checkpoints, destination, accepted, indices, INPLACE
    ).launch(
        grid=(8 * accepted.shape[0] * initial.shape[1], 1, 1),
        block=(256, 1, 1),
        stream=stream,
    )


@functools.cache
def compiled_commit(device_index, capability, fp32, indexed, inplace):
    b, h, p, q = (cute.sym_int() for _ in range(4))

    def fake(shape, dtype):
        return cute.runtime.make_fake_compact_tensor(
            dtype,
            shape,
            assumed_align=32,
            stride_order=tuple(reversed(range(len(shape)))),
        )

    dtype = cute.Float32 if fp32 else cute.BFloat16
    state = fake((p, h, 128, 128), dtype)
    return cute.compile(
        _launch,
        state,
        fake((b, q, h, 128, 128), dtype),
        state,
        fake((b,), cute.Int32),
        fake((b,), cute.Int32) if indexed else None,
        inplace,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )
