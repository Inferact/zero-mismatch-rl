"""Prepared FP32-state KDA prefill and configurable-Q decode."""

from dataclasses import dataclass, field

import torch


def _validate_contract(q, state_in, state_dtype, workspace_limit_bytes):
    if state_dtype != torch.float32:
        raise ValueError("state_dtype must be torch.float32")
    if not isinstance(q, torch.Tensor) or not isinstance(state_in, torch.Tensor):
        raise TypeError("q and state_in must be torch.Tensor")
    if state_in.dtype != state_dtype:
        raise ValueError(
            "state_in.dtype must equal state_dtype; no implicit conversion"
        )
    if q.device.type != "cuda":
        raise ValueError("q must be a CUDA tensor")
    if not isinstance(workspace_limit_bytes, int) or workspace_limit_bytes < 0:
        raise ValueError("workspace_limit_bytes must be a nonnegative integer")
    with torch.cuda.device(q.device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("prepare must run outside CUDA Graph capture")
        if torch.cuda.get_device_capability(q.device) not in ((10, 0), (10, 3)):
            raise ValueError("this package targets SM100/SM103 Blackwell")


@dataclass(frozen=True)
class PrefillPlan:
    """Own bounded workspace; keep this plan and its buffers alive during execution."""

    _backend: object = field(repr=False)
    out: torch.Tensor = field(repr=False)
    state_out: torch.Tensor = field(repr=False)
    state_dtype: torch.dtype
    workspace_bytes: int

    def run(self):
        """Enqueue on the bound device's current stream; return output and state."""
        with torch.cuda.device(self.out.device):
            self._backend.run()
        return self.out, self.state_out


@dataclass(frozen=True)
class DecodePlan:
    """Terminal state or per-token checkpoints under one fixed contract."""

    _backend: object = field(repr=False)
    out: torch.Tensor = field(repr=False)
    state_out: torch.Tensor = field(repr=False)
    checkpoint_states: torch.Tensor | None = field(repr=False)
    state_dtype: torch.dtype
    workspace_bytes: int

    accepted_state: torch.Tensor | None = field(default=None, repr=False)
    _commit_call: tuple | None = field(default=None, repr=False)

    def commit(self):
        """Submit the multi-token accepted prefix using the bound device acceptance counts.

        Call after run and acceptance evaluation on the same stream. Zero
        preserves the original state. Slot/checkpoint identities must remain
        unchanged between run and commit. Returns the destination state pool.
        """
        if self._commit_call is None:
            raise RuntimeError(
                "commit requires Q>1 with num_accepted_tokens and state_out"
            )
        if self.accepted_state.requires_grad and torch.is_grad_enabled():
            raise RuntimeError("commit supports forward without autograd only")
        kernel, arguments = self._commit_call
        with torch.cuda.device(self.out.device):
            if kernel is not None:
                kernel(*arguments)
        return self.accepted_state

    def run(self):
        """Enqueue decode and return outputs and terminal state; commit selects an accepted prefix."""
        with torch.cuda.device(self.out.device):
            self._backend.run()
        return self.out, self.state_out


def prepare_prefill(
    q,
    k,
    v,
    g,
    beta,
    A_log,
    dt_bias,
    state_in,
    out,
    state_out,
    *,
    state_dtype=torch.float32,
    cu_seqlens=None,
    state_indices=None,
    scale=128**-0.5,
    eps=1e-6,
    lower_bound=-5.0,
    workspace_limit_bytes=256 * 1024**2,
):
    """Prepare dense [B,L,H,128] or packed [1,T,H,128] no_grad forward.

    state_dtype defaults to FP32; other state storage types are rejected.
    Q/K/V/g/beta/output are BF16; A_log/dt_bias are FP32. Packed cu_seqlens
    is device Int32 [N+1]. States are [N,H,128,128], V then K, or [P,H,128,128] pools
    indexed by optional device Int32 state_indices [N].
    See README.md for metadata, ownership, alignment and Graph constraints.
    """
    _validate_contract(q, state_in, state_dtype, workspace_limit_bytes)
    from .kernels.api import prepare_fp32_prefill as prepare

    backend = prepare(
        q,
        k,
        v,
        g,
        beta,
        A_log,
        dt_bias,
        state_in,
        out,
        state_out,
        cu_seqlens=cu_seqlens,
        state_indices=state_indices,
        scale=scale,
        eps=eps,
        lower_bound=lower_bound,
        workspace_limit_bytes=workspace_limit_bytes,
    )
    return PrefillPlan(backend, out, state_out, state_dtype, backend.workspace_bytes)


def prepare_decode(
    q,
    k,
    v,
    g,
    beta,
    A_log,
    dt_bias,
    state_in,
    out,
    state_out=None,
    *,
    state_dtype=torch.float32,
    checkpoint_states=None,
    state_indices=None,
    num_accepted_tokens=None,
    scale=128**-0.5,
    eps=1e-6,
    lower_bound=-5.0,
    workspace_limit_bytes=64 * 1024**2,
):
    """Prepare dense [B,Q,H,128] decode. Q is a positive host-visible shape dimension.

    Without checkpoints, state_out is required; run writes all outputs and the
    final state, with exact state_in/state_out aliasing allowed. Q>1 may instead
    provide checkpoint_states [B,Q,H,128,128]; run returns its last checkpoint.
    Optional state_indices maps requests to state pool slots. Multi-token decode can bind
    state_out and num_accepted_tokens for an explicit plan.commit() after
    acceptance evaluation. Zero accepted tokens preserve the initial state.
    No input dtype conversion or device-to-host metadata read is performed.
    """
    _validate_contract(q, state_in, state_dtype, workspace_limit_bytes)
    if (
        q.ndim != 4
        or not 1 <= q.shape[1] < 2**31
        or q.shape[2] < 1
        or q.shape[3] != 128
    ):
        raise ValueError("q must have shape [B,Q,H,128], 1 <= Q < 2**31, H>0")
    b, length, h, _ = q.shape
    if length == 1:
        if not isinstance(state_out, torch.Tensor) or checkpoint_states is not None:
            raise ValueError("Q1 requires state_out and no checkpoint_states")
    elif checkpoint_states is None:
        if not isinstance(state_out, torch.Tensor):
            raise ValueError("decode without checkpoints requires state_out")
        if num_accepted_tokens is not None:
            raise ValueError("num_accepted_tokens requires checkpoint_states")
    else:
        if not isinstance(checkpoint_states, torch.Tensor) or (
            state_out is not None and num_accepted_tokens is None
        ):
            raise ValueError(
                "Q>1 requires checkpoint_states; state_out requires num_accepted_tokens"
            )
        if (
            checkpoint_states.shape != (b, length, h, 128, 128)
            or checkpoint_states.dtype != state_dtype
            or checkpoint_states.device != q.device
            or not checkpoint_states.is_contiguous()
        ):
            raise ValueError(
                "Q>1 checkpoints must be contiguous [B,Q,H,128,128] with matching device/state_dtype"
            )
    args = (q, k, v, g, beta, A_log, dt_bias, state_in, out)
    options = dict(
        scale=scale, eps=eps, lower_bound=lower_bound, state_indices=state_indices
    )
    from .kernels.api import prepare_fp32_decode

    backend = prepare_fp32_decode(
        *args,
        state_out if checkpoint_states is None else None,
        checkpoint_states=checkpoint_states,
        **options,
    )
    workspace_bytes = backend.workspace_bytes
    terminal = checkpoint_states[:, -1] if checkpoint_states is not None else state_out
    commit_call = None
    if num_accepted_tokens is not None:
        if length == 1:
            raise ValueError("num_accepted_tokens requires Q>1")
        commit_call = _prepare_commit(
            q,
            state_in,
            state_out,
            checkpoint_states,
            state_indices,
            num_accepted_tokens,
            state_dtype,
            (q, k, v, g, beta, A_log, dt_bias, out),
        )
    return DecodePlan(
        backend,
        out,
        terminal,
        checkpoint_states,
        state_dtype,
        workspace_bytes,
        state_out if commit_call is not None else None,
        commit_call,
    )


def _prepare_commit(
    q, initial, destination, checkpoints, indices, accepted, dtype, buffers
):
    from ._validation import _range

    if not isinstance(destination, torch.Tensor):
        raise ValueError("multi-token commit requires state_out")
    if (
        destination.shape != initial.shape
        or destination.dtype != dtype
        or destination.device != q.device
        or not destination.is_contiguous()
        or (destination.numel() and destination.data_ptr() % 32)
    ):
        raise ValueError(
            "state_out must match the state_in pool shape, device and dtype, contiguous and aligned"
        )
    if (
        not isinstance(accepted, torch.Tensor)
        or accepted.shape != (q.shape[0],)
        or accepted.dtype != torch.int32
        or accepted.device != q.device
        or not accepted.is_contiguous()
        or (accepted.numel() and accepted.data_ptr() % 32)
    ):
        raise ValueError(
            "num_accepted_tokens must be contiguous aligned device Int32 [B], values 0..Q"
        )
    if destination.requires_grad and torch.is_grad_enabled():
        raise RuntimeError("state_out requires no_grad")
    values = [initial, checkpoints, destination, accepted, *buffers]
    if indices is not None:
        values.append(indices)
    for i, tensor in enumerate(values):
        lo, hi = _range(tensor)
        for j in range(i + 1, len(values)):
            ulo, uhi = _range(values[j])
            if max(lo, ulo) < min(hi, uhi):
                if (i, j) == (0, 2) and (lo, hi) == (ulo, uhi):
                    continue
                raise ValueError(
                    "overlapping commit buffers are forbidden except identical state_in/state_out"
                )
    kernel = None
    if q.shape[0]:
        from ._checkpoint import compiled_commit

        with torch.cuda.device(q.device):
            kernel = compiled_commit(
                q.device.index,
                torch.cuda.get_device_capability(q.device),
                dtype == torch.float32,
                indices is not None,
                initial.data_ptr() == destination.data_ptr(),
            )
    return kernel, (initial, checkpoints, destination, accepted, indices)
