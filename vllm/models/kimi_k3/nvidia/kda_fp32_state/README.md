# FP32-state consistent KDA kernels

[简体中文](README.zh-CN.md)

KDA forward kernels for K3/K3.1 with BF16 inputs/outputs and FP32 recurrent state.
Prefill and decode execute the same ordered per-token FP32 program: normalize
Q/K, evaluate gates, compute the prediction and delta, update state, then project
the output. Prefill parallelizes coefficient generation while preserving the
state recurrence. Decode supports any positive fixed Q, including Q4/Q6/Q8 for
MTP3/MTP5/MTP7, with optional checkpoints after every token.

Within this contract, effective outputs, continuation state and checkpoints are
bitwise invariant to prefill/decode partitioning, batch composition/order and
complete-head sharding on the same GPU model/software stack. This is a separate
floating-point contract from CF-BI; it is not a bitwise replacement for a trainer
that still uses CF-BI. Both engines must adopt the same contract for alignment.

This addition provides kernels and a prepared API. It does not change model
routing, convolution, scheduler/cache management or communication. It does not
provide backward or guarantee full-model logits. UP-style complete-head sharding
is supported; Ring CP is not implemented.

## Requirements

SM100/SM103 (B200/B300), CUDA-enabled PyTorch and CuTe DSL 4.8.1.

```bash
python -m pip install -r vllm/models/kimi_k3/nvidia/kda_fp32_state/requirements.txt
```

## Example

```python
import torch
from vllm.models.kimi_k3.nvidia.kda_fp32_state import prepare_decode

B, Q, H, D = 2, 4, 12, 128
q, k, v, g = [torch.randn(B, Q, H, D, device="cuda", dtype=torch.bfloat16)
               for _ in range(4)]
beta = torch.randn(B, Q, H, device="cuda", dtype=torch.bfloat16)
A_log = torch.zeros(H, device="cuda")
dt_bias = torch.zeros(H, D, device="cuda")
state = torch.zeros(B, H, D, D, device="cuda")
out = torch.empty_like(q)
checkpoints = torch.empty(B, Q, H, D, D, device="cuda")
accepted = torch.tensor([0, 3], device="cuda", dtype=torch.int32)
plan = prepare_decode(
    q, k, v, g, beta, A_log, dt_bias, state, out, state,
    checkpoint_states=checkpoints, num_accepted_tokens=accepted,
)
plan.run()
# Update accepted after the acceptance decision, then commit on the same stream.
plan.commit()
```

## API contract

Exports: `prepare_prefill`, `prepare_decode`, `PrefillPlan`, `DecodePlan`.
Both prepare functions begin with `q,k,v,g,beta,A_log,dt_bias,state_in,out`;
`state_out` follows. `state_dtype=torch.float32` is the only accepted state dtype.
All buffers must be contiguous, on the same CUDA device and 32-byte aligned when
nonempty. State axes are **[V,K]**. D=128 and H>0.

| Argument | Prefill | Decode | dtype |
|---|---|---|---|
| q/k/v/g/out | dense `[B,L,H,D]` or packed `[1,T,H,D]` | `[B,Q,H,D]`, Q≥1 | BF16 |
| beta | matching first three axes | `[B,Q,H]` | BF16 |
| A_log / dt_bias | `[H]` / `[H,D]` | same | FP32 |
| state_in/state_out | `[N,H,D,D]` or indexed pool `[P,H,D,D]` | `[B,H,D,D]` or indexed pool `[P,H,D,D]` | FP32 |
| cu_seqlens | device `[N+1]` for packed input; omit for dense | unused | Int32 |
| state_indices | optional `[N]` | optional `[B]` | Int32 |
| checkpoint_states | unsupported | optional `[B,Q,H,D,D]` for Q>1; forbidden for Q1 | FP32 |
| num_accepted_tokens | unsupported | optional `[B]`, values 0..Q for Q>1 commit | Int32 |

Defaults: `scale=128**-0.5`, `eps=1e-6`, `lower_bound=-5.0`.
Q/K normalization and gate activations are internal; g/beta are raw activations.
Scalars must be finite FP32 values, eps positive and lower_bound in [-5,0].
No intermediate coefficients are rounded to BF16 and state is never rounded to
BF16. Output is rounded to BF16; approximate FP32 elementary functions are used.

Prefill requires state_out; Q1 also requires state_out. They permit identical
input/output state buffers. Without `checkpoint_states`, Q>1 also requires
`state_out`: `run()` writes all token outputs and the final state, and permits exact
input/output state aliasing. This mode has no acceptance selection or rollback.
With `checkpoint_states`, Q>1 `run()` leaves the state pool unchanged and returns
`(out, checkpoint_states[:, -1])`. This terminal view may be noncontiguous: copy
before rebinding it as input. Supply both state_out and num_accepted_tokens to
use `commit()`: accepted a>0 copies checkpoint a−1; a=0 preserves/copies initial
state. `commit()` returns the destination pool, also exposed as `accepted_state`.

Pool slots must be unique within a call and in `[0,P)`, with `N≤P<2³¹`.
Unselected slots remain untouched. **Slot 0 is an ordinary slot, not a padding
sentinel.** The integration must remove padded/NULL_SLOT rows or give them valid,
unique scratch slots before calling this API. Partial buffer aliasing and
checkpoint/pool aliasing are forbidden.

Packed offsets start at zero, are nondecreasing, and end at or below T; unused
output tails are untouched. Empty requests/batches are valid. Prefill supports
up to 1023 requests, subject to Int32/grid limits. Decode has uniform Q across
requests, with Q and B×Q below 2³¹. Metadata contents stay device-resident and
must be valid; prepare does not read them back to the host. Historical KV is
represented only by the supplied state. Noncontiguous sequence fragments need
separate correct entry states.

## Plans and CUDA Graphs

Prepare validates, compiles and binds addresses/capacities outside capture.
Prefill workspace is bounded by `workspace_limit_bytes` (default 256 MiB);
decode's default is 64 MiB and this FP32 path requires no internal workspace.
Insufficient prefill workspace raises an error. Run/commit allocate no device
memory, compile nothing and use the bound device's current stream.

Keep plans, buffers and workspace alive until completion. Update input values,
slots, offsets and acceptance counts in place for Graph replay; recreate plans
for new addresses/shapes/scalars. Between run and commit, retain slot identities,
initial state and checkpoints. Do not overlap reads/writes to a state slot across
calls/streams without synchronization, or concurrently reuse a plan/workspace.
An in-place commit advances subsequent calls; an out-of-place commit requires
binding the destination as the next input.

Prefill B/H/lengths are runtime values, not compilation variants. Decode Q can
select an instance; exact batch size, historical length, device pointer and
metadata contents cannot. Invalid host-visible shape/type/dtype/alignment/alias
or scalar arguments raise TypeError/ValueError. Prepare during capture or
requires-grad inputs with autograd enabled raise RuntimeError. Forward requires
finite valid inputs; kernels do not scan for NaN/Inf at runtime.

## Contract fingerprint

`kda_fp32_state.contract` records what decides the bits these kernels produce: the
contract id, a digest of the kernel sources, the CuTe DSL packages, `CUTE_DSL_*`
environment variables and the GPU model, plus the torch, CUDA, driver and `ptxas`
versions for diagnosis. Log it in the rollout and in the trainer process and compare:

```python
from vllm.models.kimi_k3.nvidia.kda_fp32_state import contract

fingerprint = contract.contract_fingerprint()
print(contract.fingerprint_digest(fingerprint))
contract.assert_same_contract(other_process_fingerprint, fingerprint)
contract.require_pinned_toolchain()
```

`assert_same_contract` raises `ContractMismatch` for the fields that can change the
bits and ignores the diagnostic ones unless `informational=True`.
`require_pinned_toolchain` is an opt-in check of the installed CuTe DSL and tvm-ffi
against `requirements.txt`. Equal fingerprints rule out these toolchain differences;
they do not prove bitwise equality, so the bytewise comparison of dumped outputs and
states stays the gate. Bump `CONTRACT_ID` when a change alters output bits. No other
module calls this one.

## Tests and measurements

```bash
python -m pip install pytest triton
python -m pytest -q tests/kernels/kda_fp32_state
```

Two matching GPUs are needed for the complete-head and PD handoff checks; these
checks skip on single-GPU systems. [Tests](../../../../../tests/kernels/kda_fp32_state/README.md)
compare prefill, Q1/configurable Q, arbitrary splits, batch/reordering, varlen,
checkpoint continuation, dynamic metadata, Graphs and compilation-cache reuse.
[Benchmark data](../../../../../benchmarks/kda_fp32_state/README.md) includes
FlashInfer and vLLM CF-BI C16/C64 comparisons with explicit precision/timing scopes.

## License

[MIT](LICENSE), Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
FlashInfer-derived portions retain [Apache-2.0](licenses/FlashInfer-Apache-2.0.txt)
and [attribution](licenses/FlashInfer-NOTICE.txt).
