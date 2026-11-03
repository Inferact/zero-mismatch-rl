# Sync robustness: collect everything, verify, stop on failure

The subject is trainer → rollout online sync (verl `BucketedWeightSender/Receiver` + vLLM layerwise reload). It answers four questions: how to decide a sync has received everything, how to prove the received bytes are the sent bytes, how to avoid deadlocking both sides on failure, and why a small adapter takes minutes to sync. Mechanism taxonomy is in the [01 case-map](01-case-map.md); version semantics in [02 policy versions](02-policy-versions.md).

## Three bugs and their fixes

| bug | Symptom | Root cause | Fix |
|---|---|---|---|
| Legacy-format `input_scale` arriving late crashes | W4A16 layers are processed as soon as weights arrive and `input_scale` is deleted; when the old checkpoint's `input_scale` arrives in a later bucket, "no module or parameter named …input_scale" is raised, MoE layers raise AttributeError | After the completion criterion stopped waiting for placeholders, layers are processed early; old checkpoints still carry 5935 such tensors | Receiver checks completeness by name: ModelOpt configs containing W4A16 layers add `.input_scale` to `_ignore_unexpected_suffixes`, effective only when the parameter no longer exists; FP8 layers' `input_scale` is a real parameter, loaded and waited for as usual |
| Router bias not in the integrity check | Sync still "succeeds" when `e_score_correction_bias` is not sent | It is an in-place-loaded parameter, in `SKIP_LOAD_TENSORS`, and not counted in the layer's `load_numel` | Generate expected from the manifest: the `load` callback collects the mapped names returned by `load_weights`; before finalize, every parameter in `named_parameters()` whose last name segment is in the skip list must have appeared, otherwise it counts as incomplete |
| Duplicate sends hang the sender | A duplicate tensor raises in the bucket callback, the receiver returns, the sender is still blocked in `recv()` after 300 s | The exception is raised before the ack, the handshake is not completed | Receive the full batch before reporting the error: wrap the whole callback in try/except, record the first exception, subsequent buckets are only acked and not loaded; after `receive_weights` returns, raise that exception first, then run the integrity check; stop on failure (see below) |

Duplicate tensors within the same bucket are deduplicated by the sender keyed on name; the receiver sees them only once and results are unaffected; the duplicate check can only see cross-bucket cases.

## How to write the integrity check

Completeness is a property of one whole sync, not of one `load_weights` call. Weight-transfer engines (NCCL, IPC) call the model's
loader once per tensor or bucket, so a loader that asserts "every parameter loaded" inside each call rejects every streamed update
(Model A's vLLM loader did, failing each base update with `Parameters not loaded from the checkpoint`), while a loader that simply accepts
partial calls lets a tensor that never arrives keep stale or uninitialized values without an error. Accept partial calls, and check
completeness once, when the sync finishes; fail the reload if anything expected is missing (Model A fixed both halves in vLLM: allow
partial loads from transfer engines, and fail reloads that do not deliver every weight).

- Record received names before `load_weights`; raise when the same name appears a second time within one sync; otherwise duplicate tensors can fill up some layer's count and get it processed early.
- Before finalize, iterate the layerwise state: for layers still pending, `load_numel` must equal the sum of `weight_loader_numel` of all their parameters (excluding placeholders that are never loaded and attention's `q_scale`/`prob_scale`). Use element counts, not names, because the same `w2_weight` is loaded separately by 128 experts; with one expert missing, the name is still present.
- This criterion counts only parameters and differs from vLLM's `load_numel_total` definition (parameters + buffers, including padding); layers with padding will be reported as incomplete, which is a false positive, not a false negative (fail closed). The only layers still pending at this point are attention layers, which only need to load the two k/v scale elements.
- In-place-loaded parameters are checked by name set (second row of the table above). Runtime buffers registered with `register_buffer(persistent=False)` in vLLM are not in `named_parameters()` and need no exemption.
- Set `_refit_failed = True` before the sync starts, and clear it only after finalize runs successfully; any later call that sees it True refuses. After a failure the model may contain meta tensors or garbage layers, and the only option is to restart rollout; this is by design.
- The exemption for non-local experts is unaffected; under EP each rank receives all experts and keeps only the local ones, so "received n tensors" was never a completion criterion.

Acceptance uses a single-GPU probe (real sender/receiver, shm 1 GiB bucket):

| Mode | worker | Sender | Post-sync logprob vs cold load |
|---|---|---|---|
| Normal | Success | Exits normally | Bitwise identical |
| Legacy `input_scale` all placed last | Success | Exits normally | Bitwise identical |
| Duplicate in the same bucket | Success (sender already deduplicated) | Exits normally | Bitwise identical |
| Router bias not sent | `incomplete: [...gate.e_score_correction_bias]` | Exits normally | — |
| One expert's `down_proj.weight` not sent | `incomplete: [...experts.routed_experts]` | Exits normally | — |
| Duplicate across buckets | `received a tensor twice` | Exits normally (HUNG before the fix) | — |

After each kind of failure, the next normal sync must be refused.

## Byte verification: how 291,472 tensors with 0 diff was done

1. **Sender**: the exporter computes sha256 per tensor and writes a manifest (name, shape, dtype, hash); when scales are fixed, each refresh verifies that scale bytes did not change.
2. **Receiver**: inside the worker, sha256 the `state_dict` (parameters + persistent buffers) and record `data_ptr`; require no meta leftovers, unchanged pointers (CUDA Graph addresses valid), state that actually changed for θ1/θ2, and 0 changes in the scale category.
3. **online == cold**: each version's payload is saved as a directory and an engine cold-started from it; state hashes on every DP rank are all identical, and generated tokens and logprobs are identical too. This proves deployed bytes = the bytes of the trainer's quantized payload after vLLM's first-load processing.
4. **Per token**: FP32 logprobs from greedy generation on the resident engine are compared bitwise with the actor eval forward on the same tokens (mismatch=0, nonfinite=0, valid token count > 0).
5. **At least two updates**: θ0→θ1→θ2; θ1 uses an update that is exact on the grid and must change deployed bytes, such as "negate the first half of the output rows of the quantized master"; θ2 does a real optimizer step; with too small an lr the bytes may not move and the check passes vacuously.

Result of the Nemotron-H 100-step delivery: 16 syncs, 291,472 tensors byte-identical, old-logprob vs rollout about 8 million tokens with original-dtype 0 mismatch ([delivery record](../../../cases/nemotron/consistency-case.md#2026-10-lightning-nvfp4-true-on-policy-delivery)).

## Sync time is per tensor, not per byte

A LoRA adapter is small in bytes and large in count: per-expert adapters of a large MoE model are 55,374 tensors (Model A,
rank 32). Every per-tensor step (a `.cpu()`, a hash, a clone, a pinned allocation, a registration) is multiplied by that count, so sync
time is set by the number of tensors and the per-tensor work, not by bandwidth. Model A Megatron's colocated sync took 118–120 s per step:
export 8–16 s, send 51–54 s, vLLM registration 28–31 s. The causes, all per tensor:

| Cost | Cause | Fix |
|---|---|---|
| Gather (FSDP) | One `DTensor.full_tensor()` all-gather per tensor, 55,374 collectives per sync | One all-gather per bucket (256 MiB) |
| Send | 27,264 of the tensors were per-expert copies of shared-outer factors | Send each shared factor once and let the receiver rebuild the aliases from its existing alias table |
| Send and register | A sender manifest (`.cpu()` + sha256, about 20 GiB) on rank 0 and the same hash in every vLLM worker (14.4 s of a 16.6 s `add_lora`; building the LoRA itself took about 0.5 s) | Keep the manifest, but on the first sync of a segment and every N-th sync; align N with the checkpoint interval so the first sync after a resume can be compared byte for byte with the saved one |
| Register | The receiver cloned tensors the packed-IPC consumer had already cloned | One copy |
| Host memory | A GB-scale adapter can sit on the host three times at once: the trainer's send staging, the receiver's staging, each TP worker's swap copy; with the level-1 sleep backup this fills host memory | Gather and send per module group without a full host copy; the receiver copies each bucket into one pinned buffer reused every step; with a single adapter that is never swapped out, drop vLLM's host copy after `add_lora` activates it. When the colocated sync wakes vLLM's weights before exporting, stage the export on the host (Model A Megatron: exporting on the GPU added 28 GiB beside 128 GiB of woken weights and ran out of memory) |

On a full-size bench (one TP8 engine, real sender, IPC and vLLM in-memory load) the per-sync time went from 47.8 s to 10.2 s with
the manifest off and 32.0 s with it on; in production the sync fell to 33–82 s. The remaining variance came from memory pressure, not
transfer: the slowest syncs ran with the GPU at 170 of 184 GiB during receive, or with host memory pressure during export. Pre-stacking
expert LoRA per layer would cut the count from about 28k to 1.4k tensors; it uses a different receive path and was not done.
Every change was accepted by the same rule: receiver bytes equal sender bytes for every public key, and vLLM logprobs within the
same-load repeat noise.

## Compare update and reload stage by stage

Compare sender/receiver policy fingerprints, full weight keys/shards, aliases, derived weights and scales. Check whether storage addresses captured by Graph are stable, whether the update writes in place or replaces the tensor, and whether broadcast/transposed/fused weights are refreshed. Compare stage by stage: cold load → same-weight reload → real update → sync → sleep/wake (if used) → Graph replay → reload in a new process. Localize by the first failing stage; a new process passing does not prove the old process's cache/Graph has been fixed.

## Other constraints

- Every update must send the full checkpoint-format state completely; layers not fully sent are treated as partial and processed into garbage ([precision P14](../precision-quant-qat/03-pitfalls.md#p14-partially-updating-fused-parameters)).
- `export_dtype=null`, otherwise FP8/FP32 scales are converted to BF16.
- Immediate copy rather than clone: tensors are copied in before the loader returns, only names are recorded; deferred + reused IPC buffers read overwritten data (NaN has occurred).
- `gc.collect()` + `empty_cache()` before and after sync, so stale temporary tensors and the cuMem pool do not compete for GPU memory.
- GPU memory related to legacy formats/placeholders: when the W4A16 placeholder is counted in the layer total, one sync accumulates an extra 16.5 GiB; see [01 case-map](01-case-map.md#internal-bug--rfc-class).

Sources: [ds4form r1 fixes](../../../cases/nemotron/reports/cross-review/ds4form-r1-fixes.md), [r2 fixes](../../../cases/nemotron/reports/cross-review/ds4form-r2-fixes.md), [online sync design](../../../cases/nemotron/reports/nvfp4-online-sync.md), [W4A16 reload PR body](../../../cases/nemotron/reports/pr-bodies/vllm-split/2-modelopt-w4a16-reload.md).
