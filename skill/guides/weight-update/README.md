# Weight Update and Policy Versions

The single source for mechanisms and failure taxonomy is the public RFC [vllm#48312 Weight Reload Correctness for RL](https://github.com/vllm-project/vllm/issues/48312) (seven classes: storage identity, runtime value refresh, loader lifecycle, state preservation, parameter routing/sharding, name mapping, post-update cache coherence) and its sibling RFCs: [#48310](https://github.com/vllm-project/vllm/issues/48310) sleep/wake, [#54477](https://github.com/vllm-project/vllm/issues/54477) reload refactor, [#48478](https://github.com/vllm-project/vllm/issues/48478) fail-closed storage contract, [#48305](https://github.com/vllm-project/vllm/issues/48305) train-inference consistency. Tools (trainer-free publisher, lifecycle wrapper, fixed-token oracle, provenance sidecar) live in a companion verifier toolkit (not included in this repo). This guide does not restate them; it holds only what is extra from our own runs.

Boundary with the [precision guide](../precision-quant-qat/README.md): dump the bytes at the sender and compare with expected values; a mismatch is a conversion-chain problem (precision); a match while inference is still wrong is a transport/loading/derived-state problem (here).

## Find by symptom

| Symptom | Go to |
|---|---|
| Garbled output / silent errors after refit | RFC classes 2 and 7; [precision pitfalls P13–P16](../precision-quant-qat/03-pitfalls.md#export-and-receive) |
| A–A outputs identical but unsure whether the update happened | [02 policy versions](02-policy-versions.md): report master change, payload content ID, and server-side load proof separately |
| Only the CUDA Graph path is wrong | RFC class 1; [01 case-map: Model F graph reload, MoE kernel dead references](01-case-map.md#internal-bug--rfc-class) |
| Wrong only from the second update | RFC class 2 (derived values folded repeatedly); [01 #57954](01-case-map.md#57954-nvfp4-reload-stale-derived-values) |
| Wrong after sleep/wake | #48310; after Mamba2 wake, restore every state in each layer's declared state list |
| Sync hangs and never finishes / late scale crashes / missing sends go unnoticed | [03 sync robustness](03-sync-robustness.md) |
| A LoRA sync takes a minute or more although the adapter is small | [03 sync time is per tensor](03-sync-robustness.md#sync-time-is-per-tensor-not-per-byte) |
| Received n tensors; does that mean the layer is complete | [01 what cannot count as completion](01-case-map.md#what-cannot-count-as-completion) |
| Cold start is correct, online sync on resident workers is wrong | [02 policy versions](02-policy-versions.md#this-mistake-and-the-delivery-boundary) |

## Key rules

1. Completion is defined by the manifest/schema or a record of one valid load; the number of received buckets, physical numel, or number of named parameters cannot count as "this layer is complete", and expected cannot be defined by this run's observed.
2. Record a receipt only after the loader actually accepts and applies; a return of `False` does not count, `None` is judged by the specific loader's contract; missing/duplicate must have explicit failure semantics, not keep serving on a warning.
3. Transport complete → loader applied → post-load derived state → sleep/wake restore → old graph replay → output oracle; a later step does not replace an earlier one; neither HTTP 200 nor identical A–A output is evidence that the update happened.
4. CUDA Graph and kernel objects read the original storage: updates must `copy_` into the original address, derived values are recomputed on every update, and clearing/recapturing the graph is not a fix.
5. The minimum granularity of a partial update is all shards of one fused parameter; the first update after level 2 sleep must be full.
6. Reload acceptance does at least two updates (on the first, the stale derived value happens to equal the cold-load value); report master change, payload content ID and server-side load proof separately.
7. When the no-update repeat already shows the same kind of floating-point noise and meets the baseline noise rule, stop the determinism investigation; do not change the model/kernel just to support A–A.

## Terminology

- **layerwise reload**: stage received tensors per layer/fusion group, post-process once all have arrived, then free the staging; not every load needs it.
- **load plan**: "what each layer should receive", recorded on the first valid load; adopting a historical plan requires verifying compatibility with the current structure, sharding and load method.
- **receipt**: rank-local record that the loader actually accepted and applied a tensor; input to completion decisions and integrity checks.
- **post-load derived state**: values computed from weights and used by kernels (packed layout, folded scale/alpha/clamp, KV scale cache); must be refreshed after every update.
- **storage identity**: a tensor's underlying address; CUDA Graphs and kernel objects bind to it, not to the Python attribute.
- **level 1 / level 2 sleep**: vLLM sleep modes; level 2 discards weight storage, and contents must be restored after wake.
- **A–A**: update with the same weights twice and compare whether outputs are identical; can only prove determinism, not that the update happened.
- **θk / policy version**: the policy after the k-th update; master, exported payload and server-side load each have their own ID; see [02](02-policy-versions.md).

## Files

| File | Content |
|---|---|
| [01-case-map.md](01-case-map.md) | One sentence per each of the seven classes; internal bug → RFC class table (#57954, FlashInfer KV scale, partial update of fused params, `FusedMoEWithLoRA` no-op, MIXED_PRECISION taking the wrong branch, W4A16 `input_scale` placeholder and late arrival, MoE kernel dead references, Mamba2 wake, Model C MXFP4, Model F graph reload, completion contract), which have been upstreamed and which are internal only; mechanism criteria (why layer-level completion exists, what cannot count as completion, what comes after completion, restore responsibility and minimal checks for five kinds of state); minimal localization order |
| [02-policy-versions.md](02-policy-versions.md) | Four artifacts of θk, three IDs not mixed, the minimum two-update threshold, cold start does not replace online sync |
| [03-sync-robustness.md](03-sync-robustness.md) | Legacy-format `input_scale` arriving late → collect by name; router bias not in the integrity check → generate expected from the manifest; duplicate sends hang → receive the full batch before reporting an error, stop on failure; method for checking 291,472 tensors with zero diff per tensor; completeness checked per sync, not per loader call; sync time set by tensor count (Model A 55,374 adapter tensors) |

## Candidates to upstream into the RFC

FlashInfer attention lazily caching KV scale (dummy start → refit does not invalidate) appears in the RFC table only as the level-2 sleep version from #57163; this item is worth adding a row for (internal vLLM fork PR). The two items W4A16 `input_scale` placeholder counted in the layer total and legacy `input_scale` arriving late are already in fork PR split 2, targeting upstream.
