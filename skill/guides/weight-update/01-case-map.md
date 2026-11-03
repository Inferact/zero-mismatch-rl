# Weight-update case map: internal bugs mapped to RFC classes

The single source for mechanisms and failure taxonomy is [vllm#48312 Weight Reload Correctness for RL](https://github.com/vllm-project/vllm/issues/48312). This page does not restate the RFC; it does three things: writes each of the seven classes as one sentence, maps internally encountered bugs to classes and marks whether they have been upstreamed, and keeps a short minimal localization order.

## The seven classes in one sentence each

| # | Class | One sentence |
|---|---|---|
| 1 | Storage identity | CUDA Graphs and kernel objects read the original storage address; reload replaces the attribute with a numerically identical new tensor, and the old address does not follow. |
| 2 | Runtime value refresh | Post-load derived values (packed layout, folded scale/alpha/clamp, aliases held by kernels) must be recomputed after every update; "return if already initialized" leaves stale values. |
| 3 | Loader lifecycle | The "layer complete" criterion of layerwise reload, uninitialized GPU memory produced by materialize, and the finalize fallback for partial layers determine whether missing or partial sends silently turn into garbage. |
| 4 | State preservation | Storage discarded by sleep level 2, non-persistent buffers, and draft/MTP shared tensors need an explicit owner responsible for restoring them; frozen parameters not updated by sync can be kept by vLLM itself via `--sleep-preserve-parameter-names` ([02 §5.4](../checkpoint-resume-memory/02-host-memory.md#54-sleep-level-1-and-level-2)). |
| 5 | Parameter routing/sharding | Non-local experts under EP/TP, shards of fused parameters, and `use_global_sf` scale allocation determine whether "received n tensors" can count as completion. |
| 6 | Name mapping | In the three-level mapping checkpoint name → vLLM parameter name → kernel field, guessing at any level by suffix or class name drops layers or takes the wrong branch. |
| 7 | Post-update cache coherence | Values lazily cached on the first forward by attention impls, prefix cache, KV scale, etc. must be invalidated after an update. |

Sibling RFCs: [#48310](https://github.com/vllm-project/vllm/issues/48310) sleep/wake, [#54477](https://github.com/vllm-project/vllm/issues/54477) reload refactor, [#48478](https://github.com/vllm-project/vllm/issues/48478) fail-closed storage contract, [#48305](https://github.com/vllm-project/vllm/issues/48305) train-inference consistency.

## Internal bug → RFC class

| bug | Symptom | Class | Fix | Status | Evidence |
|---|---|---|---|---|---|
| NVFP4 reload stale derived values | After reloading weights ×1.1/×0.9, `g1_scale_c` uses the previous round's alpha, relative RMS error 9.58%/22.49%; clamp/beta folded again every round | 2 | See the section below | Upstream PR [#57954](https://github.com/vllm-project/vllm/pull/57954) (head eb59f582e7) | [Precision pitfalls cross-reference](../precision-quant-qat/03-pitfalls.md#cross-reference-derived-value-and-cache-bugs) |
| FlashInfer attention lazily caches KV scale | After starting with `load_format=dummy`, refit writes the real k/v scales, but `bmm1_scale`/`bmm2_scale` are still the cache from scale=1, output is garbled (prompt NLL about 11); yet all parameter checksums match a normal load | 7 | Invalidate the cache at reload finalize, or compute from the layer every step; workaround `load_format=auto` | internal vLLM fork PR, not upstreamed; the RFC table has only the level-2 sleep version from #57163; this item is an upstreaming candidate | [Precision P15](../precision-quant-qat/03-pitfalls.md#p15-flashinfer-attention-lazily-caches-kv-scales), [bug report](../../../cases/nemotron/reports/vllm-flashinfer-kvscale-reload-bug.md) |
| Partial update of fused params | Sending only the k/v shards of qkv leaves the q rows as uninitialized GPU memory from materialize; the model silently becomes garbage, exposed only by the spec-decode acceptance rate | 3, 5 | Minimum granularity is all shards of one fused parameter, safest is the whole layer; the first update after level 2 sleep must be full | Internal rule | [Precision P14](../precision-quant-qat/03-pitfalls.md#p14-partially-updating-fused-parameters) |
| `FusedMoEWithLoRA` replaces post-load with a no-op | Drift after NVFP4 identity reload, lifecycle same as BF16 | 2, 3 | Cannot simply delegate: also update the `_moe_kernel`/context held by LoRA and satisfy the Graph storage contract | Internal (Model B) | [Precision P16](../precision-quant-qat/03-pitfalls.md#p16-post-load-replaced-by-a-no-op); Model B support audit P0-3 (not included) |
| Layerwise reload on LoRA-wrapped fused MoE | `weight_loader` looks for "ModelOpt" by class name, the wrapped `FusedMoEModularMethod` does not match, the per-tensor scale path is skipped, NVFP4 scales load wrongly; the wrapper has no `process_weights_after_loading`, so reloaded tensors are copied into kernel tensors unconverted | 6, 2 | The loader checks the wrapped quant method; hand post-load to it, provided it does not rebuild the kernel on reload (otherwise see the previous row) | internal fork PR | Model N 2026-10 G1 base reload validation |
| Model's own `load_weights` calls `self.weight_loader` directly | Fused QKV/O-gate projections bypass the wrapper that layerwise reload attaches to the parameter; the Graph + level 2 lane reports `Failed to load weights` | 3, 6 | Dispatch via `param.weight_loader` | internal fork PR | Model N 2026-10 G1 base reload validation |
| Non-module LoRA state after level 2 | After base sync by a self-written receiver, weights are bitwise correct but the adapter has no effect: MoE LoRA's `adapter_enabled` and `LogitsProcessorWithLoRA.sharded_to_full_mapping_gpu` are not in the module tree and are garbage after level 2 | 4 | After base sync call `model_runner.reset_lora_state()` (vLLM's own `reload_weights` and weight transfer both call it) | Internal (Model N, VERL receiver) | Model N 2026-10 G1 level 2 validation |
| ModelOpt `MIXED_PRECISION` falls into plain `load_weights` | The verl sync patch identifies the quant config by class name; MIXED takes the unquantized branch, checkpoint-format tensors are written into the already-processed kernel layout, then a non-idempotent post-load runs again | 6, 2 | Use the layerwise reload lifecycle; the sender sends in checkpoint format | Internal extension + fork PR | [Precision P13](../precision-quant-qat/03-pitfalls.md#p13-quantization-config-identified-by-class-name-mixed-falls-through-to-plain-load_weights), [online sync](../../../cases/nemotron/reports/nvfp4-online-sync.md) |
| W4A16 `input_scale` placeholder counted in the layer total | The placeholder is never loaded, so every W4A16 layer hangs until finalize; one sync accumulates an extra 16.5 GiB of clones | 3 | Placeholder `weight_loader_numel = 0`, process the layer as soon as it is complete | fork PR split 2 | [PR body](../../../cases/nemotron/reports/pr-bodies/vllm-split/2-modelopt-w4a16-reload.md), [r1 fixes](../../../cases/nemotron/reports/cross-review/ds4form-r1-fixes.md) |
| W4A16 `input_scale` from old checkpoints arriving late | After the previous fix, the layer is processed early and `input_scale` deleted; when it arrives in a later bucket, "no module or parameter named …" is raised | 6, 3 | Configs containing W4A16 layers add `.input_scale` to `_ignore_unexpected_suffixes` (only takes effect when the parameter does not exist; FP8 layers still wait for it) | fork PR split 2 | [r2 fixes](../../../cases/nemotron/reports/cross-review/ds4form-r2-fixes.md), [03 sync robustness](03-sync-robustness.md) |
| MoE kernel rebuilt on every sync | The new quant config references new scales while the layer swaps back to the original storage; a resident +1.75 GiB of dead references | 1, 2 | Kernel reads scales via the layer; scale/alpha in the descriptor set to None | fork PR | [r1 fixes](../../../cases/nemotron/reports/cross-review/ds4form-r1-fixes.md) |
| Layers that received no tensors after dummy start + level 2 sleep | Finalize puts the old kernel tensors back, contents are garbage | 4, 3 | First update is full; integrity check fails closed | Internal rule | [online sync §5](../../../cases/nemotron/reports/nvfp4-online-sync.md#5-open-risks) |
| Mamba2 restores only the conv/ssm two states after wake | BI=1 exact replay needs 5 states; inconsistent after a prefix cache hit | 4 | Copy every state in each layer's declared state list | fork PR | [Nemotron Mamba prefix caching](../../../cases/nemotron/consistency-case.md#mamba-prefix-caching-align-mode) |
| Model C MXFP4 derived state, draft wake | After reload, MXFP4 derived scales and the draft buffer are not refreshed | 1, 4 | Derived state rebuilt with the reload lifecycle; parts shared by draft and target have a revision contract | Internal | Model C timeline (not included) |
| Model F graph reload | Still drifts with all registered state identical: three faults — unregistered state, kernel storage, and padding | 1, 4 | Check storage actually reachable from the graph, not just attributes | Internal | Model F CUDA Graph reload report (not included) |
| DeepSeek-V4 completion contract | The number of received buckets, physical numel, or number of named parameters cannot count as "this layer is complete" | 3, 5 | Completion is defined by the manifest/schema or a record of one valid load; rank-local receipts record only what the loader actually accepted | Internal | [DeepSeek-V4 history audit](../../../cases/deepseek-v4-flash/reports/history-audit.md) |

### Mechanism criterion: why layer-level completion exists

Some post-processing needs to see a group of weights at once, e.g. fusing gate/up, computing a shared scale, reordering kernel layout. So receiving one parameter does not necessarily allow producing the final runtime state immediately; accumulate first, then process. Staging the whole model has too high a peak, so the processing boundary is narrowed to a layer or fusion group.

```text
a group of inputs jointly determines the post-processing result
→ all inputs in the group must be present
→ stage per layer/group, avoid accumulating the whole model
→ an explicit "complete" criterion is needed
→ free that group's staging after processing
```

Not every load must be layerwise, and not every layerwise load must cache the first load plan. When each parameter can be processed independently, no accumulation is needed; which mechanism to use depends on the real post-processing dependencies, not on the framework's name. If a historical plan is adopted, verify that it is compatible with the current structure, sharding and load method.

Declaring completion too early runs post-processing on partial weights; finalize's fallback processing also cannot prove inputs are complete. Behavior on missing/duplicate inputs must have explicit failure semantics and cannot keep serving on a warning alone (the concrete fail-closed design is in [03](03-sync-robustness.md)).

### What cannot count as completion

| Object | Why it cannot count as "this layer is complete" |
|---|---|
| Number of tensors/buckets received over the network | This rank rejects non-local experts, or one input is mapped to multiple targets |
| Physical numel of the runtime parameter | Includes padding or derived data that the checkpoint does not send |
| Number of named parameters | Fused parameters can receive multiple shards/experts; aliases can point to the same storage |
| Loader application | Closest to the actual write, but identity, rejection semantics and the expected input set still need defining |

Expected cannot be defined by this run's observed; dummy initialization has no real load record and cannot count as an established plan. Record a receipt only after the loader actually accepts and applies: interfaces where returning `False` means rejection cannot be counted, and not every `None` can be treated as failure either; judge by the specific loader's return contract. Identify duplicate applications by identity; equal totals can still be "one duplicated, one missing". When staging piles up, print the expected / accepted / missing identities of incomplete layers, rejection reasons, and staged bytes: if everything missing is padding, derived state or non-local experts, the completion contract is wrong; if it is just the send order interleaving across layers, the contract may be right but multiple layers are staged at once. The `RoutedExperts` large-allocation warning by itself cannot distinguish the two.

### What comes after completion

```text
transport complete → loader applied → post-load derived state restored
→ sleep/wake state restored → existing graph replay → output oracle
```

A later step does not replace an earlier one. HTTP 200 is not evidence of load completion; identical A–A output does not prove the update happened; real transport and application evidence must be kept as well.

| State | Restore responsibility | Minimal check |
|---|---|---|
| checkpoint parameter / persistent buffer | Loader, restore path | Name, logical value, shape/dtype |
| kernel-owned transformed weight / scale | Post-load conversion | Values and storage actually referenced by the kernel |
| Non-persistent buffer / unregistered CUDA tensor / metadata | Explicit rebuild or in-place refresh entry point | Owner, call timing, value after update |
| shared target/draft tensor | Shared owner and revision contract | References on both sides, hash of the shared part, actual update responsibility |
| padding / workspace | The operator consuming it or runtime initialization | Distinguish the valid range from padding that is actually read |

`named_parameters()` and `named_buffers()` do not enumerate all kernel-reachable state; identical registered lists only rule out differences within those lists.

Paths captured by CUDA Graph keep accessing the original addresses; replacing a Python attribute with a numerically identical new tensor does not mean the old graph switches to the new address, so check the storage actually reachable from the graph. An unchanged address does not mean the value is correct either: after level-2 discard, the contents of the original storage still need restoring; a post-load written as "return if already initialized" skips weight-derived state that needs refreshing; distinguish one-time structure creation from numeric refresh that must run on every update. Hook names and call timing are also part of the contract: a builder without a real caller does not count as restoration; building early before the bucket is complete does not count as correct either. A shared module must not be staged separately by two reload owners, or one side will wait for inputs it will never receive. Do not use clear/invalidate/recapture graph as a correctness fix; what must be proven is that the old graph still reads the correct state after the update.

## #57954: NVFP4 reload stale derived values

- Cause: [#50074](https://github.com/vllm-project/vllm/pull/50074) made online NVFP4 reload reuse the same MoE kernel object. Layerwise reload copies new parameters back into the original storage only **after** post-processing, while TRTLLM computes derived values **during** post-processing using old tensors held by the config, so `g1_scale_c` uses the previous round's alpha.
- Second problem: clamp/beta are re-folded on top of already-folded values on every reload; the n-th time = clamp/(g0·g1·…·gn), with g about 4e-4, amplifying about 2500× per round.
- Fix: before post-processing, `copy_` the new alpha into the original storage (cannot reassign: both CUDA Graph and "discard B at the end" require the config to keep pointing at A); clamp/beta store the original pre-fold values and fold from the originals every time.
- Leftovers: under EPLB, derived parameters become orphans; `copy_` happens before multiplying by the input scale (on the online path the input scale is always 1, so no effect for now).
- Why "wrong only from the second update": on the first reload the stale alpha happens to equal the cold-load value, so derived values are correct; from the second on the old value is stale. So reload acceptance does at least two updates.

## Minimal localization order

1. Fix input, batch, backend and graph config; first do a no-update repeat.
2. Check the actual image/overlay, checkpoint and load receipts, ruling out running the wrong code or missing inputs.
3. Compare values and addresses of the relevant parameters/buffers; if still unexplained, find the first layer divergence, then narrow to component and internal state. Checksums identical but output bad → state is cached on the impl or kernel object; use per-layer forward hooks to find the first divergence.
4. Probes label phase, active rows and graph slots; do not read unmapped allocations, do not treat inactive rows as valid output. Add observation only near the divergence.
5. After the minimal fix, retest the second update, production graph and MTP where applicable; a combined experiment passing does not replace trigger evidence for each individual fix.

When the no-update repeat already shows the same kind of floating-point noise and meets the baseline noise rule in the protocol, stop the dedicated determinism investigation; do not modify the model/kernel just to support A–A. Scripts for the lifecycle wrapper, fixed-token oracle and provenance sidecar are in a companion state-transition verifier toolkit (not included in this repo). Execution order and pass criteria follow the serving validation protocol of the day-0 support workflow (not included).

## Cases and boundaries

- Model C timeline (not included): MXFP4 derived state and draft wake.
- [DeepSeek-V4 history audit](../../../cases/deepseek-v4-flash/reports/history-audit.md): completion contract, rank-local receipts, and attribution boundaries of control experiments.
- Model F CUDA Graph reload report (not included): different faults from unregistered state, kernel storage and padding; includes validation evidence for the final fixed stack.
- [Nemotron online sync](../../../cases/nemotron/reports/nvfp4-online-sync.md): how the layerwise reload lifecycle is wrapped in the verl extension, and why immediate copy is more correct than clone.
