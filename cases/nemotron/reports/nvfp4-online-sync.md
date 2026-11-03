# NVFP4 online weight sync: verl_mlite layerwise-reload extension

## 0. Update (2026-10-02): switched to immediate copy

**Trigger:** a design review decided to keep verl's native per-bucket immediate loading and not to require layerwise reload by default. verl upstream points the same way: #7136 (FP8 resync via layerwise) was closed, with the closing note "no longer using `reload_weights` after #7224".

**Problem:** the original implementation kept vLLM layerwise reload's default deferred mode. Until all tensors of a layer arrive, it only stores the loader arguments and does not copy, so every bucket had to be cloned (a series of internal commits): first, the clone takes extra GPU memory; second, vLLM's deferred design itself relies on "the incoming tensor outlives `load_weights`".

**Change:**
- vLLM (an internal fork commit): `initialize_layerwise_reload(model, copy_immediately=True)`. The layer is materialized in checkpoint format the first time it is loaded; each tensor is copied in before the loader returns, and only its name is recorded. Once the layer is complete it is processed as usual and written back to the original kernel storage; attention scales are handled at finalize.
- Upstream vLLM precedent: `weight_transfer/sharded_rdt_engine.py` also reuses the layerwise state (`materialize_layer`, `_copy_and_restore_kernel_tensors`) and scatter-copies data as soon as it arrives. Humming's kernel preservation depends on `LAYERWISE_INFO.kernel_tensors` (`modelopt.py:1038`), so this state is reused instead of building a separate one.
- Megatron (an internal commit): the extension drops the clone; `session.load` calls the model's native `load_weights` tensor by tensor, with no deferred references. The fail-closed audit is unchanged.

**Evidence:**
- vLLM unit tests: 2 new cases (with a reused buffer, the result is bitwise identical to deferred mode with fresh tensors and the kernel storage address is unchanged; immediate loading of an attention layer), 8 passed on CPU.
- Negative confirmation: deferred mode with buffer reuse produces NaN; immediate mode is correct.
- The extension unit tests and the GPU sync gate must be rerun once an image containing the vLLM immediate-copy commit is built. The old receipts (r2e, cr002, on an earlier image) are kept only as evidence for the clone version.


- Megatron commit: an internal commit on the mLite-on-dev branch, not pushed
- vLLM PR checkout and verl checkout: internal fix branches
- GPU gate script: `proxy_sync_gate.py` (**not run**)
- Status: code written, ruff passes; neither the CPU unit tests nor the GPU gate has been run. Pending cross-review.

## 1. Conclusion

Upstream `vLLMColocateWorkerExtension` takes the "unquantized" branch when it meets `ModelOptMixedPrecisionConfig`. The new subclass `verl_mlite.rollout.layerwise_reload.LayerwiseReloadWorkerExtension` overrides `update_weights_from_ipc`: every sync is wrapped in vLLM's own layerwise reload lifecycle. verl needs no changes; the recipe just sets `ROLLOUT_WORKER_EXTENSION_CLS=verl_mlite.rollout.layerwise_reload.LayerwiseReloadWorkerExtension`. Note that this variable currently defaults to empty in the recipe (`run_nemotron_h_lightning_nvfp4_true_on_policy_megatron.sh:98,171-172`), and if it is empty, it falls back to upstream's wrong path.

## 2. Required update semantics

### 2.1 Why upstream is wrong

- Quantization detection only recognizes `Fp8Config`/ascend MX (`verl/utils/vllm/vllm_quant_utils.py:83-92`), compressed-tensors QAT, and `ModelOptNvFp4Config` (`vllm_rollout/utils.py:194-206`). MIXED_PRECISION matches none of the three.
- So it takes the unquantized branch: `stage_unquantized_moe_params` (`utils.py:299-307`) → `model.load_weights` writes checkpoint-format tensors into the already-processed Humming layout (`utils.py:414-423`) → after all buckets arrive, `process_weights_after_loading` runs again (`utils.py:359-365`). This step is not idempotent.

### 2.2 What the trainer sends (Nemotron export)

- Output of `NemotronExport.iter_export_tensors` (`experimental/lite/megatron/lite/model/nemotron_h/checkpoint.py:96-123`):
  - Unquantized tensors: BF16/FP32 dense, including Mamba, norm, router, and attention's `k_proj.k_scale`/`v_proj.v_scale` (`:107-110`, `hf_tensor_views` `:159-184`);
  - NVFP4 linear: `weight` (uint8 packed), `weight_scale` (FP8 E4M3 block), `weight_scale_2` (FP32) (`quantization.py:236-245,270-278`);
  - FP8 linear: `weight` (FP8), `weight_scale`, `input_scale` (FP32) (`fp8_training.py:89-97,146-153`);
  - Routed experts: each expert's `up_proj/down_proj.{weight,weight_scale,weight_scale_2}` (`nvfp4_experts.py:185-196,227-238`).
- Scales are fixed: NVFP4/FP8 STE does not recalibrate, and each expert refresh checks that the scale bytes did not change (`nvfp4_experts.py:197-219`). So what really changes on each update is the packed weight, and the scale bytes should equal the previous version, but scales are still sent with the payload every time.
- The export yields the complete, unsharded HF state, one copy on every participating rank (`primitive/ckpt/hf_weights.py:1369-1386`). Every rank of the EP4 rollout receives all 128 experts, and the vLLM loader keeps only the local 32.
- dtype: the recipe sets `actor.engine.export_dtype=null` (recipe `:237`); otherwise `_cast_export_tensor` would cast the FP8/FP32 scales to BF16.
- After the optimizer step, `post_optimizer_step_hook` re-encodes the deployment (`nemotron_h/protocol.py:203-205,291`). The export must happen after that.

### 2.3 What must happen on the rollout worker (in order)

1. **initialize**: `initialize_layerwise_reload(model)` (`vllm/model_executor/model_loader/reload/layerwise.py:81-116`).
   - Stores the current kernel tensors (i.e. the original storage referenced by CUDA graphs) in `info.kernel_tensors` (`:108`).
   - Restores each layer's parameters to meta tensors according to the checkpoint format recorded at build time by `record_metadata_for_reloading` (`model_loader/utils.py:96,128`) (`:113`), then wraps them with `online_process_loader` (`:116`).
   - vLLM's own `reload_weights(is_checkpoint_format=True)` uses the same three steps (`v1/worker/gpu_model_runner.py:5544-5549`), and so does the IPC/NCCL transfer engine (`distributed/weight_transfer/ipc_engine.py:162-175`); verl's delta path does the same (`delta_weight_transfer.py:281-287,307-317`).
2. **load**: hand each bucket as is to `model.load_weights` (Nemotron uses `AutoWeightsLoader` + `hf_to_vllm_mapper`, `models/nemotron_h.py:1035-1037`; unknown names raise immediately). Buffers also go this way instead of verl's `apply_buffer_updates`; otherwise they would be copied into a meta buffer and silently dropped.
   - **Must clone**: until the layer is complete, the loader arguments stay in `info.loaded_weights` (`layerwise.py:184`); and the receiver reuses the same IPC buffer for every bucket (`bucketed_weight_transfer.py:295`). Without a clone, a layer that spans buckets reads overwritten data when it is processed. The comment in the old DS4 patch records the same pitfall (`verl_mlite/compat.py:873-879`).
3. **per-layer process**: once all elements of a layer have arrived (`layerwise.py:215-217`), `_layerwise_process` runs (`:336-378`): materialize (`torch.empty_strided`, contents uninitialized, `meta.py:49-61`) → replay with the original loader → `quant_method.process_weights_after_loading` (Humming repacks here, `oracle/nvfp4.py:438-466`) → `_copy_and_restore_kernel_tensors` **copies the result back into the original storage** (`:395-411`). The addresses do not change, so **no CUDA graph recapture is needed**.
   - Humming MoE: the PR branch skips `_build_moe_kernel` on reload, keeps the original kernel object and the scale aliases it holds, and checks that the storage aliases, layout and Humming recipe have not changed (`layers/quantization/modelopt.py:1028-1110`; switch at `models/nemotron_h.py:349`).
   - The `w13/w2_input_scale` parameters of W4A16 experts are created but never loaded (`modelopt.py:957-972`), so MoE layers are never complete and wait for finalize's "Delayed processing" branch (`layerwise.py:272-274`). This is by design.
4. **finalize**: `finalize_layerwise_reload(model, model_config)` (`layerwise.py:225-287`). Processes all incomplete layers, attention last (`:278-281`).
   - Attention layers with checkpoint KV scales: rebuild scale → replay → process → write back (`:305-333`). Nemotron uses `_preserve_checkpoint_scale_dtype` to keep the scale rounding of the first load (`:317-323`, `nemotron_h.py:596`).
   - Layers that received no tensor at all get their old kernel tensors back (`:254-266`). If that happens after a level 2 sleep, those tensors contain garbage.
5. **Afterwards**: verl calls `clear_kv_cache`/`reset_prefix_cache` (`vllm_rollout.py:247-248`). In hybrid mode the order is sleep(level 2) → wake(weights) → sync → wake(kv_cache) (`vllm_async_server.py:855-905,1414-1442`).
6. **Second and later updates**: `info.reset()` runs after each layer is processed, and the next initialize fetches `kernel_tensors` again (`layerwise.py:103-108`), so the lifecycle can be re-entered. Humming's preservation checks rerun every time. Every update must send the full checkpoint-format state; "send only changed tensors" is not supported, because layers that are not fully sent are treated as partial.

## 3. Implementation (an internal Megatron commit)

File `experimental/lite/examples/verl/verl_mlite/rollout/layerwise_reload.py`:

- `require_layerwise_reload_support`: checked in `__new__`; if the quant config is not `ModelOptMixedPrecisionConfig`, it raises `NotImplementedError` and the worker does not start, i.e. fail closed.
- `update_weights_from_ipc` overrides the base-class method, with the signature matching verl's call (`vllm_rollout.py:229-233`):
  - Refuses LoRA, MTP drafter sync, or a previous failed reload outright;
  - Inside `set_current_vllm_config` it runs `initialize_layerwise_reload` → receives data with verl's public `BucketedWeightReceiver` → calls `load_checkpoint_bucket` per bucket (`drop_tied_alias_updates` + clone + `model.load_weights`) → `finalize_layerwise_reload`;
  - When the callback errors, it first completes the bucket handshake and then raises (following the delta path), so the sender does not hang;
  - After a failure it sets `_layerwise_reload_failed` and refuses all later syncs, because meta tensors may still remain in the model;
  - At the end it runs `gc.collect()` and `empty_cache()` to release the now-invalid clones so they do not compete with cuMem for GPU memory.
- Uses only verl's public classes and functions plus a subclass override; no monkeypatching of verl or vLLM. `monkey_patch_model` is still called by verl; its MoE loader patch has no effect on NemotronH (`verl/utils/vllm/patch.py:17-72,120`).
- Unit test `experimental/lite/tests/unit/examples/test_verl_layerwise_reload_extension.py` (marked `optional`, needs vllm and verl):
  - Uses the real vLLM reload API on CPU to load one layer across two reused buckets, two rounds in a row, asserting the values are correct and the storage pointers unchanged;
  - Asserts that an unsupported quant config is rejected.
  - No verl environment on this machine, **not run**.

## 4. GPU gate (`proxy_sync_gate.py`, not run)

- **Layout**: single process (`torchrun --nproc-per-node 1`). The mLite actor (PP1 proxy) runs on cuda:0; an `AsyncLLM` frontend in the same process starts DP4/EP4 engine core subprocesses per `serve-args-bi1-proxy.json` (cuda:0-3, cuda:0 shared with the actor to simulate colocation). torchrun environment variables are stripped before starting the engine.
  - Launch args: `--load-format dummy --worker-extension-cls <new class> --enable-sleep-mode`.
  - The sync matches verl: `collective_rpc("update_weights_from_ipc")` (the DPLB client broadcasts to all DP engines), with one `BucketedWeightSender` (CUDA IPC) per rollout rank. The sender calls `set_device(rank)` in a thread, and the zmq address matches `_get_zmq_handle`.
  - **Fallback two-process layout** (if the actor's in-process vLLM runtime conflicts with AsyncLLM): process A runs only AsyncLLM and the sender; process B (torchrun) runs the actor. B writes the payload to safetensors in `/dev/shm`, and A reads it and pushes it via IPC to the resident engine. Actor scores are passed back and forth through files. The disk is only used to move data; the engine does not restart.
- **Update sequence** (all pushed to the same resident engine):
  1. θ0: the actor exports the checkpoint state, replacing the dummy weights. Corresponds to verl's first sync.
  2. θ1: negate the first half of the output rows of each quantized master (NVFP4 linear, FP8 linear, NVFP4 experts), then refresh. Negation is exact on the FP4/FP8 grids and the scales are fixed, so the deployed bytes must change.
  3. θ2: one real optimizer step on θ1's rollout, default `--lr 1e-3`. With too small an lr the bytes may not move at all, and the check would pass vacuously.
  - By default each push is preceded by sleep(level 2) → wake(weights) and followed by wake(kv_cache), matching verl HYBRID.
- **Checks per update** (written to `receipt.json`):
  - payload: scale dtype not downcast; count of tensors changed vs the previous version, per category. θ1 requires a change in every quantized weight category, θ2 requires at least one change, and the scale categories should be 0.
  - rollout: inside the worker, sha256 over `state_dict` (parameters + persistent buffers) and record `data_ptr`. Requires no meta leftovers, unchanged pointers (CUDA graph addresses valid), and that the θ1/θ2 state actually changed.
  - alignment: the resident engine does greedy generation (graph path) to get FP32 logprobs, compared **bitwise** with the actor's eval forward logprobs on the same tokens (mismatch=0, nonfinite=0, valid tokens > 0).
  - The byte diff between the θ0 payload and the `/model` checkpoint is recorded for information only.
  - Cold-start reference: at the end, cold-start an engine from a directory saved from each payload version; every DP rank's state hash must be identical, and the generated tokens and logprobs must all be identical, i.e. online == cold. This proves "deployed bytes = the bytes of the actor's quantized payload after vLLM's first-load processing".

## 5. Open risks

1. **Level 2 sleep and the Humming preservation path**: the preservation path does not rebuild the kernel. If tensors held by the kernel but not registered on the layer (e.g. locks, workspaces) are allocated in the "weights" cuMem pool, their contents are lost after level 2 sleep, and reload does not restore them. A hang or wrong results may occur here. The gate tests level 2 by default precisely to expose this.
2. **CUDA graph / unregistered derived state**: correctness relies on "all derived values are registered tensors and are copied back into the original storage". If a linear kernel (e.g. `NemotronSharedNvFp4LinearKernel`, FP8 `QuantFP8` on-device quantization under BI, `nemotron_h.py:97-117`) stores derived tensors on the kernel object in `process_weights_after_loading`, or replaces the object, then after reload the graph still reads old data, or reads temporary tensors that have already been freed. The gate uses the online==cold and actor-bitwise checks as a backstop.
3. **Incomplete layers silently become garbage**: finalize processes partial layers, and the materialized memory is uninitialized. There is no completeness check in formal runs. MoE layers already take this path because input_scale is never complete, so if an expert tensor is missed, it is not visible. A check can be added later that compares `load_numel` per layer: after removing the known never-loaded input_scale, it must equal the total.
4. **Receiver GPU memory peak**: the export order is all unquantized tensors first, then all quantized linears, then all experts, not grouped by layer. Clones stay until the corresponding layer is complete; MoE layers wait until finalize. Non-local experts are also cloned and kept (`loaded_weights` is appended before counting), so the per-rank peak is roughly the whole payload (rough estimate on the order of 20 GB for full52 NVFP4), plus the materialize overhead of one layer. Mitigations: sort on the sender by `resync_layer_cluster_key` (`verl_mlite/rollout/layer_cluster.py` already has the key), or drop non-local experts before cloning. The receiver's peak GPU memory must be recorded at θ0 admission.
5. **EP4 sharding**: each DP rank receives all 128 experts and keeps only 32, so the transfer volume is 4x. With `use_global_sf` the scale parameters are allocated by the global expert count (`modelopt.py:954-956`); this combination under EP4 needs the gate's online==cold to confirm.
6. **MTP**: the extension refuses MTP drafter sync outright; the main model's mapper drops `mtp.*` (`nemotron_h.py:843`). If the recipe enables MTP rollout, it fails immediately (fail closed), and verl also lowers the sleep level to 1 (`vllm_async_server.py:1414-1431`).
7. **Scale handling**: relies on `export_dtype=null` and fixed scales. If the recipe later allows recalibrating scales, Humming's preservation check only verifies layout, not values; value correctness is covered by the gate.
8. **Failure recovery**: after a mid-sync failure, the worker refuses all later syncs and the rollout can only be restarted; this is intentional.
9. **Single-process layout**: the actor process has its own vLLM global state (`ensure_vllm_runtime`), and an AsyncLLM frontend in the same process may conflict with it. If so, switch to the two-process layout of section 4.
10. **Unverified items**: neither the CPU unit tests nor the GPU gate has been run; the recipe's default extension has not been changed yet (changing the recipe is a verl recipe change and needs human review).
