# DS4-form correctness audit r1 fixes (2026-10-04)

Input: the three external r1 correctness reviews (reviewers A, B and C). Keep the DS4 form with numerics unchanged. All committed locally, not pushed.

Commit labels used below refer to commits on unpushed internal fix branches; they are labels, not SHAs.

## New branch heads

| Repo | Branch | Before | Now | Commits this round |
|---|---|---|---|---|
| vLLM | the vLLM ds4form fix branch | V-pre | V0 | V-2a, V0 |
| Megatron | the Megatron ds4form fix branch | M-pre | M0 | M-1, M-3d, M0 |
| verl | the verl ds4form fix branch | R-pre | R0 | R-3, R0 |

The Dockerfile is re-pinned to: vLLM V0 (full SHA and tree hash, version `0.30.1rc1.dev625+g<sha>`, per `git describe` = `v0.30.1rc0-625`), Megatron M0. `VLLM_REF` (the pinned vLLM BI branch) does not have these two vLLM commits yet; they must be pushed together.

## Item-by-item handling

| # | Finding | Handling | Commit |
|---|---|---|---|
| 1 | Weight sync has no completeness check, and does not latch after a failure (Reviewer C P1-1, Reviewer A P2-1) | Fixed | Megatron M-1 |
| 2a | NVFP4 layers wait for `*input_scale` that never arrives, so whole-layer clones pile up until finalize (Reviewer A P2-2) | Fixed | vLLM V-2a |
| 2b | Every sync rebuilds the MoE kernel, leaving dead references (Reviewer C P2) | Confirmed, fixed | vLLM V0 |
| 3a | The script dropped the prompt+response ≤ PPO_MAX_TOKEN_LEN_PER_GPU check (Reviewer A P2-6) | Not fixed | — |
| 3b | COMPOSE_ONLY requires real inputs (Reviewer C P2) | Fixed | verl R-3 |
| 3c | gpu_memory_utilization does not follow the optimizer (Reviewer C P2) | Fixed: fsdp2 0.65, dist_opt 0.7 | verl R-3 |
| 3d | The two `~grad_offload` lines and the leftover key in mlite.yaml (Reviewer A P2-4) | Fixed: delete the yaml key (the source), and delete both script lines | Megatron M-3d, verl R-3 |
| 4 | impl=vllm + param_offload only errors at runtime (Reviewer A P2-5) | Fixed, added a config-time check | Megatron M0 |
| 5 | `ssd_state_passing.py` uses `tl.fma` unconditionally; is it neutral under BI=0 (Reviewer C item to verify) | Not changed: proven bitwise equivalent under BI=0 | — |
| 6 | sha256 manifest, bucketed sender patch, DP sampler seed, checkpoint/EP DCP keys (Reviewer C P1-2, Reviewer B P1-3/P1-2/P0) | Left untouched as requested | — |

### 1. Fail-closed weight sync

DS4's `vllm_worker.py` has no corresponding check: it only calls `initialize_layerwise_reload` / clone / `finalize_layerwise_processing` inside verl's quantization hook. So a check as small as possible was written here (`verl_mlite/rollout/vllm_worker.py`, +40 lines, no monkeypatch):

- The `load` callback records the names received; if the same name appears a second time within one sync, it raises. This check happens before `load_weights`, so a duplicate tensor cannot fill up a layer's count and trigger early processing.
- Before finalize, `_incomplete_layers` walks `LAYERWISE_INFO`: for a layer still pending (`can_load()`), its `load_numel` must equal the sum of `weight_loader_numel` over all its parameters (excluding `SKIP_LOAD_TENSORS` and attention's `q_scale`/`prob_scale` placeholders), otherwise it raises. Element counts are used instead of name matching because the same `w2_weight` is loaded separately by 128 experts, so the name is still present when one expert is missing.
- `_refit_failed = True` is set before the sync starts and only cleared after finalize completes successfully; any later call that sees it True refuses.

This check depends on 2a: before that fix all W4A16 layers stayed pending until finalize (`input_scale` never arrives), and the check would flag all of them falsely. After the fix only 6 attention layers remain pending (vLLM already processes them in finalize), and their `k_scale`+`v_scale` are exactly 2 elements.

The single-GPU probe (below) verifies: a normal sync passes; withholding `backbone.layers.1.mixer.experts.5.down_proj.weight` → `RuntimeError: MLite refit left layers incomplete: ['model.layers.1.mixer.experts.routed_experts']`; sending `backbone.layers.0.mixer.norm.weight` twice → `RuntimeError: MLite refit received a tensor twice`; after either failure the next normal sync is refused (`An earlier MLite refit failed; refusing further updates`).

### 2. Sync memory

Method: a reload probe script (`reload_probe.py` + `probe_worker.py`). Single GPU (one node), BI=1, V2 runner, CuTe-DSL MoE, FP8 KV, with `worker_extension_cls` set to the real `MLiteVLLMColocateWorkerExtension`; verl's `BucketedWeightReceiver` is replaced by a fake receiver that reads tensors from a local checkpoint: tensors are put into a preallocated 1 GiB device buffer (equivalent to an IPC bucket, counted in the baseline), sent in the exporter's order (non-quantized → quantized projections → routed experts, bucketed per layer), and KV scales are first rounded to BF16 as the exporter does. Peak is measured with `max_memory_allocated`, the resident increment with `memory_allocated` before and after sync (after gc + empty_cache). After sync, 4 prompts with greedy 32 tokens are used to check whether logprobs are bitwise identical to before the sync.

A single GPU has no EP, so one layer's routed experts have 128 experts. Under EP4 each rank has only 32 local experts, so the expert part is estimated at 1/4 (this number is an estimate, not an EP4 measurement).

| | Before fix (V-pre) | 2a only (V-2a) | 2a+2b (V0) |
|---|---|---|---|
| Layers pending at finalize | routed experts 23, shared up/down 23 each, lm_head, attention 6 | Only attention 6 (48 bytes) | Same |
| Clones accumulated until finalize | 16.46 GiB (routed 16.52 GB, shared 0.26 GB, lm_head 0.20 GB) | ≈0 | ≈0 |
| Extra peak device memory during sync, 1st | 17.20 GiB | 3.09 GiB | 1.42 GiB |
| Extra peak device memory during sync, 2nd | 17.19 GiB | 1.41 GiB | 1.42 GiB |
| Resident increment after sync, 1st | +1.75 GiB | +1.75 GiB | 0 |
| Logprobs after sync bitwise identical to cold load | Yes | Yes | Yes |
| EP4 estimated per-rank peak | ≈4.3 GiB | — | ≈0.4 GiB |
| EP4 estimated per-rank resident dead references | ≈0.44 GiB | — | 0 |

- **2a**: `ModelOptNvFp4FusedMoE.create_weights` marks `w13_input_scale`/`w2_input_scale` with `weight_loader_numel = 0` when `use_a16`; the `_DropInputScale` placeholder of W4A16 linear layers is handled the same way. This attribute is only read by the layerwise reload trigger condition; neither the loader nor the kernels are affected. After the fix each layer is processed as soon as its weights are complete, and the peak is only one layer's clone + materialized copy + processing intermediates. gate1's log had vLLM's "allocating buffers for multiple layers at once" warning; it did not appear in this gate.
- **2b**: the dead references do exist. On reload, `_build_moe_kernel` builds the quant config with the newly processed scales; afterwards `_copy_and_restore_kernel_tensors` swaps the layer back to the original storage, but the new quant config still references those new scales. `FlashInferCuteDSLW4A16Experts` already reads scales through the layer (`_registered_scale`), so after its `process_weights_after_loading` binds the layer, the `scale`/`alpha_or_gscale` of the `_w1`/`_w2` descriptors are set to None (+4 lines). Resident increment 1.75 GiB → 0, logprobs bitwise unchanged. Numerically the size is exactly 23 layers × 128 experts of swizzled block scales.

### 3. verl script

- 3a not fixed: the DS4 script (`run_deepseek_v4_true_on_policy_preview_megatron.sh` at the pinned DS4 recipe commit) does not have this check. The defaults satisfy it; when `MAX_RESPONSE_LENGTH`/`PPO_MAX_TOKEN_LEN_PER_GPU` are overridden manually and exceed the limit, it fails at verl's dynamic bsz packing.
- 3b: the condition of the input-check block is changed to `DRY_RUN != 1 && COMPOSE_ONLY != 1`, and the `MODEL_PATH/config.json` check is moved into this block too. The environment variables still have to be set, but may point to nonexistent paths. DS4 here is the same as ours was before (only exempting DRY_RUN), so this is a deviation from DS4.
- 3c: defaults are set in the `ACTOR_OPTIMIZER` case (DS4 sets them in the MODE case; same pattern).
- 3d: `grad_offload: false` was brought in by the base mlite.yaml (the yaml DS4 references has it too, and the DS4 script also deletes it with `~`). config.py no longer declares this field, so it is deleted from the yaml, and the script no longer needs those two `~` lines. Two other old example scripts in the repo (qwen3moe, deepseek_v4_dapo) still set `engine.grad_offload=`. They already fail on the base because the config does not have this field; not touched this round.
- Measured in the image: both optimizers with `COMPOSE_ONLY=1` and all inputs pointing to nonexistent paths exit 0; the composed rollout `gpu_memory_utilization` is 0.65 for fsdp2 and 0.7 for dist_opt; there is no `grad_offload` anywhere in the config.

### 4. param_offload

A check was added in `MegatronLiteEngineConfig.__post_init__` (an `if` plus a `raise`): `impl == "vllm" and param_offload` → `ValueError("impl='vllm' requires param_offload=False")`. The old `export_dtype` check was not restored; it is still blocked by `protocol.export_hf_weights` at the first export, and the audit did not ask to restore it. Verified in the image: the default config constructs; `param_offload=True` raises the ValueError above.

### 5. BI=0 neutrality of `tl.fma`

Existing evidence: item 4 of an earlier reply to Reviewer B's audit, which tested 3 shape sets with `scripts/fma_equivalence_check.py` in an earlier image and found them bitwise identical. This time it was rerun with BI=0 on the currently pinned toolchain (the delivery candidate image: triton 3.7.1, torch 2.13.0a0 nv26.07, CUDA 13.3, GB200):

- The original form `fast_exp(dA_cs) * states + new_states` and `tl.fma(...)` have 0 bit differences on (4,8,4096), (33,16,8192, with init) and (128,64,8192, with init), 71.56 million elements in total;
- autotune compiles 12 `_state_passing_fwd_kernel` PTX files for each form; across the 24 files there are 0 `add.f32`/`add.rn.f32` instructions, only `fma.rn.f32`. In other words, Triton's default fp fusion already compiles the original form into FMA.

Conclusion: bitwise unchanged under BI=0, no gate added. The log is on shared storage (`fma.log`).

### 6. Not touched

The input sha256 manifest (human decided to delete it per DS4), the verl bucketed-weight-sender patch (a human-approved exception), the DP sampler seed (awaiting human decision), checkpoint/EP expert DCP keys (another agent is porting #234/#235 on the Megatron ds4form checkpoint branch; this round did not touch `primitive/ckpt` or the fsdp2 checkpoint code). Reviewer A P2-3 (requant runs twice on restore), P2-7 and P2-8 are outside this round's scope and were not changed.

## Verification

Image `<image>` (a candidate image with vLLM V0, Megatron M0 and verl R-3), built with the ds4form image build script (base image is the DeepEP image; source snapshot on shared storage, matching the three branch heads).

- **Unit tests** (one node, `unit-overlap.sh`, logs on shared storage):
  - Megatron `test_nemotron_h_serving_parity.py`: 22 passed (including the three 4-GPU tests FSDP2, DeepEP EP4, one-sided);
  - `test_nemotron_h_training_vjp.py`: 66 passed;
  - engine config + qat yaml + runtime: 50 passed;
  - vLLM `test_nemotron_h_quantization.py`, `test_flash_attn_fixed_split.py`, `test_nemotron_h_batch_invariance.py`: 24 passed.
  - No new unit test files; the probes above are the verification for the sync check and memory.
- **Quick gate** (one node, `gate.sbatch`, `--mode quick_alignment_test`, FSDP2, EP4, 3 steps, recipe run unmodified inside the image): exit 0. Over 3 steps `training/rollout_probs_diff_max = 0.0`, `diff_mean = 0.0`, pearson all 1.0. All 3 weight syncs passed the completeness check (no refit errors in the log, and no vLLM "Failed to load weights"). `update_weights` 7.8 / 14.0 / 9.4 s (gate1 was 11.3 / 11.9 / 9.1 s), `update_actor` 88.5 / 21.2 / 28.1 s, node host peak 441.9 GiB (gate1 was 448 GiB). Logs on shared storage.
- **Probes**: per-probe `.log`/`.json` files A2, B1 and C on shared storage (A2 = before fix; B1 = 2a + check, including drop; C = 2a+2b + check, including dup).
