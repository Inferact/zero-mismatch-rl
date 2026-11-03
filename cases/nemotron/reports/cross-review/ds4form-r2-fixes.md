# DS4-form correctness audit r2 fixes (2026-10-04)

Input: the three external r2 correctness reviews (reviewers A, B and C), plus two items the coordinator added this round (the attention backward D term; merging the checkpoint P0 branch + the DCP local staging default). Keep the DS4 form with numerics unchanged (except the D term, a backward change explicitly approved by the human). All committed locally, not pushed.

Commit labels used below refer to commits on unpushed internal fix branches; they are labels, not SHAs.

## Branch heads

| Repo | Branch | Before | Now |
|---|---|---|---|
| vLLM | the vLLM ds4form fix branch | V0 (r1 head) | V2 |
| Megatron | the Megatron ds4form fix branch | M0 (r1 head) | M-ckpt-head |
| verl | the verl ds4form fix branch | R0 (r1 head) | R1 |

Reviewer C P2-5: in the vLLM worktree the branch ref was still at an older commit, and the fix commits were hanging on a detached HEAD. The vLLM ds4form branch was first fast-forwarded to V0 and checked out (`git rev-parse` of the branch = V0), and further commits were made on the branch.

The Dockerfile (an internal verl commit) is pinned to: vLLM V2 (full SHA and tree hash), version `0.30.1rc1.dev627+g<sha>` (`git describe` = `v0.30.1rc0-627-g<sha>`); Megatron M-ckpt-head. `VLLM_REF` does not have these vLLM commits yet; they must be pushed together.

## Item-by-item handling

| # | Finding | Handling | Commit |
|---|---|---|---|
| 1 | Reviewer B P0: W4A16 layers are processed early, so an old checkpoint's `input_scale` arriving in a later bucket errors with "no module or parameter named …input_scale" | Fixed | vLLM V1 |
| 2 | Reviewer B P1: `e_score_correction_bias` is loaded in place and is not in the completeness check | Fixed | Megatron M1 |
| 3 | Reviewer A P2-1: a duplicate tensor throws inside the bucket callback without an ack, so the sender blocks forever | Fixed | Megatron M2 |
| 4 | Reviewers A/C P2-2: `_incomplete_layers` only counts parameters, a different definition from vLLM's `load_numel_total` (parameters + buffers, including padding) | Add a one-line note, do not change the criterion | Megatron M1 |
| 5 | Reviewer C P2-3: under BI an explicit `-quant_fp8` returns early without setting the fusion switches to False | Fixed (set the switches first, then check the op) | vLLM V2 |
| 6 | Added: attention backward D term (adopted by human decision) | cherry-pick the prototype commit, add docstring | Megatron M3 |
| 7 | Added: checkpoint P0 branch (the Megatron ds4form checkpoint branch) | cherry-pick 16 commits in order, no conflicts | Megatron M-ckpt-1…M-ckpt-head |
| 8 | Added: DCP local staging (#196) default | Script default `/tmp/mlite-dcp-stage` | verl R1 |
| — | EP4 DCP keys (another agent's branch, already merged via item 7), sampler seed, sender patch, manifest, export_dtype config-time check (Reviewer C P2-1) | Left untouched as requested | — |

### 1. W4A16 input_scale of old checkpoints (vLLM V1)

Cause: after `weight_loader_numel = 0`, W4A16 layers are processed as soon as their weights are complete, and processing deletes `input_scale` (linear layers via `_DropInputScale`, MoE layers via `replace_parameter(..., None)`). An `input_scale` arriving later then cannot find its parameter: linear layers raise ValueError in `AutoWeightsLoader`, MoE layers raise AttributeError in `RoutedExperts.load_weights`. BI=0 refit is the same.

Fix (two places, both only take effect when the parameter no longer exists):
- Linear layers: ModelOpt configs containing W4A16 layers (`ModelOptNvFp4Config` with `W4A16_NVFP4`, or a mixed config with `W4A16_NVFP4` layers) add `.input_scale` to `_ignore_unexpected_suffixes`. This list is an existing vLLM mechanism and is only consulted when the module does not have that parameter. FP8 layers (Mamba in/out_proj) always keep the `input_scale` parameter, so they load as usual, and the layer still waits for it to arrive; unaffected. `base_config.py` annotates this class attribute as `tuple[str, ...]`, otherwise mypy errors.
- MoE: when `RoutedExperts.load_weights` cannot find a `*input_scale` parameter and `moe_quant_config.use_nvfp4_w4a16`, skip it instead of raising.

Unit test `tests/quantization/test_modelopt.py::test_modelopt_w4a16_reload_ignores_legacy_input_scale[False|True]`: mixed config, one W4A16 linear layer and one FP8 linear layer. Cold load first (with legacy input_scale), then a layerwise reload where `input_scale` arrives in a bucket before and after the weights respectively. Asserts: the W4A16 layer is done processing at the first bucket; the FP8 layer waits for the FP8 input_scale before processing; finally all tensors are bitwise identical to the cold load. With the pre-fix `modelopt.py` mounted back, `[True]` (late arrival) fails with exactly `There is no module or parameter named 'w4a16.input_scale'`; after the fix both orders pass. The MoE path has no unit test and is covered by the probe below (real model, 5935 legacy input_scales, most of them routed experts').

### 2. Parameters loaded in place (Megatron M1)

Inside the `load` callback, collect the mapped names returned by `model.load_weights`. Before finalize, every parameter in `model.named_parameters()` whose last name segment is in `SKIP_LOAD_TENSORS` must appear in this set, otherwise it counts as incomplete. The other names in `SKIP_LOAD_TENSORS` (`_expert_map`, `expert_mask`, `expert_*_to_*`) are all runtime buffers registered with `register_buffer(persistent=False)` in vLLM and are not in `named_parameters()`, so they need no separate exemption. The exemption for non-local experts is unaffected, because experts are still judged by the layer's element count. This matches the pre-refactor check of `e_score_correction_bias` in `LayerwiseReloadSession.deficits()`. Probe: when `backbone.layers.1.mixer.gate.e_score_correction_bias` is withheld, it reports `MLite refit left layers incomplete: ['model.layers.1.mixer.gate.e_score_correction_bias']`; a normal sync gives no false positive.

### 3. Hang caused by a duplicate tensor (Megatron M2)

The whole callback is wrapped in try/except: the first exception (including duplicate tensors and errors from `load_weights` itself) is recorded, and later buckets are only acked, not loaded; after `receive_weights` returns, this exception is raised first, before `_incomplete_layers`. The `_refit_failed` latch is unchanged.

Reproduced and verified with the real sender (see probes R4/Old): with the duplicate tensor in a later bucket, the pre-fix worker reports "received a tensor twice", but the sender is still blocked in `recv()` 300 s after the worker returned (HUNG); after the fix it reports the same error and the sender finishes normally.

One more observation: if the duplicate tensor is in the **same** bucket, verl's `BucketedWeightSender` keys `bucket_meta` by name, so the latter overwrites the former and the receiver sees it only once. The duplicate check cannot see this case, but since it is the same tensor, results are unaffected (probe R1's dup: sync succeeds, logprobs bitwise identical). Not changed this round.

### 4. The `_incomplete_layers` criterion (Megatron M1, comment only)

The criterion is not changed, because: the only layers still pending at this point are attention layers (W4A16 input scales are no longer waited for), and the only parameters they load are the two k/v scales; their buffers are runtime copies of the k/v scales, not loaded from the checkpoint. Switching to vLLM's `load_numel_total` would require relisting all attention buffers in the exemption table, which means more code. The comment states: only parameters are counted; a padded layer will be reported as incomplete, i.e. fail closed (false positives, never false negatives). In the delivered config, vocab 131072, TP=1, there are no padded layers. The 3 syncs in the gate and the probe's normal/legacy syncs had no false positives.

### 5. Fusion under `-quant_fp8` (vLLM V2)

`use_cuda_fp8_quant` now first sets `fuse_norm_quant` / `fuse_act_quant` from None to False, and only then decides whether to append `+quant_fp8`. Fusion values the user set explicitly and an explicitly written `-quant_fp8` are both preserved. "Reject `-quant_fp8` under BI" was not chosen, because this fix just moves three lines and is smaller. The expectation of the corresponding unit test parameter `(["-quant_fp8"], None)` is changed to False.

### 6. Attention backward D term (Megatron M3)

Cherry-picked from a scratch attention D-fix branch, no conflicts: the `backward.py` forward only saves the dequantized Q/K/V, and the backward first recomputes BF16 O/LSE with `_flash_attn_fwd`, then calls `_flash_attn_bwd`; `module.py` only changes a docstring; the unit test's relaxation factor goes from 4.0 back to the default. As requested, one sentence was added to the backward docstring: FA4's FP8 forward is declared forward-only upstream (interface.py rejects requires_grad outright); the actor calls `_flash_attn_fwd/_bwd` directly and recomputes BF16 O/LSE so that D matches the P recomputed in backward. The commit message cites the attention D-fix prototype report and [`fp8-attn-bwd-survey.md`](../fp8-attn-bwd-survey.md). Forward and serving are unchanged: `rollout_probs_diff` is 0 in both gates.

### 7. Merging the checkpoint P0 branch

The 16-commit range of the checkpoint branch was cherry-picked onto M3; none of the 16 commits conflicted, and commit messages were kept verbatim (including references to PRs #195/#196/#234/#235/#236/#237 of a community Megatron-LM fork): M-ckpt-1, PP4 dist_opt save, #195, #196, #237 (two commits), #236, HDO reuse, #234 (two commits), #235 (three commits), test fixes (two commits, the second being M-ckpt-head), and a new 4-rank test. After the merge, the tree differs from the checkpoint branch head only by this round's and r1's own commits (6 files).

### 8. DCP local staging default (verl R1)

The DS4 recipe (verl PR 7688) only passes `MLITE_DCP_LOCAL_STAGE_DIR` to Ray when it is already set, and sets no default itself. As requested we set a default: `MLITE_DCP_LOCAL_STAGE_DIR=${MLITE_DCP_LOCAL_STAGE_DIR:-/tmp/mlite-dcp-stage}`, add a one-line comment, and add it to `RAY_ENV_NAMES`. Note: inside a pyxis/enroot container `/tmp` is a private tmpfs, so staging consumes host memory. The cluster sbatch still sets `/stage` explicitly (bound to the node-local disk); see the save→restore measurement below.

## Verification

Images (built by the ds4form image build script on top of a DeepEP base `<image>`):
- img1 `<image>` (candidate image with vLLM V-test, Megatron M1, an earlier verl commit): r2 items 1–5.
- img2 `<image>` (vLLM V2, Megatron M3, another verl commit): adds the D term.
- img3 (**final candidate**) `<image>` (vLLM V2, Megatron M-ckpt-head, verl R1): adds the checkpoint branch and the staging default. Its source snapshot on shared storage matches the three branch heads.

vLLM V-test and V2 differ only in test files (the dtype fix in the unit test was squashed into V1), so the vllm package installed in img1 and img3 is identical.

**Unit tests** (img3, one node, logs on shared storage):
- Megatron `test_nemotron_h_serving_parity.py`: 22 passed.
- `test_nemotron_h_training_vjp.py`: 67 passed. Including the default tolerance of the D term, and the new 4-GPU test `test_ep4_fsdp2_checkpoint_restores_each_ranks_expert_bank`; run once more separately in `unit-ds4f-all`, also 67 passed.
- engine config + qat yaml + runtime: 50 passed.
- runtime/checkpoint + examples: 75 passed, 2 skipped, 1 failed. The failure is `test_hf_model_only_save_uses_protocol_and_writes_hf_metadata` (a lambda does not accept `export_dtype`), already recorded in the ckpt report as failing on the baseline.
- fsdp2: 40 passed, 1 skipped; HDO: 74 passed; the torchrun 4-rank test of #235: 2 passed.
- vLLM `test_nemotron_h_quantization.py`, `test_flash_attn_fixed_split.py`, `test_nemotron_h_batch_invariance.py` and the new W4A16 reload test: 26 passed.
- Same suite on img2 (without the checkpoint tests): 22 / 66 / 50 / 26 passed.

**Reload probe** (`reload_probe2.py`, single GPU, one node, BI=1, V2 runner, CuTe-DSL MoE, FP8 KV). This time it uses verl's real `BucketedWeightSender`/`BucketedWeightReceiver` (shm, 1 GiB bucket): the driver sends and the worker runs `MLiteVLLMColocateWorkerExtension`. Compared with r1's fake receiver, it additionally covers the ack protocol. The probe ran on img1; changes after that were only to the attention backward and checkpoint code, with no change to worker or vLLM code.

| Run | Mode | Worker result | Sender | Logprobs after sync bitwise identical to cold load |
|---|---|---|---|---|
| R1 | legacy_late: 5935 legacy W4A16 input_scales (shared/routed experts, lm_head), sent after all tensors | Success | Finished normally | Yes |
| R1 | normal | Success | Finished normally | Yes |
| R1 | dup (within the same bucket) | Success (sender already deduplicated, see section 3) | Finished normally | Yes |
| R2 | drop_bias | `incomplete: ['model.layers.1.mixer.gate.e_score_correction_bias']` | Finished normally | — |
| R3 | drop_expert (layers.1 experts.5 down_proj.weight) | `incomplete: ['model.layers.1.mixer.experts.routed_experts']` | Finished normally | — |
| R4 | dup (in a later bucket) | `received a tensor twice` | **Finished normally** | — |
| Old | Same as R4, with the pre-fix `vllm_worker.py` (M0) mounted back | `received a tensor twice` | **HUNG** (still blocked 300 s after the worker returned) | — |

Logs are the per-run probe `.log`/`.json` files (R1, R2, R3, R4, Old).

**EP4 FSDP2 quick gate** (`gate.sbatch`, `--mode quick_alignment_test`, recipe run unmodified inside the image, 3 steps):

| Image | rollout_probs_diff max/mean (3 steps) | update_weights (s) | update_actor (s) | Host peak |
|---|---|---|---|---|
| img1 | 0 / 0, pearson 1.0 | 7.3 / 15.1 / 14.2 | 91.5 / 21.8 / 33.3 | 438 GiB |
| img2 | 0 / 0 | 10.1 / 12.6 / 8.0 | 89.4 / 22.3 / 28.5 | — |
| **img3** | **0 / 0** | 9.7 / 15.6 / 10.9 | 91.8 / 28.5 / 20.1 | 448 GiB |

All three runs exit 0, with no refit errors and no "Failed to load weights" in the logs. One DataLoader worker is killed at exit, same as gate1; it does not affect results. Logs are on shared storage.

**Save→restore** (img3, one node, the ckpt-run sbatch with the ds4f fast env file in `fast` mode: quick mode, `SAVE_FREQ=1`, `MLITE_DCP_LOCAL_STAGE_DIR=/stage`):
- Phase A exit 0. All 3 steps saved (global_step_1/2/3), each step `rollout_probs_diff_max = diff_mean = 0`, host peak 637 GiB (consistent with 626–637 GiB in the ckpt report).
- Phase B restores from global_step_2, exit 0. Step 3 `rollout_probs_diff_max = diff_mean = 0`, and global_step_3 was saved. The only Traceback in the log comes from torch `dump_compile_times` at exit, harmless.
- Logs are on shared storage.

## Leftovers

- Duplicate names within the same bucket are silently merged by the verl sender (section 3); no numerical effect, not changed this round.
- The recipe's default staging directory is tmpfs inside an enroot container (consuming host memory); when running on the cluster, bind `/stage` to the local disk as ckpt-run does.
- Nodes: all three node holds have been released (two of them automatically at job end).
