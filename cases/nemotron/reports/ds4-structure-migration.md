# ds4-structure-migration

This report records the work of converting the Nemotron-H Lightning NVFP4 training actor (Megatron mLite, `impl=vllm`) into the DS4 aligned actor structure. Midway the user reordered priorities and asked to first make PP4 a delivery candidate and validate it; EP4, DeepEP and FSDP2 were paused.

Below, the conclusions and evidence for the PP4 candidate come first, followed by a record of the parked EP4 and FSDP2 work.

Path abbreviations:
- `W` = the worktree of the internal Megatron-LM fork used for this work
- `M` = `W/experimental/lite/megatron/lite/model/nemotron_h`
- `R` = the run's results directory (receipts and scripts)

Reviewers are referred to as Reviewer A, Reviewer B and Reviewer C (three independent automated reviewers); `rN` is the review round. Commit SHAs of the internal forks are omitted.

## 0. Branches

All branches below live in internal forks (Megatron-LM and verl).

| Branch | Content |
|---|---|
| **PP4 candidate branch** | PP4 delivery candidate (36 commits, see below) |
| EP4 WIP branch | Parked: EP4 + DeepEP + route fix (EP version) |
| original DS4-structure working branch | Same as the EP4 WIP branch (the original working branch) |
| FSDP2 WIP branch | Parked: FSDP2 WIP (whole-model OOM) |
| PP4 diagnostic branch (local, not delivered) | candidate + observer + main_grad checksums; used only for gates |
| verl EP4 recipe branch | Parked: EP4 topology + Dockerfile adds DeepEP |
| verl EP4 FSDP2 recipe branch | Parked: FSDP2 recipe |

The mLite dev branch and an earlier DS4 diagnostic branch were not touched. The PP4 candidate's recipe is a commit on the verl recipe-on-main branch: it requires `BF16_MASTER_PATH`, checks its config.json, and passes `impl_cfg.bf16_master_path`.

Commits on the PP4 candidate (built on top of the mLite base commit, kept rebasable):

```
Call the serving kernels directly; remove vLLM layer / config / process groups
Deterministic FlashAttention backward
Upstream #7849: LR scheduler multiplies by wd_mult (cherry-pick)
Upstream #7849: also multiply by wd_mult when there is no scheduler (cherry-pick)
Attention VJP uses the visible out/LSE (DS4 style)
Resolve MoE device capability at construction time
dist_opt: gather params before requantize
Route weight gradient changed to <dy, visible expert output>
First review round fixes
Mamba backward made deterministic
Second review round fixes (temperature, reduction required, EP4-rank oracle)
BF16 master initialized from the official BF16 release (merge of the bf16-master branch)
BF16 master read directly to device (saves host RAM)
BF16 review fixes (θ0 floor, checked keys, host RAM)
θ0 floor and BF16 source tests; README adds table of 24 calibrated FP8 tensors
θ0 floor re-set from the measured per-tensor distribution
Checkpoint restore reinstalls the saved deployment bytes (Reviewer B r1 P0)
BF16 master checks release identity, not just compatibility (Reviewer B r5 P0)
Compatibility check compares FP4 values, not products (Reviewer C r5 P1-2)
Route input gradient summed in DS4 deterministic scatter-backward order (Reviewer B r1 P1)
LR scheduler uses per-group max_lr/min_lr, rejects lr_mult (Reviewer B r1 P1)
Oracle of training-side EP4 rank vs vLLM EP4 serving rank (Reviewer B r1 P1)
Exported KV scale rounded to BF16 first (paired with vLLM removing _preserve_checkpoint_scale_dtype)
Error when router GEMM runs before init_batch_invariance
Measure at build time that the router GEMM is independent of row count
Capability seeding uses torch's own device UUID list (Reviewer B r1 P2)
verl engine checkpoint restore goes through the restore hook (Reviewer B / Reviewer A r1b P1)
FP8 KV cache paged by request length (Reviewer B r1b P2; output bitwise unchanged)
EP4 Humming scheduling uses the delivered vLLM's nemotron_humming_schedule (Reviewer A r1b P1-2)
Restore byte-equality test goes through the verl engine load path (test)
impl=vllm rejects export_dtype≠null and param_offload at config time (Reviewer A P2)
Routed deployment must be given the EP4 reduction; remove the direct-sum branch (Reviewer A r1b P2-3)
Vendor backward function signature check (test)
Attention docstring (docs)
```
verl recipe-on-main branch: one commit adds a host-RAM check before launch (MemAvailable ≥ `HOST_MEM_MIN_GIB`, default 760 GiB).

The wd fix I originally wrote has been replaced by two upstream commits. The upstream version only multiplies by `wd_mult`, does not involve lr_mult, and does not claim per-group max_lr/min_lr semantics, so Reviewer A P2-1 is resolved as well.

## 1. Phase 1: call kernels directly

- `M/kernels.py` follows DS4 and calls the following kernels directly on tensors:
  - `humming_forward` (dense and indexed), `may_process_input`, `moe_align_block_size`, `moe_fused_mul_sum`;
  - FlashInfer CuTe-DSL `mm_bf16_fp4`;
  - `scaled_fp8_quant`.
- Weight preprocessing calls vLLM's own helpers throughout, in the same order as the ModelOpt method:
  - `prepare_humming_linear_layer_config`, `convert_to_humming_moe_kernel_format`, `get_humming_moe_quant_config`;
  - `pad_nvfp4_weight_for_cutlass`, `swizzle_blockscale`, `flashinfer_prepare_bf16_fp4_weights`;
  - global scale is taken as `1/(1/s)`.
- Removed `vllm_runtime.py`, `init_distributed_environment`, `ensure_model_parallel_initialized`, and the vLLM group checks. VllmConfig, forward context and workspace are now **not needed**.
- Oracle tests: build the vLLM objects on the same checkpoint bytes and run `torch.equal` for each M ∈ {1, 7, 64, 513, 8192}; all results are bitwise equal. Coverage:
  - Humming linear, 3 shapes;
  - CuTe-DSL linear, 3 shapes;
  - 128 experts, per-route FC1 and down;
  - FP8 query quantization.
- Added an EP4-rank oracle (responding to Reviewer A P1-3). It uses 32 local experts with serving's `configure_nemotron_humming`, a `valid_shape_m` computed from the global token count, an expert_map, and feeds only the tokens routed to that rank. The routes and per-rank partial sums it computes are bitwise equal to the training-side 128-expert deployment (across all M). This way the training-side single-GPU EP4 emulation no longer relies on an untested tile-independence assumption.

## 2. Side fixes and review fixes

**Deterministic attention backward**
- Changed to `_flash_attn_bwd(..., deterministic=True)`.
- Cost:
  - 16k single sequence: 3.71 → 3.93 ms (+6%)
  - 4×4k: 1.32 → 1.60 ms (+21%)
  - 16×1k: 0.70 → 1.19 ms (+70%)
- Added a full-geometry bitwise test: Q32/KV2/D128, packed 8192+3000+1+5000.

**Attention switched to the DS4 style**
- Save the visible out, LSE and the dequantized Q/K/V; the backward no longer recomputes.
- Noise-floor ratio relative to the FP64 reference: dq 2.9×, dk 2.4×, dv 0.9×. The cause is that serving's P@V goes through FP8; the user accepted this.
- Removed the unreachable Triton path.

**Route weight gradient**
- `d_routes = <dy, visible per-route output>`. The visible output is Humming's BF16 output, before `moe_fused_mul_sum` and the EP4 reduction.
- dx, d_up, d_down are still the high_precision VJP of the BF16 master.
- The frozen per-input contract is written in `M/README.md`.
- Memory: 31.5 KiB more per token per MoE layer. One PP4 stage has 6 MoE layers, about 1.5 GiB at 8k tokens.
- Validation:
  - CPU contract test, with an FP32 autograd reference implementation where FC1 and the output take the visible values (STE);
  - measured on the real 128-expert Humming at θ0 and two post-update snapshots (deployment requantized from the updated master), and checked that the saved output is indeed the one used in the forward combine.

**dist_opt gather**
- The post-step hook first calls MCore's `start_param_sync(force_sync=True)`.
- Under PP4 (DP1) this is a no-op. It fixes the problem of θ1 mismatching everywhere under EP4/DP4.

**First review round** (Reviewers A/B/C)
- CuTe-DSL is constructed on the device where the weights live (Reviewer C P1-2).
- Input validation consistent with serving: NaN group scales raise; the global scale must be a single FP32 (Reviewer C P2-1).
- Warning when a tuning interval is missing, consistent with serving (Reviewer A P2-5).
- README corrections (P2-7).
- Capability handling (Reviewer A P2-4 / Reviewer C P1-1):
  - Changed to seed only once at build time, with a comment documenting the real call chain: `moe_fused_mul_sum → _heuristic_config(inputs.device.index) → get_device_capability`, which maps the torch ordinal to a physical id through `CUDA_VISIBLE_DEVICES` (source in the d9 image: `moe_fused_mul_sum.py:157`, `platforms/interface.py:342`).
  - Reviewer C said "this function does not call capability", which does not match the d9 vLLM; the traceback from the first proxy run proves it.
  - The `set_assigned_physical_gpu_ids` suggested by Reviewer A does not affect this mapping (vLLM docs verbatim: "independent of any logical-to-physical mapping"), so there is no supported mechanism available. The environment variable still has to be temporarily removed, but this is only triggered when CUDA sees more GPUs than the variable lists (i.e., the verl Ray actor case), and is executed only once during single-threaded build.

**Mamba determinism** (found by A/A)
- causal_conv1d's dweight/dbias are accumulated with atomics. Changed to keep the vendor-provided dx and sum dweight/dbias in FP32 in a fixed order.
- The mamba_ssm SSD backward's dA/dD/ddt_bias are accumulated with atomics. verl's full_determinism does not turn on `torch.use_deterministic_algorithms`, so under `impl_cfg.deterministic` we explicitly call `set_deterministic_mode(True)`.

**Second review round** (Reviewer A PP4 review)
- When temperature≠1, follow DS4 and the vLLM sampler: first cast to FP32, then divide by T. The test compares against exactly this serving algorithm. The recipe uses T=1 and is not affected.
- `routed_forward_reduction=None` is rejected directly at config time (P1-2).
- Added the EP4-rank oracle mentioned above (P1-3).

**BF16 master merge**
- The master comes from `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16@a9904d24`. θ0 deployment = requant(BF16 master): NVFP4 uses TE 4over6 MSE with global amax/1536; FP8 uses amax/448 (scheme A, unchanged).
- The θ0 floor is where I got it wrong. The first integration gate (pp4g2) failed at `layers.1.experts.51.up_proj`: values agreement 98.23%, below the per-tensor floor of 99% I had set. That floor was not calibrated on the full data.
  - Measured over all 5935 NVFP4 tensors (`R/theta0-pertensor.jsonl`): values min 97.87%, codes min 98.07%, block scale min 97.69%; per-layer aggregate min 99.88%; whole model 99.93%.
  - Floor changed to: per-rank aggregate values ≥ 99.8%, nonzero blocks ≥ 99.7%; per tensor ≥ 95%; global scales must be exactly equal.

**Third review round** (Reviewer B r1, Reviewer B r5, Reviewer C r5)
- **P0 checkpoint restore**.
  - Original problem: after load, the post-optimizer hook was called, which requantizes, so on a step-0 restore the checkpoint bytes were replaced by requant(master) even though no update had happened in between.
  - Another problem: the `_packed` of linear and FP8 is a non-persistent buffer. The dist checkpoint saves it, but `load_state_dict(strict=False)` drops it, so the restored scales got combined with the old codes.
  - Fix: `_packed` made persistent; added `post_checkpoint_load_hook`, which only reinstalls the restored bytes and does not requantize. The runtime calls this hook first and falls back to the original hook if absent.
  - Definition of restore: reproduce the bytes last deployed at the moment of saving. That is, the θ0 deployment before the first update, and requant(master) afterwards.
  - Test: at step 0 and after one update, do save→new→load→restore→export and compare byte by byte. Step 0 uses a checkpoint for which requant changes the bytes, so the test is sensitive to this bug.
- **P0 BF16 identity**. The original check only proved compatibility.
  - `M/bf16_release.json` records the HF LFS sha256 of the 14 shards and the git blob sha1 of the index. After these all verify, it computes the sha256 of each layer (or each top-level tensor), 57 groups in total. The generator is a release-manifest helper script kept with the run's scripts.
  - Each rank only checks the groups it reads. The proxy declares, via `impl_cfg.bf16_master_layers`, which release layer each of its layers corresponds to; full-depth must leave it empty, the proxy must fill it.
  - The original check was renamed the compatibility check.
  - Adversarial tests: change the master by one BF16 ulp, the compatibility check still passes and the identity check rejects; a wrong proxy layer mapping is also rejected.
- **P1 route input gradient**. Following DS4 `communication.py:116`: in slot order, round to BF16 after each addition. `[1, 1/256, -1, 0, 0, 0]` gives 0 (previously 1/256). The user's standing requirement is to follow DS4, and there is no reason not to here.
- **P1 lr_mult**. This repository's MCore scheduler no longer uses lr_mult; it only honors per-group max_lr/min_lr (lr_mult is a compatibility field that is always 1). So following MCore, each group now decays within its own bounds, and lr_mult≠1 raises an error. Tests cover step and scheduler restore.
- **P1 EP4 oracle**. The earlier EP4-rank oracle (second review round) compared our own 32-expert instance against the 128-expert deployment, and did not compare against vLLM's EP4 rank; Reviewer B was right.
  - New oracle: vLLM FusedMoE with an EP4 parallel config, 32 local experts, vLLM's own expert_map; ModelOpt applies `configure_nemotron_humming` under the Nemotron config; DP metadata gives the global token count.
  - Covers 4 ranks × 7 values of M (including both sides of the scatter threshold) × two sets of weights (random checkpoint bytes, a post-update requantized snapshot); per-route FC1 and down are all bitwise equal.
  - A single process has no EP process group, so the test monkeypatches `make_parallel_config`.
  - Found along the way: the oracle fixture originally did not set `VLLM_HUMMING_MOE_GEMM_TYPE`, so vLLM under EP4 would pick grouped Humming, unlike the serving recipe. It is now consistent with the recipe, pinned to indexed.
- **Reviewer C P1-2**. Originally both sides' products used the same unit, so algorithmically a 1-ulp difference cannot occur. Still changed to compare FP4 values directly where unit>0, making it fully independent of product rounding.
- **Reviewer C P1-1** is covered by the aggregate plus per-tensor floors above. P1-3 (SFT rerun) is handled by another track. P1-4 (CUBLAS in the provenance) has been corrected.

### Backward recompute audit ("whatever DS4 saves, we also save as visible tensors")

| Location | Quantities used in backward | Conclusion |
|---|---|---|
| attention | q/k/v_ref, out, LSE | Changed to no recompute, consistent with DS4 |
| routed MoE | x, visible FC1, visible output, master | Consistent with DS4 grouped MoE |
| linear | input + master | Closed-form VJP |
| RMSNorm / gated RMSNorm | input + weight, rstd recomputed in closed form | Closed-form VJP |
| router | logits + fixed ids | Closed-form replay |
| shared expert | autograd intermediates | No recompute |
| log-prob | the same logits | Consistent with DS4 |
| Mamba conv1d | x/w/b/seq_idx | DS4 has no corresponding module. The vendor recomputes the pre-activation inside the kernel; dW/db are now summed deterministically by us in FP32, likewise recomputed from the inputs |
| Mamba SSD | x/dt/A/B/C/D/dt_bias/visible output | DS4 has no corresponding module. vLLM's visible SSD does not expose chunk states in mamba_ssm format, so it cannot be changed to save; keeps the MCore approach |

## 3. PP4 candidate validation

Environment: d10 image, with the worktree overlaid via `PYTHONPATH`; gates run on the diagnostic branch (candidate + observer + main_grad checksums). As required, only three designated nodes were used (node A, node B, node C).

**Unit tests** (`MLITE_TEST_HARNESS=1 pytest test_nemotron_*_unit.py` + engine config tests)
- At the commit "error when router GEMM runs before init_batch_invariance": 111 passed, 1 skipped (log archived in `R`).
- At the EP4 oracle commit: 109 passed, 1 skipped (log archived in `R`).
- The router tests added by the build-time router measurement commit (including subprocess positive and negative cases): 6 passed.

**Raw gradient A/A** (whole model, `actor_l3l4.py --no-step`, dapo_0/dapo_1/synthetic_8192, 378 parameter tensors, run twice on the same node, node A)
- At the θ0-floor re-set commit: 0 of 378 tensors differ (receipt in `R`).
- At the EP4 oracle commit (including the DS4 route dx summation): 0 of 378 tensors differ (receipt in `R`).
- Earlier, at the first-review-round commit, only Mamba conv1d/dt_bias differed; fixed by the Mamba determinism commit.

**RL gate** (whole model + BF16 master + fast profile, 3 RL steps, 6 optimizer steps)
- pp4g2 (node B, at the BF16 review fix commit): aborted fail-closed at the θ0 per-tensor floor (see §2; fixed by the θ0-floor re-set commit).
- pp4h2 (node B, at the θ0-floor re-set commit): θ0 check passed (rank0 99.939%), but another user's containers occupied 95 GB each on GPU0/1, so vLLM failed at startup with insufficient memory. Node B was not used after this.
- **pp4k1 (node C, at the EP4 oracle commit, d10 vLLM): PASS** (`R/pp4k1-receipt.json`)
  - θ0/θ2/θ4 old-logprob vs rollout: 0 mismatches (16–19 micro-batches per version, 115–122 thousand tokens); training forward vs the same version: 0/0.
  - The 6 grad norms are bitwise identical across the 4 ranks; master changed; exit 0.
  - update_actor steps 2/3: 57.6 / 31.6 s; peak allocated 75.96 GiB, reserved 87.9 GiB.
- **pp4m1 (node C, at the build-time router measurement commit, delivered vLLM integration branch): PASS** (`R/pp4m1-receipt.json`)
  - vLLM overlaid onto d10 from a staged overlay bundle (bundle-vpr-int6): 26 files, SHA verified per file.
  - θ0/θ2/θ4 old-logprob vs rollout: 0 mismatches (16–17 micro-batches, 112–119 thousand tokens); training forward 0/0.
  - grad norms bitwise identical across the 4 ranks; exit 0; the build-time router row-count probe passed in all 4 processes.
  - KV scales are exported rounded to BF16; still 0 mismatches after the θ1/θ2 weight syncs (the vLLM-side counterexample p2neg already has 131052/131072 tokens mismatching at θ0).
  - update_actor steps 2/3: 30.4 / 29.9 s; peak allocated 75.5 GiB, reserved 87.9 GiB; host peak 826/882 GB (`R/pp4m1-hostmem.txt`).

**Unit tests (delivered vLLM)**: on d10 with `bundle-vpr-int6` overlaid, the vendor-signature-test commit runs 119 passed, 1 skipped (log archived in `R`). The build-time router measurement commit on plain d10 runs 113 passed, 1 skipped. The earlier numbers on plain d10 cannot prove that the delivered stack is fine: `configure_nemotron_humming` has been removed in the delivered vLLM, as Reviewer A r1b P1-2 pointed out.

- **pp4r1 (node C, final PP4 candidate head, delivered vLLM overlay, verl recipe with the host-RAM check run as is, including manifest checks and the host RAM precheck): PASS** (`R/pp4r1-receipt.json`)
  - θ0/θ2/θ4 old-logprob vs rollout: 0 mismatches (16–18 micro-batches, 112–122 thousand tokens); training forward 0/0.
  - grad norms bitwise identical across the 4 ranks; exit 0.
  - update_actor steps 2/3: 33.6 / 29.9 s; allocated 75.9 GiB; container host peak 737.8 GiB (precheck threshold 760 GiB).

**Host RAM** (pp4k1, sampled every 10 s, `R/pp4k1-hostmem.txt`)
- About 440 GB while loading BF16 and doing the θ0 check (16:53). Now the master is read directly to GPU, and loading is no longer the peak.
- The peak occurs at the end of the first optimizer step: node used 811/882 GB, container 734 GB. This is the CPU-offloaded optimizer state, consistent with Reviewer C's judgment, not the loader. Headroom is about 70 GB; if the node has other large processes, step 1 may OOM.

## 4. Review items not yet handled

PP4 candidate (no numerical impact):
- Reviewer A P2-(4): the engine default `export_dtype` conflicts with the protocol; should error at config time.
- P2-(6): the hook fires twice on `load_checkpoint` (idempotent, harmless).
- P2-(7): explicitly reject param_offload.
- P2-(8): source of the 6768 page size, and deduplication of `(128,2688,1856,6)`. 6768 has no constant in the vLLM source; need to find serving's block-size derivation and then write a comment.
- P2-(10): allocate FP8 KV blocks per sequence for each attention layer.
- P2-(11): remaining oracles (FP8 linear vs `FlashInferFP8ScaledMMLinearKernel`, FA4 vs `NemotronHFixedFA4Impl`, θ1 requantize→reload→logprob, vendor function signature check).
- ~~router `aten.mm.dtype`~~: measured on the vLLM side, after `init_batch_invariance()` it is bitwise identical for all M. The candidate now measures this property at build time in every training process, and the router also errors when BI is not initialized. The recipe does not use KL and has no ref model; old-logprob is computed by the actor itself, through the same build_model.
- Capability seeding: still needs to temporarily set `CUDA_VISIBLE_DEVICES`, but now uses the device UUID list enumerated by torch itself, so the correct GPU is found even when the starting list is restricted or reordered. Tests cover narrowing `1,0` to `0`.
  - A solution that does not change environment variables would require changing vLLM: when `get_device_capability` gets a visible ordinal it should use torch's device (or UUID) rather than converting via `CUDA_VISIBLE_DEVICES`. That is outside the Megatron side's scope.
- Host RAM precheck: added in the verl recipe.
- SFT provenance manifest: added `scripts/bf16src/sft/manifest.py`, called by run-a/run-c after `docker wait`. It records the image ID, command, environment variables, per-file sha256 of all read-only mounts plus an aggregate, and the sha256 of output files. The staged run-a on Lustre is updated by atomic replacement. Takes effect on the next SFT run.
- Still not done: bitwise oracle of FP8 linear vs `FlashInferFP8ScaledMMLinearKernel`, oracle of FA4 vs `NemotronHFixedFA4Impl`, comparison of the one-sided reduction vs FlashInfer combine, and centralizing hard-coded geometry into one constants module. Currently backstopped by the 0 mismatches of the e2e gate.
- Reviewer C r5 P1-3: SFT rerun on the integrated HEAD, owned by another track.

EP4 WIP branch (paused at the user's request, untouched):
- Guards and tests for zero receive rows / zero tokens (Reviewer A P2-2, Reviewer C P2-3);
- make `fill_uninitialized_memory` explicit (P2-3);
- per-layer host sync (P2-6);
- dx transport contract: send back per route, sum on the source rank in the declared order (Reviewer A P2-8 / Reviewer B P2-1);
- `ctx.state=None` is incompatible with retain_graph (Reviewer B P2-2);
- DeepEP's `num_rdma_bytes=0` must reject non-intranode groups (Reviewer B P2-3).

## 5. Phase 2: EP4 (new EP4 branch, cut from the PP4 candidate head)

After PP4 validation finished, EP4 was started in the user's order. The old WIP remains on the EP4 WIP branch.

**Commits**
- Port the WIP DeepEP implementation onto the PP4 candidate. Conflicts were resolved with the candidate's semantics: reduction required, restore hook, BF16 master only handles this rank's 32 experts (the θ0 check and release digest likewise only look at this rank).
- Change the backward per the EP review.
  - Expert ranks no longer return per-rank BF16 partial sums; instead they return per-route input gradients: one row per (token, rank), width topk×hidden, with exactly one contributor. The source rank adds them in slot order, rounding to BF16 after each addition (DS4's deterministic scatter backward). This makes **dx bitwise equal to the single-GPU VJP**, asserted in the tests.
  - The backward keeps the saved state, so under `retain_graph` the backward can be run again; there is a test.
  - The cases of 0 tokens on the source rank and no rows received by an expert rank are both handled, and both have test cases.
  - The EP group must be on the same node with at most 4 ranks. No RDMA buffer is allocated; cross-node errors out directly.
  - `fill_uninitialized_memory` is turned off only during DeepEP calls and restored afterwards.
- verl EP4 recipe branch (the old branch was renamed with a `-wip` suffix):
  - EP4 topology. The Dockerfile builds a community DeepEP fork at a pinned commit with DS4's arguments; the manifest records the deepep revision and nvshmem version; the runtime smoke test imports deep_ep.
  - EP4 host RAM floor set to 780 GiB.

**Test environment**: d10 + delivered vLLM overlay (`bundle-vpr-int6`) + deep_ep 1.2.1 (community fork, pinned commit) taken from the DS4 alignment verl image (`<image>`), copied from that image's DeepEP NVL4 overlay and `hybrid_ep_cpp` into a staging directory and loaded via PYTHONPATH. This is only for testing; the delivered image builds it from the Dockerfile.

**Validation**
- Unit tests (delivered vLLM + DeepEP): 121 passed, 1 skipped (log archived in `R`). EP4 DeepEP test coverage: forward bitwise equal to the single-GPU EP4 serving reduction, dx and d_routes bitwise equal to EP1, backward bitwise identical run to run, retain_graph, 0 tokens/0 rows.
- Raw gradient A/A (node A, run twice on the same node, PP1/EP4): 1512 tensors across 4 ranks, 0 differ (receipt in `R`).
- **RL gate ep4g3 (node C, dev profile): PASS** (`R/ep4g3-receipt.json`)
  - θ0/θ2/θ4 old-logprob vs rollout: 0 mismatches (16–20 micro-batches, 113–122 thousand tokens); training forward 0/0.
  - grad norms bitwise identical across the 4 ranks; exit 0.
- **Comparison with PP4 (both dev profile, 8192-token micro-batches)**

| | PP4 (pp4r1) | EP4 (ep4g3) |
|---|---|---|
| update_actor steps 2/3 | 33.6 / 29.9 s | 57.5 / 50.6 s |
| One RL step | 70.6 / 66.4 s | 93.6 / 86.5 s |
| Peak GPU allocated / reserved | 75.9 / 87.9 GiB | 161.5 / 166.3 GiB |
| Container host peak | 737.8 GiB | 764.8 GiB |

  - Under EP4 every rank holds the full dense part of all 52 layers plus 32 experts, with no PP split. Static usage 63 GiB: BF16 params 17.8, buffers 5.1, FP32 main_grad 35.6. Activations and other transient usage at 8192 tokens are about 98 GiB.
- **Formal-length OOM (ep4r1, r16 profile, `PPO_MAX_TOKEN_LEN_PER_GPU=16384`, response 14000)**
  - Rollout and old-logprob both finished. update_actor's training forward OOMed at `HummingRoutedExperts.routes` in some MoE layer: PyTorch had allocated 176.75 GiB, GPU 1 had only 1 GiB left (`R/ep4r1-oom.txt`).
  - Cause 1: with PP1 every rank carries the activations of all 52 layers. Under the dev profile (8192 tokens) transient usage is already about 98 GiB, doubling at 16384 tokens.
  - Cause 2: the per-route state saved by the EP backward is not compressed. For each (token, rank) row, FC1 [rows×6×1856] and the visible output [rows×6×2688] are stored for all 6 slots, including slots that do not belong to this rank. Each token goes to about 3.3 ranks on average, so this part is about 3.3× the PP emulation. At 16384 tokens and 23 MoE layers that is about 68 GB; storing only this rank's routes would save about 47 GB.
  - Rough estimate: compressing only the per-route state still gives about 210 GiB > 184 GiB at 16384 tokens. Fitting the formal length also needs activation recompute (BI kernels are deterministic so recompute is bitwise identical, but the DeepEP dispatch must also be replayed) or CP. **This needs a human decision.**

## 6. Formal length: activation recompute, compressed EP state, and FSDP2

Decision (relayed by the coordinator): follow DS4 with `impl_cfg.recompute=full` and compress the per-route state; then test the formal length, then try DS4's FP32-shard FSDP2.

**Commits** (EP4 branch)
- The EP backward saves only this rank's routes (FC1 and visible output) and expands them in the backward. The no-grad path saves no state.
- `impl_cfg.recompute=full`, directly using DS4's `primitive.recompute.apply_recompute` (reentrant CheckpointFunction; each Block reruns its forward in the backward, and the DeepEP dispatch is replayed with it).
- Cherry-pick the WIP FSDP2 optimizer primitive (replicated FP32 params, `cast_forward_inputs`, consistent with the DS4 aligned branch).
- `impl_cfg.optimizer=fsdp2`, following DS4 `build_training_backend`:
  - Deferred wrap after HF loading; FP32 shards with no separate master; FP32 params replicated; BF16 compute.
  - Dense blocks sharded over DP×CP, experts over expert DP.
  - Quantized layers use `bind_master` to bind the FP32 shard that the optimizer actually updates. The staleness check looks at the versions of both the DTensor and the local tensor: measured, an in-place add on a DTensor only bumps the DTensor's version, not the local tensor's version.
  - On the post-step refresh, the gathered full matrix is cast to BF16 and then requantized. This is exactly the matrix the forward sees after unshard, so scales come from the full matrix and forward and export use the same bytes.
  - FP32 shards are exported in the original model dtype. Grads are freed after the step.
  - Added a 4-GPU test: after an in-place update on a shard, the deployment is judged stale; the refreshed bytes equal the full-matrix requantize result.
- verl: EP4 defaults to `ACTOR_RECOMPUTE=full`, PP4 defaults to none; `ACTOR_OPTIMIZER=dist_opt|fsdp2`.

**Activation recompute is bitwise**
- EP4 whole-model raw gradients: recompute on vs off, 1512 tensors across 4 ranks, 0 differ.
- Peak allocated (dapo samples) drops from 123 GiB to 82 GiB.

**EP4 + dist_opt + recompute, formal length (ep4r2, r16 profile)**
- The first update finished on GPU (the log shows after_to_cuda then after_to_cpu).
- It was then killed by Ray's host memory monitor: the node reached 838.8/882.9 GB, exactly Ray's default 95% threshold. Container host peak 765 GiB, same as the dev profile; other processes on the node resident at about 73–93 GB (`R/ep4r2-ray-oom.txt`, `R/ep4r2-hostmem.txt`).
- This is a host memory headroom problem, not a GPU problem. PP4's node peak is 826–830 GB, also hugging this line.
- Rerun with `RAY_memory_usage_threshold=0.99` (test environment only): **ep4r3 PASS** (`R/ep4r3-receipt.json`; at formal length each RL step has 4 optimizer steps, so the reducer uses `--steps 12`).
  - θ0/θ4/θ8 old-logprob vs rollout: 0 mismatches (544–604 micro-batches, 7.6–8.6 million tokens per version); training forward 0/0.
  - The 12 grad norms are bitwise identical across the 4 ranks.

**EP4 + FSDP2 (DS4 FP32-shard) + recompute, dev profile (ep4f1, node A): PASS** (`R/ep4f1-receipt.json`)
- θ0/θ2/θ4 old-logprob vs rollout: 0 mismatches (16–20 micro-batches, 116–120 thousand tokens); training forward 0/0.
- grad norms bitwise identical across the 4 ranks.

| dev profile | PP4 dist_opt | EP4 dist_opt | EP4 FSDP2 + recompute |
|---|---|---|---|
| update_actor (steps 2/3) | 33.6 / 29.9 s | 57.5 / 50.6 s | 35.7 / 23.9 s |
| One RL step | 70.6 / 66.4 s | 93.6 / 86.5 s | 74.2 / 63.1 s |
| GPU peak allocated | 75.9 GiB | 161.5 GiB | 141–144 GiB |
| Container host peak | 738 GiB | 765 GiB | 441 GiB |

- FSDP2's Adam runs on GPU without CPU-offloaded FP32 state, so host memory drops sharply and the update is also faster.
**Formal length (r16 profile: 16384-token micro-batches, response up to 14000)**

| | EP4 dist_opt + recompute (ep4r3) | EP4 FSDP2 + recompute (ep4f2) |
|---|---|---|
| Result | PASS (needs Ray threshold 0.99) | PASS (default Ray threshold) |
| update_actor (steps 1/2/3) | 590 / 475 / 521 s | 571 / 462 / 515 s |
| One RL step | 1026 / 880 / 971 s | 1029 / 879 / 984 s |
| GPU peak allocated / reserved | 87.7 / 100.8 GiB | 153.2 / 161.1 GiB |
| Container host peak | 762.7 GiB (node 840 GB) | 451 GiB |

- ep4f2: θ0/θ4/θ8 old-logprob vs rollout 0 mismatches (536–600 micro-batches, 7.56–8.43 million tokens per version); training forward 0/0; the 12 grad norms bitwise identical across the 4 ranks (`R/ep4f2-receipt.json`).
- At formal length the two backends take similar time, dominated by the update (one step is about 1.95M tokens/rank). FSDP2 uses more GPU memory, but over 300 GiB less host memory, and does not trigger Ray's memory protection.
- Whether PP4 needs activation recompute at formal length: answered by the packaging agent's PP4 default-workload acceptance on another node (node D). I have no PP4 formal-length data myself.

**FSDP2 raw gradient A/A (node C, same node, EP4 + recompute)**
- Ran three times (a/b/c): a and c are bitwise identical; b differs from both a and c only in 19 elements of the rank0 shard of `backbone.embeddings.weight`, relative difference 1.8e-7 (receipt in `R`).
- A standalone test of `nn.Embedding`'s backward is identical over 20 reruns; the three dist_opt A/A groups are also all 0. So the nondeterminism comes from the FSDP2 path.
- Turning off NVLS (`NCCL_NVLS_ENABLE=0`) and running three more times gives the same result (the same 19 elements). NVLS is not the cause.

**Cross-process gradient nondeterminism: fixed on the PP4 candidate**
- Symptom: doing A/A with only a single short sample (dapo_0, 1224 tokens), the loss is bitwise identical between two runs but gradients differ widely (PP4 dist_opt: 365 of 378 tensors differ; the same for both EP4 backends). A/A with 3 samples packed is usually identical by luck.
- There are two sources, both "each process picks a kernel config by timing":
  1. **mamba_ssm SSD backward.** mamba_ssm decides the autotune candidates of each Triton kernel at import time (`mamba_ssm.utils.determinism.autotune_configs`): in deterministic mode only the cheapest one is kept, otherwise all are kept and each process selects by timing.
     - The mode is read from `MAMBA_DETERMINISTIC` at the moment of import. The `set_deterministic_mode(True)` we called before came too late: `from mamba_ssm.utils.determinism import ...` first executes the package's `__init__` (→ `modules.mamba2` → `ops.triton.ssd_*`), and the kernel decorators have already been evaluated by then.
     - This is why a standalone probe that does not import vLLM also sees 6–9 candidates: measured with the deterministic switch True and `TRITON_CACHE_AUTOTUNING` unset, the dx kernel still has 9 configs.
     - In addition, even with deterministic mode on at import time, `TRITON_CACHE_AUTOTUNING=1` (which vLLM's `env_override` sets to 1 by default) also keeps all candidates and hands them to Triton's cached autotune.
     - Fix: build_model temporarily sets `MAMBA_DETERMINISTIC=1`, `TRITON_CACHE_AUTOTUNING=0` while importing mamba_ssm, and restores them immediately after import. It then checks that every SSD autotuner has only 1 config left; if mamba_ssm was already imported without pinning, it errors out. `set_deterministic_mode(True)` is still called afterwards; it selects non-atomic reductions at runtime.
     - The pinned mamba_ssm (0048fbf) has no more formal interface than the environment variable: the mode is decided only at import time by env or `torch.are_deterministic_algorithms_enabled()`. The latter switches the whole process to deterministic algorithms, which has a wider blast radius, so it was not used.
     - Tests: SSD backward bitwise identical across 3 independent processes; importing the unpinned version first raises an error.
  2. **Inductor-compiled RMSNorm VJP.** After fixing SSD, per-module backward gradient digests (the A/A harness's `AA_TRACE`) located it: a block norm's output gradient was identical while its input gradient differed. Inductor by default benchmarks candidate reduction kernels per process. Fix: the three VJPs use `torch.compile(..., options={"deterministic": True})`, scoped to these three functions only, without changing global configuration.
- Other kernels on the training path: the A/A harness lists Triton autotuners that this process actually autotuned and that have more than 1 candidate; after the fix the list is empty. causal_conv1d is a CUDA kernel; vLLM's visible kernels are pinned by BI; `_visible_combine` is an elementwise forward.
- **Validation** (PP4, delivered vLLM, each group run twice in independent processes):

| Sample | Before fix | After fix |
|---|---|---|
| dapo_0 (1224 tokens) | 365–376/378 differ | 0/378 |
| dapo_1 (1170 tokens) | 370/378 differ | 0/378 |
| synthetic_8192 (9216 tokens) | 0/378 | 0/378 |
| 3 packed (11610 tokens) | 0/378 | 0/378 |

- **Time** (3 packed, fwd+bwd, 4 repetitions, two rounds):
  - Steady state (repetitions 2–4): 1.13–1.19 s before the fix, 1.12–1.16 s after; no measurable overhead.
  - First call drops from 367–383 s to 152–161 s: autotune benchmarking is no longer done in-process; this includes compile time.
- Full-model fast-profile gate (p5 image + the fixed tree): see below.
- **New PP4 candidate head** (two fix commits — the mamba_ssm import pinning and the deterministic Inductor VJPs — plus one docs-only commit). The EP4 branch has cherry-picked them.

## 7. Images and environment

- **Three pitfalls running the recipe on d9**:
  - The vLLM version string is `0.1.dev1`, rejected by verl's version gate;
  - omegaconf `{}` defaults;
  - Ray device ordinal: the old code would `IndexError` already at vLLM `init_distributed_environment`.
  - The first two were fixed by a verl recipe commit.
- **NCCL in the unit-test container**:
  - The cloud-provider NCCL NET plugin cannot find the NIC inside the container; test workers switch to spcx;
  - NCCL variables exported by vLLM's `init_batch_invariance` pollute child processes; test workers restore the environment from collection time.
- Ray occasionally reports "node timed out during startup"; rerunning is enough.
