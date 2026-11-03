# Host memory for checkpoint save/restore (PP4 dist_opt and EP4 FSDP2, GB200 cluster, 2026-10-04)

## Conclusion

- **PP4 dist_opt (fallback)**: after the fix, save → restore works under the fast profile (run ckoom-stage-pp4).
  - Save peak 811 GiB. The original was killed by Ray at the first save.
  - On restore, load-phase peak 734 GiB, first step after restore 825 GiB. With only the DS4 port and without this round's new commit, these two numbers are 849 and 883 GiB.
  - After restore `rollout_probs_diff_max = 0`, and global_step continues from 3.
  - The formal load did not finish: the node went down just as the step 5 save started; see "Not done".
- **EP4 FSDP2 (main configuration)**: saving does not OOM.
  - Fast profile: save peak about 625–644 GiB, 444 GiB when not saving.
  - Formal load: 631–637 GiB while saving at step 5, about 250 GiB below the 892.8 GiB limit.
  - After restore, rollout vs training logprob diff is 0 and the Adam step is continuous. Both the old EP4 FSDP2 fix branch and the restructured DS4-form checkpoint branch pass under both the fast and the formal load.
- **After the DS4-form restructure (audit P0: EP4 FSDP2 expert banks have duplicate coordinates in DCP)**: fixed on the DS4-form checkpoint branch; see the "DS4-form" section.
  - The new 4-rank round-trip test fails on the restructure base commit (rank 2's up_proj master is wrong) and passes on this branch.
  - Save → restore PASSes in both quick_alignment_test and aligned (formal) modes: after restore `rollout_probs_diff_max = diff_mean = 0`, Adam step continuous.
- **Newly found delivery risk**: after 06:00, dist_opt saves writing directly to Lustre failed repeatedly with `inline_container.cc:672 unexpected pos 704 vs 598`.
  - The control used the same image that had saved 4 times in a row successfully at 05:00, and it now failed too, so it is not caused by this round's code.
  - Two of my nodes went down during checkpoint writes (one at 07:11, another at 08:04).
  - With #196's local staging (`MLITE_DCP_LOCAL_STAGE_DIR`) turned on, both save and restore succeed. Recommend turning it on by default in the recipe (needs a verl change, not done this round).
- **The limit must be computed as 892.8 GiB, not 956**: on this cluster a job's cgroup limit is 892.8 GiB (Slurm RealMemory 914192 MiB). Ray kills processes at 0.95 of that, about 848 GiB. The original error text: `854.52GB used / 892.77GB limit`, where GB is actually GiB.

## What the 740 GiB baseline consists of (PP4, original p6, after the first update, independent of load)

Data sources: the job cgroup's memory.stat and each WorkerDict's smaps (observer, read-only /proc). Total about 740 GiB = anonymous pages 457 + locked shmem 206 + reclaimable file cache 76.

| Item | Amount | Attribution |
|---|---|---|
| Paged memory of 4 actors | about 380 GiB (92–99 GiB per rank, 2.4 GiB large mappings) | HDO's FP32 master + Adam exp_avg + exp_avg_sq, i.e. 12 B/param × 8.2B params/rank, times 4 |
| CUDA pinned memory of 4 actors | about 206 GiB (about 52 GiB per rank, /dev/zero 4 GiB slabs) | HDO's FP32 gradient staging (4 B/param, about 30.6 GiB/rank), plus the BF16 pinned copy created and then discarded during HDO initialization (about 15 GiB/rank, kept in the CUDA host caching allocator and never returned) and rounding overhead |
| Other anonymous pages | about 60–75 GiB | 4 vLLM workers at about 3.2 GiB each, Ray, TaskRunner, AgentLoop, etc. |
| File cache | about 76 GiB | Model copy on the node's /tmp, etc., reclaimable |

- The breakdown of the pinned item is inferred: the total is measured, the parts are estimated from parameter counts.
- I tried merging the pinned gradients into large allocations; measured shmem only dropped from 203.5 to 199.6 GiB, so it was withdrawn and is not on the branch.
- The one item that can really be saved is the BF16 pinned copy from HDO initialization (about 61 GiB left in the cache), not changed this round.

## Root cause and fix (PP4 dist_opt)

| Phase | Root cause | Fix |
|---|---|---|
| Save | `fully_reshardable` first gathers the whole parameter state into newly allocated FP32 host buffers (param/exp_avg/exp_avg_sq, ≥3x) | Own commit: switch to Megatron's default `dp_reshardable`, which hands over the optimizer's own tensors directly; on load, read according to the type recorded in the checkpoint, so old checkpoints still load |
| Save | dist_checkpointing copies the GPU model to pinned memory with non_blocking D2H, rounded up to powers of 2, and the allocator does not return it (a previous agent measured +119 GiB on another cluster) | Own commit: before saving, copy the main replica to pageable memory, and free it after save returns. No separate ablation on this cluster this round |
| Load | HDO's `load_state_dict` post-hook rebuilds the sub-optimizers and clones every parameter again (pinned BF16 + FP32) while the old copies are still referenced. dist_opt loading rebuilds twice: once in `sharded_state_dict(is_loading)`, once in the real load. After the rebuild the old FP32 master stays attached to the state until the next step | New commit **HDO reuse**: reuse each parameter's original copies when rebuilding. The copy that serves as the FP32 master is not refreshed from BF16, because `_update_fp32_params_by_new_state` overwrites it with the master right after; the other copies are refreshed as usual, giving the same result as a fresh clone. Measured load-phase peak drops from 849 to 734 GiB, first step after restore from 883 to 825 GiB |

## Measurements (cluster, fast = dev-profile + SAVE_FREQ=1, 4×GB200 per node)

| Run | Megatron in the image | Save peak | Restore: load phase / first-step peak | Result |
|---|---|---|---|---|
| ckoom-base-fast | p6 Megatron commit, original | 793 → killed by Ray | — | OOM at the first save |
| ckoom-fix1-fast | fix1 (2 own commits + #195/#196/#237/#236) | 816–823 | 849 / 883 | 4 saves succeeded, probs_diff_max = 0 at step 3 after restore |
| ckoom-stage-pp4 | fix1 + HDO reuse + malloc_trim*, `MLITE_DCP_LOCAL_STAGE_DIR=/stage` | 811 | **734 / 825** | 4 saves succeeded, probs_diff_max = 0 at step 3 after restore, grad_norm 0.245 |
| fix2b, fix3, obs, bis-reuse, ctl-fix1 (06:04–07:58) | Various combinations, including an image identical to fix1 | — | — | Lustre write failure at the 1st or 2nd save |
| ckoom-fin-formal, formal load | fix1 + HDO reuse + malloc_trim, no staging | Steady at 739–748 for steps 1–4 | — | The node went down as the step 5 save started, at 812 GiB |

\* The `malloc_trim` commit measured no effect: 775 vs 778 GiB after saving; withdrawn from all branches. It only affects the dist_opt path, so it does not affect the conclusions above.

Evidence of restore continuity:

- End to end:
  - verl loads from global_step_2; `Loaded dist_opt checkpoint ... at step 2`; the next global_step is 3.
  - The first rollout after restore has 0 logprob diff against training (`training/rollout_probs_diff_max = 0.0`), showing that the weights deployed to rollout are exactly the restored actor weights.
- Optimizer state, bitwise:
  - New unit test `test_hdo_dp_reshardable_load_restores_state_in_place` (4 ranks, in the image) checks, after a dp_reshardable save → load into a new HDO:
    - master, exp_avg, exp_avg_sq bitwise identical;
    - host copy objects not reallocated;
    - still bitwise identical on both sides after one more step.
  - It fails without the fix (verified).
- Rollout resamples, so the step 3 data differs from A, and step 3 metrics cannot be compared bitwise.
- The end-to-end hash comparison with the optauth observer did not complete: the obs run hit the Lustre write failure above.

## EP4 FSDP2 (the EP4 FSDP2 fix branch)

- Save (ckoom-ep4-fast A, one job, one node, a diagnostic image = fix + one diag commit that only acts at load):
  - all 3 saves succeeded;
  - peak 643.6 GiB, steady at 571 GiB after saving;
  - 444 GiB peak when not saving (ep4c-fast2).
- `rollout_probs_diff_max` is 0 at steps 1–3.
- Phase B of that job loaded successfully, log `Loaded training checkpoint ... at step 2`. The diag commit then immediately re-saved a copy after loading, for an offline byte-by-byte comparison; the node went down during the re-save (07:11), and the comparison did not complete.
- **Restore**: ckoom-ep4-resume (one job, the same node, clean image, `MLITE_DCP_LOCAL_STAGE_DIR=/stage`), phase B only, restoring from A's global_step_2 above. PASS:
  - load-phase peak about 418 GiB, peak at step 3 after restore (with a save) 624 GiB;
  - next global_step is 3, `rollout_probs_diff_max = 0`;
  - global_step_3 saved;
  - Adam step count continuous: in optimizer_rank_0.pt, global_step_2 has 4; the restored global_step_3 has 6, the same as the uninterrupted A.
- **Formal load**: ckoom-ep4-formal (one job, another node, clean image, recipe default load, SAVE_FREQ=5, `MLITE_DCP_LOCAL_STAGE_DIR=/stage`). PASS:
  - host peak 445 GiB at steps 1–4;
  - peak 631 GiB while saving at step 5, 515–620 after saving;
  - step 6 after restoring from global_step_5 has `rollout_probs_diff_max = 0`, grad_norm 0.085, phase B peak 434 GiB.
  - Note: ckpt-run's SIGINT to stop srun did not take effect, so A ran on to step 7, which does not affect the step_5 checkpoint; afterwards switched to scancel to stop my own job step.
- The extra ~127 GiB after saving:
  - is not the glibc heap, since malloc_trim had no effect on PP4;
  - presumably related to the CPU staging of GPU tensors in `torch.save` during the DCP save, not located;
  - measured not to be an OOM risk: about 250 GiB below the limit.

Relevance of the PRs (Megatron fork PRs):

- **#235** (EP-local expert key collision) applies directly:
  - the FSDP2 path stores experts as DTensors, and a DTensor describes only DP sharding, not EP identity. Without the fix, the 32 experts of the 4 EP ranks land on the same DCP key.
  - After the fix keys carry `_ep{size}_rank{rank}`. **Restore must use the same EP topology (EP=4)**; a different topology errors out directly instead of silently loading only part.
  - The optimizer state is one torch.save file per rank, which also requires the same rank mapping.
- **#234**:
  - offloaded native Adam moments are local tensors when loaded back and must be rebuilt in the owning param's DTensor layout;
  - it also clones the step counter on CPU, so that one scalar does not keep the whole mmap alive (together with #195).
- **#236**:
  - with FSDP2 and `optimizer_offload=True`, the optimizer must be moved back to CPU after loading; otherwise it stays on the GPU;
  - a no-op for PP4's HDO (offload_fraction=1).
- **#195**: rank-local optimizer files are loaded with mmap, dcp/FSDP2 path only.
- **#196 / #237**: optional local staging, and #196's stream-close fix. Given the Lustre write failures above, recommended on.
- **#197**: an equivalent implementation is already in the baseline (two baseline commits).
- **#226**: confirms Nemotron takes the dist_opt path; no checkpoint memory changes.
- **#233**: DS4 rollout Top-K, unrelated to checkpoints.
- **#232** (carried in the baseline of #234/#235): ChunkedOptimizerStateOffloader master residency issue, unrelated to both HDO and FSDP2, not ported.

Relation to the restructure:

- Cherry-picking the 13 commits of the EP4 FSDP2 fix branch in order onto the DS4-form restructure base commit had no conflicts (temporary checkout, deleted).
- The restructure commit that requantizes from the master after loading only touches nemotron_h, and replaces the post-load hook with a post-optimizer-step hook. #236's `finally` block can coexist with it and there is no semantic conflict: requantization reads the restored master, while #234/#235 handle the moment layout and keys.

## DS4-form (the DS4-form checkpoint branch, on the restructure base commit, audit P0)

Problem:

- The restructured FSDP2 wraps each EP rank's expert bank on an expert-DP mesh containing only itself (4 single-element meshes at EP4).
- `_dcp_tensor_from_param` uses a parameter's own mesh directly if it is already a DTensor, and that mesh carries no EP identity. So the banks of all 4 ranks are written into DCP as `[32,…]` at offset 0, with duplicate coordinates.
- The result is a wrong restored master; the post-optimizer hook then requantizes from those masters, so the deployed bytes are wrong too.
- This is exactly the kind of problem #235 fixes.

Fix and commits:

- The first 13 commits are an in-order cherry-pick of the EP4 FSDP2 fix branch, no conflicts. They only touch shared mLite code (primitive/ckpt, optimizers, verl_mlite engine) and mcore HDO, not nemotron_h:
  - dp_reshardable
  - pageable staging
  - #195
  - #196
  - #237 (2 commits)
  - #236
  - HDO reuse
  - #234 (2 commits)
  - #235 (3 commits)
- Test commits added to adapt to the restructure:
  - one commit: the engine's checkpoint-load test gets `optimizer_offload` (#236 reads this field);
  - one commit: the stub ParallelState in the DCP sidecar test gets `ep_size` (#235 reads this field).
- New test `test_ep4_fsdp2_checkpoint_restores_each_ranks_expert_bank` (one commit), flow:
  1. 4 ranks, synthetic NVFP4 checkpoint, 8 experts, 2 per rank, each rank's bank different;
  2. wrap with the real `Nvfp4ExpertWeights` and `build_fsdp2_training_optimizer` into expert-DP single-element meshes;
  3. take one optimizer step, then run `_post_optimizer_step`, i.e. requantize from the master;
  4. save, rebuild from θ0 and load, then run the hook;
  5. bitwise compare each rank's FP32 master shard, Adam state (exp_avg / exp_avg_sq / step) and `export_quantized` deployed bytes.
- Verification of the new test:
  - fails on the restructure base commit with `AssertionError: (2, 'up_proj')`; passes on this branch.
  - Tested on the same path as the restructure's checkpoint load (running the post-optimizer hook after loading, requantizing from the master), consistent, no conflict.

Unit tests (the ds4form candidate image with this branch's sources mounted, one node):

| Scope | Result |
|---|---|
| `test_nemotron_h_training_vjp.py` whole file | 67 passed |
| runtime/checkpoint | 32 passed, 2 skipped |
| examples | 43 passed, 1 failed (hf_model test, fails on baseline too) |
| fsdp2 | 40 passed |
| HDO | 74 passed |
| #235's torchrun 4-rank tests | 2 passed |

End to end (dev image `<image>` = DeepEP base + a vLLM fork commit + this branch + a verl fork commit; recipe used as is with `--hardware gb200 --mode quick_alignment_test`, setting only `SAVE_FREQ=1` and `MLITE_DCP_LOCAL_STAGE_DIR=/stage`):

- ckoom-ds4f-fast (one job, one node) PASS.
- Phase A:
  - 3 steps, 3 saves;
  - `rollout_probs_diff_max = diff_mean = 0` at every step;
  - peak 626 GiB.
- Phase B restores from global_step_2:
  - step 3 `rollout_probs_diff_max = diff_mean = 0`, grad_norm 0.316;
  - peak 629 GiB;
  - Adam step in optimizer_rank_0.pt: global_step_2 has 4, A's global_step_3 has 6, B's global_step_3 also has 6.
- Formal load ckoom-ds4f-formal (`--mode aligned`, recipe default SAVE_FREQ=5, one job, the same node) PASS:
  - steps 1–4 `rollout_probs_diff_max = diff_mean = 0`;
  - host peak 637 GiB while saving at step 5 (about 445 GiB at steps 1–4);
  - step 6 after restoring from global_step_5 has `diff_max = diff_mean = 0`, grad_norm 0.122, phase B peak 451 GiB.

## Branches (committed locally only, not pushed; commit trailers added)

| Branch | Based on | Commits (in order) |
|---|---|---|
| p6 host-OOM fix branch | the p6 Megatron commit | dp_reshardable; pageable staging; #195; #196; #237 (2 commits); #236; HDO reuse |
| p5 host-OOM fix branch | the p5 Megatron commit (with two determinism commits removed) | 2 own commits (dp_reshardable, pageable staging), #195, #196, #237 (2 commits), #236, HDO reuse |
| EP4 FSDP2 fix branch | the EP4 dist_opt tree | the 8 above, plus #234 (2 commits), #235 (3 commits), one commit adding ep_size to a test stub |
| DS4-form checkpoint branch | the DS4-form restructure base commit | cherry-picks of the 13 above, plus 3 test commits |
| EP4 FSDP2 diag branch | the older fix + one diag commit | diag: re-save after load, not delivered |

Whether each commit is necessary:

- dp_reshardable and HDO reuse: backed by measurements.
- Pageable staging: based on the +119 GiB a previous agent measured on another cluster; no separate ablation on this cluster this round.
- #195, #236, #234, #235: no effect on PP4's OOM, but necessary or relevant for EP4 FSDP2 restore.
- #196, #237: guard against Lustre write failures (the env must be turned on).

Unit tests (in the image):

- `test_optimizer_cpu_offloading.py` 74 passed.
- dist_checkpointing `-k "hdo or chunked_offload or dp_reshardable"`: 7 passed. One more failed because it requires 8 GPUs.
- lite checkpoint and FSDP2: 67 passed, GPU cases 40 passed.
- `test_hf_model_only_save_uses_protocol_and_writes_hf_metadata` already fails on the baseline (`export_dtype` argument mismatch), unrelated to this change.

## Not done / risks

- **Lustre write failures**: the error is the torch zip writer's position mismatch at `writeEndOfFile`, i.e. Python's `write` raised.
  - No OST was full (≤81%). At failure the cgroup was about 780–790 GiB, below the limit.
  - Root cause not located. Mitigation is #196's local staging, verified once.
- **Two node crashes**: both during checkpoint writes (one node at 07:11 and another at 08:04, sinfo shows `Not responding`). Cause unknown; needs the admins to look. Until the cause is understood, large checkpoint writes may bring nodes down.
- **PP4 formal load**: no result for the step 5 save and restore: the node went down, and since PP4 is the fallback it was not rerun.
- **Memory that can still be saved**: the BF16 pinned copy from HDO initialization stays in the CUDA host cache, about 61 GiB, not changed this round.

Data:

- Each run is under `$WORKDIR/runs/ckoom-*`, including run.log, hostmem, meminfo, cgroup.txt, procs.txt.
- Scripts are in `$WORKDIR/ckpt-oom/`: ckpt-run.sbatch, memsample3.sh, smaps_probe.sh, cmp_roundtrip.sh.
