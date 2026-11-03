# Long formal runs of the delivery recipe (GB200 cluster, 2026-10-04)

Status: phase 1 (r5 image, about 7 hours) **finished**. All three runs exited during checkpoint writes because Lustre filled up (ENOSPC), not as a planned stop. Phase 2 (resume on a new image to 100 steps) in progress, see the end.

## Conclusion (phase 1)

- **rollout_probs_diff**: 93 training steps across the three runs (including resumes), max = mean = 0 at every step, `valid` always 1.
- **Resume**: kill → `resume_mode=auto` works. Loaded from global_step_10; step 11 after resume has diff 0, and its metrics are close to the uninterrupted run. It then ran 28 more steps in a row, saving 5 checkpoints along the way.
- **Training trend is normal**:
  - mean score rises from about −0.1 to 0.4–0.58;
  - mean response length drops from about 7700 to 4400–5900, truncation rate from about 0.32 to near 0;
  - grad norm stays in 0.057–0.100, no divergence;
  - step time falls as responses get shorter, from about 900 s to 510–650 s (FSDP2); PP4 from about 1100 s to about 750 s.
- **Host memory**:
  - FSDP2: 436–455 GiB normally, 629–657 GiB while saving, no later growth.
  - **PP4: 755–775 GiB normally, up to 844.3 GiB while saving (step 20)**, only about 4 GiB below Ray's kill line of about 848 GiB. Long PP4 runs are at OOM risk.
  - While saving, cgroup `memory.current` hits 892.7 GiB; the excess is page cache.
- **GPU memory**: FSDP2 peak 182.7 GiB (of 184.3), PP4 peak 175.0 GiB.
- **Checkpoints**:
  - FSDP2: 353 GB each, 15 files in 5 file groups, 78–96 s to save;
  - PP4: 429 GB each, 117–133 s to save;
  - Staged on node-local disk; no Lustre write errors and no node crashes (until the disk filled).
- **Failures**:
  - 21:34 (run 1, save at step 30), 22:26 (PP4, save at step 25), 23:57 (run 2, save at step 40) all reported `OSError: [Errno 28] No space left on device`, at `os.sendfile` in `local_stage.publish_staged_file`.
  - At the time total Lustre usage was 99%, with several OSTs having only 23–76 GB left.
  - My checkpoints are not rotated (see event 8) and took about 6.2 TB in total, one of the causes. The rest of the usage came from elsewhere.
  - Run 1 and PP4 stopped at around the 7 hours I had set (6h24m / 6h46m), but not as a planned stop; run 2 stopped 5h48m into its resume.
  - All 3 jobs show COMPLETED 0:0 in Slurm. This is because the launcher captures the exit code itself; the recipe's real exit code is 1, see `exit-code.raw`.
- **User-visible problems** (by severity):
  1. `max_actor_ckpt_to_keep` has no effect under the mLite engine; the default 100 steps accumulate 7–8.6 TB, and it crashes on save once the disk is full;
  2. PP4 dist_opt rollout util: neither 0.7 nor 0.65 starts (the new recipe has changed it to 0.55);
  3. PP4 has only about 4 GiB of host memory headroom while saving;
  4. Restarting for resume overwrites the metrics jsonl;
  5. VAL_FILES cannot be a large file; `val_batch_size` defaults to the whole dataset.

## Configuration

- Image `<image>` (the ds4form candidate image used for r5, sha256 a1fca9e6…, recomputed at 15:08 and matching). Contains only delivery code, no observer.
  - Image content check (one job, one node): vLLM changed files 0 diff; megatron/lite, verl_mlite and recipe identical to the sources; megatron/core and verl differ only in README/.md and the compiled helpers .so.
  - Source directory check: the Megatron commit (megatron/core, experimental/lite) and verl commit (verl, recipe) exported with `git archive` are identical to the directories in the source checkout; the 18 changed vLLM files are byte-identical to the pinned vLLM commit.
- Entry point: the recipe in the image run as is, `--hardware gb200 --mode aligned`, i.e. the recipe defaults: 100 steps, SAVE_FREQ=5, TEST_FREQ=10, `trainer.resume_mode=auto` (verl default), `max_actor_ckpt_to_keep=2`.
- Only the recipe's own env is set:
  - `WANDB_MODE=disabled`, logger console+file;
  - `MLITE_DCP_LOCAL_STAGE_DIR=/cache/dcp-stage`, i.e. node-local `<node-local-disk>/cache`. The recipe comment asks cluster launchers to set this; the checkpoint still lands in `<run>/checkpoints/` on Lustre.
  - The PP4 run additionally sets `ACTOR_OPTIMIZER=dist_opt ACTOR_PP=4 ACTOR_EP=1`.
- Launcher `results/long-run/run-long2.sbatch`: the same as the ds4form reval launcher, plus:
  - read-only sampling: host memory, cgroup, GPU memory (every 10 s), checkpoint du (every 120 s);
  - a separate run log per job.
- Validation set: `VAL_FILES` uses AIME-2024 in DAPO format (960 rows, a DAPO-format AIME-2024 parquet from an earlier e2e test), for the reason in "Events" item 1.

## Events (chronological)

1. 15:08 first submission (three jobs): `VAL_FILES` followed the reval setup and pointed to the training parquet (1.79 million rows).
   - verl printed `Size of val dataloader: 1`: the whole file is one validation batch. At step 10 it would generate for 1.79 million prompts, which in practice can never finish.
   - 15:10 cancelled my own 3 jobs (still initializing) and resubmitted with AIME-2024 (three new jobs).
   - Impact on users: the recipe's aligned default is TEST_FREQ=10, but there is no hint about VAL_FILES size, and `val_batch_size` defaults to the whole dataset.
2. **15:18 PP4 dist_opt failed to start (one job, one node)**: vLLM reported at startup `Free memory on device cuda:3 (125.03/184.31 GiB) on startup is less than desired GPU memory utilization (0.7, 129.01 GiB)`, cuda:1 and cuda:2 were 128.76 / 128.78 GiB, exit 1.
   - Cause: the recipe's default `ROLLOUT_GPU_MEMORY_UTILIZATION=0.7` for dist_opt is the PP4 value from the p6 era. Under the ds4form delivery code, the PP4 actor takes 56–60 GiB on each of GPU1–3 at startup (GPU0 49 GiB), so 0.7 does not fit.
   - Adding `ACTOR_OPTIMIZER=dist_opt ACTOR_PP=4 ACTOR_EP=1` to the recipe as is will fail. The recipe header comment lists only PP1/EP4 as a validated hardware configuration.
   - Handling: added only the script's own `ROLLOUT_GPU_MEMORY_UTILIZATION=0.65` (equal to the fsdp2 default) and resubmitted. Failed directory `runs/lr-pp4-distopt-fail-gpuutil07/`.
3. **15:35 PP4 dist_opt still failed with 0.65**: this time vLLM's FlashInfer autotune warmup OOM'd.
   - Location: `flash_attn_fixed_split` allocating an 8 GiB `out_partial`.
   - At the time the vLLM process held 124.6 GiB (over the 0.65 budget), the actor processes 54–58 GiB, and GPU3 had only 1.6 GiB left.
   - This is the same as what the recipe comment describes for fsdp2 at 0.7.
   - Resubmitted with 0.55; failed directory `runs/lr-pp4-distopt-fail-gpuutil065/`.
   - Conclusion: the dist_opt branch of the ds4form delivery recipe (default 0.7) cannot run directly under PP4; PP4 needs at least about 0.55.
   - The coordinator then confirmed: both 0.7 and 0.65 fail under PP4, already known from the ds4form r3 review fixes (not included); PP4 save → restore was validated with 0.55. So this PP4 run **carries the override `ROLLOUT_GPU_MEMORY_UTILIZATION=0.55`**.
   - The 0.55 job started successfully.
4. 16:38 step 5: the two FSDP2 runs each saved one checkpoint, 353 GB each, taking 96 s / 94 s.
   - Host memory (MemTotal−MemAvailable) peak 642 GiB.
   - cgroup `memory.current` hit 892.7 GiB, i.e. the limit, while saving. The excess is page cache from writing the checkpoint, reclaimable; no OOM.
5. 17:26 PP4 step 5: checkpoint 429 GB, 132 s, host memory peak 822.5 GiB. Ray kills processes at about 848 GiB, so the headroom is only about 26 GiB.
6. 18:09 planned `scancel` of run 2's job: at this point the run had finished step 11 and was running step 12; the latest checkpoint was global_step_10.
   - The metrics of the first segment saved separately as `phaseA-<job-id>.jsonl`. Reason: the recipe's file logger opens the jsonl with `wb`, so a restart overwrites it. This is a user-visible problem: auto resume wipes the earlier metrics file.
   - 18:10 resubmitted on the same node with exactly the same env and arguments; verl's default `resume_mode=auto` resumes.
7. **18:42 resume PASS**:
   - The log shows in order `Found checkpoint … global_step_10`, `Resuming from …/global_step_10`, `[megatron.lite] Loaded training checkpoint … at step 10`; the first resumed step is global_step 11.
   - 32 minutes from resubmission to the end of step 11: about 18 minutes of copying the model to the node, initialization and loading; step 11 itself 861 s.
   - Step 11 after resume compared with the other two step 11s:

| Metric | Run 1 (uninterrupted) | Run 2 before kill | Run 2 after resume |
|---|---|---|---|
| rollout_probs_diff max / mean | 0 / 0 | 0 / 0 | **0 / 0** |
| score mean | 0.0790 | 0.0677 | 0.0677 |
| response length mean | 7088 | 7253 | 7174 |
| grad norm | 0.0769 | 0.0672 | 0.0939 |
| entropy | 0.550 | 0.550 | 0.555 |
| ppo_kl | 0.0040 | 0.0043 | 0.0039 |

   - Conclusion: the metrics after resume are close to the other two, not bitwise identical, as expected.
   - Why they cannot be bitwise identical: see below.
8. **18:52 checkpoint retention does not take effect**: after run 1 saved at step 15, global_step_5, 10 and 15 were all still there, 1059 GiB in total.
   - Cause: the recipe sets `trainer.max_actor_ckpt_to_keep=2`, but the mLite engine's `save_checkpoint` simply does `del … max_ckpt_to_keep` (`verl_mlite/engine/mlite_engine.py:456`) and does not rotate at all.
   - Impact: every 5 steps adds 353 GB (FSDP2) or 429 GB (PP4); running the default 100 steps, each run accumulates about 7–8.6 TB. Users must clean up themselves.
   - This time Lustre had 26 TB free and 7 hours was expected to accumulate about 6 TB, so nothing was deleted by hand during the runs; only the last checkpoint is kept after they finish.
9. After 19:44, every step of runs 1 and 2 was normal, PP4 also normal, with validation every 10 steps.
10. **21:34 run 1 exited with ENOSPC while saving at step 30**. Step 29 had finished, the last complete checkpoint is global_step_25; global_step_30 had written only 118 GB, with no `.metadata`.
11. **22:26 PP4 exited with ENOSPC while saving at step 25**. `publishing staged checkpoint shards failed on another rank`; all ranks errored together. The last complete checkpoint is global_step_20.
12. **23:57 run 2 exited with ENOSPC while saving at step 40**. The last complete checkpoint is global_step_35.
    - In all three failures, r5's "all ranks fail together and `latest_checkpointed_iteration` does not advance" worked: the tracker still points to the previous complete checkpoint, so resuming is direct.
13. 10-05 00:12 (coordinator's instruction) cleanup: kept only the checkpoint the tracker points to for each run (25 / 35 / 20), deleted the other 15 directories, about 5.0 TB in total. List in `results/long-run/deleted-checkpoints.txt`.
    - After deletion Lustre had 6.9 TB free (94%).
    - The node-local working directories (model copies, compile caches, staging directories) were not cleaned.

### Per-step data

`results/long-run/run1-fsdp2-<node>-steps.txt`, `run2-fsdp2-resume-<node>-phaseA.txt` / `-phaseB.txt`, `run3-pp4-distopt-<node>-steps.txt`; columns are diff, score, length, truncation rate, grad norm, per-phase time, save/validation time, validation score, and per-step host/GPU peaks.

- Note: the script on the login node that recorded per-step timestamps stopped at 21:25 (session restart), so the per-step host/GPU peaks after that are empty. The whole-run peaks still come from the complete sampling files `hostmem.txt` and `gpumem.txt`.

Validation (AIME-2024, greedy, acc mean@1, 30 problems × 32 repeats):

| Step | Run 1 | Run 2 | PP4 |
|---|---|---|---|
| 10 | 0.633 | 0.600 | 0.633 |
| 20 | 0.633 | 0.567 | 0.700 |
| 30 | — | 0.667 | — |

### On bitwise identity

- From step 1, the scores of the two uninterrupted runs (run 1 and the first segment of run 2) already differ (−0.081 vs −0.068). That is, even with the same seed, rollout sampling is not bitwise consistent across runs, so only "close" can be compared.

## Phase 2: resume to 100 steps on a new candidate image (from 10-05 00:15)

- Image `<image>` (the r5 candidate image with the final-candidate vLLM, Megatron and verl commits):
  - sha256 253dc2d3…, recomputed and matching;
  - Image content check (one job, one node): vLLM changed files 0 diff; megatron/lite, verl_mlite, verl and recipe identical; megatron/core has only one extra compiled .so;
  - Source check: the Megatron and verl commits exported with `git archive` are identical to the directories in the source checkout; the 20 changed vLLM files are identical to the pinned vLLM commit; the changed/deleted file list matches the git diff between the base and the pinned vLLM commit (excluding tests);
  - The HEADs of the three repository checkouts are exactly these three commits.
- The recipe changed relative to the phase 1 verl commit only in: dist_opt default util 0.7 → 0.55, and comments. PP4's env still sets 0.55 explicitly, the same as the new default.
- Resume: same run directory, same env, same arguments. `resume_mode=auto` reads the checkpoints saved by the old code (the phase 1 Megatron commit): run 1 uses global_step_25, run 2 global_step_35, PP4 global_step_20.
  - The old checkpoints contain `actor/lr_scheduler.pt`, `data.pt`, `step_N/{.metadata, __r_0.distcp, optimizer_rank_r.pt, rng_state_rank_r.pt}`.
- The phase 1 jsonl files have been saved separately: `phase1-<job-id>.jsonl` (run 1), `phaseA-<job-id>.jsonl` + `phaseB-<job-id>.jsonl` (run 2), `phase1-<job-id>.jsonl` (PP4).
- Launcher `run-long3.sbatch` = run-long2 plus an in-job `ckpt-rotate.sh`:
  - checks every 60 s and keeps only the latest 2 complete checkpoints up to the tracker, without touching the directory being written; deletions are logged in `<run>/ckpt-rotate.txt`;
  - also records each step's completion time inside the job, no longer relying on the login node.
- Jobs: three jobs, one per node (run 1, run 2, PP4).
  - 12 h limit per job. Run 1 and PP4 are expected to need two segments: after the first segment ends I check it and then manually continue one more segment, rather than relying on a dependency chain to resubmit automatically.

### Phase 2 events

1. **00:33 new code loads the checkpoints saved by the old code (the phase 1 Megatron commit): PASS for all three runs**.
   - Run 1: `Loaded training checkpoint … global_step_25/actor at step 25`;
   - Run 2: `… global_step_35 … at step 35`;
   - PP4: `Loaded dist_opt checkpoint … global_step_20/actor at step 20`.
   - verl's `lr_scheduler.pt` and `data.pt` both loaded successfully, no errors.
- 00:48 first step after resume (new code): run 1 step 26, run 2 step 36, PP4 step 21, rollout_probs_diff_max 0 for all; grad norm 0.087 / 0.087 / 0.083, score 0.17 / 0.50 / 0.33, continuing the earlier trend.

### Run 1 val is 0.633 at both step 10 and step 20: a coincidence caused by the coarse granularity of the validation set

The coordinator asked about this; the conclusion is as follows.

- **The AIME-2024 parquet has duplicate rows**: the 960 rows are actually 30 distinct problems, each repeated 32 times (checked by deduplicating on prompt). This is the DAPO convention, for sampling 32 times.
- **Validation uses greedy sampling**: `val_kwargs` is `do_sample=False, temperature=0, n=1`, and vLLM has batch invariance on. So the 32 copies of the same problem generate the same result.
  - verl gives each row its own uuid, so `mean@1` averages over 960 samples, which is effectively only 30 problems.
- **All val scores are integer multiples of 32/960 = 1/30**:

| Run | Step | Score | Problems correct |
|---|---|---|---|
| Run 1 | 10 | 0.6333 | 19/30 |
| Run 1 | 20 | 0.6333 | 19/30 |
| Run 2 | 10 | 0.600 | 18/30 |
| Run 2 | 20 | 0.5667 | 17/30 |
| Run 2 | 30 | 0.6667 | 20/30 |
| PP4 | 10 | 0.6333 | 19/30 |
| PP4 | 20 | 0.700 | 21/30 |

  - Each value times 960 is exactly a multiple of 32, showing that the 32 copies are right or wrong together.
  - `score` always equals 2·acc−1 (0.2667 = 2×0.6333−1), because DAPO's score is ±1. So an identical score is not separate evidence.
- **The two validations used different weights**: run 1's `val-aux/math_dapo/reward/mean@1` is −0.124 at step 10 and −0.064 at step 20.
  - reward equals score plus the overlong penalty, and the penalty depends only on response length, so the length distributions of the two generations differ and the model really changed.
  - It just happened to get 19 problems right both times. PP4 and run 2 also vary between 17 and 21 at the same steps.
- Validation generations are not saved: the recipe does not set `trainer.validation_data_dir`. So the generations cannot be compared one by one; the above is inferred from aggregate metrics.
- Impact on users:
  - with this validation set plus greedy, the metric's resolution is only 1/30, and changes over 10 steps are mostly drowned in noise;
  - the 32 copies are repeated computation; of each ~4-minute validation, 31/32 is wasted.
  - For a more meaningful val, use `val_kwargs.do_sample=True` (e.g. temperature 1.0) so the 32 copies are 32 independent samples (avg@32); or just remove the duplicate rows.
2. **02:02 run 2 cancelled by the coordinator per the user's decision**. Reason: kill → resume was already verified (step 11), and old-code → new-code checkpoint loading was also verified; running on to 100 steps would only repeat run 1. Not resubmitted; the node has been returned.
   - Phase 2 completed steps 36–44 and was cancelled after the step 45 checkpoint was saved. The tracker is 45 and the jsonl only goes to 44, because metrics are written after saving.
   - diff max/mean 0 at every step; score 0.41–0.58; mean response length 3600–4500, truncation rate near 0; grad norm 0.067–0.087.
   - Step 40 val acc 0.700 (21/30), save 80 s.
   - Host memory peak 634 GiB (while saving), about 440 GiB normally.
   - update_actor at the first step after resume (step 36) is 345 s, then stable at 215–270 s. The extra time is compile / autotune overhead in the new process.
   - Kept global_step_45 (353 GB), deleted 35 and 40.
   - Run 2 in total: steps 1–11 (first job), 11–39 (second job, r5 image), 36–44 (third job, new image), 49 training steps in all, **diff 0 throughout**.
3. **02:04 PP4 cancelled by the coordinator per the user's decision**, stopping after step 26. Not resubmitted; the node has been returned.
   - Phase 2 completed steps 21–26, diff 0 throughout, grad norm 0.065–0.084, score 0.17–0.39.
   - Step 25 save 152 s, host memory peak 842.9 GiB, essentially the same as phase 1's 844.3 GiB. The new code did not lower PP4's memory while saving; it is still only about 5 GiB from Ray's kill line of about 848 GiB.
   - Kept global_step_25 (429 GB), deleted 20.
   - PP4 in total: steps 1–24 (first job), 21–26 (second job), diff 0 throughout.
4. Only run 1 (FSDP2) keeps running to 100 steps.

## Part 3: two more seeds (SEED=43 / 44), matching the baseline's 2 seeds (from 10-05 07:40)

- Image `<image>` (the final image):
  - sha256 ed73e69d…, recomputed and matching;
  - Image content check (one job, one node): vLLM changed files 0 diff; megatron/lite, verl_mlite and recipe identical; megatron/core has only one extra compiled .so; verl lacks only README/.md;
  - Source check: the final Megatron and verl commits exported with `git archive` are identical to the directories in the source checkout; the HEADs of the three repository checkouts are exactly the three commits in the image.
  - Relative to the Megatron/verl commits used in run 1's phase 2, it changes only one test file and the Dockerfile pin. The runtime code is the same as run 1's phase 2.
- Configuration exactly the same as run 1: `--mode aligned`, FSDP2, same env, VAL_FILES AIME-2024, `run-long3.sbatch` with checkpoint rotation. The only difference is one extra env line `SEED=43` / `SEED=44` (the recipe's own env).
  - SEED in the recipe sets data.seed, the actor and rollout seeds, and PYTHONHASHSEED at once.
- Run directories (handed to W&B sync): `$WORKDIR/runs/lr-fsdp2-s43` (one job, one node), `$WORKDIR/runs/lr-fsdp2-s44` (one job, another node).

### Run 1 finished 100 steps (10-05 10:14, exit 0, phase 2 took 9h59m)

- How the trajectory is stitched: steps 1–25 come from the r5 image (phase 1 job); steps 26–100 are the new image resuming from global_step_25 (phase 2 job). Steps 26–29 run by the phase 1 job are discarded, because the resume restarts from 25.
- **rollout_probs_diff: steps 1–100, max = mean = 0 at every step.**
- Per-5-step means (full table in `results/long-run/compare-seeds.txt`, per-step data in `run1-fsdp2-<node>-steps.txt` and `run1-fsdp2-<node>-phase2.txt`):

| Steps | score | Response length | Truncation rate | grad norm |
|---|---|---|---|---|
| 1–5 | −0.078 | 7589 | 0.319 | 0.082 |
| 11–15 | 0.219 | 6437 | 0.220 | 0.075 |
| 21–25 | 0.317 | 5912 | 0.153 | 0.076 |
| 31–35 | 0.432 | 4705 | 0.035 | 0.084 |
| 41–45 | 0.410 | 3801 | 0.001 | 0.082 |
| 51–55 | 0.448 | 3232 | 0.003 | 0.086 |
| 61–65 | 0.419 | 3090 | 0.007 | 0.095 |
| 71–75 | 0.446 | 3094 | 0.005 | 0.087 |
| 81–85 | 0.489 | 3286 | 0.006 | 0.082 |
| 91–95 | 0.554 | 3326 | 0.008 | 0.085 |
| 96–100 | 0.455 | 3498 | 0.013 | 0.089 |

- Training course:
  - In the first 35 steps the score rises from −0.08 to about 0.45. The main reason is that responses get shorter and the truncation rate drops from 0.32 to near 0, i.e. the overlong penalty disappears.
  - After that it plateaus and oscillates between 0.41 and 0.55.
  - Response length reaches its minimum of about 3100 around step 60, then slowly rises back to about 3500.
  - grad norm stays in 0.07–0.10, no divergence.
- **val (AIME-2024 greedy, in units of 1/30)**:

| Step | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 | 100 |
|---|---|---|---|---|---|---|---|---|---|---|
| acc | 0.633 | 0.633 | 0.700 | 0.667 | 0.667 | 0.733 | 0.700 | 0.633 | 0.567 | 0.633 |

  That is, it fluctuates between 17 and 22 problems, with no visible trend within 100 steps. The reason is in the previous section: this validation set does not have enough resolution.
- Step time: from about 900 s down to 360–510 s, mainly following the drop in generation length. update_actor from about 530 s down to about 200 s.
- **Host memory rises slowly**:
  - Steps without saving: about 445 GiB at step 1, about 455 at step 25, about 433 at step 26 after resume, about 460 at step 70, about 476 GiB at step 100. After resume it grows by about 0.5 GiB per step.
  - Steps with saving: 642 GiB at step 5, 667 at step 75, 681.6 GiB at step 100.
  - Still far from the 892.8 GiB cgroup limit and Ray's line of about 848 GiB; at this slope it will not OOM within 100 steps. But it is monotonically rising, cause not investigated; a longer run is needed to tell whether it is a leak.
- **GPU memory**: phase 2 peak 187,717 MiB, i.e. 183.3 GiB, out of a total of 184.3 GiB. The phase 1 peak was 182.7 GiB. Only about 1 GiB of headroom, no OOM.
- Checkpoints:
  - saved every 5 steps from step 30 to 100, 353 GB each, 72–83 s each, all successful;
  - the rotation script deleted the oldest each time, at most 2 kept throughout, 14 deleted in total.
  - After finishing only global_step_100 is kept; the step 95 one was also deleted.
