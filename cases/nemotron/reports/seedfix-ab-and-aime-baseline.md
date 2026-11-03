# DP sampler-seed fix A/B long run + step-0 AIME baseline (GB200 cluster, 2026-10-05)

Status: step-0 baseline complete. The A/B ran to step 30 and was then **stopped by decision** (the owner considered the comparison unnecessary; the coordinator cancelled both jobs at 08:16 UTC) and will not be resubmitted. The rollout dumps and the last two checkpoints of each run (global_step_25, 30) are kept.

## Conclusion

- Step-0 AIME-2024 greedy: **acc 0.4333 (13/30)**, score −0.133, reward −0.713.
- The seed fix does take effect: in the fixed arm, over 30 steps, **all 8 samples of every GRPO group are distinct from each other**; the control arm averages only 7.43 distinct samples, 25–107 groups (out of 128) per step have duplicates, and the same sample is repeated at most 4 times (= rollout DP 4). Step 1 of the control arm is the worst: average 5.34, 107/128 groups with duplicates.
- But at a scale of batch 128×8 and 30 steps, **the training metrics show no difference**: zero-advantage group fraction, score, response length and grad norm go back and forth between the two arms, within step-to-step noise. Entropy is slightly higher in the fixed arm (last-10-step mean 0.889 vs 0.858), but with only one run each this is not a conclusion.
- AIME val: at steps 10/20/30 the fixed arm got 18/20/23 problems, the control arm 17/19/20. Resolution is 1/30, single run, not usable as evidence.
- rollout_probs_diff_max is 0 for all 30 steps in both arms.

## 1. Step-0 AIME baseline (one job, one node)

- Image: the delivery-candidate image `<image>`; its sha256 recomputed inside the job was f9cae7eb…, matching the `.sha256` file.
- Launcher: `$WORKDIR/seedfix-ab/run-ab.sbatch` (= `long-run/run-long3.sbatch`, plus Lustre headroom protection and stop-at-step, both of which only observe or scancel their own job), env = `long-run/aligned-fsdp2.env` unchanged, `VAL_FILES` = the AIME-2024 parquet (960 rows).
- Parameters: `--hardware gb200 --mode aligned trainer.val_before_train=True trainer.val_only=True`, plus `TOTAL_TRAINING_STEPS=0 SAVE_FREQ=-1`. verl supports `val_only`: it returns right after validation, with no training and no checkpoint saved (the checkpoints directory is empty).
- Validation sampling (resolved config): `do_sample=False, temperature=0, n=1, top_p=1.0, top_k=-1`, i.e. greedy. 960 rows = 30 distinct problems × 32 copies; together with batch invariance, the 32 copies of the same problem give identical results, so the score resolution is 1/30.
- Timing: started 01:45, initialization about 18 min, validation about 4 min, exit 0 at 02:07.

| Metric | Step 0 |
|---|---|
| val-core/math_dapo/acc/mean@1 | **0.4333** (13/30) |
| val-aux/math_dapo/score/mean@1 | −0.1333 (= 2·acc − 1) |
| val-aux/math_dapo/reward/mean@1 | −0.7133 |

- reward is 0.58 lower than score because of the overlong penalty: the step-0 model's responses are very long (training rollout step 1 averages about 7700 tokens, truncation rate about 0.32).
- Compared with the phase-1 long run: 0.60–0.63 at step 10, 0.57–0.70 at step 20, 0.667 at step 30. That is, the first 10 training steps bring about +5 problems (13 → 18–19).

## 2. seed fix A/B (two jobs on two nodes: fixed arm and control arm)

### Images

- Fixed arm: `<image>` (the delivery-candidate image with the minimal seed fix overlaid), sha256 f61eb8a6… (built by one job on one node).
  - Done like a docker commit: the delivery-candidate image (sha recomputed inside the job f9cae7eb…, matching) + `$WORKDIR/seedfix/make-seedfix-sqsh.sh` overlaying the 2 vLLM files from a local branch (the delivery vLLM tree + the minimal seed fix) onto site-packages: `vllm/utils/torch_utils.py`, `vllm/v1/worker/gpu_worker.py`.
  - The overlay source `$WORKDIR/seedfix/src-min/vllm` is byte-identical to those files in the branch's worktree. These two files did not change between the candidate image's vLLM base and the delivery vLLM tree, so the old versions in the candidate image are the delivery tree's versions.
  - Content check: sha256 of every file under `/opt/venv` and `/opt/nemotron` in both images, then diff: **only these 2 files differ**, they cmp equal to the source, and the old .pyc files were deleted.
- Control arm: the delivery-candidate image itself.

### Configuration

- Both arms were submitted at the same time (01:46), with the same launcher `run-ab.sbatch`, the same `aligned-fsdp2.env`, the same `--hardware gb200 --mode aligned`, starting from step 0, each with a new run directory (`$WORKDIR/runs/ab-seedfix`, `ab-control`). The only added parameter is `trainer.rollout_data_dir=/out/rollout-data`.
- ckpt-rotate.sh keeps only 2 checkpoints, with a deletion log. Lustre free space stayed at 14.1–16.4 TB throughout (the < 2 TB protection never triggered).
- Step time about 11–13 min. Host memory peak 659 / 657 GiB.
- Statistics method: group by prompt text from the rollout dumps. "Distinct samples" only counts groups of 8 samples; the data order is the same in both arms, and a few steps have duplicate prompts in the batch, so those merged 16-sample groups are not counted. "Zero-advantage group" means all scores in the group are equal (norm_adv_by_std=False, so these groups have advantage 0).

### Summary (30 steps)

| | Fixed arm | Control arm |
|---|---|---|
| Distinct samples per group (mean) | **8.000** (8 at every step) | 7.431 |
| Groups with duplicates per step (out of 128, mean) | **0** | 48.8 (25–107) |
| Zero-advantage group fraction (mean / last 10 steps) | 0.663 / 0.643 | 0.675 / 0.664 |
| score mean (all / last 10 steps) | 0.200 / 0.350 | 0.207 / 0.376 |
| Response length mean | 6398 | 6415 |
| entropy (all / last 10 steps) | 0.663 / 0.889 | 0.641 / 0.858 |
| rollout_probs_diff_max | 0 for all 30 steps | 0 for all 30 steps |

- `critic/rewards/mean` and `critic/score/mean` are identical at every step; the table only lists score.
- In both arms entropy rises from about 0.46 to about 1.0 and response length drops from about 7700 to about 4800, consistent with the trend of the phase-1 long run (entropy 0.96 at step 26).

### AIME val (greedy, 30 problems)

| Step | Fixed arm | Control arm |
|---|---|---|
| 0 (the baseline job, delivery image; greedy is not affected by seed, shared by both arms) | 0.433 (13) | 0.433 (13) |
| 10 | 0.600 (18) | 0.567 (17) |
| 20 | 0.667 (20) | 0.633 (19) |
| 30 | 0.767 (23) | 0.667 (20) |
| 40 | stopped, not reached | stopped, not reached |

### Per-step data

`results/seedfix-ab/ab-table.txt`, columns: mean distinct samples, groups with duplicates, minimum distinct samples, zero-advantage group fraction, group size distribution | score, reward, length, truncation rate, entropy, probs diff max, val, grad norm. Generated by `summ.py`. The metrics jsonl is also copied into the same directory.

### Kept artifacts (Lustre)

- `$WORKDIR/runs/ab-seedfix/`, `ab-control/`: `rollout-data/1..30.jsonl` (about 565 MB per run), `checkpoints/nemotron_h_gb200_aligned/global_step_{25,30}` (353 GB each, about 1.4 TB for both arms), plus run logs and memory sampling files.
- Both arms were scancelled, so there is no `exit-code.raw`.
