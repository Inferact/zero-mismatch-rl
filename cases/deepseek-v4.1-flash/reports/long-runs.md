# Long runs

Full-model DAPO runs on the #257 line, and the trainer memory leak found on the earlier parity
branch that blocked the first long run.

## DAPO long run

| Item | Value |
|---|---|
| Megatron | #257 at `8df84b5d3` + #264, #265, #266 |
| vLLM | #62365 build with the level-2 sleep and idle-DP fixes (no small-M router, no DeepSelect) |
| Hardware | 16 nodes × 4 GB200 |
| Recipe | [recipe and memory](../recipe.md#launcher-overrides-used-by-the-validated-runs): CP2 / EP64, sparse FlashMLA on both sides, FP32 head, gradient release, chunked offload, lr 1e-6, weight decay 0.1, MegaMoE chunk 2048, `full_determinism=False` on the rollout, DSpark, sleep level 2, gmu 0.65, batch 128 × 8, response 8192 |
| Checks | `algorithm.rollout_correction.bypass_mode=False`: the trainer recomputes old logprobs every step, so `training/rollout_probs_diff_max` is defined at every step |
| Schedule | 100 steps planned; AIME 2024 before training and every 5 steps; checkpoint every 10 steps to node-local disk (the first attempt's Lustre write hung two nodes, see below) |
| Outcome | One job from scratch, no resume, ran 84 steps; it was cancelled at step 84 to free the nodes. Checkpoints for steps 30–80 were complete on all 16 nodes |

Results over steps 1–84:

- `training/rollout_probs_diff_max` = 0.0 and `training/rollout_probs_diff_mean` = 0.0 at every step.
- `actor/ppo_kl` = 0.0 at every step.
- Score mean 0.553–0.781; grad norm 0.0343–0.0644; response clip ratio ≤ 0.0049.
- Step time 366–543 s; generation 170–336 s; weight sync 105.7–132.9 s (about 106 s typical; this is
  the slow sync path that #267 replaces, see [weight sync](weight-sync.md)); checkpoint save 85–106 s.
- Rank-0 trainer memory at export 37.5 → 38.1 GiB over the first 32 steps (no per-step growth);
  actor peak allocated 133.96–136.48 GB.
- Host peak over the first 32 steps: 871.8 GB on the driver node at the step-10 save, 811–841 GB on
  the other nodes at that step.

AIME 2024 (n=8, temperature 1.0, 4096 output tokens):

| Step | mean@8 | best@8 | maj@8 |
|---|---|---|---|
| 0 | 0.733 | 0.896 | 0.805 |
| 10 | 0.736 | 0.907 | 0.813 |
| 20 | 0.739 | 0.905 | 0.810 |
| 30 | 0.750 | 0.907 | 0.814 |
| 40 | 0.754 | 0.901 | 0.816 |
| 50 | 0.755 | 0.904 | 0.816 |
| 60 | 0.762 | 0.905 | 0.818 |
| 70 | 0.760 | 0.897 | 0.814 |
| 80 | 0.763 | 0.901 | 0.815 |

The run sampled siblings independently (`full_determinism=False`) on a trainer without the sentinel
routing fix (#268). It stayed at 0 for 84 steps; whether any vision sentinel id was sampled was not
checked. On SWE-bench Pro, with 8K–24K-token responses, the same gap did show up (see
[SWE-bench Pro](swe-agentic-rl.md#vision-sentinel-ids)).

## First attempt: node failure during the first Lustre checkpoint

Same code and recipe, checkpoint every 10 steps to Lustre.

- Steps 1–9: `rollout_probs_diff_max` 0 at every step. AIME mean@8 0.736 before training, 0.740 at
  step 5.
- Step 10: two nodes stopped responding about 30 s into the first full-model save. At the last
  sample the hosts showed about 817 GB used / 210 GB available, so `free` did not indicate OOM. 7
  EFAULT retries during the write all succeeded on the first retry. A partial `global_step_10` (64
  distcp shards, 1.07 TB) was left without a tracker file.
- The cause was not resolved. The long run above wrote to node-local disk instead. See the
  [checkpoint pitfalls guide](../../../skill/guides/checkpoint-resume-memory/03-pitfalls.md).

The run watchdog was set to resubmit only when the last completed step advanced and to stop
otherwise, and it was allowed one from-scratch retry; it refused a second from-scratch submit after
this failure.

## Trainer memory leak on the earlier branch

On the closed parity branch, the first 100-step DAPO attempt died at step 15 with a CUDA OOM at the
rollout's wake.

- Symptom: trainer `memory_allocated` at each export 32.0, 35.1, 35.9, 36.6, 37.3, 38.0, 38.8 GiB:
  +0.73 GiB per step on a reproduction with 2048-token responses, about +0.9 GiB per step on the full
  recipe.
- Root cause (snapshot diff): the training forward's compressor latents and indexer
  top-k / candidates / q stayed alive. The per-forward state object formed a reference cycle through
  the activation-checkpoint function's `ctx.run_function`, so each training forward's cross-layer
  tensors survived until a full GC.
- Fix: break the cycle. Verification: 22 steps on the full model, rc=0; per-rank trainer memory flat
  on all 64 ranks (exports 2 → 22 within +0.06–0.08 GiB, a 34.42–34.49 GiB sawtooth with period 7);
  `rollout_probs_diff_max` 0 at all 22 steps; the two-path per-token probe checked 125.9M tokens with 0
  mismatch; sync payload identical across 64 ranks for v1–v23.
- #257 does not have this leak by construction (cross-layer state is cleared when the micro-batch
  forward returns); the G6 short job on #257 confirmed flat memory
  ([#257 gaps](pr257-gaps.md#g6-short-job-full-model)).

Lesson: in long runs, record trainer `memory_allocated` at every export. A slope of a fraction of a
GiB per step is invisible in short checks and shows up as an OOM at a rollout wake many steps later.
