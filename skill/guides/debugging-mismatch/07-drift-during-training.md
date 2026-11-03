# Gap grows during training: split tokens first, then choose a correction

The train-inference gap at θ0 is within budget, but after a few dozen steps it keeps growing — this is a different class of problem from θ0 already being wrong. θ0 being wrong means one side computes incorrectly; find the fork point via [01](01-isolate-the-side.md), [02](02-bisect.md). Growth during training is usually not a new bug, but **the same numerical noise amplified by a model that becomes more sensitive as training proceeds**, and the remedy is on the training side. All numbers on this page come from Model D (an MoE model; 2026-10, FSDP2 + LoRA, R3, DAPO + AIME, rollout has rounding points not simulated on the training side, such as NVFP4 KV).

## 1. What to monitor

Take a window every 10 steps and look at the window mean, not single steps (single steps jump between 0.029 and 0.044):

- KL: verl `rollout_corr/kl`, the KL between rollout and the trainer-recomputed old logprob on online samples;
- Split by token: split each response token's `|log(π_old / π_rollout)|` into the majority ≤ 1 and the tail > 1 (about 1.25–2.25% on Model D), compute each part's contribution to KL, and the tail token share;
- `actor/entropy`: look at entropy together with KL.

Compute the split offline from each step's logprob dump (`old_log_probs`, `rollout_log_probs`, `response_mask`); if the login node environment has no torch, compute it inside the training container.

## 2. Where the growth is: all in the tail

Formal training without any correction: KL rose from about 0.03 to 0.2, while entropy dropped from 0.78 to 0.12 over the same period (steps 30 → 99), with a jump at step 87.

Retraining with only geometric-mean masking (`rollout_rs: seq_mean_k1`), split:

| Window | KL | Majority-token contribution | Tail contribution | Tail share |
| --- | --- | --- | --- | --- |
| 1–10 | 0.0358 | 0.0196 | 0.0163 | 1.57% |
| 21–30 | 0.0355 | 0.0174 | 0.0181 | 1.64% |
| 31–40 | 0.0384 | 0.0172 | 0.0212 | 1.85% |
| 41–50 | 0.0449 | 0.0175 | 0.0274 | 2.25% |

The majority tokens are flat or even slightly decreasing; all the growth comes from the tail. So to judge whether a correction works, look at the tail column, not overall KL.

## 3. Comparison of three corrections

| Approach | verl config | Result |
| --- | --- | --- |
| No correction | — | Entropy collapse, KL rises to 0.2 |
| Geometric-mean masking (sequence level) | `rollout_rs: seq_mean_k1` | At ~1% rejection the tail grows as in the table above; tightening the window to ~20% rejection, the tail contribution at steps 31–40 is 0.0210, the same as 0.0212 at 1% |
| Token-level masking + geometric mean | `rollout_rs: token_k1,seq_mean_k1` | Continued from step 30 to step 70: KL 0.0377 → 0.0391, tail contribution 0.020–0.021, no clear further rise, but plateaus higher than TIS |
| Token-level TIS | `rollout_is: token`, `rollout_is_threshold: 2.0` | Trained from step 0: per-window KL 0.0356 → 0.0345 → 0.0334 → 0.0328 → 0.0328 → 0.0321 → 0.0293 (steps 1–63), tail contribution 0.0157 → 0.0118, tail share 1.53% → 1.25%, entropy from 0.89 down to 0.7–0.8, no collapse, AIME maj@8 0.355 → 0.46–0.50. Another run continuing from step 30 of uncorrected training with TIS: window KL over steps 31–186 between 0.029 and 0.038, no trend |

**Why sequence-level masking doesn't help.** It decides whether to keep a response by the mean log ratio over the whole response. Model D responses have thousands of tokens; 70–90% of responses contain at least one token with ratio < 0.1; a few extreme tokens are averaged out by thousands of normal tokens, and the whole-response mean doesn't show them. Tightening the threshold just throws away more whole responses; the distribution of extreme tokens in discarded responses is about the same as in retained ones, so the tail is unchanged.

**How token-level TIS works (inferred, not separately proven).** In decoupled mode, the PPO ratio is the new policy over the trainer-recomputed old logprob; the rollout-vs-trainer difference does not enter that ratio. Without correction, tokens sampled by rollout that the trainer considers extremely unlikely get full gradient weight, and training is pushed in the direction of rollout's numerical error; TIS weights each token by `min(π_old/π_rollout, 2)`, bringing these tokens' weights down to the order of 0.1.

Conclusion: when rollout has rounding points without a training-side counterpart (items marked "none" in the [inference-side quantization inventory](../precision-quant-qat/04-inference-quant-inventory.md)), or when the train-inference gap grows by the §1 windows, use token-level TIS; if the condition holds before training starts, enable it from step 0. For long-response tasks, sequence-level masking cannot replace it and can only be an addition. For the verl syntax see [baseline hyperparameters §4](../recipe-evaluation/03-day0-baseline-hparams.md#4-how-to-write-it-in-verl).

## 4. Handle fixed gap and growing gap separately

- **Fixed gap**: the part present at θ0 and roughly constant during training, coming from inference-side rounding points without a training-side counterpart. Measure item by item with single switches per the [inference-side quantization inventory](../precision-quant-qat/04-inference-quant-inventory.md), and decide whether to simulate, raise precision on the inference side, or keep it (for Model D's KV item, see §1 of that page).
- **Growing gap**: held down by token-level TIS. It does not shrink the fixed gap; when the goal is "doesn't grow during training", this alone is enough.
- First confirm it is noise amplification, not a new bug: on the current checkpoint measure the two noise floors and see whether the three pairwise gaps grow in the same proportion ([how to measure and interpret](01-isolate-the-side.md#the-cheapest-comparison-do-not-dump-all-layers-yet)).

## 5. Stop conditions need a trend term

Besides an absolute cap, G0's stop conditions include a trend: one window every 10 steps, compute the contribution of tokens with `|d| > 1` to KL; if two consecutive windows exceed the steps 1–10 window by the ratio set in G0 (e.g. 30%, set from that model's window-to-window fluctuation), stop and investigate — do not wait for the absolute value to cross a line. An absolute cap alone is late: in Model D's geometric-mean masking retrain, the tail contribution at steps 41–50 was already 68% higher than at steps 1–10, while overall KL of 0.045 was still far below any absolute cap.
