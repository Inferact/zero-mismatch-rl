# BI=0 vs BI=1 long-run comparison (seed 42, common steps 1–100)

Same image, same recipe, differing only in `VLLM_BATCH_INVARIANT`. BI=0 is Megatron-Bridge + ModelOpt weight QAT + R3; BI=1 is mLite calling vLLM kernels directly. One seed per configuration; the conclusions only represent this one run. This table holds the pre-fix numbers; for the 100-step retrain after adding the R3 return-path and frontend logprobs fixes, see the [100-step retrain section of r3-return-path](../r3-return-path.md#100-step-retrain-2026-10-07).

| Metric | BI=1 | BI=0 QAT |
| --- | --- | --- |
| Train-inference probability diff mean (whole run) | 0.0000 | 0.0103 |
| Train-inference probability diff max (whole-run max) | 0.000 | 1.000 |
| Training score, first 10 steps | -0.030 | 0.009 |
| Training score, last 10 steps (91–100) | 0.536 | 0.668 |
| Mean response length, last 10 steps | 3364 | 3427 |
| Time per step (whole-run mean) | 530 s | 601 s |
| Total rollout throughput (whole run) | 25,931 tok/s | 22,611 tok/s |
| Per-request generation speed (whole-run mean) | 42.7 tok/s | 36.7 tok/s |
| Training update time (whole-run mean) | 267 s | 260 s |
| old-logprob time (whole-run mean) | 62 s | 106 s |

AIME 2024 validation set (`val-core/math_dapo/acc/mean@1`):

| Step | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 | 100 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BI=1 | 0.700 | 0.700 | 0.767 | 0.767 | 0.700 | 0.767 | 0.667 | 0.667 | 0.800 | 0.700 |
| BI=0 QAT | 0.644 | 0.681 | 0.768 | 0.764 | 0.756 | 0.743 | 0.766 | 0.752 | 0.791 | 0.765 |

The validation set is the 30 AIME 2024 problems each repeated 32 times (960 rows). Validation uses greedy decoding (`do_sample=False`). BI=1 is batch invariant, so the 32 outputs of the same problem are exactly identical and the scores are all integer multiples of 1/30; BI=0's greedy results vary with batch composition, so the 32 outputs are not all identical and the score is their average. The two arms' validation scores cannot be compared point by point; only the trend matters.

Rollout throughput goes in the opposite direction from a standalone `vllm serve` load test: in the standalone test (prefix cache off, no R3, 8K/1K) BI=1 is about 8–12% slower (see section 1 of [results/final/README.md](../results/final/README.md)); in training the two arms generate a similar number of tokens per step (whole-run mean response length about 4760 for BI=1 and about 4680 for BI=0), and BI=0 also has to record R3 routing; the specific cause has not been broken down yet. The old-logprob gap comes from the two arms using different training backends (BI=0 is Megatron-Bridge, BI=1 is mLite).

BI=0 bf16 (no rollout correction) collapsed at about step 41 and was stopped at step 62; whole-run train-inference probability diff mean 0.0228.
