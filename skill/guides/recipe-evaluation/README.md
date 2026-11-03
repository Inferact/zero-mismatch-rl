# Recipe, evaluation and paired acceptance

Which training recipe items must be mathematically aligned with the reference implementation (not aligned by name), what makes evaluation the same metric, and how two arms must be paired before drawing a conclusion. Reference recipe: for the day-0 route, Miles's recipe for that model; for the train-inference consistency route, [verl#7688](https://github.com/verl-project/verl/pull/7688) (DS4 exact 0-diff true-on-policy). "What we aligned with and which values we used after aligning" is a per-model fact, written in `cases/<model>/recipe.md`, not here.

## Find by question

| Question | Where |
|---|---|
| What optimizer, clip and sample count to fill in for a new model's first GRPO recipe | [03](03-day0-baseline-hparams.md): three tiers (fixed / default / not fixed), items where verl's defaults differ |
| Which modules LoRA goes on, rank / alpha / dropout, shared or independent expert factors | [03 §1.1](03-day0-baseline-hparams.md#11-when-training-with-lora): all fixed; how to enable shared outer factors in Miles / vLLM / verl is in 03 §4 |
| Same loss name, different results | 01: check loss aggregation, advantage normalization, clip, and correction/replay terms separately |
| Whose learning rate to copy | 01: take the reference's validated value, not the default (the same entry point can differ by more than an order of magnitude); do not compare grad_norm across runs |
| Is one rollout one optimizer step | 01 |
| Reward rising too fast; how to test the grader before training | [02 grader](02-evaluation-and-acceptance.md#three-grader-checks): known-answer test, format-agnostic accuracy |
| A switch written in the yaml did not take effect | 01 §3: read the engine's actual launch args and logs; frameworks silently override |
| Train-inference gap grows during training; whether to enable TIS | [Gap grows during training](../debugging-mismatch/07-drift-during-training.md) |
| Does a 5% difference between two arms count | 02 paired acceptance: noise floor, median over multiple pairs, confounders |
| Validation score takes only a few discrete values | 02 evaluation contract: under greedy the score can only be a multiple of 1/number-of-questions |
| How to design the BI=0 baseline arm | [bi-switch-recipe](bi-switch-recipe.md) |

## Key rules

1. Align the math, not the names; read the final CLI/resolved config, not the README or model name.
2. List every deviation from the reference in a table: item, reference value, our value, why, which conclusion it affects (goes into cases/<model>/recipe.md).
3. One recipe, one image, a switch selects the arm (modeled on DS4 `EXACT_ALIGNMENT`, default BI=1; BI=0 goes through Bridge/mcore + R3 + QAT); deviations live only in the switch.
4. If any of the six evaluation items differ, it is not the same score: dataset version, prompt template, sampling params, repeat count, grader, length limit.
5. Training reward rising, probs_diff being 0, and held-out improvement are three conclusions; none substitutes for another.
6. Before running, fix what to compare, how to compute it, the pass line and the noise floor; inject a known error to confirm the criterion can return FAIL.
7. One artifact answers one question: run and report performance, quality and consistency separately.
8. List confounders before comparing; each is either aligned or quantified separately ([02 paired acceptance](02-evaluation-and-acceptance.md#paired-acceptance)).
9. Keep failure records as is and append new results; distinguish reruns from changing the question set.
10. Before trusting a reward increase, pass the grader's three checks ([02 grader](02-evaluation-and-acceptance.md#three-grader-checks)).

## Files

| File | Content |
|---|---|
| [01-recipe-alignment.md](01-recipe-alignment.md) | Alignment method: choose the reference / align four math items / read the final CLI and take validated values / deviation table / two-arm rules / minimal record card |
| [02-evaluation-and-acceptance.md](02-evaluation-and-acceptance.md) | Six-item evaluation contract, the grader's three checks, greedy 1/number-of-questions; paired acceptance: frozen criteria, confounders, pairing and computation, failure and recovery |
| [03-day0-baseline-hparams.md](03-day0-baseline-hparams.md) | Day-0 GRPO baseline hyperparameters: 12 fixed items (including R3, mandatory for MoE models) and 5 for LoRA, 7 default items, non-fixed items (day-0 route; numbers live only on this page), verl syntax, statistical method |
| [bi-switch-recipe.md](bi-switch-recipe.md) | How the two arms are defined, rules, BI=0 arm integration pitfalls, smoke-test criteria |

The list of five Miles reports, the LoRA recipe summary table, the three lr cases, specific batch/reward/eval differences and the source index are in a separate Miles recipe audit (not included here); the Model F Flash-Next section is in that model's recipe notes (not included here).

## Evidence index

- Nemotron-H 100-step two-arm comparison: [bi0-vs-bi1-100step](../../../cases/nemotron/reports/bi0-vs-bi1-100step.md); confounders and the R3 return path: [r3-return-path](../../../cases/nemotron/r3-return-path.md); deviation table vs verl#7688: [nemotron/recipe.md](../../../cases/nemotron/recipe.md).
- Miles public recipe audit (blog, LoRA summary table, lr cases): kept separately, not included here.
- Model F recipe comparison: not included here.
