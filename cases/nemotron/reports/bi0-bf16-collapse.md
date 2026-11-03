# BI=0 BF16 arm collapses at about step 41: symptoms and hypothesis (single seed)

Date: 2026-10-06. Run `lr-bi0-bf16-s42` (the BI=0 arm of the one recipe, `ACTOR_PRECISION=bf16`, seed 42). There is only this one seed; the cause below is an evidence-based hypothesis, not confirmed by a controlled experiment.

## Symptoms

- **Drift starts at step 33**: entropy rises; ppo_kl rises from about 2e-5 to 6e-4; rollout KL 0.016, versus 0.0037 for the QAT arm (`lr-bi0-s42`) over the same period; log_ppl_diff rises from 0.0143 to 0.0170.
- **Score drops at steps 38–40, collapse at steps 41–44**: score about −0.38; entropy first rises then collapses; truncation rate rises from 0.14 to 0.43.
- **No recovery**: afterwards grad_norm 0.4–1.48, clipfrac about 100× its pre-collapse value. Manually stopped at step 62.

## Ruled out

- Not caused by the image switch or resume at step 11: metrics are continuous across the resume; there was no restart around step 41.
- Configuration: `rollout_is`, `rollout_rs` and `bypass_mode` are all off, i.e. there is no rollout importance-sampling or rejection-sampling correction at all. DS4's BI=0 configuration has none either.

## Hypothesis

The NVFP4 rollout cannot see the update direction of BF16 training until a weight crosses an FP4 quantization bin; the train-inference mismatch accumulates during training and finally collapses once the degree of off-policy becomes large enough. The QAT arm's forward uses the quantized weights, its ppo_kl stays on the order of 1e-5, and it did not collapse over the same period.

## Proposed validation

1. Rerun the BF16 arm from step 0 with `rollout_is=token` (token-level importance-sampling correction) and see whether it still collapses around step 40.
2. Add another seed to confirm the collapse is not a fluke of seed 42.

Until both are done, "missing rollout correction caused the collapse" is only a hypothesis.
