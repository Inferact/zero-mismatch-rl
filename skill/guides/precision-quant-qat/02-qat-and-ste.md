# QAT, low-precision training and STE

## 1. Choose the precision route first

First rule out errors in inputs, weights, adapters, layout and operator correctness, then choose precision:

1. If GPU memory suffices and rollout can use BF16, use BF16 for both training and inference; check FP32-exception parameters one by one.
2. If rollout must use low precision (e.g. NVFP4), choose between weight QAT and a BF16 actor on the training side: on Nemotron-H the weight-QAT arm was stable; the BF16 actor, with no rollout correction enabled, collapsed around step 41 (single seed, cause is a hypothesis, see [§4.2](#42-bf16-arm-collapses-around-step-41-single-seed-cause-is-a-hypothesis)). Whether enabling `rollout_is=token` stabilizes the BF16 actor is **untested**.
3. Consider true low-precision training last: backward/kernel support and maintenance cost are higher.

The same dtype does not guarantee identical computation, and QAT does not guarantee bit-exactness. QAT and low-precision training shrink the **weight-value** gap; gaps from kernels and reduction order can only be eliminated by having the training forward call the same kernels as inference (the BI=1 arm of the [BI switch recipe](../recipe-evaluation/bi-switch-recipe.md)). A natively low-precision model is not guaranteed to run just by changing to `dtype=bf16`; do a capacity pre-check first.

## 2. What the four kinds of paths are

| Path | Training forward | backward / optimizer | Is it QAT |
| --- | --- | --- | --- |
| High-precision training + low-precision rollout | BF16 weights | High precision | No; quantized only at sync |
| fake-QAT | `DQ(Q(W))`, GEMM still BF16 | STE sends gradients back to the high-precision master | Yes |
| True low-precision GEMM (FP8/MXFP8/NVFP4) | TE changes the compute kernel | High-precision master/exception tensors remain | Different from fake-QAT |
| Frozen quantized base + high-precision LoRA | Dequantized base + adapter | Train adapter only | No |

- **Frozen quantized base + LoRA: the weight rounding point is zero; everything else as usual.** The training-side base takes the checkpoint's dequantized values, frozen and never requantized, and the adapter is not merged into the base. NVFP4 `code × block scale` has at most 6 significant bits, which BF16 represents exactly; the tensor-level scale is multiplied onto the FP32 accumulation result as the inference kernel does, not into the BF16 weight (multiplying it in adds one more rounding). Then the weights read by the GEMMs on both sides are bitwise identical (`DQ(Q(W)) = W`), and the weight rounding that QAT would simulate is zero. It only removes the weight term: activations, KV cache and other rounding points are handled item by item per the [04 Inference-side quantization inventory](04-inference-quant-inventory.md); when the inference side quantizes activations (W4A4), the training side still needs simulation. The prerequisite is that the base uses dequantized values and only the adapter is trained: when the base uses the original BF16 release, the weight term is not zero (a Miles LoRA recipe for Model G trains in BF16 while rollout uses a separately quantized FP8 checkpoint, full-model KL 2.7×10⁻³, a public PR); training all parameters requires requantization, and starting from a dequantized point also enters the [§6.1 dead zone](#61-dead-zone). Two runs of the same route (both with R3): Model N (2026-10) dropped rollout to W4A16 after enabling expert LoRA (a deviation as defined in [LoRA §1](../lora/README.md#part-1-1a-serving-lora)), activations not quantized, 150 steps `rollout_corr/k3_kl` 1.5–2.0×10⁻⁴; Model D (2026-10) kept rollout at W4A4 with NVFP4 KV cache, the training side simulated expert activations but not KV, long run with token-level TIS had KL about 3×10⁻², and at step 80 turning off only KV and indexer KV quantization reduced KL by about 40% ([04 §1](04-inference-quant-inventory.md#1-why-this-is-a-separate-step)).
- Neither fake-QAT nor true low-precision training means the optimizer/gradients are entirely low precision. W4A16 quantizes only weights; W4A4 also quantizes inputs.
- True low precision: Miles uses `--fp8-format e4m3 --fp8-recipe blockwise`, `mxfp8`, or `--fp4-format e2m1 --fp4-recipe nvfp4`. NVFP4 forward does not mean FP4 backward: there are configurations such as high-precision backward and dequantized-operand backward (backward using dequantized operands). [Low precision][low]
- Only weight-only formats can compute scales from the complete weight; formats with activation quantization also need activation scales/calibration state, which cannot be derived from weights. QAT simulation and export follow the same format rules, but that does not mean all scales stay constant during training.
- When LoRA is applied separately, do not arbitrarily fake-quant the merged base+adapter.
- Known recipes (Miles, 2026-09-14 audit): only the Kimi K2.5 INT4 launcher (routed-expert fake quant, group 32 W4A16) and the Qwen3-30B-A3B low-precision recipe explicitly enable QAT/low-precision training; DeepSeek-V4 is BF16 training + FP8 rollout; Model C and Model F (a very large MoE) are frozen quantized base + adapter-only; Model G `--fp8-rollout` only controls rollout (an FP8 KV cache does not prove the weights are FP8); a smaller Model F variant and Model B default to BF16. Sources: [DeepSeek][dsblog], [INT4][int4], [Low precision][low], and the per-model Miles docs and launch scripts at the audited commit. This is not a support whitelist; launcher defaults and framework capabilities cannot be upgraded to "enabled and validated for this model". Another survey round (verl [#6992](https://github.com/verl-project/verl/pull/6992), [#7971](https://github.com/verl-project/verl/pull/7971), NeMo-RL, miles): open-source frameworks mostly only do W4A16; activation inconsistency only appears at W4A8; details in [activation quantization survey](../../../cases/nemotron/reports/qat-activation-quant-survey.md), [NeMo-RL low-precision survey](../../../cases/nemotron/reports/nemo-rl-low-precision-survey.md).

## 3. fake-QAT implementation

Forward formulation (Nemotron-H 2026-10):

```text
w_visible = w + (dequant(requantize(w)) - w).detach()
```

- `requantize` must be the same function as the exporter; replace the library's built-in quantizer if its rules differ.
- Requantize on every forward, and again in full recompute; Nemotron-H step time about +60%.
- The new `QuantModuleRegistry` temporary replacement and the old class-attribute patch give the same probs_diff mean on the same quick workload (0.0095/0.0091/0.0100 vs 0.0096/0.0092/0.0101); for module hit checks see [P7](03-pitfalls.md#p7-qat-switch-on-target-modules-not-replaced).
- Miles: INT4 uses `OPEN_TRAINING_INT4_FAKE_QAT_FLAG`/`OPEN_TRAINING_INT4_GROUP_SIZE` ([INT4][int4]), NVFP4 uses `OPEN_TRAINING_NVFP4_FAKE_QAT_FLAG=1` ([PR #2864](https://github.com/radixark/miles/pull/2864), [adapter][nvqat], [QDQ][nvqdq]). Both do QDQ in forward and STE passes gradients back to the high-precision parameters (NVFP4's STE returns `grad_output` directly); NVFP4 keeps `main_grad` for compatibility with TE gradient accumulation, and optimizer parameters are not replaced by packed FP4. Checking env vars alone is not enough; confirm the modules hit the hook.
- Miles NVFP4's five-layer W4A16 e2e test (on Model G, under `tests/e2e/megatron/` at the audited commit) requires Blackwell and keeps shared experts in high precision; rounding, RHT, 2D and 4over6 must be aligned with the exporter/receiver. The [unit tests][nvunit] validate quantized values, STE/`main_grad` and export layer by layer, not just loss decrease.

## 4. How much mismatch QAT can remove

Nemotron-H 2026-10 measurements (`rollout_probs_diff` mean, NVFP4 mixed-precision deployment, single node 4×GB200):

| Approach | Training forward | probs_diff mean |
| --- | --- | --- |
| BF16 training + quantized rollout | BF16 weights | About 0.025; no rollout correction, collapsed around step 41 (single seed) |
| Weight QAT (STE) | Dequantized values of deployed weights, BF16 GEMM | About 0.010 (pearson 0.986→0.997); one recipe, three quick smoke runs of 3 steps each, 0.0091–0.0102 |
| Weight QAT + activation fake-quant | Same as above, plus static FP8 activations | About 0.0107, about 10% worse than weight-only |
| true on-policy (BI=1) | Deployed bytes + the same kernels as vLLM | 100 steps, every step max = mean = 0 |

The QAT arm's probs_diff max is still 0.73–1.0; a small mean does not mean no individual token has a large deviation. Magnitudes on other models (different metrics, not usable as thresholds): DeepSeek-V4 BF16 train + FP8 rollout first-step logprob drift about 0.023, still reported reward/eval increasing ([DeepSeek][dsblog]); Model C frozen quantized base + LoRA train/rollout KL noise floor about 0.002, which is a Schulman KL estimate, not mean absolute logprob diff (from a published day-0 support blog).

### 4.1 Training-side quantization simulation: per-layer aligned, end-to-end worse

Adding the same quantization simulation as vLLM on the training side aligned per-layer outputs, yet end-to-end mismatch got larger. Seen three times so far:

- **Nemotron-H static FP8 activation fake-quant.** Per-layer output aligned down to the BF16 noise floor, end-to-end probs_diff about 10% worse: FP8 rounding is discontinuous, so any elementwise mismatch in upstream inputs gets amplified by code flips. The way to judge is to add relative noise δ only to training-side inputs and sweep the output error; the crossover is around δ≈1%, and when upstream mismatch exceeds it, adding activation quantization is worse. This is one measurement on one model, not a threshold; for a new model, measure upstream mismatch first and do the same sweep. Nemotron-H decided not to merge it. All numbers (per-layer rel_l2, δ sweep table, smoke and formal runs) are in the [B1 activation fake-quant report](../../../cases/nemotron/reports/b1-activation-fakequant.md).
- **Model D NVFP4 KV simulation (2026-10).** Rollout's KV cache is NVFP4; after rounding training-side attention K/V in the same format, per-layer attention output error dropped from 15% to 2.4%, but full-model KL at step 81 (starting from the step-80 checkpoint) rose from 0.084–0.087 to 0.095–0.097; with only the expert adapter kept, from 0.030–0.032 to 0.035. Non-expert MXFP8 activation simulation was also slightly worse. The cause was not found; it may be the same mechanism as Nemotron-H.
- **DeepSeek V4.1 KV quantization simulation** brought only about 1.93% extra improvement on top of R3 (V4.1 VeOmni record; raw history archive, not included).

Rule: the weight-side range/axis/scale/rounding must match serving ([README rule 3](README.md#key-rules)); whether to simulate activations and KV depends on end-to-end results, not an unconditional requirement. Before and after adding simulation, measure end to end per [Inference-side quantization inventory §4](04-inference-quant-inventory.md#4-single-switch-measurement); if it gets worse, do not adopt it. Mainstream frameworks (verl, NeMo-RL, miles, slime) use the same **dynamic** activation rule on both sides and do not reuse the external checkpoint's static activation scales; NeMo-RL's Nemotron recipe quantizes only routed experts, leaving Mamba projections, attention and lm_head in BF16.

### 4.2 BF16 arm collapses around step 41 (single seed, cause is a hypothesis)

With BF16 actor + NVFP4 rollout and no rollout correction enabled, Nemotron-H single seed started drifting at step 33 and collapsed at steps 41–44 without recovering; the hypothesis is that BF16 updates are invisible to rollout until a weight crosses one FP4 grid step, so mismatch accumulates with training; the QAT arm was stable over the same period. To do: rerun with `rollout_is=token` and a second seed. Symptoms, exclusions and numbers are in the [BI=0 BF16 arm collapse report](../../../cases/nemotron/reports/bi0-bf16-collapse.md).

### 4.3 When QAT is worth it

Only evaluate unified FP8/QAT when evidence shows the train-inference quantization difference is the main limitation; do not switch precision just because "there was no improvement". For DSv4.1 rank-2 LoRA, the post-merge quantization delta is only 5.1–7.7% of the pre-quantization delta (a norm ratio, not an information retention rate), which only supports treating step size/quantization as controlled variables (V4.1 archive source-57; raw history archive, not included); the smaller Model F variant's long-running experiments were actually same-source BF16/BF16 and cannot count as validation of the FP8 rollout route.

## 5. STE contract

STE's surrogate function (identity or clipped), whether dX uses master values or dequantized values, how scales are handled, and how to validate are all covered in [Training side 06 Backward validation](../training-model-support/06-backward-validation.md#first-write-down-the-gradient-contract). By default dX uses dequantized values; when using master values the deviation must be quantified.

## 6. Why QAT does not train

Train-inference consistency is itself the strictest QAT: forward computes loss with quantized weights, the high-precision master accumulates updates, and STE passes gradients back. Whether it trains depends on one thing only: how far the master is from the nearest quantization decision boundary. W4A16 (FP4 codes decoded to BF16 in registers and then run on BF16 tensor cores, the benefit being 1/4 weight bandwidth), W4A4 and W8A8 all compute with "code × scale"; this is the same across all three.

### 6.1 Dead zone

- Each NVFP4 element has only 8 possible values 0/0.5/1/1.5/2/3/4/6 × scale, RNE to nearest. Starting from a dequantized checkpoint, the master is entirely on grid points, every element is half a spacing from the decision boundary, and it must travel half a spacing before the code changes.
- lr 1e-6 moves the master about 0.01% per step; the cumulative movement over a few hundred steps is far short of half a spacing, so deployed values stay byte-identical; RL gradients are more random than SFT, closer to a random walk, and need more steps.
- Result: parameters covered by NVFP4 (on Nemotron-H, routed/shared experts and lm_head, the bulk) have 0 change in deployed values, and validation NLL is clearly worse than the native BF16 baseline; freezing these weights and training again explains about half of the gap.
- Recomputing block scales does not flip codes earlier: starting from grid points, requant is idempotent; an E4M3 scale only changes codes after block amax changes by 3–6%; tiny changes in the global scale are just a uniform rescaling of the whole tensor.
- Mainstream QAT (NeMo-RL QARL, slime/Kimi INT4, Jet-RL) starts from original BF16 weights, whose values lie between grid points, so any small drift flips a proportional share of codes; there is no such dead zone. INT4 with 1×32 groups starting from a dequantized checkpoint stagnates the same way.
- FP8 is less affected: the floating-point format's relative spacing is 6–12%, small elements have small absolute spacing, and some elements are always near a boundary. DS4 also starts from dequantized FP8 and is affected the same way, just mildly.

Magnitudes (master movement percentage, extrapolated steps needed to flip codes, NLL gap, fraction of FP8 layers propagating to deployed values) are in [NVFP4 stagnation research](../../../cases/nemotron/reports/research-nvfp4-rl-stagnation.md) and [proxy SFT three-arm comparison](../../../cases/nemotron/reports/sft-three-arm.md).

### 6.2 Fix

Preferred: use the original BF16 weights (true BF16 release) as the master and requantize every step with the same rule (Nemotron-H: TE 4over6, E4M3 upper bound 256). Two points must be stated explicitly:

- **θ0 does not equal the official checkpoint bytes.** Requantizing from the true BF16 release, Nemotron-H's NVFP4 bytes are only 99.776% identical, and 24 of the 46 FP8 tensors (ModelOpt-calibrated scales) differ; "verified that the official checkpoint was produced by this rule" holds only on dequantized weights, since on grid points MSE and MAE cannot distinguish rules ([P2](03-pitfalls.md#p2-validating-rules-on-dequantized-weights-gives-false-positives)). The cost comes from the proxy SFT comparison: 300-step val NLL A-bf16 3.4685 vs C-dequant 3.4641 / C-bf16 3.4373, about 0.03 nat, already present at θ0 and not growing during training. This is a policy that requires human confirmation. Sources: [DS4 practice differences §E.1](../../../cases/nemotron/reports/ds4-practice-differences.md#e1-correction-2026-10-03-pending-human-confirmation), [skill-admission](../../../cases/nemotron/reports/skill-admission.md#items-needing-a-human), [proxy SFT three-arm comparison](../../../cases/nemotron/reports/sft-three-arm.md#addendum-same-tree-comparison-2026-10-03-1730z-one-node).
- **Three acceptance items**: ① bytes identical before and after sync with zero update (not identical to the official checkpoint); ② quick-tier RL θ0/θ1/θ2 still 0 diff and deployed bytes start changing; ③ the gap between proxy SFT and the native BF16 baseline is within budget. Nemotron-H has completed all three; results as above.

Alternative: without changing θ0 bytes, add a fixed small jitter to the master to move it off grid points (toward the neighboring grid point, no more than half a step; after requant verify bytes unchanged block by block, and roll back blocks that changed). Stochastic-rounding requantization (deployment error about 1.4×, requires writing your own quantizer, no mainstream precedent), increasing lr (under Adam equivalent to QuRL UAQ, only shortens the dead zone; 10× lr is outside the usual RL range), and frozen base + LoRA (QeRL, gives up full-parameter updates) are all not recommended. Whichever is chosen, state explicitly: starting from a dequantized checkpoint with lr 1e-6, NVFP4 deployed values do not change for roughly the first 10³–10⁴ steps. Pitfall table [P28](03-pitfalls.md#p28-master-starting-point-lies-on-the-quantization-grid).


[low]: https://github.com/radixark/miles/blob/d2fc97ce581577e255e494801d7568747d5a10d7/docs/advanced/low-precision.md
[int4]: https://github.com/radixark/miles/blob/d2fc97ce581577e255e494801d7568747d5a10d7/docs/advanced/int4-qat.md
[dsblog]: https://www.lmsys.org/blog/2026-04-25-deepseek-v4/
[nvqat]: https://github.com/radixark/miles/blob/7daba3a4f2ef94497c5af6a502bdcc3de1a2c9c5/miles/utils/nvfp4_fake_qat.py
[nvqdq]: https://github.com/radixark/miles/blob/7daba3a4f2ef94497c5af6a502bdcc3de1a2c9c5/miles/utils/fused_nvfp4_qdq.py
[nvunit]: https://github.com/radixark/miles/blob/7daba3a4f2ef94497c5af6a502bdcc3de1a2c9c5/tests/fast-gpu/test_nvfp4_quantizer.py
