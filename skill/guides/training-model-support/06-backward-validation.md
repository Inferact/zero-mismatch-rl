# Backward validation: is the implementation right, is the approximation reasonable, is training led astray

Forward 0 diff only shows the training and inference sides compute the same logprobs; it does not show the gradients are right. Backward must be validated separately, and it must answer three different questions, one layer of validation each:

| Layer | Question answered | Method | Does not replace |
|---|---|---|---|
| 1. Module-level VJP comparison | Does the implementation match the agreed gradient contract (missing scale, wrong transpose, broken gradient chain) | Compare each module's dX, dW and one update step against an independent reference that does not call the backward under test | Layers 2, 3 |
| 2. Gradient contract comparison | Is the contract itself reasonable: does an approximation (e.g. dX using the master value) lead training astray | On the same batch, compare gradients from the two contracts per class and per layer | Layer 1 |
| 3. Proxy SFT three-arm comparison | Is there systematic bias in the whole training pipeline: first compare gradients per module on the same batch, then compare training curves | Three SFT paths — under test, gradient reference, native baseline — compared with the same data and hyperparameters | Layers 1, 2 |

The order is 1 → 2 → 3: if layer 1 fails, fix the implementation first; do not launch SFT hoping for luck; when layer 3 is anomalous, go back to the first two layers to localize. Only when all three layers pass can you say backward has no problem; none of them proves long-term RL stability.

## First write down the gradient contract

### The three tiers of weights

In quantized training the same parameter has three copies, and the gradient must ultimately return to the one the optimizer manages:

| Tier | Precision | What it is |
|---|---|---|
| main | FP32 | Optimizer state; updates accumulate here |
| master | BF16 | The model parameter itself, converted from main each step; unquantized layers compute with it directly |
| Deployed value | NVFP4 / FP8 etc. | Quantized from master by the deployment rule; both training forward and inference use these bytes |

Keep the optimizer's parameter identity: when replacing the forward, do not create new parameters the optimizer does not know about; the deployed value is not a trainable parameter.

### Contract table

An inference kernel having no autograd does not mean its math needs a surrogate gradient. Non-quantized GEMM, norm and attention reuse existing backward directly when the math semantics agree; STE is only needed when the gradient has to pass through discrete operations such as quantization rounding. Before implementing, write down for each module class:

| Item | What needs to be determined |
|---|---|
| Visible forward | Inference implementation; input/output dtype, layout; quantize/dequantize position, scale granularity, rounding, clamp, accumulation precision |
| Trainable variables | input, master weight, bias, norm, router; whether scale is fixed, stop-gradient, or trainable |
| Surrogate function | Whether dX uses the master value or the dequantized value (see below); identity STE or clipped, value at the threshold |
| Discrete state | Fixed expert IDs, sparse indices, sorting; continuous score/combine weights that still need gradients |
| backward | Reused implementation or custom VJP; cast, accumulation/reduction precision, communication and normalization |
| Save/recompute | Quantized values, scale, norm/attention intermediates, routing, RNG; memory cost and recompute rules |
| Reference and acceptance | Independent reference implementation, target shape/execution path, metrics and pre-agreed tolerance |

Identity STE defines the surrogate derivative of quantization Q(x) as 1; clipped STE defines 1 inside the range and 0 outside, and the value at the threshold must be stated and must not be silently switched in the implementation. Declare weight quantization and activation quantization separately; do not infer from "frozen weights" that the activation upstream needs no gradient.

### Linear: which weight dX uses

For `Y = X · Wᵀ + b` with upstream gradient `G`:

```text
dW = Gᵀ · X_ref        # back to master via STE, then the optimizer updates main
db = sum_over_tokens(G)
dX = G · W_ref
```

`W_ref` can be taken two ways:

- **Dequantized value**: `W_ref = dequant(quant(W_master))`, i.e. the value the forward actually used. This is the true derivative of `Y = X · Wqᵀ` w.r.t. X, and it is what fake-quant QAT naturally gets when left to autograd. **Use this one by default.**
- **Master value**: `W_ref = W_master` (BF16). The benefit is no need to dequantize the deployed value; the ordinary BF16 GEMM backward can be reused directly. When the inference kernel only takes packed quantized bytes and has no ready-made backward, this is the easiest wiring. The cost is that dX is no longer the exact derivative of the forward function, ignoring the effect of quantization error on the input gradient.

The two choices give different gradients and cannot be silently swapped. When using the master value, layer 2 comparison is mandatory to quantify and record the bias. Activation quantization is the same: whether `X_ref` uses the quantized activation or the original activation must also be stated.

Before reusing a linear or grouped GEMM backward, check flatten/transpose, padding and valid tokens, bias reduction, saved inputs, scaling, and accumulation/return dtype; identical output shape does not prove it can be reused.

## Layer 1: module-level VJP comparison

### How to build the reference implementation

The reference can use PyTorch primitives and autograd, but **must not call the backward under test**, otherwise both paths replicate the same error. When discrete operations are involved, the agreed surrogate backward must be implemented explicitly; do not differentiate bare rounding and treat the resulting zero gradient as the reference.

The bridge below makes the output value come from the inference implementation and the gradient from an independent surrogate; it is used to build the reference path:

```python
import torch


class VisibleWithSurrogate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, visible, surrogate):
        assert visible.shape == surrogate.shape
        assert visible.device == surrogate.device
        assert visible.dtype == surrogate.dtype
        return visible

    @staticmethod
    def backward(ctx, grad_output):
        return None, grad_output


with torch.no_grad():
    visible = inference_linear(x, deployment_weight, scales)
w_ref = dequantize(deployment_weight, scales)  # default contract; switch to master_weight under the master-value contract
surrogate = torch.nn.functional.linear(x, w_ref, bias)
y = VisibleWithSurrogate.apply(visible.detach(), surrogate)
```

- The surrogate graph must be built in normal grad mode, outside the custom Function's forward.
- When using the dequantized value as the surrogate, `w_ref`'s gradient must flow back to master via STE: first bridge out a tensor "whose value is the dequantized value and whose gradient passes straight through to master", then feed it into an ordinary linear.
- Do not substitute `visible + surrogate - surrogate.detach()`: floating-point add/subtract and compiler transforms change the result. The bridge output must not be modified in place downstream.
- This bridge is an independent reference for validation, not the production wiring; for production wiring see [03](03-reuse-backward.md#wiring-compute-the-forward-once-the-backward-uses-what-the-forward-saved).
- This bridge and `visible_forward(visible, native, *inputs)` are two ways of writing the same thing: `visible` only saves inputs and builds no graph, `native` is the same-math implementation in the training framework, and at backward time the forward is rerun on the same inputs to differentiate; the resulting precision and speed depend on which `native` is chosen. In mLite it is model-agnostic and belongs in `primitive/ops/`, not the model directory ([01 §5](01-module-inventory.md#5-example-nemotron-h-reuse-audit-against-mlite)).

### Two kinds of baselines, not sharing one pass standard

| Question | Baseline | Verdict |
|---|---|---|
| Does the new backward implement the agreed gradient | A validated backward with the same semantics, or a reference with the same quantization/STE rule | Per the pre-agreed precision standard |
| Does the new numerical scheme affect training | The HF/Megatron Core native training path | Module gradient bias and proxy SFT within budget (layers 2, 3) |

The names HF/Megatron Core are not proof of correctness; record the actual version and config. For the first diverging module, prefer VJP comparison with inputs/weights that actually trigger it; random shapes are not guaranteed to hit quantization midpoints. Whole-model gradients being finite and non-zero is only a smoke test.

### Comparison steps

The general steps are below; the Nemotron instance (tolerances, covered shapes and length tiers) is in the [case](../../../cases/nemotron/consistency-case.md#backward-independent-validation-protocol-and-per-op-status).

1. **Fix the state.** Fix inputs, weights, quantization config, mask/routing and other state; the reference and candidate each use independent copies, with accumulated gradients cleared. When bitwise gradients are required, unify determinism settings first and do A/A on the reference itself (for the Nemotron-H attention 835-token comparison, before unifying, random dQ had 52 unequal BF16 elements; after unifying, 0). First check the visible forward matches the inference output, then check the saved state the backward needs; at module integration also confirm that state is actually produced by the real forward.
2. **The same upstream gradient.** Both sides use exactly the same external dY, with the same dtype and representable values. Cover fixed random, zero, and sparse gradients non-zero only at selected positions, then add samples captured from real training. Do not use only `y.sum().backward()`; it misses transpose, permutation and broadcast errors.
3. **Compare item by item.** Compare dX, dW and every bias/norm/router/scale that requires grad: missing, shape, finiteness, whether those that should be non-zero are non-zero. By default do not ignore unused differentiable inputs; list separately the parameters that truly do not participate in the current branch. Report max absolute error, relative L2 `‖g − g_ref‖₂ / max(‖g_ref‖₂, eps)`, norm ratio and cosine — do not look only at direction; when the norm is zero, cosine is undefined, use absolute error instead.
4. **Fix tolerances in advance.** Set atol/rtol and relative L2 thresholds by dtype, reduction length and reference precision, and record the rationale; do not use one threshold for all modules, and do not relax after a failure. When only optimizing a same-semantics kernel, do not use "STE is an approximation anyway" to relax implementation error; cross-framework bitwise identical backward is not required. First do the injected-error discrimination check, and accept multiple training snapshots separately ([04 training-side correctness acceptance](04-layout-and-length.md#training-side-correctness-acceptance-gradient-criteria-v2-and-optauth)).
5. **Coverage.** Decode small M, training/prefill large M, different context lengths, unaligned tails, padding/packing/mask, zeros and extremes; both sides of quantization thresholds, the saturation region, different scales. For MoE cover empty experts, duplicate routing, shared expert, top-k ties; for shared parameters check gradient accumulation across multiple use sites. Test under the target dtype/layout, activation checkpointing/recompute, autocast, CUDA Graph, gradient accumulation and the actual parallel reduction, local first then integrated.
6. **One update step.** From the same parameters and optimizer state, each side performs one update; compare the change in master. If gradients are close but updates differ, check scaling, clipping, optimizer state and parameter binding.

Smooth non-quantized modules can use small-size, high-precision [gradcheck](https://docs.pytorch.org/docs/stable/generated/torch.autograd.gradcheck.html) or finite differences as an auxiliary check: fix discrete branches, avoid non-smooth boundaries, and compute the predicted loss change by perturbing the weights the forward actually sees. **Do not accept STE with finite differences of the real quantized forward**: that measures the discrete forward, and STE's surrogate derivative is by design different from it.

**Tolerances themselves can be wrong.** Of the four "failures" in Nemotron's independent gradient validation (report [gate4-vjp](../../../cases/nemotron/reports/gate4-vjp.md)): F1 was the contract saying FP32 while the compact kernel was actually TF32 (weight operand truncated by 13 bits); F2 was a multi-term sum exceeding the preset 1e-3, but a reference modeled on the real addition order was bitwise identical; F3 was PyTorch CUDA `embedding_dense_backward` accumulating at reduced precision with ≤3072 indices (6.1e-3); F4 was TE FusedAdam flushing subnormal gradients to zero. None were implementation errors in the production backward, but all must be written into the contract or criterion, not erased after the fact.

### What to check per module

| Module | Prefer to reuse | Semantics to check |
|---|---|---|
| Dense/QKV/LM head | Training linear/grouped VJP, gradient back to master | dX value convention, cast, bias, gradient accumulation of shared parameters |
| Norm/RoPE | Existing training VJP, FP32 statistics if needed | epsilon, RMS vs LayerNorm, position offset, rotation layout |
| Activation + quantization | The actual combination of activation derivative and STE | Saturation region, whether scale is detached, fusion order |
| Attention/sparse attention | Training backward with the same mask/index semantics | Causal boundary, softmax scale, LSE definition, GQA, selected KV order; auxiliary quantities from different forwards must not be mixed |
| MoE | Expert linear/grouped VJP and differentiable combine | Inverse dispatch permutation, gradient accumulation for duplicate tokens, empty experts, shared expert, top-k weight normalization |
| Router/sparse selection | Fixed discrete selection, keep derivatives of continuous scores | No continuous gradient w.r.t. IDs/indices; do not detach the entire router output |
| Mamba2 conv/scan | causal_conv1d bwd, mamba_ssm `_mamba_chunk_scan_combined_bwd` | Sequence boundaries, chunk grid, dW accumulation order; atomics in SSD backward need deterministic mode |
| Selected logprob | log-softmax/gather VJP matching the definition | Vocab-shard reduction, temperature, token mask, loss normalization |

Recompute must save or deterministically rebuild scale, routing/indices, RNG and necessary intermediates, and must not switch the surrogate function during recompute; after an optimizer update, refresh deployed weights, scale, packs and Graph references by version. Whether FP8 attention backward uses the forward-visible O or a BF16-recomputed O is a contract choice for a human to decide; see [Nemotron step 7](../../../cases/nemotron/consistency-case.md#step-7-how-to-make-the-gradient-comparison-discriminating) and [the D term in 04](04-layout-and-length.md#the-d-term-in-attention-backward).

## Layer 2: gradient contract comparison

Layer 1 only answers "does the implementation equal the contract", not "is the contract reasonable". When the contract contains an approximation — most commonly dX using the master value rather than the dequantized value — quantify how far it is from the exact derivative:

1. Fix a checkpoint and the same batch; for each quantized layer compute dX (and gradients related to activation quantization) under each of the two contracts.
2. Report cosine and norm ratio per module class and per layer, and look at outlier layers.
3. Measure separately under the target precision combinations: changes in weight precision and activation precision both change the bias.

On Nemotron-H (NVFP4 routed experts, FP8 Mamba projection), the two dX choices (master value vs dequantized value) give per-class per-layer cosine about 0.99 and norm ratio about 1.00, the same under both BF16-activation and FP8-activation configs, so this approximation is not the cause of the training score gap. This conclusion only holds for the measured precision combinations; for lower precision, larger learning rate or longer training, re-measure.

## Layer 3: Proxy SFT three-arm comparison

After the first two layers pass, use controlled SFT to examine the whole training pipeline:

| Path | Visible forward | backward | Purpose |
|---|---|---|---|
| Under test | Aligned inference numerics | Production backward | The scheme to be delivered |
| Gradient reference | Same as under test at the same parameter snapshot | Independent surrogate reference (the bridge above can be used) | Isolates backward implementation errors |
| Native baseline | Original training numeric path | Validated original training backward | Measures the overall recipe quality change |

In two steps: first compare gradients per module on the same batch, then run curves.

### 3a. Per-module gradient comparison on the same batch

At a fixed checkpoint and the same SFT batch, each side does one forward + backward (no optimizer step), dumps every parameter's gradient, and compares module by module. It sits between layer 1 and curves: integration issues that layer 1's per-module unit tests cannot catch — wrong parameter binding, gradients not flowing through a custom Function, loss normalization, gradient accumulation, parallel reduction done once too few or once too many — show up here as the gradients of some modules deviating as a whole.

**Two comparisons, interpreted separately:**

| Comparison | The two sides | Relationship of the two forwards | Expectation |
|---|---|---|---|
| Same-value comparison | Under test vs gradient reference | The same visible values (the reference reproduces the under-test forward output via the bridge) | Only backward implementation differences remain: each module within the pre-agreed numeric tolerance |
| Native comparison | Under test vs native baseline | The native baseline computes along its own precision path | Differences include quantization and forward path: small and uniform overall, no outlier modules |

With only the native comparison, forward numeric differences mix into gradient differences, and you cannot tell whether backward is wrong or it is the effect of quantization itself; so first confirm the backward implementation with the same-value comparison, then use the native comparison to see the overall bias from quantization. If the native baseline can run with "dequantized deployed values as BF16 weights", its forward is closest to the one under test and can serve as an intermediate point between the two.

**How to do it:**

1. **Same starting point.** Both sides load the same master; if the native baseline uses dequantized values as weights, state so. Same batch, same labels/mask, packing and loss normalization (per valid token or per sequence), same number of gradient accumulation steps.
2. **Fix discrete selections.** MoE routing and sparse indices must be identical on both sides, otherwise experts receive different tokens and expert gradients are inherently incomparable. Have the native baseline replay the routing recorded by the side under test (R2/R3-style routing replay), or have both sides replay the same routing.
3. **Dump location and which gradient to take.** After all gradient reductions complete and before clipping and the optimizer, save `{name: grad.float().cpu()}` with an in-process hook. Each framework takes it at a different place:
   - Megatron's gradient is in `param.main_grad` (FP32), not `.grad`; read it after `finalize_model_grads`. Under the distributed optimizer gradients are reduce-scattered and each rank only has its own segment reduced, so dump with DP=1, or temporarily disable the distributed optimizer and switch to all-reduce.
   - FSDP2's `param.grad` is a DTensor; use `full_tensor()` to assemble the full tensor.
   - Assemble TP shards by parallel style: column-parallel along the output dim, row-parallel along the input dim; for EP order each rank's experts by global expert ID; compare tied embeddings only once.
   - Expert gradients may be multiplied by expert-DP/DP scaling only inside the optimizer step; the dump location must be consistent with that.
4. **Align names and scaling.** Map parameter names on both sides to one common set (e.g. HF checkpoint names); split fused parameters (qkv, gate_up, stacked experts) by the same rule before comparing. Parameters that cannot be matched must be listed, not silently skipped. Also check loss scaling: Megatron averages over micro-batch count and DP, while HF/FSDP implementations often average over valid tokens; the two may differ by a constant. If the norm ratio is the same constant across all parameters, check here first.
5. **Per-parameter metrics.** Compute in FP32/FP64; for each parameter report cosine, norm ratio, relative L2 and finiteness; list zero-norm parameters separately. Aggregate by "module class × layer" (replace layer and expert numbers in names with `*` to form the class), and look at per-layer curves:

   ```python
   for name in sorted(ref):
       g, r = cand[name].double().flatten(), ref[name].double().flatten()
       rn = r.norm()
       cos = (g @ r / (g.norm() * rn)).item()
       ratio = (g.norm() / rn).item()
       rel_l2 = ((g - r).norm() / rn).item()
   ```

   Set thresholds before looking at results. First measure the noise floor with A/A (the same side run twice), then corrupt one parameter's gradient and confirm it gets flagged.
6. **Interpretation.** Look at the whole first: all classes within tolerance means integration is correct. Then look for outliers: if one module class or a few layers clearly deviate (cosine drops, norm ratio away from 1, relative L2 jumps), first suspect backward, scale, transpose, routing or reduction at those locations; uniform small deviations are usually numeric precision. Take outliers back to layer 1 and do a VJP comparison on that module alone to localize.
7. **One update step.** Once gradients match, both sides take one update from the same optimizer state and compare the change in master; if gradients are close but updates differ, check clipping, scaling, optimizer state and parameter binding.

### 3b. Training curve comparison

- Run curves only after 3a passes; once trajectories diverge, you cannot assume later visible values are still identical. Differences between under-test and native baseline include quantization and forward-path differences and cannot all be attributed to backward. When budget does not allow a full curve for the gradient reference, keep comparisons on representative batches and state the missing range.
- Controlled variables: initial weights and optimizer state, data and order, labels/mask, packing, loss normalization, global batch and gradient accumulation, learning rate and schedule, clipping, precision and loss scaling, seed, recompute strategy.
- Record: train loss, fixed validation-set NLL, gradient norm, parameter update RMS, clipping ratio, non-finite values, throughput; use optimizer step or cumulative valid tokens as the x-axis consistently. Agree on the comparison window and acceptable deviation before running; short runs only prove a short-range smoke.
- Localization order: if loss differs already at identical initial weights, check forward, data, mask; if initial loss matches but one update differs, check backward, reduction, scale, clipping, optimizer; if divergence appears only after a few steps, save the first anomalous checkpoint, check recompute, quantization refresh, learning rate, RNG, then do a same-weight replay.
- Differences that cannot be explained or that lack budget are marked inconclusive; do not cover them with forward 0 diff.

On Nemotron-H, three paths (the aligned trainer with BF16 activations, Megatron QAT, Megatron BF16) ran 40 steps on gsm8k with the same data and order: loss 0.3388 / 0.3383 / 0.3374, gradient norm 0.685 / 0.674 / 0.728, curves overlapping; this shows the training loop itself is fine and the score gap in RL comes from RL-specific factors ([sft-three-arm](../../../cases/nemotron/reports/sft-three-arm.md), [mcore-grad-optimizer-compare](../../../cases/nemotron/reports/mcore-grad-optimizer-compare.md)).

After all three layers pass, accept at the native training entry point per [two-step RL and final sync probe](../../validation.md#two-step-rl-and-the-final-sync-probe). SFT supports judging training effectiveness, RL 0 diff supports forward and sync correctness; neither independently proves long-term stability.

## Common false differences

These have all made gradients look mismatched, but were actually problems with the comparison method:

- **Different upstream gradient precision.** FP32 dY on one side and already-BF16-rounded dY on the other manufactures differences unrelated to backward. Both sides must use the same external dY.
- **Dump timing.** Expert gradients are multiplied by expert-DP and DP scaling only inside the optimizer step; dumping too early shows a fake multiplicative difference.
- **FSDP2's unsharded copies.** After forward-only, the root module does not reshard, so parameters fetched by name are the unsharded compute copies, not the real master; perturbing them or reading their gradients leads to wrong conclusions.
- **"Frozen" is not really frozen.** If gradients are only zeroed, AdamW's weight decay still changes the parameters; truly freezing requires grad to be None or removing the parameter from the optimizer.
- **Test environment differs from production.** Unit tests missing the production config context, or importing in an environment without GPU visibility, pick the wrong implementation or fail to import (see [03 reusing backward](03-reuse-backward.md#two-test-infrastructure-false-negatives)). These are test errors and cannot be attributed to train-inference differences.

## What to write in the report

Per module class: gradient contract (including dX value choice, STE type, scale handling), reference implementation, upstream gradient construction, covered shapes/paths, metrics and tolerances with rationale; layer 2's per-class per-layer cosine/norm ratio and outlier layers; layer 3's config difference table for the three paths, curves, and conclusion over the comparison window. List missing coverage separately; do not wave it off with "the rest is analogous".

For the basic definition of STE see [the paper by Bengio et al.](https://arxiv.org/abs/1308.3432); it does not determine the surrogate gradient rule for a specific model.
