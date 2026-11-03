# B1: adding activation FP8 fake-quant (Mamba in/out_proj)

Date: 2026-10-05. A local scratch verl branch (based on the verl used by the B1 s43 image), in its own worktree, not pushed.

## Conclusion
- Single-layer alignment holds: with identical inputs, the training-side FP8 codes are **100% identical** to vLLM's; the out_proj output gap to vLLM drops from rel_l2 0.0149 to 0.0021, the latter being the bf16 noise floor (vLLM itself is 0.0017 from the exact fp32 value).
- **End to end it gets about 10% worse instead**:
  - Same-image A/B smoke (quick, seed 42): actq probs_diff 0.0108 / 0.0102 / 0.0110, weight-only 0.0096 / 0.0093 / 0.0099; pearson 0.9965 vs 0.9972.
  - Formal run steps 1 and 2: 0.0107, 0.0110; B1 s43 for the same two steps: 0.0097, 0.0104 (B2 about 0.025).
- Cause: FP8 rounding is discontinuous, so any tiny difference between the training-side and deployment-side inputs is amplified into a code flip of a whole bin (one e4m3 bin is about 12.5%). The sweep shows that **once the upstream elementwise mismatch exceeds about 1%, adding activation quantization is worse than not adding it** (see section 3). The upstream mismatch in deep layers (kernel differences in MoE / attention / FP8 KV / norm accumulating layer by layer) clearly already exceeds this threshold.
- Recommendation: under the current setup, closing this gap is actually harmful. Activation-quantization alignment only makes sense once the train/inference upstream is aligned to well below 1% first (for example the true-on-policy BI=1 path). The formal run is in progress to confirm the long-horizon trend.

## 1. Which deployment-side layers quantize activations (verified from the checkpoint and vLLM code)
- `hf_quant_config.json`: `MIXED_PRECISION`. 46 layers are `FP8`, namely `mixer.in_proj` and `mixer.out_proj` of the 23 Mamba layers. 5935 are `W4A16_NVFP4` (experts, lm_head), which do not quantize activations. The KV cache is FP8.
- `quantization_config.config_groups.group_0` in `config.json` states `input_activations: {dynamic: false, num_bits: 8, type: float}`, with exactly these 46 layers as targets.
- There are 46 `input_scale` tensors in the safetensors, only on these 46 layers: fp32, shape (1,), one scalar per layer, ranging from 0.0028 (L0 out_proj) to 1.22 (L50 out_proj). Neither the experts nor lm_head have an input_scale.
- Path in the delivery vLLM tree: `ModelOptMixedPrecisionConfig` parses `FP8` as `QuantSpec(weight=kFp8StaticTensorSym, activation=kFp8StaticTensorSym)`. During processing `input_scale` takes the `max`, so the scalar is unchanged. The kernel selected is `FlashInferFP8ScaledMMLinearKernel` (measured in the unit test). Activation quantization uses `QuantFP8(static, per-tensor)`:
  - numerically `fp32(x) * (1/s)` → clamp ±448 → RNE to e4m3fn;
  - both `forward_native` and CUDA `static_scaled_fp8_quant` compute 1/s first and then multiply; the unit test confirms the codes are 100% identical;
  - the GEMM is `bmm_fp8(scale_a=s, scale_b=w_s)`, fp32 accumulation, bf16 output.
- The actual rollout configuration is `custom_ops=['none']`, `fuse_norm_quant=False`, so norm and quant are compiled by Inductor. In the unit test the eager and `torch.compile` variants produce bitwise identical outputs.

## 2. Changes (verl, 4 files, +96/−8)
- `verl/utils/modelopt/quantize.py`
  - `mtq.quantize` additionally enables `*mixer.{in,out}_proj.input_quantizer`, reusing ModelOpt's own TE plugin:
    - `_QuantTELayerNormLinear` quantizes `ln_out` after RMSNorm and before the GEMM;
    - `_QuantTELinear` quantizes the out_proj input.
  - These quantizers are then replaced by `StaticFP8InputQuantizer`:
    - the scale is taken from `backbone.layers.N.mixer.*.input_scale` in the served checkpoint and stored as a non-persistent buffer; it is not trained and does not go into the checkpoint;
    - the forward is exactly the same as vLLM's `QuantFP8`, dequantized back to bf16 output;
    - the backward uses ModelOpt's default clipped STE (gradient zeroed where |x| > 448·s). The in_proj path goes through ModelOpt's LN-Linear Function, the gradient is pure pass-through, and wgrad uses the quantized input. This matches ModelOpt's own behavior.
  - Must match exactly 46 modules, otherwise fail closed; raises directly when PP≠1.
- `qat_utils.py` and `transformer_impl.py`: when `qat.activation_observer == "checkpoint"`, pass `model.path` (the NVFP4 checkpoint) in.
- The recipe adds `ACTOR_PRECISION=qat_actq`, which equals `qat` plus `qat.activation_observer=checkpoint`; everything else is identical to B1.

## 3. Unit test (L0, real activations: README text 4096 tokens → embedding → HF Mamba mixer)
Script `$WORKDIR/baseline-r3/checks/actq_unit_check.py` (one job, re-run on the new image in another job). The training side is TE LayerNormLinear / Linear plus this branch's `apply_mixed_precision_qat`. The vLLM side is a real `ReplicatedLinear` + `ModelOptFp8Config`, with weights from the exporter's `requantize()`, i.e. the weights vLLM actually receives after sync.

| Item | in_proj | out_proj |
|---|---|---|
| Fraction of GEMM inputs bitwise identical to vLLM | 74.4% (TE fused RMSNorm and vLLM native RMSNorm differ by 1 bf16 ulp; already present in B1) | 100% |
| FP8 code agreement (at positions with identical inputs) | 98.3% (100%) | 100% (100%) |
| Saturation fraction | 0 | 0 |
| Output rel_l2: actq vs vLLM | **0.0089** | **0.0021** |
| Output rel_l2: weight-only (current B1) vs vLLM | 0.0141 | 0.0149 |
| Output rel_l2: vLLM vs exact fp32 | 0.0017 | 0.0017 |
| Gradients reach input, weight and norm weight | Yes | Yes |

Upstream mismatch sweep (out_proj: relative noise δ added only to the training-side input, vLLM-side input unchanged; table shows rel_l2):

| δ | 0 | 0.2% | 0.5% | 1% | 2% | 4% |
|---|---|---|---|---|---|---|
| actq | 0.0021 | 0.0060 | 0.0109 | 0.0152 | 0.0211 | 0.0302 |
| weight-only | 0.0149 | 0.0149 | 0.0152 | 0.0159 | 0.0186 | 0.0267 |

The crossover is at about δ≈1%.

## 4. Image and runs
- Dev image: `<image>` (the B1 base image with the actq verl branch installed)
  - Method: the base image (the B1 verl image) only reinstalls verl and the recipe, then enroot export.
  - Build scripts `baseline-r3/make-actq-sqsh.sh` and `build-actq.sbatch`, built on one node (one job).
  - Source in `$WORKDIR/src/baseline-verl-<commit>`.
- env: `baseline-r3/formal-b1-actq-s43.env` and `smoke-b1-actq.env`, both only change `ACTOR_PRECISION` to `qat_actq`.
- Smoke (one node, quick):
  - actq: one job, exit 0, 46 modules attached, steps 2/3 normal;
  - control (same image, weight-only): one job, exit 0.
- Formal run `br3-b1-actq-s43` (one node, seed 43, same configuration as B1 s43):
  - segment 1: one job;
  - resumes: an afterany chain of continuation jobs (12 segments in total), recorded in `continuations.txt`.

## 5. Not covered
- FP8 KV cache: the deployment side quantizes with static k/v scales while the training side is still BF16. This is out of scope for this task.
- The bf16 rounding of the RMSNorm before in_proj differs between TE and vLLM, causing 1.7% code flips.

## 6. Decision (2026-10-06)

- Activation fake-quant does not go into the delivery recipe: per the sweep in section 3, the current upstream mismatch already exceeds the crossover δ≈1%, so adding it is harmful. The crossover is the result of one measurement on one model, not a threshold; for a new model, first measure the upstream mismatch and then do the same sweep.
