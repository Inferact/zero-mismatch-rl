# L3 parameter gradients and L4 one optimizer step: mLite actor vs MCore reference (2026-10-03)

The object under test is the actor's currently delivered code, at an internal Megatron commit. This round serves as the "before" baseline ahead of replacing the Mamba SSD and attention backward. For setting up the environment and the reference model, see the MCore reference setup notes (not included).

## Conclusions
- **L3 parameter gradients: all 5 categories sit at the noise floor; no implementation defect in the actor was located.**
  - Of the 7 outliers seen earlier against HF, 5 are back to ≤1.3x against MCore; they are specific to the HF reference, not an actor problem.
  - The layer-41 norm is still 2.1x. Block-level VJP (same input, same dY) and whole-model per-layer h/g are both at the floor, so it is judged a statistical outlier caused by a single-sample floor that happened to be small.
- **L4 one optimizer step: all three checks pass (see below for the FP8 one).**
  - (i) The FP32 main update differs from FP64 AdamW by ≤3.04 FP32 ulp;
  - (ii) BF16 parameters = RNE(FP32 main), 0 mismatches;
  - (iii) NVFP4 deployed bytes are bitwise equal to TE 4over6. FP8 is not bitwise equal to TE current-scaling, but is bitwise equal to the vLLM/ModelOpt rule; this is a convention difference, not a defect.
- **Finding 1 (real defect): parameters with wd_mult=0 also get weight decay.**
  - verl_mlite's LR scheduler ignores `wd_mult`, so 168 1-D parameters (all norms, A_log, D, dt_bias, conv1d.bias) get wd 0.1 applied.
- **Finding 2: actor gradients are not bitwise reproducible.**
  - Two identical runs: 304 of 378 gradient tensors differ.
  - The source is dQ of the layer-42 attention (k/v/o are all bitwise identical), which then propagates to shallower layers.
- **The 1-ulp deviation of lm_head is not a meaningful defect.**
  - The cause is two-step rounding in FP32, plus the exact value happening to land near a BF16 midpoint. The excess error is ≤6e-8 (relative).
  - This phenomenon is not limited to lm_head; my earlier `check_theta0` itself has the same problem.

## Method
- **Data**: dapo_0 and dapo_1 packed, 2394 tokens, of which 2048 are response. Labels are exactly the rollout tokens. The loss is the token-mean NLL over response tokens, consistent with the actor's `_token_mean_loss`.
- **Three runs**:
  - (a) actor: the real path of the verl recipe, i.e. impl=vllm, PP4, dist_opt, precision-aware, CPU offload AdamW. Take the FP32 `main_grad`.
  - (b) MCore BF16: STE fake-quant added on the θ0 exported weights, aligned with the actor:
    - Mamba in/out_proj: x·fp32(1/s) quantized to e4m3, using TE high_precision's backward;
    - the fused in_proj+norm forward is rewritten as RMSNorm (FP32 math) plus linear, math unchanged;
    - attention: Q scale is 1, K/V use x/bf16(s), identity STE.
  - (c) MCore FP32 as ground truth. TF32 off (cuBLAS, torch and mamba_ssm's Triton dot all off), unfused attention.
- **Metric**: for each tensor compute rel L2 and cosine of actor vs (c), take (b) vs (c) as the floor, and report the ratio of the two.
- **Weight mapping**: MCore uses the same batch on 4 ranks, and routed-expert gradients are divided by EP. The mapping was checked with the weights (QKV split by group, global expert numbering); all 1804 tensors are bitwise equal.
- **loss**: actor 4.9612, MCore BF16 5.0236, MCore FP32 4.9931. Mean per-token logprob difference from FP32: actor 0.171, floor 0.181.

## L3 results (actor error ÷ floor)

| Category | Tensors | Median | p95 | Max | actor rel median / floor rel median | Norm ratio actor / floor (median) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 Quantized/attention linear layers (Mamba in/out FP8, shared experts, q/k/v/o) | 116 | 1.07 | 1.41 | 1.60 | 0.059 / 0.056 | 0.988 / 0.995 |
| 2 routed experts (experts concatenated per layer) | 23 layers | 1.07 | — | 1.53 | 0.083 / 0.082 | 0.990 / 0.988 |
| 3 router gate | 23 | 1.00 | 1.12 | 1.61 | 0.150 / 0.197 | 0.987 / 0.996 |
| 4 Small parameters (norm, conv1d, dt_bias, A_log, D) | 191 | 1.08 | 2.05 | 6.82 | 0.061 / 0.056 | 0.990 / 0.999 |
| 5 embedding / lm_head | 2 | 1.36 | — | 1.36 | 0.045 / 0.043 | 0.993 / 0.999 |

- **Routed experts should not be compared per expert.**
  - Per-expert rel median is about 0.26, ratio median 1.08.
  - 242 experts get no tokens in the ground truth, and 26 of them do get tokens in the actor, because top-k flips near ties. So per-expert ratios are meaningless; instead all experts of a layer are concatenated and compared.
- **Gradient norm**: actor total norm 227.80, FP32 231.2, BF16 230.6. Per-layer norm ratios are all about 0.98–0.99, consistent with the slightly lower loss; direction cos is within the floor.

**The 7 earlier outliers against HF, now as ratios against MCore:**

| Parameter | Ratio against MCore | Attribution |
| --- | --- | --- |
| L0 norm | 1.03 | HF-specific |
| L0 in_proj | 1.04 | HF-specific |
| L0 conv1d.weight | 1.30 | HF-specific |
| L0 conv1d.bias | 1.26 | HF-specific |
| L40 shared down | 0.95 | HF-specific |
| L48 dt_bias | 0.86 | HF-specific |
| L41 norm | 2.11 | Persists |

Besides L41 norm, `41.mixer.norm` in the same block is 6.8x, and `4.mixer.norm` is 3.5x.

**Localization:**
1. **Block-level VJP** (actor and MCore blocks given the same real input and the same dY; `block_compare.json`). For the Mamba blocks of layers 41/4/50/2/39/0 and the layer-42 attention block, the ratios of output, dX and all parameter gradients are within 0.53–1.58. Layer 41's norm is 0.91 and mixer.norm is 1.11. The in-block implementation is fine.
2. **Whole model, per layer** (`layer_dx_compare.json`). Each layer's residual stream h_i and its gradient g_i, actor error ÷ floor: h within 0.94–1.06, g within 0.94–1.15. No layer deviates.
3. **Noise of the reference itself** (`fp32_self_noise.json`):
   - Two MCore FP32 runs are not bitwise identical, loss 4.9931 vs 4.9694; suspected to be mamba_ssm Triton autotune picking a different config each time, with the FP8 quantization boundaries amplifying the difference;
   - FP32 self-noise is about 0.34–0.49x of the floor;
   - The single-sample floor of `41.mixer.norm` is only 0.65%, while FP32 self-noise is 1.5%.

Conclusion: the gradients of these small parameters are sums over 2394 tokens with heavy cancellation, the most sensitive to upstream perturbations, and single-sample ratios have heavy tails. There is no evidence of an actor defect. To judge this category accurately in the future, either sample the floor multiple times or switch to an error normalized by the condition number of the sum.

## Gradients are not bitwise reproducible (actor)
- Two identical runs, comparing FP32 `main_grad` (`grad_repro.json`):
  - Layers 43–51, lm_head and norm_f are bitwise identical;
  - Layer 42 (attention): only q_proj differs, k/v/o are identical;
  - Everything from layer 42 toward shallower layers differs, rel typically 1e-3–1e-2, a few dt_bias up to 3e-2, embedding 4e-3.
- Total gradient norm over three runs is 227.7798 / 227.7827 / 227.7968, while the forward loss is bitwise identical all three times.
- Source: the surrogate VJP of fp8_attention uses the SDPA backward, whose dQ accumulation is nondeterministic.
- The magnitude is far below the BF16 floor (about 5%), but the recipe says deterministic. The line of work replacing the attention backward must make sure the new kernel's dQ is deterministic too.

## L4 one optimizer step
**Config**: recipe defaults. lr 1e-6, constant schedule, betas (0.9, 0.95), eps 1e-8, wd 0.1, clip 1.0, decoupled, precision-aware, `offload_fraction=1.0`.
- The scheduler uses the verl_mlite engine's own `_build_lr_scheduler`, built the same way as in the recipe.
- lr was not increased: the recipe lr already changes 1–10% of BF16 elements, so requant is genuinely exercised.

**Results** (`l4.json`):
- grad norm: runtime 227.796753, FP64 227.796843, relative difference 4e-7; clip is active, coefficient 0.00439.
- **(i) FP32 main vs FP64 AdamW**:
  - The reference uses the group's actual hyperparameters and the runtime clip coefficient. Error in FP32 ulp of max(|x0|, |Δ|): max 3.04 ulp over all parameters, ≤1.5% of elements above 1 ulp.
  - exp_avg ≤2 ulp, exp_avg_sq ≤3 ulp.
  - With a clip computed in FP64 instead, the max is 8.8 ulp. The difference comes from rounding in the FP32 global norm reduction.
  - Elements whose first or second moment is in the FP32 subnormal range are excluded, mainly zero-gradient elements and the in_proj dead rows already recorded by gate4 F4 (FTZ, update about 1e-36).
- **(ii) BF16 parameters = RNE(FP32 main)**: 378/378 tensors, 0 mismatches.
- **(iii) Deployed bytes after requant**:
  - NVFP4: 47 linears (shared experts and lm_head) and 23×256 routed-expert projections are bitwise equal to TE `NVFP4Quantizer(4over6, e4m3_max=256)` applied to the new BF16 parameters.
  - FP8, 46 tensors (Mamba in/out_proj):
    - The deployed scale equals the correctly rounded amax/448 (46/46);
    - The codes equal e4m3(x·fl32(1/s)), bitwise equal (46/46);
    - TE `Float8CurrentScalingQuantizer` computes 448/amax first and then takes the reciprocal; scale differs in 10/46, codes differ in 3/46 tensors, 1220 bytes in total.
    - This is a convention difference: the actor uses vLLM's dynamic per-tensor rule, and serving reads exactly the exported bytes, so training and inference are consistent.
    - Note: torch's `tensor / 448.0` is implemented as a multiplication by the reciprocal, so 22/46 tensors are 1 ulp below the correctly rounded value. My first reference version fell into this trap.

## Finding 1: weight decay is applied to parameters with wd_mult=0
- **Evidence**: in L4, 168 parameter groups have `wd_mult=0.0` but `weight_decay=0.1`. With the in-group wd 0.1 as reference, error ≤0.81 ulp; with the Megatron rule (wd×wd_mult=0), the update-direction cos for A_log, D, dt_bias and norm drops to 0.69–0.92. This shows wd really is applied.
- **Location**: `_MegatronLiteLRScheduler._apply` in `experimental/lite/examples/verl/verl_mlite/engine/mlite_engine.py` (around lines 133–139):
  ```python
  param_group["lr"] = lr
  if param_group.get("weight_decay", None) is not None:
      param_group["weight_decay"] = wd
  ```
  - MCore's `get_standard_config_overrides` sets `wd_mult=0` for 1-D parameters and biases. It relies on `OptimizerParamScheduler` writing `wd * wd_mult`; see `optimizer_param_scheduler.py:310`.
  - Without building a scheduler, the group's wd is also the configured value; `l4_no_scheduler.json` agrees.
- **Suggested fix** (not committed):
  ```python
  param_group["weight_decay"] = wd * param_group.get("wd_mult", 1.0)
  param_group["lr"] = lr * param_group.get("lr_mult", 1.0)
  ```
  lr_mult is ignored too. This recipe has no decoupled lr, so it has no effect on this recipe.
- **Impact**: every step x ← x·(1 − 1e-7); small, but systematic in direction. A_log being pulled toward 0 changes A=−exp(A_log), and norm weights are slowly shrunk as well. HF, NeMo and Megatron all default to not decaying these parameters.

## Finding 2: rounding of the lm_head θ0 master vs exact dequantization
- **Location**: `QuantizedWeight.initial_master` in `experimental/lite/megatron/lite/model/nemotron_h/quantization.py` (around lines 185–215); the computation order is `values * (scale.float().repeat_interleave(16) * global_scale)`, rounded twice in FP32 before converting to BF16.
- **How it was checked** (`lm_head_rounding.py`): compute the exact value in FP64, apply explicit RNE to get BF16, and compare with the actor master:
  - lm_head: 3.57% of elements differ by 1 BF16 ulp;
  - layer 1 expert 7 down_proj: 10.3%;
  - expert 0, shared experts, FP8 projections: 0.
- **Cause**: global scale = amax/1536, whose FP32 mantissa is a repeating pattern like 1/3 (e.g. `0x39455555`, `0x39a2aaab`). So the exact value lands within ±2^-24 relative of a BF16 midpoint, and any intermediate FP32 rounding can push it across the midpoint.
- The neighbor the actor picks is only ≤6e-8 (relative) farther than the correct one; numerically negligible.
- The (v·s)·g form I used earlier in `check_theta0` also misrounds: lm_head 7.9%. The "4.3%" reported earlier was the difference between the two FP32 forms.
- **Suggested fix** (if strictness is needed): compute the exact value in FP64, then convert to BF16 via a round-to-odd FP32 intermediate or explicit RNE (see `lm_head_rounding.rne_bf16`). Not fixing it is also fine; it does not affect train-inference consistency: the forward reads the deployed bytes.

## Rerun (after replacing SSD and attention backward)
- **One command**: run `scripts/mcore_ref/l3-actor.sh <commit> <tag>` locally.
  - It ships that commit's `megatron/` and `experimental/lite/megatron/` via git archive to the reference node;
  - then runs the actor in the test image with a PYTHONPATH override (L3 gradients plus L4), then runs the L3 comparison;
  - The comparison target is fixed to the MCore reference gradients `l3/base/mcore-{bf16,fp32}`.
  - Results are in `$WORKDIR/mcore_ref/l3/<tag>/{l3_compare.json, actor/l4.json}` on the reference node.
- **Localization**:
  - On the node, `run-block.sh <tag> <src> <layers>` does the block-level check;
  - `run-layer-dx.sh <tag> <src>` does the whole-model per-layer check;
  - Reproducibility check: run two tags with the same commit, then compare with `grad_repro.py`.
- **Disk**: each actor run's gradient dump is about 118 GB, the MCore reference about 236 GB, both on the reference node's local data disk.

## Files
- **Scripts** (`scripts/mcore_ref/`):

| Purpose | File |
| --- | --- |
| actor L3 gradients and L4 | `actor_l3l4.py` |
| MCore BF16/FP32 gradients, including block-level and per-layer modes | `mcore_grads.py` |
| L3 comparison | `l3_compare.py` |
| Block-level localization | `actor_block.py`, `block_compare.py` |
| Whole-model per-layer localization | `actor_layer_dx.py`, `layer_dx_compare.py` |
| Gradient reproducibility | `grad_repro.py` |
| actor per-token logprobs | `actor_logprobs.py` |
| lm_head rounding analysis | `lm_head_rounding.py` |
| Run scripts | `run-*.sh`, `l3-chain.sh`, `l3-actor.sh`, `fp32-noise.sh` |

- **Results** (`results/mcore-ref/l3l4-base-54d0/`): `l3_compare.json`, `block_compare.json`, `layer_dx_compare.json`, `fp32_self_noise.json`, `grad_repro.json`, `l4.json`, `l4_no_scheduler.json`; the lm_head analysis is in `results/mcore-ref/lm_head_rounding.txt`.
