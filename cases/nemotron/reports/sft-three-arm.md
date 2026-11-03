# Proxy SFT three-arm comparison (A under test / B gradient reference / C native baseline)

Date 2026-10-03. The criteria were written before the runs started (10:45 UTC) and have not been changed since; issues found afterwards are marked separately in the results.

## Conclusion
- **A/B (backward implementation): under A's own gradient contract, all 4 snapshots pass.**
  - At θ0, it passes in one go under B's original contract: per-class tensor ratio medians 0.87–1.01, max 1.27; one-step update relL2 0.066, floor 0.065.
  - After training (steps 100/200/300), the router gradient deviates by 60–80% (ratio 106–165, cos 0.77–0.88), and this propagates through backward to all upstream parameters.
  - Localization: the difference comes from the VJP contract of the routing weights. A follows DS4's (#7050) `routed_vjp`, computing d_routes with `down_master · relu(fc1_visible)²`; B originally used the visible expert output. At θ0 the master equals the deployed value and the two forms agree; afterwards the master drifts while the NVFP4 deployed value does not move, and only then do they diverge.
  - After switching B to A's contract, all 3 snapshots are back at the floor: class median ≤1.07, single-tensor max 1.38.
  - Conclusion: A's backward correctly implements its own contract. But the README does not state this contract; it is a case of "a master proxy and the deployed value must not be silently swapped".
- **A/C (training effect): S1 and S2 fail.**
  - Within the window, A's validation NLL drop is 0.72–0.95 nat smaller than C's, about 70–90x the reference noise (data-order change, 0.011 nat). Step 300 val NLL: A 4.405, C 3.464.
- **The A/C gap is not caused by backward, but by the deployment lag of the quantized forward. Three pieces of evidence:**
  1. **The NVFP4 deployed values do not move at all in 300 steps.** routed/shared experts and lm_head together make up 86.7% of the parameters. Their master moved 0.23–0.86% relative to θ0, while the deployed value moved ≤3e-5 (mostly 0). The reason is that the θ0 master sits exactly on NVFP4 grid points, and at lr 1e-6 the accumulated movement is far less than half a grid step, so RNE requantization does not flip codes. For the FP8 Mamba in_proj, only 13–16% of the master movement reaches the deployed value.
  2. **An independent implementation reproduces A's curve.** The B curve is a pure-PyTorch rewrite of the same quantization numerics; it calls none of A's code and shares only TE's NVFP4 quantizer. Under A's d_routes contract it differs from A by ≤0.008 nat throughout; under the visible-output contract it differs by at most 0.020 nat (ending 0.019 lower).
  3. **Freezing the corresponding weights in the native model brackets the gap.** C with NVFP4-class weights frozen ends at 3.952, explaining half of the gap; with the FP8 projections also frozen it ends at 4.788, worse than A. A lies between the two, consistent with "NVFP4 fully frozen, FP8 partly updated".
- **What this means for the delivery.**
  - At the recipe's lr 1e-6, 86.7% of the parameters do not change in their inference-visible values for at least several hundred steps (for the full model this is inferred from the same mechanism, not measured). Short RL training effectively updates only the BF16 parameters and part of the FP8 projections.
  - This is a property of the quantization scheme (starting from grid points, RNE requantization), not an implementation defect. But it determines the training-effect gap between this recipe and native BF16, and it must be stated clearly in the delivery notes.

## Setup
| Item | A (under test) | B (gradient reference) | C (native baseline) |
| --- | --- | --- | --- |
| Code / image | the d9 Megatron commit (an internal commit), i.e. the d9 image (`<image>`, the d9 delivery image, sha256:885bfdd5…) | `scripts/sft3/b_ref.py`, `b_curve.py`, pure torch implementation, run in the d9 image | `<image>` (MCore reference image t1): MCore `52554d5`, Megatron-Bridge `1f8873bb` |
| forward | vLLM kernels plus deployed bytes (NVFP4 W4A16, FP8 W8A8, FP8 QKV) | Snapshot comparison: bridges A's visible values. Curve: its own quantized forward, deployed values dequantized from the checkpoint bytes, requantized every step with TE NVFP4 4over6 and FP8 amax/448 | BF16, no quantization at all |
| backward | Production VJP: TE high_precision, mamba_ssm, FA4, closed-form norm VJP | torch autograd, calls none of the VJPs under test | MCore native (TE, mamba_ssm a14b1df, cuDNN/flash attention) |
| Parallelism | PP4, dist_opt, precision-aware CPU offload Adam | Single GPU | Single GPU TP1/PP1/EP1, DistributedOptimizer DP1 |
| mamba_ssm | 0048fbf | Not used | a14b1df (different from the delivered version) |

- **Common controls:**
  - θ0 always comes from the dequantized master of the proxy NVFP4 checkpoint. The checkpoint is `proxy5-contiguous-r1/model`, made of real layers 2–6, with structure M/E/M/*/E. C and the B curve read the θ0 exported by A itself (`export_theta0.py`, 548 tensors).
  - Data is the official GSM8K `main`: `Question: …\nAnswer: …<eos>`, loss on the response only.
  - 16 samples per step, one micro-batch, no gradient accumulation, 3024 tokens on average, of which 1980 are response. Data order fixed by seed 42.
  - Loss is the token-mean NLL over response tokens.
  - Optimizer: Adam lr 1e-6 (constant), betas (0.9, 0.95), eps 1e-8, wd 0.1, global clip 1.0, FP32 main, BF16 parameters = RNE(main). 300 steps in total.
- **lr:** kept at the recipe's 1e-6, not increased, because the trend is already very clear at 1e-6 (val NLL drops from 16.3 to 3.5).
- **Known and intentionally kept differences:**
  - **weight decay:** A's verl_mlite scheduler also applies wd to 1-D parameters; this is finding 1 in `mcore-grad-optimizer-compare.md`, still not fixed in the d9 commit. C follows the Megatron rule, with wd_mult 0 for 1-D parameters and biases. The B curve follows A. The per-step magnitude is 1e-7.
  - **router GEMM:** A uses BF16 input and FP32 output; C is set to FP32 by Bridge.
- **Validation set and probe:**
  - Validation set: the first 128 items of GSM8K test, 15247 response tokens in total.
  - Probe: the batch of training step 0, which also serves as the representative batch for A/B.
- **Reference noise:**
  - Rerunning C with the same seed differs by 0.0014 nat, so it is not bitwise reproducible;
  - C with a different data order (seed 43) differs by 0.011 nat.

## Pre-agreed comparison windows and tolerances (written before the runs; original text kept)

### A/B: backward implementation (representative batch)
- Representative batch = probe batch = the batch of training step 0 (16 GSM8K items, about 3000 tokens, of which about 2000 are response tokens).
- Snapshots: steps 0, 100, 200, 300 on A's training trajectory (step 300 is after the last update). At each snapshot, B uses A's BF16 master and the visible values of A's inference forward.
- B runs in BF16 (B16) and in FP32 (B32, TF32 off). B32 is the reference, and the B16-vs-B32 difference is the noise floor. The per-tensor statistic is ratio = relL2(A, B32) / relL2(B16, B32).
- Pass conditions:
  - G0: B forward sanity. The relL2 of each block's surrogate output against the visible values is ≤ 0.05 for both B16 and B32; otherwise B itself is judged wrong and the comparison is invalid.
  - G1: all of A's gradients are finite, no missing tensors; tensors non-zero in B32 are also non-zero in A.
  - G2: per tensor class, ratio median ≤ 1.5.
  - G3: each tensor has ratio ≤ 3, or relL2(A, B32) ≤ 0.01. Tensors over the limit are listed and explained one by one; anything unexplained is a failure.
  - G4: the total gradient norm satisfies |‖g_A‖/‖g_B32‖ − 1| ≤ 3·|‖g_B16‖/‖g_B32‖ − 1| + 0.005.
- One-step update (only at step 0, when the optimizer state is empty):
  - Replay Adam's first step in FP64 with the parameter-group hyperparameters actually used by A (lr, wd, eps, betas) and each side's own global-norm clip.
  - U1: relL2(ΔA, ΔB32) ≤ 3 × relL2(ΔB16, ΔB32), judged globally.
  - U2: A's FP32 main update vs the FP64 update replayed from A's own gradient, global relL2 ≤ 1e-5, to confirm the optimizer path.

### A/C: training effect
- Window: steps 100–300, evaluated every 20 steps, 11 points in total. The x-axis gives both step and cumulative response tokens.
- Primary metric: fixed validation-set NLL (first 128 items of GSM8K test, 15247 response tokens).
  - What is compared is the change from the start, ΔNLL(t) = NLL(t) − NLL(0). The A/C forward difference at θ0 (quantization) is reported separately and not counted.
- Reference noise is the larger of:
  - the difference between C with a different data order (seed 43) and C (seed 42), |ΔC42 − ΔC43|, max over the window;
  - the difference between repeated C seed 42 runs, |ΔC42 − ΔC42b|.
- Tolerance τ(t) = max(reference noise, 0.05 × |ΔC42(t)|).
- Pass conditions:
  - S1: every evaluation point in the window satisfies |ΔA(t) − ΔC42(t)| ≤ τ(t).
  - S2: probe NLL judged by the same rule.
- Systematic anomaly checks (whole run):
  - A1: 0 non-finite values, 0 skipped steps;
  - A2: median of the grad norm ratio A/C within the window in [0.8, 1.25];
  - A3: median of the update RMS ratio A/C in [0.8, 1.25].
- Decision rules:
  - An unmet item that can be explained by known non-backward factors (forward quantization, known optimizer differences) is recorded as inconclusive, with the reason;
  - Unexplained means failure.
  - A short run only proves short-range behavior; it is not used to claim long-term convergence equivalence.

## A/B results

### Construction of B
- `b_ref.py` plugs in the visible values of A's inference forward at each block boundary with `VisibleWithSurrogate`:
  - Values used: A's block output stream, route ids, and the visible logits recomputed with A's `projection`.
  - All surrogate blocks are implemented in torch:
    - Mamba2: F.conv1d plus HF's chunk scan (FP32);
    - attention: SDPA math backend, FP8 STE on Q/K/V;
    - Quantized linear layers: forward value uses the deployed dequantized value and FP8 input; gradient uses the BF16 master and unquantized input;
    - MoE: route ids fixed to A's, router uses FP32 sigmoid-gather-renorm.
- **Bridge check:** the losses of all three are bitwise identical (e.g. at step 0 A/B16/B32 are all 16.055996). The surrogate forward's relL2 against the visible values is ≤ 3.3%, the largest being layer 0 Mamba in B32. G0 passes.

### Gradient comparison (ratio = relL2(A,B32) / relL2(B16,B32))

| Snapshot | B's d_routes contract | Max of per-class ratio medians | Max single-tensor ratio | router relL2(A) / cos | ‖g‖ A / B16 / B32 | G1–G4 |
| --- | --- | --- | --- | --- | --- | --- |
| step 0 | Visible expert output | 1.01 | 1.27 (L0 D) | 0.053 / 0.999 | 242.39 / 242.36 / 243.12 | All pass |
| step 100 | Visible expert output | 106 | 106 (L4 gate) | 0.60 / 0.88 | 37.28 / 37.45 / 37.86 | G2, G3 fail |
| step 200 | Visible expert output | 155 | 155 (L4 gate) | 0.60 / 0.84 | 20.31 / 20.27 / 19.85 | G2, G3 fail |
| step 300 | Visible expert output | 165 | 165 (L4 gate) | 0.81 / 0.77 | 20.09 / 20.02 / 20.08 | G2, G3 fail |
| step 100 | Master expert output (A/DS4) | 1.07 | 1.38 (L3 norm) | 0.084 / 0.997 | 37.28 / 37.34 / 37.81 | All pass |
| step 200 | Master expert output (A/DS4) | 1.03 | 1.15 | 0.040 / 0.999 | 20.31 / 20.32 / 19.88 | All pass |
| step 300 | Master expert output (A/DS4) | 1.16 | 1.21 | 0.017 / 1.000 | 20.09 / 20.05 / 20.11 | All pass |

- **How the difference propagates:**
  - The difference caused by the contract starts at the router of layer 4 (the last MoE);
  - it passes through the layer 4 norm (ratio 59) and dX to shallower layers;
  - at step 300, attention o/v (ratio 22–30), the norms of each layer, FP8 projections and experts (ratio 5–12) all exceed G3.
- **Why the deviation is so large:** for a renormalized top-k, d_logits is the residual of dw_k minus its weighted mean, with heavy cancellation. So a ~0.3% difference in the expert output (master vs deployed value) is amplified into a 60–80% difference in the router gradient.
- **Impact on training:** small. The two B curves (two contracts) differ at the end by 0.011 nat, comparable to the data-order noise.
- **Recommendation:**
  - State in the README's Backward section: d_routes uses the expert output recomputed from the master;
  - or switch to the visible output saved in forward, at the cost of storing an extra [M·topk, K] BF16 tensor.
  - Whether to change it is the owner's decision. For now it can only be called a defensible but undeclared contract, not a bug.
  - **Correction (late 2026-10-03)**: later, checked against DS4 (item 2 of the strict delivery audit reply in the cross-review notes, not included), DS4's route gradient is `<dy, visible expert output>`, i.e. the exact derivative; our recomputation from the master is an undeclared proxy, **recorded as our bug by the delivery standard**, and the delivery has been changed to the visible-output contract. The passes under the "master expert output" contract in the table above no longer represent the delivered implementation; the variable-weight snapshot A/B under the visible-output contract on p5 (the p5 Megatron commit) is in `results/final-p5/step7-ab/`.

### One-step update at step 0
- **U1 passes:**
  - relL2(ΔA, ΔB32) = 0.0663, relL2(ΔB16, ΔB32) = 0.0651, threshold is 3x the latter;
  - Sign agreement: A vs B32 99.67%, B16 vs B32 99.78%;
  - Clip coefficient: A 0.004126, B32 0.004113.
- **U2 fails under the original threshold, but the criterion itself was written wrong:**
  - A's global relL2 against its own FP64 replay is 1.15e-3;
  - Merely storing the exact result in FP32 already gives 0.88e-3. The reason is Δ is about 1e-6 and ulp(x0)/Δ is about 1e-3. The 1e-5 threshold cannot be reached at all.
  - Using the measure of L4 in `mcore-grad-optimizer-compare.md` instead (error measured in FP32 ulp of max(|x0|, |Δ|), excluding elements whose second moment is an FP32 subnormal): max 3.22 ulp, 1.4% of elements above 1 ulp. Consistent with L4's ≤3.04 ulp, ≤1.5%; judged a pass.
  - This measure was changed after the fact, and is flagged here as such.
- **requant check (`check_requant.py`, step 300):**
  - All NVFP4 deployed bytes equal the result of TE 4over6 applied to the BF16 master, byte for byte;
  - FP8 codes and scales equal the result of the amax/448 rule. Layer 2 has 34/25 codes and 1 scale different, the same kind of difference as the torch division vs multiply-by-reciprocal recorded in L4.

## A/C results
| Metric (step 300) | A | C | C rerun | C seed43 | B curve (visible d_routes) | B curve (master d_routes) | C with NVFP4 class frozen | C with NVFP4 class and FP8 projections frozen |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| val NLL | 4.405 | 3.464 | 3.466 | 3.469 | 4.386 | 4.397 | 3.952 | 4.788 |
| probe NLL | 4.000 | 3.090 | 3.089 | 3.091 | 3.988 | — | 3.552 | 4.392 |

- **Starting point (forward difference at θ0):** A − C val NLL is −0.007, probe −0.054. The B curve differs from A by −0.0001.
- **S1 (val) fails:**
  - ΔA − ΔC at steps 100/200/300 is 0.72/0.83/0.95 nat;
  - τ is 0.56–0.64, almost entirely from the 0.05·|ΔC| term;
  - the actual reference noise is only 0.011.
- **S2 (probe) fails:** the difference is 0.73–0.96.
- **A1 passes:** non-finite values and skipped steps are 0 on all branches. Every branch triggered clipping at every step (grad norm 10–300, far above 1); Adam normalizes per element, so clipping only affects the relative size of eps.
- **A2 passes:** median grad norm ratio A/C is 1.23, right at the edge. A's loss is higher, so its gradient is larger too.
- **A3 passes:** median update RMS ratio is 1.08.
- **Verdict:**
  - The S1 and S2 gaps can be fully explained by known non-backward factors, see below; by the pre-agreed rules they are recorded as **inconclusive (due to the quantized forward)**, not a failure.
  - But it does reflect a training-effect loss of the delivered recipe relative to native BF16.
- **Attribution evidence (`deploy_lag.py`, `check_requant.py`, comparing step 0 and step 300):**

| Weight | Master relative movement | Deployed-value relative movement | Projection of deployed movement on the master direction |
| --- | --- | --- | --- |
| routed experts (L1, L4) | 0.23–0.29% | ≤4e-6 | 0 |
| shared experts | 0.53–0.86% | ≤3e-5 | 0 |
| lm_head | 0.56% | 0 | 0 |
| Mamba in_proj (FP8) | 0.37–0.54% | 0.15–0.25% | 0.13–0.16 |
| Mamba out_proj (FP8) | 0.52–0.62% | 3.0–3.1% | 1.0–1.5 (cos 0.18–0.30) |

  - 7–8% of NVFP4 code bytes changed but the values did not; presumably ±0 encoding differences.
  - **The out_proj phenomenon (new finding):**
    - The checkpoint's FP8 scale is 0.004255, larger than amax/448 = 0.002738, so the largest code is only 288;
    - After the first requant switches to amax/448, 100% of the codes are re-encoded, and the deployed value shows a 2.9% quantization error relative to the master;
    - This jump happens once; it is not step-by-step learning;
    - It shows that the README's claim "requantizing with the checkpoint's own rule reproduces the checkpoint" does not hold for out_proj (proxy layers 0 and 2, i.e. full-model layers 2 and 4; other layers not checked).
  - Freezing experiments:
    - C with routed/shared experts and output_layer frozen (2.947B parameters, 86.7%) reaches val 3.952, explaining 0.49 of the 0.94 nat gap;
    - with Mamba in/out_proj also frozen, val is 4.788, above A.
    - A lies between the two, consistent with "NVFP4 does not move at all, FP8 projections partly update".
  - The grad norm and update RMS of the frozen branches count only trainable parameters and are not directly comparable to A/C.
- **Throughput (median):**
  - A: 0.60 s/step, 5.0k tokens/s (4 GPUs PP4, CPU offload Adam);
  - C: 0.17 s/step, 18k tokens/s (single GPU);
  - B curve: 1.08 s/step (single GPU, pure torch).

## Missing scope and limitations
- B has a full 300-step curve (`b_curve.py`), but its forward is an independent implementation with 1–3% block-level differences from A's visible values; it is not A's bitwise visible values. The bitwise-bridged B was compared only at 4 snapshots, and the one-step update comparison was done only at step 0; there is no A/B one-step update comparison at steps 100–300 (no optimizer state dump).
- The proxy has only 5 layers (M/E/M/*/E) and is a slice of the full model: initial NLL 16.3, above the uniform 11.8. Its response to SFT differs from the full model's.
- "NVFP4 deployed values stay unchanged for a long time" is inferred for the full model from the same mechanism, not measured.
- C is not bitwise reproducible (rerun differs by 0.0014 nat). The reference noise has only 1 rerun and 1 data-order seed.
- The 0.05·|ΔC| term in τ is set too wide, about 50x the actual noise. Even so S1/S2 fail, so the conclusion is unaffected.

## Files
- **Scripts `scripts/sft3/`:**

| File | Purpose |
| --- | --- |
| `prep_data.py` | GSM8K tokenization |
| `sft_a.py`, `run-a.sh` | A training and snapshot dumps |
| `sft_c.py`, `run-c.sh` | C training; `--freeze` for the freezing diagnosis |
| `b_ref.py`, `run-b.sh` | B snapshot comparison; `B_ROUTE_DW=master` switches to A's d_routes contract |
| `b_curve.py`, `run-bcurve.sh` | B curve |
| `deploy_lag.py`, `check_requant.py` | Deployment lag and requant checks |
| `analyze.py` | Curves, CSV, criteria computation |
| `guard.sh` | GPU idle check and occupancy guard |
| `sync.sh`, `fetch.sh`, `stage-images.sh`, `run-export.sh` | Sync, fetch, image staging, θ0 export |

- **Results `results/sft3/`:**
  - `curves_step.png`, `curves_tokens.png`
  - `train.csv`, `eval.csv`
  - `criteria.json`
  - `runs/*/metrics.jsonl`, `runs/*/meta.json`
  - `b/b-s42-step{0,100,200,300}[-dwmaster].json`
  - `b/deploy_lag_0_300.json`
- **Shared storage `$WORKDIR/stage/sft3/`** keeps the following so the runs can be repeated: data, the proxy θ0 BF16 export (6.3 GB), and the MCore reference image tar (14 GB).
- **Cleanup:** snapshots, images and data on the two nodes used have all been deleted; the guard log shows no processes from other users.

## Addendum: A initialized from a BF16 master (A-bf16, 2026-10-03 14:28Z)

Change: A's master is no longer obtained by dequantizing the checkpoint, but loaded from the checkpoint's BF16 source (`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16@a9904d2`, layers 2–6 cut the same way as the proxy). θ0 deployment = requant(BF16 master), with rules TE 4over6 MSE, global amax/1536, FP8 amax/448 (code E4M3(BF16(w/s))). Everything else is identical to the original A: data, seed 42, lr 1e-6, 300 steps, PP4 dist_opt.

- Code: the BF16-master Megatron commit (an internal commit on a BF16-master branch, based on the DS4-structure branch).
- Image: `<image>` (a d9 DeepEP image variant), tar sha256 5555178c…, TE 2.17.
- Full provenance and commands: `results/bf16src/sft-a-bf16/provenance.json`; curve: `metrics.jsonl`.

| step | A-bf16 val NLL | C | Original A |
| --- | --- | --- | --- |
| 0 | 16.244 | 16.283 | 16.276 |
| 100 | 5.131 | 5.137 | 5.850 |
| 200 | 3.942 | 3.941 | 4.768 |
| 300 | **3.4685** | **3.4641** | **4.4046** |

- Under the criteria written before the runs:
  - S1: within the 100–300 window, max |ΔA − ΔC| is 0.045 nat, τ is 0.56–0.64, worst ratio 0.07, **pass**.
  - S2: the probe difference within the window ≤ 0.013 nat, **pass**.
  - The original A differed by 0.72–0.95 nat in the same window, fail.
  - Absolute val NLL: within the window A-bf16 and C differ by ≤ 0.006 nat, within C's data-order noise of 0.011 nat. The ~0.04 difference between ΔA and ΔC comes almost entirely from the starting difference at θ0: A's forward is quantized, C's is native BF16.
  - A1: 0 non-finite values, 0 skipped steps.
- **The deployed bytes finally move.**
  - After the first step, NVFP4 experts have 1.3e-5 code flips, then on the order of 1e-4 per step. By step 300, relative to θ0, 1.5% of expert codes and 6.2% of shared/lm_head codes differ, and 11.8% of FP8 codes differ.
  - The original A's NVFP4 deployed values moved ≤3e-5 in 300 steps, mostly 0.
- Conclusion: the gap between the original A and C is entirely caused by "the master starting on NVFP4 grid points, and RNE requantization not flipping codes". Starting from a true BF16 master, the quantized actor's SFT effect matches native BF16.

## Addendum: same-tree comparison (2026-10-03 17:30Z, one node)

Responding to review: Reviewer A P1-2 (A-bf16 and the original A are on different code trees), Reviewer B P1 (C's θ0 comes from the old actor's dequantized master, not the release BF16).

- **A-ckpt (same tree)**: the same staged tree as A-bf16 (the BF16-master Megatron commit, the staged `lite` tree, unchanged since A-bf16 started), the same image (tar sha256 5555178c…, id ff9bc57e…), the same sft_a.py, data, seed, lr, step count and env. The only difference is `MLITE_BF16_MASTER=` (empty), so the θ0 master is the dequantized checkpoint.
  - CUBLAS: both runs were started with `-e CUBLAS_WORKSPACE_CONFIG=:4096:8`. vLLM batch_invariant initialization changes the in-process value to `:16:8` on SM100, so both meta.json files record `:16:8`. The provenance and meta do not actually contradict each other.
- **C-bf16**: uses sft_c.py and the MCore reference image t1, the same as the old C; the only difference is that the θ0 weights are replaced with the release BF16 proxy cut used by A-bf16 (a9904d24, layers 2–6, safetensors byte-identical). Config and tokenizer kept from the old C.
- Results: `results/bf16src/sft-a-ckpt-sametree/` (provenance, metrics, deploy-delta, `eval.csv`, `train.csv`, `curves.png`, `summary.json`), and `results/bf16src/sft-c-bf16/`.

| step | A-ckpt same tree | A-bf16 | Original A (d9 commit) | C (dequantized θ0) | C-bf16 (release θ0) |
| --- | --- | --- | --- | --- | --- |
| 0 | 16.276 | 16.244 | 16.276 | 16.283 | 16.201 |
| 100 | 5.798 | 5.131 | 5.850 | 5.137 | 5.108 |
| 200 | 4.721 | 3.942 | 4.768 | 3.941 | 3.909 |
| 300 | **4.3533** | **3.4685** | 4.4046 | 3.4641 | **3.4373** |

- **The 7 intermediate commits are not the cause.** Same-tree A-ckpt is 0.045–0.061 nat lower than the original A within the 100–300 window, and 0.67–0.88 nat higher than A-bf16. The gap comes from the BF16 master.
- **With C switched to the release θ0, the A-bf16 vs C gap is no longer ≤0.006.**
  - Within the window A-bf16 − C-bf16 is 0.023–0.038 nat, 0.031 at step 300, above C's data-order noise of 0.011.
  - This gap already exists at θ0 (0.044) and does not widen during training: the drop from each side's own step 0 is −12.775 for A-bf16 and −12.763 for C-bf16.
  - So the remaining ~0.03 nat is the cost of the quantized forward itself (A uses NVFP4/FP8, C native BF16), not a gap in training dynamics. In the old table A-bf16 ≈ C partly because the old C's dequantized θ0 started about 0.03 higher.
- Reruns are not bitwise repeatable: two A-ckpt runs with the same configuration (try2 stopped at step 120) differ by up to 4e-3 nat at the same step.
- Run log: jobs from another session on the same node took the GPUs twice, and the guard stopped A-ckpt twice (try1 was stopped before step 0, try2 stopped at step 120). The third attempt ran to completion; partial outputs are kept on shared storage at `runs/a-ckptst-s42.try{1,2}`.
- Scripts: `scripts/bf16src/sft/run-a-ckpt-sametree.sh`, `run-c-bf16.sh`, `driver-sametree.sh`, `analyze_sametree.py`.
