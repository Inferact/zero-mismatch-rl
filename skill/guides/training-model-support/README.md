# Training-Side Model Support

Answers one question: on the day a model is released, how do we get a trustworthy training implementation at the lowest cost, and prove it computes the same math as the inference side. Where the training weights come from (master → training forward weights) is not covered here; see [Precision, quantization and QAT](../precision-quant-qat/README.md).

## What you have on release day

On Day-0 you can rely on only three things: the released checkpoint (weights, config, and the code shipped with the same revision, if any), the vLLM inference implementation, and model-agnostic training libraries (PyTorch, TE, PEFT, FSDP2, mamba_ssm and the like). The training implementation for this model does not exist that day: the Transformers modeling file, the NeMo AutoModel model directory, the Megatron-Bridge provider and the Miles plugin all appear only after release. We write the training implementation ourselves; it is the deliverable.

Three consequences:

1. **Semantics follow the vLLM implementation and the checkpoint.** Rollouts are produced by vLLM, and training must compute the same function. The config-parsing code shipped with the checkpoint (`configuration_*.py`) is part of the config format and can be loaded directly; the shipped modeling code is reading reference only. Where it disagrees with vLLM, vLLM wins, and the divergence is recorded.
2. **The self-written implementation goes into real source files, is maintained by us, and is eventually registered in the training framework and submitted as a PR.** The FSDP2 route delivers a verl PR; see [01 §3.3](01-module-inventory.md#33-fsdp2-verls-hf-interface-path). Third-party implementations that appear only after release are not used as the training path or as dependencies.
3. **When reviewing or rehearsing an already-released model, first check the source and appearance date of the implementations in the image.** Newer Transformers, NeMo AutoModel, etc. may already ship this model. Implementations that appeared after release must not be used as training path, dependency or "reference answer". Only after the self-written implementation passes G2 against an independent reference may they be used for cross-checking; agreement is not extra evidence, and disagreement must be investigated on both sides to find which is wrong. Correctness is decided only by the independent reference and vLLM.

## Three routes

The routes differ in where parallelism and kernels come from, not in whether the model must be written by us: all three require writing it ourselves.

| | MCore (Megatron-Bridge) | FSDP2 (VERL's HF interface path) | mLite |
|---|---|---|---|
| Positioning | Mature parallelism (TP/PP/EP/CP), main path for production RL | Lowest development cost: no parallelism to write, only the model | Built for train-inference consistency: forward calls vLLM kernels, backward reuses the training implementation |
| Forward source | MCore's own kernels (TE, grouped GEMM) | A PyTorch model we write following vLLM semantics, wrapped in the Transformers `PreTrainedModel` interface for the VERL loader and PEFT | vLLM kernels (`visible_forward(visible, native, ...)`) |
| Achievable consistency | BI=0, via R3 + importance sampling | BI=0 | BI=1, logprob raw bytes identical |
| What a new model requires | Bridge weight mapping + modules MCore lacks (Mamba/SSM, special norms) | The whole model: config class, checkpoint loading (including dequantization), per-layer forward, per-operator precision contract; backward prefers autograd, filled in where libraries lack it | One `model/<name>/` directory: config, checkpoint mapping, model assembly, protocol; modules picked from `primitive/modules` |
| Typical pitfalls | Mapping checks values, not semantics; MCore kernels and vLLM use different math (MoE activation clamp definition) | bf16 autocast outside VERL's forward rewrites module precision; `--system-site-packages` ABI; PyTorch reference implementations (Mamba scan) too slow to use; mistakenly using post-release implementations in the image | Privatizing modules into the model directory, bypassing the shared skeleton |

Order for choosing a route: first do the [01 module inventory](01-module-inventory.md), then choose by "is BI=1 needed, how many backwards are missing, what parallelism is needed". Nemotron lesson: from the model name we assumed it was a low-cost model, and only three weeks later found that three modules — FA4, Mamba scan, W4A16 MoE — had no backward.

## Find by symptom

| Symptom / question | Go to |
|---|---|
| New model loads and config parses, but unclear whether it can be trained | 01 module inventory |
| The image already has a Transformers / AutoModel implementation of this model; should we use it | This page, "What you have on release day" |
| Want to reuse vLLM forward code, tangled in paged KV / cudagraph / scheduling | 02 call kernels only, do not port the scheduler |
| Weight mapping shapes match, but model output silently degrades | 02 mapping checks values and semantics; fused params overwritten as whole params |
| Inference kernel has no backward; should we write one | 03 three-case decision |
| Wrote a VJP; how to prove it correct | 03 module-level comparison steps, two kinds of baseline |
| Unit tests pass but real training fails / occasional non-bitwise at step 0 | 03 two test-infrastructure false negatives; BI kernel appendix |
| Changes after altering packing, CP/TP/EP; only long sequences fail | 04 layout and length |
| After changing the rollout parallel layout (e.g. DP+EP to TP1 per GPU), `ppo_kl` is non-zero | [04 bitwise consistency after changing parallel layout](04-layout-and-length.md#bitwise-consistency-after-changing-the-parallel-layout) |
| Without recomputing old-logprob, update fails with `modified by an inplace operation` | [05 entropy when not recomputing old-logprob](05-framework-adapter.md#entropy-when-old-logprob-is-not-recomputed-bypass) |
| How to hook into VERL / Megatron | 05 framework integration |
| Which kernel an operator uses in BI mode and at what cost | [Appendix](appendix-bi-kernels.md) |

## Key rules

1. Write the training implementation ourselves; semantics follow the vLLM implementation and the checkpoint; third-party implementations that appear after release are not path, dependency or reference answer (see the section above).
2. Loads ≠ trainable; before starting, list for each module whether its inference kernel has a backward.
3. Call kernels only, do not port the scheduler. mLite (BI=1) compares FP32 logprob raw bytes against trajectories captured from vLLM, target 0 diff; FSDP2 / MCore (BI=0) do not chase 0 diff; per the day-0 acceptance criteria, first use an independent reference to judge whether each side is correct, then look at precision.
4. Mapping checks cover values and semantics, not just shape; fused params must be overwritten as whole params.
5. The precision contract lives in the model. Where vLLM computes in FP32 (positional-encoding angles, router logits and bias, norm statistics, etc., each per vLLM), the training module specifies dtype explicitly and disables the outer autocast for that computation. VERL's training forward runs under bf16 autocast; for matmul-type ops (including `F.linear` and `einsum` that gets split into bmm), casting only the inputs does not prevent downcasting; acceptance must measure output dtype and values under the real autocast.
6. Backward is determined by the math, independent of which kernel the forward uses; only rounding differs → reuse directly; quantization rounding present → define the STE contract, then reuse BF16 GEMM; no existing backward for the whole math segment → write glue, GEMM still uses TE. "No backward kernel at the same precision" is not a reason to rewrite.
7. What gets reused is model-agnostic operators: first look in mLite `primitive/modules` and `ops` for an implementation of the same math, then in MCore/TE/mamba_ssm; only when neither has it does it count as new work.
8. Backward is not required to be bitwise; report bitwise and tolerance separately; the reference builds its graph independently with ATen autograd; tolerances are fixed before looking at results; synthetic inputs are labeled as not real activations.
9. Do not recompute activations: use intermediates saved by the forward directly (DS4 uses gate_up / LSE).
10. Module placement: the model directory holds only config / checkpoint / model / protocol; reusable pieces go into `primitive/`; do not create a same-named `primitive/` subtree under the model directory; impl name is `lite`.
11. Long-sequence boundary points (KV page, packing prepend, the 7000/9216 tiers) must be tested specifically; passing short sequences does not mean long sequences pass.
12. Parts that do not participate in training (MTP block, EPLB redundant experts, draft) must be explicitly skipped in mapping and sync.

## Files

| File | Content |
|---|---|
| [01-module-inventory.md](01-module-inventory.md) | How to build the inventory (Nemotron filled example); loads ≠ trainable (three kinds of omission, cost map, seven questions); mLite / MCore-Bridge / FSDP2 placement rules; parts not participating in training (MTP, EPLB redundant experts, draft); summary of the Nemotron 20-file reuse audit (full text in [cases/nemotron](../../../cases/nemotron/mlite-reuse-audit.md)); appendix: Miles cross-model comparison |
| [02-reproduce-forward.md](02-reproduce-forward.md) | Call kernels only, do not port the scheduler; read the inference implementation along the full support chain; mapping checks values and semantics, fused params overwritten as whole params; packing boundaries of stateful modules; compare FP32 logprob raw bytes using captured trajectories; small tasks and stop conditions |
| [03-reuse-backward.md](03-reuse-backward.md) | Three-case decision, DS4 comparison, three reuse rules, two false negatives |
| [04-layout-and-length.md](04-layout-and-length.md) | World-size formula and layout-dependent GPU memory; packing/CP boundaries; EP and collectives; long-sequence boundary point table; attention backward D term; microbatch gradient diff 2e-3; training-side correctness acceptance (gradient comparison criterion v2, optauth) |
| [05-framework-adapter.md](05-framework-adapter.md) | Where the integration layer is, capture location, historical DS4 entry, minimal adapter contract, training extensions; recipe review and two-step acceptance convention moved to [validation.md](../../validation.md#recipe-review-materials-and-two-step-acceptance-conventions) |
| [06-backward-validation.md](06-backward-validation.md) | Three layers of backward validation: gradient contract (three weight tiers, dX values), module-level VJP comparison and reference bridging, gradient-contract comparison, proxy SFT three-group comparison (including per-module gradient comparison on the same batch), common false differences, reporting requirements |
| [appendix-bi-kernels.md](appendix-bi-kernels.md) | Per-operator BI kernel table (attention, dense GEMM, mHC, two kinds of MoE, top-k, Mamba2, RMSNorm, FP8 quantization, cudagraph, determinism, KV zeroing, prefix caching); top-k output contract; index fusion; the M of split-K; determinism only on the main thread; how to break down cost; minimal record per candidate |

## Evidence index

- Nemotron-H: [train-inference consistency delivery](../../../cases/nemotron/consistency-case.md), [DAPO failure case](../../../cases/nemotron/dapo-20260927.md); the 2026-09-30 validation log for independent VJPs (Mamba, MoE, FA4) has been deleted from the source history.
- DeepSeek V4: [case](../../../cases/deepseek-v4-flash/consistency-case.md), [CP boundary](../../../cases/deepseek-v4-flash/cp-compressed-boundary.md).
- Model E: training lessons (case notes not included here).
