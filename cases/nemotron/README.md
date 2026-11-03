# Nemotron-H 30B-A3B NVFP4 true on-policy RL

## Current progress

- Status: **delivery complete** (2026-10-06, three PRs + one image, see the table below).
- Current phase: post-delivery performance wrap-up. The reason the BI=0 arm's rollout is slower than BI=1 has been located to two floors, the R3 return path and frontend logprobs unpacking; fixes are in two draft PRs on an internal vLLM fork (not in the delivery image), see [r3-return-path.md](r3-return-path.md). 2026-10-07 100-step retrain: with both fixes applied, BI=0 rollout per token is on par with BI=1 (−0.6%), while the whole step is still 6.6% slower, the difference being old-logprob ([100-step retrain table](r3-return-path.md#100-step-retrain-2026-10-07)).
- Consolidated conclusions are in the guides: [Precision, quantization and QAT](../../skill/guides/precision-quant-qat/README.md), [Checkpoint, resume and host memory](../../skill/guides/checkpoint-resume-memory/README.md); this directory keeps the facts and numbers the guides reference.

## Deliverables

| Item | Location |
| --- | --- |
| vLLM | An internal vLLM fork PR |
| Megatron-LM mLite | An internal Megatron-LM fork PR (the image was built from an earlier head, after which only a version check for `empty_host_cache` was added); the dense.py recompile fix is in a separate fork PR |
| verl recipe | An internal verl fork PR; the AgentLoopWorker host memory fix is in a separate fork PR |
| Image | `<image>` (a unified Nemotron-H true on-policy image, linux/arm64), built from the three fork branches above; the verl commit in the image has only one extra test commit beyond the verl PR |
| BI=0 vs BI=1 comparison | An internal W&B project (not linked) |

Delivery notes and acceptance data: [results/final/README.md](results/final/README.md); p6 inference-side acceptance: [results/final-p6/inference/README.md](results/final-p6/inference/README.md).

## Key results

- [consistency-case.md](consistency-case.md): cross-stack arithmetic, Mamba2 state and effective updates; 7-layer localization and 52-layer training experiments.
- [dapo-20260927.md](dapo-20260927.md): 2026-09 failure case where DAPO ran five steps but train-inference logprobs were misaligned from θ0 onward.
- [recipe.md](recipe.md): deviation table and open items of the internal verl recipe PR against verl#7688.
- [reports/bi0-vs-bi1-100step.md](reports/bi0-vs-bi1-100step.md): comparison of the first 100 steps of BI=0 and BI=1 with seed 42.
- [r3-return-path.md](r3-return-path.md): R3 return path overhead in the BI=0 arm, root cause, fixpack and equivalence evidence.
- [mlite-reuse-audit.md](mlite-reuse-audit.md): 20-file reuse audit of the mLite scaffold.
- Acceptance JSON (2026-09-29): [direct-fa4](history/nemotron-direct-fa4-acceptance-20260929.json), [shared-fi](history/nemotron-shared-fi-20260929.json).

## Reports

| Topic | Reports |
| --- | --- |
| Admission and decisions | [Skill admission record](reports/skill-admission.md), [Long formal run](reports/long-run-formal.md) |
| Training correctness | [Gate 4 VJP](reports/gate4-vjp.md), [Gradients and optimizer vs MCore](reports/mcore-grad-optimizer-compare.md), [SFT three-arm comparison](reports/sft-three-arm.md), [mbinv first step](reports/mbinv-first-step.md), [EP4 + CuTe-DSL](reports/ep4-cutedsl.md) |
| Inference BI and prefix caching | [BI Mamba2 prefix cache](reports/bi-mamba2-prefix-cache.md), [Prefix cache performance](reports/prefix-cache-perf.md), [Cost of enabling PC under BI=0](reports/bi0-prefix-caching-cost.md), [BI=0 BF16 collapse](reports/bi0-bf16-collapse.md) |
| Precision and QAT | [B1 activation fake-quant](reports/b1-activation-fakequant.md), [QAT activation quantization survey](reports/qat-activation-quant-survey.md), [NeMo-RL low-precision survey](reports/nemo-rl-low-precision-survey.md), [FP8 attention backward survey](reports/fp8-attn-bwd-survey.md), [NVFP4 RL stagnation](reports/research-nvfp4-rl-stagnation.md) |
| Weight sync | [NVFP4 online sync](reports/nvfp4-online-sync.md), [ds4form r1 fixes](reports/cross-review/ds4form-r1-fixes.md), [r2 fixes](reports/cross-review/ds4form-r2-fixes.md), [W4A16 reload PR body](reports/pr-bodies/vllm-split/2-modelopt-w4a16-reload.md), [FlashInfer kv-scale reload bug](reports/vllm-flashinfer-kvscale-reload-bug.md) |
| Checkpoint and host memory | [Checkpoint host OOM](reports/ckpt-host-oom.md), [Host memory growth](reports/host-mem-growth.md), [BI=0 baseline checkpoint crash](reports/baseline-ckpt-crash.md) |
| Sampling seed | [seedfix A/B and AIME baseline](reports/seedfix-ab-and-aime-baseline.md), [SGLang DP seed collision](reports/sglang-dp-seed-collision.md), [vLLM DP sampler seed PR](reports/vllm-dp-sampler-seed-pr.md) |
| Comparison with DS4 | [Practice differences](reports/ds4-practice-differences.md), [Structure migration](reports/ds4-structure-migration.md) |

Raw scripts, results-data, cross-review records, PR/issue text drafts, learning materials and plan.md were deleted on 2026-10-07; they are not included in this public copy. Paths written in backticks in the reports such as `scripts/`, `results-data/`, `results/*` refer to these files; absolute cluster paths, job numbers and node names referred to locations on the cluster at the time and are not guaranteed to still exist.

## Historical report SHA256

Hashes of the workspace reports at 2026-10 compilation time, used to verify that the reports above match the originals.
The [paired acceptance numbers summary](history/nemotron-direct-fa4-acceptance-20260929.json) keeps only the four full-problem quality cells, three-round fixed-length throughput and the diagnostic
profile's DP0 aggregates; shared expert kernel attribution and the final image are still not proven by that summary.

| Report | SHA256 at compilation time |
| --- | --- |
| Workspace report `long-run-formal.md` | `a77d4214036722e8e967e729f63f3ceaa2a36a3495fbf2a2be03b0858ce86dcb` |
| Workspace report `recipe-bi0.md` | `16b8d2046620a9906951e0dce6fd9c21c54b0cb09a3a4c556d859dda6002e710` |
| Workspace report `baseline-ckpt-crash.md` | `f8a7c712d55b3a45c79a3bc8d4d591c689bd8afd9b96e332a5d06e13028c862d` |
| Workspace report `host-mem-growth.md` | `8960f970400723bf4264ad2cfe89329f3b596660c31d04ba0c4927676f49be84` (includes §7 fix results; earlier versions `590d070a…`, `ed8a5dfc…`, `b08fd491…`) |
| Workspace result `final/README.md` | `53d66036e9dcbaf6f2f9567b86582d7c01f3ef3b381ceeb2acdd8d5f1d20f616` |
| Workspace report `b1-activation-fakequant.md` | `a14207fc94972221e419a6d54b8de2316ac11ce232103449bf0216a00d36788b` |
| Workspace report `baseline-r3-feasibility.md` | `92d681ba04760d5b99b13c81fe7ade17abf071fbc6e20a7ca022643114cad68e` |
| Workspace report `bi0-bf16-collapse.md` | `e44a6ea792900407e0ddfa6132b21be8e91d0f9f826b764e11dfc294613a9a32` |
| Workspace report `prefix-cache-perf.md` | `502c208798931d3f9a6535760aa0b57123022772ebc2b4752fa0478dda2de169` |
| Workspace report `vllm-dp-sampler-seed-pr.md` | `dc15cecd7db5caa41fae411db32116c4ccc659bf1c6d960e2a2220e86d220b40` |
| Workspace report `vllm-flashinfer-kvscale-reload-bug.md` | `9be983c7e33870ecbde0ac42744ab15270e4b2ec49893b415339901769d8ea85` |
