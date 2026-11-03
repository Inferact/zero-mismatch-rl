# Classification and order of fixes

After fixing the side ([01](01-isolate-the-side.md)) and bisecting ([02](02-bisect.md)), classify the difference first, then decide what to change; precision comes last. When adapter transfer/application is involved, also read the [LoRA guide](../lora/README.md). Model-specific probes, runtime patches and measurements stay in the model working directory, not in this document.

## Classification table

| Class | Evidence | Action |
|---|---|---|
| Correctness error | stale/wrong adapter, missing weights, wrong mask/position/layout, wrong scale/block, wrong quantization range, operator semantics not matching the contract | Must be fixed and the original failing path re-verified; for diagnosis tasks, deliver the reproducer and evidence first, do not automatically expand modification permissions |
| Normal floating-point difference | Explainable by same-version repeatability/reduction/kernel noise, no contract error found | Report the noise floor and task impact; bit-exact is not the only pass criterion |
| Error from correct quantization | Mapping and quantization rules are correct; same-input comparison locates the divergence to the quantization path | Do precision ablation in the priority order below |
| Not located | Insufficient evidence or several variables confounded | Keep it as unknown, continue with minimal probes, do not declare "it's just quantization" |

Precision ablation order: supported and affordable high-precision rollout → QAT matching the serving rules → true low-precision training (most backend requirements, consider last). For a natively quantized model, do not assume BF16 is supported just by changing dtype. Fake quant must match serving's weight/activation rules, but does not automatically reproduce its GEMM. QAT is not a workaround for a stale adapter or a wrong scale. For the QAT contract and route selection, see the [precision guide](../precision-quant-qat/02-qat-and-ste.md#1-choose-the-precision-route-first).

Keep a single checkpoint source; if needed, open a separate high-precision adapter branch.

### Noise or defect: read the pattern, then the size

The size of a difference does not classify it; its pattern does. Rounding noise follows the rounding: it changes when the batch size or
kernel changes, is spread over tokens and slots, and shrinks when the computation is made smoother. A defect follows structure: it is
the same on every repeat, appears only at a slot, a position, a segment or a token id, and is absent in a neighbouring configuration that
differs only structurally. Size is then read against two floors measured on the same model: the repeat floor (same input twice) and
the self-perturbation floor (the trainer with one rounding changed, e.g. 0.1% of one layer's inputs moved by one BF16 ulp).

- **Reduced-layer slices amplify.** On Model A's six-layer slice that one-ulp perturbation moved logprobs by 7.7–8.2 nats; with QAT on, the
  trainer's distance to its own perturbed self (0.0228) equalled its distance to vLLM (0.023). A slice can show that a path is
  structurally right; it cannot size a small difference. Measure batch effects and train-inference gaps on the full model: there Model A's
  vLLM batch-vs-alone difference (0.009–0.013 per token) matched its own repeat noise (0.007–0.010).
- **But a clean structural pattern is not chaos.** On the same slice with CUDA graphs, every prompt in batch slot 1 or later was off by
  3.1–13.8 nats while slot 0 was right. It was first explained away as slice chaos, and it was a defect: the sparse indexer had been
  recorded inside a breakable CUDA graph segment with the capture batch's metadata. Amplified rounding does not single out slots.
  (A slot-0-only pattern in eager mode on that slice, `[0, 8.37, 8.37, 8.37]` for four copies of one prompt, was never explained; the
  same probe on the full-model engine showed no slot dependence.)
- **Non-repeatable is not wrong.** Atomic split-K reductions (vLLM's LoRA shrink, Megatron's MoE combine) make repeats differ in the last
  bits; the value is correct. Record the repeat floor and keep the default kernel; do not adopt a deterministic variant as a fix at
  day-0; when one is worth upstreaming it goes behind the batch-invariant mode, not into the default path (a day-0 boundary). Sample that floor with enough repeats: in Model A's
  G1 only 2 of 6 repeats differed at all, the envelope came out too small, and two gate pairs stopped on noise. Judge against the noise
  with a statistic the noise does not saturate: the same adapter scored twice differed by up to 7.2 nats on a single token, so a rule on
  max |diff| failed every real update while the mean separated them clearly; compare means and percentiles, keep the max as a diagnostic.
- **For MoE models, rule out routing first.** When the gap stays put across precision changes, compare with routing replay (R3) on:
  a top-k flip changes a token's whole expert mix, so a gap made of flips looks like a precision floor that nothing moves. Another MoE model's θ0 went
  from MAE 0.037 to 0.019 with R3; Model A's full-model trainer against vLLM went from 0.027 (native routing) to 0.018 (R3). Then report the native-routing flip
  rate separately, because R3 hides it ([rollout engine §4](../rollout-engine.md#4-r3-routed-experts-replay-functional-constraints)).

## Order of fixes

Input mismatch → weight mismatch → discrete path mismatch (top-k, index) → kernel numerics → precision contract. Do not move to the next item until the previous one is ruled out.

Acceptance: after the fix, the same failing probe passes, the original path is not broken, and error growth on fixed samples is explained/controlled; only then look at training-process metrics and fixed-task evaluation. Set tolerances from the model and baseline; do not copy another model's 0.002. Statistics on a fixed old trajectory are diagnostics, not a KL estimate of the current policy.

This procedure does not authorize stopping jobs, switching dtype, adding GPUs or changing the production recipe; when the next diagnostic gate needs new permissions/resources, report the smallest next probe and its cost.

## Deliverables

Follow the [06 save and compare protocol](06-artifacts.md): manifest, bounded dump, comparison table, semantic boundary of the earliest divergence, classification and confidence, one next step; state which paths were actually measured. Do not build a separate framework, and do not set cross-model universal thresholds.

## Two cases this week: looked like precision, were something else

Both occurred during the Nemotron-H 2026-10 delivery; in both, the μ/π difference was first treated as a numerical problem.

### 1. FlashInfer caches the KV scale and does not refresh it after reload (update chain)

Symptom: start with `load_format=dummy`, layerwise reload writes the real checkpoint, then rollout is all garbage. Trigger: checkpoint carries static FP8 KV scales, `kv_cache_dtype=fp8_e4m3`, attention backend is FlashInfer; verl rollout by default does dummy init and then the first refit, so RL hits it right at startup.

Root cause: the FlashInfer impl lazily computes and caches `bmm1_scale`/`bmm2_scale` (including `_k_scale_float`, `_v_scale_float`) on the first forward; reload updates the scale on the layer, but the cache on the impl is never invalidated, so K/V are dequantized with the dummy-era scale (1.0).

Classification: correctness error (stale derived value), part of the [weight update](../weight-update/README.md) chain, not kernel precision. Distinguishing clues: appears only after reload, always reproduces with dummy start, coexists with byte-identical weights. Same class: NVFP4 reload stale derived values (vllm#57954). Details in the [case report](../../../cases/nemotron/reports/vllm-flashinfer-kvscale-reload-bug.md).

### 2. determinism only takes effect on the main thread (switch scope)

Symptom: at training step 0 the backward is occasionally not bitwise identical (node-dependent, 5 out of 8 on one node); from step 1 on every step is stable and consistent; forward, selected logprob and loss are all bitwise identical; the first differing tensor is `grad_x` of the `norm_f` VJP.

Root cause: in PyTorch 2.13, Inductor config overrides are thread-local. `torch.use_deterministic_algorithms(True)` sets `inductor.config.deterministic` to True on the main thread, while autograd's CUDA device thread is still False; the `torch.compile`d norm VJP is first compiled in that thread during backward, so `dynamic_scale_rblock` is turned on, adding an extra `R0_BLOCK=512` candidate that is selected by first-launch timing, with an in-row reduction order different from 1024. At the end of dynamo compilation, `use_deterministic_algorithms(prior)` is called, and only after that is the thread True, so all later compilations have a single candidate.

Classification: correctness error (switch scope), not reduction noise. The fix is one line: use `torch.compile(..., options={"deterministic": True})` for the three VJPs, independent of which thread the compilation happens in; after the fix, the easily-reproducing node showed 0 occurrences in 12 runs. Distinguishing clues: affects only the first step, only backward, only the shape compiled first; the train-inference consistency contract (forward logprob) is not affected. Details in the [mbinv first-step report](../../../cases/nemotron/reports/mbinv-first-step.md); the kernel-side entry is in the [BI kernel appendix](../training-model-support/appendix-bi-kernels.md).
