# Nemotron: cross-stack arithmetic, state and effective updates

This case contains lessons from the public Mamba2/Nemotron-H implementation, plus 7-layer localization and 52-layer training experiments on the internal Nemotron 3.5 Lightning 30B-A3B. Passing results from different models/configurations cannot be merged into a single support claim. See also the [2026-09 NVFP4 DAPO failure case](dapo-20260927.md): that full five-step run completed but train-inference logprobs were misaligned from θ0 through θ4, and the historical passing evidence here cannot vouch for it.

## SSD/SSU and state

The two sections below are mechanism conclusions accumulated on the upstream Mamba2 implementation before the 2026-10 delivery; the delivered BI implementation (exact replay, PC state copy, gate/norm order) is built on them; the corresponding delivered implementation is in "Key corrections from the 7-layer localization" and "2026-10 Lightning NVFP4 true on-policy delivery" below.

Mamba2 prefill uses chunked SSD and decode uses recurrent SSU; mathematical equivalence does not guarantee identical rounding. Replaying a partial chunk from the FP32 boundary state through the same SSD path can match prefill; it needs the same chunk grid and a state of sufficient precision. Validate aligned/unaligned splits, cross-chunk, and preemption/recompute separately.

Public sources: [vLLM #55524](https://github.com/vllm-project/vllm/issues/55524), [#55627](https://github.com/vllm-project/vllm/pull/55627). The RFC contains historical design updates and retracted conclusions; read it with dates/versions in mind, and do not copy old configuration restrictions as the current API.

## Autotune and gate

These two items determined that, at delivery, the reference implementation had to be pinned to the BI configuration and the training-side SSD had to replicate the gate/norm/cast order.

Some chunk256/multi-head shapes could not reveal tile differences; later chunk64/8 heads found that the cumsum tile changes FP32 prefix sums. Cover cross-process and actual shapes, rather than inferring from a few passes that autotune does not affect numerics. [#55895](https://github.com/vllm-project/vllm/pull/55895)

Applying the gate inside the FP32 scan and then RMSNorm has a different rounding path from rounding the scan first and then applying the gate in gated RMSNorm. A shared SSD kernel still needs to match the order of gate, D residual, norm and cast. [#56244](https://github.com/vllm-project/vllm/pull/56244)

## Key corrections from the 7-layer localization

The internal run's training-forward progress report and alignment code audit (not included) record the process:

1. Locate the first difference in norm/projection layer by layer. Different reduction blocks were observed, but measuring both blocks with the same captured kernel showed both match, refuting that single-factor attribution.
2. The direct and compiled paths of the same gated RMS kernel differ; after calling the same existing kernel through an opaque custom-op boundary, most of the difference disappeared.
3. A remaining logprob difference of about 4.768e-7 still does not count as a bitwise pass. The two scorers agree on the same FP32 logits, proving that "only swapping the scorer" is not enough to fix it.
4. The actual HF forward unconditionally applies `.float()` after lm_head. Keeping the real BF16 head output and sharing scoring made it pass; this is not truncating a real FP32 projection to BF16 to manufacture equality.

This phase covered four prompts, 544 independent response tokens, and targets at different positions in mixed batches; it is not a full dataset, native VERL multi-step training, or proof of long-term stability. The shared norm changed the stock inference arithmetic baseline and needs an independent quality regression.

## Backward, CP and updates

The shared inference forward and the native VJP are validated separately. One CP gradient difference came from the test using an FP32 cotangent while the other side had already rounded to BF16; after matching representable inputs, the op comparison passed. Root-only FSDP OOM'd on temporary FP32 reduction memory; this was solved by per-layer grouping, and gradient precision should not be quietly swapped.

Full 52-layer CP2 standalone update/reload validation and native Megatron/VERL integration are different phases. Later the native two steps had effective gradients and weight fingerprint changes; a still later four-step record had 139,260 bitwise-identical raw-logprob tokens, but shutdown hit a DataLoader error, and a separate long run failed at a later step. Four-step evidence cannot be upgraded to 50-step stability.

Internally this corresponds to the run's full-52-layer CP2 completion audit, its task index and the integration PR body drafted for the 2026-09-18 image release (all not included). Public integration: a PR on a community Megatron-LM fork.

## Performance lessons

Row-gated emit only emits the outputs currently needed and updates state at chunk boundaries, which reduces exact-replay work and restores Graph, but must keep the scan's gate/rounding order. An internal experimental fast path measured 0.1874 ms/token versus BI0 at 0.1726, still about 8.57% slower, so it was not kept as a default path that successfully sped things up.

When migrating, keep the successful methods and the refuted attributions: fixed rollout, full semantic boundaries, identical op inputs, the real head dtype, validation after update. Do not hard-code any particular tile, norm replacement or attention backend as the configuration for all Nemotron models.

## 2026-10 Lightning NVFP4 true on-policy delivery

Nemotron-H Lightning 30B-A3B NVFP4 (mixed precision: experts/shared/lm_head
W4A16, Mamba in/out_proj FP8 static activations, FP8 KV), single node 4×GB200, actor
EP4 + FSDP2 (PP4 dist_opt as fallback), rollout vLLM DP4/EP4 CuTe-DSL W4A16.

- **Results.** 100 steps at the default workload, every step `rollout_probs_diff` max = mean = 0;
  still 0 after kill→resume and after new code reads an old checkpoint. Closed-loop θ0→θ1→θ2 probe: each version
  about 7.80–8.35 million old-logprob tokens with 0 mismatch in the original dtype, and 16 syncs with 291,472 tensors byte-identical.
  8K/1K 5-pair paired bench median loss at c=1/8/32/64 = 8.5/8.6/11.9/1.9% (budget 20%).
  GSM8K: on the p6 pre-candidate image (an internal vLLM commit) BI0 84.15% vs BI1 83.78%
  (McNemar p=0.70); on the final image, 40-shot first 500 problems BI1 426/500.
- **Delivery.** An internal vLLM fork PR (later also split into six smaller PRs),
  an internal Megatron-LM fork PR, an internal verl fork PR, and one unified image. Side fix already merged
  upstream: vllm-project/vllm#59788 (DP sampling seed).
- **Conventional baseline** uses the same recipe and same image as true on-policy, see
  [BI switch recipe](../../skill/guides/recipe-evaluation/bi-switch-recipe.md); quantization rules, refit and QAT data are in
  [Precision, quantization and QAT](../../skill/guides/precision-quant-qat/README.md); 100 GB checkpoints and host
  memory are in [Checkpoint, resume and host memory](../../skill/guides/checkpoint-resume-memory/README.md).

### Necessary differences from the DS4 approach

- The training process directly reuses vLLM layer modules (FusedMoE, ModelOpt linear), which need
  `get_tp/ep_group`, so the training process must build vLLM process groups; DS4's self-written modules do not need this.
- `param_offload=True` moves parameters, changing the quantized master's `data_ptr`, so the pre-export
  check raises; like DS4, only optimizer offload is enabled.
- The FP32 quantized master resident on GPU squeezes KV capacity, so rollout `gpu_memory_utilization`
  must be set per training mode; the measured values for the three modes are in
  [checkpoint 02 §5](../../skill/guides/checkpoint-resume-memory/02-host-memory.md#5-gpu-memory-trainingrollout-colocation-is-time-sharing-compute-per-phase).

### Step 7: how to make the gradient comparison discriminating

- Freeze the criteria (tolerance, denominator, threshold) before running and record the sha; the denominator is max(B16 reference floor,
  the implementation under test's own noise floor).
- First do a planted-error discrimination check: gradients ×1.25 and ×1.05 must both be judged FAIL.
- Accept multiple training snapshots (steps 0/50/150/250) separately; passing at step 0 does not mean later steps pass.
- On failure, bisect by injecting the reference gradient block by block. Nemotron found that attention backward's
  `D = rowsum(dO⊙O)` used the visible O of the FP8-P forward, inconsistent with the P recomputed in backward
  (deviation about 0.03 → 0.006–0.009 after the fix). The fix is to recompute O and LSE in BF16 in backward;
  mainstream FP8 attention (TE fp8_dpa, SageBwd) uses the visible O directly, so this is not the default
  practice, and a human decided to include it with full knowledge.
- Cross-process bitwise reproducibility of backward is not a requirement; train-inference consistency only constrains the forward.

## Mamba prefix caching (align mode)

- **Copy all state under BI.** BI's Mamba2 declares 5 states (conv + temporal×4);
  when the align-mode copy function copies only 2, every prompt longer than one block is wrong after a hit.
  Bind the number of copy functions to the BI state-shape list under the same switch, and add a unit test asserting
  the two are equal.
- **You pay even without hits.** Turning PC on as-is: BI0 throughput −4 to −20%, TPOT +13 to 21%, c=1
  TTFT roughly doubles, reproduced likewise with a BF16 checkpoint and the default MoE backend
  (vllm-project/vllm#60008). Three sources: eager block-table operations per KV group
  (eager launches per step 9→37), the align kernel outside the graph, and prefill being split into two steps at block
  boundaries.
- **Fix combination.** Decode side: precompute multi-group aligned indices once, skip
  align postprocess when there is no spec decode, and skip preprocess/precopy when the host side determines this step has no cross-block,
  bringing launches back to 9. Prefill side: export block-boundary checkpoints from inside the scan
  (the #57329 approach), with a separate export on the BI path, otherwise BI1 silently goes wrong; chunks truncated by the token budget
  also export the last checkpoint (under DP+EP, any rank taking one extra prefill step
  is a global decode stall). Result: BI1 roughly on par with PC off.
- **Two individually correct fixes combined can silently go wrong.** After directly merging #58851 (multi-group precompute) and
  #57329 (prefill checkpoint), 28-shot GSM8K dropped from 432 to 4/500:
  later groups reused the first group's checkpoint target block; after adding per-group re-gather
  it was 438.
- **Conv window misalignment with spec decode on.** The checkpoint is written tail-aligned by `conv_state.size(-1)`,
  while the kernel reads from `[0, width-1)`; when the cache is widened for spec decode,
  the window is offset after a hit. Write only `[0, conv_kernel_size-1)`.
- **Gates** (run for every candidate): bitwise probes against the PC-off gold standard (hit positions not on chunk
  boundaries, every DP rank); recomputed hits after forced preemption; probes of the budget-truncation path, confirming in the trace
  that truncation actually happened; GSM8K per-problem identical to PC off; a hook that checks GPU state index step by step
  against the host-side prediction; rerun gate3 (the inference-side BI/PD/chunk full gate, corresponding to C3, see §2 of the [final-p6 inference results](results/final-p6/inference/README.md#2-gate3-full-run-inference-bi--pd--chunk)). BI0 itself is nondeterministic (two runs on the same image had only
  0–3/19 sequences bitwise identical), so BI0 is judged by GSM8K and token agreement rate intervals.
- Under BI, Mamba2 rejects the KV connector, so real PD disaggregation cannot be tested; state explicitly that it is not covered.

## Duplicate in-group samples in DP rollout

This is a general vLLM/GRPO issue; the check method is in
[DP seed collision](../../skill/validation.md#dp-seed-collision-identical-in-group-sibling-samples). On Nemotron,
before the fix up to 43% of sibling pairs were fully identical; the fix has been merged as vllm-project/vllm#59788.

## Backward: independent validation protocol and per-op status

The general training-side criteria and steps are in [training side 03 reuse backward](../../skill/guides/training-model-support/03-reuse-backward.md); this is the Nemotron instance.

### Independent validation protocol (2026-09-30)

- The reference builds its graph independently with ATen autograd, without calling the backward under test or the intermediates of the production compact dispatch; candidate and reference keep the same STE, BF16 edges and reduction contract.
- Tolerances (`atol=1e-5`, `rtol=1e-4`, relative L2 ≤ 1e-5) are fixed before looking at results and not loosened afterward.
- Separately check dX, dUp, dDown, dRoute, that unused-expert gradients are strictly zero, and a one-step fixed SGD update from identical initial weights.
- Report "bitwise identical" and "within tolerance" separately: for MoE M=17 all four gradient types are bitwise identical; for M=128 dX/dRoute are bitwise, while dUp/dDown only pass within tolerance; do not conflate them.
- State that synthetic inputs and upstream gradients are not activations captured from real training; weights come from the real checkpoint.
- Test long sequences separately: FA4 at two lengths 7000/9216, three kinds of dY random/zero/sparse, sparse gradients crossing the 6768-token KV page boundary, and a packing comparison with two groups of 17 tokens prepended.
- The production code's source and kernel SHA must match the training manifest, otherwise the test refuses to run (to avoid testing a same-named diagnostic copy).
- When the base is frozen and only LoRA is trained, base modules must still propagate input gradients.
- Do not recompute activations: use the intermediates saved by the forward directly (DS4's gate_up, LSE).

### Per-op backward status (2026-10-03)

| Op | Forward (vLLM) | Case | Backward |
|---|---|---|---|
| Mamba SSD scan | `mamba_chunk_scan_combined_varlen` | A | mamba_ssm `_mamba_chunk_scan_combined_bwd` (previously 160 lines of PyTorch copied from HF) |
| Attention (FP8 KV) | FA4 + FP8 Q/K/V | B→A | Dequantize Q/K/V and recompute BF16 O/LSE, FA4 deterministic bwd (previously per-sequence SDPA) |
| conv1d | `causal_conv1d_fn` | A | causal_conv1d bwd + fixed-order FP32 dW |
| Mamba in/out_proj (FP8), shared expert (NVFP4) | FlashInfer / NVFP4 GEMM | B | Math same as DS4 (TE GEMM, master weights); the implementation is the model directory's private `native_linear_vjp`, calling the TE API itself, without reusing `primitive/utils/moe.py:_te_general_gemm` |
| Routed experts (NVFP4) | CuTe-DSL / Humming MoE | B+C | `routed_vjp` glue; grouped GEMM math same as DS4, but `_te_grouped_gemm` is a private rewrite that does not reuse `te.GroupedLinear` |
| RMSNorm / gated RMSNorm, router scoring | BI kernel / grouped top-k | A | PyTorch formula + autograd |

"Same as DS4" means the backward math and source (TE GEMM) are the same; "reuse" does not hold; the rewrite and placement issues are in the [mLite reuse audit](mlite-reuse-audit.md) (the `dense.py` and `grouped.py` rows).

## Numbers for training-side GPU memory, length boundaries and optimizer acceptance

The criteria are in [training side 04](../../skill/guides/training-model-support/04-layout-and-length.md); only Nemotron's numbers are here.

- **Layout-dependent GPU memory**: with EP4 (PP1) each rank carries activations for all 52 layers, transiently about 98 GiB at 8192 tokens, doubling at 16384 tokens; if the per-route state saved by EP backward also stores a copy for slots not belonging to this rank, it is about 68 GB at 16384 tokens and 23 MoE layers, and storing only this rank's routes saves about 47 GB. Source: §5 (Phase 2: EP4) of [ds4-structure-migration](reports/ds4-structure-migration.md).
- **FA4 fixed split**: 32 splits × 640 keys, `MAX_SEQ_LEN` 16384; replicates vLLM's 6768-token KV page; long-sequence validation uses two lengths 7000/9216, three kinds of dY random/zero/sparse, sparse dY crossing the page boundary, and a packing comparison with two groups of 17 tokens prepended. Source: the protocol in the previous section; [BI kernel appendix](../../skill/guides/training-model-support/appendix-bi-kernels.md).
- **optauth v2**: 20 PASS / 1 FAIL (same for EP4 and PP4, criteria frozen before running). The failure is a gap in how the criteria were modeled, not an implementation error: A1's check does not apply when there are no replicated parameters (occurs in the FSDP2 final version); A6: Adam's first step applies a 6–20 ulp equal-magnitude update to `norm.weight`, the rounding errors systematically share a sign, and the γ criterion breaks down; these coordinates all satisfy the per-coordinate error bound (A4), so it is accepted as a known gap. A real bug found only after v1 was used: the `wd_mult=0` group actually had wd=0.1 (fixed). Sources: the supplementary gates (p5) section of [skill-admission](reports/skill-admission.md), finding 1 (weight decay applied to `wd_mult=0` parameters) in the [MCore gradient/optimizer comparison](reports/mcore-grad-optimizer-compare.md).
