# LoRA: route selection, training, sync and recovery

LoRA is a mandatory test path in the Day-0 flow. This guide has two parts: inference-side adaptation and sync route selection (Part 1), and the minimal FSDP + LoRA training loop (Part 2). A model's target coverage, actual launch arguments and evidence go in `cases/<model>/`, not here.

## Find by symptom

| Symptom | Go to |
|---|---|
| Unsure whether to go adapter-only or merge first | [Close the loop on one route first](#2-close-the-loop-on-one-route-first) |
| Log shows `matched ... could not be wrapped ... ignored` | [Minimal deliverable](#4-minimal-deliverable): cross-check against the trainable list; an ignored module that should be trained is a failure |
| Adapter sent successfully but output unchanged / large train-inference mismatch | [Three gates](#1-three-gates-can-forward-is-not-full-support): prove separately that it was trained, transferred correctly, and actually used; Model B shared-outer launch-argument case |
| Adapter unchanged after the first step / grad_norm is 0 | [Adapter contract table](#2-one-adapter-contract-table-is-enough): zero-B init and identical rewards within a group are both legitimate causes |
| Need to prove the transfer is correct | [Reusable Miles checks](#3-checks-actually-reusable-from-miles): compare checksums at a consistent logical-layout boundary; this proves transfer, not that forward reads the values |
| Medium-magnitude adapters mismatch across the two sides, while zero and very small adapters match | [§4](#4-minimal-implementation-principles-on-the-fsdp-side): PEFT merges expert LoRA; [Where the delta is rounded](#where-the-delta-is-rounded) |
| vLLM scores the same request twice with different results (nonzero adapter) | [Three gates](#1-three-gates-can-forward-is-not-full-support) "actually used": split-K atomic add in shrink |
| A hook captures the adapter vLLM uses as all zeros | [Three gates](#1-three-gates-can-forward-is-not-full-support) "actually used": an all-zero placeholder is installed first on load |
| Serving does not support LoRA on some module; expert LoRA cannot attach to the production quantized kernel; rollout is noticeably slower with LoRA on (even when only attention is trained) | [Serving-side LoRA must be implemented yourself](#part-1-1a-serving-lora) |
| With LoRA on, vLLM weight reload reports a shape error or silently degrades; adapter reread from disk every batch; Rust frontend fails to load adapters; NVFP4 expert LoRA rejected | [Known vLLM-side issues](vllm-serving-issues.md#find-by-symptom) |
| Same adapter gives different outputs on TP1 and TP4; MoE LoRA init fails only on the full model | TP and multimodal-wrapper rows of [List the actual training scope first](#part-1-1-list-the-actual-training-scope-first) |
| On a reduced-layer model the trainer-vs-rollout diff is as large as the adapter's effect | [Minimal verification order](#3-minimal-verification-order): scale up LoRA-B and add a counterfactual adapter |
| Can training resume after recovery? | [Save and resume](#5-save-and-resume-must-also-be-measured) |
| What next after it runs end to end | [What to add first after it runs](#part-2-6-what-to-add-first-after-it-runs) |

## Key rules

1. A capability claim is not implementation evidence: `SupportsLoRA` does not mean every layer in the allowlist has a wrapper; check item by item from training key family → serving module.
2. Of the two routes (`merge=false` sends the adapter / `merge=true` sends the full model), close the loop on one first; do not mix partial reload, offline checkpoint writes, or a standalone publisher into normal sync.
3. The verification order is fixed: zero adapter → nonzero adapter → swap revision and swap back → production sleep/wake and CUDA Graph → then add MTP; it must be measured under Graph, and recapture after reload does not count as a fix.
4. Trained, transferred correctly, and actually used are three gates, each with independent evidence; equal checksums do not prove forward read the new values. When the engine is not bitwise repeatable (default split-K LoRA kernels), "the synced adapter scores like the same adapter cold-loaded" is judged against the measured repeat envelope E, and the check is valid only if planted controls fall outside it: Model A's G1 recheck required a stale adapter, LoRA-B ×1.1 and one zeroed layer each to differ by more than 1.5×E.
5. Inference-side launch arguments (e.g. `enable_moe_shared_loras`) must match the training factor layout; verify with a nonzero adapter, because zero B masks layout errors.
6. New models start with packing off and turn it on only after the [eight-class check](../training-model-support/04-layout-and-length.md#packing-and-dynamic-batching); reduced-layer models keep the attention period and special layers; passing on a reduced-layer model does not prove full-model capacity.
7. Save/resume testing (reduced-layer stages of G3, G5): after resume, score fixed tokens first, then do the next update/sync. Resumes of full-model training jobs (including the short job before formal submission) do not insert fixed-token scoring (acceptance item 6 of the day-0 support workflow's train-inference consistency pass criteria).
8. Serving-side LoRA must be implemented yourself at day-0: what to train follows the recipe, it computes the same function as training, and it is implemented on the production base path (kernel and activation quantization unchanged); falling back to another kernel is a deviation, see [§1](#part-1-1a-serving-lora).
9. Training-side LoRA computes base and delta separately and does not merge the delta into weights. First establish at which step each side rounds the delta: serving engines all round base to BF16 first and then add the delta; which form the training side uses is decided per model by the owner, and if undecided it follows serving. Record the chosen form and the resulting known differences from serving in the recipe deviation table; see [Where the delta is rounded](#where-the-delta-is-rounded).

<a id="part-1"></a>
## LoRA inference-side adaptation and sync route selection

> Source audit date: 2026-09-05; the version conclusions below do not represent re-verification against latest upstream.

Use this document to choose a route before preparing training, and record stage acceptance back in the
Day-0 overall flow. The principle is to compare actual development cost,
and prefer the native adapter path when it is feasible; a model claiming SupportsLoRA alone does not count as a pass.

<a id="part-1-1-list-the-actual-training-scope-first"></a>
## 1. List the actual training scope first

Freeze the PEFT target allowlist, rank, alpha, dtype and base revision; from the real
trainable parameters, export a key/shape list of training name → HF export name → serving module,
noting scaling and transpose/fusion direction. Map each item to a serving module:

| Check item | Behavior to confirm |
|---|---|
| Model registration and module wrapping | The actual class supports LoRA; every layer in the allowlist has a corresponding wrapper, none silently ignored |
| packed modules / namespace | QKV, gate/up mapping and name conversions such as `.base_layer.` are correct |
| MoE / special layers | routed/shared experts, convolution or custom layers are verified item by item per the real training scope; untrained modules need no extension |
| Multimodal wrapper | When the full checkpoint is `*ForConditionalGeneration`, the text config may lack `architectures` and MoE LoRA init fails; build the reduced-layer model with the same top-level class as the full checkpoint |
| Quantized base | The LoRA wrapper does not break scale forwarding or the base quant method |
| TP | Adapter slicing, distribution and aggregation match the actual production TP; also test TP=1, since LoRA-B slicing of fused projections may only be exercised at TP>1 |
| MTP / draft | Clearly state which adapters apply to target/draft; excluded keys have a justification; shared weights are consistent |

<a id="part-1-1a-serving-lora"></a>
**Serving-side LoRA must be implemented yourself.** At day-0 a new model has no LoRA implementation on the serving side, so "does the existing kernel support it" is not a precondition. Implement by three rules:

1. **What to train is set by the recipe.** Attention projections (including fused qkv, o-gate), dense MLP, routed/shared experts and model-specific projections — whatever the recipe trains lands on both sides; if a wrapper, mapping or kernel is missing, add it, do not shrink the scope (shrinking the scope changes the algorithm).
2. **Compute the same function as training.** Add the delta where the model formula puts it (example: when expert output goes through a norm before being multiplied by the routing weight, the W2 delta is added before the norm and is not multiplied by the routing weight). The serving side follows the engine's existing form and adds onto the base output already rounded to BF16; the training-side rounding point follows the decision in [Where the delta is rounded](#where-the-delta-is-rounded).
3. **Do not change the base's production path.** Whatever kernel the base uses in production, and whether activations are quantized, stays the same after attaching LoRA; if the kernel does not expose intermediates LoRA needs, extend the kernel. Model D (2026-10, NVFP4 W4A4) uses the LoRA interface of FlashInfer routed MoE (introduced in #3153, NVFP4 supported by #3987): the W13 delta is fused into FC1 before SwiGLU via `gemm1_lora_delta`; the W2 delta is computed outside the kernel from the FC1 activation output the kernel returns (`gemm1_activation_output`) and added to the post-finalize output. This path only holds when FC1 activations use per-token scale: under static scale the kernel returns packed FP4 activations and the W2 delta cannot be computed (details and upstream status in [Known vLLM-side issues](vllm-serving-issues.md#nvfp4-expert-lora-supports-only-per-token-activation-scale)).

Falling back to another path (W4A16, BF16) when rule 3 cannot be met is a deviation: rollout is no longer the production model, activation numerics change (the activation rows of the [precision 04](../precision-quant-qat/04-inference-quant-inventory.md) inventory change with it), and throughput loss counts against G0 walltime; the adapter is trained against the degraded policy, and deploying it back on the production kernel adds another layer of inconsistency. Model N (2026-10) dropped MoE from W4A4 to Marlin W4A16 when expert LoRA was enabled, so its train-inference diff is small ([QAT §2](../precision-quant-qat/02-qat-and-ste.md#2-what-the-four-kinds-of-paths-are)); that is a side effect of the deviation, not a reason to choose it. vLLM as of 2026-10 also violates rule 3 in one place: enabling LoRA at all switches MoE to the LoRA kernel even if the targets exclude experts, which lowers throughput, and BF16 MoE errors out outright (fix [vllm#61165](https://github.com/vllm-project/vllm/pull/61165) not merged), so training only attention does not avoid this cost.

A capability claim is not implementation evidence. Check the source of the image in use and the actual diff of the corresponding PRs; for specific module scope and
historical fix leads see the per-model case notes.

## 2. Close the loop on one route first

| Route | What is sent each time | New work |
|---|---|---|
| `merge=false` | Full adapter of the current revision | Add needed LoRA wrappers/key mapping, verify adapter replacement |
| `merge=true` | Full model weights after merge | gather/merge, quantization format conversion, full-model loading and memory budget |

Estimate minimal code change, verification scale, init/sync peak and per-step time, then record the rationale for the choice.
Do not add partial model reload, per-step offline checkpoint writes, or a standalone publisher to the normal
VERL adapter sync flow. A publisher is only for upfront standalone serving A–A verification.

Format and owner for the merge route are unified in the
[weight conversion guide](../precision-quant-qat/01-conversion-chain.md#4-who-converts-conversion-owner). If adapter-only holds, the base
keeps using the existing low-precision checkpoint and does not need re-quantization each step.

## 3. Minimal verification order

Verify interfaces on a model with reduced layers/experts first, then recheck capacity and quality on the full model:

1. Cold-start the real base; outputs with a zero adapter and with no adapter are within repeated-run noise.
2. Load a deterministic nonzero adapter and compare trainer/rollout logprobs on the same tokens.
   Check the key/shape/dtype the adapter actually loaded; API success alone is not enough.
3. Swap to a second adapter revision, then back to the first; compare against the cold-start
   reference of the corresponding adapter to prove no stale adapter/cache remains.
4. Run the production sleep/wake and CUDA Graph paths; add MTP after target passes.
5. Verify training injection, optimizer, export and save/resume per the
   [training loop](#part-2) below.

When the production config enables CUDA Graph it must be measured; clearing graphs/recapturing after reload does not count as a
correctness fix. Run eager only for diagnosis.

On a reduced-layer model the trainer-vs-rollout noise floor may be as large as the effect to detect (skipped layers change the inputs of later layers, and near-tied
top-k flips). In that case, in step 2 multiply all LoRA-B by one factor (e.g. 10) and compare again: layout correctness is independent of magnitude, but shrink's split-K noise is also multiplied onto B,
so first set `split_k=1` per ["actually used"](#1-three-gates-can-forward-is-not-full-support); also prepare a
counterfactual adapter with one class of target modules removed (e.g. experts): the diff between the two sides should stay near noise, and the diff from the counterfactual should be clearly larger. Scaling up only verifies layout: once the delta is far above the BF16 last bit of the base, you cannot detect small deltas being rounded away; measure that with the magnitude sweep (B×0.1/1/10) in [§4](#4-minimal-implementation-principles-on-the-fsdp-side).

## 4. Minimal deliverable

One table in the model evidence is enough: training key family → serving module →
supported/needs fix/out of scope → test or PR. Attach the final target allowlist, the chosen sync mode,
verification results for two adapter revisions, and the repro commands.

When the warning "matched ... could not be wrapped ... ignored" appears, first cross-check against the trainable
list: if a module that really needs training is ignored, that is a failure; the training scope must not be silently shrunk.

<a id="part-2"></a>
## FSDP + LoRA Day-0: minimal correctness loop

> Source audit date: 2026-09-14; the version conclusions below do not represent re-verification against latest upstream.

Borrow verification boundaries from Miles, do not port the Megatron training implementation. This document ran no new experiments.
For route selection see the [LoRA serving guide](README.md), for quantization the
[quantization guide](../precision-quant-qat/README.md), for numerical localization the
[mismatch guide](../debugging-mismatch/README.md).

## 1. Three gates; "can forward" is not full support

| Gate | Minimal check | Where to look on failure |
|---|---|---|
| Trained | Real parameter names after target resolution, requires_grad, gradients, adapter changed after optimizer step; base frozen | Injection point, optimizer param groups, whether fused experts were missed |
| Transferred correctly | Exported adapter non-empty, logical names/shape/dtype/values consistent, all target replicas receive the same version | FSDP gather, HF mapping, expert sharding/ordering, receiver return values |
| Actually used | On the same fixed tokens, model computation responds after loading a changed adapter; compare against training-side recomputation | Request adapter ID, active slot, scaling, actual kernel, version switching |

For "actually used", two things on the vLLM side must be ruled out first. Neither shows up with zero or very small adapters; they only surface after some training, once the delta grows to medium magnitude, and the symptom looks like the two sides' implementation or sharding is wrong:

- **The same request scored twice gives different results.** LoRA shrink in `vllm/lora/ops/triton_ops/utils.py` defaults to `split_k=64`, merging the partial sums with `tl.atomic_add` in an order that differs each time; the noise is multiplied onto B — invisible at B≈1e-4, different every time at 1e-3 and above. With the Model D step-80 adapter: vLLM prefill-scoring the same sequence twice gave a mean per-token diff of 0.221, and 0.094 with `split_k=1`; this does not change the diff between decode and trainer, it only removes vLLM's own non-repeatability. The fix needs no code change: put `<GPU name>_SHRINK.json` (structure `config[max_loras][num_slices][m][k][n]`) under `VLLM_TUNED_CONFIG_FOLDER=<dir>` with every `split_k` set to 1. With `VLLM_BATCH_INVARIANT=1`, shrink uses deterministic split-K (fixed 8 segments since [vllm#59377](https://github.com/vllm-project/vllm/pull/59377), merged 2026-10-04; earlier versions set 1); the default non-BI path is still nondeterministic upstream. First have the same service score the same batch of tokens twice with a post-training-magnitude adapter; only once it is repeatable use vLLM's scores as the reference for any comparison.
- **The adapter a hook captures is all zeros.** When loading an adapter vLLM first installs an all-zero placeholder, then writes the real weights; a hook on the LoRA layer that captures "the A/B vLLM actually uses" gets zeros the first time, easily misread as sync not taking effect — take only the nonzero capture. Model D was checked this way: the A vLLM actually loaded matches the checkpoint slice by slice, and B equals checkpoint times `alpha/r = 2`, residual 1.6e-3, i.e. BF16 rounding.

First complete one round of train → sync → score → save/resume on a reduced-layer model, then scale to the full model; score here is fixed-token scoring, which goes into the training loop only in the reduced-layer stage; for the full model it is done only in separate processes before and after training.
Passing on reduced layers does not prove full-model capacity; keep the necessary attention period and special layers, and do not cut modules under test.
The FSDP2 route uses the forward of our own model, saving model-parallel adaptation, but does not automatically guarantee correct packing, position IDs,
frozen large-table residency or LoRA export. Reduced-layer tests must keep these boundaries, not just reduce parameter count.
New models default to packing off; consider enabling it only after an isolation test of effective-token outputs/gradients between standalone and packed samples
passes, and do not treat it as the same switch as dynamic batching.

## 2. One adapter contract table is enough

The contract follows the [LoRA inference guide](README.md#part-1-1-list-the-actual-training-scope-first);
the training side adds expert ID, whether factors are shared, and target_modules/target_parameters.
`alpha/r` scaling differs from rsLoRA; sharing an entire A/B pair is not the same as sharing only one factor.
"The model supports LoRA" does not mean GDN, MoE and special projections are all supported.

**Case: Model B shared-outer misconfiguration (2026-09-19).** The training adapter's per-expert
factors were independent, but the actual vLLM launch explicitly enabled `enable_moe_shared_loras=True` (default False),
which takes expert 0's gate/up LoRA-A and down LoRA-B and broadcasts them, silently dropping the other experts' values.
No exception was thrown; wrong weights caused the train-inference mismatch. With a fixed step-80 checkpoint and the same
137-token prompt and 2,043-token response, per-token probability MAE dropped from 0.08833 in shared mode
to 0.02798 in per-expert mode; after a full merge vLLM was 0.02288 and SGLang 0.02105.
Here MAE is `mean(abs(exp(logp_train)-exp(logp_serve)))` over fixed response tokens,
not logprob MAE, full-vocab error or test accuracy; a single sample does not represent a training batch, and the remaining error is not yet attributed.
The SGLang comparison used merged BF16 and did not verify the native adapter. Before going live, check the **actual launch arguments** against the
training factor layout; do not enable a shared layout on the inference side alone just because the same model's Miles recipe uses one.
Verify with an updated nonzero adapter: an initial zero B masks this kind of error.

With default zero-B init, a zero gradient for A at the first step is normal; not every adapter tensor is required to change every step.
Nonzero init or resume cannot reuse the "all B must be zero" init assertion.
If nothing changes, first check legitimate causes such as zero advantage, no effective tokens, or skipped optimizer;
do not unconditionally conclude a sync bug.
For example, when GRPO normalizes advantage per prompt group, even if rewards across the batch include both `-1` and
`1`, identical candidate rewards within each group still give `pg_loss=0`, `grad_norm=0`; first check in-group
rewards and the next batch, and do not misdiagnose it as an FSDP, LoRA or weight sync fault.

## 3. Checks actually reusable from Miles

Audited `miles@d2fc97ce581577e255e494801d7568747d5a10d7`:

- [`WeightUpdater.update_weights()`](https://github.com/radixark/miles/blob/d2fc97ce581577e255e494801d7568747d5a10d7/miles/backends/training_utils/weight_update/updater.py)
  records the sender's adapter checksum list under `--check-lora-weight-equal`,
  and `end_weight_update(expected_lora_checksums=...)` hands it to the engine for verification; only on success is the version updated and generation resumed.
  The list currently requires the iterator's `gather_pp`, so it cannot be summarized as a cost-free check for arbitrary topologies.
- [`session.py`](https://github.com/radixark/miles/blob/d2fc97ce581577e255e494801d7568747d5a10d7/miles/backends/training_utils/weight_update/session.py)
  checks the receiver's failure return; an RPC return is not treated as success.
- A published day-0 support blog (Model C, validation section)
  also describes initial-B, parameter-update and fixed-trajectory canary checks; these are a public experimental method,
  not a unified main switch found this time. The full cross-step canary scheduling source has not been located.

Checksum comparison must be done at a consistent logical layout/dtype boundary; sender HF adapter and receiver packed/sharded
storage cannot be compared as raw bytes. A checksum proves the transfer is correct, not that forward read those values.
Canary curves changing together are only supplementary evidence; numerical gaps must still be examined, and they cannot replace module comparison.

## 4. Minimal implementation principles on the FSDP side

- adapter-only gathers only the adapter and does not gather the huge frozen base to send the adapter.
  Keep rank participation consistent for necessary collectives, stage bounded per tensor/bucket, and do not let every rank copy the full adapter to CPU (stacked with the host backup of level-1 sleep this causes host OOM; per-expert adapters have tens of thousands of tensors, so copy whole buckets instead of allocating per tensor); do not introduce a partial-update protocol. For concrete handling of GB-scale adapters see [weight update 03](../weight-update/03-sync-robustness.md#other-constraints).
- meta init, FSDP wrapping and freezing must ensure the base is not moved whole to GPU before sharding; CPU-resident
  parameters such as PLE keep their intended residency, and observation and saving must not re-materialize them.
- The sync route follows the LoRA inference guide; for dtype, layout and quantization owner see the
  [conversion chain](../precision-quant-qat/01-conversion-chain.md); no separate flow is defined here.
- Initial deployment, new engine, sleep/wake and resume may need the base restored; distinguish them from steady-state adapter-only.
- MoE expert LoRA must not be left to PEFT's default `target_parameters` implementation: each forward it composes a full BF16 stack of `W + s·B·A`, taking an extra full stack of GPU memory, deltas below W's last bit vanish outright, and it is not the same thing serving computes. Keep PEFT's parameters and naming, and replace the forward with the low-rank path `base(x) + s·(x·Aᵀ)·Bᵀ`.
  When implementing with `torch._grouped_mm`: the `lhs` from `get_delta_factors` is a non-contiguous view, call `.contiguous()` first; the inner dimension needs 16-byte alignment, so with BF16 rank < 8 takes the fallback path, and each expert's row count is zero-padded to a multiple of 16.
- Per-expert adapters mean tens of thousands of DTensor parameters (Model D: 44,616). torch's `foreach` optimizer path is quadratic in that count; set `override_optimizer_config: {foreach: false}` and clip sharded gradients on the local shards.
- Implement the delta's rounding point per the decision in the next section.

Localize low-precision mismatch per the [diagnosis guide](../debugging-mismatch/README.md);
passing BF16/BF16 only proves that combination and cannot be upgraded to "native low-precision rollout verified".

### Where the delta is rounded

A LoRA layer's output is `base(x) + delta`, finally stored as BF16. There are two forms for the rounding point:

| Form | Method | Who uses it |
|---|---|---|
| A | Round the base GEMM result to BF16 first, add the delta, round again | vLLM (punica expand for non-expert layers; `intermediate_cache.add_(lora_delta)` in expert `fused_moe/experts/triton_moe.py`; row-parallel layers add on each TP rank's own partial sum), SGLang (`srt/lora/lora_moe_runners.py`, shrink/expand intermediates are also BF16), Miles (`torch.add(base_out, delta)`) |
| B | Keep the base product in FP32, add the delta, round once | The internal verl fork's native Model D implementation (an internal fork PR) |

Example: base 1.0029, delta 0.0020: A gives 1.0 (delta rounded away), B gives 1.0078, one BF16 ulp apart; when base and delta cancel the difference can be larger, so do not write "at most one ulp". Under form A, deltas smaller than half an ulp of the base element are deterministically dropped: early in training deltas are small, so the adapter is updated but forward barely sees it; each rank of a row-parallel layer drops its own share, so how much is dropped varies with TP.

**Who decides.** The serving side is not changed. Which form the training side uses is decided per model by the owner; if undecided, follow serving with form A, and do not change only the training side to make small deltas take effect. Neither form is bitwise identical to serving; once chosen, list differences like those below one by one in the recipe deviation table, rather than writing "same rule as serving".

**Model D decision and evidence.** On 2026-10-16 it was first decided to follow serving; on 2026-10-22 the owner changed it to form B, covering all adapter-bearing projections of the internal verl fork's native Model D implementation, including routed expert FC2 (w2) (in that internal fork PR; output is bitwise unchanged without an adapter), with serving unchanged. The reason for B is that the delta fully enters the training-side forward, not lower KL: the measurement recorded in that PR is an overall constant offset of about 0.002–0.003 higher train-inference KL, not a reduction (the change measured also included MXFP8 activation QAT for non-expert projections, and the two were not measured separately); the KL 0.09 → 0.03 that appeared in a commit message is the effect of a different change and cannot serve as evidence for B. The upstream Model D PR a public PR lists five known differences from vLLM serving in model comments and the PR description:

1. Rounding point of the delta addition (table above);
2. Finalize order of routed expert FC2: the training side rounds `base + delta` per expert before multiplying by the routing weight; per the audit's reading of the source, vLLM's finalize sums the weighted outputs in FP32 (not traced ourselves on that side);
3. After residual addition, round to BF16 before RMSNorm; vLLM uses FlashInfer's fused add + RMSNorm (`gemma_fused_add_rmsnorm`), doing the norm on the FP32 sum;
4. With TP>1, each vLLM rank rounds its own share to BF16 before reduction, while the training side computes one full-width product;
5. Inference-side NVFP4 KV cache and indexer cache are not simulated on the training side ([inference-side quantization inventory](../precision-quant-qat/04-inference-quant-inventory.md)).

**Verification.** Sweep delta magnitude with nonzero B (B×0.1/1/10) and compare the fraction of the delta that takes effect on each side; zero B cannot detect these problems.

## 5. Save and resume must also be measured

After resuming from the same checkpoint: check adapter, optimizer, step and the required random state;
in the reduced-layer stages (G3, G5) first score on fixed tokens, then complete the next update/sync; full-model resume does not insert fixed-token scoring
and goes straight into the next round. The mere existence of adapter files does not prove training can resume: Model A's PEFT-format
export wrote a well-formed adapter with 0 tensors (PEFT 0.21 selects state-dict entries by full module name, the exporter keyed them
relative), so every export checks its tensor count against the adapter contract and fails on a mismatch.
Records cover layer count, target families, rank/topology, precision, policy version, test inputs and source SHA.

The minimal day-0 deliverable is: contract table + single-step parameter evidence + receiver verification + fixed-token scoring on both ends
+ one resume. Reuse existing logs and tests first; do not build an audit framework per model.

<a id="part-2-6-what-to-add-first-after-it-runs"></a>
## 6. What to add first after it runs

1. **First close a reproducible deliverable**: the successful config, necessary fixes, dependencies and overlays must be rebuildable from pinned versions;
   gains from an integrated stack cannot be treated as independent validation of a single PR diff.
2. **Then keep cheap regressions**: reuse the reduced-layer model to cover train/sync/score and save/resume with a nonzero adapter;
   keep minimal tests from localized bugs, and do not make full-model long training the entry point for every change.
3. **Finally extend capability or performance**: low-precision rollout, packing, more training modules and multimodal (router replay for MoE models is not in this list:
   it must be on for formal GRPO, see [baseline hyperparameters §1](../recipe-evaluation/03-day0-baseline-hparams.md#1-fixed))
   are accepted separately; absent a corresponding failure or resource bottleneck, do not copy another framework's implementation just because it supports it.

See the same model's a public PR reduced-layer CI script:
it provides a reduced-layer multi-step test and logs mismatch metrics, but also disables the built-in KL/logprob checker.
When borrowing its test layering, check the assertions actually executed; metrics in logs do not mean a correctness gate that blocks regressions already exists.
