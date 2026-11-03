# Reproducing the forward: call only the kernels, do not port the scheduler

Once the module inventory ([01](01-module-inventory.md)) is done, the first deliverable is a training forward that computes the same math as vLLM. This page answers: how to read the inference implementation, what to fill in, and how to prove it computes the same math. The logprob definition and consumption of R3 routing replay are not covered here; see [workflow](../../README.md) and [rollout engine](../rollout-engine.md).

## 1. First separate the responsibilities of the model implementation and the training framework

Training frameworks (FSDP2, MCore, mLite runtime) can shard a trainable model, but they cannot automatically turn an inference-only kernel into a differentiable op, and they will not write the model for us. Having no Transformers model file for the model on release day is normal; do not wait for upstream: write a minimal PyTorch model implementation compatible with the current loader interface, covering the config, checkpoint loading, forward, gradients and save interfaces actually used; a full upstream Transformers integration, generation, or every parallel feature is not required first. The semantic reference is the vLLM implementation and the checkpoint; an implementation of the same model that only appeared in the image after release is not a reference, see [README](README.md#what-you-have-on-release-day).

| Capability | What the training route still has to do | What is not copied from inference / Megatron |
|---|---|---|
| Model math | Real semantics of attention/SSM, norm, positional encoding, routing, residuals, special embeddings | vLLM's paged KV, decode scheduling, cudagraph management; MCore's provider and ModuleSpec wiring style |
| Backprop | Input gradients of every op on the training path; gradients of parameters that need training; recompute correctness (see [03](03-reuse-backward.md)) | TP/SP/CP-specific collectives and gradient reduction |
| Weights | Names, axes, fused order, expert IDs, sharing relations, derived state | HF↔Megatron shard layout and PP export protocol |
| LoRA | Target coverage, A/B direction and scaling, non-zero adapter takes effect | Parallel linear wrappers, TP-specific adapter aggregation |
| Memory | Init peak, residency of frozen large tables, save/export peak | Megatron memory saver and PP scheduling details |

dtype, dequantization, QAT and update format belong to the [precision guide](../precision-quant-qat/README.md); here we only require each module's input/output precision and differentiable boundary to be stated explicitly.

## 2. Wiring that only calls kernels

Each layer's forward runs once: call vLLM's inference kernel to get the visible output, guaranteeing bitwise equality with inference, while saving the quantities the backward needs; for how to wire the backward see [03](03-reuse-backward.md#wiring-compute-the-forward-once-the-backward-uses-what-the-forward-saved). The forward borrows only kernels, never any vLLM runtime object:

- Do not copy KV paging, decode scheduling, or CUDA Graph management; training uses a full-sequence teacher-forced forward, and packing passes boundaries via thd/cu_seqlens.
- Reuse order: existing model-agnostic differentiable ops → a concise PyTorch reference written to vLLM semantics → existing ops that have a backward → custom autograd/kernel only when truly necessary. This order only decides where to look for an implementation first; the cost of each choice on the training step is accepted at C6 against the C1 budget — do not pick whatever runs first and swap it out at C10.
- Handle discrete top-k / index selection separately from the differentiable tensors being selected; which branches are trained, frozen, or get an auxiliary loss follows the model definition. Do not detach arbitrarily just to get it running, and do not automatically add router/indexer training objectives.
- If vLLM layer modules (FusedMoE, ModelOpt linear) are used directly, they need `get_tp/ep_group`, so the training process must build vLLM process groups (Nemotron did this; DS4's self-written modules do not need it).
- Kernel calls must take the same branch as inference: quantization config, attention backend, and kernel selection under the BI switch must all be reproduced in the training process, otherwise you are comparing a different path. Test wrappers use the same `CompilationConfig(custom_ops=...)` as production; see [the two false negatives in 03](03-reuse-backward.md#two-test-infrastructure-false-negatives).

Three corrections from localizing Nemotron's 7 layers, showing that "the same kernel" also has pitfalls:

1. The direct and compiled paths of the same gated RMS kernel differ; after calling the same existing kernel through an opaque custom-op boundary, the difference disappeared.
2. Attributing to a single factor just because the reduction block differs is wrong; measuring with the same captured kernel showed both blocks match, which refuted that attribution.
3. The remaining 4.768e-7 logprob difference came from the HF forward unconditionally calling `.float()` after lm_head; keeping the real BF16 head output and sharing scoring made it pass. This is not truncating an FP32 projection to BF16 to manufacture equality.

## 3. Read the inference implementation along the full support chain

Order for reading the vLLM implementation: model file (layer types and order, residual and norm positions, positional encoding, attention visibility range, router/indexer, special embeddings) → the kernels and quantization variants each layer actually calls → attention backend and KV dtype → weight loader (name mapping, fused order, derived scales) → branches under the BI switch. Confirm the parameter/buffer list from the same checkpoint/config.

The Miles/Megatron implementation of the same model usually appears only after release; on Day-0 what you can read is the implementation of older models in the same family, used to find structural clues and known pitfalls, not the answer for this model. Reading order: model recipe → model/plugin → ops and backward → bidirectional checkpoint mapping → LoRA → later fixes → Megatron-LM / Bridge dependencies. Do not look only at the first PR or blog highlight charts. The PR index by model family is in [01 Appendix A.2](01-module-inventory.md#part-1-2-what-else-was-fixed-after-first-support).

| What to learn from | Concrete source | Lesson |
|---|---|---|
| Qwen3.5/3.6 GDN | [#740](https://github.com/radixark/miles/pull/740) | A similar attention name does not mean ordinary attention can be reused; conv/recurrent state and fused projection mapping must be confirmed item by item |
| Model F Flash-Next HC/PLE/QSA | a public PR and the HC source snapshot in that plugin (Miles commit `6aea72c18bd9`) | HC uses per-stream/per-feature gates, stream mean and identity residual, not a copy of DeepSeek mHC; the source explicitly rejects the not-yet-supported mHC recompute arena |
| DeepSeek V4 sparse MLA | [#1181](https://github.com/radixark/miles/pull/1181), [#1642](https://github.com/radixark/miles/pull/1642) | #1642 reports a finite forward but NaN in backward due to a shared-memory merge; passing inference does not replace gradient tests |
| DeepSeek V4.1 new paths | [#3179](https://github.com/radixark/miles/pull/3179) | The file set includes Engram, compressor, sparse MLA, HC, quantized ops and bidirectional mapping; training compatibility cannot be claimed just because it is "also V4" |
| Model G sparse index boundaries | a public PR | Invalid top-k/padding and fully invalid rows need defined behavior; disabling sequence packing does not remove padding issues inside the op |
| Model G non-zero LoRA | a public PR, a public PR | The shared-expert delta was reduced twice; a zero adapter hides the error and mismatch only grows after training. Learn the non-zero validation; do not port the TP fix into FSDP |
| Model B / Model C | a public PR, a public PR, a public PR | Model support and native LoRA are separate work; do not assume special experts/shared factors are automatically covered by ordinary Linear adapters |
| Nemotron conv mapping | [#3163](https://github.com/radixark/miles/pull/3163) | Old/new parameter name differences caused 46 tensors to be missed in loading/sync; successful model construction is not proof of completeness |
| Gemma4 / GPT-OSS | a public PR, a public PR | When the structure is close to an existing architecture, the generic provider/skeleton can be reused; a ready-made training implementation of the same model only exists after release and cannot be counted on at Day-0 |
| Model D | a public PR | Text training, VL, indexer training and high-performance kernels are accepted separately; the reference implementation proves correctness first |
| Multimodal | Same as above | The reference implementation proves correctness first; do not claim VL support just because a vision module was constructed |

Do not rewrite a porting tutorial when different sizes reuse the same implementation. For ordinary dense models, first check for a reusable generic skeleton; the model itself still has to be written by us. Multimodal is validated separately per actual task; diffusion is outside the scope of autoregressive language models. Miles is a source of borrowable implementations and problem leads, not an inherently correct oracle; PR authors' experimental results and our reproductions must be labeled separately.

## 4. The minimal implementation that actually needs filling in

### 4.1 Reproduce the math

From the same checkpoint/config and the actual inference code, confirm: layer types and order, parameters/buffers, residual and norm positions, positional encoding, attention visibility range, router/indexer and special embeddings. Model-specific math (`sqrt(softplus)` scoring, SwiGLU with clamp, mHC Sinkhorn, ReLU² experts) is not in libraries and must be written out one by one following the inference source, not the paper.

Self-written implementations must also target the branch inference actually takes (last item of §2): the same op may go to different kernels depending on config, with rounding at different places. One model's dense/shared MLP is configured with `swiglu_limits_shared=7.0` on every layer, so it takes the limited `SwigluStepAndMul` (computed in FP32, rounded once), not `SiluAndMul` (SiLU rounded to BF16 first); writing the training side after the latter makes 27% of activations differ by 1 ulp, which the overall logprob metric does not reveal. In the image, compare the kernel of the actual branch against the training implementation bit by bit.

### 4.2 A frozen base still needs to propagate input gradients

```python
W.requires_grad_(False)
y = linear(x, W) + scale * B(A(x))  # W gets no gradient, but must still propagate to x
loss(y).backward()
```

Do not wrap an entire frozen block in `no_grad()`: it cuts off LoRA gradients of earlier layers. The exception is a frozen lookup path that truly needs no derivative w.r.t. inputs or parameters. With initial B=0, A's gradient being zero can be normal; check with a non-zero adapter/real step, and do not judge backward correct just because the loss is finite.

<a id="33-mapping-checks-cover-values-and-semantics-not-just-shape"></a>
### 4.3 Mapping checks cover values and semantics, not just shape

For each weight class, list source name → training name → serving name, axis transforms and sharing relations. Focus on fused QKV, gate/up order, expert numbering, extra norm/bias, tied weights, non-persistent buffers and derived state (FP8 KV scale, NVFP4 alpha). Missing/extra entries must have an explicit reason; quantization-derived entries are checked per the precision guide and are not required to be byte-equal to the low-precision source. FSDP does not need an HF↔Megatron bridge, but may still need checkpoint→training model and export mappings.

**Fused parameters must be overwritten as whole parameters.** If only the k/v shards of qkv are sent, the q rows are left as uninitialized GPU memory and the model silently becomes garbage, exposed only by indirect metrics such as spec-decode acceptance rate. This applies to partial updates, LoRA merge and weight sync: write whole blocks along training-side parameter boundaries, not along inference-side shards. Generate the receiver's expected list from the manifest (router bias was once missed because it was not in the completeness check); see [weight update](../weight-update/README.md).

### 4.4 Packing boundaries for Mamba / stateful modules

Every stateful op (conv1d, SSM scan, recurrence, PLE shift) must consume sequence boundaries; a correct attention mask does not mean they are correct. Two real incidents:

- When verl went through Megatron-Bridge, packing did not pass sequence boundaries to Mamba, so states of different sequences bled into each other; logprobs were silently wrong and the only metric symptom was a larger mismatch.
- Model F Flash-Next's PLE shift made the second sample reference the last token of the first ([01 Appendix A.5](01-module-inventory.md#a5-how-to-use-model-fs-packing-evidence)).

Check: run unequal-length A/B separately and merged, verifying sample isolation; initialize outputs with different sentinels to detect missed writes. In mLite the boundary metadata is derived from `PackedSeqParams`/`ThdPackMeta`; do not invent a separate `SSMMeta`. Long-sequence boundary points (KV page, packing prepend, the two lengths 7000/9216) are in [04](04-layout-and-length.md).

### 4.5 Recompute and small-model trimming

If activation checkpointing is actually enabled, verify that recompute reads the same state and produces reasonable gradients. Use the minimal layer set that covers every non-equivalent path to be validated plus necessary dependencies; the layer count is determined by coverage, usually about 4–6 layers. Mixed attention, cross-layer sharing and lookups that appear only in specific layers must not be trimmed away. Record the original layer indices kept and the config transforms; slice results only prove the covered paths and cannot pass as full-model mismatch or score-improvement results.

## 5. How to prove the same math is computed

Feed the training forward with trajectories captured from vLLM (fixed tokens, fixed weight version, position, mask, routing); do not compare probability summaries or argmax. The criterion depends on the route: mLite (BI=1) compares raw bytes of FP32 logprobs, target 0 diff; FSDP2 / MCore (BI=0) does not chase 0 diff — first use an independent reference to judge whether each side is correct on its own, then use the day-0 acceptance criteria (precision-rule attribution) to check whether the remaining difference comes only from precision rules.

- Compare under the same policy version; differing dtypes are first judged as a contract mismatch — do not convert and fake bitwise equality.
- Frozen trainer image, full actor, 2048 tokens FP32 0 diff, and five real updates with 883,998 tokens of identical raw FP32 logprobs: this is the deliverable form of this gate for Nemotron.
- Old-logprob recompute and the grad-enabled training forward are two paths; keep evidence for each separately, see [dual-path contract](../../validation.md#dual-path-contract-for-old-logprob-and-training-forward).
- Localize the first divergence per [debugging](../debugging-mismatch/README.md): per-module dump, single-request prefill, then narrow to the op.

## 6. Small tasks handed to agents and stop conditions

Each task outputs only a short conclusion, source version, reproducible command and artifacts, reusing existing acceptance cards without adding new schemas.

| Task | Minimal acceptance | What to do first if it fails |
|---|---|---|
| Implementation inventory | Find the real training model class, op backwards, checkpoint loader; list gaps | If there is no implementation, write the minimal model; do not continue a large-model cold start |
| Module comparison | Forward with same weights, same inputs, aligned semantics; gradient comparison against a differentiable reference | Localize the earliest difference |
| Few-layer training | Non-zero loss, finite gradients, expected adapter actually changes; explicit zero-init exception | Distinguish broken backward, op bug, and target genuinely having no gradient |
| Native sync | Non-zero adapter/merge takes effect via the framework path, repeated updates leave no residue | Fix exporter/receiver; do not introduce a publisher or partial update as a substitute |
| Save/restore | After restoring in a fresh process, forward and the next step are reasonably consistent | Check restoration of model state, adapter, optimizer and special residency |

Move to distributed only after the first two pass; scale up only after the few-layer two-step loop passes. For training/inference comparison, prefer the vLLM + same-source checkpoint we actually use; SGLang support at Day-0 is not required.

## 7. How to judge development cost

- **Training implementation already exists**: the focus is loading, LoRA, sharding and sync wiring; validate with a few layers first, do not rewrite the model.
- **Only an inference implementation, but ops have differentiable references**: fill in model assembly, weight mapping and necessary interfaces.
- **Key ops lack backward or the math semantics are unclear**: list as a separate blocker, do reference and gradient validation first; do not describe the work as "adding a model registration". If a new kernel is needed, estimate it separately; do not promise a uniform person-day figure.
