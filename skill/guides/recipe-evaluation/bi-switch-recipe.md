# One recipe, one image: deliver true on-policy and a conventional baseline with the BI switch

The structure follows DS4's EXACT_ALIGNMENT: the same delivery image and the same recipe, with
`VLLM_BATCH_INVARIANT`, defined by the recipe itself, selecting the whole arm. The two arms may differ in training
engine, precision and routing replay; the switch is only the single entry point. The pitfalls below come from
Nemotron-H Lightning NVFP4 (Mamba hybrid MoE, 2026-10); re-check each item for a new model, do not copy the config.

## How the two arms are defined

BI=1 is true on-policy: the training forward calls vLLM kernels and reads the deployed bytes; rollout uses the vLLM
BI path; deployed bytes are shipped as is. BI=0 is the conventional baseline: training uses the default MCore path
(verl Megatron engine / Megatron-Bridge, without injecting the alignment implementation of
[C6](../../README.md#gate-table)), rollout uses vLLM's non-BI kernels, with the
config audited per [C1](../../README.md#gate-table). Under BI=0 the following items
are decided per model, recording who approved each:

| Decision | When needed | Nemotron's choice (example) |
| --- | --- | --- |
| Training precision: bf16 or weight QAT (STE) | QAT is an option only when the deployment uses quantized weights | QAT by default, bf16 optional |
| MoE routing replay (R3) | MoE only | Always on |
| Requantize from master at sync | Quantized deployment only | Requantize per the checkpoint rules at every sync |
| Rollout importance-sampling correction (TIS etc.) | Arms whose train-inference mismatch is not 0 | Off; the BF16 arm crashes at about step 41, see [Quantization and QAT](../precision-quant-qat/02-qat-and-ste.md#42-bf16-arm-collapses-around-step-41-single-seed-cause-is-a-hypothesis) |
| Rollout MoE backend | When the backend chosen by `auto` does not run | Explicit `flashinfer_cutedsl` (Marlin, chosen by `auto`, raises a PTX version error), consistent with the BI0 arm of the BI=1 bench |

Do not build BI=0 as "the BI=1 actor with non-BI kernels swapped in". That requires changing the alignment
implementation, and the result is neither a conventional baseline nor true on-policy; Nemotron built that version,
the human rejected it, and it was discarded.

## Rules

- **Same image, same recipe.** If the recipe does not support a switch, first change the recipe on the delivery
  branch, then build one image shared by both arms; do not create a separate image, recipe or source mount for a
  single experiment. Delete the old standalone baseline recipe.
- **Separate output and checkpoint paths.** Add a suffix only for BI=0; BI=1 paths stay the same as history. With a
  shared path, `resume_mode=auto` resumes from the other arm's checkpoint (review actually caught this). Changing the
  run name makes an in-progress resume chain lose its first-segment checkpoint: do not change the image of a running
  chain.
- **BI=1 command unchanged byte for byte.** The recipe must provide a mode that only prints the resolved full training
  command without launching the job (Nemotron uses `DRY_RUN=1`), and a mode that only does hydra compose. For each
  mode × optimizer, print the command with the recipe before and after the change and `diff` the results (or compare
  sha256): BI=1 must be identical; BI=0 compared with the original baseline recipe may differ only in names, paths
  and newly added explicit switches. Every compose combination must have rc 0.
- **Header comments state only what was measured to hold.** Review has caught false descriptions such as "the BI=0
  actor uses vLLM default kernels". Known differences between the two arms' rollout configs (e.g. R3 forcing prefix
  caching on) go into the comparison report.
- Delete untested switches (e.g. unverified mxfp8, routing replay variants); do not leave them for users to trip on.

## BI=0 arm integration pitfalls

| Problem | Consequence | Handling |
| --- | --- | --- |
| THD packing does not pass `total_tokens` into `PackedSeqParams` | Mamba gets no `seq_idx` and scans the whole pack as one sequence; logprobs are silently corrupted | Pass it when CP=1; raise an error for CP>1 while unverified |
| The actor needs the BF16 release, rollout needs the quantized checkpoint, and both share `model.path` | The actor is built from quantized weights | Separate `hf_weights_path` |
| MTP not fully disabled (the provider splices the pattern back) | Extra untrained layers | Clear both `mtp_num_layers` and the MTP pattern |
| Expert bias is updated during training by default | Routing drifts and no longer matches inference | `moe_router_bias_update_rate=0` |
| The framework's built-in QAT only modifies the GPT layer spec | Hybrid models read the hybrid stack spec, so QAT does not take effect | Attach quantizers after building the model from the TE spec, and replace the quantizer with the same `requantize` the exporter uses |
| QAT library signature incompatible with TE grouped linear | Startup error or wrong quantization | Quantize weights only; unit test forward = deployed bytes, gradients flow back to master |
| First refit after rollout `load_format=dummy` | Stale FlashInfer KV scale cache, garbled output | See [pitfall P15](../precision-quant-qat/03-pitfalls.md#p15-flashinfer-attention-lazily-caches-kv-scales) |
| param + optimizer offload, HF safetensors export | Host OOM, EFAULT on save | Chunked optimizer offload, save only dist ckpt; see [checkpoint and host memory](../checkpoint-resume-memory/03-pitfalls.md#2-efault-root-cause-unknown), [§4 pinned rounding](../checkpoint-resume-memory/03-pitfalls.md#4-pinned-buffers-rounded-up-to-powers-of-2) |

R3 checkpoints: enabling R3 in vLLM requires prefix caching and Model Runner V2; capture is written by
**absolute layer number** (non-MoE layers are listed as 0), the training side takes columns by the global
`layer_number`, which must also match under PP; exclude the MTP router. Pitfalls recorded by miles: omitting
`routed_scaling_factor` causes a drift of about 0.28; when a monolithic backend bypasses top-k capture, the replay is
all zeros.

## Smoke-test criteria

- BI=1 quick: every step has `rollout_probs_diff` max = mean = 0.
- BI=0 quick: mean is the same order of magnitude as an independent reference (Nemotron QAT about 0.010, BF16 about
  0.025), max can approach 1, grad norm is finite.
- When grad norm is exactly 0 at some step, first check in-group rewards: if all rewards in the group are equal, the
  advantage is 0; it is not a numerical problem.
- After changing the BI=0 QAT implementation, compare probs_diff against the old implementation on the same quick
  workload, and print the number of replaced modules (consistent across ranks).
