# DeepSeek-V4.1-Flash: full-parameter zero-mismatch RL

Full-parameter RL on DeepSeek-V4.1-Flash where the vLLM rollout and the Megatron-LM (mLite)
trainer compute bitwise-identical token log-probabilities in the release precision
(block FP8 / MXFP4 weights, nvfp4 KV cache, MegaMoE, DSpark speculative decoding).

- Status (2026-11-03): the code is published as open pull requests (see [Code and PRs](#code-and-prs)).
  A full-model DAPO run on 64 GB200 kept `rollout_probs_diff_max` and `ppo_kl` at exactly 0 for all
  84 steps it ran; SWE-bench Pro agentic RL kept 0 for 20 steps at 32K context and 10 steps at 128K.
- Scope: this case covers only the full-parameter work. An earlier LoRA-based attempt on this model
  is not part of this public copy.
- No run in this case used the exact published Megatron head. Each result below names the code it
  ran on; [Known limits](#known-limits) lists what the published head lacks relative to those runs.

## What was made exact

**Inference (vLLM, `VLLM_BATCH_INVARIANT=1`).** Token ids and FP32 logprobs are bitwise identical
regardless of batch composition and order, `max_num_batched_tokens`, decode vs `prompt_logprobs`
re-feed, prefix-cache hit vs miss, DP rank placement and MTP on/off, with CUDA graphs on
(FULL_AND_PIECEWISE), DEP4, DSpark and MegaMoE. The pieces: a BI CUDA op library, row-invariant
BF16/MXFP8 projections and norms, sparse FlashMLA and indexer under BI (every causal row runs on
the unsplit decode kernel), exact SWA bounded replay, MegaMoE with an ordered EP combine, fixed-split
mHC, DSpark under BI, and opt-in faster tiers. Details:
[inference batch invariance](reports/inference-batch-invariance.md).

**Training (Megatron-LM mLite, `impl=vllm`).** The actor forward calls the rollout's own ops in the
rollout's order and layout over the native mLite parameter tree; the backward is owned (module VJPs,
identity STE through every quantization, BF16-operand tensor-core GEMMs accumulating into FP32
`main_grad`). The forward covers wq_a/wkv/q_norm/kv_norm/wq_b, compressor, frozen indexer,
sparse attention core (FlashMLA by default, mega attention selectable), wo_a/wo_b, Engram, mHC,
MegaMoE over the EP group, the head and the log-softmax. Context parallel, activation recompute,
packed micro-batches and DistOpt checkpoints are supported. Details:
[training forward and backward](reports/training-forward-backward.md).

**Weight sync.** Each step the trainer exports weights in the release format (block FP8 32×32
UE8M0, MXFP4 experts, BF16/FP32 rest) through the same quantizers its forward applies; vLLM reloads
them layer-wise, so the rollout holds the actor's per-step fake-quant weights byte for byte. A
layer-wise reload of the export is byte-identical to a cold load of the same bytes. Frozen Engram
tables are not synced; both sides read the release rows. Details: [weight sync](reports/weight-sync.md).

## Code and PRs

State checked on GitHub on 2026-11-03.

| Component | PR | State | What it contains |
|---|---|---|---|
| vLLM | [vllm-project/vllm#62365](https://github.com/vllm-project/vllm/pull/62365) | Open, head `4345c936af` | DeepSeek-V4.1-Flash batch invariance with CUDA graphs, RL integration fixes (level-2 sleep state, idle-DP dummy batch), opt-in small-M router, opt-in DeepInfer calls (`VLLM_BI_WO_FP8_EPILOGUE`, `VLLM_BI_INDEXER_DEEPSELECT`) |
| Megatron-LM (mLite) | [ISEEKYAN/Megatron-LM#257](https://github.com/ISEEKYAN/Megatron-LM/pull/257) | Open (draft), head `5f14c7ef6` | The DS4.1 vLLM-kernel implementation on native mLite by Yan Bai (NVIDIA): `impl=vllm` forward and owned backward, export and online sync, verl wiring, `run_deepseek_v41_dapo.sh` |
| Megatron-LM | [#269](https://github.com/ISEEKYAN/Megatron-LM/pull/269) | Merged into #257 (2026-10-27) | DAPO recipe: `BATCH_INVARIANT=0` arm changes only the rollout; vLLM default CUDA graph capture sizes; sparse-attention backward determinism off by default |
| Megatron-LM | [#271](https://github.com/ISEEKYAN/Megatron-LM/pull/271) | Merged into #257 (2026-10-28) | Indexer top-k on DeepSelect, paired with vLLM `VLLM_BI_INDEXER_DEEPSELECT=1` (trainer passes `zero_starts=True`); recipe default on |
| Megatron-LM | [#264](https://github.com/ISEEKYAN/Megatron-LM/pull/264) | Open | MegaMoE training chunk size from `DSV41_MEGAMOE_MAX_TOKENS` |
| Megatron-LM | [#265](https://github.com/ISEEKYAN/Megatron-LM/pull/265) | Open | Chunked optimizer-state offload pins host buffers at exact size |
| Megatron-LM | [#266](https://github.com/ISEEKYAN/Megatron-LM/pull/266) | Open | Bounded save-time host memory (streamed D2H, page-cache drop) and Lustre EFAULT retry for dist-checkpoint |
| Megatron-LM | [#267](https://github.com/ISEEKYAN/Megatron-LM/pull/267) | Open | Weight sync: owner-side expert quantization, routed `all_to_all`, `gc.freeze` (about 106 s to about 9 s per step) |
| Megatron-LM | [#268](https://github.com/ISEEKYAN/Megatron-LM/pull/268) | Open | Route sampled vision sentinel ids like vLLM (`bias_vl`, Engram dead token, resync of `bias_vl`) |
| Megatron-LM | [#270](https://github.com/ISEEKYAN/Megatron-LM/pull/270) | Open | SWE-bench Pro agentic RL: verl agent loop over uni-agent |
| verl | [verl-project/verl#8317](https://github.com/verl-project/verl/pull/8317) | Open | GB200 image for mLite + vLLM batch-invariant rollout (on verl `b9c513c44c06`) |

The Megatron PRs other than #257 are Inferact follow-ups that target #257's branch. An earlier
Inferact parity branch ([#260](https://github.com/ISEEKYAN/Megatron-LM/pull/260)) was closed in favor
of building on #257; its design and validation are kept in the reports because #257 implements the
same contract.

Snapshots in this repository (see the top-level [README](../../README.md#code-branches)):
`vllm/deepseek-v4.1-flash` = #62365 at `4345c936af` on vllm-project/vllm@36e0ae2df288;
`megatron/deepseek-v4.1-flash` = #257 at `5f14c7ef6` on NVIDIA/Megatron-LM@7b150acfacee (#269 and
#271 included; #264–#268 and #270 not included); `verl/deepseek-v4.1-flash` = verl#8317 on
verl-project/verl@b9c513c44c06.

## Acceptance evidence

Topology for every full-model run: 16 nodes × 4 GB200 (64 GPUs), training and rollout colocated
(time-shared). Actor TP1 / EP64 / CP2 (CP16 at 128K); rollout one DEP4 vLLM engine per node,
DSpark (5 draft tokens), level-2 sleep, `nvfp4_ds_mla` KV cache, MegaMoE. "0" means
`training/rollout_probs_diff_max` = 0.0 (the max absolute probability difference between the
rollout's logprobs and the trainer's old-logprob recompute over every response token of the step).

### Runs on the #257 line

| Run | Model | Steps | Megatron code | Result |
|---|---|---|---|---|
| DAPO long run | Full | 84 (planned 100; cancelled at step 84 to free the nodes) | #257 at `8df84b5d3` + #264, #265, #266 | 0 at every step; `actor/ppo_kl` 0 at every step; AIME 2024 mean@8 0.733 (step 0) → 0.763 (step 80). [Details](reports/long-runs.md#dapo-long-run) |
| First DAPO attempt | Full | 9 | same | 0 at steps 1–9; two nodes stopped responding during the first checkpoint write to Lustre (step 10). [Details](reports/long-runs.md#first-attempt-node-failure-during-the-first-lustre-checkpoint) |
| G6 short job | Full | 6 (2, save, resume in a fresh Ray cluster, 4) | same | 0 at all 6 steps; save-time host peak 873.5 GB on the driver node; no per-step trainer memory growth. [Details](reports/pr257-gaps.md#g6-short-job-full-model) |
| Proxy resume study | 40-layer proxy, 16 experts, 2 nodes | 3 per run, 3 runs | same | After resume, optimizer masters and moments, model params, LR scheduler, RNG, the step-3 batch, rollout logprobs and old logprobs are bitwise equal to an uninterrupted run; the step-3 backward is not (grad norm differs in the 5th digit). [Details](reports/pr257-gaps.md#proxy-resume-study) |
| SWE-bench Pro, 32K | Full | 20 | integration branch: #257 at `3bc04c83a` + the changes later opened as #264–#270 and the trainer side of #271 | 0 at every step; held-out resolved rate 0.742 (step 0) → 0.707 (step 20). [Details](reports/swe-agentic-rl.md#runs-and-results) |
| SWE-bench Pro, 128K | Full | 3 (G6) + 10 + 2 | same | 0 at every step; CP16 fits (actor peak 129.0 GB allocated). [Details](reports/swe-agentic-rl.md#128k-context) |
| DeepSelect pairing (#271) | Proxy on GB300 | 3 | #257 at `066851478` + #271, with #264–#267 merged for the test | 0 at steps 1–3 |

### Earlier parity branch (#260, closed)

Run on the closed #260 branch (mega attention core, BF16 head), not on published code; kept as
evidence for the design. Formal 3-step run: θ0/θ1/θ2 old-logprob (no grad) and train-forward
(grad) paths 0 mismatch with max_abs 0 over 3,553,540 / 4,072,110 / 3,789,788 tokens, 1024
distinct responses per step, sync payload identical across 64 ranks (93,414 tensors); a 22-step run
after the leak fix: 0 at every step over 125.9M checked tokens. [Details](recipe.md#formal-runs-on-the-earlier-parity-branch).

### Inference gates (vLLM #62365)

All 0 diff with BI=1 and graphs on: alone vs batch vs reversed (pinned and unpinned DP ranks),
decode vs `prompt_logprobs` re-feed, `max_num_batched_tokens` 2048 / 8192 / 16384 cross-run,
prefix-cache hit vs miss, MTP on, batch combinations; on the sparse FlashMLA path also graph vs
eager and short prompts of 1–31 tokens. [Details](reports/inference-batch-invariance.md#gates).

## Rollout cost

BI=1 / BI=0 with the same image and recipe; only `VLLM_BATCH_INVARIANT` differs.

| Measurement | Attention path | BI=1 / BI=0 |
|---|---|---|
| Serving output throughput, 8192 in / 1024 out, c = 1 / 8 / 32 / 64, 12 AB/BA pairs | mega attention (opt-in tier) | 0.911 / 0.922 / 0.919 / 0.907 |
| Same, 4 pairs | sparse FlashMLA (recipe default) | 0.911 / 0.890 / 0.876 / 0.835 |
| RL rollout step, 128 trajectories per DEP4 node, response ≤ 8192, fixed length / natural EOS | sparse FlashMLA | 1.072 / 1.023 |
| Same | mega attention | 1.034 / 1.036 |
| DAPO RL generation time, 16 nodes, 2 steps, with the opt-in small-M router | mega attention | 1.081 |
| SWE-bench Pro 128K agentic RL, 2 steps, LLM time per generated token | sparse FlashMLA | 1.034 |

The agentic step time itself is set by sandbox and tool latency; see
[SWE-bench Pro](reports/swe-agentic-rl.md#rollout-speed).

## Known limits

- **Published head vs validated code.** The published #257 head does not contain #264–#268 or
  #270. Without them: the trainer MegaMoE chunk is fixed at 8192 rows (the validated recipe uses
  2048 to fit CP2 micro-batches of 5120 tokens); chunked optimizer offload pins host buffers with
  power-of-two rounding (measured +41% host memory, enough to exceed the 16-node host budget);
  `run_deepseek_v41_dapo.sh` forces `MCORE_DIST_CKPT_STREAM_D2H=0` (the unbounded save path added
  84–94 GiB per node on a proxy); weight sync takes about 106 s per step instead of about 9 s;
  sampled vision sentinel ids (129264–129268) are not routed with `bias_vl` on the trainer side, so
  once the policy samples one, the rest of that response mismatches. See
  [#257 gaps](reports/pr257-gaps.md).
- **Launcher defaults are not the validated recipe.** The published launcher defaults to CP1,
  `GRAD_OFFLOAD=False`, lr 1e-5 and weight decay 0. The validated runs set CP2, `GRAD_OFFLOAD=True`,
  lr 1e-6, weight decay 0.1, `DSV41_MEGAMOE_MAX_TOKENS=2048` and the streamed checkpoint save.
- **Resume is bitwise for state and rollout, not for the backward.** The step after a resume
  recomputes the same old logprobs but a slightly different gradient (also seen across node pairs
  without resume), so the trajectory after resume is not bitwise equal to an uninterrupted run.
- **Checkpoint writes to Lustre.** Writing a 7.7 TB full-model checkpoint from 64 ranks to Lustre
  made two nodes stop responding once; the long runs wrote to node-local disk. Resuming from
  node-local disk on another node set needs the rank-0-only files copied to every node first.
- **Training quality of the quantized forward.** `impl=vllm` trains with the deployed weight
  quantization in the forward. In a 150-step full-model SFT comparison against the BF16 `impl=lite`
  path the gap was 0.005 nat at the end of epoch 1 and grew at each epoch boundary to 0.17 nat in
  epoch 3 (A/A noise 0.0011). Interpretation: weight updates smaller than one quantization step are
  invisible to the forward, which limits memorizing repeated samples. Single-epoch RL is less
  affected; this is not proven for long RL runs.
- **Gradient check.** Isolated module VJPs pass a 1e-2 relative threshold against an FP64 reference
  except three mHC scalar sums at the FP32 cancellation floor; on the whole model 35 entries exceed
  1e-2 (attributed to FP32 cancellation and low-token experts, with a planted 2% error detected).
  No formal waiver.
- **Evaluation.** One seed per configuration. AIME 2024 rose by 0.03 over 80 DAPO steps; the 32K
  SWE-bench Pro held-out rate fell from 0.742 to 0.707 over 20 steps. These runs show stability of
  the zero-mismatch contract, not a tuned training recipe.
- **Scope.** TP=1 and PP=1 on both sides (DEP rollout, EP/CP/DP trainer). V2 model runner only.
  GB200 for every full-model run; GB300 only for proxies and kernel tests. Inference under DEP>1 is
  bitwise consistent within the DEP group but not bitwise equal to an unsharded run (EP combine can
  turn a BF16 -0 into +0).

## How to reproduce

1. **Image.** vLLM built from #62365, plus Transformer Engine 2.20.2, verl at `b9c513c44c06`,
   FlashMLA, fast-hadamard-transform and a `flash_attn.bert_padding` shim, in one Python
   environment. verl#8317 carries the Dockerfile and its README.
2. **Trainer.** Megatron-LM #257 plus the open follow-ups you need (#264, #265, #266 for the
   16-node recipe; #267 for fast sync; #268 when siblings are sampled independently). The validated
   DAPO run used `8df84b5d3` plus #264–#266.
3. **Launch.** `experimental/lite/examples/verl/scripts/run_deepseek_v41_dapo.sh` with the
   overrides in [recipe and memory budget](recipe.md#launcher-overrides-used-by-the-validated-runs).
   DAPO-Math-17K training data, AIME 2024 validation (n=8, temperature 1.0, 4096 output tokens).
4. **Check.** `training/rollout_probs_diff_max` must be exactly 0 at every step with
   `algorithm.rollout_correction.bypass_mode=False` (the trainer recomputes old logprobs). Run a G6
   short job (two steps, save, resume in a fresh Ray cluster, more steps) before any long run.

## Reports

| Report | Content |
|---|---|
| [Inference batch invariance](reports/inference-batch-invariance.md) | Gates, paired throughput, every deviation from vLLM main and whether it is forced or opt-in, uncovered items |
| [Training forward and backward](reports/training-forward-backward.md) | Starting point, module contract, owned backward, CP, recompute, packing and loss checks, gradient check, attention benchmark, SFT comparison with the BF16 path |
| [Recipe and memory budget](recipe.md) | Recipe vs the DeepSeek-V4-Flash recipe, level-2 sleep state, GPU and host ledgers, checkpoint fixes, formal runs on the earlier branch |
| [#257 gaps and follow-ups](reports/pr257-gaps.md) | Item-by-item audit of #257, the three follow-up PRs, G6 short job, proxy resume study |
| [Long runs](reports/long-runs.md) | DAPO long run, first attempt, trainer memory leak found on the earlier branch |
| [Weight sync](reports/weight-sync.md) | Profile of the 106 s sync and the changes that bring it to about 9 s |
| [SWE-bench Pro agentic RL](reports/swe-agentic-rl.md) | Recipe and deviations from the uni-agent recipe, prefix-hit replay, cold start, 128K, speed, vision sentinel ids |
