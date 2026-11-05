# Kimi-K3.1

Zero-mismatch RL for Kimi-K3.1 (hybrid KDA linear attention + sparse MLA, MXFP4 MoE, LoRA on all modules incl. experts): the vLLM rollout and the trainer produce bit-identical token log-probs.

## Status

| Item | Value |
|---|---|
| Rollout | vLLM, TP8 / EP8 per replica, `VLLM_BATCH_INVARIANT=1` |
| Trainer | Megatron-LM (mLite) + verl, PP4 / EP4, LoRA r=32 |
| Evidence (BI on) | 100 DAPO steps with LoRA, rollout/trainer log-prob mismatch 0 at every step; AIME 24/25/26 mean@8 0.451 → 0.664, pass@8 0.633 → 0.867 |
| Hardware | GB300 (100-step run: 4 nodes × 4 GPUs, rollout DP16/EP16, actor PP4×EP4) |

## What was hard: KDA

KDA's prefill path (chunkwise, 64-token blocks) and decode path (per-token recurrence) round differently, and mixed batches route decode rows into the prefill path, so results depend on the batch. The deployed fix is chunk-form batch invariance (CF-BI): row t of a chunk depends only on the chunk entry state, rows < t and token t; decode keeps the entry state plus the rows already computed in the chunk and computes exactly that row, so prefill, decode and the trainer forward all evaluate the same expression in the same order (fixed 3xTF32 split, ascending k, log2-domain gates). See `reports/kda-cfbi-design.md`.

Other pieces: fixed-tree TP all-reduce and TP-ordered reduction (`reports/k31-tp-ep-invariant-design.md`), FlashInfer W4A16 MoE with CUDA expert LoRA and fixed-order combine (`reports/DELIVERY-moe-fi-w4a16.md`), mLite K3.1 layout (`reports/c5-mlite-k31-layout.md`), CP=2 (experimental: bitwise vs CP=1 in microbenchmarks, no RL run yet), deployment overlays (`reports/k31-bi-deploy.md`).

## BI rollout cost

Same image and recipe, only `VLLM_BATCH_INVARIANT` switched, LoRA on every request, rollout 2 × TP8/EP8 on 4 GB300 nodes.

| Measure | BI off | BI on | BI cost |
|---|---|---|---|
| RL rollout phase, step 1, GPU memory utilization 0.87 | 353 s | 412 s | 17% longer |
| Whole RL step, step 1 | 1330 s | 1525 s | 15% longer |
| Serving with LoRA active, util 0.90, 128 concurrent | 1708 tok/s | 1543 tok/s | 9.7% lower |

At utilization 0.82 the BI-off arm had less free KV cache than BI-on (larger CUDA-graph footprint) and its rollout was slower; the root cause is being measured. See `reports/bi0-bi1-claim-conditions.md`.

## Code

| Branch | Content |
|---|---|
| `vllm/kimi-k3.1` | changed vLLM files (`CHANGED.txt`) and the K3.1 CUDA kernels in `csrc/k31/` |
| `verl/kimi-k3.1` | verl tree used for the runs, recipe `examples/grpo_trainer/run_k31_true_on_policy_preview_megatron.sh` |
| `megatron/kimi-k3.1` | mLite tree with the K3.1 model |

These branches are a single commit of the working trees (no separate upstream snapshot commit yet).

Paths, hosts and image names are replaced by placeholders (`$WORK`, `$MODEL`, `<image>`, `nodeN`).
