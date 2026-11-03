# DeepSeek-V4-Flash recipe

GRPO on DAPO-Math with verl, Megatron-LM (mLite) as the actor and
batch-invariant vLLM as the rollout. The actor runs the same kernels as the
rollout (`engine.impl=vllm`), so rollout and trainer log-probs match bit for
bit. The recipe is organized by hardware.

| Hardware | Status | Launcher | Evidence (BI on) |
|---|---|---|---|
| [NVIDIA GB200](#nvidia-gb200) | ✅ Validated | verl#7688 | 100 RL steps, KL = 0 at every step (8 nodes × 4) |
| [NVIDIA H100](#nvidia-h100) | ✅ Validated | verl#7688 | `--hardware h100` preset |

The algorithm, data, lengths, batch and optimizer are the same on all
hardware (see [Shared settings](#shared-settings)); only the topology, the
kernel backends and the image differ.

## NVIDIA GB200

### Code and launcher

All three code branches come from the stack used by the upstream launcher in
[verl#7688](https://github.com/verl-project/verl/pull/7688). The Megatron
branch is the exact tree of the 100-step run, and the verl branch differs only
in launcher presets. The vLLM branch carries the same alignment kernels but is
not the exact tree of that run, which used an older upstream base; rerun the
`quick_alignment_test` before relying on it.

| Branch | Source | Notes |
|---|---|---|
| [`vllm/deepseek-v4-flash`](../../../../tree/vllm/deepseek-v4-flash) | ISEEKYAN/vllm `ds4-v9-rc1` @ `168a040e` | Baked into the image by `docker/Dockerfile.dsv4_true_on_policy_preview` |
| [`megatron/deepseek-v4-flash`](../../../../tree/megatron/deepseek-v4-flash) | ISEEKYAN/Megatron-LM `ds4_vllm_align_preview` @ `327a2382` | Megatron Lite (`experimental/lite`) actor; mount as `MEGATRON_ROOT` |
| [`verl/deepseek-v4-flash`](../../../../tree/verl/deepseek-v4-flash) | verl#7688 @ `0c849d86` | Contains the launcher and the Dockerfile; mount as `VERL_ROOT` |

```bash
git clone --single-branch -b verl/deepseek-v4-flash https://github.com/Inferact/zero-mismatch-rl.git verl
git clone --single-branch -b megatron/deepseek-v4-flash https://github.com/Inferact/zero-mismatch-rl.git megatron
VERL_ROOT=$PWD/verl MEGATRON_ROOT=$PWD/megatron \
MODEL_PATH=<DeepSeek-V4-Flash> TRAIN_FILES=<dapo-math-17k.parquet> VAL_FILES=<aime.parquet> \
  bash verl/examples/grpo_trainer/run_deepseek_v4_true_on_policy_preview_megatron.sh \
  --hardware gb200 --mode aligned
```

Run inside an image built from `verl/docker/Dockerfile.dsv4_true_on_policy_preview`
(it builds vLLM from the same commit as the vLLM branch).

| Mode | What it runs |
|---|---|
| `quick_alignment_test` | 1 node × 4 GPUs, four-layer model, 2 steps; checks the 0-diff path end to end |
| `aligned` | Full model, BI on (`VLLM_BATCH_INVARIANT=1`), vLLM-visible mLite actor |
| `baseline-r3` | Full model, BI off, Megatron MXFP8 actor with routing replay (R3); the comparison arm. It needs the Megatron `dev` branch with Megatron-Bridge, not the branch above |

### Settings of the `aligned` mode

| Item | Value |
|---|---|
| Topology | 8 nodes × 4 GB200; actor PP1 / CP4 / EP32; rollout TP1, DP8 / EP8 |
| Steps | 100; save every 5, validate every 10 |
| Rollout | `moe_backend=deep_gemm`, `linear_backend=deep_gemm`, `all2all_backend=deepep_low_latency` (MNNVL on), `VLLM_DS4_DECODE_KERNEL=sparse`, FP8 KV cache, `max_num_seqs=128`, `gpu_memory_utilization=0.65` |
| Determinism | `VLLM_BATCH_INVARIANT=1`, `VERL_FULL_DETERMINISM=1`, actor and rollout `full_determinism=True`, fixed seed 42 |

## NVIDIA H100

The verl#7688 launcher has an H100 preset (`--hardware h100`, 8 nodes × 8):
actor PP4 / CP2 / EP16 and rollout EP16, on the same branches and image as
GB200.

## Shared settings

| Item | Value |
|---|---|
| Data | DAPO-Math-17K; train batch 128 prompts × n = 8, mini-batch 32; prompt 2048, response 14000; overlong buffer 4096 |
| Optimizer | lr 1e-6 constant, no warmup; optimizer offloaded to host (`offload_fraction=1.0`), parameters stay on GPU |
| Rollout | FP8 W8A8 weights (UE8M0 block scales), FP8 KV cache, `max_num_batched_tokens=8192`, custom all-reduce off |
| Actor | `engine.impl=vllm`, cross-entropy fusion on, `ppo_max_token_len_per_gpu=4096` |

## Why these settings

The reasons behind each setting, and what failed without them, are in
[consistency-case.md](consistency-case.md) and
[cp-compressed-boundary.md](cp-compressed-boundary.md). How the Nemotron
recipe deviates from this one is in
[the Nemotron recipe](../nemotron/recipe.md).
