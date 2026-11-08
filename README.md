# zero-mismatch-rl

Bitwise-identical token probabilities between the RL rollout engine (vLLM) and
the trainer: zero mismatch, not small mismatch.

This repository holds three things:

- **The skill** ([`skill/`](skill/)): an agent-executable workflow for making
  rollout and training compute the same token probabilities, plus the
  mechanism guides, process rules and templates it relies on. Start at
  [`skill/README.md`](skill/README.md).
- **Model cases** ([`cases/`](cases/)): one folder per model with the
  recipe (`recipe.md`), what broke and how it was fixed, and the evidence.
- **Code**, one branch per framework and model (see below). Each code branch
  has exactly two commits: an unmodified upstream snapshot, then the
  zero-mismatch changes on top.

## Why zero mismatch: REINFORCE Zero

[REINFORCE Zero](https://github.com/hijkzzz/REINFORCE-Zero) shows what zero
mismatch buys in practice. The gap between the rollout policy and the trainer
has two sources:

- **Weight lag**: rollouts come from older weights. This gap is smooth, and
  ordinary importance sampling (IS) corrects it.
- **Kernel mismatch**: rollout and training compute the same weights with
  different kernels. This gap is spiky: with standard kernels the IS
  diagnostic peaks at 442.54× its median, versus 1.50× with aligned kernels.

Remove kernel mismatch with aligned kernels, as this repository does, and
plain unclipped token-level IS is enough to correct weight lag. Asynchronous
RL then trains stably through 20,000 policy updates, at 2.40× the
training-step throughput of synchronous training.

## Support matrix

| Family | Model | Status | Trainer | Evidence (BI on) | BI rollout overhead¹ |
|---|---|---|---|---|---|
| Sparse attention | [DeepSeek-V4-Flash](cases/deepseek-v4-flash/README.md) | ✅ Supported | Megatron-LM (mLite) + verl | 100 RL steps, KL = 0 at every step (32 GB200); AMD MI355X: [verl#8505](https://github.com/verl-project/verl/pull/8505), [by hardware](cases/deepseek-v4-flash/recipe.md) | 8.7% lower RL rollout throughput |
| Sparse attention | [DeepSeek-V4.1-Flash](cases/deepseek-v4.1-flash/README.md) | ✅ Supported | Megatron-LM (mLite) + verl | 84 DAPO steps, KL = 0 at every step; SWE-bench Pro agentic RL, 20 steps at 32K and 10 at 128K, 0 mismatch (64 GB200) | 2.3–7.2% longer RL rollout step |
| Linear attention (hybrid) | [Nemotron-3.5-Lightning-30B-A3B](cases/nemotron/README.md) | ✅ Supported | Megatron-LM (mLite) + verl + uni-agent | SWE-bench Pro agentic RL, KL = 0 at every step (8 GB200 nodes per run) | 12.5% lower RL rollout throughput |
| Linear attention (hybrid) + sparse attention | [Kimi-K3.1](cases/kimi-k3.1/README.md) | ✅ Supported | Megatron-LM (mLite) + verl | 100 DAPO steps with LoRA, 0 mismatch at every step; AIME 24/25/26 mean@8 0.451 → 0.664 (16 GB300) | 17% longer RL rollout phase |

Rollout is vLLM for every model.

¹ Rollout-side cost only: the model's BI-on arm against its standard-kernel
(BI-off) arm on the same hardware and workload. The BI-off arm uses that
model's fastest standard rollout backends, so the number is the whole cost of
the aligned rollout path, not just of one flag.

- DeepSeek-V4-Flash: generated tokens per millisecond during rollout,
  averaged over the 100-step RL run, 24.22 vs 26.53. The BI-off arm is the
  recipe's `baseline-r3` mode (default MoE backend, paged decode kernel,
  routing replay).
- DeepSeek-V4.1-Flash: wall time of the rollout phase of one RL step, same
  image and recipe with only `VLLM_BATCH_INVARIANT` switched; sparse FlashMLA
  path, natural EOS (+2.3%) and fixed-length responses (+7.2%). In standalone
  serving at 8K in / 1K out, output throughput is 9–16% lower
  ([details](cases/deepseek-v4.1-flash/README.md#rollout-cost)).
- Nemotron: rollout throughput over the first 30 steps of SWE-bench Pro
  agentic RL; the BI-off arm is NVFP4 QAT with routing replay.
- Kimi-K3.1: wall time of the rollout phase of the first RL step, same image
  and recipe, LoRA on every request, rollout GPU memory utilization 0.87 on
  both arms: 412 s vs 353 s. The BI-off arm uses FlashMLA sparse with an fp8
  KV cache and FlashInfer W4A16 MoE with CUDA expert LoRA. In standalone
  serving with LoRA active, output throughput is 9.7% lower (1543 vs 1708
  tok/s). Measurement is still in progress
  ([details](cases/kimi-k3.1/README.md#bi-rollout-cost)).

### Key features

| Dimension | DeepSeek-V4-Flash | DeepSeek-V4.1-Flash | Nemotron | Kimi-K3.1 |
|---|---|---|---|---|
| Attention | Sparse MLA + indexer | Sparse MLA + indexer, sliding window | Mamba-2 (decode replays the prefill chunk math) + fixed-split attention | KDA linear attention (decode computes its row of the chunk-form expression from the chunk entry state) + sparse MLA with stable top-k |
| MoE | FP8, DeepGEMM, fixed-order sum | MXFP4 experts, MegaMoE, rank-ordered EP combine | NVFP4 (W4A16) experts, fixed slot order | MXFP4 experts, FlashInfer W4A16, CUDA expert LoRA, fixed-tree combine |
| LoRA | — | — | — | ✅ All modules incl. experts, FP32 add with one rounding |
| Training context parallel | ✅ CP4 | ✅ CP2, CP16 | ✅ CP2 | 🧪 CP2 (experimental: bitwise vs CP1 in microbenchmarks, no RL run yet) |

Data parallel > 1 needs a DeepEP / NIXL all-to-all
backend on DeepSeek-V4-Flash; the default `allgather_reducescatter` breaks
invariance there.

### Hardware

| Hardware | DeepSeek-V4-Flash | DeepSeek-V4.1-Flash | Nemotron | Kimi-K3.1 |
|---|---|---|---|---|
| NVIDIA GB200 | ✅ Validated | ✅ Validated | ✅ Validated | — |
| NVIDIA GB300 | — | — | — | ✅ Validated |
| NVIDIA H100 | ✅ Validated | — | — | — |
| AMD MI355X | ✅ Validated | — | — | — |

- `—`: not verified yet.
- GB200 runs: DeepSeek-V4-Flash 100 RL steps on 32 GPUs; DeepSeek-V4.1-Flash
  every full-model run on 64 GPUs; Nemotron agentic RL on 8 nodes × 4 GPUs.
- GB300 runs: Kimi-K3.1 100 RL steps on 4 nodes × 4 GPUs (rollout DP16/EP16,
  actor PP4×EP4); BI-on/off rollout comparison on 4 nodes (rollout 2 × TP8/EP8).
- H100: DeepSeek-V4-Flash with the `--hardware h100` preset (8 nodes × 8, CP2).
- AMD MI355X: [verl#8505](https://github.com/verl-project/verl/pull/8505),
  36 RL steps at diff 0 (4 nodes × 8);
  see [the recipe by hardware](cases/deepseek-v4-flash/recipe.md).

## Code branches

| Model | vLLM | Megatron-LM (mLite) | verl | Other |
|---|---|---|---|---|
| DeepSeek-V4-Flash | [`vllm/deepseek-v4-flash`](../../tree/vllm/deepseek-v4-flash) ([changes](../../commit/vllm/deepseek-v4-flash)) | [`megatron/deepseek-v4-flash`](../../tree/megatron/deepseek-v4-flash) ([changes](../../commit/megatron/deepseek-v4-flash)) | [`verl/deepseek-v4-flash`](../../tree/verl/deepseek-v4-flash) ([changes](../../commit/verl/deepseek-v4-flash)) | — |
| DeepSeek-V4.1-Flash | [`vllm/deepseek-v4.1-flash`](../../tree/vllm/deepseek-v4.1-flash) ([changes](../../commit/vllm/deepseek-v4.1-flash)) | [`megatron/deepseek-v4.1-flash`](../../tree/megatron/deepseek-v4.1-flash) ([changes](../../commit/megatron/deepseek-v4.1-flash)) | [`verl/deepseek-v4.1-flash`](../../tree/verl/deepseek-v4.1-flash) ([changes](../../commit/verl/deepseek-v4.1-flash)) | — |
| Nemotron | [`vllm/nemotron`](../../tree/vllm/nemotron) ([changes](../../commit/vllm/nemotron)) | [`megatron/nemotron`](../../tree/megatron/nemotron) ([changes](../../commit/megatron/nemotron)) | [`verl/nemotron`](../../tree/verl/nemotron) ([changes](../../commit/verl/nemotron)) | [`uni-agent/nemotron`](../../tree/uni-agent/nemotron) ([changes](../../commit/uni-agent/nemotron)) |
| Kimi-K3.1 | [`vllm/kimi-k3.1`](../../tree/vllm/kimi-k3.1) | [`megatron/kimi-k3.1`](../../tree/megatron/kimi-k3.1) | [`verl/kimi-k3.1`](../../tree/verl/kimi-k3.1) | — |

Clone only the branch you need; code branches contain full framework trees:

```bash
git clone --single-branch -b vllm/deepseek-v4-flash https://github.com/Inferact/zero-mismatch-rl.git
```

Each branch is a release snapshot pinned to the upstream commit named in its
first commit message. The Kimi-K3.1 branches are a single snapshot commit of
the working trees for now (no separate upstream commit yet). Changes that have since landed upstream are noted in the
model's case README.

## Acknowledgments

We thank [RL-Kernel](https://github.com/RL-Align/RL-Kernel) from RL-Align for
its support. RL-Kernel builds deterministic operators that keep rollout and
training numerically aligned across engines and hardware, and it shares the
goal of this repository: zero train-inference mismatch.

## License

The skill and case notes on `main` are under [Apache-2.0](LICENSE). Each code
branch keeps the license of its upstream project (vLLM: Apache-2.0,
Megatron-LM: see its `LICENSE`, verl: Apache-2.0).
