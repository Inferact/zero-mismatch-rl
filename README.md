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

## Support matrix

| Family | Model | Status | Trainer | Evidence (BI on) | BI rollout overhead¹ |
|---|---|---|---|---|---|
| Sparse attention | [DeepSeek-V4-Flash](cases/deepseek-v4-flash/README.md) | ✅ Supported | Megatron-LM (mLite) + verl | 100 RL steps, KL = 0 at every step (32 GB200); AMD MI355X: [verl#8505](https://github.com/verl-project/verl/pull/8505), [by hardware](cases/deepseek-v4-flash/recipe.md) | 8.7% lower RL rollout throughput |
| Sparse attention | [DeepSeek-V4.1-Flash](cases/deepseek-v4.1-flash/README.md) | ✅ Supported | Megatron-LM (mLite) + verl | 84 DAPO steps, KL = 0 at every step; SWE-bench Pro agentic RL, 20 steps at 32K and 10 at 128K, 0 mismatch (64 GB200) | 2.3–7.2% longer RL rollout step |
| Linear attention (hybrid) | [Nemotron-3.5-Lightning-30B-A3B](cases/nemotron/README.md) | ✅ Supported | Megatron-LM (mLite) + verl + uni-agent | SWE-bench Pro agentic RL, KL = 0 at every step (8 GB200 nodes per run) | 12.5% lower RL rollout throughput |

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

### Key features

| Dimension | DeepSeek-V4-Flash | DeepSeek-V4.1-Flash | Nemotron |
|---|---|---|---|
| Attention | Sparse MLA + indexer | Sparse MLA + indexer, sliding window | Mamba-2 (decode replays the prefill chunk math) + fixed-split attention |
| MoE | FP8, DeepGEMM, fixed-order sum | MXFP4 experts, MegaMoE, rank-ordered EP combine | NVFP4 (W4A16) experts, fixed slot order |
| Training context parallel | ✅ CP4 | ✅ CP2, CP16 | ✅ CP2 |

Data parallel > 1 needs a DeepEP / NIXL all-to-all
backend on DeepSeek-V4-Flash; the default `allgather_reducescatter` breaks
invariance there.

### Hardware

| Hardware | DeepSeek-V4-Flash | DeepSeek-V4.1-Flash | Nemotron |
|---|---|---|---|
| NVIDIA GB200 | ✅ Validated | ✅ Validated | ✅ Validated |
| NVIDIA H100 | ✅ Validated | — | — |
| AMD MI355X | ✅ Validated | — | — |

- `—`: not verified yet.
- GB200 runs: DeepSeek-V4-Flash 100 RL steps on 32 GPUs; DeepSeek-V4.1-Flash
  every full-model run on 64 GPUs; Nemotron agentic RL on 8 nodes × 4 GPUs.
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

Clone only the branch you need; code branches contain full framework trees:

```bash
git clone --single-branch -b vllm/deepseek-v4-flash https://github.com/Inferact/zero-mismatch-rl.git
```

Each branch is a release snapshot pinned to the upstream commit named in its
first commit message. Changes that have since landed upstream are noted in the
model's case README.

## License

The skill and case notes on `main` are under [Apache-2.0](LICENSE). Each code
branch keeps the license of its upstream project (vLLM: Apache-2.0,
Megatron-LM: see its `LICENSE`, verl: Apache-2.0).
