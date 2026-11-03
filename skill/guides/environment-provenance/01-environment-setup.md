# Container and uv environment configuration

> Source audit date: 2026-09-05; the version conclusions below do not represent re-verification against the latest upstream.

For onboarding a new model when the image already has the model implementation but VERL's general dependency lock is incompatible with it.
Phase order and status are still managed by the overall day-0 support workflow.

## 1. Confirm the image first, then resolve training dependencies

1. Pin the image digest, CPU architecture and Python version; record the actual vLLM/Torch/CUDA,
   Transformers and Triton versions and import paths.
2. In the image, check the model class corresponding to the checkpoint architecture and complete one minimal inference run.
   Image tags, PyPI version numbers or upstream PR merge status cannot replace this step.
3. Read the extras, sources and lock of the chosen VERL commit; dry-run first to confirm it will not
   unexpectedly replace the model-specific vLLM or pull in mismatched CUDA/kernel packages.
4. Then create the uv environment, freeze a compatible lock, and record the build commands. Redo compatibility verification when versions change.

Division of labor: the image pins the native runtime; uv manages training dependencies; application source / pure-Python fixes are deployed via
a traceable code snapshot or minimal overlay. Concrete container code sync follows
[02 boundary between overlay and native wheel](02-runtime-provenance.md#boundary-between-overlay-and-native-wheel);
do not mount an entire vLLM worktree of another version over the image.

## 2. Visible dependencies do not mean binary compatibility

`--system-site-packages` only lets the environment see image packages; it cannot stop uv from installing a same-named package that is imported first.
So check the install plan, actual import paths and GPU operators together; a consistent lock or a successful import
alone cannot prove that Torch and vLLM compiled extensions are compatible.

Prefer keeping the image's native stack. If you really need a matching wheel or a dependency metadata override,
record it as an exception specific to that digest and verify the wheel, shared libraries and operators; it must not become a general
"ignore dependencies" template. Specific versions and historical exceptions are in the
Model F environment record (case notes).

## 3. Minimal reproducible steps and acceptance

Run inside the container, in the reviewed training project directory; use the extra names the project actually provides:

```bash
uv venv --system-site-packages .venv
uv lock --check
uv sync --frozen --dry-run --extra fsdp --extra vllm
# After reviewing the sources of native packages in the plan, run the same frozen sync.
uv sync --frozen --extra fsdp --extra vllm
.venv/bin/python -c 'import torch, vllm, triton; print(torch.__version__, torch.version.cuda, torch.__file__); print(vllm.__version__, vllm.__file__); print(triton.__version__, triton.__file__)'
```

This requires a project lock already modified to be image-compatible; do not blindly run the above
extras on an arbitrary upstream checkout. system-site-packages also does not stop uv from installing a same-named package and importing it first.

Acceptance in order: import/lock consistent on all nodes → small GPU operators and cross-node collectives →
FSDP2 LoRA forward/backward on a scaled-down model → rollout → one complete RL step.
A successful `import torch` alone does not prove vLLM compiled extensions are compatible.

Keep a single copy of environment evidence: image digest, code commit/diff, lock hash, actual install and import
list, build commands, smoke log. After upgrading dependencies, rerun the affected gates.

## 4. Version pins must go into the runtime manifest

The compatibility matrix between dependencies is not only in the lock: the Megatron fork is only compatible with Megatron-Bridge v0.6.2;
pins like this go into the source/installed-package layers of the [02 runtime manifest](02-runtime-provenance.md#use-one-runtime-manifest-to-answer-what-exactly-is-running-now);
upgrading either side reruns the §3 acceptance.

## 5. Avoid paying the cold-start cost repeatedly

Where to put compile caches, how to read weights on Lustre, how to build third-party kernel packages from source, and the two enroot import pitfalls:
see [03 cold start and cache](03-cold-start-and-cache.md).

## 6. Do a tokenizer round-trip check before training

transformers 5 rebuilds the pipeline from the class definition for some tokenizer classes (`LlamaTokenizer` etc.), ignoring
the pre-tokenizer stored in `tokenizer.json`. When a byte-level BPE vocabulary is registered under such a class name, encoding
drops all spaces and newlines without error. Train-inference consistency checks cannot catch it: training and rollout use the same wrong
token ids, so logprobs still align; evaluation scores are not necessarily abnormal either.

On 2026-10-11, LoRA training of another model (not the Model D runs referenced elsewhere) ran like this for three days:
transformers 5.17, `tokenizer_config.json` said `LlamaTokenizerFast`, training prompts, AIME evaluation and
canaries were all space-less text, AIME 0.35 looked like a normal number, and it was finally found from the context of
the tail tokens with the largest decoded logprob differences. The tokenizer vLLM loaded itself was correct.

Before training, in the image, do one round-trip through the entry point training actually calls (e.g. `hf_tokenizer`, not separately written loading code):
encode then decode a text containing consecutive spaces, leading/trailing spaces, newlines, tabs, digits and punctuation, with
`clean_up_tokenization_spaces=False`, and compare character by character with the original; also compare ids against a
`tokenizers.Tokenizer` loaded as-is from `tokenizer.json`; then decode the tokens of one real training prompt back to text and eyeball it.
If whitespace is only being removed, it is this problem; if whitespace is only being added (with `add_prefix_space` configured), that is configured behavior, not corruption.
The fix is to load `tokenizer.json` as-is with `PreTrainedTokenizerFast.from_pretrained`. The launcher makes this
a preflight check; if it fails, the job does not start.

## 7. Bringing a run up on another cluster

A second cluster gives more nodes but a different architecture, filesystem and network, and nothing on it is verified. Model A brought its
Megatron run up on GB300 (sm_103) while the main run used GB200 (sm_100), 2026-10:

- **Fetch large inputs on the target cluster.** There was no direct route between the clusters, and relaying through a workstation ran
  at 0.2–0.5 MB/s. Weights were downloaded from Hugging Face on the target cluster in about 10 minutes, at the pinned revision; images
  were rebuilt there as the same file-level changes on a base image pulled from its registry, and checked with a full-tree content
  manifest against the source image.
- **Rebuild architecture-specific native modules.** FlashInfer selects its prebuilt module by GPU architecture, so the sm100a LoRA MoE
  module built for GB200 was silently ignored on sm_103; it had to be rebuilt as sm103a. List every native or JIT module the run
  depends on and confirm in the worker which build it loaded.
- **Check each node's disk and memory layout.** The Megatron frozen-weight offload needs about 205–220 GB of local disk per node; on the
  GB300 nodes the default directory sat on a host `/tmp` of about 150 GB, so it had to be mounted per node, and falling back to host
  memory put a node under memory pressure ([checkpoint 03 §19](../checkpoint-resume-memory/03-pitfalls.md#19-file-backed-offload-of-frozen-weights)).
  The checkpoint layout and size change with the backend too (Megatron: about 93 GB per step for Model A).
