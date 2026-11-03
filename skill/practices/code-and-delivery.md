# Code, experiment snapshots and delivery

Shared by both workflows: how RL-specific code is managed from the base image to a releasable image (§1–7), code
ownership across the three repositories and "three PRs, one image" for cross-stack train-inference consistency
delivery (§8), and PRs and cleanup (§9). Does not repeat weight-update, lifecycle, cold-start
or model acceptance details.

## 1. Branch model

The RL branch must be based exactly on the source commit of the current base image, and be independent of the main branch:

```text
<base-image-source-commit>
├── main
└── <base-image-commit>-rl
```

## 2. Change capsule layout

- One independent semantic change maps to one `<capsule-name>` worktree;
- Changes of the same kind go in a subdirectory, e.g. `reload/`, `draft/`, `loading/`,
  `observability/`;
- Each capsule registers its base SHA, branch, modified files, dependencies, corresponding PR and status;
- Unregistered ad-hoc overlays are forbidden.

<a id="part-1-3-experiment-source-snapshot"></a>
## 3. Experiment source snapshot

Before each experiment starts, turn the current code of the change capsule into a local snapshot commit
without changing the uncommitted state of the original worktree. From that commit create a
`run-snapshot/<run-id>` local branch and a clean worktree; use it as the overlay source,
and following the [overlay vs native wheel rules](../guides/environment-provenance/02-runtime-provenance.md#boundary-between-overlay-and-native-wheel)
deploy the minimal set of pure Python files, verify the actual import path and file hashes, and do not overwrite the image's full vLLM source tree.

The result directory records at least branch, snapshot commit, base SHA, binary patch, tree hash,
image digest and `rl-patch-stack.lock.yaml`. The branch name cannot be the only proof: ordinary experiments
are cleaned up periodically after archiving commit, patch and tree hash; snapshots used for PASS, images, PRs or formal reports
are pushed to the internal remote and kept long-term, except for a confidential model (one whose existence is under embargo), whose snapshots stay on local
branches and in the workspace until the owner lifts the designation.

Use
a source-snapshot helper script
(a small tool kept next to the project, not part of this repo)
to run `prepare`, `verify` and `cleanup`; keep the patch-stack manifest
in a fixed template (`rl-patch-stack.yaml`, see §5).
The pre-submission preflight script's source-manifest and container-source checks
reuse the same manifest before submission to verify the changed files and resolved image digest in every container.

## 4. Patch promotion order

Code is submitted to `vllm-project/vllm` first; content containing sensitive information goes to the internal
repository (nothing about a confidential model goes anywhere until the owner lifts the designation); only patches that are not yet upstreamed but required for RL are temporarily stacked on `<base>-rl`.

Public PRs, commits and logs must not contain internal models, paths, recipes, checkpoints, job IDs
or experiment context.

## 5. Patch-stack manifest

Use one machine-readable `rl-patch-stack.yaml` to record:

```text
base image/source commit
capsule order and dependencies
upstream/internal PR
validation status
RL image digest
```

It is the single source of truth for the patch stack and does not depend on chat history or manual reconstruction.
The manifest lives in the project control directory, not in the source branch it describes, to avoid the circular dependency
of a manifest commit having to describe its own commit.
The tool requires the first-parent commits of `base → HEAD` to be listed completely and in order in
`patches[].commits`; anything missing, extra or out of order fails immediately.

## 6. Base image update

When the base image is updated:

1. Lock the new image digest and source commit;
2. Rebase every active capsule separately;
3. Resolve conflicts and run minimal validation;
4. Rebuild `<base>-rl` in dependency order;
5. Regenerate the RL-specific image and provenance.

## 7. Validation and release

Before release, follow the existing RL validation protocol: target before draft, use two real
critical recipes, cover the declared CUDA Graph path, and pass the fixed-token and tiered
GSM8K/acceptance gates.

The image tag and provenance must trace back at least to the base commit, RL branch SHA, patch-stack
manifest and image digest.

## 8. Three-repository delivery: from experimental implementation to three PRs and one image

This rule applies generally to cross-stack RL delivery for future models: one vLLM PR, one Megatron PR,
one verl PR, and a unified image built from the versions fixed by those three. Default targets are
`vllm-project/vllm:main`, `NVIDIA/Megatron-LM:dev`, `verl-project/verl:main`;
forks may be used to push heads but do not replace the main repository target branch. This section defines change ownership and delivery checks; it does not replace
the [ten-step process](../README.md), the [numerical thresholds](../validation.md) or human recipe review.

### Before changing code: give a change map first

Determine the inference-side change locations at step 3 and complete the training-side and integration interfaces at step 5; you are not required
to design every interface at step 1. Submit a short "feature → existing module/interface →
proposed change location → reason for reuse or addition" table, covering at least model construction, forward/backward,
quantization, parameter registration, checkpoint, online sync and build entry points.

Inspect the implementation and callers in the chosen checkout before deciding file names; the paths below are design
references, not a requirement for every model to copy DeepSeek's directory layout. Avoid writing a standalone engine first
and only discovering at the end that an integration layer already exists or that the real trainer cannot use it.

### vLLM: implement in the real execution path, do not deliver a patch loader

- For the development environment and code sync, use your team's development tooling; you must follow
  [same-commit worktree, per-file sync and build verification](../guides/environment-provenance/02-runtime-provenance.md#boundary-between-overlay-and-native-wheel);
  development sync is not how the final image is installed, and does not change the target branch of the main repository PR.
- Model-specific selection goes in the model construction/execution entry; shared kernel, backend, metadata
  and reload changes go in the existing responsible modules, stating which models and callers are affected.
- BI=0 keeps the reviewed native path; BI=1 explicitly selects the aligned implementation. Check the actual backend,
  and fail early when a required BI capability is missing; never silently fall back and keep claiming consistency.
- Deliver a kernel's numerical implementation, binding and build registration together; CUDA/C++ changes go into the normal
  build, and JIT changes such as CuTe DSL record the compiler and specialization. Do not submit only
  the Python call while depending on uncommitted `.so` files, source files or build scripts inside the experiment container.
- Sync is not just copying parameters: follow the existing reload lifecycle to handle quantization scales,
  packing, derived buffers, cache invalidation and Graph address constraints; test real online reload,
  including a second update; success of model reconstruction does not substitute for it.
- Verify the default path was not changed by accident, and cover the modified shared interfaces; when the target only supports a given
  model/runner/topology, restrict it explicitly; do not claim BI support for other models along the way.

Reference points of [vLLM #54955](https://github.com/vllm-project/vllm/pull/54955):
the formal kernel integration in `csrc/batch_invariant/`, the DeepGEMM BI capability check,
the stable tie-break / prefill-decode semantics of sparse selection, and layerwise reload
calling the model's `finalize_weights_after_layerwise_reload`. These respectively address operator
numerics, backend authenticity, execution shape changes and post-update state; it is not just adding a BI switch.

### Megatron mLite: separate model, aligned operators and runtime

The train-inference consistency implementation uses mLite; the training baseline uses the default MCore path of the existing verl recipe,
and the aligned implementation is not injected into the baseline. The training baseline and the vLLM BI=0 inference performance baseline are recorded separately and must not
be treated as the same comparison; when the target model has support gaps, state them and get them reviewed; do not switch training approach on your own.

| Change | Preferred owner | Should not go into |
| --- | --- | --- |
| Model layer structure, parameters and checkpoint mapping | model/protocol/checkpoint modules under `experimental/lite/model/<model>/` | temporary model implementation inside a verl recipe |
| vLLM-aligned forward, autograd/STE, model-specific quantization | that model's aligned implementation and primitive modules | miscellaneous top-level mLite scripts, global monkeypatches |
| Reusable training/communication/optimizer capabilities | existing shared primitive/runtime, modified only when there is a real interface gap | one model's recipe constants and diagnostic branches |
| verl engine adaptation, batch/loss/sync bridging | existing `experimental/lite/examples/verl/verl_mlite/` interface | rebuilding an equivalent engine inside verl |

- Native and aligned implementations are selected via existing protocols/factories/explicit config; do not scatter
  environment checks across many low-level forwards, or replace instance methods/global functions at runtime.
  Small local branches may stay; do not copy the whole model just to keep directories tidy; reuse shared math and backward
  as much as possible, and implement a primitive separately only when its numerical semantics really differ.
- Do not assemble production operators with `inspect.getsource`, AST rewriting, or string replacement plus `exec`,
  and do not use `sitecustomize` or ad-hoc `PYTHONPATH` hijacking as a model registration mechanism.
  Import stable interfaces from real dependencies, and submit necessary changes normally to the owning module.
- Preserve the contracts for parameter registration, state_dict/checkpoint, distributed reduction and recomputation; do not
  replace the training master parameters with an inference cache not managed by the optimizer just to align forward.
  Backward is accepted separately per the [gradient contract](../guides/training-model-support/06-backward-validation.md#first-write-down-the-gradient-contract); it is not required to be
  "bitwise identical" to inference, and a kernel without autograd must not automatically be treated as needing STE.

[Megatron #7050](https://github.com/NVIDIA/Megatron-LM/pull/7050) separates
DeepSeek-V4's `lite/` from `vllm/primitive/` and reuses the existing `verl_mlite`;
it can serve as an example of these boundaries. `verl_mlite` already exists in official Megatron dev; do not assume
it is only a private integration in a downstream fork; still check the version pinned for this task and its actual capabilities.

### verl: a thin recipe, not a second training framework

Deliverables depend on the route:

- **mLite route**: for non-observability changes, deliver as far as possible only one recipe script and one image build script,
  reusing the real trainer of the current version and the installed `verl_mlite`.
- **FSDP2 Day-0 route**: verl does not have this model; the deliverable is a self-written model implementation placed into verl mainline, its
  Auto-class registration in verl, and the recipe, combined into one verl PR; the model implementation itself is the deliverable and does not go through a separate "generic interface gap" pre-review.
  For how to write and register it, see [training-side model support 01 §3.3](../guides/training-model-support/01-module-inventory.md#33-fsdp2-verls-hf-interface-path).

Same for both routes: when there is a real generic interface gap, first give call-chain evidence, the proposed interface change and impact scope,
**implement only after human review confirms**, and make the minimal fix in the owning repository; do not hide code in scripts or move it to the wrong repository to keep the file count low.

The two-file shape of [verl #7688](https://github.com/verl-project/verl/pull/7688) and the
`verl.trainer.main_ppo`/`model_engine=mlite` chain are the reference; its development mounts and overlays
are not how the final image is delivered here. For data, config differences and review rules use
[DAPO reference and review material](../validation.md#dapo-reference-and-review-materials);
do not create a new "approximate DAPO" script that bypasses the trainer.

### Step 10: from clean PRs to verifiable delivery

For a confidential model, delivery is local: the same diffs and evidence, on local branches, with no PR and no push.

1. **Three complete diffs.** For each repository record the user-agreed target owner/branch/base
   SHA and PR head; check merge-base, commit history and the actual diff. Required model support
   must be included in the PR or already merged into the target branch; another unmerged feature branch must not be used as the base.
   Do not carry over history drift or unrelated deletions from a preview PR; pin cross-repository dependencies to commits
   normally. After changing base / rebasing, re-review the diff; evidence from the old head does not migrate automatically.
2. **One rebuildable image.** Build from three clean checkouts, pin the base image digest,
   dependencies and necessary native/JIT build inputs, and install all runtime packages normally, including external adapter packages.
   Training and inference workers use the same immutable platform digest and the same Python/Torch
   and runtime dependencies; different nodes/processes are allowed, but do not use two independent environments inside the image to dodge compatibility issues.
   It must not depend on host source mounts, startup-time patches or uncommitted overlays to run.
3. **Re-run acceptance from a fresh container.** Check the actual workers' interpreter, imports, libraries and versions,
   validate proxy, full zero-update and the real online RL loop of the reviewed recipe against the existing thresholds,
   then record performance/quality. A successful image build, offline disk replay and historical 0 diff do not replace
   validation of the final artifact; when a difference is found, first shrink to a fixed failing replay rather than repeatedly launching full training to localize it.
4. **One correspondence record.** Use the [report template](../templates/consistency-report.md) to list the three
   PRs, image digest, reviewed recipe version and acceptance receipts; mark each item pass/fail/
   inconclusive/not run. Update this record after code cleanup, build or dependency changes; an unvalidated
   clean implementation cannot inherit the pass conclusion of the overlay prototype.

When slimming down implementation PRs, keep non-essential diagnostic tools, docs and experiment artifacts in internal / the model workspace;
necessary regression tests may go with the PR; do not delete existing tests on the target branch, and do not interpret a slimmed PR
as not running tests. Follow each target repository's contribution requirements, record the regressions and model evals actually
run, and submit for human review; if thresholds are not met, keep it in the not-accepted state.

### Reference versions, not proof of passing

As checked on 2026-10-01, all three PRs are open, and the first two are explicitly marked DO NOT MERGE:

| Reference | Reviewed head SHA |
| --- | --- |
| vLLM #54955 | `4b5c627c23e6690b8987100d1cbf78e34ac14bf9` |
| Megatron #7050 | `327a238239448de6771beea185dc6902a8de3eaa` |
| verl #7688 | `640c5f0a317307e99636924804c40624242fa19b` |

PR bodies may describe different historical pins; when citing a design, distinguish the head you reviewed, the experiment version
claimed in the body and the version measured this time; do not treat the reference implementation as fully correct or already merged code.
The Triton implementation in the reference also does not override this workflow's tooling constraints on new/replacement kernels.

## 9. PRs and cleanup

For a confidential model no PR is opened and nothing is pushed until the owner lifts the designation. What lifts it is
written in G0; for a model that is released, it is the public release of its weights, and from then on code goes out only on the owner's
explicit instruction for that delivery. Credentials, cluster paths and hostnames never go out. For a released model the designation is lifted
by the public weight release; the changes then go out as described below.

A day-0 produces its changes on private branches with private history (experiment commits, case material, code names). An upstream
reviewer needs the opposite: one reviewable change at a time, on the public base, that fixes something real with the least new code.

- **Port, do not push.** Re-create each change as a commit on the public target base (cherry-pick or re-apply the diff) and push only
  those commits; private history, overlays and experiment tooling never reach a public remote. Before pushing, grep the outgoing commits
  and PR text for private names, paths and tokens.
- **One fix per PR, in dependency order.** Do not merge independent fixes into one PR; a stack of small PRs, each based
  on the one it needs, with the order stated in the description. Model A's vLLM work went out as 19 fork PRs: expert LoRA for the NVFP4 MoE
  kernel, model LoRA support and its name-contract test, five reload fixes, four MoE-LoRA kernel changes, and single fixes for the sleep
  backup, sparse-indexer top-k order, LoRA shrink split-K, CUDA graph capture, reload completeness, the checkpoint-cache hash and a port of
  an upstream reload fix.
- **Only what is still needed.** Drop a change that upstream already merged (Model A's port of vllm#63319 was closed because the upstream fix had already merged); replace
  a fork PR by its upstream PR once that is open; and say for each fix whether it was hit in a run (with the symptom and the number) or
  found by reading code (preventive). A fix for a problem the delivery does not have is not part of it.
- **Minimal and reusing.** Reuse the existing mechanism the code already has (a config option next to an existing one, an existing alias
  table, an existing kernel path) before adding a new one; prefer configuration over environment variables (Model A's sleep-backup PR moved
  its switch from an environment variable to `--sleep-mode-retain-cpu-backup`, written the same way as the option beside it). No code
  comments in upstream PRs.
- **Tests that fail on the old code.** Run every new test on the old code first; a test that passes there is removed. Assert observable
  behavior (memory returned, output changed, error raised), not one internal access pattern.
- **Look for siblings.** When a fix's root cause is a pattern, search the codebase for other instances before sending: the
  sleep-backup fix found the same pinned-allocation pattern in vLLM's XPU sleep allocator and fixed it in the same PR.
- **Results in the description, plots in the house style.** Claims, validation commands with old-vs-new results, and the effect measured
  in the delivery; when a figure goes to someone else's publication (the vLLM blog), match that publication's existing figures rather than
  pasting a dashboard.
- From `<base>-rl`, open a `<name>-rl → main` PR; three-repository delivery uses the target branches from §8. The PR description records:
  minimal purpose, base and patch stack, related existing PRs / non-duplication rationale, upstream/internal PRs, RL-only content, test
  commands and results, model impact, rollback method, AI assistance and human review requirements.
- Split large PRs into small PRs that can be reviewed independently, and use tree hashes to prove equivalence before and after the split;
  open PRs only on your own fork unless the target repository is ours.
- Resolve conflicts on the head branch of the PR the user specified; do not open a new PR with a similar name and pretend the original PR is fixed.
- The usual cross-repository change order is kernel/runtime → framework integration → recipe/pins → final image acceptance;
  later stages lock the actual revisions of earlier ones.

After the patch is merged and a new image is generated, archive the corresponding worktree, delete temporary overlays, and keep the PR,
commit, manifest and validation report.
