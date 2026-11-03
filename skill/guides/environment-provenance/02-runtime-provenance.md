# Environment identity: code existing does not mean it is executing

## Use one runtime manifest to answer "what exactly is running now"

Record recipe, trainer, Megatron/MCore, vLLM, DeepEP, DeepGEMM, FlashAttention/FlashInfer and NVSHMEM separately. Each entry contains at least:

| Layer | Evidence to save | What cannot replace it |
| --- | --- | --- |
| Source | remote, target branch, HEAD, dirty patch hash, submodule gitlink/actual HEAD | `latest`, repo directory name |
| Installed package | actual interpreter, import path, package version, wheel/build revision | `pip show` from another venv |
| Native library | path/hash of the `.so` actually loaded, build inputs, target architecture, dynamic dependencies | Python source HEAD, extension file name |
| JIT kernel | DSL/source hash, compiler version, specialization/config, cache origin | `.so` unchanged |
| Container | OCI manifest/platform digest or `.sqsh` SHA256, parent image, build record | mutable Docker tag, image creation time |
| Run | resolved config, worker environment, model/weight digest, real kernels | exports in the head node shell |

Write `unknown` for unknown items; do not infer the whole source tree from a wheel version string. Audits should go into the **actually running worker, interpreter and mount namespace**; a shell logged into the container and a Ray worker may differ. When dynamic linking is suspected, check the actual process mappings/loader results, not just `ldd`'s static prediction.

The same image tag may point to different digests on different dates/nodes; ARM64 and AMD64 tags may also be independent builds or members of a multi-arch manifest. Compare the target platform's manifest/config/layers; do not judge whether it's the same image just from similar tags or update times. General scheduling switches such as `CUDA_DEVICE_MAX_CONNECTIONS=1` must also be recorded per the task convention; do not quietly add them and then attribute the effect to BI.

Cross-repo compatibility pins are also a line in the manifest: the Megatron fork is only compatible with Megatron-Bridge v0.6.2; the manifest states the pin and its basis, and upgrading either side re-verifies per [01 §3](01-environment-setup.md#3-minimal-reproducible-steps-and-acceptance); do not assemble mutually incompatible releases just so every item shows "latest".

### Communication implementations: do not use one name for four things

Communication stacks such as DeepEP/NVSHMEM are recorded as four lines in the manifest: which implementation (including fork/revision), which path of that implementation (the actual class and kernel for HT/LL/v2), how it communicates (NVLink / RDMA / MNNVL), and what it was built as (peer macro, target GPU, NVSHMEM dependency). For the four-item table, why topology names don't indicate the mode, and the isolation ladder for communication faults, see [debugging 05](../debugging-mismatch/05-comm-fault-ladder.md#do-not-use-one-name-for-four-things).

## Prove the numerical path ran

A version check proves that the files you hashed are the files you meant; it does not prove they are the files the workers imported,
and a config switch proves only that it was set. Both gaps produce runs that look correct: the train-inference diff stays in its usual
range because most of the computation is unchanged, and an A/B on the switch shows no difference. Model A (2026-10) hit both:

- **A stale module behind `trust_remote_code`.** The trainer loaded the model through `trust_remote_code`, which imports the `.py`
  files in the model directory (via the `transformers_modules` cache), not the source tree the run tuple hashed. The model directory
  held a copy from a day earlier, so for 18 steps of the full-model training run `fp8_kv_cache=true` did nothing (that copy had no FP8-KV simulation) and R3
  replay used an older path. Every preflight version check passed; a profiler run found it. An earlier "the FP8-KV switch does not change
  θ0" result was probably measured on the same stale copy.
- **Runtime proofs that were no-ops on another backend.** The FSDP run's per-rank proofs were ported to the Megatron backend, where the
  hooks they counted were never called, so they passed with nothing to count.
- **`auto` resolved from the checkpoint.** The recipe said `kv_cache_dtype: auto` with a comment planning BF16; the engine resolved it to
  `fp8_ds_mla` from the checkpoint's own config while the trainer simulated no FP8 cache. Standalone vLLM scripts passed `bfloat16`
  explicitly and never showed it.

Rules:

1. **Hash what was imported.** On every rank, after the model is built, take the source file of each model class with
   `inspect.getsourcefile` (this resolves the `transformers_modules` copy) and compare its hash with the run tuple; fail the run on a
   mismatch. Rebuild the model directory from the pinned commit whenever trainer code changes, and keep the directory's file hashes in
   the tuple.
2. **Count calls on the numerical paths that matter.** Each special path (quantization simulation, KV-cache simulation, R3 replay in
   forward and in recompute, activation QAT) increments a per-rank counter; at the first training step every rank checks each counter
   against the count the layer layout implies and fails the run if one is zero or wrong. Model A Megatron rank 0 per step: 63 FP8-KV
   writes, 142 NVFP4 input quantizations per expert projection, 474 FP8 activation quantizations, 71 R3 replays in forward and 71 in
   recompute. A proof is itself code to test: plant a missing path once and see it fail.
3. **Write every dtype and quantization switch explicitly** (`kv_cache_dtype`, MoE activation scheme, LoRA kernel options) and read the
   resolved value back from the engine after the first generate (Ray forwards worker lines late, and deduplicates identical lines across
   workers unless `RAY_DEDUP_LOGS=0`, so per-worker counts need it). Logs can mislead: this
   vLLM version printed an explicit `bfloat16` KV cache as `kv_cache_dtype=auto`, so the check used the cache's bytes per token (FP8 47,136
   vs BF16 78,384 for Model A).
4. **Fail closed.** A check that cannot run (the expected spec is missing on a worker, a rank skipped it) is an error, not a skip;
   Model A's remote-code check at first passed silently on workers that had not received its spec.
5. **Suspect the switch first** when an A/B on a numerical switch shows almost no difference.

## "Latest" must carry the target branch

Pin each repo's target branch per the [delivery target](../../practices/code-and-delivery.md#8-three-repository-delivery-from-experimental-implementation-to-three-prs-and-one-image); a reference PR is a design reference and does not automatically become the PR base. When the user specifies a different fork/branch, record the override. fetch is not checkout; submodule update usually follows the parent repo's gitlink, which is also not the same as every submodule being at its own remote latest.

```bash
git -C /explicit/repo status --short --branch
git -C /explicit/repo remote -v
git -C /explicit/repo fetch origin
git -C /explicit/repo rev-list --left-right --count A...B
git -C /explicit/repo log --left-right --oneline A...B
git -C /explicit/repo merge-base A B
git -C /explicit/repo diff --stat A B
git -C /explicit/repo submodule status --recursive
```

The left/right unique commit counts of `A...B` answer "how many apart on each side"; the common ancestor and actual diff answer "what differs". When branches diverge, there is no single ordering. Neither commit time nor short hash magnitude can determine ancestry. If a shallow clone can't find the ancestor, fetch more history first; do not claim the two lines are entirely unrelated.

By default keep the compatible combination declared by the recipe; if the user explicitly asks to update a particular development branch, record this deviation from the pin and check the actual changes and ABI. Do not assemble mutually incompatible releases just so every item shows "latest".

## Boundary between overlay and native wheel

During development you must first consult your team's container code sync and build-and-test procedures,
and record the commit of the tooling actually used in the runtime manifest.
Use "modify in the host worktree, execute in the container"; do not maintain a separate overlay stack detached from the official source:

1. First confirm the native image/checkpoint's config, model registration and import identity; pin the image
   digest, full source commit and real install directory; do not guess missing implementations from the model name or tag.
2. The `HEAD` of the dedicated sync worktree must equal the image's full commit, with changes under test as
   a diff on that base; they may be ported from the official development branch, but do not bypass the check with a different HEAD.
3. In a container not yet running workloads, sync all changed in-package Python files one by one; before copying, verify
   the original file hash against the `HEAD` version; after copying, verify the hash of the file under test; save the sync manifest.
   For new files, verify neither side has an original file; never overwrite whole packages or modify a running container.
4. Keep the image's versions, generated files and unchanged extensions; for native/binding changes, recompile and
   verify the artifacts; for JIT changes, check compile inputs and caches; if ABI/toolchain changes, rebuild the image.
5. After startup, recheck the actual worker's import/library paths and test. Finally, build the unified image from the official PR's clean
   commit and accept it; the user is not required to redo the development sync; details in
   [implementation and delivery](../../practices/code-and-delivery.md#8-three-repository-delivery-from-experimental-implementation-to-three-prs-and-one-image).

The overlay notes below are only for development/diagnosis, not the final delivery format.

- An overlay is "overwriting the corresponding files in the container with a few external files"; a wheel is an installed Python package that may also carry compiled machine code.
- Prefer overwriting only the necessary files at exact target paths; do not mount a whole repo over the installed package and hide existing extensions. First verify the target file's original hash; apply once at startup and stop on failure; save the overlay manifest, final hashes and import paths.
- Python-only changes can also change binding/JIT/shape contracts; changing only Python does not automatically mean no recompilation or testing is needed.
- Changes to CUDA/C++, bindings, compile macros, submodule kernels, or CUDA/Torch ABI need checking and recompilation as needed. Do not jump to a conclusion from a "missing extension" symptom: first check for the wrong venv, a whole-repo mount shadowing, search paths and dynamic dependencies.
- If the image already has the correct artifact, verify and use it directly. Do not recompile a DeepEP already proven usable; also do not only change the recipe's source pin and assume the old `.so` will change.
- Do not mechanically fix a FlashAttention extension load failure by `unset`ting some variable. Accept a config fix only after proving the variable wrongly selected an unneeded/incompatible path and that restoring the user's target path passes.
- Keep observability overlays separate from numerical/performance changes, off by default; when cleaning up the main PR, do not delete still-useful diagnostic code without a backup.

## Delivery gate for immutable images

1. Commit production changes to the corresponding PR; freeze a clean source commit, all native/JIT build inputs and dependencies; record the base image digest, not just the tag. A dirty patch can identify a diagnostic candidate but cannot be an uncommitted final runtime dependency.
2. Build the final image and check import/dependencies/hashes in a new container with **no development source mounts and no inherited JIT cache**. Device, model and data mounts required for production remain.
3. Run the agreed correctness, Graph/cache and performance gates on that image; if observability is enabled, re-accept with the official probe-free config.
4. Publish exactly the same accepted artifact and verify the registry digest/architecture; if format conversion or repackaging changes content, verify the equivalent closure and do a final smoke.
5. Sync the recipe, PR description and image receipt: failed items cannot be covered by "image pushed successfully". When the user asks to push after testing, follow this order.

How the image is produced is itself part of its identity; the receipt states: whether a Dockerfile rebuilt it from a clean commit, or changes were installed in a container which was then committed and exported as `.sqsh` (when the coordinating machine has no docker permission only the latter is possible — acceptable during development, but formal delivery must return to the former); which registry it was pushed to (Nemotron pushes to public Docker Hub); the tag carries the vLLM/Megatron/verl three-repo SHAs, not just a date. Under the same tag, record both the `.sqsh` SHA256 and the registry digest.

Adding a "new project source layer" on top of an old OCI base does not automatically bring in another experimental image's vLLM/FA4 changes. A deleted file not being in the top-layer tar does not prove it is absent from lower layers; deleting a file in the base requires a correct whiteout, or verifying its absence in the final rootfs. A concrete counterexample was seen in an earlier model's image.

Upgrading FlashInfer/NVSHMEM should not incidentally upgrade Torch/CUDA unintentionally. Pin the target interpreter and constraints, compare packages/dynamic dependencies before and after the upgrade, and confirm the APIs used and build compatibility. A version number by itself is not evidence of an HCA fix or model correctness.
