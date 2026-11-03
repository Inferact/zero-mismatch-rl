# Environment, images and runtime version provenance

Answers two questions: how to set up the environment (image pins the native runtime, uv manages training dependencies, source goes through a traceable snapshot or minimal overlay), and once set up, how to prove that is what is running (code existing does not mean it is executing).

## Find by symptom

| Symptom / question | Go to |
|---|---|
| The image has the model implementation, but VERL's dependency lock is incompatible with it | [01 §1–3](01-environment-setup.md) |
| Lock consistent, import succeeds, GPU operators still error | [01 §2 visible dependencies do not mean ABI compatibility](01-environment-setup.md#2-visible-dependencies-do-not-mean-binary-compatibility) |
| Unsure which version of source / wheel / .so / image is actually running | [02 runtime manifest](02-runtime-provenance.md#use-one-runtime-manifest-to-answer-what-exactly-is-running-now) |
| A numerical switch makes no difference in an A/B; a remote-code model may be importing an old copy; `auto` dtype resolved to something unplanned | [02 prove the numerical path ran](02-runtime-provenance.md#prove-the-numerical-path-ran) |
| How many commits "latest" is actually behind the target branch | [02 "latest" must carry the target branch](02-runtime-provenance.md#latest-must-carry-the-target-branch) |
| Overlay, or modify source and rebuild | [02 boundary between overlay and native wheel](02-runtime-provenance.md#boundary-between-overlay-and-native-wheel) |
| Recompiles on every startup / unsure where the cache should go | [03 §1–2](03-cold-start-and-cache.md#2-compile-caches-where-they-go-decides-whether-they-can-load) |
| `failed to map segment from shared object` during `torch.compile` | [03 §2 /dev/shm noexec](03-cold-start-and-cache.md#2-compile-caches-where-they-go-decides-whether-they-can-load) |
| Multi-GPU vLLM on the same node hangs at kernel warmup on restart, no progress and no error | [03 §2 one cache per rank for multiple ranks on the same node](03-cold-start-and-cache.md#2-compile-caches-where-they-go-decides-whether-they-can-load) |
| Loading weights on Lustre sits at 0% for a long time, prefetch doesn't help | [03 §3](03-cold-start-and-cache.md#3-weight-loading-two-layers-of-mmap-pitfalls-on-lustre) |
| Kernel package has no prebuilt wheel and won't build in the image | [03 §4](03-cold-start-and-cache.md#4-building-third-party-kernel-packages-from-source) |
| Unsure which path DeepEP actually runs / how it was built | [02 four things for communication implementations](02-runtime-provenance.md#communication-implementations-do-not-use-one-name-for-four-things) |
| Can the Bridge/Megatron version be upgraded | [01 §4 version pins](01-environment-setup.md#4-version-pins-must-go-into-the-runtime-manifest) |
| Moving a run to another cluster or GPU architecture | [01 §7](01-environment-setup.md#7-bringing-a-run-up-on-another-cluster) |
| Spaces and newlines all gone from training text, yet train-inference consistency is fine | [01 §6 do a tokenizer round-trip check before training](01-environment-setup.md#6-do-a-tokenizer-round-trip-check-before-training) |

## Key rules

1. First pin the image digest, CPU architecture and Python version, then resolve training dependencies; image tags, PyPI version numbers and upstream PR merge status cannot replace a minimal inference run inside the image.
2. `--system-site-packages` does not stop uv from installing a same-named package that is imported first; check the install plan, actual import paths and GPU operators together.
3. Each of the six layers of the runtime manifest (source, installed packages, native libraries, JIT kernels, container, run) has something nothing else can replace: `latest` cannot replace a digest, `pip show` from another venv cannot replace the actual import path, the Python source HEAD cannot replace the .so actually loaded.
4. Audit inside the actually running worker, interpreter and mount namespace; a shell logged into the container and a Ray worker may differ. A version check is not an execution check: hash the files each rank actually imported (`trust_remote_code` imports the model directory's copy) and count calls on every special numerical path at step 1 ([02](02-runtime-provenance.md#prove-the-numerical-path-ran)).
5. The same image tag may point to different digests on different dates/nodes; ARM64 and AMD64 tags may be independent builds.
6. "Modify in the host worktree, execute in the container": same-commit worktree + per-file hash verification; do not maintain an overlay stack detached from the official source.
7. Version pins such as "the Megatron fork is only compatible with Megatron-Bridge v0.6.2" must be written into the runtime manifest.
8. How the image is produced (committing a container and exporting sqsh vs rebuilding from a Dockerfile), where it is pushed, and the three-repo SHAs in the tag are all part of the run identity.

## Files

- [01-environment-setup.md](01-environment-setup.md): container + uv environment configuration; §4 version pins (Bridge v0.6.2); §6 tokenizer round-trip check; §7 bringing a run up on another cluster.
- [02-runtime-provenance.md](02-runtime-provenance.md): environment identity and runtime manifest; proving the numerical path ran (imported-file hashes, per-rank call counts, explicit dtypes); which four things to record for communication implementations (table in [debugging 05](../debugging-mismatch/05-comm-fault-ladder.md)), overlay vs wheel boundary, immutable image delivery gate and how images are produced.
- [03-cold-start-and-cache.md](03-cold-start-and-cache.md): cold start and caches: compile cache cannot go in /dev/shm (noexec); safetensors prefetch on Lustre only helps for ≲50 GB, client cache cap 54 GB, copy large models to /dev/shm for loading; the three sticking points and gating checks for building third-party kernel packages from source; two enroot import pitfalls.

## Evidence index

- Nemotron-H: [case](../../../cases/nemotron/README.md) (image commit export, Bridge version pin, three-repo SHA tag).
- Cluster-level enroot/pyxis/Lustre/NCCL facts belong to cluster operations notes; a one-line list is in [checkpoint guide 05 cluster pitfalls](../checkpoint-resume-memory/05-cluster-notes.md).
