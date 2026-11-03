# Checkpoint, resume and memory budget

Start here to learn about and troubleshoot checkpoint save, load and resume correctness in long RL runs, as well as host (CPU) memory peaks and growth. For per-phase GPU memory peaks see [04 Locating GPU OOM by lifecycle](04-gpu-oom-and-staging.md); for LoRA save/restore requirements see the [LoRA guide](../lora/README.md#5-save-and-resume-must-also-be-measured).

## Find by symptom

| Symptom | Where to go |
| --- | --- |
| Save reports `unexpected pos` / EFAULT (`Bad address`) / `CheckpointException` | [03 §1–2](03-pitfalls.md#1-unexpected-pos-is-a-masked-efault) |
| Killed by Ray or cgroup OOM during or after save | [03 §3–5](03-pitfalls.md#3-killed-by-ray-after-writing-a-checkpoint) |
| OOM while loading a checkpoint | [03 §9–10](03-pitfalls.md#9-optimizer-load-duplicates-anonymous-memory) |
| Train-inference mismatch on the first step after restore | [03 §13](03-pitfalls.md#13-first-step-after-restore-uses-stale-quantization-scales) |
| Some ranks stalled, GPU around 3%, stack stopped in `cudaMemcpy*Async` | [03 §18](03-pitfalls.md#18-large-pageable-copies-hang-in-libcuda) |
| Resume is configured but training starts from step 0 | [03 §14](03-pitfalls.md#14-resume_modeauto-starts-from-step-0) |
| Host memory jumps at the first sleep and does not come back after wake | [03 §11](03-pitfalls.md#11-sleep-backup-in-torch-pinned-memory-rounded-up-and-never-returned) |
| Restoring offloaded frozen weights takes minutes on some steps; anonymous memory grows by the restored size | [03 §19](03-pitfalls.md#19-file-backed-offload-of-frozen-weights) |
| Multi-node Megatron resume from node-local disk fails; ranks other than 0 resume with rank 0's RNG | [03 §14](03-pitfalls.md#14-resume_modeauto-starts-from-step-0), [01 §6](01-checkpoint-and-resume.md#6-how-to-prove-a-resume-is-correct) |
| Host RSS rises every step | [02 §3.3](02-host-memory.md#33-locating-slow-per-step-growth) |
| Planning host memory for a checkpoint in the hundreds of GB | [02 §4](02-host-memory.md#4-capacity-planning) |
| vLLM fails to start under training colocation (insufficient KV budget) | [02 §5](02-host-memory.md#5-gpu-memory-trainingrollout-colocation-is-time-sharing-compute-per-phase); for MoE via verl first confirm vLLM's EP is not disabled ([recipe alignment §3](../recipe-evaluation/01-recipe-alignment.md#3-read-the-final-cli-take-validated-values)) |
| Under colocation the math says "GPU memory does not fit"; want a larger TP, more GPUs, or to split inference and training onto separate nodes | [02 §5.1–5.3](02-host-memory.md#5-gpu-memory-trainingrollout-colocation-is-time-sharing-compute-per-phase) |
| Choosing sleep level 1 or level 2; how much remains on the GPU after sleep | [02 §5.4](02-host-memory.md#54-sleep-level-1-and-level-2) |
| GPU OOM during initialization / export / update / wake | [04](04-gpu-oom-and-staging.md) |
| Save staging eats hundreds of GB of host memory; cannot reach Ray across steps | [05 §1](05-cluster-notes.md#1-container-tmp-is-a-per-container-private-tmpfs) |
| The checkpoint from before a crash was deleted by rotation | [05 §5](05-cluster-notes.md#5-rotation-retention-policy) |
| Job exits with code 1 and no traceback; ENOSPC writing a small file | [05 §6](05-cluster-notes.md#6-shared-nfs-it-fills-up-and-it-can-be-overwhelmed): shared NFS is full; write runtime output to node-local disk |
| Long run stuck at a step, GPU utilization 0, no error | [06 §1–3](06-unattended-long-run.md#1-what-went-wrong): preserve the scene, cancel, auto-resume |
| Changing trainer code or config in the middle of a long run | [06 §5](06-unattended-long-run.md#5-changing-code-during-the-run) |
| Hold job about to expire; a requeued job cannot start steps; nodes lost | [06 §6](06-unattended-long-run.md#6-holding-the-allocation), [§7](06-unattended-long-run.md#7-node-failures) |
| Time limit hit and nobody resumed; auto-resume submitted the wrong step | [06 §2](06-unattended-long-run.md#2-what-the-daemon-does), [§4](06-unattended-long-run.md#4-the-daemon-scripts-own-pitfalls) |

## Reading order

1. [01 Checkpoint and resume](01-checkpoint-and-resume.md): three independent contracts, which state to save, save and commit protocol, loading, derived state after restore, how to prove resume is correct, rotation and disk.
2. [02 Memory budget: host and GPU](02-host-memory.md): who occupies memory, whether cgroup or Ray does the killing, how to measure, capacity planning, per-phase GPU memory accounting under colocation, sleep level trade-offs.
3. [03 Pitfalls](03-pitfalls.md): incidents grouped by save / load / sleep-wake / resume / long-run growth, each with symptom, cause, fix, verification and source.
4. [04 Locating GPU OOM and staging peaks by lifecycle](04-gpu-oom-and-staging.md): minimum to record, peaks set by objects alive at the same time, why wake still has no GPU memory when reserved is high, telling a fix from a downgrade.
5. [05 Cluster pitfalls](05-cluster-notes.md): container `/tmp` tmpfs, Lustre page cache charged by Ray, direct Lustre write EFAULT, Lustre client cache cap, rotation retention policy, shared NFS; one-line index of other cluster pitfalls.
6. [06 Unattended long runs](06-unattended-long-run.md): on hangs, time limits, node loss or an unreachable login node, a daemon on the cluster preserves the scene, stops, and resumes from the latest checkpoint; the daemon script's own pitfalls; changing code at a checkpoint boundary; holding the allocation; node failures.

For errors go straight to 03; when designing save and resume for a new model read 01 and 02 first; for what looks like an environment failure scan 05 first; read 06 before submitting a formal long run.

## Terms

- **pinned memory:** page-locked host memory the GPU can DMA directly; obtained via `cudaHostAlloc` or `cudaHostRegister`, and cached by PyTorch's caching host allocator.
- **cgroup:** the job's memory limit is `memory.max`; `memory.stat` splits usage into anon (anonymous), file (page cache) and shmem; `memory.current` is total usage.
- **RSS and PSS:** RSS is process-resident memory, with shared pages counted in full in every process; PSS splits shared pages by the number of sharing processes, so only the sum of PSS across processes approximates the physical total.
- **DCP / Megatron dist_ckpt:** PyTorch Distributed Checkpoint and Megatron's distributed checkpoint format; each rank writes its own shard, and `.metadata` is written last.
- **staging / D2H:** before saving, GPU tensors are copied into a host (device-to-host) buffer and then written to disk; that buffer is the staging.
- **Ray object store (plasma):** the shared memory region Ray uses to pass objects between processes, mapped into each worker and counted via PSS split.
- **bounce buffer:** when a disk write fails, an intermediate buffer where the data is first copied into ordinary (pageable) memory and then written.
- **old-logprob:** the logprob recomputed on the training side for rollout tokens with the pre-update policy, used to compare against rollout-side logprob for train-inference consistency.
- **Ray OOM threshold:** `RAY_memory_usage_threshold`; the Ray memory monitor kills workers when usage exceeds this fraction of the limit; default 0.95.

## Key rules

- Split "the checkpoint is broken" into three separate checks: files, memory, derived state. Complete files can still be inconsistent on the first step after restore.
- Capture the first exception (errno, rank, file, offset); do not fix the wording of the last error. `unexpected pos 704 vs 598` is a masked EFAULT; the kernel root cause of EFAULT is still not located, and everything existing is a fallback (publish from local disk, write into owned bytes, chunked retries at the lowest-level `write()`), so do not claim it is fixed; retries that wrap only one layer of calls miss writes at `flush` / `close`.
- Write the completion marker last; clear the old directory before overwriting; all ranks check together and fail together; a missing file must raise, not silently fall back to a new optimizer.
- Under EP4 + FSDP2, each rank's expert shards in DCP have the same key and same offset, so restore cross-reads other ranks' experts; keys must include the EP rank (port the existing upstream fix that adds it), add a 4-GPU bitwise read-back unit test, and raise directly on EP size mismatch.
- Rotation only deletes directories within the tracker that are complete and not being written; long runs keep one copy every 20 steps outside rotation ([05 §5](05-cluster-notes.md#5-rotation-retention-policy)).
- Colocating training and rollout is time-sharing: account GPU memory separately for the rollout phase and the training phase, do not add them; the constraint that varies with TP and layout is the sleep backup in per-node host memory. Before saying "it does not fit", name which item of which phase exceeds the limit ([02 §5](02-host-memory.md#5-gpu-memory-trainingrollout-colocation-is-time-sharing-compute-per-phase)).
- sleep level 1 backs weights up to host and they are usable right after wake; level 2 discards weights and requires a full reload after wake. Neither level frees everything; read the worker logs for the residue ([02 §5.4](02-host-memory.md#54-sleep-level-1-and-level-2)).
- Set rollout `gpu_memory_utilization` from the measured residency of the training process during rollout, written into the recipe per training mode, not as a global default ([02 §5.2](02-host-memory.md#52-the-two-ledgers)).
- Do not default the staging directory to `/tmp`: inside a container it is tmpfs and eats job memory directly; have the launch script pass a local disk explicitly.
- Shared NFS only holds inputs read at startup: runtime output and checkpoints go to node-local disk; before submitting, check free space on the shared disk and each node's local disk ([05 §6](05-cluster-notes.md#6-shared-nfs-it-fills-up-and-it-can-be-overwhelmed)).
- In long runs a daemon on the cluster handles "hang or time limit → preserve the scene → cancel → resume from the latest checkpoint", not a human watching and not a local ssh session ([06](06-unattended-long-run.md)).
- Fixes to save problems must pass the [01 §3.2 acceptance](01-checkpoint-and-resume.md#32-reproduce-and-discriminate); a single success does not count. Before a long run, a new save path or layout first runs to the first save point and then does one kill→resume.
- Evidence of correct resume: per-rank summaries match in the new process, the first export is byte-identical to the save, the first old-logprob aligns, followed by one non-zero update. Setting only `resume_mode=auto` does not count.
- Look at host memory split into anon/file/shmem: `memory.current` includes page cache, and Ray kills based on it.
- Pinned buffers get rounded up to powers of 2 by the caching host allocator and are cached, not returned, after release: Nemotron optimizer offload with 88.2 GiB of state occupied 146.9 GiB pinned; Model A's vLLM sleep backup held 578 GiB per node for 341 GiB of weights after wake. Large pinned blocks are allocated at exact size and either released or deliberately retained ([03 §4](03-pitfalls.md#4-pinned-buffers-rounded-up-to-powers-of-2), [§11](03-pitfalls.md#11-sleep-backup-in-torch-pinned-memory-rounded-up-and-never-returned)).
- The save step's peak is closest to the limit; raising the Ray threshold or reducing microbatch does not solve a headroom problem.
- For per-step growth, find the source from the per-process PSS floor, and reproduce the mechanism in a small CPU-only process before changing code.
- Releasing host caches after save must be verified per path ([03 §2](03-pitfalls.md#2-efault-root-cause-unknown), [§11](03-pitfalls.md#11-sleep-backup-in-torch-pinned-memory-rounded-up-and-never-returned)).


## Evidence sources

| Source | Content |
| --- | --- |
| Three DS4 full32 failure reports ([DeepSeek-V4-Flash historical sources](../../../cases/deepseek-v4-flash/README.md#historical-failure-reports)) | Lustre EFAULT / `unexpected pos`, optimizer load OOM, stale FP8 scale after restore; PRs #195/#196/#197 on a community Megatron-LM fork (PRs are for looking up implementation; merge status must be verified live) |
| Nemotron-H 2026-10 reports ([`baseline-ckpt-crash.md`](../../../cases/nemotron/reports/baseline-ckpt-crash.md), [`host-mem-growth.md`](../../../cases/nemotron/reports/host-mem-growth.md), [final delivery README](../../../cases/nemotron/results/final/README.md)) | pinned rounding, page cache, staging cache, EFAULT fallback, commit protocol, mimalloc, dynamo; fixes in the internal Megatron-LM and verl forks; delivery overview in the [Nemotron case](../../../cases/nemotron/consistency-case.md#2026-10-lightning-nvfp4-true-on-policy-delivery) |
| DS4.1 training history, inference history, restore contract, step28 saved state dtype (raw history archives, not included) | FSDP2 resume state and verification, safetensors errno 14, duplicated CPU offload pinned memory, sleep pinned cache, resume after node failure |
| Model E training lessons, Model B long training and restore, Model F FSDP memory (case notes) | offload stacked with sleep backup, missing tracker, Ray guard |
