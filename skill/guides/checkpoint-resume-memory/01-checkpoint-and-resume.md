# Checkpoint and resume

In long RL runs: what a checkpoint must store, how to save it reliably, how to load it without blowing up memory, and how to prove a resume is correct. Host-memory measurement and growth sources are in [02](02-host-memory.md); each concrete incident is in [03](03-pitfalls.md).

## 1. Three independent contracts

Do not lump every save/load/first-rollout problem under "the checkpoint is broken". First decide which kind it is:

| Contract | Typical failure | Example |
| --- | --- | --- |
| File: serialization and publishing | Write error; an incomplete directory is treated as complete | [EFAULT masked as `unexpected pos`](03-pitfalls.md#1-unexpected-pos-is-a-masked-efault) |
| Memory: transient peak during save and load | Killed by Ray after writing; cgroup OOM during load | [Page-cache accounting](03-pitfalls.md#3-killed-by-ray-after-writing-a-checkpoint), [optimizer load duplicates anonymous memory](03-pitfalls.md#9-optimizer-load-duplicates-anonymous-memory) |
| Derived state: first export after restore | Files are complete, but the first step after restore has a train-inference mismatch | [Stale quantization scale](03-pitfalls.md#13-first-step-after-restore-uses-stale-quantization-scales) |

Even with fully intact files you can hit the third kind of bug; it belongs to the lifecycle of derived state, not to file corruption.

## 2. What state exact resume needs

| State | Requirement |
| --- | --- |
| Model parameters or LoRA adapter shards | Save per rank; load strictly, reject silent missing/unexpected keys ([VERL/Megatron integration](../training-model-support/05-framework-adapter.md)) |
| Optimizer state, including the part offloaded to CPU | Same step as the model. Nemotron chunked optimizer offload is 88.2 GiB per rank, 94.7 GB per rank on disk; DS4.1 saves the LoRA parameters and Adam `exp_avg`/`exp_avg_sq` as FP32 |
| LR scheduler | Check the values actually saved, not just the source-code defaults (DS4.1's constant schedule with no warmup was also checked rank by rank) |
| RNG | DS4.1 saves four streams: CPU, CUDA, Python, NumPy |
| Counters and data position | Global step, optimizer step, data cursor; DS4.1 also saves the cumulative count of steps with a valid signal |
| Run contract | Hashes of code, data and recipe. Compare each item on restore; by default reject a change in total steps or data. To extend training, use an explicit new-stage declaration; do not edit the old contract file and do not add a generic ignore switch |
| Completion marker | `.metadata`/tracker (e.g. `latest_checkpointed_iteration.txt`) is written last, see §3.4 |

Not in the checkpoint, must be rebuilt after restore:

- **Rollout engine weights.** After restore, do one full sync first. After DS4.1 restored, the 320 exported files were byte-identical to those at the saved step; the restore sync took about 696–698 s.
- **Parameter-dependent derived caches.** Quantization scales, packed/transposed/fused weights, version tags; see §5.
- **Records outside training.** The metrics jsonl gets overwritten ([03 §15](03-pitfalls.md#15-resume-overwrites-the-metrics-jsonl)); for W&B continuation and curve merging see [long-running operations](../../practices/operations-and-handoff.md#resume-eval-and-wb).

## 3. Save

### 3.1 Capture the first error, do not patch the last error message

Keep the writer callback's original exception/errno, rank, file path, offset, requested/returned byte counts and the exception chain; DCP and Ray wrapping can drop these. Also record the filesystem/mount, free space and quota, and cgroup OOM events; do not look only at the final `ActorDiedError`. The same `unexpected pos` message can also come from disk space, quota, permissions or I/O faults; distinguish them by the first exception.

Megatron dist_ckpt's `_write_item_checked` (internal Megatron-LM fork, `filesystem_async.py`) raises an error carrying the file, offset, item name and the original OSError when an item write fails. A `write()` that does not go through it (e.g. flushing the buffer at `close`) still raises a bare `OSError`; when you see a bare errno, first determine which `write()` it was.

### 3.2 Reproduce and discriminate

A fast repro keeps the real tensors/storage, serialization implementation and concurrency, skips rollout/backward, and only repeats the save. Run a bounded number of rounds sized to the failure probability; stop at the first failure and preserve the scene. In DS4 model-only saving, one path failed only on the third attempt, another on the sixth; Nemotron with host-cache release failed on save 2–4. A single success does not prove a fix.

**Acceptance criterion for fixing a save problem (Nemotron): N consecutive saves + kill→resume→save again.** Everywhere else in this guide refers to this. Also run it before the first long run of a new save path, parallel layout or node count: EFAULT concentrates on a process's first save ([03 §2](03-pitfalls.md#2-efault-root-cause-unknown)), so running only up to the first save point exposes it; no need to wait for the long run to reach that step.

Discrimination order:

1. Same buffer/serializer, Lustre vs. node-local XFS; look at the errno of the first failure.
2. Control concurrency, pinned/pageable, memory peak; if switching filesystem also switched the serializer, the result cannot be attributed to a single factor.
3. Check growth of threads, processes and file handles, and verify it separately from write errors (DS4 example in [03 §8](03-pitfalls.md#8-repeatedly-creating-devicemesh-accumulates-threads)).
4. Only if needed, look at kernel/filesystem client code and patches. Matching a historical Lustre issue is only supporting evidence; it does not mean the underlying issue is fixed.

### 3.3 Reliable publishing

First finish serialization on node-local storage and close the files; copy to a **temporary name on the target filesystem**, handle short copies, flush/fsync according to durability requirements, verify integrity, and finally rename within the same filesystem. Publish metadata/tracker/completion marker only after all ranks finish. A cross-filesystem move cannot be claimed to be an atomic rename. Once an underlying write-path problem has shown up, separately verify the copy from local disk to shared storage too, and check size and checksum after the copy. DS4 implementation hints are in PR #196 of a community Megatron-LM fork; Nemotron mLite DCP (EP4 + FSDP2) uses `MLITE_DCP_LOCAL_STAGE_DIR` to set the node-local staging disk.

### 3.4 Commit protocol: write the completion marker last

| Rule | If you don't |
| --- | --- |
| Write `.metadata`/tracker last, after every rank's shards and auxiliary files (optimizer, RNG, scheduler) succeed and are confirmed collectively | A rank fails, the completion marker already exists, and restore reads incomplete state |
| Before overwriting an existing step directory, delete all its old files first (uncommit) | Leftover old optimizer/RNG gets paired with the new model without any error |
| Before restore, all ranks check file completeness together and fail together if anything is missing | The rank with missing files exits, the other ranks hang until NCCL timeout |
| Ranks with no shard to write still join the same collective | Collective order diverges across ranks and hangs |
| Error when loading is requested but files are missing, not just a log line | Training silently continues with a fresh optimizer |

Acceptance: when a save fails midway (Nemotron actually hit ENOSPC), all ranks exit together, the tracker does not advance, and you can resume directly; new code loads checkpoints saved by old code. When stopping a job after a save, first verify the checkpoint is complete; do not kill mid-write.

### 3.5 Verify a written checkpoint

Check the expected shard set, file sizes / non-empty, leftover temp files, any required checksums, and do one real full restore. Checking only the save summary is not testing restore: DS4.1 periodic checkpoints 20/24 had correct save summaries but no separate new-process restore was done, and the report states this plainly.

## 4. Load: enough capacity does not mean enough for the transient peak

During load the optimizer may be fully duplicated into anonymous memory ([03 §9](03-pitfalls.md#9-optimizer-load-duplicates-anonymous-memory)), and FSDP CPU offload may make another pinned copy ([03 §10](03-pitfalls.md#10-fsdp-cpu-offload-duplicates-pinned-memory-on-load)).

Acceptance loads model + optimizer + RNG/extra at real scale, not just one small shard.

Read a checkpoint with the same image that wrote it: DS4.1's host PyTorch 2.11 could not deserialize the DTensor `_FlatLayout` saved by the image's 2.13; it only read successfully inside the image with `torch.load(weights_only=True, mmap=True, map_location="cpu")`. Do not modify the checkpoint or turn off `weights_only` to make it readable.

## 5. Derived state after restore

DS4 chain: HF initial load saves the FP8 source scales → DCP restores trained BF16 parameters → the first export still uses the old registry → the actor re-quantizes by parameter version, but rollout uses the old scales. Result: the first resumed step mismatches, and after the next optimizer hook clears the cache it matches again. The prediction "it recovers by itself on the next step", plus a before/after comparison on the same checkpoint, pins down causality better than "a diff appears after saving".

Put the fix in the model/load lifecycle that owns the state: after load/update, invalidate or correctly rebuild parameter-dependent caches such as scales, packed/transposed/fused weights and version tags; reuse a suitable existing hook, but avoid side effects such as running a real optimizer update just to invalidate. Do not mask the problem by waiting one training step, disabling resume, or manually clearing all caches. DS4's minimal patch calls the existing post-update hook, see PR #197 of a community Megatron-LM fork. List all parameter-dependent caches and the code responsible for invalidating them, add the missing invalidation calls, and add a regression test.

## 6. How to prove a resume is correct

1. **Before launch.** If the checkpoint is on node-local disk, every node hosting a newly allocated rank has the corresponding shard; the tracker/latest pointer exists and points to a complete checkpoint; the `resume_from_path` and global step actually resolved by the launcher are as expected. Setting `resume_mode=auto` alone is not evidence; confirm the restored step before the first training log line.
2. **New-process restore.** All worker PIDs have changed; each rank's adapter/optimizer/scheduler/RNG digests match those at save time item by item, on every rank, not just rank 0. A distributed checkpoint saves a plain object inside a sharded state dict once, from rank 0: Model A's Megatron path stored the RNG state that way, and after a resume 31 of 32 ranks held rank 0's CPU RNG (51,386 items compared, 31 differed); per-rank state must be saved as a per-rank sharded object, with a fallback that still reads old checkpoints. Include negative controls once: restore without the optimizer and with a reset RNG, and confirm the comparison reports both (Model A: 89–113 optimizer items and 7 RNG items per rank).
3. **First export and old-logprob.** Exported files are byte-identical to those at the saved step; also check the exporter scale/policy fingerprint and the full valid logprobs. Looking only at the second step returning to 0 misses the §5 bug.
4. **One non-zero update.** After restore, complete one update and sync with non-zero gradients. A zero-advantage step only validates the flow: in two DS4.1 resume segments, step 2 had zero gradients on all ranks (see the DS4.1 training history; raw history archive, not included), which cannot prove that newly learned adapter values were restored correctly.
5. **Gate coverage.** step0 cold load, reload of identical weights, first export/old-logprob after a non-zero-step restore, one subsequent update/sync, address/state validity under Graph and sleep/wake; new code reading an old checkpoint.
6. **What acceptance compares.** What the sources actually compared: each rank's restored state digest matches save time (DS4.1); the first exported files after restore are byte-identical to the saved step, and the full outputs and logprobs of two repeated runs after sync are identical (DS4.1); per-step train-inference `probs_diff` after restore, which must be 0 under BI=1 (Nemotron kill→`resume_mode=auto`, first step was 0). The resumed trajectory is not required to be bitwise identical to an uninterrupted run: Nemotron's rollout sampling is not bitwise across processes. The sources give no threshold for "metrics are close"; curves such as reward are only a reference over the same step range and the same measurement.

## 7. Rotation, disk and timing

| Item | Record |
| --- | --- |
| Size per copy and save time | DS4 full32: 99 files, about 3.2 TiB, save about 269 s. Nemotron-H 30B-A3B single node 4×GB200: 353 GB (EP4 + FSDP2) or 429 GB (PP4 dist_opt), save 72–152 s; saving every 5 steps costs about 2–3% of step time. These are records for that layout only, not a standard for other models; write these items for your own model in the plan |
| Rotation | The mLite engine ignores `max_actor_ckpt_to_keep`, see [03 §6](03-pitfalls.md#6-no-checkpoint-rotation-fills-the-disk) |
| Checkpoint for eval | Rotation may already have deleted the step to evaluate; do not substitute a checkpoint from another chain, see [long-running operations](../../practices/operations-and-handoff.md#resume-eval-and-wb) |
| Resuming on a new node after node failure | DS4.1 first copied the checkpoint, dequantization cache, pinned image and dependency caches to the new node, and only started after that succeeded; disabled automatic job requeue to avoid overwriting results; changed only the node selection, and wrote a transfer record |
| Paths for two-arm experiments | BI=0/BI=1 output and checkpoint paths must be separate; changing the run name breaks an in-progress resume chain, see [BI switch recipe](../recipe-evaluation/bi-switch-recipe.md#rules) |
