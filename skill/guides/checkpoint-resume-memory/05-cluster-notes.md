# Cluster pitfalls that get mistaken for checkpoint or memory problems

This page expands only the four cluster facts that directly change the checkpoint save path or the memory ledger, plus one on rotation retention and one on shared NFS; other cluster pitfalls get one line each. Cluster paths and job IDs do not go here.

## 1. Container `/tmp` is a per-container private tmpfs

The pyxis/enroot fstab has `tmpfs /tmp tmpfs ...`: every container start creates a fresh private tmpfs as `/tmp`. `container_scope=job` only applies to named containers; srun steps without `--container-name` each start a new container.

- **Memory ledger:** when the checkpoint staging directory defaults to `/tmp`, hundreds of GB of staging land directly in job memory (about 235 GiB for Nemotron), showing up as a Ray kill during save. Following DS4, remove the default from code and have the launch script pass a node-local disk explicitly (mLite: `MLITE_DCP_LOCAL_STAGE_DIR`); see [03 §7](03-pitfalls.md#7-container-tmp-used-for-staging).
- **Cannot reach Ray across steps:** after `ray start --head --temp-dir=/tmp/ray-...`, a driver in another srun step, even on the same node, cannot open the raylet/plasma unix socket paths returned by GCS, and `ray.init` always fails. Put the driver and head in the same step/container, or bind a host directory as a temp-dir shared by both. `/dev/shm` is bound from the host and shared across containers, but the sockets are not there.
- Container `/tmp` is not noexec, so it can hold compile caches, but every new container recompiles; see [environment 03 §2](../environment-provenance/03-cold-start-and-cache.md#2-compile-caches-where-they-go-decides-whether-they-can-load).

## 2. Lustre page cache is charged by Ray

After a 410 GB checkpoint is written to Lustre it stays in page cache and is charged to the job cgroup's `memory.current`; that is exactly what the Ray memory monitor watches, and it then kills the worker. The fix is `fsync` + `posix_fadvise(DONTNEED)` on new files of completed directories, applied only to completed directories within the tracker ([03 §3](03-pitfalls.md#3-killed-by-ray-after-writing-a-checkpoint)). Verify via the file entry in `memory.stat`, not `free`.

## 3. Direct Lustre writes hit EFAULT

The torch ZIP writer writing directly to Lustre reports `inline_container.cc unexpected pos`; the real error is EFAULT returned by `write()`. The kernel root cause is not located, and all fixes are fallbacks ([03 §1–2](03-pitfalls.md#1-unexpected-pos-is-a-masked-efault)). The path-level countermeasure is to write to node-local disk first and then publish to Lustre ([01 §3.3](01-checkpoint-and-resume.md#33-reliable-publishing)); paths that still write Lustre directly (Megatron dist_ckpt) retry in chunks at the lowest-level `write()`. Node crashes have happened during writes; Nemotron SWE 256K run also had two node heartbeat losses right next to the process's first save ([03 §2](03-pitfalls.md#2-efault-root-cause-unknown), cause not located), and on one GB200 cluster six Model A nodes died while large writes went to Lustre through the client's direct-IO path ([06 §7](06-unattended-long-run.md#7-node-failures)); write to Lustre in pieces of at most 1 MiB, and the publish step must be rerunnable from a half-finished state: temporary name, verification, same-filesystem rename, completion marker written last.

## 4. Lustre client cache cap is about 54 GB

`llite.max_cached_mb` is about 54 GB. When reading weights, safetensors mmap prefetch only helps models ≲50 GB; larger ones sit at shard 0 for tens of minutes; copy to `/dev/shm` and then load. Details in [environment 03 §3](../environment-provenance/03-cold-start-and-cache.md#3-weight-loading-two-layers-of-mmap-pitfalls-on-lustre). Effect on checkpoints: reading optimizer shards on resume also goes through this cache; `torch.load(mmap=True)` saves anonymous memory, not page cache ([03 §9](03-pitfalls.md#9-optimizer-load-duplicates-anonymous-memory)).

## 5. Rotation retention policy

In a Nemotron long run, checkpoint rotation deleted the copy from just before the crash, so the crash could not be replayed. Rules:

- Rotation only deletes directories that are within the tracker, complete and not being written, and deletion records are persisted to disk ([03 §6](03-pitfalls.md#6-no-checkpoint-rotation-fills-the-disk)).
- Long runs keep one copy every 20 steps outside rotation, so a crash or score anomaly can be replayed from a retained point.
- Rotation may already have deleted a step you want to evaluate; do not fill it in with a checkpoint from another chain ([01 §7](01-checkpoint-and-resume.md#7-rotation-disk-and-timing)).

## 6. Shared NFS: it fills up, and it can be overwhelmed

On some clusters (for example one GB300 cluster) the only shared filesystem is a single NFS (about 14T); home and shared software directories are on it, the whole cluster uses it, and it sits near 100% for long periods. Whether it is full depends on others and cannot be predicted.

- **Full disk kills jobs.** The Model D formal training run hit ENOSPC at step 20 while writing a 2 KB sync receipt to the results directory, and the whole job failed; that job itself used only 41 MB on the shared disk. Another LoRA training run's first checkpoint save (LoRA 34 GB + Adam state 68 GB, about 100 GB) filled the shared disk down to 1 MB free; the job exited with code 1 and no traceback, the checkpoint was truncated and useless, and during the few minutes it was full other people's jobs on the same filesystem may have been affected too.
- **Overwhelms everyone.** It is a cluster-wide single point; multiple nodes with every rank writing to it at once (logs, rollout dumps, compile caches, venv builds, model downloads) slow the server down, and login nodes and other people's jobs stall along with it.

Rules:

1. All runtime output (trainer logs, metrics, receipts, rollout dumps, checkpoints) goes to compute-node local disk (e.g. `/raid`, mounted as `/out` in the container). The shared disk only holds inputs read at startup and small coordination files between nodes. Experiment records and their provenance sidecars (`results/<run-id>/`) are also written locally first, then mirrored per rule 2.
2. To watch progress from the login node, have one process mirror local files to the shared disk best-effort every minute, ignoring failures; failures writing audit or receipt files must not interrupt training.
3. Before submitting a job that writes large files, `df -h` the shared disk and the local disk of every allocated node; add a free-space floor check in the launcher and do not start if there is not enough. Node-local disks fill up too (during Model D training one node's `/raid` was seen at 98% used).
4. When estimating checkpoint size, include optimizer state (Adam has two states with the same shape as the parameters), multiplied by the number of retained copies.
5. When a job exits with "code 1, no traceback", check disk first.
6. When checkpoints are on node-local disk, resume requires the rank-to-node mapping to stay unchanged; check per [03 §14](03-pitfalls.md#14-resume_modeauto-starts-from-step-0).

## 7. Other cluster pitfalls (one line each)

These belong to cluster operations notes; only an index is kept here, so that "environment failures that look like code bugs" can be ruled out at a glance:

| Symptom | One line |
| --- | --- |
| `enroot import` of a large image runs over an hour with no `.sqsh` | `ENROOT_TEMP_PATH` is on Lustre; overlayfs does not accept a network FS as lowerdir; `ENROOT_CACHE_PATH` can be there |
| `enroot import docker://` gets 401 at 20 minutes | Bearer token expires after 20 minutes; large images take the local tar fallback path, first replicate the enroot frontend config block |
| Converted image loses GPUs / Megatron refuses to start | The whole NGC env is lost: without `NVIDIA_VISIBLE_DEVICES` the nvidia hook does not run, without `TORCH_CUDA_ARCH_LIST` Megatron raises `RuntimeError`; diff `/etc/environment` first |
| `enroot start IMG bash -c ...` reports `--compilation-config: Invalid JSON` | The vllm-openai image entrypoint is `vllm serve`, which swallows the command as arguments; bypass with an `--rc` script doing `exec "$@"` |
| NCCL `unhandled system error` in a GB200 container | The container has no IMEX channel; set `NCCL_MNNVL_ENABLE=0 NCCL_CUMEM_ENABLE=0` |
| Resubmitted job dies instantly with `Free memory on device (23/184 GiB)` | A crashed job's GPU memory is freed slower than Slurm schedules; the launcher needs a GPU-clean barrier (>2 GiB/GPU counts as dirty) |
| `srun ... bash <script>` exit 127 | The script is in the login node's `/tmp`/scratchpad, invisible to compute nodes; scripts executed by srun must be on a shared filesystem visible to compute nodes (Lustre; NFS on GB300) |
| Multi-node vLLM fails to start with `No module named 'ray'` | The image has no Ray; use `distributed_executor_backend="external_launcher"`, pitfalls in [rollout engine §5](../rollout-engine.md#5-lifecycle-and-gpu-memory-ledger) |
| Multi-node NCCL init hangs forever (all ranks pass pre-init, none pass post-init), 2-node test works | A node's BMC NIC (e.g. `usb0`) is chosen by NCCL as the bootstrap address and other nodes cannot reach it; for all multi-node jobs set `NCCL_SOCKET_IFNAME` and `GLOO_SOCKET_IFNAME` to the data NIC |
| Ray worker cannot connect to GCS (`Failed to connect to GCS at <hostname>:<port>`), port is reachable | Ray's gRPC cannot resolve the node hostname; use the head node's IP for `ray start --address` (`getent hosts`) |
| Ray is up, no actor for over ten minutes, logs show `process_failed_pending_registration` | Ray prestarts workers per CPU count by default (144 cores means 144); with slow imports each one times out registering; set `ray_init.num_cpus` small (e.g. 32), `RAY_worker_register_timeout_seconds=900`, and pass slow-import-related env vars to actors only via `ray_init.runtime_env`, not in the `ray start` environment |
| NCCL `Duplicate GPU detected`, all FSDP ranks land on the same GPU | sitecustomize imports the model early, so CUDA initializes before Ray sets each actor's `CUDA_VISIBLE_DEVICES`; do not put model registration in sitecustomize ([Code and delivery](../../practices/code-and-delivery.md)), and existing sitecustomize asserts `torch.cuda.is_initialized()` is False after running |
| verl worker refuses to start: `Please don't set ROCR_VISIBLE_DEVICES` | Slurm resets this variable in every srun step, so `unset` in the batch script does nothing; unset it inside srun, before `ray start` |
| `sbatch` reports `Unexpected message received` | The job may already be queued; confirm with `squeue -u $USER` before deciding whether to resubmit, otherwise you occupy nodes twice |
| Someone else's job dies after cleaning up leftovers | Never `pkill` by name on shared nodes; only start with `setsid` and kill by process group; `ray::WorkerDict`, `VLLM::Worker_DP*_EP*` belong to others by default |
