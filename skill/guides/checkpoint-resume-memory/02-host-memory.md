# Memory budget: host and GPU

In long RL runs, host memory has two kinds of problems: the **peak** during phases such as save/load/sleep hitting the limit, and slow per-step **growth**. They are measured differently (§1–4). GPU memory is a third kind: when the training process and the rollout engine are colocated, the two sides time-share the same GPUs, and the budget is computed separately per phase (§5); for locating GPU OOM by lifecycle see [04](04-gpu-oom-and-staging.md).

## 1. What occupies host memory

| Source | Mechanism | Details |
| --- | --- | --- |
| Pinned buffers for optimizer offload | The torch caching host allocator rounds up to powers of 2, so actual state is tens of percent larger than nominal; summed over ranks, it pushes the resident amount before saving near the limit | [03 §4](03-pitfalls.md#4-pinned-buffers-rounded-up-to-powers-of-2) |
| Parameter shards for FSDP CPU offload | Each rank's parameter shard stays resident on CPU; if load creates a second pinned copy it doubles | [03 §10](03-pitfalls.md#10-fsdp-cpu-offload-duplicates-pinned-memory-on-load) |
| D2H staging during save | Stays in the pinned cache pool and is not returned; whether `torch.accelerator.empty_host_cache()` can release it must be verified per save path (removed from the Megatron dist_ckpt path due to the EFAULT association, see 03 §2) | [03 §5](03-pitfalls.md#5-save-staging-stays-in-the-pinned-cache-pool) |
| Freshly written checkpoint files | Writes to Lustre stay in the page cache, are charged to the job cgroup, and Ray counts them as used memory and kills processes; fsync then drop cache | [03 §3](03-pitfalls.md#3-killed-by-ray-after-writing-a-checkpoint), [05 §2](05-cluster-notes.md#2-lustre-page-cache-is-charged-by-ray) |
| CPU backup for serving sleep | L1 sleep keeps a full backup per rank; after wake the pinned cache is not returned | [03 §11](03-pitfalls.md#11-sleep-backup-in-torch-pinned-memory-rounded-up-and-never-returned), [§12](03-pitfalls.md#12-actor-offload-stacks-with-the-serving-sleep-backup) |
| Per-thread allocator heaps | mimalloc keeps pages in the thread that allocated them | [03 §16](03-pitfalls.md#16-mimalloc-per-thread-heaps) |
| Compile cache | Each dynamo guard failure leaves a small resident record; after hitting the recompile limit it accumulates every step | [03 §17](03-pitfalls.md#17-dynamo-hits-the-recompile-limit) |
| Ray object store | Plasma mappings are charged to each process's PSS as shared pages; node shmem does not grow with it | §3 |
| `/tmp` inside the container | In pyxis/enroot it is tmpfs; when the staging directory defaults to `/tmp` it eats job memory directly. Following DS4, remove the default and have the launch script explicitly specify a node-local disk | [03 §7](03-pitfalls.md#7-container-tmp-used-for-staging), [05 §1](05-cluster-notes.md#1-container-tmp-is-a-per-container-private-tmpfs) |

`param_offload` adds CPU pressure; turning it off uses more GPU; optimizer offload is a separate dimension. Change only one offload switch at a time, and measure the phase peak once before and once after.

Measured sizes for each item (pinned rounding, staging leftovers, sleep backup, breakdown of resident memory) are in the corresponding 03 entries and case reports: Nemotron [checkpoint host memory](../../../cases/nemotron/reports/ckpt-host-oom.md#what-the-740-gib-baseline-consists-of-pp4-original-p6-after-the-first-update-independent-of-load), [BI=0 baseline save crash](../../../cases/nemotron/reports/baseline-ckpt-crash.md); DS4.1 training history (raw history archive, not included).

## 2. Limits and who kills

- **cgroup.** The job limit is `memory.max`, usually below the node's physical memory (Slurm allocates by RealMemory); budget against the cgroup limit, not physical memory. Signs of a cgroup OOM are Slurm `OUT_OF_MEMORY`, exit `0:125`, an `oom_kill` event and a process SIGKILL, with no Python stack.
- **Ray memory monitor.** By default kills workers at `RAY_memory_usage_threshold`=0.95; the log line looks like `exceeded threshold of <bytes>B (95.0% ...)`. The usage Ray looks at includes the page cache. Raising the threshold does not solve a headroom problem: Nemotron set 0.985 and was still killed during save; Model F historically raised it to 0.98 after accounting for headroom, which is a record for that configuration, not a general recommendation (Model F FSDP memory case notes).
- Distinguish three kinds of termination: GPU allocation exception, host cgroup OOM, and an active Ray kill. After CUDA allocator allocation-failure/retry warnings the step may still complete; a warning is not a terminal OOM, but it also does not mean there is no memory pressure.

Concrete limit values (Nemotron single-node cgroup, Ray kill line, DS4.1 node allocation) are in the [ckpt-host-oom conclusion](../../../cases/nemotron/reports/ckpt-host-oom.md#conclusion) and [baseline-ckpt-crash](../../../cases/nemotron/reports/baseline-ckpt-crash.md).

## 3. How to measure

### 3.1 One record per phase

Emit a log line at load, save, sleep, weight update, wakeup, reset KV cache, and generation start/end, with monotonic time, node/rank/PID, and policy version. Per node, record cgroup `memory.current`/peak/events and anon/file/shmem from `memory.stat`; sample process RSS/PSS, thread count and FD count as needed, without high-frequency scans of all processes.

`generate_sequences` includes phases such as wake/refit/init; an error inside this method does not prove token generation had started, let alone that it is a backward/optimizer OOM.

### 3.2 Reading pitfalls

| Reading | Problem |
| --- | --- |
| `memory.current` | Includes page cache, which masks per-step growth; but it is exactly what Ray and the save peak look at |
| `free`, `psutil.virtual_memory().used` | System-visible memory, not process RSS and not job cgroup usage |
| `scontrol` FreeMem | Excludes reclaimable cache |
| `ru_maxrss`, Slurm MaxRSS | Cumulative peak, not current RSS |
| VmPin/VmLck | Cannot be used alone to infer CUDA pinned memory |
| Sum of per-process RSS | Shared mappings are double-counted, so it is not a physical total; use PSS |

Nemotron measurement: cudaHostAlloc pinned memory is counted as shmem in cgroup `memory.stat`.

### 3.3 Locating "slow per-step growth"

1. Look at the node's `MemTotal−MemAvailable` and the cgroup anon/file/shmem to determine which category is growing.
2. Every minute, sample per-process PSS from `/proc/<pid>/smaps_rollup`, recording only comm/argv0; take the per-step minimum as the floor and find processes that rise monotonically. In-phase peaks swing with offload and cannot be used.
3. For rising processes, read `smaps`: large 1 GiB-aligned anonymous mappings are mimalloc arenas (torch CPU tensors), `[heap]` is the Python heap. For other people's or running jobs, only read `/proc`; do not attach and do not send signals.
4. Reproduce the candidate mechanism with a small CPU-only process before deciding what to change.

The floor dropping back after resume (Nemotron run 1 high plateau went from about 455 back to about 433 GiB) shows that the growth accumulates over the process lifetime.

For the Nemotron 100-step long run (per-step growth, which processes grew, the two sources and fixes) see [host-mem-growth](../../../cases/nemotron/reports/host-mem-growth.md); mechanism entries are in [03 §16](03-pitfalls.md#16-mimalloc-per-thread-heaps), [§17](03-pitfalls.md#17-dynamo-hits-the-recompile-limit).

## 4. Capacity planning

The peak budget for a save step sums the parts alive at the same time, then compares against cgroup limit × Ray threshold:

```text
peak ≈ resident pinned optimizer offload (actual usage after allocator rounding, see 03 §4)
     + save D2H staging (and the part left in the cache pool, 03 §5)
     + page cache of newly written checkpoint files (03 §3)
     + serving sleep CPU backup and unreleased pinned cache (03 §11–12)
     + resident amount of other processes (rollout, driver, Ray workers, etc., floor per §3.3)
limit = cgroup memory.max × RAY_memory_usage_threshold
```

Use measured values wherever possible; the expanded size computed from headers is not the peak (see below).

- **The save-step peak is closest to the limit.** A long-run plan states, for each training mode, the measured cgroup peak at the save step and during a kill→resume segment, and which workloads have not yet been measured. For Nemotron values see [long-run-formal](../../../cases/nemotron/reports/long-run-formal.md) (the PP4 save step is only a few GiB from the Ray kill line) and [baseline-ckpt-crash new-image validation](../../../cases/nemotron/reports/baseline-ckpt-crash.md#verification-of-the-new-image-same-method-as-the-previous-round).
- **Estimate from headers before going up.** Without reading payloads, reading all safetensors headers is enough to compute the expanded size per dtype (the full DS4.1 checkpoint expanded to BF16 is about 1421.66 GiB, including auxiliary content, not only dequantized weights; training history source-23 (raw history archive, not included)); the expanded size is not the peak RSS, you still have to add serializer copies and mapped source pages.
- **A slope only signals headroom risk.** Resume from a verified checkpoint and observe long enough; you cannot linearly extrapolate a definite OOM step from a short timeline.
- **Do not reduce the microbatch or raise a guard as soon as you see an OOM.** First determine which phase and which objects are alive at the same time.

## 5. GPU memory: training–rollout colocation is time-sharing, compute per phase

Colocation means the training process and the rollout engine (vLLM) use the same GPUs. The two sides do not work at the same time: during the rollout phase vLLM is awake and the training process is idle; during the training phase vLLM is asleep. A GPU's budget is therefore **two ledgers, one per phase, each not exceeding single-GPU capacity**, not the sum of both sides' usage.

### 5.1 The recurring mistake

Adding "training peak + vLLM weights + KV", comparing it to single-GPU capacity, concluding it does not fit, and then increasing TP, lowering `gpu_memory_utilization`, requiring inference and training on separate nodes, or adding GPUs. This calculation stacks usage from two different moments.

A day-0 example (4 nodes × 4 GPUs, 277 GiB per GPU, full weights about 1.4 TiB): vLLM used TP16, 111 GiB of weights per GPU, `gpu_memory_utilization` 0.55; when asked why not TP8, the reason given was "TP8 is 222 GiB per GPU, plus the training segment's 61–96 GiB it does not fit", and on that basis it was proposed to split inference and training onto different nodes. In reality, the training segment moved only 2 layers off the GPU during rollout and left the rest on the GPU, which is a training-side implementation choice; once the whole segment is moved to host, each GPU holds only vLLM during the rollout phase. The constraint that really changes with TP is in host memory (§5.3), not GPU memory.

Before a conclusion says "does not fit", "need more GPUs", "need larger TP" or "need separate nodes", first write out the two ledgers per §5.2 and point out which item in which phase exceeds the limit.

### 5.2 The two ledgers

Per GPU, per phase, list the objects alive at the same time:

| Phase | Per-GPU usage | Designable quantity |
| --- | --- | --- |
| rollout (vLLM awake) | vLLM weight share + KV cache + runtime (CUDA graph pool, workspace, communication buffers) + **training-process residency during this period** | Training-process residency. It is decided by the training side: frozen weights, LoRA master and optimizer state can all be moved to host before rollout and moved back after rollout |
| training (vLLM asleep) | Training peak (weights + activations + optimizer + export staging) + **vLLM residue after sleep** | Sleep level and residue (§5.4) |

- **Rollout-phase `gpu_memory_utilization`** = 1 − (training-process residency + off-ledger overhead) / single-GPU capacity. Off-ledger overhead means allocations in the vLLM process not governed by this ratio (e.g. FlashInfer autotune warmup, about 8 GiB) and CUDA contexts of other processes on the GPU; there is no general value, back it out from `nvidia-smi` at first launch and the KV budget in the vLLM log. First ask "why is the training process still holding this during rollout", move off whatever can be moved, then set the value; do not plug the training peak into it. Moving has a time cost (host-to-GPU a few seconds to a dozen-plus seconds per segment, occasionally slower); weigh it together with the KV, concurrency or smaller TP it buys.
- **Usable GPU memory in the training phase** = single-GPU capacity − vLLM residue. Read the residue from the vLLM log `Sleep mode freed X GiB memory, Y GiB memory is still in use`, one line per worker; do not estimate.
- **How much KV you need can only be measured.** The startup log gives the available KV size; run one rollout at the formal concurrency and length limits and read the KV usage peak and preemption count from `/metrics`. A usage peak below about 80% with 0 preemptions is enough. For the same batch of requests, per-GPU KV usage is proportional to the number of attention heads per GPU; halving TP roughly doubles it per GPU.
- Look at the resolved values inside the container and the actual KV budget in the vLLM log, not the outer shell variables.

Measured values when the training process is not moved off or not moved off cleanly (Nemotron single node 4×GB200; last row Model A). They only show how much that training mode keeps resident during rollout; they are not an inherent cost of colocation:

| Training mode | Training process still holds per GPU during rollout | rollout `gpu_memory_utilization` |
| --- | --- | --- |
| EP4 + FSDP2 (mLite; BI=1 arm default `ACTOR_OPTIMIZER=fsdp2`) | Not measured separately | 0.65 (verified) |
| dist_opt (BI=1 arm optional `ACTOR_OPTIMIZER=dist_opt`; numbers from historical PP4 dist_opt measurements) | 54–64 GiB (FP32 quantized master resident on GPU) | 0.7, 0.6 and 0.65 all failed to start; 0.55 measured |
| EP4 + Megatron (mbridge) dist_opt + chunked optimizer offload (BI=0 arm, QAT + R3) | Not measured separately | 0.45 (current recipe value) |
| Model A, Megatron EP32 + LoRA, frozen base offloaded to file, 4 engines × TP8 (8×4 GB200) | 28–35 GiB (allocator 5–7, NCCL and CUDA context about 20; rises with dynamic batching) | 0.8 ran out of memory when waking KV in step 2; 0.7 measured (KV about 33 GiB per GPU, still about twice what 64 requests × 6144 tokens per engine need) |

The three values match the recipe scripts; see the `gpu_memory_utilization` row of [cases/nemotron/recipe.md](../../../cases/nemotron/recipe.md). Another direct cause of failing to start is that FlashInfer autotune warmup needs about 8 GiB more. When comparing two arms, differences such as 0.45/0.65 are confounders and must be listed first ([Recipe and evaluation](../recipe-evaluation/README.md)). `param_offload=True` saves GPU memory but moves parameters, changing the quantized master's `data_ptr`, so the pre-export check errors out; as with DS4, enable only optimizer offload.

### 5.3 Host memory is the item that changes with layout

GPU memory is time-shared; host memory is not necessarily:

- Level-1 sleep backs up each rank's weight share to host. Per-node backup = number of GPUs vLLM uses on that node × per-GPU weight share, plus the CUDA graph pool backup (when the corresponding switch is on). With multiple DP replicas each replica has its own full weights, and the per-GPU share is computed from a single replica's TP, so using the same number of GPUs, DP2's backup is twice DP1's. With vLLM's default torch pinned backup, actual usage exceeds nominal by the power-of-two rounding of [03 §4](03-pitfalls.md#4-pinned-buffers-rounded-up-to-powers-of-2) (Model A TP8: +578 GiB per node for 341 GiB of weights); an exact-size backup ([03 §11](03-pitfalls.md#11-sleep-backup-in-torch-pinned-memory-rounded-up-and-never-returned)) costs the nominal amount. Go by the measured difference before and after sleep.
- When rollout and training are each replicated into multiple copies (each copy a group of nodes handling part of the sequences), per-GPU and per-node ledgers are the same as for a single copy; what must be confirmed separately is the training side: gradients from the copies merge into one optimizer update, sample count matches the recipe, and the same adapter version is synced to every vLLM replica. This is a layout change that modifies a verified path.
- Amount the training process moves to host = the size of tensors it moves off the GPU before rollout, taken from the training side's own load log, not estimated from checkpoint file size (quantized weights grow after dequantization).
- After wake, the default torch pinned backup is not returned ([03 §11](03-pitfalls.md#11-sleep-backup-in-torch-pinned-memory-rounded-up-and-never-returned)), and a retained exact-size backup is kept by design, so during the rollout phase the host may hold both the backup and the weights moved off by the training process at once. Which applies must be measured per image and overlay: record per-node used memory before sleep, after sleep, and after wake. Model A's per-node peak came at a third moment: right after vLLM fell asleep, when the trainer started restoring its frozen weights while the backup was already in place (685–732 GiB against a Ray kill line of about 790 GiB).
- During the training phase, host holds sleep backup + training-process residency + save staging + page cache; sum them per §4.

TP choice is therefore jointly determined by two constraints: in the rollout phase each GPU fits the weight share and KV; each node's host fits the backup. Lowering TP increases the per-GPU share, so you can use fewer GPUs per node to keep the per-node backup unchanged. The example above: TP16 uses 16 GPUs at 111 GiB each; TP8 uses 2 GPUs on each of 4 nodes at 222 GiB each; both placements have about 445 GiB of backup per node. Concentrating TP8's 8 GPUs on 2 nodes gives about 890 GiB of backup per node, which does not fit the host (956 GiB) — that is the real reason that layout is infeasible.

Two boundaries:

- When one TP group spans nodes (e.g. TP8 placed as 2 GPUs per node), all-reduce goes over the inter-node link; whether it starts and how much slower it is depends on the cluster interconnect and communication backend and must be measured; equal host memory alone does not make the two placements equivalent.
- Splitting inference and training onto different nodes is not wrong, but it is a different resource scheme: no sleep or moving needed, both sides own whole GPUs, at the cost of summed node counts, and at any moment one side's GPUs are idle. Consider it only when an item in the colocation ledgers exceeds the limit under every TP, sleep level and moving method, or when sleep/wake/moving time takes the bulk of a step; when proposing it, present the two ledgers together with the node-count cost.

### 5.4 Sleep level 1 and level 2

The interface and the actions of the three levels are in [rollout engine §5](../rollout-engine.md#5-lifecycle-and-gpu-memory-ledger). The differences that matter for budgeting:

| | level 1 | level 2 |
| --- | --- | --- |
| Weights | Copied to host, copied back on wake, bitwise identical to before sleep | Discarded; only small state such as `named_buffers` is kept on host |
| KV cache and prefix cache | Discarded | Discarded |
| Host cost | One weight share per rank | Very small |
| After wake | Usable directly | `wake(weights)` only remaps GPU memory, contents are uninitialized; you must fully reload weights first, then wake KV |
| Time for one round trip | Seconds to a dozen-plus seconds, varies with weight size and host memory pressure | Equals one full weight reload |
| Suited for | Incremental weight sync (LoRA, partial parameters); base unchanged each step | Weights are fully synced every step anyway (full-parameter update); host cannot fit the level-1 backup |

- There are no general timing numbers; measure both on the full model. One reference: a TP16 engine with 111 GiB of weights per GPU, level-1 sleep 8–14 s, wake 5–12 s (wakes of several minutes have occurred under host memory pressure); the same engine reads its weights once at cold start in about 400 s, and each level-2 round trip is at least of that order.
- Confirm sleep actually took effect: `/is_sleeping` returns true, and every worker printed a `Sleep mode freed …` line; `/sleep` returning 404 means `VLLM_SERVER_DEV_MODE=1` is not set; level 0 only pauses scheduling and does not free GPU memory.
- Level 2 requires that the model's full reload path is available and verified (the full-reload rows of the day-0 serving validation lifecycle table). In the first serving gate, actually run both levels once; if level 2 does not work (the reload interface is unavailable for this model), record the error and reason as a known limitation, use only level 1 in formal runs, and it does not count as a gate gap.
- Choosing level 2 for LoRA training means rereading the base every step, which is usually more expensive than the level-1 host backup; work it out before choosing.
- Level 2 does not have to drop everything: `--sleep-preserve-parameter-names <pattern...>` (`ModelConfig.sleep_preserve_parameter_names`) copies matching parameters to host before sleep and back on wake(weights); `draft.*` matches the drafter. Use it to keep parameters that are frozen and not part of weight sync (e.g. DSv4.1's Engram non-table parameters, Model K's drafter), without falling back to level 1 for them; tensors already offloaded on host (`param.is_cpu`, e.g. the Engram table's `cpu_offload`) are untouched by level 2. Host cost counts only the preserved part. See `_save_sleep_parameters` in vLLM `gpu_worker.py`.
- Neither level frees GPU memory **completely**. Per the attribution in [vllm#59055](https://github.com/vllm-project/vllm/issues/59055), when only the `weights` and `kv_cache` pools are managed by sleep, a sleeping engine still keeps 14–21 GiB per GPU (that issue's measurements on 4×GB300 for the DeepSeek-V4 family and another MoE model; varies with model, TP/DP shape, `max_num_batched_tokens` and CUDA graph config; for other models go by your own logs): workspace staging, runtime state built together with KV (block table, sparse attention and indexer metadata, MTP buffers), the CUDA graph memory pool, FlashInfer all-reduce workspace, NCCL buffers, CUDA graph executables and the CUDA context. The set of PRs tracked by that issue plus `--sleep-mode-offload-cudagraph` and `--enable-nccl-comm-suspend` can bring this down to about 4–7 GiB per GPU; what remains is mostly driver-side state and cannot be freed. When the residue in the log is clearly above this range, first check whether some worker did not sleep (on multiple nodes check each worker), and whether other processes are on the GPU, then find the live owner per [04](04-gpu-oom-and-staging.md).
- Whether these switches and PRs are in the image you use must be checked in the image source; before using them, verify that fixed-token results after wake are bitwise identical to before sleep, and remeasure sleep and wake times (with NCCL suspend enabled, wake is 0.2–1.4 s slower; graph pool backup increases host pinned usage).
- "Reserved is high but there is still no GPU memory" after wake is an allocator boundary problem; see [04](04-gpu-oom-and-staging.md).

### 5.5 Must answer before writing a budget

1. Which objects are on each GPU in the rollout phase and in the training phase? List them separately, do not add them.
2. How much does the training process keep resident during rollout? Why can't it be less?
3. How much does vLLM leave per GPU after sleep (measured from logs)?
4. When is per-node host memory tightest? Which items are present at that moment (sleep backup, unreturned pinned cache, weights moved off by the training side, save staging, page cache)?
5. Which sleep level, and why? Has the level-2 reload path been verified?
6. How were TP and GPUs per node decided? Which constraint limits each (rollout-phase GPU memory, per-node host backup)?

Write the values into the recipe per training mode, not as one global default; the recipe's hardware comments list only verified topologies.
