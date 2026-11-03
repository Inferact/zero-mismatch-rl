# GB200: first distinguish topology, implementation, mode and build

This document distills historical GB200 four-GPU-node experiments; it does not guarantee that future DeepEP/NVSHMEM releases keep the same support matrix. Before use, check the source and installed artifacts of the target fork/commit.

## Do not use one name for four things

| What to ask | Record example |
| --- | --- |
| Which implementation | AllToAll, a specific DeepEP version, HybridEP; note fork/revision |
| Which path of that implementation | The actual class, API, kernel corresponding to high-throughput / low-latency / v2 |
| How it communicates | Intra-node NVLink; inter-node RDMA; or the MNNVL path actually enabled |
| What it was built as | 4-peer/8-peer macro, target GPU, NVSHMEM dynamic dependency |

`NVL4 + RDMA fallback` describes the communication topology and **says nothing about the DeepEP mode**. Do not answer it as low-latency or high-throughput unless the call site proves it. The relationship between `DeepEP v2`, `HybridEP` and HT/LL depends on the specific version; do not draw a general containment relationship from the names.

Not having MNNVL enabled does not mean cross-node EP is impossible: a verified path may use intra-node NVLink and inter-node RDMA NICs. Nor can you conversely assert that this path works on all GB200s; the devices, network, peer build, underlying libraries and calling API all need verification.

## Get the world size right first

Write trainer and rollout process groups separately; for the dense world and expert world formulas and grouping traps such as "two DP4/EP4 groups are not one EP8 group", see [training-side layout and boundaries](../training-model-support/04-layout-and-length.md). How the actual framework groups ranks is determined by the runtime; do not change GPU counts to fit a config string.

## Fault-isolation ladder

1. Confirm job/node ownership and no duplicate workers; use the GPU, RDMA, IMEX, IPC and shared-memory mounts prescribed by the project infrastructure. Check visible devices and permissions on host and in container; do not use `--privileged` in place of diagnosis. First confirm the GPUs are clean (see dirty GPU below).
2. Inside the container, verify GPU count, GPU/NIC topology, HCA names, loaded driver/user-space libraries; check device and environment propagation for each rank.
3. Two-node NCCL test: same message size, warmup, topology and timing definition; distinguish algorithm bandwidth from bus bandwidth. Successful initialization does not mean reasonable bandwidth.
4. DeepEP's own version-matched benchmark: single node first, then two nodes; record the actual HT/LL/v2 API, dtype, token/expert distribution. The two-node test must prove cross-node transfer occurred, not just that two machines were started.
5. dispatch/combine correctness: same input, routing, scales, reference implementation; compare valid outputs, inverse mapping and non-finite values. A single `max_abs=0` does not prove bitwise consistency for arbitrary batch/mode.
6. Only after passing, run small-model inference, batch/prefill-decode invariance, and finally training; on initialization failure do not start expensive RL retries.

## How to check historical symptoms

| Symptom | Minimal discriminating check | Conclusion you cannot draw directly |
| --- | --- | --- |
| `num_ranks > NUM_MAX_NVL_PEERS or low_latency_mode` | Call mode, ranks, peer build of the loaded binary, and the source of that assertion | Asserting macro=4 just from the source pin |
| `invalid resource handle` | First CUDA error, device/IPC/IMEX, processes and libraries, actual call stack | Always attributing it to NVSHMEM or model numerics |
| HCA `rocep*` not recognized | Actual HCA name, NVSHMEM version/load path, the worker's `NVSHMEM_HCA_PREFIX` | Single-node pass means two-node RDMA passes |
| Passes after recompile | Compare the full before/after closure and launch arguments | The compile action itself proves the peer macro is the only cause |

Historically there are records both of a peer-8 build not fitting the four-GPU path and passing after rebuilding as peer-4, and of a new ARM image working out of the box. Later, some NVSHMEM 3.4.5 environments passed the two-node test just by setting `NVSHMEM_HCA_PREFIX=rocep`. So "always upgrade GB200 to 3.6.5 and rebuild DeepEP" is not a conclusion; first reuse an environment with evidence, then do single-variable tests.

When dynamic NVSHMEM updates are involved, check ABI/linking; if a rebuild is needed, keep Torch/CUDA unchanged and record before/after. Propagate environment such as `NVSHMEM_HCA_PREFIX` to workers before they are created and verify the value inside the worker; an export in the head shell is not proof.

If a DeepEP performance A/B enables MNNVL for BI=1 while BI=0 uses a different communication backend, report "each side's best available stack" and "the causal comparison toggling only BI" separately. General network gains cannot all be counted as BI-specific kernel optimization. Use the interaction ablation in [Performance and profiling](../performance-profiling.md) and [04 ablation method](04-ablation-method.md).

## Same-fabric fairness

Before any communication-bound A/B, first prove both arms use the same transport. It happened once on GB200 NVL: the BI=0 arm's NCCL AllGather/ReduceScatter went `via P2P/MNNVL` (same clique, cliqueSize 8), while the arm environment script pinned the BI=1 arm's DeepEP low-latency a2a to RoCE (`VLLM_DEEPEP_LOW_LATENCY_USE_MNNVL=0`, `NVSHMEM_HCA_PREFIX=rocep`). A whole round of optimization attributed the gap to "non-overlappable fixed communication latency" when actually the wrong NIC was used; two environment variables recovered most of the gap (numbers in the [DS4 case](../../../cases/deepseek-v4-flash/README.md#serving-performance-levers-gb200-dp8ep8)).

Check steps:

1. `nvidia-smi -q | grep -A4 Fabric`; ClusterUUID/CliqueId must match on both nodes.
2. Run `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET` once per arm and see whether the transport is P2P/MNNVL or NET/IB.
3. grep the arm environment files for DeepEP/NVSHMEM pins. A hard `export` overrides values passed via container `-e`; write it as `${VAR:-default}`.

Mechanism: `VLLM_DEEPEP_LOW_LATENCY_USE_MNNVL=1` → `deep_ep.Buffer(allow_mnnvl=True)`, DeepEP no longer exports `NVSHMEM_DISABLE_MNNVL=1`, and NVSHMEM 3.4.5 automatically uses MNNVL for peers in the same clique. `DEEPEP_MAX_NVL_PEERS` is not a runtime switch (it is DeepEP's compile-time `NUM_MAX_NVL_PEERS`); peer-4/peer-8 builds are selected by switching overlays. MNNVL is not hybrid-specific: the DeepEP v1 LL kernel branches per destination on `nvshmemi_get_p2p_ptr` (NVLink direct write vs IBGDA); MNNVL just makes cross-tray peers P2P-mappable; "hybrid mode" is a v2/elastic concept. MNNVL was on by default before DeepEP #190 (2025-06-06) and turned off afterwards to save NVSHMEM memory.

## dirty GPU: a crashed job releases GPU memory slower than scheduling

Slurm takes only a few seconds to schedule the next job onto a node whose "previous job just ended"; GPU memory release by crashed (OOM, NCCL hang) processes takes tens of seconds to several minutes. A new job's vLLM engine or Megatron worker starts on GPUs still holding 100+ GiB, reports `Free memory on device (23/184 GiB)` or `CUDA out of memory ... Process <pid> has 158 GiB`, and dies within seconds; 5 jobs died like this in one day, including ones hitting leftovers from someone else's just-finished job. Resubmitting immediately after a failure is exactly when you're most likely to hit it.

Fix: the launcher has a built-in GPU-clean barrier (check `nvidia-smi` per node, >2 GiB per GPU counts as dirty, 600 s timeout); new launchers copy this section; before manual resubmission confirm with `srun nvidia-smi`, or wait 2 minutes. When you see these two errors, first check for leftover processes; do not treat it as a model GPU memory budget problem.

## Order assertion across two dispatches

The physical return order of two communications on the same input is not necessarily the stable semantic token order. If new Megatron code requires two dispatches to have the same order, locate the call site, metadata, rank/expert mapping and DeepEP's real ordering guarantee; compare the fork's specific changes rather than only asking "is it the same commit".

Do not simply delete the assert or restore the old fingerprint sort to mask the new interface contract, and do not declare the underlying issue fixed because it didn't trigger once. Build a minimal loop of double dispatch on the same input, check each mapping and inverse combine by request/token/expert ID, record batch, rank, mode, concurrency and versions; then decide whether same order should be guaranteed by the communication implementation, or whether the caller must match by metadata.
