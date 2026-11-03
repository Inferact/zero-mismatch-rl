# Locating GPU OOM and staging peaks by lifecycle

Use this page when the per-shard model estimate fits, but initialization, export, weight update or wake still OOMs. Host (CPU) memory peaks and growth are in [02 Memory budget](02-host-memory.md), and save/load incidents are in [03 Pitfalls](03-pitfalls.md); this page only covers how to locate GPU-side problems by phase.

First establish which step made the allocation and which objects are still alive, then change the corresponding boundary. Do not lower batch
or change sharding as soon as you see OOM, and never attribute a failure that never entered update to update.

## 1. Minimum to record

One failure needs only:

```text
phase / process / rank / topology
requested bytes / allocated / reserved / device free
host RSS / container or Ray limit
last completed phase / first failing call
```

Mark missing fields as not obtained; do not guess. Distinguish GPU allocation errors, host OOM and an active Ray kill.
If there is no error and progress just stops printing, first check CPU/GPU/I/O, log timestamps and job phase to confirm
whether it is reading weights, compiling, in a collective or generating; taking long is not by itself a hang.

## 2. Memory peak is set by "objects alive at the same time"

Total parameter count divided by number of GPUs only estimates part of the steady state; it does not cover these peaks:

| Phase | Objects and boundary to look for | Minimal verification |
|---|---|---|
| Initialization / FSDP wrap | Whether it is fully materialized before sharding; largest single parameter; CPU-resident/tied parameters | Whether meta → shard → load actually takes effect; record per-rank peak |
| forward / backward | All-gathered full module, activations, workspace | Measure logprob and actor update separately; frozen parameters may also be gathered |
| Export / merge / quantize | Whether adapter-only export wrongly gathers base; single full tensors, fused groups, lazy generator references | Check the actual exported set, and when things are freed after yield/exception |
| weight update | Staging for incomplete layers and post-load temporary tensors | Missing receipts, number of layers incomplete at once, staged bytes per layer |
| compile / graph capture | Autotune workspace and graph pool | Locate the specific allocation from logs; eager is only for isolation, recheck the production graph separately |
| KV planning / wake | KV budget, trainer residency, allocator free cache | Actual resolved GMU/TP and allocator metrics, not the outer configured values |
| checkpoint / host | CPU full copies, optimizer gather, shared mappings, Ray guard | Compare per-process and per-node usage; RSS cannot be summed directly as physical total |

A small model that keeps the special layers, attention period and scale layout is for verifying the mechanism; the full model verifies capacity.
A small model passing does not prove the all-gather peak of a huge single parameter is affordable.

## 3. Why wake still has no GPU memory when reserved is high

The PyTorch allocator keeps a cache for later tensor allocations; after update's temporary tensors are freed,
some blocks become reserved-but-unallocated, but are not yet returned to the device allocator.
Another allocation path such as CuMem cannot directly use PyTorch's private free blocks to satisfy its own allocations.
So "reusable by PyTorch" does not mean "available to another allocator".

`empty_cache()` releases reclaimable free cache; it does not free tensors that are still alive, nor does it guarantee releasing
all reserved memory constrained by active blocks or graph private pools.
`gc.collect()` targets unreachable Python reference cycles; only when collecting those references actually leaves tensors
with no owner does it indirectly increase releasable cache. High reserved alone does not prove GC is needed.

Evidence should compare allocated/reserved/device-free before and after cleanup, and the result of the same allocation:
if allocated is still high, find the live owner first; if reserved-unallocated is small, stop using
empty-cache to explain a workspace OOM of tens of GiB. The cleanup location should match the resource hand-back boundary;
whether it goes at the end of update or before wake is decided by the real call path and a retest.

## 4. Telling a fix from a downgrade

- Lifecycle error: fix materialization/release timing, the completion contract, or the wrong gather set.
- Legitimate peak exceeds capacity: choose a topology that fits or shrink the declared configuration, and record the support boundary.
- Config not passed through: first confirm the resolved value inside the container; repeatedly changing shell variables is not a new comparison.
- Ray/container policy kill: account for the actual headroom before evaluating limits; do not raise the guard by default.
- Long duration: first time it by phase; only optimize the parts that dominate, and do not treat GPU utilization as evidence of completion.

Do not add resident full-tensor dumps to chase OOM; first use the one-line phase record above to decide the next step.

## Further reading and cases

For completion criteria and staging pile-up see the [weight update guide](../weight-update/README.md);
for the full-tensor boundary of export quantization see [conversion chain](../precision-quant-qat/01-conversion-chain.md#41-fsdp-path-verl);
for NVMe and environment reuse see the [environment guide](../environment-provenance/01-environment-setup.md);
for checkpoint save/load peaks and host memory growth on long runs see [02](02-host-memory.md) and [03](03-pitfalls.md) in this guide.

The Model F FSDP memory case notes keep the evidence of specific allocation sizes,
config not passed through, and host policy kills; those GMU/guard values are not general recommended settings.

## Evidence boundaries of historical experiments

Distilled from the 2026-08-28 DeepSeek-V4 audit; historical model conclusions are in
the per-model cases (e.g. [DeepSeek-V4-Flash](../../../cases/deepseek-v4-flash/README.md)). A path being configured successfully does not mean it was executed: record markers/counters when needed,
and mark zero hits as invalid verification. When comparing tensors, first align request, token position, rank/owner and valid region;
identical byte hashes alone do not prove the compared objects are the same. Verify candidate root causes with single-variable interventions and negative controls that actually trigger the failure.

Keep records of failed, invalid and later-superseded results along with the reasons. A small model can locate a mechanism but cannot be extrapolated to full-model capacity or performance;
the profiler is for explaining structure, product latency is measured by independent timing. Bind each result to the actual patch stack and imported-file
hash; a container that only overlays newly added files leaves old overlays behind, so restore and apply according to the manifest; details in
[Code and delivery](../../practices/code-and-delivery.md).
