# DeepSeek V4: deterministic path and performance recovery

Public integration entry point: [vLLM #54955](https://github.com/vllm-project/vllm/pull/54955) is a preview containing validation items still to be filled in; the plans in it must not be treated as passed results. Internal experience sources are the model workspace's knowledge base, its kernel optimization plan (2026-09-02) and its batch invariance reports (not included here). Everything below is limited to the corresponding historical versions/hardware.

## Five classes of differences

1. Reduction order: MHC split count, GEMM geometry, collectives.
2. Two implementations: sparse prefill vs paged/split-KV decode.
3. Discrete decisions: router/indexer Top-K ties, output index order.
4. Batch composition: chunk/capacity and dispatch formed by neighbor lengths.
5. State: Graph-reused buffers, cache, derived weights and scales after reload.

MHC's upper-layer post and this layer's pre may be fused, so hooking by module name does not necessarily give a comparable boundary. Carried residual/mix state must be aligned together. MoE must be covered from gate, route, quant and expert GEMM through combine; declaring "supports BI" does not mean all paths have been verified.

## Weights and scaled-down models cannot be guessed from directory names

History repeatedly confused Flash-Base's FP8 weights, Flash's MXFP4 weights, and converted FP8 derived weights. The actual checkpoint/config/tensor dtype is authoritative; "all called FP8" does not prove that quantization blocks, scale layout, clamp, ignored layers or loading paths are the same.

If the recipe requires a four-step conversion, read each step's inputs/outputs and conversion tool revision from the current scripts; do not reconstruct steps from memory of old chats. Record for each stage the source hash, target path, config diff, tensor key/shape/dtype/scale metadata and completion marker; the final launcher points to the verified product of the last step. Do not first get pre-conversion weights to run and then claim to have reproduced the post-conversion experiment.

When expanding from 4 layers to full scale, keep the hidden/head/expert geometry and the compression/attention/MHC paths actually used, and choose the smallest model that covers the necessary layer types and satisfies the PP split, rather than blindly picking 8/9 layers. Scale length, batch, layer count, cross-node communication and training layout separately; first estimate per-rank weight/optimizer/activation/KV capacity. An initialization OOM only shows that the capacity of that layout failed; it has not yet entered RL and provides no numerical conclusion.

`full-bi-no-r3`, trainer/rollout BI, `VERL_FULL_DETERMINISM`, `VLLM_BATCH_INVARIANT`, model runner v2 and Graph are different control dimensions. Trace the precedence launcher → framework config → worker env → kernel selector; two VERL switches set to 1 do not automatically cancel an explicit `VLLM_BATCH_INVARIANT=0` in the worker. Draw conclusions only from the resolved path, not by guessing from script titles.

## Sparse attention and Top-K

Top-K may select different sets or orders on ties, and the latter also changes gather/reduction. After fixing the tie-break and canonical index order, keep verifying the caller's relative/global index, C4/C128, tail sentinels and reused buffers.

If prefill's chunk plan depends on batch neighbors, single-request consistency can still fail in mixed batches. Having decode reuse the sparse prefill path helps match arithmetic, but may add paged KV dequant/gather cost. Optimization should first reduce idle blocks and data movement, then consider fusing into attention, keeping the same scale rounding and KV traversal order.

## A corrected Graph attribution

An old experiment had full-scale 0-diff in a single round; a later soak found mixed prefill/decode differences in 29 of 75 rounds. The prefix cache had caused the target prefill not to be re-executed at all; the dummy model also never formed the mixed steps seen in production.

Turning off Graph also changes scheduling speed, so step composition must be fixed offline before comparing combinations of eager/FULL/PIECEWISE and split plans. In one historical version PIECEWISE was the actual trigger condition; other versions/later training used different modes, so "always disable PIECEWISE" cannot be written as a general solution.

## Optimization paths

- Quantization: a fixed 1024 blocks per expert idles heavily on small decodes; adjust the grid by the effective upper bound known on the host, and adjust the loop stride accordingly.
- dequant/gather: a fixed 1024 blocks per request is equally wasteful for a few SWA rows; partition by workload and keep element-wise output.
- Top-K and index post-processing: optimize and ablate separately, and check that every slot is written; the tail-clearing contracts of the fused and non-fused paths must be covered separately.
- Graph: reduce host launch overhead, while proving that replay buffers and paths are really effective.
- Communication: reduction determinism may have a cost; kernel optimization cannot automatically eliminate collective overhead.

## Do not exaggerate micro gains

An internal report on the PR16/PR22 rollouts (2026-09-05) records six full43 rollouts, each with 1024 request receipts identical to the baseline. In the same node pool, PR16 improved latency by about 1.3–1.9%, PR22 alone by about 3.1%, and the combination by about 2.9%. Each arm/pool ran only once, so this does not show the combination beats PR22, and single-item gains cannot be added.

These are fresh-process rollout timings with no new training step executed; what is equal is sampled-token logprob receipts, not full-vocabulary logits. The PR22 simplification also has a micro A/B over 1200 shapes and Graph tests; micro results cannot substitute for end-to-end speedup.

## CP and reload

The training CP2 host/device compressed boundary misalignment is detailed in the [separate case](cp-compressed-boundary.md). Inference-internal 0-diff did not rule out a training packing bug.

After an update, check routed/shared weight aliases, quantization scales, MHC derived weights and Graph pointers. Repeated rollouts with no parameter change can only verify the frozen model, not dynamic sync. Pass consistency first, then do effective updates and model quality regression.

For the communication/environment troubleshooting ladder see [GB200 communication](../../skill/guides/debugging-mismatch/05-comm-fault-ladder.md); for the three classes of checkpoint failures see [checkpoint, resume and host memory](../../skill/guides/checkpoint-resume-memory/README.md).
