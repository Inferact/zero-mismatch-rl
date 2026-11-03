# Numerical contracts and shared forward

## Check the arithmetic that actually executes

The same mathematical formula does not guarantee the same floating-point result. Check the accumulation dtype, cast positions, FMA, reduction tree, tiling/split-K, approximate activations, quantization scale/rounding, epilogue, and the dtype of the final logprob input. The same source code can also yield different execution configs through autotune, compile specialization, or a different GPU.

When training and inference share a kernel, still check the caller, arguments, layout and compile boundaries. A direct call and a torch.compile fused path can differ. When needed, keep the same forward behind an existing opaque custom-op boundary; do not introduce unproven arithmetic rewrites just to mimic the output.

For quantization-related execution checks (scale taken from the actual runner, quantization dispatch, quantizer not in the `nn.Module` tree, training vs deployment quantization grid) see the [precision, quantization and QAT guide](guides/precision-quant-qat/03-pitfalls.md#training-side-numerics); for the grid see [P8](guides/precision-quant-qat/03-pitfalls.md#p8-training-grid-differs-from-deployment-grid); for format and scale records see [conversion chain §1](guides/precision-quant-qat/01-conversion-chain.md#1-four-stages).

First check that the semantics exist on both sides: when a feature is implemented inside the kernel on one side and outside or missing on the other, the difference will not go away with precision settings. Do two-layer minimal comparisons per model feature (attention sink, RoPE variant, final norm, etc.), enabling one at a time. In an external MiMo case, SWA alone was 0 while SWA+sink was `3.906e-03`, because only one side's kernel included the sink.

## Discrete decisions before last-bit error

After comparing router/indexer logits, also compare the selected set, order and tie-break. The same Top-K set in a different order still changes the downstream gather/reduction. A deterministic tie-break must define the index order on equal scores explicitly and keep the invalid/padding index contract.

For MoE, also fix or verify dispatch, expert capacity, grouped GEMM geometry, quantization and combine. Do not hide router differences by forcing expert indices; forced routing is only for isolating the downstream.

The top-k/top-p truncation set is also a discrete decision: last-bit logits differences move tokens near the cutoff in or out of the candidate set, and the normalization denominator of the processed logprob jumps accordingly. Compare the candidate set before comparing processed logprobs. MiMo-V2.6 in production replays the candidate set together with R3: rollout records a fixed-shape, full-vocab-width bitmap (avoiding host syncs and truncation), and the training side renormalizes within the set. If the contract uses such a replay, state in the report that it masks set differences, and still report the set flip rate separately.

## Test shapes and process boundaries

For the same victim, construct batch1/multiple neighbors, different physical positions, lengths before and after boundaries, cross-process autotune, prefill/decode, Graph/eager. Finding a different launch by changing only M does not mean finding a numerical difference; you must test the output. Finding no difference only covers the shapes tested.

SSM chunk recurrence and per-token recurrence can be two legitimate but different rounding paths. Unifying them requires matching the chunk grid, boundary state precision and gate order; it is not just turning on a deterministic switch.

When localizing history-dependent operators layer by layer, looking only at one token's scan/conv inputs and outputs can misattribute an upstream history difference to that operator. First compare the first differing token per layer over the **full prompt batch**, then within that layer capture norm, projection, conv, scan, gate and output projection in order; single-position probes are only for narrowing candidates, not proof of the first cause. Plain Python hooks in a compiled graph may keep stale warmup data; first verify capture freshness with predictable sentinels such as embedding/residual, and when needed use the custom-op boundary that actually executes with independent buffers.

## Boundary between shared forward and backward

When using a shared inference forward + native differentiable VJP, separately verify forward parity, that every trainable parameter gets a finite gradient, and a gradient comparison under cotangents representable in the same dtype. Operator-level VJP verification does not automatically prove that whole-model gradients are bitwise equal to the original training implementation.

Never detach inference outputs and use them as a trainable substitute. Gradients existing/being finite still does not mean effective learning; also check whether advantage/reward has degenerated, whether gradients are non-zero, and whether parameters and optimizer state are updated.

## Changing the arithmetic baseline

If you intentionally change both sides' arithmetic baseline to align them, keep the old baseline, build a new frozen rollout, state explicitly "old outputs are not guaranteed bitwise unchanged", and run model quality and performance regressions. Do not attach the old baseline's success numbers directly to the new implementation.
