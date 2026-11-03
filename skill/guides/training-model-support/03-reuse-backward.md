# Reusing backward: when you actually need to write your own

When to reuse a backward and when to write glue. How to validate backward (gradient contract, module-level comparison, gradient-contract comparison, three-arm SFT comparison) is in [06 Backward validation](06-backward-validation.md); the Nemotron example (independent validation protocol, current backward status per operator) is in [cases/nemotron](../../../cases/nemotron/consistency-case.md#backward-independent-validation-protocol-and-per-op-status).

## Conclusion

The backward is determined by the **math formula**, not by which kernel the forward uses. The backward of any layer `y = f(x, W)` needs only the formula and the saved inputs: `dX = dY·W`, `dW = dYᵀ·x`. Switching the forward to a vLLM kernel does not by itself require rewriting the backward. "There is no backward operator at the same precision" is not a reason to rewrite: under the high_precision contract the backward is computed in BF16/FP32 anyway.

## Wiring: compute the forward once, the backward uses what the forward saved

The training forward is the inference forward. Each module is an autograd Function: forward calls the inference kernel to produce the visible result and saves what its own backward needs (inputs, visible output, LSE, fixed routing IDs, quantized values and scales); backward uses only these, calling the reused library backward or a closed-form VJP. If the inference kernel does not output what the backward needs, modify the kernel so it does.

The delivery bar is 0 diff, plus a training step within budget on the main topology fixed in C1; both are accepted together in C6. Nemotron ran one audit against DS4 on this rule, see the [audit table](../../../cases/nemotron/reports/ds4-structure-migration.md#backward-recompute-audit-whatever-ds4-saves-we-also-save-as-visible-tensors); Mamba SSD is not there yet, because vLLM's visible SSD does not output chunk state.

## Three cases

| Case | Characteristic | Example | Approach |
|---|---|---|---|
| A rounding-only difference | Same math; the kernel's summation order/tiling differs; the difference is in the last BF16 bit | RMSNorm, Mamba SSD scan, attention, conv1d, unquantized linear | Reuse the training framework's backward directly (MCore/TE/mamba_ssm); no code to write |
| B forward has rounding | FP8/NVFP4 quantization rounds values; the true derivative is 0 almost everywhere | Quantized linear, FP8 Q/K/V | First fix the STE proxy (pass rounding through as 1; dX uses the dequantized value by default; following DS4 we use TE high_precision, BF16 master, BF16 input, which is a master-value approximation and must be quantified per [06 layer 2](06-backward-validation.md#layer-2-gradient-contract-comparison)), then reuse the ordinary BF16 GEMM backward |
| C no ready-made backward for the whole segment | Composite of several operators; no library provides the whole backward | Routed experts: sort → grouped GEMM → relu² → unsort → top-k weighted sum | Write glue chaining existing operators; GEMM still uses TE grouped GEMM; compare gradients against an independent reference |

Decision flow: ① Does the forward have rounding/quantization (top-k ID selection does not count; IDs are fixed and not differentiated)? → no → A; yes → ② fix the STE contract → ③ is there a packaged training backward for the whole segment (TE Linear/GroupedLinear, MCore MoE)? → yes → B reuse (feed master values); no → C glue. If A cannot find a same-math training implementation, likewise write only glue. Look in mLite `primitive/modules`, `ops` first, then MCore/TE/mamba_ssm.

## How DS4 did it

The only hand-written kernel in DS4 is a small forward FP8 quantization Triton kernel; there is no hand-written backward kernel, only PyTorch closed-form formulas and glue, all in FP32.

| Operator | Case | Backward source | What was written |
|---|---|---|---|
| FP8 dense linear (qkv, shared expert, o-proj) | B | TE `general_gemm`, BF16 master | Nothing |
| Routed experts (grouped) | B+C | TE `general_grouped_gemm`; uses gate_up directly from the forward vLLM FC1 output, no recompute | Glue: sort, activation derivative, sum |
| Sparse attention | A | cuDNN sparse attention backward, consuming out and LSE saved by the forward | Nothing |
| SwiGLU (with clamp), RMSNorm, q/kv norm, mHC | A | Hand-written closed-form derivative, FP32, `torch.compile` | PyTorch formula |
| Routing scores | A (IDs fixed) | FP32 replay of `sqrt(softplus)` + normalization, handed to autograd | PyTorch formula |
| Others | A | Generic bridge: forward in vLLM, backward recomputes the PyTorch graph then `autograd.grad` | One utility function |

The four reasons DS4 wrote its own, none of which is "missing a same-precision backward kernel": ① the library backward is tied to intermediates from the library's own forward (TE RMSNorm needs rstd, MCore MoE needs token permutation; the vLLM forward does not have them); ② model-specific math is not in any library (`sqrt(softplus)` scoring, clamped SwiGLU, mHC Sinkhorn); ③ discrete selection must copy the forward (recomputing top-k in backward may pick a different expert on ties, computing gradients for the wrong expert, so IDs are saved and only the continuous scores are differentiated); ④ the generic bridge costs too much GPU memory and time (mHC's 20 iterations under autograd save intermediates for every iteration; switched to closed-form derivatives fused with compile). Also: inside a custom backward function there is no autograd graph, so any activation derivative called there must be closed form (`dynamic_clamped_swiglu_vjp`; same for the relu² derivative in Nemotron's `routed_vjp`).

## Three rules when reusing

1. Intermediate definitions must match: attention backward uses the forward's out and LSE; whether LSE is ln or log2, the softmax scale, and the memory layout must line up; if they don't, recompute the forward with the training kernel inside the backward.
2. Backward does not need to be bitwise identical: recomputed intermediates differ from the vLLM forward in the last BF16 bit, and gradients differ by about as much; this is allowed. It only needs to satisfy the gradient contract with error near the noise floor.
3. Reused backward must be validated too: an FP32 reference that does not call the backward under test, the same dY, and compare dX and dW error as a multiple of the noise floor.

## Two test-infrastructure false negatives

- A unit test that instantiated QuantFP8 directly missed the vLLM config context of production training (`CompilationConfig(custom_ops=['none','+quant_fp8'])`, Humming, PP4); the first long-sequence test failed before computing. The fix is for the test wrapper to use the same config as production; production operators and tolerances are unchanged.
- An import check without GPU visibility makes vLLM pick a non-CUDA platform, so `reshape_and_cache_flash`, conditionally exported for CUDA, cannot be imported. Dependency checks need real device visibility.
- Both are test errors and must not be attributed to train-inference differences.

## How to validate

Gradient contract, reference bridging, per-module checklist, two kinds of baseline, module-level VJP comparison, gradient-contract comparison and the three-arm proxy SFT comparison are all in [06 Backward validation](06-backward-validation.md).
