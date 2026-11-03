# Survey of mainstream practice for the D term in FP8 attention backward (2026-10-04, read-only)

Question: once the forward rounds P to FP8, the visible output O = fp8(P)·V no longer equals softmax(QKᵀ)·V. In that case, for the backward D = rowsum(dO⊙O), which O do mainstream frameworks use?

## Conclusion

1. **Mainstream training configurations do not do FP8 attention.** The attention core (QKᵀ, softmax, PV) stays in BF16; only the linear layers use FP8. Evidence: Megatron's `fp8_dot_product_attention` is off by default and none of the local recipes turns it on; the DeepSeek-V3 paper states that attention keeps the original precision; Nemotron-H only puts the linear layers into FP8; torchao/torchtitan only convert Linear; FP8 in FA3 and FA4 is forward-only.
2. **The few implementations that do train with FP8 attention compute D directly from the visible O**, with no extra reconciliation step. This includes TE + cuDNN `fp8_dpa` and SageBwd. TE does not acknowledge the resulting bias anywhere; its tests only check that RMSE is below 11–15% of the value range. SageBwd measures a rel-L2 of 7.4% for δ relative to FPA, and large-batch pretraining does show a loss gap, but the paper attributes it to the low SNR of dS and does not fix D.
3. **"Make D consistent with the recomputed P" is not the mainstream default for FP8 kernels, but two places explicitly endorse the principle**:
   - Upstream FA4 (Dao-AILab `71dbd9372d`, 2026-09-23) added an O residual for sparse-MLA training. The stated reason is that the dpsum error (even if it only comes from the BF16 rounding of O) becomes the dominant term of the dQ error when attention is concentrated.
   - TE's own FP8 numerical-emulation reference implementation (unfused + STE) goes through autograd and gets a D consistent with the unrounded P, the same semantics as our fix.

   So our fix is not invented from scratch: when the visible O and P are inconsistent, it restores D according to "the correct gradient under STE semantics".

**Why it is a systematic bias and not noise**: when D is correct, rowsum(dS_i) = 0, which corresponds to softmax being invariant to a row shift. When D is inconsistent, rowsum(dS_i) = sum of P × (D_true − D_vis) ≠ 0, and dQ_i gets an extra term (D_true − D_vis)·Σ_j P_ij K_j. This term accumulates with the same sign per row, so it shows up as a bias and does not cancel out. This matches the roughly 3% dQ/dK bias measured at step 7.

## Comparison table

| Framework / work | Is forward P FP8 | Which O backward D uses | Bias acknowledged | Evidence (commit) |
|---|---|---|---|---|
| TE `fp8_dpa` + cuDNN FP8 fused attn, default `NVTE_FP8_DPA_BWD=1` | Yes. S_quantizer: P is first multiplied by scale_s, then cast to E4M3, and used for P·V | Visible O: delayed scaling uses the FP8 O; current scaling by default uses the F16 O output of this FP8 forward kernel (`NVTE_DPA_FP8CS_O_in_F16=1`); MXFP8 uses the F16 O | No. Tests only check RMSE < 0.11–0.15 × value range | TE `fc1fc0e7` `backends.py:180, 1500-1548` (saves O), `1825-1840` (O passed to `fused_attn_bwd`); `test_attention.py:2204-2206, 2474-2476`; `docs/envvars.rst:241-251` |
| Same, `NVTE_FP8_DPA_BWD=0` (FP8 forward + F16 backward) | Yes | Still the `out_f16` produced by the FP8 forward; moreover Q/K/V are the original BF16 values (when the inputs are not FP8), so both P and D are inconsistent | No | `backends.py:1545-1548, 1896-1910` |
| cuDNN frontend `sdpa_fp8_backward` | Yes (fwd inputs include Scale S / Descale S) | Only accepts the O passed in by the caller (the API has no other source). In the open-source DSL implementation `delta = rowsum(dO_fp8 * O_fp8) * descale_o * descale_dO`, then the fp32-recomputed P is used for `dS = P*(dP − delta)` | No. The SM107 DSL offers `external_delta=True`, letting the caller pass its own D (designed for gated attention) | cudnn-frontend `9ecb75e7`: `docs/operations/Attention.md` "SDPA FP8 Backward"; `python/cudnn/sdpa/bwd/kernels/bprop_chain_common.py:13-25, 219-235`; `sm107/bprop_d256_fp8.py:20-35`; `sm107/prepared_host.py:19-22` |
| TE UnfusedDotProductAttention FP8 emulation | Yes (S quantized then dequantized) | autograd's softmax backward uses the unquantized P, and S quantization uses STE pass-through, so D = rowsum(P⊙dP), consistent with P | Not applicable (this is the reference implementation) | `backends.py:202-207, 700-705` |
| Megatron-Core | No by default. `fp8_dot_product_attention=False`; when on, it hands off to TE's `fp8_dpa` and goes through cuDNN FusedAttention | Same as TE | None | Megatron `15c83d2fc`: `transformer_config.py:626-629`, `fp8_utils.py:778-790`, `extensions/transformer_engine.py:3288-3289`; none of the Megatron, NeMo-RL, verl, slime, miles or torchtitan recipes turns it on |
| FlashAttention-3 (Hopper) | Forward supports FP8 | Not applicable: no FP8 backward | README says "FP16/BF16 forward and backward, FP8 forward" | flash-attention `e9515d5d`: `hopper/flash_api.cpp:1300-1302` (bwd only accepts fp16/bf16), `hopper/setup.py:595`, `README.md:43` |
| FA4 CuTe (dense) | Forward supports FP8 (P's smem uses q_dtype) | The public interface raises directly when `requires_grad`. The preprocess of `_flash_attn_bwd` uses the `out` passed in | Explicitly forward-only | `flash_attn/cute/interface.py:862-864` ("FA4 CuTe FP8 backward is not supported yet (forward-only)"), `2714-2723`; `flash_bwd_preprocess.py:5`; vLLM's bundled copy `vllm/vllm_flash_attn/cute/interface.py:545-546` has the same guard; our actor bypasses the public interface and calls `_flash_attn_fwd/_bwd` directly |
| FA4 CuTe (sparse-MLA, BF16) | No | The forward additionally saves `o_lo = fp32(O) − bf16(O)`, and preprocess uses `out + o_lo` to compute dpsum | **Yes**. It states that the dpsum error is the dominant error when attention is concentrated: at peak 0.8, dq rel-L2 0.89% → 0.24% | `interface.py:1243-1245`; `AI/SPARSE_MLA_DPSUM_PRECISION.md`; commit `71dbd9372d` |
| PyTorch SDPA / torchao float8 / torchtitan | No. float8 training only converts Linear | Not applicable | torchao's FP8 attention prototype notes "inference, backward not supported" | ao `a701b6a`: `torchao/prototype/attention/__init__.py:7-10`; torchtitan `65fa556b`: `components/quantization/utils.py:19` |
| SageBwd (SageAttention3 / arXiv 2603.02170) | Yes (P̃ quantized per token to INT8, then P̃V) | Visible O: Alg.2 line 2 `D = rowsum(dO◦O)`, with O from the quantized forward; P recomputed in fp32 | Partially acknowledged: Table 2 shows rel-L2 of δ relative to FPA is 0.0736, and the paper says the dS error "combines quantization noise from forward-pass tensors (P, O)"; 2.1M TPS pretraining loss 2.640 vs FPA 2.586, but D is not fixed | arXiv 2603.02170 §3, §4.2, §5.2, Appendix A; thu-ml/SageAttention `d1a57a5` contains no SageBwd code |
| DeepSeek-V3 | No | Not applicable | Paper §3.3.1: "maintain the original precision (BF16 or FP32) for … attention operators" | arXiv 2412.19437 |
| DeepSeek-V4 (our DS4-aligned actor) | No. KV is dequantized to BF16, then `flash_mla_sparse_fwd` is called | Visible O; its P is BF16, so it is self-consistent | Not applicable | the attention D-fix prototype report, §4 |
| Nemotron-H pretraining | The paper only says the linear layers (including the QKV/O projections) use FP8 and does not mention the attention core. Inferred from Megatron defaults to be BF16, **not verified** | Not applicable | — | arXiv 2504.03624 §2.3 |

Reachability: GitHub (gh / git clone) and arXiv (WebFetch) are both accessible. cudnn-frontend, flash-attention, ao, SageAttention and DeepSeek-V3 were shallow-cloned into a scratch directory; TE, Megatron and torchtitan used local reference checkouts.

Unverified points:
- The cuDNN SM90/SM100 FP8 backward is a closed-source engine. "D comes from the O passed in" is inferred from the API, and the open-source SM107 DSL implementation corroborates it.
- No public code for SageBwd was found; only the paper's pseudocode was used.
- The Nemotron-H paper does not state the precision of the attention core.

## Implications for our choice

- **Serving parity requires the forward to be FP8**: the forward must be bitwise identical to vLLM, rollout_probs_diff = 0. So "use BF16 for all of attention like the mainstream" would break alignment and is not an option.
- **Copying the TE/cuDNN approach (compute D from the visible O) is exactly our pre-fix state**: it leaves a systematic dQ/dK bias of about 3%, and the v2 criterion FAILs at steps 150 and 250 under both PP4 and EP4. TE can accept this bias because its tests are loose and it has no FP32 gradient criterion; that does not show the bias is harmless. SageBwd's pretraining gap at high TPS is also consistent in direction with this kind of error.
- **Recommendation: keep the D fix**: semantically it is equivalent to TE's emulation reference and to the correct gradient under STE, and in principle consistent with the upstream FA4 dpsum fix. The cost is +4–15% per-layer backward, < 0.05% end to end.
- Optional cheaper implementations, not required:
  - Have the forward additionally output an O computed from the "unrounded P", which needs one more MMA.
  - Switch to a backward that supports an external delta, such as cuDNN's `external_delta`, but currently that only exists in the SM107 DSL.
  - Add a parameter for passing in D to the FA4 preprocess.

  None of these is worth doing at this stage.
