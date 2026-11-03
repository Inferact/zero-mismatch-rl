# NVFP4 deployed weights do not move at small lr: mainstream practice, quantitative estimates and recommendations

Date 2026-10-03. Read-only survey; no repository was modified. For the trigger see `sft-three-arm.md`: with lr 1e-6 for 300 steps, the NVFP4 deployed values (routed/shared experts, lm_head, 86.7% of parameters) did not change.

## Conclusions
- **Mainstream QAT-RL masters never start from grid points, so they do not have our dead zone.** NeMo-RL QARL / TE NVFP4, Kimi's and SGLang/slime's INT4 QAT, and Jet-RL all do the same thing: a BF16 (or FP32) master, requantized from the master with RNE every step. But their θ0 is always the **original BF16 weights**, not a dequantized checkpoint.
  - An off-grid master is uniformly distributed within its grid cell, so any small drift flips a proportional fraction of codes, and the deployed values follow the master as a whole without bias.
  - Our θ0 sits exactly on grid points, so every element is Δ/2 away from its nearest decision boundary. That is the dead zone.
- **Mainstream schemes do not use stochastic rounding (SR) in the forward quantization of weights.** Both the NVFP4 paper and TE only apply SR to gradients, and explicitly say that SR on forward tensors is harmful.
- **A paper has already pointed out this exact problem.** QuRL (arXiv 2602.13953) §4.3 points out that at lr 1e-6 the RL update magnitude (1e-7 to 1e-6) is far smaller than the quantization error. Its fix, UAQ, essentially amplifies the relative update size, which under Adam is equivalent to increasing lr.
- **Estimates (Gaussian synthetic weights, simulation with 4over6 and E4M3 block scales):**
  - Codes only start to flip once drift reaches about 2% of the tensor RMS.
  - Deployed values only largely catch up with the master once drift reaches about 10%.
  - Measured: over 300 steps the routed experts drifted only 0.23–0.29%. Even with linear extrapolation it takes about 2000–2600 steps before codes start to flip, and 5000+ steps before there is noticeable learning. With random-walk (√t) extrapolation it takes more than ten thousand steps.
- **Recomputing the block scale does not make codes flip earlier.** Starting from grid points, requant is idempotent, and an E4M3 scale only changes code once the block amax changes by 3–6%. Tiny changes of the global scale only rescale the whole tensor uniformly, on the order of 1e-5. This is the source of the measured "deployed values move ≤3e-5, projection onto the master direction is 0" (inference).
- **Recommendations (see section 4 for details):**
  1. First choice: **(f) use the original BF16 weights as master θ0**, which is exactly the mainstream practice; if the original weights are unavailable, fall back to **(f′) deterministic dithering of θ0 within its RNE cell**. Neither changes the quantizer or vLLM.
  2. Next: **(c) fixed-seed SR requantization**. Consistency still holds, but it needs a custom quantizer, and deployment error grows by about 1.4x.
  3. (b) Larger lr only as an auxiliary measure.
  4. (a) Must be written into the documentation.
  5. (d) and (e) do not fit the true on-policy goal.

## 1. How mainstream systems handle it

| System | Training forward | Master | θ0 | Measure against losing small updates | Consistency |
| --- | --- | --- | --- | --- | --- |
| **NeMo-RL QARL** (ModelOpt fake-quant, NVFP4 W4A16/W4A4, real-quant pushed back) | fake-quant NVFP4 (quantized from master every step with dynamic e4m3 block scales), STE in backward | BF16 parameters + FP32 optimizer | **The original BF16 model**, e.g. `NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16` (`examples/modelopt/qa_grpo_nano3_megatron.yaml:16`), `...Nano-30B-A3B-BF16` (`grpo-nanov3-...-w4a16-real.yaml:8`) | No dedicated measure; lr 5e-6 (`grpo_math_1B.yaml:290`) or 1e-6 (`grpo-qwen3-8b-base-dapo-...-w4a16.yaml:44`); IS correction enabled (same file :23) | Approximately consistent, monitored via Generation KL (`docs/guides/quantization-aware-rl.md:1-31`) |
| **NeMo-RL TE NVFP4 per-token** | TE NVFP4 forward; backward takes the dequantized path | BF16 parameters + FP32 optimizer, `fp4_param=false` (`docs/design-docs/te-nvfp4-per-token-rollout.md:18-26,48-51`) | Qwen3-30B-A3B-Base (BF16) | SR/RHT/2D disabled (same file :128-132), lr 1e-6 (recipe :57), TIS ratio 2, router replay | Not bitwise consistent; TIS as the safety net (recipe :21-23) |
| **TE NVFP4BlockScaling** | RNE for weights, **SR only for gradients** (`transformer_engine/common/recipe/__init__.py:512,600-608`; `docs/.../nvfp4.rst:107`) | Decided by the framework | — | 4over6 is positioned as "targets RL and post-training" (`recipe/__init__.py:537-543`), but it provides no weight SR or any mechanism aimed at small updates | — |
| **NVFP4 pretraining paper** (arXiv 2509.25149) | NVFP4, 2D 16×16 weight scales; RHT only on Wgrad inputs | FP32 master weights and optimizer states | Random init | Gradient SR; about 16% of linear layers kept in BF16 (first 2 blocks plus last 8); switching back to BF16 before lr decay recovers the loss | Not applicable. The paper explicitly says SR on weights or activations causes divergence |
| **QeRL** (arXiv 2510.11696, NVIDIA/MIT) | NVFP4 frozen base + BF16 LoRA; training forward and rollout use the same NVFP4 base | LoRA in BF16 | Quantized base | The base is never updated at all; LoRA lr 1e-5 (BF16 LoRA collapses above 5e-6); AQN folds noise into the RMSNorm weights | Consistent, no IS |
| **FlashRL / TIS** (Feng Yao et al.; github.com/yaof20/Flash-RL) | BF16 | BF16 | BF16 | Online quantization to INT8/FP8 on the rollout side after each update; accepts off-policy, corrected with truncated IS | Not consistent |
| **QuRL** (arXiv 2602.13953) | BF16 (only rollout is quantized, INT8/FP8) | BF16 | BF16 | **§4.3 points out the update magnitude (1e-7 to 1e-6) is far smaller than the quantization error**; UAQ applies a one-time invariant scaling W/s, X·s (s=1.5, absorbed by LayerNorm); TIS + ACR | Not consistent |
| **Jet-RL** (arXiv 2601.14243) | Unified FP8 precision flow, training and rollout consistent | BF16 | BF16 | Stresses that it must be on-policy; does not mention the grid dead zone | Consistent (FP8) |
| **Kimi K2 Thinking** (HF model card) | "uses QAT during the post-training phase, INT4 weight-only quantization of the MoE components" | Not published | Not published | lr and details not published | Inference is INT4 |
| **SGLang/slime INT4 QAT RL** (lmsys.org/blog/2026-01-26-int4-qat) | fake-quant INT4, 1×32 groups, amax dynamic scaling, STE | "the training side maintains a BF16 master" | BF16 | No mention of lr, SR or dead zone | W4A16 inference; logprob diff close to BF16, smaller than FP8 |
| **Systematic study of reasoning QAT** (arXiv 2601.14888) | fake-quant | BF16 | PTQ (RTN/GPTQ) init | lr 2e-5 in the KD stage, lr 5e-7 in the RL stage | — |
| **vime (vLLM RL) low-precision docs** | BF16 training + FP8 rollout (recommended); FP8 training + FP8 rollout (experimental); INT4 QAT (beta) | BF16 | — | Not mentioned | — |
| **DeepSeek-V4 aligned (verl #7688)** | W8A8 block FP8 identical to deployment, 0-diff | BF16 master | Dequantized FP8 checkpoint, also on grid points | None | Bitwise consistent |

**Is DS4 affected the same way?** Yes, but much more mildly.
- E4M3 is a floating-point format: each element's spacing is proportional to its own magnitude (relative half-step 3–6%), and small elements also have small absolute spacing, so some elements can always flip code.
- The minimum NVFP4 spacing is 0.5·s (s is the block scale), uniform across all elements of the block, with no finer grid.
- Our measured FP8 Mamba in_proj passes 13–16% of master movement through to the deployed values, while NVFP4 passes 0 (`sft-three-arm.md:155-160`), consistent with this difference.

## 2. Who does FP4/INT4 true on-policy? How do the quantized weights get to move?
- Among schemes whose training forward equals quantized serving, only Kimi/slime INT4 QAT and NeMo-RL QARL real-quant **update the base**; QeRL freezes the base and only trains LoRA.
  - Both Kimi/slime and NeMo-RL do: BF16 master plus fake-quant forward plus STE, RNE requantization every step.
  - Neither **uses SR, nor a separately increased lr for the quantized parameters**.
  - The reason they move at all is that the master comes from the original BF16 model, which is not on the grid to begin with.
- INT4 has the same dead zone. With 1×32 groups and amax/7 scaling, spacing = amax/7, about 30–50% relative to a typical element, the same order as NVFP4.
  - If θ0 were a dequantized INT4 checkpoint (on grid points), it would stagnate just the same at lr 1e-6.
  - Kimi's lr and θ0 are not published. Published QAT-RL lrs are 5e-7 to 1e-6; see 2601.14888 and the NeMo-RL recipes.

## 3. Quantitative estimate (our configuration)
**The FP32 main accumulates small updates.**
- The report says "FP32 main, BF16 parameters are RNE(main)" (`sft-three-arm.md:36`), so each step's BF16 parameters are regenerated from the FP32 main, and small updates accumulate in FP32.
- BF16 rounding error is 2^-9 of the value (about 0.2%), much smaller than the NVFP4 half-step (≥ 0.25·s, about 8% of σ), so BF16 is not the bottleneck.
- The bottleneck is only the NVFP4 RNE.

**Minimum steps for a single element (analytic lower bound):**
- For |q| ≤ 2, half-step = 0.25·s_b·g (s_b is the E4M3 block scale, g the global scale).
- The magnitude of a single Adam update is about ≤ lr (when gradient signs are consistent).
- So at least 0.25·s_b·g / lr steps are needed, which equals amax_b/(24·lr) (map-to-6) or amax_b/(16·lr) (map-to-4).
- Example: with amax_b = 0.03, even if every step has a consistent direction, at least about 1250 steps are needed.
- This quantity can be computed as a distribution directly from the checkpoint's scale tensors, without guessing σ. The proxy checkpoint is on the cluster's Lustre and not visible from this machine, so it was not actually computed this time.

**Simulation (a local `sim.py` script):**
- Setup: Gaussian weights, block 16, 4over6 (MAE), E4M3 scale, global scale set to amax/(6·256). θ0 is the dequantized grid value; add a random drift of RMS r·σ to each element; remove the effect of the global scale when counting code flips.

| r = drift RMS / σ | From grid: code-flip fraction | From grid: projection of deployed movement onto drift direction | From BF16 (off-grid): code-flip fraction / projection |
| --- | --- | --- | --- |
| 0.003 | 0 | 0 | 4.2% / 0.97 |
| 0.01 | 0.003% | 0 | 13% / 0.99 |
| 0.02 | 0.6% | 0.01 | 25% |
| 0.03 | 6.5% | 0.10 | 35% |
| 0.05 | 32% | 0.44 | 52% |
| 0.10 | 70% | 0.90 | 71% |

- Measured drift r after 300 steps: routed experts 0.0023–0.0029, shared experts 0.0053–0.0086, lm_head 0.0056.
- Linear extrapolation (upper bound with consistent gradient direction):

| Weight | Codes start flipping (r ≈ 0.02) | Projection about 0.44 (r ≈ 0.05) |
| --- | --- | --- |
| routed experts | about 2100–2600 steps | about 5000–6500 steps |
| shared experts | about 700–1100 steps | — |
| lm_head | about 1100 steps | — |

- With random-walk extrapolation, routed experts need more than 18,000 steps before codes start flipping. RL gradients are noisier than SFT, so RL is closer to the random-walk case.
- Reference: starting from BF16 (off-grid), r = 0.003 already gives a projection of 0.97, i.e. every bit of drift reaches the deployed values.

**Effect of recomputing the 4over6 block scale:**
- Starting from grid points, requant is idempotent (bitwise identical in the simulation, and also measured byte-identical in arm A).
- The selected candidate has zero error, so a small perturbation does not let another candidate win.
- An E4M3 scale only changes once the block amax changes by more than half an E4M3 spacing (3–6%), later than element code flips.
- When the global amax changes by ε, the block scale is pulled back to its original value by RNE, while the deployed value q·s_b·g' is multiplied by (1+ε) as a whole. This is uniform scaling, not learning, consistent with the measured "≤3e-5, projection 0" (inference).
- So recomputing the block scale does **not** make codes change earlier.

## 4. Ranking of candidate fixes (by how mainstream they are and compatibility with bitwise consistency)
1. **(f) Initialize the master from the original BF16 weights** (mainstream practice: NeMo-RL QARL, slime/Kimi INT4 QAT, Jet-RL).
   - Deployed bytes every step = TE 4over6(RNE(main)); actor and rollout read the same bytes, so consistency holds naturally.
   - Cost: the deployed values at step 0 equal requant(BF16), which may not be byte-identical to the official NVFP4 checkpoint. If the official checkpoint was quantized from this same BF16 with TE 4over6, it is byte-identical; this needs verification.
   - Prerequisite: access to the BF16 weights of Lightning (or the master used during QAT).
   - **(f′) If unavailable, deterministic in-cell dithering of θ0:** θ0' = θ0 + u·(half-step toward the neighboring grid point), with u uniform within ±0.995, and no dithering on block-amax elements; after requant, check per block that the bytes are unchanged, and revert to undithered for any block whose bytes changed.
     - Deployed bytes at step 0 are completely unchanged; neither the quantizer nor vLLM needs changes.
     - In simulation, under 4over6 about 45% of blocks must revert because of candidate switching; dithered blocks have a drift projection of about 0.3–0.46 (r = 0.001–0.01), versus 0 without dithering.
     - Cost: the master used in backward is offset from θ0 by at most the order of the quantization error. STE already makes the master differ from the deployed values, so this introduces no new contract issue.
2. **(c) Fixed-seed SR requantization.**
   - Each element's threshold u_i is generated by a counter hash of (tensor id, element index, seed), and **is kept fixed across steps**, so codewords do not jitter from step to step.
   - Actor and rollout read the same exported bytes, so consistency holds naturally.
   - Simulation: starting from grid points with r = 0.003, projection is 0.93 and error relative to master is about 0.028σ.
   - Three costs:
     - In the long run, deployed quantization error is about 1.4x that of RNE (simulated 0.124σ vs 0.088σ), offsetting part of the gain from 4over6;
     - A custom quantizer is needed, because TE's weight forward hardcodes `stochastic_rounding=False`;
     - There is no mainstream precedent, and the NVFP4 paper's data even points the other way (though that is SR resampled every time, used in pretraining).
3. **(b) Increase lr only for the NVFP4 parameter group.**
   - Under Adam this is equivalent to QuRL's UAQ: block scaling is invariant, W/s does not change where the code sits within its cell, it only amplifies the relative step size.
   - To start flipping codes within about 300 steps needs roughly 10x (1e-5), which is already outside the common range for full-parameter RL training (order 1e-6), with high stability risk.
   - Starting from grid points, it only shortens the dead zone and cannot eliminate it.
   - Only suitable combined with (f′)/(c), with a modest increase.
4. **(a) Accept and document.** Whichever fix is chosen, document that: starting from a dequantized checkpoint at lr 1e-6, NVFP4 deployed values do not change for the first roughly 10³–10⁴ steps.
5. **(d) Switch the training forward to high precision.** This is the mainstream practice of FlashRL, QuRL and NeMo-RL TE per-token, but it gives up true on-policy and needs TIS or IS.
6. **(e) NVFP4 frozen base plus high-precision LoRA** (QeRL).
   - Consistency is achievable, but rollout must carry LoRA, taking the NVFP4 MoE plus LoRA path in vLLM, and bitwise alignment is a large amount of work.
   - Learning capacity is limited by the rank.

**Recommendations for the user:**
1. First ask whether the BF16 weights of Lightning (or the QAT master) are available; if so, use (f). It is consistent with mainstream practice and the lowest-cost option.
2. If not, use (f′).
3. In either case, rerun 300 steps of proxy SFT and compare val NLL and the projection of deployed-value movement against C.
4. (c) as a fallback.
5. Regardless of the outcome, write (a) into the README.

## Sources
- NeMo-RL: `docs/guides/quantization-aware-rl.md:1-31,131-209`; `docs/design-docs/te-nvfp4-per-token-rollout.md:18-26,43-58,120-160`; `docs/fp8.md:15-16,41`; `examples/modelopt/qa_grpo_nano3_megatron.yaml:16`; `examples/configs/recipes/llm/grpo-qwen3-30ba3b-base-8n4g-megatron-te-nvfp4-pertoken.yaml:21-23,57`
- TransformerEngine: `transformer_engine/common/recipe/__init__.py:505-551,598-608`; `docs/features/low_precision_training/nvfp4/nvfp4.rst:73-108`
- NVFP4 pretraining: https://arxiv.org/abs/2509.25149
- Four Over Six: https://arxiv.org/abs/2512.02010
- QeRL: https://arxiv.org/abs/2510.11696
- QuRL: https://arxiv.org/abs/2602.13953
- Jet-RL: https://arxiv.org/abs/2601.14243
- Reasoning QAT: https://arxiv.org/abs/2601.14888
- FlashRL: https://github.com/yaof20/Flash-RL
- INT4 QAT RL (slime): https://lmsys.org/blog/2026-01-26-int4-qat
- Kimi K2 Thinking: https://huggingface.co/moonshotai/Kimi-K2-Thinking
- vime: https://docs.vllm.ai/projects/vime/en/latest/advanced/low-precision.html
- verl DS4: https://github.com/verl-project/verl/pull/7688
- Simulation script: a local `sim.py` (Gaussian synthetic weights, not a real checkpoint; not included)
