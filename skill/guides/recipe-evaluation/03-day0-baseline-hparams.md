# Day-0 GRPO baseline hyperparameters

The first formal GRPO recipe for the new-model day-0 route (Miles as reference; DAPO-Math-17K training, AIME-2024 validation) takes its values from here; the defaults in the day-0 support workflow's GRPO profiles point to this page. The train-inference consistency route aligns to verl#7688; that route's values (for example Nemotron-H's betas `[0.9, 0.95]` and no std division) are not constrained by this page, and only the old logprob row also holds for it. There are three tiers: **fixed** items are copied directly with no ablation; **default** items are copied, may be changed with a reason, and changes go into the deviation table in `cases/<model>/recipe.md`; **not fixed** items require choosing a reference for that model per [01](01-recipe-alignment.md). If the model already has a validated recipe, that recipe wins and this page only covers what it does not specify; the only item that does not follow the reference is the AIME max output (§2).

A fixed-tier item may take another value in only two cases: the model's own validated recipe (a blog experiment with run results, or a command marked validated) specifies another value, in which case follow it; or the framework or inference engine in use cannot do it (meaning it cannot be implemented; for things that must be written in the framework but have not been yet, see the shared-outer-factor paragraph in §4). Both are recorded in the deviation table in `cases/<model>/recipe.md` and confirmed in G0. Launcher defaults, another model's recipe, or wanting to try one more configuration do not count as reasons. The old logprob row in §1 has neither exception: every model recomputes it every round.

Selection criterion: stated in the LMSYS blog body and cross-checkable against recipes in the Miles repo; for items the blog does not state, all Miles recipes corresponding to the blogs must agree. The denominator in the table is the number of recipes that specify the item: most items are specified by all 17, validation-related items by only 13. Statistical method and limitations are in §5.

## 1. Fixed

| Item | Value | Blog | Miles (the 17 recipes in §5) |
|---|---|---|---|
| Algorithm | GRPO, divide by std within group | ROCm, MXFP8/NVFP4, Model B, Model F, AMD DeepSeek-V4 | 17/17; std normalization on by default, nobody turns it off |
| Samples per prompt | 8 | MXFP8/NVFP4, Model B, another large MoE model, AMD DeepSeek-V4, ROCm | 13/17, the rest are scaled-down 4 |
| optimizer | Adam | Model B | 17/17 |
| lr schedule | constant, no warmup | — | 17/17 |
| weight decay | 0.1 | — | 17/17 |
| Adam beta1 / beta2 | 0.9 / 0.98 | — | 17/17 |
| KL penalty | 0 | MXFP8/NVFP4 (coefficient 0.0), ROCm (no KL loss) | 17/17; 10 keep the KL path on for diagnostics only |
| entropy coefficient | 0 | — | 17/17 |
| Gradient clipping | 1.0 | — | 17/17 use the `--clip-grad` default 1.0, nobody changes it |
| PPO clip | lower 0.2, upper 0.28 | — | 16/17 explicitly write 0.2 / 0.28; the other 1 does not set these two parameters and uses the default (lower 0.2, no separate upper) |
| chat template | on | — | 17/17 |
| PPO old logprob | training engine recomputes for all samples every rollout round; rollout-returned logprobs are not used | — | 17/17 do not set `--use-rollout-logprobs` (no reuse of rollout logprob seen); 0 of the 98 launchers in the whole checkout either |
| R3 routing replay (MoE models) | on | — | 8/17 (the other 9 include non-MoE models and early recipes); this item is a project decision, not the Miles majority value; reason below the table |

Why R3 must be on: for MoE models the training side and rollout side are not numerically identical (quantization, kernels, precision), so the same token may select different experts on the two sides; the logprob computed on the training side is then not the policy that rollout sampled from, and every step carries this part of the train-inference difference. R3 makes the training side use the expert ids recorded by rollout directly, eliminating the part caused by routing differences (numerical differences inside experts remain); TIS only reweights by ratio and does not change which experts the training side uses. Non-MoE models have no routing, so it does not apply. Checkpoints for enabling R3 are in [bi-switch-recipe](bi-switch-recipe.md#bi0-arm-integration-pitfalls); after start, confirm that the routing returned by rollout is not all 0 and that every MoE layer on the training side uses replay.

The old logprob row holds for both GRPO and GSPO, and does not deviate with a model's validated recipe (acceptance item 5 of the day-0 support workflow). Other than it and R3, items in this tier hold only for GRPO. Miles GSPO recipes use a clip three orders of magnitude smaller (Qwen3-235B and Qwen3-Next are 4e-4, GLM-4.5 is 1e-4 / 2e-4, see §5 of the Miles recipe audit, not included here); do not apply 0.2 / 0.28. This page gives no GSPO values; take them from the chosen reference recipe.

### 1.1 When training with LoRA

When training with LoRA instead of full parameters, the five items below are also fixed, with the same deviation rules as §1 (see top of page). The statistics cover Miles single-policy LoRA recipes for math GRPO (excluding terminal-use and multi-LoRA examples), which overlap with the 17 in §5 in only 3 (the LoRA recipes of Model G-1, Model G-2, Kimi-K2.5). Two sources: the LoRA list in §4 of the Miles recipe audit (not included here; it grades evidence as B / V / D in its §2.3), and the literal parameters of the `examples/lora/` scripts and Model G-1 / G-2 LoRA launchers read one by one on 2026-10-11 in Miles checkout `bf5d45a` (module names, layout switch, and dropout are three items that audit's §4 does not list line by line). Counts in the table are numbers of recipes, not numbers of independent experiments.

| Item | Value | Miles |
|---|---|---|
| Modules with LoRA | attention linear projections + all MLPs (dense layers, shared experts, routed experts). Not: router, embedding, `lm_head`, norm, and modules not in the training forward that get no gradient (e.g. DSA indexer). List the actual module names added per model in the recipe | Large MoE launchers are all "attention + all MLP": Kimi-K2.5, Model C, Model G-1 / G-2 / G-Flash, Qwen3.5 / 3.6 (the attention part follows each model's structure, including GDN, DSA / MLA, KDA projections; Model G excludes the 3 DSA indexer modules by default). Model B and small dense model examples write `all-linear` directly. Two exceptions: the gpt-oss-20B example only adds MLP, Model F (2.4T) only adds attention q/k/v/o. These are launcher parameters; they do not mean each recipe's existing results cover all MLPs (the old Model G-2 results exclude the experts' `down_proj`, see §10 of the Miles recipe audit, not included here) |
| MoE expert factor layout | shared outer factors (shared-outer) | 7 enable it: Kimi-K2.5, Model C, Model G-1, Model G-2, Qwen3.5 / 3.6, Model B, Model B-Small (same launcher as Model B); 3 are per-expert independent: AMD Qwen3-30B, Model G-Flash, GPT-OSS-20B |
| rank | 32 | The 4 with blog or validated results are all 32 (Model C blog experiment, Model F (2.4T) validated, Model B-Small validated, AMD Qwen3-30B validated); 16 appears in launcher defaults and in one bring-up still being cleaned up (Model G-Flash, only 20 steps of training reward, no held-out); 64 appears in only one example script |
| alpha | 64 (2× rank) | Among the 4 with results it is 2 vs 2: 32/64 for the Model C blog experiment and Model F validated, 32/32 for Model B-Small and AMD Qwen3-30B. Among launcher defaults 16/32 (also 2×) is most common; of the 8 `examples/lora/` scripts, 7 are 32/32 and 1 is 64/32 |
| LoRA dropout | 0 | All 8 `examples/lora/` scripts and the Model G-1 / G-2 LoRA launchers are 0.0 |

What shared outer factors are: a routed expert has three linear layers, gate, up, and down, and each layer's LoRA delta is B·A, six factors in total. gate's A, up's A (next to the expert input), and down's B (next to the expert output) are outer factors; all routed experts in the same layer share one copy of each, three copies in total; gate's B, up's B, and down's A are inner factors, one per expert. In the per-expert independent layout all six factors are one per expert. gate and up do not share with each other (in implementations that fuse gate/up into one linear layer they are already the same A); shared experts and dense layers do not participate. This matches the definition of vLLM `--enable-moe-shared-loras`.

All five items are project-chosen baselines. dropout, rank, and module scope are basically consistent among recipes with results; Miles itself has no consistent conclusion on the other two:

- **Shared outer factors** is the numerical majority (7 vs 3), but the only direct comparison in Miles favors per-expert independent: AMD Qwen3-30B, lr 2e-4, fixed held-out set, independent 0.793→0.824, falling back after switching to shared with the same parameters (§7.2 of the Miles recipe audit, not included here). That recipe is validated, so for Qwen3-30B use per-expert independent per the top-of-page rule, not the shared layout here. Sharing changes the optimization problem; you cannot assume the same rank / alpha / lr gives the same training strength.
- **alpha 64** is 2 vs 2 among recipes with results. 64 was chosen to match the recipe closest to the default task (Model C's blog: large MoE, shared layout, DAPO training, AIME 4096 validation); this is a project trade-off, not an evidence-based conclusion.

Without a validated recipe for the model, do not change these five items because some launcher defaults to 16/32 or because of the AMD comparison; to change them, go through the deviation table per the top-of-page rule.

Learning rate is not determined by layout; see the LoRA lr row in §2. The defaults in all three training frameworks and the inference engine are per-expert independent, and shared must be enabled explicitly; on verl FSDP it must also be implemented yourself; what to do before it is implemented is in §4.

## 2. Default

| Item | Default | Basis | When to change |
|---|---|---|---|
| Full-parameter lr | 1e-6 | Model B's blog; Miles full-parameter 14/14 | The model has a validated value (Model B-Small is 5e-5) |
| LoRA lr | 1e-5 | Model C's blog; all 3 LoRA recipes among the 17 in §5 (Model G-1, Model G-2, Kimi-K2.5) | Use the model's validated value if it has one. Of the 7 shared-outer-factor recipes, 5 are 1e-5, but only Model C has a held-out improvement result; the other 4 are launcher or example defaults (Model F (2.4T) validated, which does not share experts and only adds attention, is also 1e-5); the other two are the Model B launcher default 5e-6 and Model B-Small validated 2e-4 (the author says 5e-6 needs many rollouts before change is visible). Per-expert independent AMD Qwen3-30B is also 2e-4 (§7 of the Miles recipe audit, not included here). So 1e-5 is a starting point aligned with Model C's blog, not a Miles consensus. How to judge and what to change it to when validation score does not rise: see below the table |
| Prompts per step `P` | 32 | ROCm, AMD DeepSeek-V4; Miles 10/17 | The chosen reference recipe gives another value, or cluster size differs (in the blogs Model C uses 64 responses per step, DeepSeek-V4.1 uses 128 per step) |
| rollout max generation length | 4096 | DeepSeek-V4, AMD DeepSeek-V4, Model C, Model B | Small/medium model blogs use 8192 (Qwen3-30B, another large MoE model) |
| Validation interval | every 10 steps | INT4 QAT, AMD DeepSeek-V4, Model C | Miles repo default is 20; fall back to 20 when validation is too expensive |
| AIME samples per problem | 8 | AMD DeepSeek-V4; Miles 4/13 | 16 is more common in the Miles repo (9/13); Model C's blog uses greedy once. A greedy single-run score can only be a multiple of 1/30 and is not the same score as pass@1 over 8 samples ([02 evaluation contract](02-evaluation-and-acceptance.md#evaluation-contract)) |
| AIME max output | 4096 | DeepSeek-V4, AMD DeepSeek-V4, Model C (all three that state it use 4096); Miles 3/13 | This item does not change with the reference recipe or with rollout length: if the reference gives 16384 (e.g. AMD Qwen3 LoRA), or rollout uses 8192 per the reference, validation still uses 4096; reason below the table. Scores obtained this way are not on the reference recipe's eval basis and are not directly compared with its numbers. 16384 is more common in Miles repo launchers (10/13), but no blog's curves use it. Frozen in G0, not changed during the run |

The 4096 cap is the setting of this task, not a compute-saving compromise. Without a length cap, current large models already start very high on this problem set, leaving little room for improvement, so you cannot measure whether training helped (how high the start is must be measured for the model and written into the recipe); only with limited output is the starting score low, and what training must learn is exactly how to solve problems within a short budget. So truncation is expected: in the AMD DeepSeek-V4 blog 55%–60% of AIME answers hit the cap, and in Model C's blog overlong correct solutions counted as wrong — both belong to this setting. By default both rollout and validation are 4096, so the training objective and the eval basis agree; relaxing validation length raises the starting point and compresses the measurable improvement. Report truncation rate and accuracy of completed answers (one model: 190 steps mean@8 0.375 → 0.658, completion rate 0.40 → 0.75, accuracy of completed answers 0.94 → 0.88, the improvement almost entirely from completing more); scores at 4096 and at longer budgets are not the same score and are not compared with each other (§8.3 of the Miles recipe audit, not included here).

When LoRA training scores do not rise, check in this order; the first three steps need no rerun:

1. Whether updates are effective: what fraction of groups have reward variance (groups that are all-correct or all-wrong have advantage 0), whether the optimizer actually stepped, whether the adapter was actually synced to the rollout side ([LoRA guide three gates](../lora/README.md#1-three-gates-can-forward-is-not-full-support)).
2. How large the update is: compare the adapter change between two checkpoints 10 steps apart, or the logprob change on the same fixed batch before and after an update. Under Adam each parameter moves at most about one lr per step; 1e-5 for 200 steps moves a single parameter at most on the order of 2e-3.
3. Do not judge from `ppo_kl` and the clip-trigger fraction: with only 1 update per rollout, these two numbers are computed before the update and are always 0, regardless of lr. grad_norm cannot be compared across runs either ([01 §3](01-recipe-alignment.md#3-read-the-final-cli-take-validated-values)).
4. Only when the first two steps confirm updates are effective but small, and by the second validation (20 steps) the score is still within noise ([02 paired acceptance](02-evaluation-and-acceptance.md#paired-acceptance)), consider changing the learning rate: start a new run, change only this item, record it in the deviation table, and do not change layout, rank, or updates per round in the same run. Miles LoRA recipes with held-out results use only two lr values: 1e-5 (Model C's blog, Model F (2.4T) validated) and 2e-4 (Model B-Small, AMD Qwen3-30B, both validated), with no recipe in between; if you change, change to 2e-4; taking another value requires writing the basis in the deviation table.

## 3. Not fixed

| Item | Why |
|---|---|
| loss reduction (sequence-equal / token-equal weighting) | Miles has real recipes for both, see §1 of the Miles recipe audit (not included here); verl default is `token-mean`. Of the 7 shared-outer-factor LoRA recipes, 5 use sequence-equal weighting (Qwen3.5 / 3.6, Model C, Model B, Model B-Small, Kimi-K2.5) and 2 use token-equal weighting (Model G-1, Model G-2) |
| rollout temperature | In Miles 11 use 1 and 6 use 0.8; the blogs do not state it |
| Total steps, checkpoint interval | The blogs use 60 / 80 / 100 / 450 steps, each the result of the budget at the time, not the convergence step count; checkpoint interval is set by the job time limit |
| global batch, optimizer steps per rollout | `P` has a default, but there is no consensus on how many updates the `P×8` responses are split into (with the same 32×8, some do 1 update and some 2), see §8.1 of the Miles recipe audit (not included here) |
| TIS, indexer replay | About half of Miles recipes enable TIS (8/17); when token-level TIS is mandatory is in [Gap growing during training §3](../debugging-mismatch/07-drift-during-training.md#3-comparison-of-three-corrections), not decided by the reference recipe. R3 is not in this tier, see §1 |

## 4. How to write it in verl

The following verl defaults differ from §1 and must be written explicitly, otherwise verl defaults are silently used:

| Item | verl default | Write as |
|---|---|---|
| `rollout.enable_rollout_routing_replay` (MoE) | `false` | `true` |
| `actor.<engine>.router_replay.mode` (MoE; `<engine>` is `megatron`, `veomni`, or `fsdp_config`) | `disabled` | `R3` |
| `actor.optim.betas` | `[0.9, 0.999]` | `[0.9, 0.98]` |
| `actor.optim.weight_decay` | `0.01` | `0.1` |
| `rollout.expert_parallel_size` (MoE with EP enabled in vLLM) | `1` | vLLM's TP×DP; writing only `engine_kwargs.vllm.enable_expert_parallel` gets reset to `false` by verl ([recipe alignment §3](01-recipe-alignment.md#3-read-the-final-cli-take-validated-values)) |

Write the rest per the block below, including those equal to verl defaults, to guard against upstream default changes:

```yaml
actor_rollout_ref:
  actor:
    clip_ratio_low: 0.2
    clip_ratio_high: 0.28
    entropy_coeff: 0
    use_kl_loss: false
    # loss_agg_mode is not fixed (§3): verl default token-mean, set per the reference recipe
    optim:
      # lr see §2: full-parameter 1e-6, LoRA starting point 1e-5
      betas: [0.9, 0.98]
      weight_decay: 0.1
      clip_grad: 1.0
      lr_scheduler_type: constant
      lr_warmup_steps_ratio: 0.0
    # MoE models. For megatron / veomni engines replace fsdp_config with megatron / veomni;
    # fsdp / fsdp2 use fsdp_config and need verl#8194 (see below)
    fsdp_config:
      router_replay:
        mode: R3
  rollout:
    n: 8
    enable_rollout_routing_replay: true  # MoE models
    calculate_log_probs: true  # returns rollout_log_probs, needed to compute the per-round train-inference gap
algorithm:
  adv_estimator: grpo
  norm_adv_by_std_in_grpo: true
  use_kl_in_reward: false
  rollout_correction:
    bypass_mode: false  # trainer recomputes old_log_probs every round; true reuses rollout logprobs directly
```

For `rollout_correction` this page only specifies `bypass_mode`; write TIS and the other keys per §3, and confirm from the resolved config which corrections are actually on. When enabling token-level TIS, write:

```yaml
algorithm:
  rollout_correction:
    bypass_mode: false
    rollout_is: token
    rollout_is_threshold: 2.0
```

How to write LoRA (§1.1). The verl FSDP path uses PEFT; rank, alpha, and module names go under `actor_rollout_ref.model`:

```yaml
actor_rollout_ref:
  model:
    lora_rank: 32
    lora_alpha: 64
    # example names; fill in per the model's actual attention and MLP linear layer names, expert linear layers must be listed too
    target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
```

`target_modules` only matches modules like `nn.Linear`. If expert weights are a single parameter with an expert dimension (rather than one `nn.Linear` per expert), name matching will not find them; use PEFT's `target_parameters` (verl FSDP passes `actor_rollout_ref.model.target_parameters` to PEFT; full parameter names are required and there are constraints on parameter shape; first confirm that the PEFT version in use supports 3D parameters with an expert dimension) or the model's own LoRA layer. PEFT's default implementation for 3D expert parameters merges the delta into the BF16 weight on every forward, which is a different algorithm from serving, so the forward must be replaced ([LoRA guide §4](../lora/README.md#4-minimal-implementation-principles-on-the-fsdp-side)). This only solves whether LoRA is attached; it does not create the shared-factor structure. Either way, before starting print the actual trainable parameter names and confirm every class of target module is there ([LoRA guide "actually trained" gate](../lora/README.md#1-three-gates-can-forward-is-not-full-support)). PEFT's `lora_dropout` default is already 0; with another LoRA implementation write 0 explicitly.

Shared outer factors are not the default anywhere; the training side and inference side must match:

| Location | Default | How to enable |
|---|---|---|
| Miles training side | per-expert independent | `--experts-shared-outer-loras` (also turns on the corresponding SGLang-side switch; a mismatch between the two fails an assertion) |
| vLLM | per-expert independent | Launch argument `--enable-moe-shared-loras`: the adapter is loaded as three tensors with an expert dimension per layer, `experts.w1` / `w2` / `w3`, with expert dimension 1 for shared factors. Decided at launch; cannot be mixed with the per-expert format |
| verl | per-expert independent | No switch in mainline. The FSDP + PEFT path has no shared-factor structure and no existing implementation to copy; you must write it yourself on the model's expert layers (outer factors stored once, inner factors one per expert); the Megatron path is in the unmerged [verl#7484](https://github.com/verl-project/verl/pull/7484), with the structure in Megatron-Bridge |

While the shared structure on verl FSDP is not yet written: few-layer smoke tests and pipeline-validation runs such as G4 / G5 may first run per-expert independent (first confirm expert LoRA is actually attached); formal long GRPO runs require the shared structure to be written first and to pass the checks below. If you really must run a formal long run per-expert independent, record it in the deviation table per the top-of-page rule, confirm in G0, and state the layout in the results.

After sharing on the training side, there are two ways to sync to vLLM: expand the shared factors into one per expert at export, so vLLM loads in the default format with no extra argument (the verl#7484 approach; sync volume is the same as per-expert independent); or export in vLLM's shared format and add `--enable-moe-shared-loras` (smaller sync volume, but the adapter format and checks must change together). For the first integration use the expanded approach.

Both approaches must be checked. Confirm with a non-zero adapter delta: when B is initialized to 0 no error is visible, and even after one training step there may be no update because advantage is all 0; sync tests can use a deliberately written deterministic non-zero adapter.

- Expanded export: within the same layer, all routed experts' gate `lora_A`, up `lora_A`, and down `lora_B` are each bitwise identical; inner factors are per-expert independent parameters (they need not differ pairwise in value; experts never routed to may be unchanged); vLLM does **not** add `--enable-moe-shared-loras`.
- Shared-format export: shared factors have expert dimension 1; vLLM adds `--enable-moe-shared-loras`.
- A layout mismatch between the two sides does not necessarily error. One case already encountered: vLLM had sharing enabled while the adapter was per-expert independent; it took expert 0's outer factors and broadcast them to all experts, silently dropping the other experts' values, showing only as a larger train-inference difference ([Model B case in the LoRA guide](../lora/README.md#2-one-adapter-contract-table-is-enough)). Other directions and other versions may fail to load or may not error; they have not been verified one by one. So check the actual launch arguments, and compare training-side and vLLM logprobs on fixed tokens.

Both R3 switches must be on together (the rollout-side switch makes verl launch vLLM with the argument for returning routing; no separate vLLM configuration is needed): with only the rollout side on, the training side does not replay; with only the training side on, the batch has no routing. The plain FSDP engine (`fsdp` / `fsdp2`) does not support R3 before [verl#8194](https://github.com/verl-project/verl/pull/8194) (as of 2026-10-11 the PR is unmerged; use its branch or apply the changes to the verl version in use), and it only passes `routed_experts` into the model `forward`; the model itself must implement selecting experts by the replayed ids; the megatron and veomni engines already support it.

verl's validation default is greedy, once per problem, with length shared with training via `data.max_response_length`; to run with the §2 validation settings, configure `rollout.val_kwargs` and data length separately.

## 5. Statistical method and limitations

- **Blogs**: LMSYS articles that state RL training settings, body text read on 2026-10-11. [ROCm Miles](https://lmsys.org/blog/2026-03-17-rocm-miles-rl-amd), [DeepSeek-V4](https://www.lmsys.org/blog/2026-04-25-deepseek-v4/), a blog on another large MoE model, [AMD DeepSeek-V4](https://www.lmsys.org/blog/2026-07-10-rocm-miles-dsv4/), the day-0 support blogs of Model B and Model C, [MXFP8/NVFP4](https://www.lmsys.org/blog/2026-07-29-mxfp8-nvfp4-rl/), the day-0 support blog of Model F, [DeepSeek-V4.1](https://www.lmsys.org/blog/2026-09-10-deepseek-v41/), [INT4 QAT](https://www.lmsys.org/blog/2026-01-26-int4-qat/). Blog bodies rarely state optimizer details; items with "—" in the "Blog" column have Miles as their only source.
- **Miles**: checkout `bf5d45a` on the GB300 cluster (2026-07-15; not the same directory as the local checkout `dbbab15` used for the separate Miles recipe audit, §2.2), taking the 17 recipes corresponding to the blog models above and reading the literal parameters in each launcher: `scripts/amd/run-qwen3-4B-amd.sh`, `scripts/amd/run_deepseek_v4.py`, `scripts/run_deepseek_v4.py`, 6 under `examples/low_precision/` (Qwen3-4B / 30B FP8, Qwen3-30B / 235B, Kimi-K2-Thinking, Moonlight INT4), Model G-2's full-parameter launcher, Model G-2 / G-1 LoRA, `scripts/run-kimi-k25.sh`, `examples/lora/run-kimi-k25-megatron-lora.sh`, another large MoE model's launcher, `scripts/run-qwen3.5-35B-A3B-mtp.sh`, `examples/on_policy_distillation/run-qwen3-8B-opd.sh`. "Which blog a recipe corresponds to" is matched by model name; the blogs do not name scripts.
- **Basis of the old logprob row**: searched the 98 run scripts under `scripts/` and `examples/` across the whole checkout for both spellings `use-rollout-logprobs` and `use_rollout_logprobs` (covering shell and Python launchers): 0 hits; the switch's description is "when not set, use the actor model's logprob", from which default recomputation is inferred; run logs were not checked recipe by recipe. This row counts "not set", unlike the other rows which count "what value is written".
- **Not covered**: this checkout predates the merge of the Model B, Model C, Model F, and DeepSeek-V4.1 recipes, so for these four models this page uses only blog bodies. The Miles recipe audit (not included here) is a separate audit (Miles main `d2fc97c`, 2026-09-04, including these models' launchers and PRs) that only checked LoRA lr, rank, loss reduction, and TIS/R3, not the §1 optimizer parameters item by item; the two audits have no contradicting items, but §1 has no launcher-level cross-check for these four models. Launcher defaults can be overridden by CLI; without the original run config you cannot infer the actual values of blog experiments (§2.3 of that audit). §1 only includes items with no second mainstream value: the 4 exceptions for samples per prompt are scaled-down smoke configs, and the 1 exception for PPO clip did not set the parameter.
- **What these values prove**: they are the common starting point of multiple public recipes, not optimal values we have validated. Results from running this page's config must still pass acceptance in [02](02-evaluation-and-acceptance.md).
