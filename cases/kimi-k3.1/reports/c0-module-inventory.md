# C0：模块清单与路线（草稿，2026-10-14）

来源：
- vLLM K3/K31 实现：GB300 集群 `$WORK/k31/image-src/vllm`（镜像源码，只在集群上读）。
- IsoExec：`consistency/reference/SkyRL-IsoExec`。
- 案例：DS4、Nemotron Lightning（内部仓库 cases/）。

行号相对镜像 vLLM 树。

## 1. 模块清单

| 模块 | 推理 kernel（K31 实际走的） | 训练侧同数学实现 | 量化取整 | 离散选择 | BI 现状 | backward |
|---|---|---|---|---|---|---|
| KDA prefill | 默认 FlashKDA（`_flashkda_C.abi3.so`，**闭源**），auto 选择 `kda.py:463-502`；可选 Triton FLA chunk（`kda_prefill_backend=triton`） | FLA `chunk_kda`（Megatron-Bridge 路线） | 无 | 无 | 无 BI 分支；Triton 路径有 TF32（`chunk_intra.py:635-639`，及未指定 `input_precision` 的 FP32 dot）、autotune 改 BK（`chunk.py:237-247`）、chunk 网格从 query 起点切（chunked prefill 依赖） | FLA chunk_kda bwd 可复用 |
| KDA decode | TP4/TP8 走融合 native `fused_kda_decode`（闭源 .so）；**TP16/TP32 走 Triton** `causal_conv1d_update` + `fused_recurrent_kda_packed_decode`（`kda.py:286-338`） | — | 无 | 无 | 每行独立 program，无 atomic / split-K，状态 FP32 | 不需要（训练只走 prefill 形态） |
| KDA 混合批 | 只要批里有 prefill，整段非 spec 的 decode 行也按长度 1 走 prefill 路径（`kda_metadata.py:436-449`） | — | — | — | **最大的批依赖源**：同一 decode token 随批组成换 kernel | — |
| KDA 状态 / 前缀缓存 | 状态 FP32 `[H,V=128,K=128]`；中途 checkpoint 只有 FlashKDA 提供（粒度 16） | — | — | — | Triton 路径无中途 checkpoint | — |
| KDA conv / gate / norm | conv 权重 FP32、状态 BF16；gate = −5·sigmoid(exp(A_log)·(g1+dt_bias))，逐通道，cumsum 后 ×RCP_LN2 用 exp2；输出 FusedRMSNormGated（sigmoid），prefill 输入 BF16 舍入后的 o | Megatron-Bridge 各自实现 | 无 | 无 | 需 matched prep：prefill 与 decode 门控同一函数 | 可微 |
| DSA attention | FlashMLA sparse（prefill / decode 两份），fp8_ds_mla KV；`sparse_mla_force_mqa` 固定单一路径 | `k31_mb` 自写（absorbed，FP32 scores） | FP8 KV、FP8 indexer q/k | indexer top-k 2048（并列不确定，overlay `dsa-topk-order` 可修） | 待查：sparse decode split-KV、head padding 6→64 | 训练侧可微实现已有 |
| MoE routed | Marlin W4A16（MXFP4，896 专家，EP 下 56/28 本地） | Megatron TE grouped GEMM（BF16 反量化） | MXFP4（E8M0 block 32） | router top-16 | 行对齐 atomicAdd 已有确定性 overlay；Marlin 本身 BI 未验证；**无现成 BI MXFP4 后端**（V4.1 结论） | 待定 |
| MoE shared / dense / latent proj | BF16 GEMM（TP 下 skinny GEMM 只在 TP8 + graph + M≤14，BI 下关闭） | TE | 无 | 无 | vLLM BI 模式覆盖 aten mm | TE / 可微 |
| LoRA | punica shrink/expand（split_k=64 默认）；专家 LoRA 在 xlora-ep overlay（FP32 相加） | Megatron LoRA（Bridge） | 无 | 无 | shrink split_k 要钉 1；两侧加法顺序要入契约 | 可微 |
| norm / RoPE / embedding | vLLM RMSNorm、RoPE | TE / Megatron | 无 | 无 | 照 IsoExec：钉 tile、FP32 cos/sin | 可微 |
| TP 归约 | NCCL all-reduce（TP16/32） | Megatron TP4 | — | — | 照 IsoExec pik 叶子树 / V4.1 `VLLM_TP_INVARIANT_LEAVES` | — |
| lm_head / logprob | vLLM logits + logprob | verl logprob | — | 采样 | 照 IsoExec rowinv 叶子树；raw logprobs | 可微 |

## 2. KDA 的 BI 方案要点（CPR 设计输入）

1. **不能用 FlashKDA / native decode 做 BI 基准**：两者闭源，无法证明块边界与递推顺序。BI 臂的 KDA 走源码可读的 FLA Triton 路径 + 现有 recurrent 内核，属于复用；CPR 若需要新内核，按 C10 用 CuTe DSL。
2. **CPR 结构**：
   - 块入口状态用 `chunk_gated_delta_rule_fwd_h` 的 gk 分支（`chunk_delta_h.py:229-275`）。它存的 h 是 BF16，需要拿 FP32 链或明确 snapshot 契约（IsoExec：snapshot BF16、chain FP32）。
   - 块内按 decode 顺序回放：`fused_recurrent.py:262-269` 的逐 token 形式，逐通道 `S·exp(g[None,:])`。
3. **注意 exp 与 exp2**：state pass 用 exp2(g·log2e)，recurrent 用 exp(gk)。CPR 只要求边界状态来自同一个 state pass，块内与 decode 同一递推，但要实测证明。
4. **混合批**：BI 臂必须让 decode 行无论批里有没有 prefill 都走同一递推。CPR 天然满足：prefill 也按 decode 顺序递推。
5. **固定项**：
   - TF32 → IEEE。
   - 所有 autotune 钉 configs[0]（IsoExec 做法）。
   - chunk 网格按绝对位置对齐（chunked prefill 不改变数值）。
   - 门控 matched prep：prefill 与 decode 同一函数。
6. **前缀缓存**：BI 臂下 Triton 路径无中途 checkpoint，需要 CPR 的 boundary store（FP32 入口状态 + conv 窗口），权重同步时清空。

## 3. 路线

- **训练侧：mLite（workflow 默认）**。理由：
  - vLLM K3 实现已覆盖 K31 全部特性（KDA、DSA + FP8 KV / indexer、MXFP4 MoE、LoRA、sm100）。训练复用推理 forward，只需给各模块接 backward。
  - IsoExec 是反方向（vLLM 跑 Megatron 模型），不覆盖 DSA / FP8 KV / MXFP4 / LoRA / sm100，PP=1、engine EP=1。只借它的算子做法：GDN CPR、pik 归约树、logprob 叶子树、autotune 钉配置、执行契约哈希。
- **mLite 现有组件**（GB200 集群 `opus-GB200 集群/final3-image/Megatron-LM/experimental/lite`，commit `71d08d277`，Nemotron 交付用的那份）：
  - 模型：`kimi_k2`（MLA + MoE）、`qwen3_5`（GDN）、`glm5` / `deepseek_v4`（DSA）、`nemotron_h`、`qwen3_moe`。
  - 模块：`gated_delta_net.py`（688 行）、`lora.py`（565 行）、`router_replay.py`、`moe.py` / `experts.py`。
  - kernel：`kernels/dsa_kernels.py`（1477 行）。
  - 适配层：`examples/verl/verl_mlite`。
  - K31 ≈ kimi_k2 的 MoE 骨架 + qwen3_5 的线性注意力位置换成 KDA + glm5 / DS4 的 DSA + lora，属于拼装。
  - 但 mLite 的 GDN 只是 torch 参考的 chunk 实现（`ops/gated_delta_rule.py` 108 行），不是 BI。KDA 的 BI forward 要按 IsoExec CPR 的做法新做。
- **待量化**：mLite 移植 K31 的工作量。
  - vLLM 的 KDA / DSA / latent MoE / LoRA 层能否直接当训练 forward：paged 状态、metadata、EP 通信。
  - 各模块的 backward 来源：FLA chunk_kda bwd、DSA 可微实现、MoE grouped GEMM bwd、LoRA。

## 4. 未覆盖 / 待查

- FlashMLA sparse decode 的 split-KV 与 head padding 是否批不变。
- Marlin W4A16 在不同 M 下是否批不变（不考虑 atomic 时）。
- mLite 现有接口（`experimental/lite/examples/verl/verl_mlite/`）在我们 Megatron 版本里是否存在、能否接 K31。
- 训练 master 起点：BF16 反量化 MXFP4 还是原始？θ0 字节是否等于官方 checkpoint（专家 MXFP4 冻结，LoRA 从 0 起，base 不变 → 天然相同，需确认）。
