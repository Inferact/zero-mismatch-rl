# C5：mLite 中 K31 的 vllm impl 布局（按 DS4 逐文件对照）

日期：2026-10-16。只读设计，没有改任何集群文件，没有动作业。

## 0. 读过的来源

| 来源 | 位置 | 读了什么 |
|---|---|---|
| DS4 mLite | GB200 集群 `$NV_HOME/megatron-nemotron-unified-pr/experimental/lite/megatron/lite/`（分支 `codex/nemotron-nvfp4-unified`，HEAD `53d9e539d8`） | `model/registry.py`；`model/deepseek_v4/{config,lite/*,vllm/*,vllm/primitive/*}`；`primitive/autograd/inference_only.py`；`primitive/recompute.py`；`primitive/alignment/*`；`primitive/kernels/vllm_ds4.py`；`primitive/modules/{gated_delta_net,lora}.py`；`tests/unit/model/deepseek_v4/vllm/**` |
| DS4 vllm impl 的提交历史 | 同仓库 `git log -- .../deepseek_v4/vllm` | 共 22 个提交，起点 `07bed65e3d`（2026-08-14，checkpoint: restore DS4 vLLM DeepEP LL forward baseline），作者 Yan Bai。之后依次加入 normal DeepEP 前后向对齐、RL 对齐、R3、CP、PP、PP 导出，终点 `1d20f29cc4`（2026-08-18，align SM100 batch-invariant forward paths）。包含这些提交的分支有 `iseekyan/pr226`、`iseekyan/ds4-sm100-ep4-postmerge`、`origin/ds4-rl-cp-r3-final-20260816` |
| K31 推理侧 | GB300 集群 `$WORK/k31/image-src/vllm`（镜像源码）；`k31/bi/overlays/*`、`k31/tpinv/overlay/tp-invariant`、`k31/serving/overlay/xlora-ep` | `models/kimi_k3/nvidia/{model,kda,dsa}.py`；`ops/attn_res.py`；`model_executor/layers/activation.py`（SituAndMul）；overlay 里的 `k31_bi_gemm.py`、`k31_kda_cfbi.py`、`mxfp4.py`（fmlora-policy 版）、`base_linear.py`（lora-f32add-bi-r3）、`k31_tree_allreduce.{py,cu}`、`latent_moe_runner.py`（moe-tree-ep2） |
| K31 现有训练侧 | GB200 集群 `$NV_WORK/k31/root/k31/mg/src/{Megatron-Bridge/.../models/kimi/kimi_k3_*.py,k31_mb}` | BF16 参考实现：`kda`（FLA `chunk_kda`）、`situ_and_mul`、`attn_res_aggregate`、DSA（k31_mb，indexer 冻结、fp8_ds_mla 直通） |
| 本地报告 | `agent_run/reports/{k31-bi-deploy,kda-cfbi-design,c0-module-inventory,c2-proxy-coverage}.md`、`results/c3-kda-cpr/summary.md` | 全部 |

没读到的：verl-project/verl#7688 和 ISEEKYAN V4 lite 的 PR 页面（本机 gh 未登录）。以 GB200 集群 仓库里包含同一批提交的分支为准。

## 1. DS4 的做法（K31 照抄的部分）

DS4 的 vllm impl 不构造 vLLM 模型对象，也不读 vLLM 的 forward context：

- 参数：每个模块自己建 BF16 master（`nn.Parameter`）。
- 前向：通过 `primitive/kernels/vllm_ds4.py` 里的 adapter，用 `_symbol(module, name)` / `_op(ns, name)` 直接调 vLLM kernel，没有数值兜底。
- 反向：每个前向调用外面包一个 autograd owner（`vllm/primitive/*.py`），分三种：
  1. 解析 VJP：linear 用 `fp32_linear_vjp`，RMSNorm 用 `_rms_norm_vjp`；
  2. 重算函数图：`_recompute.visible_functional_vjp`。前向值来自 vLLM kernel，反向在一个数学等价的 torch 图上求导，用于 mHC 和 o_proj；
  3. 厂商反向 kernel：FlashMLA sparse 的反向用 `dsa_kernels._DSA.sparse_attention_backward_wrapper`。
- 离散选择：
  - router 用 `fixed_route_vjp`：ids 固定，只对选中的分数求导；
  - indexer 的 top-k 用 `inference_only` 或 `.detach()`。
- 前向值的契约：`own_visible_tensor` 只 clone、不做任何运算；`parameter_versions` 在反向时检查 master 有没有被改过。
- 运行时元数据：`runtime_metadata.py` 自己构造 FlashMLA、indexer、compressor 的 metadata，并负责 `init_workspace_manager`、`DPMetadata`、`override_forward_context`。
- 并行契约：TP=ETP=VPP=1，可以开 PP 和 CP，EP 必须用 normal DeepEP。层选择器只接受从 0 开始的连续前缀，用于逐层审计。

K31 完全沿用这套结构。K31 的特点是**基座冻结、只训 LoRA**，所以 dense 和专家的 master 都不需要 dW，只需要 dX 和 LoRA 的 dA、dB。专家可以保持 MXFP4 打包形式，不需要 BF16 master。

## 2. 逐文件映射表

前缀：mLite 侧 `megatron/lite/`，K31 包名 `model/k31/`（registry 名 `k31`）。

### 2.1 模型包

| DS4 文件 | K31 文件 | 处理 | 理由 |
|---|---|---|---|
| `model/registry.py`（`register_model("deepseek_v4", impls={"lite", "vllm"})`） | 同一文件，新增 `register_model("k31", package="megatron.lite.model.k31", hf_model_types=[<K31 HF model_type>], impls={"vllm": "megatron.lite.model.k31.vllm.protocol"})` | 改 | 只注册 vllm impl，裸 key `k31` 直接指向 vllm protocol（registry 规定第一个 impl 作裸 key） |
| `deepseek_v4/__init__.py` | `k31/__init__.py` | 原样 | 空包 |
| `deepseek_v4/config.py` | `k31/config.py` | 改 | 字段换成 K31 的：93 层；`linear_attn_config.kda_layers`、`dsa_config.dsa_layers` 是 **1 起编号**，要转成 0 起；`first_k_dense_replace=1`；latent 3584 + `latent_moe_use_norm`；SiTU β=4 / linear β=25；`attn_res_block_size=12`；indexer 32 头 × 128、top-k 2048、`index_rope_dim=0`；`mla_use_nope`；`mla_use_output_gate` |
| `deepseek_v4/lite/{checkpoint,model,moe,protocol,resync}.py` | 不建 | 不做 | DS4 的 lite 是 BF16 独立实现（约 1.8k 行）。K31 已有 BF16 参考（Megatron-Bridge K31 adapter + `k31_mb`），一致性目标只需要 vllm impl；vllm impl 也不依赖 lite（DS4 有测试 `test_package_has_no_sibling_lite_imports`）。LoRA 训练只同步 adapter，`lite/resync.py` 里的整模型 FP8 重量化不需要 |
| `vllm/__init__.py` | `k31/vllm/__init__.py` | 原样 | 导出 `K31Model`、`ImplConfig`、`SelectorConfig`、metadata builder |
| `vllm/protocol.py` | `k31/vllm/protocol.py` | 改 | 骨架照搬：`SelectorConfig`、`ImplConfig`、`_finalize_replica_grads`、`_roll_packed_targets`、`_forward_step`、`build_model`（含 `init_workspace_manager`、`ensure_runtime_assets`）、`unpack_forward_output`、R3 打包。<br>`_validate_contract` 改为：TP=ETP=VPP=CP=1，可开 PP；EP 必须整除 896；必须开 DeepEP；必须带 LoRA 配置；不开 MTP。<br>stage 名换成 `{attn_res, linear, kda, dsa, router_moe, deepep}`。<br>`ImplConfig` 加 `lora: LoraConfig`（复用 `primitive/modules/lora.py` 的 dataclass）；删掉 `dsa_indexer_loss_coeff`，因为 indexer 冻结 |
| `vllm/model.py` | `k31/vllm/model.py` | 大改 | 结构同 DS4：`_AttentionState` → 两个类 `_KDAState` 和 `_DSAState`；`DeepseekV4Layer` → `K31Layer`；`DeepseekV4Model` → `K31Model`。mHC 换成 attn_res 残差库：块写层与 `prev_valid_blocks` 的规则照抄 vLLM decoder layer。head 是「attn_res 输出聚合 → final RMSNorm → lm_head → `_vllm_selected_log_probs`」，最后这个函数原样保留。PP 的 fold/unfold 由 mHC 的 4 条流改成 `prefix_sum + bank[8]` |
| `vllm/moe.py` | `k31/vllm/moe.py` | 大改 | 保留 `_RouterState`、`_replay_route`（R3）、`TokenDispatcher`（normal DeepEP）。<br>新增：latent down/norm/up、2×shared SiTU MLP、FI W4A16 专家、专家 LoRA、canonical-slot combine。<br>删除：hash 路由层和 sqrt-softplus 打分 |
| `vllm/native_cp.py` | 不建 | 不做 | DS4 的 CP 是为压缩注意力做的。K31 的 KDA CF-BI 是按序列串行的状态递推，CP 需要跨 rank 交接 E 和行缓存，没有现成实现。rollout `max-model-len` 5120，单 rank 放得下。契约直接拒绝 CP>1 |
| `vllm/runtime_metadata.py` | `k31/vllm/runtime_metadata.py` | 改 | 照搬：`initialize_*_batch_invariance`（加检查：`k31_bi_gemm.enabled()`、`k31_kda_cfbi.enabled()`、MoE tree 状态）、`*_vllm_forward_context`、`_RuntimeGateLinear` 模式。<br>新建：DSA 用 V3.2 型 `sparse_attn_indexer` 加 FlashMLA sparse prefill 的 metadata builder（只是改写 DS4 的 `DS4SparseIndexerCompressorMetadataAdapter`，去掉 compressor 和 SWA）；KDA 训练用 metadata（每条序列 ctx=0，state 池按序列临时分配，见 3.5） |
| `vllm/checkpoint.py` | `k31/vllm/checkpoint.py` | 大改 | `WeightSpec` 和 PP 导出骨架照搬。<br>名字映射按 vLLM 的融合布局建 native 参数：KDA `in_proj_qkvgfab` 拼接、DSA `fused_qkv_a_proj`、shared `gate_up`。这样每个可见 GEMM 和 vLLM 是同一个 (K, N)。<br>专家按 MXFP4 打包原样加载（uint8 + E8M0），不反量化。<br>LoRA adapter 按 vLLM `--lora-modules` 的格式加载和导出。<br>`_is_block_fp8_weight`、FP8 重量化导出整段删除 |
| `vllm/primitive/__init__.py` | `k31/vllm/primitive/__init__.py` | 改 | 导出列表按下面几行调整 |
| `vllm/primitive/_contract.py` | 同名 | 原样 | `own_visible_tensor`、`parameter_versions`、`fp32_linear_vjp` 与模型无关 |
| `vllm/primitive/_recompute.py` | 同名 | 原样 | `visible_functional_vjp` 是 attn_res、SiTU、KDA 输出 norm 的通用 owner |
| `vllm/primitive/linear.py` | 同名 | 改 | `visible_linear` 的前向换成 `bi_linear`；`gate_linear` 换成 `bi_router`；删掉 `block_fp8_linear`；在同一文件加 `visible_lora_linear`，即 LoRA 叶子（3.2）。因为基座冻结，VJP 不算 dW |
| `vllm/primitive/norm.py` | 同名 | 基本原样 | `rms_norm`（vLLM BI RMSNorm + 解析 VJP）原样。`fused_qkv_rms_norm` 在 K31 里对应 DSA 的 q_a/kv_a 两个 norm，按 vLLM 实际走的 kernel 决定保留还是退化成两次 `rms_norm` |
| `vllm/primitive/mhc.py` | `attn_res.py` | 替换 | 位置和角色都是 DS4 的 mHC（层间残差混合），文件数不变 |
| `vllm/primitive/o_proj.py` | 不建 | 删 | DS4 的 o_proj 是带逆 RoPE 的分组 FP8 投影；K31 两种注意力的 o_proj 都是普通 BF16 linear，走 `linear.py` |
| `vllm/primitive/attention.py` | 同名 | 改 | 保留 `_VisibleSparseAttentionFunction` / `attention_core` 的写法：可见调用里定下来的 indices 和 lse 要保存下来，反向调厂商 kernel，并检查 finite。删掉 compressor、SWA、逆 RoPE 重放和 `attach_indexer_aux_loss`。加上 MLA 吸收形式（W_UK / W_UV bmm）和 `g_proj` sigmoid 门 |
| `vllm/primitive/router.py` | 同名 | 改 | `fixed_route_vjp` 的重放从 sqrt(softplus) 改成 sigmoid：bias 只参与选择，renormalize，scale=1.0 |
| `vllm/primitive/moe.py` | 同名 | 大改 | DS4 的 `deep_ep_moe` 把整个 DeepEP 当成可见调用。K31 改成 FI W4A16 专家 + 专家 LoRA 的 owner，再加 canonical-slot combine 的 owner |
| — | `vllm/primitive/kda.py` | 新增 | DS4 没有线性注意力。69 层 KDA 的可见前向（conv → CF-BI → gated norm）和 FLA 反向必须有一个 owner，放进任何现有文件都会混淆职责 |

### 2.2 共享层（`primitive/`）

| DS4 文件 | K31 | 处理 | 理由 |
|---|---|---|---|
| `primitive/autograd/inference_only.py` | 复用 | 原样 | indexer 的 top-k 输出、KDA 状态池等只用于推理的边界 |
| `primitive/recompute.py` | 复用 | 原样 | K31 的 `_RECOMPUTE_MODULE_MAP` 放在 protocol 里，键改成 `kda`、`dsa`、`moe`、`experts`、`router`、`attn_res` |
| `primitive/kernels/vllm_ds4.py` | `primitive/kernels/vllm_k31.py` | 新文件，对应 DS4 的同名文件 | 所有 vLLM、FlashInfer 和 overlay kernel 的 `_symbol`/`_op` adapter 都放这里，规则同 DS4：不构造 vLLM 对象，不做兜底 |
| `primitive/alignment/vllm_grouped_moe.py` | `primitive/alignment/vllm_fi_w4a16_moe.py` | 新文件，对应 DS4 的同名文件 | 角色相同：可见的 grouped 专家前向 + BF16 反向。kernel 和量化格式都不同，不能共用一个文件 |
| `primitive/alignment/deepep_route.py`、`deterministic_route_kernels.py` | 复用 | 原样 | dispatch 只搬数据，逐位无关 |
| `primitive/alignment/{vllm_batched_prepare,deepgemm_backward}.py` | 不用 | — | 只服务 DS4 的 DeepGEMM FP8 |
| `primitive/quantization/mxfp4.py`（`dequantize_mxfp4`） | 复用 | 原样，需核对 | 专家反向的 dX 用它反量化。要先核对它的 E8M0 和 nibble 顺序与 FI 的 W4A16 解释一致 |
| `primitive/modules/lora.py`（`LoraConfig`、`freeze_non_lora_params`） | 复用 | 原样 | 只用配置和冻结工具。不用 `LinearLoRA`，因为它的前向数值不是 vLLM 的 |

### 2.3 测试（`tests/unit/model/deepseek_v4/vllm/` → `tests/unit/model/k31/vllm/`）

| DS4 测试 | K31 对应 | 说明 |
|---|---|---|
| `test_linear_bitwise.py` | 同名 | `bi_linear` / `bi_router` 的 adapter 与直接调 overlay 逐位相同；覆盖所有 K31 (K, N) |
| — | 并入 `test_linear_bitwise.py` | LoRA 叶子：`lora_f32add` 加 punica delta，与 vLLM `BaseLinearLayerWithLoRA._apply_sync` 逐位相同 |
| `test_mhc_bitwise.py` | `test_attn_res_bitwise.py` | nb 取 0–8；包括块写层和 head 聚合 |
| `test_kv_flashmla_bitwise.py` | 同名 | fp8_ds_mla 写入 + FlashMLA sparse prefill（MLA 吸收、g_proj 门） |
| — | `test_kda_bitwise.py` | 训练整段 prefill 与 rollout 逐 token decode 输出逐位相同（G1） |
| `test_router_moe_bitwise.py` | 同名 | router + FI W4A16 + 专家 LoRA + canonical combine + latent tail |
| `test_o_proj_bitwise.py` | 删 | 没有对应模块 |
| `test_deepep_ep2.py`、`test_grouped_deepep_moe.py` | 同名 | EP2 下 combine 结果与单卡 canonical tree 逐位相同 |
| `test_checkpoint_fp8.py` | `test_checkpoint_mxfp4_lora.py` | 打包专家原样加载；LoRA 导出和 vLLM 加载的结果逐字节相同 |
| `test_runtime_metadata.py`、`test_static.py`、`test_vllm_skeleton.py` | 同名 | 契约、selector、fail-closed 检查 |
| `backward/test_modules_backward.py` 等 | 同名 | 每个 owner 的 VJP 与 FP32 参考对比，覆盖 KDA、attn_res、专家 dX 和 LoRA 梯度 |

### 2.4 结构上必须新增的东西

只有以下三项，都是 DS4 里没有对应物的模块；没有新增 manifest、yaml 目录或校验层：

1. `vllm/primitive/kda.py`：DS4 没有线性注意力。
2. `vllm/primitive/attn_res.py`：替换 `mhc.py`，不算净增。
3. `primitive/kernels/vllm_k31.py` 与 `primitive/alignment/vllm_fi_w4a16_moe.py`：和 DS4 的两个文件一一对应，不算净增。

LoRA 不单独建文件：dense LoRA 放在 `linear.py`，专家 LoRA 放在 `moe.py`。

## 3. 逐 primitive：前向 kernel、反向方案、逐位契约

下文的 overlay 路径都是挂载后在镜像 vllm 包里的位置。训练进程必须挂和 rollout 完全相同的 overlay 链（k31-bi-deploy.md §2 的 1–20 号，顺序相同）。

### 3.1 BF16 dense linear（基座冻结）

- **前向**：`vllm.models.kimi_k3.nvidia.k31_bi_gemm.bi_linear(x, w)`。
  - K≤3584：走 `_mm_s1_op`（cuBLASLt，S=1）。
  - K>3584：走 `k31bi::gemm`（tcgen05 split-K）。
  - vLLM 侧 `vllm.model_executor.determinism.batch_invariant.linear_batch_invariant` 按 `_k31_bigemm_take` 规则转发到同一函数。注意 DS4 引用的是 `vllm.model_executor.layers.batch_invariant`，K31 镜像里这个模块的路径是 `determinism`。
- **用到的形状**：

  | 模块 | K → S |
  |---|---|
  | KDA in_proj | 7168 → S=8 |
  | KDA f_b | 128 → S=1 |
  | KDA o_proj | 12288 → S=8 |
  | DSA fused_qkv_a | 7168 → S=8 |
  | DSA q_b | 1536 → S=1 |
  | DSA g_proj | 7168 → S=8 |
  | DSA o_proj | 12288 → S=8 |
  | indexer wq | 1536 → S=1 |
  | indexer wk / weights_proj | 7168 → S=8 |
  | shared gate_up | 7168 → S=8 |
  | shared down | 6144 → S=8 |
  | 第 1 层 dense down | 33792 → S=16 |
  | latent down | 7168 → S=8 |
  | latent up | 3584 → S=1 |
  | lm_head | 7168 → S=8 |

- **反向**：kernel 不提供反向。用 `_contract.fp32_linear_vjp`，只算 dX，dW 跳过。
- **契约**：
  - S 只取决于 (K, N)；
  - 每个 chunk 从零开始、k 升序做 fp32 累加；chunk 之间从左到右 fp32 相加；最后舍入一次到 bf16。
  - 训练 TP=1，与部署相同，所以没有行并行归约。将来如果要开 TP>1，用 `chunks` / `chunk_partial` / `ordered_sum` 复现，不用 NCCL all-reduce；chunk 边界在 TP∈{1,2,4,8} 时都落在分片边界上。

### 3.2 LoRA 叶子（全部 dense 目标）

LoRA 目标：q/k/v/g/f_a/b/f_b/o_proj、q_a、kv_a_with_mqa、q_b、gate/up/down_proj。

- **前向**：
  - delta：bf16 零张量 + punica `add_lora_linear(..., scale=1.0, add_inputs=False)`，底层是 `vllm.lora.ops.triton_ops.lora_shrink` / `lora_expand`，使用 BI=1 时钉住的配置；
  - 输出：`k31_bi_gemm.lora_f32add(x, w, delta)`。
- **反向**：
  - dx = dy·W（fp32 VJP）+ LoRA 支路；
  - dA、dB 用 torch fp32 计算。
- **契约**：
  - y = bf16(fp32(x·Wᵀ 按 S 计划) + fp32(delta))，只舍入一次；
  - delta 本身是 punica 输出的 bf16。
  - 训练前向用的 B 必须是 vLLM 实际持有的那份 bf16 张量。vLLM 加载 adapter 时会把 scaling 折进 B 并舍入。
  - `kv_b_proj` 被当作吸收权重读取，不能挂 LoRA。

### 3.3 RMSNorm

用在：q_a 1536、kv_a 512、latent 3584、final 7168。

- **前向**：`vllm.model_executor.determinism.batch_invariant.rms_norm_batch_invariant`。
- **反向**：`norm.py` 的解析 VJP，原样。
- **契约**：逐行固定归约；C2 单算子测试已验证 7168/1536/512 逐位相同。

### 3.4 attn_res（替换 mHC）

- **前向**：`vllm.models.kimi_k3.nvidia.ops.attn_res.attn_res(...)`，最终调用 `torch.ops._C.kimi_k3_attn_res`。adapter 要断言走的是 native 分支（hidden=7168、nb≤8、sm10x、op 已注册），Triton 回退分支不能被走到。
- **反向**：没有反向 kernel。用 `visible_functional_vjp`，函数图移植 Megatron-Bridge `kimi_k3_ops.attn_res_aggregate`（fp32），在反向时重算。
- **契约**：
  - 逐行计算，nb 取 0–8，单算子已验证批不变。
  - 训练调用必须与 vLLM 完全一致：
    - bank 布局 `[T, 8, H]`；
    - `prefix` / `delta` 的传法：块写层 `prefix_sum = attn_out, delta=None`；
    - `block_write_idx`；
    - `num_blocks = prev_valid_blocks (+1)`。
  - head 聚合用 `output_attn_res_*` 权重，`output_norm_weight=None`。
  - **注意**：kernel 在块写层会原地写 `blocks`。训练侧要把 bank 当成函数式值处理：每个写块层产生新的 bank，或者把写入单独拆成一个 owner。否则 autograd 和重算会读到被改过的数据。

### 3.5 KDA（69 层）

**前向**，按顺序：

1. in_proj（3.1 / 3.2）。
2. 卷积：q/k/v 分三段调 `vllm.model_executor.layers.mamba.ops.causal_conv1d.causal_conv1d_fn`，silu，初始状态为零。rollout 的 decode 也走这个 kernel（`K31_KDA_DECODE_CONV_FN=1`）。
3. 核心：`vllm.models.kimi_k3.nvidia.k31_kda_cfbi.mixed(q, k, v, g1, beta, A_log, dt_bias, idx, starts, lens, ctx=[0]*R, Epool, Cpool, Ppool, lower=-5.0)`。
   - ctx 全为 0，所以全部 token 都走 `pf.prefill`（`kda_cfbi6_pf4.cu`）。
   - 状态池按序列临时分配：每条序列每层约 22 MB（E 6.3 MB + 行缓存 15.7 MB），用完即释放。
4. 输出 norm：`vllm.third_party.flash_linear_attention.ops.kda.FusedRMSNormGated`（sigmoid）。
5. o_proj。

**反向**：CF-BI 没有反向。按 kda-cfbi-design.md 的既定方案：

- 前向值取 CF-BI 的结果，梯度取 FLA `chunk_kda` 的 VJP。
- 实现上，在反向里用训练 venv 的 `fla.ops.kda.chunk_kda` 重算 FLA 前向，再对 grad_output 求导。门控参数（A_log、dt_bias、lower）用同一套。
- 卷积反向用 `fla.modules.convolution.causal_conv1d` 的 autograd，或者一个等价的 torch conv1d 函数图。
- 门控 norm 用 torch 函数图。
- 不写新 Triton：FLA 的反向 kernel 是现成的，只复用；autotune 必须钉住，Megatron-Bridge 已有 `_patch_fla_kda_hopper_autotune` 可以参考。

**契约**：

- CF-BI v5：
  - 门控在 log2 域，指数用 `ex2.approx`；
  - 乘积用 3×TF32 截断拆分；
  - chunk 长 64，从序列开头对齐；
  - 块前代分 16 行一块。
- 所有引擎都与 oracle `rows6` 逐位相同。因此「训练整段 prefill」与「rollout 先 prefill 提示、再逐 token decode 回答」逐位相同（G1 门已覆盖）。
- 训练前向**与 FLA 的值不同**，约 1e-4。所以反向是对相邻函数求导，相当于 STE，见风险 1。

### 3.6 DSA indexer（24 层，冻结）

- **前向**：
  1. `wq_index`、`wk_index` 走 3.1；
  2. `k_index_norm`（LayerNorm）；
  3. `weights_proj`；
  4. `vllm.model_executor.layers.quantization.utils.fp8_utils.per_token_group_quant_fp8`（ue8m0，scale 折进 weights）；
  5. `torch.ops.vllm.sparse_attn_indexer`，经 ds4-indexer-topk overlay，top-k 索引升序。
- **反向**：不需要。indexer 冻结、不挂 LoRA，输出是离散索引，套 `inference_only`。
- **契约**：
  - FP8 q 和 FP8 k cache；
  - 打分 Σ_h w·relu(q·k)；
  - top-2048 升序，并列时位置小的优先；
  - t < 2048 的行保留全部因果 key。
- **注意**：config 里写的是 `indexer_use_fp8: False`，但 vLLM 代码无条件做 FP8 量化。训练侧以代码为准。

### 3.7 DSA MLA 核心

- **前向**：
  1. kv_a 和 q_a 的 norm（3.3）；
  2. fp8_ds_mla 写入：`vllm._custom_ops.concat_and_cache_mla`，`kv_cache_dtype="fp8_ds_mla"`；
  3. W_UK 吸收 bmm；
  4. FlashMLA sparse prefill：`vllm.v1.attention.ops.flashmla.flash_mla_sparse_fwd`，配合 sparse-mla-tp1 和 `sparse_mla_force_mqa`；
  5. W_UV bmm；
  6. 乘 `sigmoid(g_proj(h))`；
  7. o_proj。
- **反向**：
  - 稀疏注意力：`megatron.lite.primitive.kernels.dsa_kernels._DSA.sparse_attention_backward_wrapper`。现有 kernel 已支持 `d_v=512`，GLM5 和 DS4 都在用。
  - K31 没有 attn sink，要传 0 或 None 的 sink。
  - fp8_ds_mla 的量化和反量化用直通梯度，与 `k31_mb._KVQDQ` 一致。
  - bmm 和门控走普通 autograd。
- **契约**：
  - 照 DS4 `attention_core`：保存可见调用内部最终确定的 indices、topk_length、workspace 和 lse（不是 max_logits）；
  - 吸收 bmm 逐位相同，cache 写入的 prefill kernel 与 decode kernel 字节相同（C2 单算子已验证）；
  - FlashMLA 固定 split 规划（overlay 7–9）。

### 3.8 Router

- **前向**：
  - logits：`k31_bi_gemm.bi_router`，fp32 输出，由 gate_linear overlay 接入；
  - 选择：`vllm.model_executor.layers.fused_moe.router.grouped_topk_router.fused_grouped_topk`。参数：sigmoid、`e_score_correction_bias`、n_group=1、renormalize、scale=1.0。实现时要在 rollout 进程里确认 BI 下最终落到的就是这个函数。
- **反向**：`fixed_route_vjp` 的 sigmoid 版本。router 权重冻结，只回传 dX。R3 沿用 DS4 的 `RouterReplay`。
- **契约**：logits 按 S 计划 fp32 求和、不舍入；top-k 结果逐位相同（C2 已覆盖 `fused_topk_bias` / `grouped_topk`）。

### 3.9 MoE 专家（MXFP4 W4A16，带专家 LoRA）

部署走的是 **FlashInfer CuTe-DSL W4A16**（overlay 11/15/18，`K31_MOE_FI_W4A16=1`），**不是 Marlin**；部署文档明确写了不要加 `--moe-backend marlin`。

**前向**：照搬 `mxfp4.py` 里 `_k31_apply_lora` 的序列：

1. `flashinfer.fused_moe.cute_dsl.blackwell.moe_w4a16` 的 `moe_sort` 和 `moe_permute`；
2. `_run_grouped_gemm`（GEMM1），tactic 用 `_k31_pick_tactic`；
3. w13 LoRA：punica `add_lora_fused_moe`，即 `vllm.lora.ops.triton_ops.fused_moe_lora_op`，配置按 fmlora-policy 钉住；
4. `vllm.model_executor.layers.quantization.k31_moe_lora_fuse.gather_add`；
5. `torch.ops._C.situ_and_mul`（fp32）；
6. `act_cast`；
7. `_run_grouped_gemm`（GEMM2）；
8. w2 LoRA（`mul_routed_weight=True`）；
9. 得到叶子 = fp32(w·base) + fp32(d2)。

这些 FlashInfer 函数是私有 API，adapter 要固定 import 路径。

**反向**：FI 没有 W4A16 的反向。照 DS4 `VLLMGroupedMoEWithBF16Backward` 的写法：

- 按专家分块，用 `dequantize_mxfp4` 反量化成 BF16，用 torch 重算 GEMM1 → SiTU（函数图）→ GEMM2；
- 只求 dX 和 LoRA 的 dA、dB；
- 先实现、后优化，不写新 kernel。

**共享外侧专家 LoRA（`--enable-moe-shared-loras`）**：

- w13 的 lora_A、w2 的 lora_B 由全部专家共用（vLLM 里是 `expand(-1, local_num_experts, ...)`）。
- 这两个参数在 EP 各 rank 上是**复制的**，梯度必须在全体 rank 上归约。`EXPERT_CLASSIFIER` 要把它们判为非专家参数，否则会只在 `ep_dp_group` 里平均，造成错误。
- 每个专家各自的 w13 lora_B 和 w2 lora_A 才是专家参数。

**契约**：

- 各 tactic 逐行结果相同（K tile 256，每个 CTA 处理完整 K）；
- 专家 LoRA 的流水是逐行的、顺序固定；
- C2 已验证 M=1..8192、非零 adapter 下批不变。

### 3.10 Combine（canonical-slot tree）

- **前向**：
  - 训练侧不能用 DeepEP combine，它按自己的顺序做 bf16 求和。
  - 做法：叶子 `[T·16, 3584]` fp32 用 all-to-all 送回 token 所在的 rank；在本地按「该 token 的 16 个专家 id 升序」排成 canonical 顺序；再调 `vllm.distributed.device_communicators.k31_tree_allreduce._ext().local_subtree`（`.cu` 注释写明这个接口就是给 trainer 用的）做 16 叶平衡二叉树；-0 规范成 +0；最后 bf16 舍入一次。
- **反向**：求和的梯度就是把 grad 广播到 16 个叶子，再 all-to-all 回专家所在的 rank。用普通 autograd Function。
- **契约**：out[t] = bf16(tree16(按 expert id 升序的叶子))，与 EP 大小无关。无 LoRA 时叶子形式为 fp32(w)·fp32(y)，是 d2=0 的特例，和 LoRA 形式是同一个契约。

### 3.11 latent 尾部与 shared experts

- **前向**：
  - latent 尾部：canonical 结果（bf16）→ RMSNorm(3584)（3.3）→ `torch.mm(latent, up.T)`，BI 下被 aten 覆盖；
  - shared 支路：gate_up → `torch.ops._C.situ_and_mul`（`SituAndMul.forward_cuda`）→ down；
  - 两支相加：`result.add_(shared)`，bf16 相加；
  - 依据：moe-tree-ep2 的 `_k31_bi_tail`，TP=1 时 tree AR 是恒等。
- **反向**：
  - norm：解析 VJP；
  - mm：fp32 VJP；
  - SiTU：用 `visible_functional_vjp`，函数图移植 Megatron-Bridge `situ_and_mul`。SiTU 的 CUDA kernel 与 native 实现差 1 ulp，所以前向必须调 `_C` op。
- **契约**：
  - 训练必须调和 vLLM 一样的入口：F.linear 走 `linear_batch_invariant`；`torch.mm` 走 BI 的 aten mm 覆盖；
  - 第 1 层是 dense MLP（intermediate 33792，down 的 S=16），用同一套 SiTU。

### 3.12 embedding / lm_head / logprob

- embedding：普通查表。
- lm_head：`bi_linear`，K=7168，N=163840。
- logprob：`vllm.v1.worker.gpu.sample.logprob.compute_token_logprobs`，路径已在镜像中确认。沿用 DS4 `_vllm_selected_log_probs`：前向值取 vLLM 的结果，梯度取 torch `log_softmax`。

## 4. 从 DS4 原样移植的部分 vs K31 新写的部分（含工作量）

工作量单位是 agent 工作日，包括单算子逐位门和反向门。

### 4.1 原样或近似原样移植（约 4 天）

| 项 | 工作量 |
|---|---|
| `_contract.py`、`_recompute.py`、`inference_only`、`recompute.py`、`norm.rms_norm` | 0.5 |
| protocol 骨架（SelectorConfig、ImplConfig、契约检查、`_finalize_replica_grads`、`forward_step` / `ensure_runtime_assets`、目标平移、R3 打包） | 1 |
| runtime_metadata 的 BI 初始化、forward context、workspace manager | 0.5 |
| checkpoint 的 WeightSpec 和 PP 导出骨架 | 0.5 |
| router 的 fixed-route VJP（改打分函数）、RouterReplay、TokenDispatcher（normal DeepEP） | 1 |
| `_vllm_selected_log_probs`、PP chunk layout、测试框架 | 0.5 |

### 4.2 K31 新写的部分（约 33–39 天）

| 项 | 内容 | 工作量 |
|---|---|---|
| dense linear + LoRA 叶子 | `bi_linear` / `bi_router` / `lora_f32add` + punica delta 的 adapter 和 VJP；B 的 scaling 与 bf16 舍入对齐 | 2 |
| attn_res | 可见调用 + 函数式 VJP；bank 原地写的函数式化；head 聚合；PP 传 `prefix_sum + bank` | 3 |
| **KDA** | conv + CF-BI（ctx=0 的 metadata 和状态池）+ gated norm 的可见前向；FLA `chunk_kda` 的直通反向（钉 autotune）；卷积反向；对 rollout dump 的 G1 门；梯度与 FP32 参考的偏差测量 | 6–8 |
| **DSA** | V3.2 型 indexer 的 metadata 和 op adapter；fp8_ds_mla 写入；FlashMLA sparse prefill；吸收 bmm；g_proj 门；`dsa_kernels` 反向（d_v=512，无 sink）；STE | 5 |
| **latent MoE** | down、norm、up、shared 相加的 tail 和各自 VJP；第 1 层 dense MLP | 1.5 |
| **MXFP4 W4A16 专家**（FI CuTe-DSL） | 把 `_k31_apply_lora` 序列移植成 adapter；`dequantize_mxfp4` 与 FI 解释一致性的核对；按专家分块的 BF16 反向（只求 dX） | 5 |
| **专家 LoRA**，含共享外侧 A/B | 前向与上一项一起；dA、dB；共享参数的分类和全 rank 归约 | 2 |
| **canonical combine** | fp32 叶子 all-to-all + `local_subtree` + -0 规范化，前向和反向；EP2 / EP16 与单卡逐位相同 | 3 |
| SiTU | `_C` op 的可见调用 + 函数式 VJP | 0.5 |
| checkpoint | MXFP4 打包原样加载、融合布局的名字映射、LoRA adapter 的加载和导出（与 vLLM 加载逐字节相同） | 3 |
| 端到端 | 逐层前缀审计（selector 对 vLLM hook）；slice15 全模型 logprob 0 差；反向门；PP | 4 |

**合计约 37–43 天**。可以并行的三条线：KDA、DSA、MoE（专家 + LoRA + combine）。

## 5. 风险与不确定项

1. **KDA 梯度对应的是相邻函数**。前向值来自 CF-BI，梯度来自 FLA `chunk_kda`，两者前向差约 1e-4（kda-cfbi-design.md 实测 H6 1.22e-4）。按「训推验收先正确性后精度」：先测梯度与 FP32 递推参考的余弦和相对误差，再决定能否接受；不能接受时，再考虑给 CF-BI 写 CuTe DSL 反向。反向里重算 FLA 前向，KDA 反向的耗时约为 FLA 的前向加反向。
2. **attn_res 的 bank 被原地写**。可见 kernel 在块写层改写 `blocks`。如果直接把它包进 autograd 或重算，反向会读到后面层改过的 bank。必须做成函数式值或者拆出 owner，这一点要在设计评审时定下来。
3. **bi_gemm 的并发限制**。同一 (K, N) 的 GEMM 不能在两条 stream 上并发（计数器切片的限制）。DS4 的 model 用了 3 条投影 stream，K31 的投影必须串行，或者保证并行的 GEMM 的 N 互不相同。
4. **overlay 一致性**。训练前向依赖 20 个 overlay 的链式产物（`mxfp4.py` 最终版来自 18 号，`batch_invariant.py` 来自 20 号）。训练进程挂的版本必须与 rollout 相同。verl 共置时两者在同一个镜像；如果训练在另一个容器，`initialize_k31_vllm_batch_invariance` 要检查每个 overlay 的 VERSION 和源码哈希。
5. **MoE 实际走 FI，不是 Marlin**。任务描述写的是 Marlin，但部署是 FI CuTe-DSL W4A16，训练必须跟部署走。FI 的 `_run_grouped_gemm`、`moe_sort`、`_get_workspace` 是私有 API，换 FlashInfer 版本就可能失效。
6. **共享外侧 LoRA 的梯度归约组**。如果分类错误，这些参数只在 `ep_dp_group` 里平均，静默产生错误梯度。需要专门的单测。
7. **combine 的通信量**。fp32 叶子是 16×3584×4 B ≈ 229 KB/token/层；每 rank 8k token 时每层约 1.9 GB 的 all-to-all。远大于 DeepEP 的 bf16 combine，可能成为吞吐瓶颈。推理侧用的是对称内存，训练侧暂时没有对应实现。
8. **LoRA 的 scaling 与 bf16 舍入**。vLLM 加载 adapter 时把 alpha/r 乘进 B 再存成 bf16。训练前向必须用同一份 bf16(B·s)，否则 delta 不逐位相同。导出和同步格式要以 vLLM 加载后的张量为准做逐字节测试。
9. **DSA 反向 kernel 的覆盖范围**。`dsa_kernels` 的反向是在 DS4（584 B/token、有 sink）和 GLM5 上验证的。K31 是 656 B 的 fp8_ds_mla 布局、无 sink、MLA 吸收形式，在 sm_100 和 sm_103 上都要重测 finite 与精度。
10. **PP 的 P2P 体积**。attn_res 需要传 `prefix_sum + bank[8]`，约 9×hidden/token，比普通 hidden 大 9 倍。
11. **mm 的 BI 路径**。latent up 用 `torch.mm`，不是 F.linear。BI 下 aten mm 覆盖最终走到哪个 kernel，要在 rollout 进程里用 dump 确认，训练侧要调同一入口。
12. **config 与代码不一致的两处**：
    - `indexer_use_fp8=False`，但代码无条件做 FP8 量化；
    - config 写了 `qk_rope_head_dim=64`，但模型是 NoPE（`mla_use_nope`）。
    两处都以 vLLM 代码为准。
13. **CP 不支持**。KDA 的 CF-BI 是串行状态递推，契约直接拒绝 CP>1。若要支持更长的序列，需要跨 rank 交接状态，这是独立的一项工作。
14. **verl#7688 与 ISEEKYAN V4 lite 未核对**。本文以 GB200 集群 仓库里的提交为准，PR 页面上后续的修改没有看到。

## 6. 下一步（按依赖顺序）

1. 搭骨架：registry、config、protocol、checkpoint（MXFP4 + LoRA）、runtime_metadata 的 BI 初始化。
2. 三条线并行：
   - A：linear、LoRA、norm、SiTU、attn_res；
   - B：KDA；
   - C：DSA。
3. MoE：router → FI 专家 + 专家 LoRA → canonical combine → latent tail。
4. 端到端：先在 slice4 上做逐层前缀审计，再用 slice15 做 logprob 0 差和反向门，最后上全模型 PP。
