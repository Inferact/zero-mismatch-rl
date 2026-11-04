# K31 BI=1 与 BI=0 性能差距：声明的限定条件（草稿，2026-11-04）

结论只在下列条件同时成立时成立。数字待成对长跑（GB300 集群 27240 BI=1 / 27241 BI=0+R3）和 bench（27216 / 27214、27242 / 27243）出齐后填入。

## 1. Workload

- **RL 长跑**（verl recipe `run_k31_true_on_policy_preview_megatron.sh --mode aligned`）：
  - 训练：GRPO，train_batch 32 × n=8，prompt ≤ 2048，response ≤ 6144，dapo17k；
  - LoRA rank 32，attention 全投影 + 896 experts（专家共享 A/B 一侧）；
  - 初始 adapter：PEFT 默认初始化（A 随机，B=0，seed 0），GB300 集群 上按 NV adapter-spec 的形状重建；
  - 布局：4 节点 GB300 × 4 GPU；rollout TP8·EP8，2 个副本；训练 PP4 × EP4，参数和优化器都 offload；
  - 共置：level-1 sleep，备份写 ceph /home；
  - PPO max token/GPU 16384（训练侧 FlashMLA int32 限制，修复见第 5 节）；
  - 只跑 2 步，不保存 checkpoint。
- **serving bench**（辅助数据）：
  - TP8·EP8，2 节点，util 0.90；
  - random 1024 in / 3072 out，ignore_eos，1024 条请求，并发 1024；
  - max_num_seqs 256，MBT 8192，开 prefix cache 和 chunked prefill；
  - 带 adapter 时用 experts-only adapter（来自 adapter15t，48 GB）。

## 2. Kernel 配置

| 项 | BI=0 | BI=1 |
|---|---|---|
| 来源 | vLLM 默认，recipe 不传任何引擎参数，只传 LoRA 必需的两项 | recipe BI_ARGS |
| attention | FLASHINFER_MLA_SPARSE，per-tensor fp8_e4m3 KV | FlashMLA sparse，force_mqa，fp8_ds_mla KV，固定 plan |
| MoE | 默认选择 → Marlin W4A16 + punica LoRA | K31 FI W4A16（CuTe-DSL）+ CUDA expert LoRA + FP32 SiTU + 规范化 combine |
| cudagraph | 默认 | FULL_AND_PIECEWISE |
| TP 归约 | NCCL / 默认自定义 all-reduce | 固定树 + tp-ordered |
| shared-experts stream | 开（默认） | 关 |
| FlashInfer autotune | 开（默认） | 不适用 |
| R3 | 开（`bi0r3`）；另有标准 verl 臂 `bi0std`：不开 R3，重算 old_log_prob，用 TIS | 关；bypass（rollout log-prob 直接当 old policy） |

TRTLLM MXFP4 MoE 不能作为 BI=0 的 MoE：FlashInfer 没有 BF16 激活的 SiTU kernel，micro bench 已验证。

## 3. BI=1 代码对 BI=0 的影响（R2 审计，`scratchpad/audit/r2/R2-fable.md`）

- **没有 P0。** 在 d0e1ea0 → 5e9034de6 的 50 个非测试文件里，BI=0 下以下各项都与上游一致：kernel 选择、KV 块大小和容量、cudagraph、环境变量默认值、MoE 后端、all-reduce、sampler。
- **两臂共有、必须声明的偏离（P1，不涉及数值）：**
  - sleep 备份改成 `cudaMallocHost`，修复 pinned 备份只涨不还；
  - 每次加载或卸载 adapter 时做 gc + synchronize + malloc_trim。
- **LoRA 强制的改动（P1，正确性所需，BI=0 也会走到）：**
  - 关闭会绕过 adapter 的融合快速路径：KDA overlap、MLA gate、GEMM-AR、DSA 打包投影；
  - 复制分片的 `output_ids=0`。
- **纯上游 d0e1ea0 不支持 K31 LoRA**（没有 SupportsLoRA），所以带 LoRA 的「纯上游」对照不存在。BI 代码是否拖慢 BI=0，用不开 LoRA 的 bench 对照来证明。
- **不开 LoRA 的对照（BI=0，util 0.90，四次都没有抢占，KV 87.3 万 / 87.6 万 token）：**

  | 节点 | 纯 d0e1ea0（tok/s） | 带 overlay（tok/s） |
  |---|---|---|
  | node17/18 | 3014.8（27242）、3019.3（27309） | 2952.1（27243）、3042.4（27338） |
  | node09/13 | — | 3081.2（27308） |

  - **结论：没有测出 overlay 拖慢 BI=0。** 在 node17/18 上：纯上游两次均值 3017，带 overlay 两次均值 2997，差 −0.7%。带 overlay 两次之间相差 3%，这个差距落在其内。
  - 只比同一对节点上的结果：不同节点对之间差约 2%。
  - 27339（node05/06）启动时 node06 的 slurmd 超时，作业失败，与本次对照无关。它的配对作业 27340 已取消。

## 3b. NV 上已有的一对 RL 长跑（第 1 步）

两次运行用同一组节点（c07/08/10/11）、同一个镜像 f0b3c26，先后连续跑。BI=1 第 2 步引擎挂掉，属于共置显存问题，按用户决定不处理。

| 指标 | BI=1（22408） | BI=0+R3（22409） |
|---|---|---|
| gen（s） | 400.8 | 360.9 |
| 生成吞吐（256 条 × 平均长度 / gen） | 2640 tok/s | 2897 tok/s |
| 最慢一条（6144 token）（s） | 397.1 | 355.0 |
| update_actor（s） | 884.1 | 881.2 |
| 整步（s） | 1336.2 | 1292.8 |
| 平均回答长度 / 截断比例 | 4133 / 0.45 | 4085 / 0.45 |

- BI=1 的生成吞吐是 BI=0 的 91%，整步只慢 3.4%，因为训练占了整步时间的约 2/3。
- **限定：** 22409 用的树早于 9570a49 / 25aa8e7（「batch-variant 进程保留上游 MoE experts 和 packed block size」），所以它的 BI=0 不是纯上游默认，MoE 路径和块大小与 R2 审计的 5e9034de6 不同。这个 91% 只能当作「BI=0 带部分 BI 路径」的结果，不能当作最终数字。
- 每臂只跑了一次，单轮不算数。

## 4. 其它限定

- **同硬件、同节点池**：GB300 集群 rack4 GB300，同一个镜像加同一个 overlay；两臂只差 recipe 的 `--bi` 和 R3/bypass 开关。
- **RL 长跑不在 GB300 集群 跑**：
  - util 0.90 时，训练侧 offload 之后每卡仍被占约 95 GiB，vLLM 拒绝启动。
  - util 0.58 时，权重（约 187 GiB/卡）超出预算，溢出到 UVM。27301（node03）和 27303（node08）都在加载权重时触发 uvm-lockup，导致节点下线。
  - 成对长跑等 NV 节点，NV 上训练侧残留约 20 GiB。
- **博客成本数字已过时**：KV 用量是默认 BI=0 的 74.7%，不是「约 1/10」。PR #7 合入前要按最终数据更新。
- **只用 2 步测时间**：BI=1 第 2 步在 util 0.90 下有共置 OOM 的风险（分配器碎片约 21 GiB/GPU）。这一项按用户决定不处理，如出现会如实记录。
- **步时间按阶段拆分**：生成时间可以直接比较。sleep/wake 走 ceph 盘，两臂开销相同，不计入差距。

## 5. 训练侧 FlashMLA 分块（mlite-kv，未提交）

- 问题：`DSASparseMLAAdapter.forward` 把整步作为一个 `(1, T, H, 576)` 批次调用 FlashMLA，batch stride 超过 int32。
- 改法：超过 `DSA_SPARSE_MLA_ROWS = 16384` 行时，按行分块调用再拼接。
- 测试：CPU 合约测试，加一个 GPU 测试验证分块与整块逐位相同。
- 验证：GB300 集群 作业 27244。
