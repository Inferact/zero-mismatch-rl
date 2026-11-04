# K31 TP/EP 不变归约：清单、契约、原型、实施计划

日期 2026-10-15。范围：K31 推理 BI=1 臂（TP16/EP16，4×4 GB300，MNNVL）与训练（Megatron/mLite，TP4、EP 任意）在所有跨 rank 求和上逐位相同。只用代号 K31。

## 0. 结论

1. BI=1 现在的 TP 归约走 NCCL Tree、1 channel、Simple、NVLS 关。TP16 下每个 decode step 有 279 次 all-reduce，批 64 时单次 590–674 us，每步合计约 180 ms。求和顺序由 NCCL 拓扑决定，换 TP 就变，训练无法复现。
2. 契约（协调方已接受）：
   - dense 行并行层：G=16 个 K-叶子，叶子是 bf16，fp32 平衡二叉树求和，根只舍入一次，-0 改 +0。
   - MoE combine：每个 token 的 16 个槽按 expert id 升序排；每个槽的叶子是 `fp32(fp32(w)·fp32(base)) + fp32(d2)`，d2 是专家 LoRA 增量（不开 LoRA 时 d2=0），走同一棵 16 叶树。不开 LoRA 和开 LoRA 是同一份契约，任何路径上都没有别的 top-k 求和。结果与专家放在哪个 rank 无关。
   - 尾部：`routed = bf16(mm(norm(latent), up_proj))`，再 `routed.add_(shared_root)`，在 bf16 上单独相加，共两次舍入（不是 addmm）。所有随 M 切换的 tail 融合在 BI 下关闭。
3. 原型是 CUDA 写的 Lamport push 固定树（无 Triton、无 barrier、CUDA graph 安全），已做成 overlay 生成器，并接上了 MoE agent 的 FI W4A16 专家（含专家 LoRA 路径）：
   - 正确性：dense 57/57 组、合成 MoE 18/18 组逐位相同；真实 FI W4A16 专家输出（896 专家）下，EP4 与 EP16 的 combine 都与单卡参考逐位相同：不开 LoRA 和开 LoRA（非零 adapter）各 8 个 T（1..4096），eager、重跑、graph 都通过；MoE agent 的 selftest 现在按树的结果比较，也通过。走 vLLM 自己 `tensor_model_parallel_all_reduce` 的冒烟测试（TP16）也是 0 差异。
   - 速度（profile 后的版本，W=16）：prefill 8192×7168 每次 393 us，BI=0 NCCL 是 422 us（0.93 倍，目标 ≤1.1 倍）；decode 64×7168 每次 9.8 us，FlashInfer 是 7.9 us（1.24 倍，目标 ≤1.5 倍）；T=1 和 T=8 分别是 1.46 倍。
   - 整步：decode 批 64 每步归约合计约 3.0 ms（BI=0 约 2.1 ms，BI=1 现状约 180 ms）；prefill 8192 一次 forward 约 95 ms（BI=0 NCCL 约 102 ms）。
4. FlashInfer MNNVL 的顺序确定但不能复用：one-shot 是两段 8-rank 顺序和，two-shot 是 16-rank 顺序和，按大小切换，且不能拆成子树（§3.4）。
5. BI=1 下有 4 条不受 `VLLM_BATCH_INVARIANT` 控制、按 token 数切换数值路径的地方（§1.3），overlay 已全部关掉。
6. 训练侧规格见 §6，双方共用一个纯 torch 参考文件 `k31_tree_ref.py`。

## 1. 归约清单

### 1.1 TP16/EP16、DP=1、无 SP 时每个 forward 的跨 rank 求和

H=7168，latent=3584，o_proj 输入 96 头×128=12288，共享专家中间维 3072×2=6144，dense 首层中间维 33792。来源：镜像源码 `models/kimi_k3/nvidia/{model,kda,dsa,mla,latent_moe_runner}.py`、`model_executor/layers/linear.py`、`distributed/device_communicators/cuda_communicator.py`。

| 归约 | 层数 | 每 rank K | 输出形状 | decode 批 64 字节（bf16） | prefill 8192 字节 | 原来的求和内容 |
|---|---|---|---|---|---|---|
| KDA o_proj（RowParallel） | 69 | 768（6 头） | [T,7168] | 0.875 MiB | 112 MiB | 16 个 bf16 部分和 |
| DSA/MLA o_proj（RowParallel） | 24 | 768 | [T,7168] | 0.875 MiB | 112 MiB | 同上 |
| MoE routed latent（EP combine 在 TP all-reduce 里完成） | 92 | 每 rank 56 个专家 | [T,3584] | 0.4375 MiB | 56 MiB | 每 rank 先在 finalize 里按 kernel 顺序加权求和本地专家，再跨 16 rank 求和 |
| 共享专家 down_proj（RowParallel，2 个共享专家合并） | 92 | 384 | [T,7168] | 0.875 MiB | 112 MiB | 16 个 bf16 部分和 |
| dense 首层 down_proj | 1 | 2112 | [T,7168] | 0.875 MiB | 112 MiB | 同上 |
| VocabParallelEmbedding | 1 | — | [T,7168] | 0.875 MiB | 112 MiB | 每个元素只有一个 rank 非零，任何顺序都精确，唯一差别是 -0 |
| lm_head | 1 | — | gather [T,10240]/rank | — | — | all_gather，不求和；logprob 在整词表上算 |
| DSA indexer | — | — | — | — | — | wq/wk/wv_index 是 ReplicatedLinear，没有跨 rank 求和 |

合计：每个 decode step 279 次 all-reduce（93 + 92×2 + 1 + 1），外加 1 次 gather。

### 1.2 BI=1 现状

`override_envs_for_invariance` 的设置：symm-mem 关；custom AR 用 1stage，但 TP16 跨节点用不上；FlashInfer AR 关；NCCL 设为 `ALGO allreduce:tree, NCHANNELS 1, PROTO Simple, NVLS 0, NTHREADS 1`。实际走 pynccl Tree。问题：求和顺序随 TP 拓扑变；bf16 每跳都舍入；慢（decode 批 64 单次 674 us）。

### 1.3 BI=1 下没被门控、会按 token 数换数值路径的地方（镜像源码）

| 位置 | 现状 | 触发条件 | overlay 处理 |
|---|---|---|---|
| `latent_moe_runner.py` `_select_tail_tier` | tier0：CuTeDSL Lamport multicast 融合 latent 归约、RMSNorm、共享 RS、up-proj；tier1：AR 加完整 up-proj；tier2：列并行 up-proj，共享部分和用 beta-add 加 | tier0 在 T≤128；tier1 在 T≤256 且共享流开启；其余 tier2 | BI 下关 tier0（连带关 deferred finalize），尾部固定一种形式 |
| `latent_moe_runner.allreduce_norm_latent_out` | FlashInfer `trtllm_fused_allreduce_norm` | T≤max_token_num | BI 尾部不再调用 |
| `kernels/linear/cute_dsl/gemm_rs_ar.py`（`VLLM_KIMI_K3_GEMM_AR` 默认 1） | GEMM 与 all-reduce 融合，`multimem.ld_reduce` 在交换机里归约 | M≥128 | BI 下 `maybe_init_gemm_rs_ar` 返回 False |
| `cuda_communicator.reduce_scatter` | BI 下 reduce 到 root 再 scatter | SP 路径 | K31 TP16、DP1 不开 SP，不触发 |

## 2. 契约

### 2.1 通用树

- 输入是 16 个叶子 L0..L15（dense 是 bf16 值，MoE 是 fp32 加权叶子）。
- 平衡二叉树，相邻配对：第 1 层 (0,1),(2,3),…，第 2 层把结果再两两相加，共 4 层。全部 fp32 加法、就近舍入，`--fmad=false`。
- 根舍入一次：舍成 bf16，-0 改 +0（Lamport 哨兵需要，同时是契约的一部分）。
- 子树性质：持有连续 16/W 个叶子的 rank 在本地先算 fp32 子树根，跨 rank 再走剩下几层。W=1/2/4/8/16 结果逐位相同（§3.2 已验证）。W=16 时线上传 bf16，W<16 时传 fp32。

### 2.2 dense 行并行层（K-叶子，G=16）

| 层 | K 总长 | 叶子大小 | 推理 TP16 | 训练 TP4 |
|---|---|---|---|---|
| KDA、DSA o_proj | 12288 | 768（6 头） | 1 叶/rank | 4 叶/rank |
| 共享专家 down_proj | 6144 | 384 | 1 叶 | 4 叶 |
| dense 首层 down_proj | 33792 | 2112 | 1 叶 | 4 叶 |

- 叶子 g = `bf16(GEMM(X[:, Kg], W[:, Kg]))`，fp32 累加，不做 split-K。两侧必须用同一个 batch-invariant leaf GEMM。
- 树的结果和整 K 一次 GEMM 不同，所以训练 TP=1 也必须走 16 叶树。
- 推理 TP16 每个 rank 只有 1 个叶子，GEMM 不用改，只换归约。
- 反向不在契约内。

### 2.3 MoE combine（EP 不变，含专家 LoRA）

- 规范槽位：每个 token 选中的 16 个专家按 expert id 升序排，c=0..15。不用 top-k kernel 输出的顺序。
- 叶子（每个 (t,k) 槽一个，fp32）：`leaf = fl32( fl32(w·base) + fp32(d2) )`
  - base：专家 down GEMM 的 bf16 输出，未加权、未 finalize（FI W4A16 的 GEMM2 输出行）。
  - w：交给专家的 fp32 路由权重。K31 走 bias router，`routed_scaling_factor` 已在 router 里乘进权重；latent 融合路径不再缩放输出，所以 rsf 就是 w 的一部分。
  - d2：该槽的专家 LoRA 增量，bf16，由 LoRA kernel 产出且已乘过 w（MoE agent 的 `_k31_apply_lora` 里 punica w2 用 `mul_routed_weight=True`）。不开 LoRA 时 d2=0：`x + 0` 只可能改变零的符号，根又统一把 -0 改 +0，所以不开 LoRA 的叶子就是 `fl32(w·base)`，两条路径是同一份契约。
  - 乘法、加法都是 fp32 就近舍入，各自单独舍入，不融合成 FMA。运算顺序与 `_k31_apply_lora` 相同：`slot = w * base.float(); slot += d2.float()`。
- `latent[t] = bf16( tree16_c( leaf[t, c] ) )`，-0 改 +0。原来 LoRA 路径的「按 k 顺序累加」和非 LoRA 路径的 FI finalize 都被这棵树替换（`K31_MOE_TREE=1` 时）。
- 尾部（明确写入契约）：latent RMSNorm（3584，replicated）→ up_proj（replicated GEMM 3584→7168，bf16 输出，第一次舍入）→ `routed.add_(shared_root)`（bf16 + bf16，fp32 计算后第二次舍入）。shared_root 是共享专家 down_proj 的 16 叶树（§2.2）。不是 runner tier2 的 addmm（那个只舍入一次，结果不同）。
- 推理侧实现：每个 rank 把本地槽的叶子推到对应槽位 owner，owner 读 16 个槽、走树、multicast 广播。不开 LoRA 时线上传 bf16 base，乘 w 在 kernel 里做；开 LoRA 时线上传 fp32 叶子（字节翻倍，单独一套缓冲）。只看 expert id，不看专家在哪个 rank，EPLB 或冗余专家也不影响结果。EP16 时不开 LoRA 每 rank 推出的数据量与一次 latent all-reduce 相同。
- 放弃的方案：跨 EP rank 的固定树（叶子 = 每 rank 局部和）。它依赖专家放置，EPLB 一开就失效。

### 2.4 其它

- Embedding 也走树 AR，训练侧在 embedding 输出上做 -0 改 +0。
- lm_head/logprob 不在本契约内，需要单独的 logprob 契约。
- 两侧必须一致的常数：G=16、dense 叶子 bf16、相邻配对树、根 -0 改 +0、MoE 槽位按 expert id 升序、MoE 叶子 `fl32(fl32(w·base) + fp32(d2))`、shared 与 routed 在 bf16 上单独相加（两次舍入）。

## 3. 原型与实测

### 3.1 实现

文件（本机规范副本在 `agent_run/scripts/`，集群副本在 `$WORK/k31/tpinv/scripts/`，两边 sha256 一致）：
- `tpinv_ar.cu`：CUDA 扩展，包括 `tree_all_reduce`（one-shot / two-shot）、`moe_combine`（one-shot / two-shot）、`local_subtree`、`set_max_ctas`、`prof_read`（`-DK31_PROF` 时有效）。
- `tpinv_ar_test.py`：合成数据的逐位测试与 bench（对照 NCCL、FlashInfer）。
- `tpinv_prof.py`：分相 profile 与 CTA 数扫描。
- `moe_tree_fi_test.py`：真实 FI W4A16 专家输出下的 EP 不变测试。
- `k31_tree_ref.py`：单卡纯 torch 参考，推理测试和训练侧共用（§6）。
- `tpinv.sbatch`：多节点 enroot 启动，`OVL=<dir>` 挂 overlay，`PIN=1` 换成 BI 的 NCCL 环境。

传输设计：
- 缓冲区用 torch symmetric memory，跨 4 节点 MNNVL 可用，`buffer_ptrs` 和 `multicast_ptr` 都有。
- Lamport push：接收方只轮询本地缓冲，-0 作哨兵，不需要 barrier。
- 三缓冲轮转：第 k 次调用用 k%3 号缓冲，清 (k+2)%3 号。epoch 和每块已用字节数放在设备内存里，graph 安全。
- one-shot：所有 rank 把整条向量 multicast 给所有 rank，各自全量求和。two-shot：chunk c 推给 owner c，owner 求和后 multicast，各自 gather。两种模式每个元素算的是同一棵树，切换点只影响速度。
- 求和时 8 个槽一组连续发 volatile load，有哨兵就整组重读，再按槽位顺序进二进制计数器栈（等价于平衡树）。
- 所有 CTA 必须同时驻留（grid ≤ SM 数）。

profile 后改的三处（§3.3 有前后对比）：
1. two-shot scatter 按 rank 轮转目标：rank r 从 owner r+1 开始发。原来 grid-stride 每一步只覆盖约一个 chunk，16 个 rank 同时往同一个 owner 写（incast），scatter 相占 prefill 的 60%。
2. epoch 在 kernel 入口认领：每个 CTA 读 epoch 后 thread 0 计数，最后到的 CTA 把 epoch 加一。去掉了尾部的 `__threadfence` + 原子计数。
3. 清上一轮缓冲挪到 push 之后、轮询之前，与等待重叠；one-shot 按缓冲大小多开 CTA，让清缓冲每线程约 2 次 store。

MoE combine 直接吃 FlashInfer 自己的张量，主机侧没有任何算子：
- 不开 LoRA：GEMM2 输出行 `[S,3584]` bf16、`expanded_idx_to_permuted_idx [T,16]`（-1 表示非本地专家）、`topk_ids`、`topk_scales`，叶子 w·base 在 kernel 里算。
- 开 LoRA：`_k31_apply_lora` 已经算好的 fp32 叶子 `slot [T,16,3584]`（router 槽位顺序），加上同一个 e2p 和 ids；kernel 模板参数换成 fp32 线，用独立的缓冲（哨兵按线上 dtype 区分）。
- 规范位置由 warp ballot 求（本 token 16 个 id 里比它小的个数）；规范顺序的权重用寄存器里的 bitonic 排序得到。
- 一个 warp 处理一个 (t,k) 槽，非本地槽整 warp 跳过。

### 3.2 正确性

| 用例 | 覆盖 | 结果 |
|---|---|---|
| dense 树 AR vs 单卡 16 叶参考（作业 19933/19937/19941/19942 旧 kernel；19983、19987 新 kernel） | W=4（每 rank 4 叶，fp32 线）、W=8（2 叶，fp32）、W=16（1 叶，bf16）；T=1/8/64/256/1024/8192 × H=7168/3584；one-shot 与 two-shot；eager、graph、连续 3 次 | 每个作业 57 组全 0 差异；19995（最终 kernel，T=1/64/8192）30 组全 0 |
| 合成 MoE combine vs torch 参考（19977、19983、19987、19995） | EP=4/8/16；T=1/64/1024/8192；one-shot 与 two-shot；ids 乱序、本地行顺序随机 | 每个作业 18 组全 0 |
| 真实 FI W4A16 专家输出（`moe_tree_fi_test.py`，19996 EP16、19997 EP4，最终版本） | 896 个随机 MXFP4 专家，按 MoE agent 的换算准备；参考 = 单卡持有全部 896 专家，抓 GEMM2 未加权行，`k31_tree_ref.moe_combine_ref`；EP 路径 = 生成的 mxfp4.py 的 arm 一行 + `launch_w4a16_moe`；T=1/3/8/9/64/333/1024/4096；参考用 default tactic，EP 用 MoE agent 的 tactic 选择 | EP4 与 EP16 每个 T 都 0 差异，eager、重跑、graph（T=8/64/4096）都通过。所以 EP4 = EP16 = 单卡 |
| 真实 FI + 专家 LoRA（20008 EP16、20009 EP4） | 同上的 FI base 行；d2 = 每专家 rank-1 增量（`bf16(w·(x·v_e)·u_e)`，约为 base 的 5%，在每个 rank 上对全部 T×16 槽用同一形状算出，与布局无关）；叶子按 `_k31_apply_lora` 的尾部构造后调用 `moe_tree_lora`（与改过的 `_k31_apply_lora` 调用同一个函数）；参考 = `moe_combine_ref(y, ids, w, d2)`；T=1/3/8/9/64/333/1024/4096；graph 在 T=1/8/64/1024/4096 | EP4 与 EP16 全部 0 差异，eager、重跑、graph 都通过；adapter 非零（T=4096 时 2.3 亿个非零增量元素，LoRA 与不开 LoRA 的结果差 750 万个元素） |
| selftest 兼容（同两个作业） | 批量调用后、latent 尾部消费标志前，按 MoE agent 的 selftest 方式单行重算（`selftest_launch` / `selftest_call`），T>1 时取 0、T/2、T-1 三行 | 单行结果等于批量结果、等于树参考；批量调用的标志在重算后仍在；全部通过 |
| vLLM 内冒烟（`tp_tree_smoke.py`，19988，TP16） | `tensor_model_parallel_all_reduce`，T=1/64/256/8192 × H=7168/3584，eager 与 graph | 全 0 差异 |
| 测试敏感度（对照） | 同一批叶子按顺序求和 vs 按树求和 | 合成 dense 每作业 474–1580 个元素不同；真实 FI 输出 T=4096 时 1095 个元素不同，FI 自己的 finalize 与树差 1259 个 |

注意：bf16 叶子在 fp32 里求和，16 个数指数跨度不超过约 12 位时与顺序无关，所以顺序差异只出现在约 1e-5 的元素上。契约测试必须用大张量和宽量级数据。

### 3.3 速度（CUDA graph，每次 us，16 rank 取最大值，W=16 bf16 线，multicast）

同一作业里对照（19987，最终 dense kernel）；“旧 tree”是 profile 前的版本（19941）：

| T × H | tree | 旧 tree | NCCL BI-pinned（BI=1 现状，19937） | NCCL 默认 MNNVL+NVLS（BI=0） | FlashInfer MNNVL auto（BI=0） | tree / 目标基线 |
|---|---|---|---|---|---|---|
| 1 × 7168 | 7.0（one-shot） | 9.1 | 160.1 | 55.2 | 4.8 | 1.46× FI |
| 8 × 7168 | 8.6 | 10.2 | 255.4 | 46.1 | 5.9 | 1.46× FI |
| 64 × 7168（decode o_proj/共享） | 9.8 | 11.5 | 674.3 | 50.8 | 7.9 | 1.24× FI |
| 64 × 3584 | 8.9 | 10.6 | 589.9 | 49.4 | 6.7 | 1.33× FI |
| 256 × 7168 | 16.7 | 17.8 | 836.2 | 58.4 | 19.2 | 0.87× FI |
| 1024 × 7168 | 49.4 | 59.1 | 1541.7 | 108.7 | 64.6 | 0.76× FI |
| 8192 × 7168（prefill） | 393.5 | 525.7 | 8095.1 | 422.1 | 479.7 | 0.93× NCCL |
| 8192 × 3584 | 191.2 | 253.3 | 4339.1 | 255.4 | 250.9 | 0.75× NCCL |

目标（协调方）：prefill ≤1.1× BI=0 NCCL，达到（0.93×）；decode ≤1.5× FlashInfer，达到（1.24–1.46×）。T=1/8 的余量只有 3%。vLLM 冒烟里同一 kernel：T=1 7.5、T=64 10.2、T=8192 392.3。

分相 profile（`-DK31_PROF`，block 0 / thread 0 的 clock64，换算成 us，16 rank 取最大）：

| 用例 | 版本 | 相结束时刻 | 每次 |
|---|---|---|---|
| 8192×7168 two-shot | 旧（19976） | scatter 317 → reduce+multicast 417 → gather 496 → 清缓冲 514 → 尾部计数 515 | 528 |
| 8192×7168 two-shot | 新（19986） | scatter+清缓冲 195 → reduce 221 → gather 365 | 391 |
| 1×7168 one-shot | 旧 | push 1.05 → 轮询+求和 6.22 → 清缓冲 7.67 → 尾部计数 8.81 | 9.8 |
| 1×7168 one-shot | 新 | push+清缓冲 2.03 → 轮询+求和 5.64 | 6.7 |

- 结论：prefill 的瓶颈是 scatter 的 incast，轮转后 scatter 从 317 降到约 190。现在剩下的是 gather（每 rank 入向约 2×数据量），这是不能在交换机里归约的方案的固有下限（NVLS ld_reduce 的求和顺序不归我们控制，不能用）。
- decode 的固定开销：尾部的 fence + 原子计数 1.1 us 和清缓冲 1.4 us 都已经移出关键路径。剩下约 3.6 us 是等对端数据到达（multicast 写的延迟加 rank 间偏差），这部分与 FlashInfer 同类，不再有便宜的改法。
- CTA 数扫描：prefill 用满 SM 最快（96 CTA 时 415 us，128 时 395 us，全部时 392 us）；decode 对 CTA 数不敏感。
- `K31_OCC=2` 和 PDL 都没有收益（之前的作业）。

MoE combine（合成数据，EP16，19995 最终版本）：T=1 是 9.4（one-shot），T=64 是 13.5（two-shot），T=1024 是 36.6，T=8192 是 235.5。它替代的是「FI 本地 unpermute + latent 树 AR」，在模型里按真实 FI 专家计时（19996，graph，整个 MoE 含 GEMM）：

| T（EP16） | FI 专家 + 树 combine | FI 专家 + 本地 finalize + 树 AR |
|---|---|---|
| 1 | 40.4 | 40.1 |
| 8 | 68.2 | 66.9 |
| 64 | 217.3 | 215.9 |
| 1024 | 294.3 | 288.2 |
| 4096 | 435.3 | 431.9 |

EP16 下 EP 不变的代价在 0.3–2% 以内。EP4 在 T≥1024 时多 4–10%，因为每 rank 要推 4 倍的行，这在 EP4 部署时才相关。第一版 hook 在主机侧用 argsort/scatter 等约 10 个小算子算槽位，EP16 时多 27–150 us（19974），改成 kernel 内计算后消失。

每个 decode step（批 64）的归约合计：

| 方案 | 合计 |
|---|---|
| BI=1 现状（NCCL pinned） | 93×674 + 92×(590+674) + 2×674 ≈ 180 ms |
| 契约 tree（最终） | 93×9.8 + 92×(13.5+9.8) + 2×9.8 ≈ 3.08 ms |
| BI=0（FlashInfer） | 93×7.9 + 92×(6.7+7.9) + 2×7.9 ≈ 2.09 ms；BI=0 实际还有 tier0 融合，会更低 |

prefill 8192 一次 forward：tree 约 93×393 + 92×(236+393) + 2×393 ≈ 95 ms；BI=0 NCCL 约 102 ms；BI-pinned NCCL 约 1.9 s。

### 3.4 FlashInfer MNNVL 的求和顺序

作业 19933，W=16，每个 rank 传自己的 bf16 叶子：

| 模式 | 与「rank 0..15 顺序和」差异 | 与「(0..7 顺序和)+(8..15 顺序和)」差异 | 与平衡树差异 | run-to-run |
|---|---|---|---|---|
| one-shot（T=8/64/1024） | 32 / 48 / 1312 | 0 / 0 / 0 | 0 / 32 / 560 | 0 |
| two-shot（T=8/64/1024） | 0 / 0 / 0 | 32 / 48 / 1312 | 32 / 80 / 1600 | 0 |

源码（`trtllm_mnnvl_allreduce.cuh`）：`accum = 0.f` 按 rank 顺序加；W>8 的 one-shot 按 8 个一组分块（`reduceLamportRanksChunked<W,8>`）；two-shot 块大小 `min(W,16)`。每种模式各自确定，但 auto 按大小切换，顺序和也不能拆成子树。只借用了它的 Lamport/multicast 传输做法。

## 4. 推荐与实施计划

### 4.1 最终契约

采用 §2：dense G=16 bf16 叶子 fp32 平衡树；MoE 规范槽位加权 16 叶树；尾部 bf16 相加一种形式；传输为 Lamport push + NVLS multicast，T·H·2 ≤ 32 KiB 用 one-shot（MoE 是 T ≤ 8），其它 two-shot。

### 4.2 vLLM overlay（生成器，不改别人的文件）

生成器 `agent_run/scripts/mk_tp_invariant.py`：

```
python3 mk_tp_invariant.py IMAGE_SRC_VLLM OVERLAY_ROOT TPINV_AR_CU [MOE_FI_W4A16_OVERLAY_DIR]
```

集群上的生成命令（输出 `$WORK/k31/tpinv/overlay/tp-invariant/`）：

```
cd $WORK/k31/tpinv
python3 scripts/mk_tp_invariant.py $WORK/k31/image-src/vllm overlay scripts/tpinv_ar.cu \
    $WORK/k31/bi/overlays/moe-fi-w4a16-bi
```

| overlay 文件 | 来源 | 改动 |
|---|---|---|
| `k31_tree_allreduce.py`（distributed/device_communicators/，新） | 生成 | `K31TreeAllReduce`：dense 的 bf16 与 fp32 两个实例；`K31_MOE_TREE=1` 时再加 MoE 的两个实例（bf16 线给不开 LoRA，fp32 线给专家 LoRA）；`arm_moe_tree`、`moe_tree_lora`、`take_moe_reduced`、`selftest_launch`、`selftest_call`；首次调用时断言 EP 组 = TP 组 |
| `k31_tree_allreduce.cu`（新） | `tpinv_ar.cu` 原样 | 首次使用时按本机架构 JIT 编译（`K31_TREE_BUILD` 指定目录） |
| `cuda_communicator.py` | 镜像源码 | `VLLM_BATCH_INVARIANT=1` 且 `K31_TP_TREE=1` 时 TP 组 all_reduce 走树；bf16 走 bf16 线，fp32 走 fp32 线，整数仍走 NCCL |
| `latent_moe_runner.py` | 镜像源码 | BI 下关 tier0（连带不做 deferred finalize）；尾部固定为 `_k31_bi_tail`：latent（MoE 钩子已归约就直接用，否则树 AR）→ norm → up_proj mm → `add_(树 AR(shared))` |
| `model.py` | moe-fi-w4a16-bi 的 model.py（不给第 4 个参数时用镜像源码） | BI 下关 GEMM-AR / GEMM-RS |
| `mxfp4.py` | moe-fi-w4a16-bi 的 mxfp4.py | 两个必需锚点：(1) `K31FiW4A16Experts.apply` 里 8 空格缩进的 `launch_w4a16_moe(` 前插入 arm 一行；(2) `_k31_apply_lora` 末尾按 k 累加的第一行 `acc = slot[:, 0].clone()` 前插入 `if _k31_tree_lora(slot, e2p, topk_ids, output): return`，即开树时用 fp32 叶子走树，不再按 k 累加。两个可选锚点（找不到只打 warning）：`_k31_selftest` 里单行重算的 `launch_w4a16_moe(` 和 `self._k31_apply_lora(o1, ` 包上 `selftest_launch` / `selftest_call`。MoE agent 改其它地方不受影响 |

钩子工作方式：
- arm 时通过 `launch_w4a16_moe.__globals__` 把 FlashInfer 的 `moe_unpermute` 换成包装函数（只装一次）。
- 被 arm 的那一次调用里，包装函数不做本地加权和，而是用 GEMM2 行和 `expanded_idx_to_permuted_idx` 调 `moe_combine`，把完全归约的 latent 写进 `output`，并置标志。
- 未 arm 的调用（自测、其它路径）照常走原函数。
- `_k31_bi_tail` 读到标志就跳过 latent AR。标志没被尾部消费、下一次又 arm，会直接报错，不会静默错位。
- 专家 LoRA 路径：`moe_tree_lora` 拿 `_k31_apply_lora` 算好的 fp32 叶子 `slot`，调同一个 combine（fp32 线），同样置标志。
- selftest：`selftest_launch` / `selftest_call` 先把本层还没被尾部消费的标志暂存，单行重算也走树（集合通信，各 rank 重算的行相同），再恢复标志。所以 `K31_MOE_EXPERTS_SELFTEST=1` 现在比较的是树的结果。
- 走不到钩子的路径（Marlin 回退）仍用树 AR（本地部分和），只做到 TP 不变，第一次会打 warning。

开关：
- `VLLM_BATCH_INVARIANT=1 K31_TP_TREE=1`：dense 树 AR。
- 再加 `K31_MOE_TREE=1`：EP 不变 MoE combine。
- `K31_MOE_TREE_BASE` / `K31_MOE_TREE_LORA`（默认都是 1）分别分配不开 LoRA（bf16 线，约 0.35 GiB）和开 LoRA（fp32 线，约 0.5 GiB）的缓冲；RL 一直开 LoRA，可以设 `K31_MOE_TREE_BASE=0` 省掉前者。
- 调参：`K31_MOE_TREE_MAX_T`（默认 8192，需 ≥ max_num_batched_tokens）、`K31_MOE_TREE_ONESHOT_T`（默认 8）、`K31_TREE_MAX_ELEMS`、`K31_TREE_ONESHOT_MAX_BYTES`（默认 32768）、`K31_TREE_MC`（默认 1）。

限制：
- 挂载顺序：tp-invariant 必须在 moe-fi-w4a16-bi 之后，它会覆盖后者的 mxfp4.py 和 model.py。MoE agent 改了这两个文件后，要重跑生成器。
- compressed-tensors 的 MXFP4 路径没挂钩子（K31 checkpoint 走 mxfp4.py）。

### 4.3 训练侧

规格写在 §6。

### 4.4 下一步

1. 端到端 slice：等 KDA BI kernel 和 DS4 sparse-MLA 移植落地后由协调方触发。挂载顺序：…其它 BI overlay → moe-fi-w4a16-bi → tp-invariant；环境变量 `VLLM_BATCH_INVARIANT=1 K31_TP_TREE=1 K31_MOE_TREE=1`。验收：prefill==decode、逐层与训练侧参考比较。
2. 训练侧按 §6 适配，先在 4 层 proxy 上逐层比较。

## 5. 产物与作业

| 内容 | 路径 |
|---|---|
| 源码 | `agent_run/scripts/{tpinv_ar.cu, mk_tp_invariant.py, k31_tree_ref.py, moe_tree_fi_test.py, tpinv_prof.py, tpinv_ar_test.py, tp_tree_smoke.py, tpinv.sbatch}`，各带 `.meta.json`；集群副本 `$WORK/k31/tpinv/scripts/` |
| 生成的 overlay | 集群 `$WORK/k31/tpinv/overlay/tp-invariant/`（6 个文件 + MOUNTS） |
| 结果 JSON | `agent_run/results/tpinv/`：n4def-19933、n4pin-19937、n4pdl-19938、n4mc-19941、n4uc-19942、smoke1-19954、smoke4-19955（旧）；moefi4-19974、moefi1-19975（第一版钩子）；prof-19976、moesyn-19977、prof3-19986、n4v3-19987、smoke4-19988、n4v4-19995、moefi4-19996、moefi1-19997、lorafi4-20008、lorafi1-20009、moesyn2-20010（本轮），各带 .meta.json。集群原件 `$WORK/k31/tpinv/results/k31tp-*/result.json` |
| FlashInfer 源码抽取（只读） | 集群 `$WORK/k31/tpinv/fisrc/`（含 `fused_moe/cute_dsl/moe_utils.py`） |
| 本轮作业 | 19966/19967（钩子第一版，通过）、19974/19975（同上复验）、19976（profile）、19977（合成 MoE）、19982/19983（profile 与回归，中间版本）、19984（EP16，节点 NCCL NVLS 初始化失败 801，不是代码问题，排除 node01/node04 后重提）、19992（误提，自行取消）、19986/19987/19988/19995/19996/19997（不含 LoRA 的最终版本）、20008/20009/20010（专家 LoRA 叶子、selftest 兼容、回归；最终版本） |

说明：19975（1 节点）和 19976（4 节点）的依赖都写成了 19974 之后，导致两者同时运行了约 2 分钟，共占 5 个节点，超过 4 节点预算。之后的作业都改成了串行依赖。

## 6. 训练侧规格（Megatron / mLite）

### 6.1 共用参考

`agent_run/scripts/k31_tree_ref.py`（集群 `$WORK/k31/tpinv/scripts/k31_tree_ref.py`）是纯 torch 文件，推理测试与训练侧都 import 它。`python k31_tree_ref.py` 会跑自测（TP=1/2/4/8/16 子树分解一致、任意行顺序的 MoE combine 一致）。

| 函数 | 作用 |
|---|---|
| `tree_sum(leaves)` | fp32 相邻配对平衡树，返回 fp32 不舍入 |
| `combine_leaves(leaves)` | 16 个叶子的完整 all-reduce 结果：舍 bf16、-0 改 +0 |
| `row_parallel_leaves(x, weight, mm)` / `row_parallel_ref` | 整个行并行层：K 切 16 片，每片一次 GEMM 出 bf16，再 `combine_leaves`。`mm` 传两侧约定的 BI GEMM |
| `local_subtree(leaves)` / `cross_rank_tree(partials)` | TP<16 时：本地 L=16/TP 个叶子的 fp32 子树根；跨 rank 把各 rank 的子树根按 rank 顺序走剩下的层 |
| `moe_lora_leaves(base, d2, topk_weights)` | `[T,16,H]` fp32 叶子，router 槽位顺序：`w * base.float()`，再 `+= d2.float()`（d2=None 即不开 LoRA） |
| `moe_combine_leaves_ref(leaves, topk_ids)` | MoE 契约本身：按 expert id 升序重排 16 个叶子，走树，舍 bf16，-0 改 +0 |
| `moe_combine_ref(y, topk_ids, topk_weights, d2=None)` | = `moe_combine_leaves_ref(moe_lora_leaves(y, d2, w), ids)`；y 是未加权专家输出 `[T,16,3584]` bf16 |
| `moe_combine_torch(y_rows, token, slot, topk_ids, topk_weights, d2_rows=None)` | 训练 EP 任意：combine 回 token owner 的行（及其 LoRA 增量行）以任意顺序到达时，先按 (token, slot) 填表再调用 `moe_combine_ref` |
| `latent_tail_ref(routed_latent, shared_leaves, norm, up_weight, mm)` | norm → up_proj → `add_(combine_leaves(shared_leaves))` |

vLLM 侧的验收就是对着这些函数做的：`moe_tree_fi_test.py` 用 `moe_combine_ref`（不开 LoRA 和开 LoRA 两种）当参考，EP4/EP16 都 0 差异。所以训练侧只要叶子（GEMM 输出、专家输出、路由权重）逐位相同，用这个文件的表达式得到的就是推理的值。

### 6.2 dense 行并行层，TP=t（t 整除 16），以 TP4 为例

- 适用：KDA/MLA o_proj、共享专家 down_proj、首层 dense down_proj。
- 前提：Megatron RowParallelLinear 按 rank 连续切 K，rank r 持有 K 块 r。K 的列顺序必须与 vLLM TP16 的分片顺序相同：TP16 的 rank g 就是叶子 g。头的排列如果有交错（比如融合 qkv 时重排过），先还原成与 vLLM 相同的头顺序。
- 每个 rank 持有 L = 16/t 个连续叶子（TP4 时 L=4；o_proj 每叶 768，共享 down 每叶 384，dense down 每叶 2112）。
- 前向，torch 写法：

```python
kl = K // 16                      # 叶子宽度
L = 16 // tp                      # 每 rank 叶子数
leaves = [bi_mm(x_loc[:, i*kl:(i+1)*kl], w_loc[:, i*kl:(i+1)*kl].t()).to(torch.bfloat16)
          for i in range(L)]      # 每片单独一次 GEMM，bf16 输出
p = k31_tree_ref.local_subtree(leaves)          # TP4: (l0+l1)+(l2+l3)，fp32
parts = all_gather(p, group=tp_group)           # [tp, T, N] fp32，按 rank 顺序
y = k31_tree_ref.cross_rank_tree(list(parts))   # TP4: (p0+p1)+(p2+p3) -> bf16 -> -0 改 +0
```

- 通信：all_gather fp32 是正确但慢的写法（4 倍字节）。等价的快速写法是同一个 CUDA kernel `tree_all_reduce`，fp32 线、W=tp（原型测过 W=4/8 fp32 线逐位相同）。
- 开 SP（reduce-scatter）时：每个元素仍是同一棵树，rank 只保留自己的 token 块。可以先算完整树再切，也可以用 two-shot 的前两相。
- TP=1 也要切 16 片走树，不能用整 K 的一次 GEMM。
- 反向：不在契约内，照常用 Megatron 的反向和通信。

### 6.3 MoE（EP 任意，ETP=1）

1. 路由：topk_ids 和 fp32 权重必须与 vLLM 逐位相同（router 契约，另一个 agent 负责）。权重里已包含 routed_scaling_factor。
2. 专家：每个 (t,k) 的输出 base 是未加权的 bf16 `[3584]`，必须与 FI W4A16 逐位相同（MoE kernel 契约，另一个 agent 负责）。专家 LoRA 增量 d2 是 bf16、已乘 w，必须与 vLLM 的 punica 路径逐位相同（同属 MoE/LoRA kernel 契约）。
3. combine：token dispatcher 把专家输出送回 token owner 后，不能用 Megatron/TE 的加权 unpermute（scatter_add 顺序不定，`moe_permute_fusion` 带 probs 也不行），也不能先把 LoRA 增量按 k 累加。改为：

```python
# rows: [R, 3584] bf16 未加权 base；d2_rows: [R, 3584] bf16 LoRA 增量（已乘 w）；token/slot: 每行的 token 下标和 router 槽位 k
latent = k31_tree_ref.moe_combine_torch(rows, token, slot, topk_ids, topk_weights, d2_rows)   # [T, 3584] bf16
# 等价的展开写法（router 槽位顺序的 base, d2: [T,16,3584]）:
leaf = topk_weights.float().unsqueeze(-1) * base.float()
leaf.add_(d2.float())                       # 不开 LoRA 时省略这一行
order = topk_ids.argsort(dim=1, stable=True)
lc = leaf.gather(1, order[..., None].expand_as(leaf))
v = [lc[:, c] for c in range(16)]
while len(v) > 1:
    v = [v[i] + v[i+1] for i in range(0, len(v), 2)]
latent = v[0].bfloat16(); latent = torch.where(latent == 0, torch.zeros_like(latent), latent)
```

4. 关掉 `moe_apply_probs_on_input`（权重必须乘在专家输出上）。
5. 不需要知道 vLLM 的 EP 分组：结果只依赖 expert id，训练任意 EP、任意专家放置都一样。
6. latent 尾部：`latent_tail_ref` 的顺序，即 latent RMSNorm（BI kernel）→ replicated up_proj（BI GEMM，bf16 输出）→ `out.add_(shared_root)`。shared_root 按 §6.2 算（K=6144 共享 down 的 16 叶树）。这是两次舍入（mm 输出一次、相加一次），不能用 addmm。

### 6.4 其它

- Embedding：训练侧在 embedding 输出上做 -0 改 +0（vLLM 树 AR 的根会这样处理）。
- 训练侧要关掉：TP 通信重叠（userbuffers 的 reduce-scatter/all-gather 融合 GEMM）、NCCL all-reduce/reduce-scatter 用在上述层的前向上、TE 的 fused unpermute。
- 验收：同一 batch，训练 TP4/EP16 与推理 TP16/EP16，逐层比较 hidden（bf16 视为 int16，必须 0 差异）。先在 4 层 proxy 上做。叶子不一致时先查：头顺序、BI GEMM 是否同一实现、路由权重、专家输出。
