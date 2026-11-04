# K31 BI1 部署交接：全模型 DP16+EP16（GB300 4 节点）

更新日期：2026-10-16。本文是当前 BI1（批不变 + 训推一致）全模型部署的唯一交接文档：镜像、挂载顺序、环境变量、启动参数、每个 overlay 的用途和门控作业。

所有路径都在 GB300 集群 集群上，根目录 `$WORK`。模型、镜像和权重来源只在集群和本机出现，文中用路径变量表示。

## 1. 部署形态

| 项 | 值 |
|---|---|
| 并行 | attention TP=1，DP16（每节点 4 个 DP rank），MoE EP16（每 rank 56 个本地专家） |
| 节点 | 4 个 GB300 tray，同一 NVLink clique（MNNVL） |
| 镜像 | `$IMG`，定义在 `k31/bi/jobs/k31int_dpmn.sbatch` 第 19 行（enroot sqsh） |
| 模型 | `$MODEL=$WORK/models/kimi-k3.1` |
| 适配器 | `$WORK/k31/perf/ref/adapter0`（r=32，含专家 LoRA） |
| CUDA graph | `-cc.cudagraph_mode=FULL`，始终开启；eager 只用于调试 |
| 启动器 | `k31/bi/scripts/k31int_dpmn_submit.sh` → `jobs/k31int_dpmn.sbatch` → `k31int_dpmn_node.sh`（每节点一个容器，节点 0 起 API server） |

一键复现（全模型，matrix + perf + c=1 profile）：

```bash
B=$WORK/k31/bi; cd $B/scripts; R=$B/results
export ROUTER_OVL=$B/overlays/bi-gemm-r5
export EXTRA_OVL=$B/overlays/fmlora-policy+$B/overlays/kda-cfbi-slot+$B/overlays/kda-cfbi-aligncopy+$B/overlays/bi-nccl-unpin2
NODES=4 MODEL=$WORK/models/kimi-k3.1 AD=$WORK/k31/perf/ref/adapter0 \
  bash k31int_dpmn_submit.sh <name> both "-cc.cudagraph_mode=FULL+--max-num-seqs+32" \
  K31_MOE_TREE_MAX_T=16384+K31INT_MBT=1024+K31INT_GMU=0.92+VLLM_ENGINE_READY_TIMEOUT_S=5400+K31_BI_NCCL_UNPIN=1
```

`matrix` 只跑一致性矩阵，`perf` 只跑性能，`both` 两个都跑。加 `K31P_PROFILE=2+K31P_PCONC=<并发>` 会在 perf 之后再抓一次 torch profile，profile 目录要放在共享盘（`--profiler-config` 里写 `$R/prof-<name>`）。

## 2. 挂载顺序

`k31int_dpmn_submit.sh` 按下面的顺序把各 overlay 的 `MOUNTS` 逐行挂进镜像的 vllm 包（`/usr/local/lib/python3.12/dist-packages/vllm/...`）。同一目标文件出现多次时，**后挂的生效**。所以几个改同一文件的 overlay 是链式生成的：后一个以前一个的输出为基础，包含前面全部改动，顺序不能调换。

前缀说明：S = `k31/serving/overlay`，B = `k31/bi/overlays`，T = `k31/tpinv/overlay`。

| # | overlay | 主要文件 | 用途 | 门控作业 |
|---|---|---|---|---|
| 1 | S/lora | models/kimi_k3/nvidia/model.py, lora/layers/base_linear.py, column_parallel_linear.py | K31 模型的 LoRA 支持（KDA/DSA 各投影可挂 LoRA） | 19887（GB200 集群 全栈 matrix） |
| 2 | S/xlora-ep | fused_moe.py, marlin_moe.py, latent_moe_runner.py, model.py, base_linear.py | 专家 LoRA 在 EP 下可用（专家共享 LoRA，按 rank 的 token→LoRA 映射） | 19887 |
| 3 | S/xlora-ep-fastalign | marlin_moe.py | EP+LoRA 的对齐加速 | 19776, 19887 |
| 4 | B/lora-f32add-bi-r3 | lora/layers/base_linear.py | LoRA 叶子：`bf16(fp32(x·W) + fp32(delta))`，一次舍入，融合在 bi_gemm 的 epilogue 里 | bigemm_lora 门控, 20354 |
| 5 | S/lora-hostfree | lora/worker_manager.py | 去掉 LoRA 管理的主机侧开销 | 19887 |
| 6 | B/ds4-indexer-topk | sparse_attn_indexer.py | DSA indexer top-k 批不变（DS4 移植，升序索引） | 19934（52/52） |
| 7 | B/ds4-sparse-mla | flashmla_sparse.py | sparse MLA 固定 split 规划，批不变 | 19935（36/36） |
| 8 | B/ds4-sparse-mla-allow | flashmla_sparse.py | BI 下放行 sparse MLA 后端（在 7 的基础上） | 19887 |
| 9 | B/sparse-mla-tp1 | flashmla_sparse.py | TP=1 下的 sparse MLA 修正（在 8 的基础上） | 20354 |
| 10 | **B/bi-gemm-r5**（由 `ROUTER_OVL` 指定） | batch_invariant.py, gate_linear.py, mla.py, k31_bi_gemm*.{py,cu,cuh} | 所有 bf16 dense linear、router、LoRA 叶子都走 k31 split-K GEMM（K>3584）或 cuBLAS S=1（K≤3584）。r5 加了 decode 小 M 的 swap-AB 均衡 kernel 和 PDL，数值和 r4 逐位相同 | r4: 20354, 20363；r5: 20416 单算子门控 ALL_OK, 20418 DP4, 20433 全模型 matrix + COMPARE PASS |
| 11 | B/moe-fi-w4a16-bi | compressed_tensors_moe_w4a4_mxfp4.py, mxfp4.py, model.py | MoE 专家走 FlashInfer CuTe-DSL W4A16，固定 tactic，专家 LoRA 固定顺序 | 19887 |
| 12 | T/tp-invariant | k31_tree_allreduce.{py,cu}, cuda_communicator.py, model.py, latent_moe_runner.py, mxfp4.py | 固定树 all-reduce / MoE combine（K31 tree） | 19933 |
| 13 | B/moe-tree-ep2 | latent_moe_runner.py | EP 下 MoE combine 走 K31 tree | 20354 |
| 14 | B/moe-tree-dp | cuda_communicator.py, k31_tree_allreduce.py, moe_runner.py, naive_dp_ep.py | DP+EP 下用 K31 tree 替换 finalize 里的 reduce-scatter | 20354 |
| 15 | B/moe-fi-bi-start | mxfp4.py | BI 下 MXFP4 oracle 拒绝时也能启动 FI W4A16 | 19887 |
| 16 | B/kda-cfbi | kda.py, kda_metadata.py, k31_kda_cfbi.py, kda_cfbi*.cu | KDA 层 CF-BI 批不变引擎（decode rows_fast，prefill 按 chunk 对齐） | 20352/20354 |
| 17 | B/dp-dummy-nolora | v1/worker/gpu_worker.py, lora_model_runner_mixin.py | 空闲 DP rank 跑 dummy batch 时不再建/删 dummy LoRA（c=1 从 398 ms 降到 12 ms 的修复） | 20357 |
| 18 | B/fmlora-policy | mxfp4.py, k31_moe_lora_fuse.{py,cu}, lora/ops/triton_ops/fused_moe_lora_op.py | 包含 moe-lora-fuse2（专家 LoRA 逐元素融合），并按 M 选 punica fused_moe_lora 的 BLOCK_M/NPID（只改启动形状） | 20378, 20382, 20390/20391, 20392, 20394 |
| 19 | B/kda-cfbi-slot | kda_cfbi6_dsync8.cu（它的 k31_kda_cfbi.py 被 19a 覆盖） | KDA 状态直接按 slot 原地读写，decode 去掉 index_select/index_copy_ | 20367, 20373 |
| 19a | **B/kda-cfbi-aligncopy**（紧跟 19 挂；由 `mk_kda_cfbi_aligncopy.py` 以 19 的 k31_kda_cfbi.py 为底生成，只多一处补丁） | k31_kda_cfbi.py | 修前缀缓存下 KDA 状态跨块拷贝不全。开前缀缓存（R3 必开）时 vLLM 用 mamba align 模式：请求长度跨过 align 块（块长 = attention block size，DP16 TP1 是 28224 token，TP8 是 3584）时，把 KDA 状态拷进新块的槽位。上游只拷 conv 和 temporal 两个状态，CF-BI 额外的 rowcache（fp32 [H,64,640]）和 pos（int32 [H]）没拷，新槽位带着上一个请求的旧值：KDA 输出错，pos 越界时非法访存。本 overlay 把 `MambaStateCopyFuncCalculator.kda_state_copy_func` 换成四个状态都拷。不开前缀缓存时不走这段代码；没跨块的请求也不走，所以数值和 19 逐位相同 | NV：TP8 15830（无修复，FAIL）/ 15908 / 15909；DP16：16253 对 16254（跨 28224 和 56448、含前缀命中，开/关前缀缓存逐位相同），16240 / 16250（不跨块时和旧 slot 逐位相同：matrix 对 15825、R3 对 15826 COMPARE PASS）；见 results/c10/kda-aligncopy-dp16.md |
| 20 | B/bi-nccl-unpin2 | determinism/batch_invariant.py, distributed/parallel_state.py | `K31_BI_NCCL_UNPIN=1` 时不加 BI 的 NCCL 钉死参数（1 channel 等）。启动时检查前置条件：BI=1、TP=PP=PCP=DCP=1、开 EP、K31_MOE_TREE=1，不满足就拒绝启动 | 20387, 20388, 20392（正向 + 反向）, 20394 |
| 21 | S/r3-chunked（开 R3 时必挂，放在 r3-hashblock / r3-pp 之后） | distributed/aux_output_connector/worker.py | R3 路由捕获改用 engine core 的 hash 单位（`kv_cache_config.hash_block_size`），修分块预填与前缀缓存命中时只剩最后一块路由并断言的问题；是 r3-hashblock、r3-pp 两个 worker.py 的超集，不改数值 | 20057, 20058（反向对照）, 20073, 20080, 20081；见 results/r3-chunked/REPORT.md |
| 22 | S/kvb-lora-guard（放在最后；GB200 集群 与 GB300 集群 两处同 sha `2b1a1004`） | lora/model_manager.py（没有别的 overlay 挂这个文件） | MLA/DSA 的 kv_b_proj LoRA 一律报错，不再静默失效（#48974）：启动时 target 列表含 kv_b_proj 就 ValueError；加载 adapter 时 kv_b_proj 的 lora_A、lora_B 都非零就 ValueError（全零的 delta 为 0，放行并 warning）。检查在 LRU 淘汰之前，被拒的加载不动缓存。只加检查，不改数值 | GB200 集群 20133：CPU 单测 20/20；切片 4 层 DP4+EP、FULL graph、1–20 全链（未开 R3）：普通 adapter 下有/无本 overlay matrix 都 PASS 且 COMPARE PASS（逐位相同）；运行时加载 kv_b_proj adapter 报错且服务继续正常，B=0 的放行；target 含 kv_b_proj 启动失败；`--lora-modules` 指向 kv_b_proj adapter 启动失败。见 results/g1-kvb-lora/GUARD.md |

几点说明：

- 10 号的 `batch_invariant.py` 会被 20 号覆盖。20 号就是以 bi-gemm 的 `batch_invariant.py` 为底生成的，所以 bi-gemm 的改动都在里面。
- `mxfp4.py` 由 11、12、15、18 依次链式生成，最终生效的是 18 号（fmlora-policy）。
- 要退回 bi-gemm-r4：设 `ROUTER_OVL=$B/bi-gemm-r4`，或保留 r5 但设 `K31_BI_GEMM_R5_POLICY=0 K31_BI_GEMM_PDL=0`。两种方式数值都不变。
- 19a 号（kda-cfbi-aligncopy）的生成：`python3 scripts/mk_kda_cfbi_aligncopy.py $B/overlays/kda-cfbi-slot/k31_kda_cfbi.py $B/overlays/kda-cfbi-aligncopy`。NV 上 19 号 k31_kda_cfbi.py 的 sha256 是 `4aa70608…`，生成结果是 `195e1a42…`；GB300 集群 上生成后先核这两个值。启动日志里每个进程会打印一行 `K31 kda-cfbi-aligncopy: KDA align-mode state copy covers conv, E, rowcache, pos`。
- 19a 只在开前缀缓存时生效。DP16 TP1 的 align 块是 28224 token（KDA 状态按 TP 切，TP1 每 rank 的状态是 TP8 的 8 倍，attention block 被放大到同一页大小），部署的 max-model-len 5120 远小于它，所以现在的 DP16 + R3 实际上既不会跨块，也不会有前缀命中；旧链的问题是潜在的，只要 max-model-len 超过 28224 就会出现。TP8 的块是 3584，在 5120 下就会跨块，必须挂 19a。
- 22 号挂法：在 `EXTRA_OVL` 末尾加 `+$WORK/k31/serving/overlay/kvb-lora-guard`。它只在切片 DP4 上做过门控（20133），还没跑过全模型 matrix；它只加检查，不在 forward 路径上。
- 生成器都在 `k31/bi/scripts/mk_*.py`（本机副本在 `agent_run/scripts/`），都用断言过的精确字符串替换。已存在的 overlay 目录不会被原地重写；要改就换新名字。

## 3. 环境变量

| 变量 | 值 | 作用 |
|---|---|---|
| VLLM_BATCH_INVARIANT | 1 | 打开 BI |
| K31_MOE_FI_W4A16 | 1 | MoE 专家走 FI W4A16（11 号） |
| K31_TP_TREE | 1 | 固定树 all-reduce（12 号） |
| K31_MOE_TREE | 1 | MoE combine 走 K31 tree（13/14 号）；unpin 的前置条件之一 |
| K31_MOE_TREE_MAX_T | 16384 | tree combine 的最大 token 数（全模型 MBT 1024 × DP16） |
| K3_XLORA_EP_F32 | 1 | 专家 LoRA 的 fp32 加法 |
| VLLM_DISABLE_SHARED_EXPERTS_STREAM | 1 | shared experts 不开第二条 stream |
| K31_BI_NCCL_UNPIN | 1 | NCCL 不钉 1 channel（20 号）；只在 TP=1+EP 且所有归约都在 K31 tree 时合法 |
| NCCL_MNNVL_ENABLE / NCCL_CUMEM_ENABLE | 1 / 1 | 跨 tray 的 NVLink |
| NCCL_SOCKET_IFNAME / GLOO_SOCKET_IFNAME | bond0.231 | GB300 集群 的控制网卡 |
| K31_KDA_CFBI_BUILD / K31_TREE_BUILD | /local/... | JIT 编译缓存放节点本地盘 |
| TRITON_CACHE_DIR | /local/triton | 同上 |
| VLLM_ENGINE_READY_TIMEOUT_S | 5400 | 全模型从共享盘加载约 500 s，再加编译和 graph capture |

可选开关（默认值就是部署值，只在 A/B 对照时改）：

| 变量 | 默认 | 说明 |
|---|---|---|
| K31_BI_GEMM_R5_POLICY | 1 | 0 = bi-gemm 用 r4 的启动方式（数值不变） |
| K31_BI_GEMM_PDL | 1 | 0 = bi-gemm 不用 programmatic dependent launch |
| K31_FMLORA_POLICY | 1 | 0 = punica fused_moe_lora 用 BLOCK_M 64 + 原 NPID 启发式 |
| K31_MOE_LORA_FUSE | 1 | 0 = 专家 LoRA 走原 torch 序列 |
| K31_KDA_CFBI_SLOT | 1 | 0 = KDA 走 gather/scatter |
| K31_BI_GEMM | 1 | 0 = 不用 k31 GEMM（会失去批不变，只用于排查） |

不要设 `VLLM_LORA_ENABLE_DUAL_STREAM`。

## 4. 启动参数（`k31int_dpmn_node.sh`）

```
vllm serve $MODEL --served-model-name k3 --port 8311
  --tensor-parallel-size 1 --data-parallel-size 16 --data-parallel-size-local 4
  --data-parallel-address <head_ip> --data-parallel-rpc-port 13345
  [非 head 节点: --headless --data-parallel-start-rank 4*node_rank]
  --max-model-len 5120 --max-num-seqs 32 --gpu-memory-utilization 0.92
  --max-num-batched-tokens 1024 --no-enable-prefix-caching
  --trust-remote-code --mm-encoder-tp-mode data
  --enable-lora --max-lora-rank 32 --max-loras 1 --enable-moe-shared-loras
  --lora-target-modules q_proj k_proj v_proj g_proj f_a_proj b_proj f_b_proj o_proj q_a_proj
      kv_a_proj_with_mqa q_b_proj gate_proj up_proj down_proj experts
  --lora-modules ad=$AD
  --enable-expert-parallel
  --attention-config '{"sparse_mla_force_mqa":true}'
  -cc.cudagraph_mode=FULL
```

- `--max-num-seqs 32`：64 会让 Mamba 块数不够（45 < 64），启动失败（20362）。
- `--gpu-memory-utilization 0.92`：BI1 时 KV 约 25 GiB，可放 64k–87k token（20363 / 20388）。
- 不要加 `--moe-backend marlin`：BI 支持检查会在 FI 覆盖之前拒绝它（20233）。
- BI0 对照：环境变量改成 `VLLM_BATCH_INVARIANT=0 K31_TP_TREE=0 K31_MOE_TREE=0 K3_LORA_F32ADD=0`，否则 lora-f32add 会在 BI0 下预建 fp32 权重副本，导致 OOM（20365）。

## 5. 验收方法

| 门控 | 命令 / 位置 | 通过条件 |
|---|---|---|
| bi-gemm 单算子 | `sbatch jobs/k31int_gpu1dir.sbatch <bi-gemm overlay> bigemm_tp1.py gate out.json` | `ALL_OK True`：各 shape、M=1..8192 探针行逐位不变；LoRA 叶子；行并行训练复现 |
| bi-gemm 启动形状 | `scripts/k31int_bigemm_r5.py KB=<overlay>` | `R5_ALL_EQ True`：所有启动模式逐位相同 |
| 引擎 DP4（切片模型，单节点） | `scripts/k31int_submit.sh <name> "dp4:matrix:..."` | `matrix.json` 的 PASS 为 True（batch 8/32/64、PD、rerun、每个 dprank 都和单发逐位相同） |
| 跨版本 COMPARE | `python3 scripts/k31int_matrix.py COMPARE=<旧>/matrix.json,<新>/matrix.json` | PASS（同一请求在不同版本间逐位相同）。基准：切片 DP4 用 20357，全模型用 20388 / 20394 |
| 全模型 | `k31int_dpmn_submit.sh <name> both ...` | matrix PASS，并和上一版全模型 COMPARE PASS |
| 每步耗时拆解 | `python3 scripts/k31int_step_breakdown.py <trace.json.gz> SKIP=3 TOP=25 MINK=50` | 用 `(outside graph kernels) ≈ 0.4 ms` 的那个 rank（它跑 sampler） |

## 6. 数值约定（训练侧要对齐的部分）

- bi-gemm 的 split 计划 S 只取决于 (K, N)，r5 和 r4 完全相同：
  - K≤3584 时 S=1；
  - K=6144 / 7168 / 12288 时 S=8；
  - K=33792 时 S=16；
  - 其余按 `k31_bi_gemm.default_split`。
- 每个 chunk 都是一次从零开始、k 升序的 fp32 累加；chunk 之间从左到右用 fp32 相加，最后舍入一次。LoRA 叶子先加 fp32(delta) 再舍入。
- 训练侧直接用 `k31_bi_gemm.chunks / chunk_partial / ordered_sum` 复现，r5 不需要任何改动。
- r5 新增的启动方式只改执行方式，数值不变：swap-AB 128×16 tile、chunk 在 CTA 间均衡分配、partial 先写 fp32 workspace 再由最后到达的 CTA 按序求和、PDL。门控已验证逐位相同（20405–20417）。
- DSA indexer 的 top-k 索引是升序（ds4-indexer-topk），训练侧要按升序排。

## 7. 已知限制

- bi-gemm r5 的 persistent kernel 用按 (K, N) 分配的计数器切片，第一次调用时从每个设备的计数池中分配，kernel 结束时清零。**同一 (K, N) 的两次 GEMM 不能在两条 stream 上并发**。当前模型里不会出现：DSA 的双 stream 用的是不同的 N；shared experts 不开 stream。
- `K31_BI_NCCL_UNPIN=1` 只适用于 TP=1 + EP、而且所有归约都在 K31 tree 的配置，启动检查会拦住其它配置。
- profile 不要开 `with_stack`：会在后处理阶段卡 30 分钟以上。

## 8. 性能现状（全模型，BI1，TPOT ms）

| 作业 | 配置 | c=1 | c=16 | c=64 |
|---|---|---|---|---|
| 20374 | 初版（无融合） | 102.6 | – | – |
| 20388 | + fuse2 + slot + unpin | 77.5 | 96.1 | 121.0 |
| 20394 | + fmlora-policy | 67.8 | 82.2 | 122.0 |
| 20433 | + bi-gemm-r5（matrix PASS，COMPARE vs 20394 PASS，prefill 9.7k tok/s） | 65.8 | 95.7；复测 20443 交替三次 89.0 / 85.9 / 84.1（c=1 同时 66.0 / 65.0 / 65.2），95.7 是波动 | 123.9 |
| 20371 | BI0 对照 | 76.5 | 106 | 180 |
