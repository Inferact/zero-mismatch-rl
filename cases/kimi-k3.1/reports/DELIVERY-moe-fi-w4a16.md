# K31 routed MoE：FlashInfer CuTe-DSL W4A16（BI）交付说明

决定：两臂（BI=0 与 BI=1）同一 recipe，routed 专家都走 FlashInfer CuTe-DSL sm100 W4A16（下称 FI），接受每卡约 5 GB 的 scale 显存代价。

## 1. overlay 改了哪些文件

生成器 `$WORK/k31/bi/scripts/mk_moe_fi_w4a16_bi.py`，输出 `$WORK/k31/bi/overlays/moe-fi-w4a16-bi/`：

```
python3 mk_moe_fi_w4a16_bi.py <镜像 vllm 源码> $WORK/k31/bi/overlays [xlora-ep overlay 目录]
```

| 文件（vllm 包内路径） | 改动 |
| --- | --- |
| `model_executor/layers/quantization/mxfp4.py` | K31 实际走的路径（`quantization=mxfp4`，`Mxfp4MoEMethod`）。开关打开且激活为 SiTU 时，专家类换成 `K31FiW4A16Experts`，加载时做无损重排（见第 3 节），覆盖镜像自动选的 TRTLLM MXFP4×MXFP8 和 `--moe-backend marlin` |
| `model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w4a4_mxfp4.py` | 同一开关，compressed-tensors 格式的 checkpoint 用；K31 当前不走 |
| `models/kimi_k3/nvidia/model.py` | 以 serving `xlora-ep/model.py`（`lora` 的超集）为底，加一个默认关闭的整块自检 `K31_MOE_SELFTEST`。原来单独的 `moe-fi-selftest` overlay 已删，不再和 lora / xlora-ep 抢同一个文件 |

挂载顺序：`lora + xlora-ep + ...（其余 serving overlay）+ moe-fi-w4a16-bi`，本 overlay 最后挂。

## 2. 环境变量

| 变量 | 默认 | 作用 |
| --- | --- | --- |
| `K31_MOE_FI_W4A16` | 开 BI 时为 1，否则为 0 | 1 = routed 专家走 FI。**两臂都要显式设 1**（BI=0 臂不设就回到 TRTLLM / Marlin） |
| `K31_MOE_FI_ALLOW_LOSSY` | 0 | scale 跨度超过 17 binade 时默认断言失败；设 1 改为截断。全量 checkpoint 实测用不到 |
| `K31_MOE_FI_TACTIC` | auto | 只影响速度。所有 tactic 的输出逐位相同，auto 按每个专家平均行数选 |
| `K31_MOE_EXPERTS_SELFTEST` | 0 | 只用于测试（eager）：每次专家调用把第 0、M/2、M-1 行单独重算一遍，逐位比较，打印 `K31FIBI` |
| `K31_MOE_SELFTEST` | 0 | 只用于测试（eager）：整个 MoE 块（router、latent、专家、EP 归约、shared）逐行自检，打印 `K31MOEBI` |
| `K3_XLORA_EP_F32` / `VLLM_DISABLE_SHARED_EXPERTS_STREAM` | 1 / 1 | 沿用 serving recipe |

## 3. 权重：无损，训练侧直接用 MXFP4 反量化

- fp4 字节不变。w13 的行从 [gate; up] 重排成 [up; gate]，再按 64 行交错。
- 每个 E8M0 per-32 scale 写成两个 E4M3 per-16 的 2 的幂 scale。每个专家、每个 GEMM 一个 2 的幂 `alpha = 2^(max_e − 8)`，在 epilogue 里乘到 FP32 累加结果上。只要一个专家内 scale 跨度不超过 17 binade 就是逐位精确；prep 里有断言。
- 全量 checkpoint 扫描（92 个 MoE 层、82432 个专家、247296 个 scale 张量）：w13 最大跨度 13 binade，w2 最大 5，有损块 0。日志：`$WORK/k31/bi/logs/moe-scanfull2-19956.out`（`SCAN_DONE ... verdict=LOSSLESS`）。
- kernel 里每个 E2M1×E4M3 解码成 BF16 都是精确的，所以 kernel 实际乘的权重就是 `MXFP4 反量化值 / alpha`，结果再乘 alpha（2 的幂，精确）。训练侧的权重 = checkpoint 的 MXFP4 精确反量化，不需要任何仿真。

## 4. 前向契约（训练侧必须复现的）

**结论：逐位一致只能靠训练前向调用同一组 kernel，不能用 Megatron / cuBLAS 的 GEMM 自己重写。**

FI kernel 每一行的算法：
1. 每个 CTA 拿一个 tile 的全部 K（K tile = 256），不做 split-K，没有 atomic。K tile 按顺序走；tile 内每条 tcgen05 BF16 UMMA 是 K=16，按顺序累加进 TMEM 的 FP32 累加器。一行的结果和 batch 大小、同 tile 的其他行、tactic 都无关（已测：M = 1..8192、所有 tactic、多次重跑都逐位一致）。
2. UMMA 内部 16 个乘积怎么求和由硬件决定，不是 IEEE 顺序加法。cuBLAS / Megatron 的 BF16 GEMM 用不同的 tile、split-K 和指令，K 顺序不同，**做不到逐位一致**。
3. GEMM1 epilogue（无 LoRA 路径）：`alpha·acc`（FP32）→ 融合 SiTU，用的是 fastmath 的 FP32 tanh / sigmoid（`tanh = 2·sigmoid(2x) − 1`）→ 一次舍入成 BF16。GEMM2：`alpha·acc` → BF16。top-k 合并由 `moe_unpermute` 完成：每个 token 按 k 的固定顺序把 `w_k · out_k` 加起来。
4. 有专家 LoRA 时是另一条管线（`K31FiW4A16Experts._k31_apply_lora`），不能和无 LoRA 路径混用：
   - GEMM1 不融合激活，直接输出 BF16 的 [gate | up]；
   - LoRA w13 增量用 punica `fused_moe_lora` 算，Triton config 钉死（BLOCK 64/64/32，SPLIT_K 1，不随 M 变），写进一个 BF16 零缓冲；
   - `FP32(gate|up) + FP32(delta)` → `torch.ops._C.situ_and_mul`（FP32）→ BF16；
   - GEMM2 → 每个 slot 先算 `FP32(w_k·base_k) + FP32(LoRA w2 增量_k)`，再按 k = 0..topk−1 的顺序在 FP32 里累加 → 一次舍入成 BF16。
   - adapter 为 0 时，这条管线和无 LoRA 路径**不**逐位相同（由实现推断，未单独测）：SiTU 的输入先被舍成 BF16，且 SiTU 换了实现。所以 RL 里只要带 LoRA，训练就必须复现这条 LoRA 管线。
5. 训练侧要做的：
   - 前向直接调用 FlashInfer 的 `moe_sort` / `moe_permute` / `_run_grouped_gemm`，tactic 随意；
   - 有 LoRA 时，增量用同一个 punica `fused_moe_lora` 和同一组钉死的 config 算；
   - SiTU 和合并顺序按上面写的做。
   - 反向不受约束，可以用 MXFP4 反量化权重配普通 GEMM。
   - 建议下一步把 `_k31_apply_lora` 抽成不依赖 vLLM LoRA 上下文的纯函数（输入：x、topk_ids、topk_weights、重排后的权重、alpha、lora_a / lora_b），训推两侧 import 同一份代码。

## 5. 验证（slice15，TP4+EP，serving recipe，adapter xall 非零，1024 输入 / 256 输出）

- serving 内逐位自检：FI + LoRA 路径 eager，`K31_MOE_EXPERTS_SELFTEST=1`。rank 0 共 14401 次专家调用、43203 行与单行重算逐位相同；4 个 rank 都没有不一致行；约 99% 的调用 LoRA 增量非零。作业 19959。
- 解码吞吐（输出 tok/s，括号内为中位 TPOT ms），作业 19960（Marlin）/ 19961（FI）：

| 并发 | Marlin graph | FI graph | Marlin eager | FI eager |
| --- | --- | --- | --- | --- |
| 1 | 88 (10.5) | 96 (9.8) | 6.9 (146) | 8.6 (116) |
| 8 | 313 (18.5) | 374 (17.5) | 52 (148) | 59 (136) |
| 16 | 476 (20.2) | 606 (19.0) | 99 (148) | 114 (134) |
| 32 | 619 (38.2) | 878 (28.4) | 189 (152) | 210 (142) |
| 64 | 770 (55.2) | 1212 (32.9) | 330 (166) | 398 (143) |
