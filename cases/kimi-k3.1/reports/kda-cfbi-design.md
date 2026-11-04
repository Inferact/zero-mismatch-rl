# KDA chunk-form BI（CF-BI）约定：设计草稿（2026-10-14）

## 为什么

CPR 的约定是：state pass 算出各 chunk 的入口状态，chunk 内部按逐 token 递推重放。在 b64、H6、L8192 下：

| 项 | 耗时 |
|---|---|
| state pass（钉住配置） | 6.64 ms |
| CuTe replay | 9.78 ms |
| 合计 | 16.4 ms |
| BI=0 生产 prefill（FlashKDA） | 2.46 ms |

CPR 合计是 FlashKDA 的 6.7 倍。replay 只能用 FP32 CUDA core 做 rank-1 更新，按 FFMA2 打满算，地板也有约 2.6 ms。所以只要沿用「prefill 按 decode 的递推顺序走」的约定，就不可能接近 FlashKDA。

## 硬件前提

已实测（作业 19716）：mma.sync m16n8k16 bf16→fp32，沿 K=128 链式累加。第 r 行的结果既不依赖 A 的其它行，也不依赖 r 在 tile 里的位置（2000 次随机试验，0 例不同）。

因此 decode 可以把单行放进同一个 MMA 形状里计算，得到的结果和 prefill 整块 MMA 里该行的结果逐位相同。tcgen05/UMMA 还要另外测。

## 数学（单个 chunk，BT=64，从序列开头对齐）

记号：
- 入口状态 E 为 [V,K] 的 FP32 矩阵。
- 每个 token 的对数衰减 a_t[k] = lower·sigmoid(exp(A_log)·(g_t + bias))，取值 ≤ 0。
- chunk 内累加 G_t = G_{t-1} + a_t，**按 token 顺序串行相加**。prefill 和 decode 用完全相同的加法顺序。
- q、k 先做 L2 归一化，q 再乘 scale；β = sigmoid(raw_beta)。

几个量的定义：
- 子块参考点：r(t) = 16·floor(t/16)，只依赖 t 本身，是因果的。
- Akk[t,j] = β_t · Σ_k (k_t·exp(G_t − G_r))[k] · (k_j·exp(G_r − G_j))[k]，其中 j < t，r = r(t)。
- Aqk[t,j] = Σ_k (q_t·exp(G_t − G_r))[k] · (k_j·exp(G_r − G_j))[k]，其中 j ≤ t。
- 指数因子的范围：j ≥ r 时 exp(G_r − G_j) ≤ e^75；j < r 时它 ≤ 1，此时乘积的真值本身就很小，下溢不会丢精度。
- WY 行：
  - ũ_t = β_t v_t − Σ_{j<t} Akk[t,j] ũ_j
  - w_t = β_t (Γ_t k_t) − Σ_{j<t} Akk[t,j] w_j
  - 其中 Γ_t = exp(G_t)，j 从小到大串行累加。
- u_t = ũ_t − E·w_t（E·w_t 是一个按行的乘积）。
- o_t = E·(Γ_t q_t) + Σ_{j≤t} Aqk[t,j] · u_j，j 的累加顺序固定。
- chunk 结束：E' = E·Γ_end + Σ_j u_j (k_j·exp(G_end − G_j))^T。这是一个固定的转移函数；prefill 的跨 chunk 扫描和 decode 在 chunk 填满时调用的是同一个函数。

## 逐位一致的来源

- 行 t 的每一个量只依赖同一 chunk 里的 j ≤ t 和 E。归约顺序固定：点积用固定的树或 MMA 行，j 串行累加。
- prefill 并行算所有行；decode 把第 t 行单独算一遍，用的是同一段 device 函数或同一种 MMA 形状。
- 跨 chunk 的状态只在 chunk 边界通过同一个转移函数更新。

## 成本估计（每个 chunk-head）

**prefill：**
- CUDA core 部分：Akk 和 Aqk 的点积约 0.5M FMA，求逆约 0.26M，o 的 chunk 内部分约 0.26M。合计约 1M，只有 replay（4.2M）的 1/4。
- E 相关的乘积和 chunk 结束的转移走 MMA；为保留 FP32 精度，用 3×TF32 或 bf16×3 拆分。
- 跨 chunk 的扫描和 FLA 的 fwd_h 同构，每个序列内是串行的。

**decode（每个 token-head）：**
- 计算：Akk/Aqk 行约 16k FMA，求逆一行约 16k，再加两次 MMA 行乘积。
- 访存：读 E（64KB）加上当前 chunk 已有行的缓存（每行约 1.8KB：k、G、u、w），平均约 121KB。和 recurrent 每 token 读写状态的 128KB 相当。

## 需要改的地方

- vLLM KDA 状态：从「当前状态」改为「E + 当前 chunk 的行缓存」。在权重同步时清空。
- 训练 forward 复用同一个 prefill kernel；backward 用 FLA chunk_kda 的 VJP 直通，和 IsoExec 的做法一样。
- 混合批：其中的 decode 行和纯 decode 走同一个按行的函数，天然一致。

## 计划

1. CUDA 原型，正确性优先：先全部用 CUDA core 按固定顺序实现，验证 G1（prefill 与逐 token decode 逐位相同）、G2，以及和 FLA chunk_kda 的误差。
2. 再加速：把 E 的乘积和转移换成 MMA（3×TF32），chunk 并行，目标是接近 FlashKDA 的量级。
3. 和 CPR 做同节点对比，然后正式选定约定。

CPR 保留为已经验证过的保底方案。

## 约定 v4（2026-10-15 定稿）

v4 取代上面草稿里的数值细节；行缓存格式、chunk 对齐和「行 t 只依赖 j ≤ t 与 E」的结构不变。唯一真值是 oracle 行引擎 `rows5_kernel`（`agent_run/scripts/kda_cfbi5.cu`）。prefill（K1/K2）和 decode 引擎都必须与它逐位相同。

### 与 v3 的三处改动（已批准）

1. 门控与衰减改到 log2 域，所有指数用 `ex2.approx.f32`。
2. 所有矩阵乘积改为 tf32 MMA 的 3×TF32 拆分，prefill 和 decode 用同一定义。
3. 累加链变短：两个交叉项进单独的 SMALL 累加器，主项进 BIG，最后按固定顺序相加。

拆分方式在第 4 轮由舍入（rna）改为截断，因为截断在硬件上免费（见下文「已实测的硬件性质」）。

### 定义（全部显式 `_rn`，编译用 `--fmad=false`）

**门控（每个 head）**
- nah = −ex2(A_log·L2E)·L2E，low2 = lower·L2E，其中 L2E = log2(e) 的 fp32 值。
- x = g + bias。
- a2 = low2 · rcp.approx.ftz(1 + ex2(nah·x))。
- G2_t = G2_{t−1} + a2_t，在 chunk 内按 token 串行相加。
- β = rcp.approx.ftz(1 + ex2(−L2E·raw_beta))。

**归一化**
- ss = Σ x²，按 k 升序串行 fmaf。
- rn = rsqrt(ss + 1e-6)；q 还要再乘 scale。

**子块参考点**：r = 16·floor(t/16)。所有指数因子都写成 ex2(G2_a − G2_b) 的形式，只有 Γ_t = ex2(G2_t) 例外。

**tf32 乘积 P = A·Bᵀ（契约的核心）**
- 拆分：hi = trunc_tf32(x) = x 的低 13 位清零；lo = x − hi，这一步在 fp32 下精确，lo 进入 MMA 时同样按截断处理。
- 每个 K=8 步按 k 升序依次执行：SMALL += A_lo·B_hi，SMALL += A_hi·B_lo，BIG += A_hi·B_hi。第 0 步不带输入累加器。
- 结果 = BIG + SMALL，做一次 `__fadd_rn`。
- 一个 K=8 步就是一条硬件 tf32 MMA。下面三种实现逐位相同，可以混用：
  - tcgen05 UMMA kind::tf32，M=128；
  - mma.sync m16n8k8 tf32，累加器初值取 −0；
  - 「交换操作数」形式：A′=B、B′=A，两个交叉项的发射顺序互换。

**乘积清单**
- Akk/Aqk：A 为 {k_t·ex2(G2_t−G2_r), q_t·ex2(G2_t−G2_r)}，B 为 kn_j·ex2(G2_r−G2_j)，K=128。
- Akk = β_t·P。
- P1：W·Eᵀ；P2：(Γ_t·q_t)·Eᵀ。
- P3：KD·U，其中 KD[k][j] = kn_j[k]·ex2(G2_end[k]−G2_j[k])，K = j = 64，j 升序。

**串行部分（顺序固定）**
- WY：ũ_t = β_t·v_t，w_t = β_t·(Γ_t·kn_t)；每个 j < t 按 j 升序做 x = fmaf(−Akk[t,j], x_j, x)。
- u_t = ũ_t − P1_t。
- o_t = P2_t 起步，对 j ≤ t 按 j 升序做 fmaf(Aqk[t,j], u_j, ·)。
- chunk 结束：E'[v][k] = E[v][k]·ex2(G2_end[k]) + P3[k][v]，分别用 `__fmul_rn` 和 `__fadd_rn`。

### 已实测的硬件性质（GB300，sm_103a）

| 性质 | 脚本 | 试验数 | 不同 |
|---|---|---|---|
| UMMA 某行的结果与其它行、行位置、M 无关；某列与其它列、列位置、N 无关（含 2^±24 宽动态范围） | `kda_cfbi5_uprobe.py` | 1200（截断拆分）、1500（rna） | 0 |
| A/B 交换形式（交叉项顺序互换）与原形式相同 | 同上，`bad_swap_T` | 1200 | 0 |
| kind::tf32 对 fp32 输入做截断：原始 fp32 操作数 == 低 13 位清零后的操作数 | `kda_cfbi5_tprobe.py` | 1200 | 0 |
| mma.sync m16n8k8 tf32 与 UMMA tf32 在本契约序列下相同 | `kda_cfbi5_syncprobe.py` | 1000 | 0 |

**精度**
- 3×TF32 截断拆分的单个乘积：相对误差中位数 6.0e-8（rna 拆分为 2.6e-8）。
- 端到端对 FP32 recurrent 的最大绝对误差：H6 为 1.22e-4，H3 为 6.1e-5，在 ≲2e-4 以内。

### 成本模型（决定实现选型）

**UMMA tf32 指令的代价**（`kda_cfbi5_mmabench.py`）
- M=128 时，单线程连续发射，每条约 150 周期。
- 这个代价与 N（16–256）和累加器个数无关。
- 因此只有 N=256 能达到 tf32 峰值。N=32 时只有 mma.sync 吞吐（0.467 条/SM/周期）的约一半。

**一次 K=128 的 3×TF32 乘积**
- UMMA 固定为 48 条，约 7.2K 周期，与形状无关。
- mma.sync 的成本按 tile 数线性增长。
- 结论：N ≥ 128 时用 UMMA；Akk/Aqk、小 BV 的 K2 和 decode 单行乘积应改用 mma.sync。两者逐位相同，所以选哪个不改变契约。

### 实现与状态

**文件**（`agent_run/scripts/`）
- `kda_cfbi5.cu`：oracle、naive 的 K1/K2。
- `kda_cfbi5_r5.cu`：当前 prefill。bv=32/64 是吞吐模式（K2r3）；bv=164/232/264 是延迟模式（K2r4/K2r5：TMA 生产者加单线程 UMMA 发射）。
- `kda_cfbi5_dec.cu`：decode 引擎 `rows_fast`。

**gates**
- P_out、P_E、P_pos、P_cache、decode_after_prefill、packing 在 H6/H3 上全部通过，覆盖所有 bv 模式。
- `rows_fast` 与 oracle 在 150 步 decode 和打包 prefill 上逐位相同。

**最终性能**（作业 20087/20088，L=8192，单位 ms）

| 场景 | v4 最优 | FlashKDA | 倍数 |
|---|---|---|---|
| H6 b1 | 1.30（bv232） | 0.348 | 3.7 |
| H6 b8 | 2.30（bv264） | 0.451 | 5.1 |
| H6 b64 | 14.6（bv32；K1 5.34 + K2 9.09） | 2.46 | 5.9 |
| H3 b1 | 1.24 | 0.335 | 3.7 |
| H3 b8 | 1.59 | 0.393 | 4.0 |
| H3 b64 | 7.50 | 1.41 | 5.3 |

decode 引擎（H6，ms/步）：

| N | rows_fast | oracle | Triton recurrent |
|---|---|---|---|
| 1 | 0.0205 | 0.0557 | 0.0021 |
| 8 | 0.0207 | 0.0558 | 0.0037 |
| 64 | 0.0504 | 0.120 | 0.0090 |
| 256 | 0.164 | 0.490 | 0.0342 |

**未达标及原因**
- 目标是 b64 ≲ 5 ms、b1/b8 ≤ 1.5×、decode N=64 ≤ 2×，目前都没有达到。
- 主因是 UMMA tf32 每条指令约 150 周期的固定代价：
  - K2 每个 chunk 72 条 MMA，每个 V-tile 都要重复一遍；
  - K1 的交换形式每个 chunk-head 96 条；
  - decode 每步 96 条，约 7 µs。
- 其次是 K1 的串行 WY：每个 chunk-head 约 16K warp 指令，这是契约规定的。

**下一步**（不改契约）
1. 把 Akk/Aqk、decode 的两个单行乘积和小 BV 的 K2 换成 mma.sync，估计这些乘积的张量流水时间降到原来的 1/3～1/8。
2. K2 改成 BV=128（N=128），UMMA 指令数不再随 V-tile 数重复。

## 约定 v5（2026-10-15 定稿）

v5 = v4 + 分块 WY 与分块 intra 输出。门控、归一化、tf32 乘积定义、P1/P2/P3 和行缓存格式都与 v4 相同。唯一真值是 oracle 行引擎 `rows6_kernel`（`agent_run/scripts/kda_cfbi6.cu`）。prefill 和 decode 引擎都与它逐位相同。

### 与 v4 的改动（已批准）

v4 的 WY 和 intra 输出是长度最多 63 的串行 fmaf 链。v5 把 chunk 分成 4 个 16 行的块，块间改成一次 tf32 乘积，块内仍然串行。

对行 t，记块号 i = ⌊t/16⌋，b0 = 16i。

**块间乘积**
- M 是 K = b0 的 tf32 乘积，按 v4 定义（3×TF32 截断拆分，k 步升序，最后 BIG+SMALL）。
- WY：A 行取 Akk[t, 0:b0]，B 取 X_j（j < b0）的 {ũ, w} 列。
- intra：A 行取 Aqk[t, 0:b0]，B 取 u_j（j < b0）。
- i = 0 时没有这一项。

**WY**
- R^ũ = β·v，R^w = β·Γ·kn。
- Y = R − M（`__fsub_rn`，仅 i > 0）。
- 再对 r = b0 … t−1 升序做 X = fmaf(−Akk[t,r], X_r, X)。

**intra**
- o = P2 + M（`__fadd_rn`，仅 i > 0）。
- 再对 r = b0 … t 升序做 o = fmaf(Aqk[t,r], u_r, o)。

这是块前代（block forward substitution），不显式求对角块的逆。串行链从 ≤63 步缩到 ≤15/16 步。块间乘积写成 tf32 MMA，因此所有引擎可以选不同的指令和不同的 tile 划分，只要每个输出元素的 k 步顺序相同。行/列不变性和 mma.sync ≡ UMMA 在 v4 已实测（见上表，均 0 不同），v5 不需要新的探针。

### 各乘积的指令选型（逐位相同，按大小选）

| 乘积 | prefill | decode |
|---|---|---|
| Akk/Aqk | K1s：mma.sync，单元 =（子块 s，j 的 n-tile），ks/qs 两个 tile 共用 kjr 的 B 片段 | mma.sync，A 只有第 0、1 行（ks、qs），每个 warp 负责 8 个 j |
| WY 块间 M | K1s：mma.sync，B 片段和结果都用 warp shuffle 在列所有者之间搬运，不经过 smem | mma.sync，B 取自 smem 中的行缓存 |
| P1/P2 | 小 BV：K2s mma.sync；大 BV：K2r5 UMMA | mma.sync，单行 A，B 为 E 行 |
| intra 块间 M | K2s：片段所有者直接把 M 加到 P2 片段上 | mma.sync |
| P3 | K2s mma.sync / K2r5 UMMA | mma.sync，每个 chunk 一次 |

UMMA BV=128（让 UMMA 指令数不随 V-tile 重复）没有实现：K2 需要 E、U 的 hi/lo 共约 200KB 的 smem，放不下。

### 门禁与精度（最终，作业 20209 / 20205）

- prefill（`kda_cfbi6_pf4.cu`，bv 316/332/464）：P_out、P_E、P_pos、P_cache、decode_after_prefill、packing 在 H6/H3 全部通过。
- 对 FP32 recurrent 的最大绝对误差：H6 为 1.22e-4，H3 为 6.1e-5，与 v4 相同。
- decode（`kda_cfbi6_dsync8.cu`）：`rows_fast` 与 oracle 在 150 步 decode 上逐位相同，在打包 prefill 上也逐位相同。打包 prefill 用 7 条序列，H6 走整头核（dsync7），H3 走 V 切分核（dsplit），两条路径都覆盖到。

### 各轮记录（v5 阶段，单位 ms；prefill L=8192，decode 为 H6 avg64）

| 轮 | 改动 | H6 b1 | H6 b8 | H6 b64 | dec N=1 | dec N=64 |
|---|---|---|---|---|---|---|
| v4 终版 | — | 1.30 | 2.30 | 14.6 | 0.0205 | 0.0504 |
| 1 | 契约 v5：oracle、K1r3 分块 WY、K2 分块 intra、decode 改 v5 | 1.09 | 2.41 | 16.3 | 0.0187 | 0.0399 |
| 2 | decode 全部乘积改 mma.sync；新增 K2s（mma.sync，小 BV） | 0.676 | 2.60 | 17.5 | 0.0189 | 0.0481 |
| 3 | K1s：Akk/WY 改 mma.sync，去掉 TMEM/sX，2 CTA/SM；decode 行缓存整体进 smem，E 进寄存器 | 0.633 | 2.05 | 13.6 | 0.0121 | 0.0365 |
| 4 | K2s 改 cp.async.bulk + mbarrier，P3 与 intra 分到不同 warp；decode 行缓存改逐行 bulk | 0.535 | 1.97 | 12.7 | 0.0123 | 0.0365 |
| 5 | decode：E 经 bulk 暂存再进寄存器、首 token 输入提前加载；小 batch 走 V 切分核 | — | — | — | 0.0109 | 0.0352 |

第 2 轮 decode 的 16B 向量化 E 载入加 shuffle 转置（dsync4/5）变慢，已放弃。

### 最终性能

prefill（作业 20209，ms）：

| 场景 | v5 最优 | FlashKDA | 倍数 | 目标 |
|---|---|---|---|---|
| H6 b1 | 0.536（bv316；K1 0.10 + K2 0.43） | 0.348 | 1.54 | ≤1.5× |
| H6 b8 | 1.98（bv316；K1 0.66 + K2 1.28） | 0.451 | 4.4 | ≤1.5× |
| H6 b64 | 12.7（bv332；K1 5.22 + K2 7.27） | 2.46 | 5.2 | ≲5 ms |
| H3 b1 | 0.480（bv316） | 0.336 | 1.43 | ≤1.5× |
| H3 b8 | 1.03（bv332） | 0.393 | 2.6 | ≤1.5× |
| H3 b64 | 6.66（bv332） | 1.41 | 4.7 | — |

decode（作业 20205，ms/步，avg64）：

| N | H6 rows_fast | H6 Triton | H3 rows_fast | H3 Triton |
|---|---|---|---|---|
| 1 | 0.0109 | 0.0021 | 0.0110 | 0.0021 |
| 8 | 0.0118 | 0.0037 | 0.0112 | 0.0026 |
| 64 | 0.0352 | 0.0090 | 0.0230 | 0.0049 |
| 256 | 0.127 | 0.0341 | 0.0684 | 0.0159 |

### 未达标及原因

**达标项**
- H3 b1 为 1.43×，达标。
- H6 b1 为 1.54×，与 1.5× 只差一点。

**未达标项**
- b8 和 b64 都没有达到。
- decode 目标是 N=1 ≤ 0.01 ms、N=64 ≤ 0.018 ms，实测分别为 0.0109 和 0.035。

**decode**
- 时钟打点（`kda_cfbi6_dprof.py`）显示每步约 17–20K 周期，由三条串行的乘积链组成：Akk → 块间 M → WY → P1/P2。
- 契约规定每个输出元素的 k 步串行累加。SMALL 累加器每个 K=8 步有两条相互依赖的 mma.sync。
- 实测单条 mma.sync tf32 的依赖延迟约 50 周期，因此一次 K=128 乘积最少约 1.6K 周期，仅链上 MMA 就有约 4K 周期。其余主要是首个 token 的访存延迟，约 4–5K。
- V 切分把每个 CTA 的字节数减半，只省下约 6%，可见瓶颈是延迟而不是带宽。

**K1**
- 主要是 ALU：每个单元都重算 ks/qs/kjr，包括 ex2 和 bf16 解包。WY 初始化也占一部分。b64 时每个 CTA 约 31K 有效周期。

**K2**
- 每个 chunk 的链为：P（K=128）→ U → intra/P3（K=64）→ E。
- b1 时为 4.2K 周期 / chunk，约 3.3 µs。
- b64 的吞吐被 per-CTA 的 WG/KD 重复读取限制：每个 chunk-head 96KB，乘以 V-tile 数。

**下一步（不改契约）**
1. K1 的 Akk 改为每个 warp 固定一个子块 s、复用 A 片段，预计 ALU 减少约 1/3。
2. K2 b64：用 cluster + TMA multicast 共享 WG/KD，读取次数不再随 V-tile 数重复。
3. decode：链上的 MMA 延迟是契约决定的。剩下可做的是把首 token 的访存（posbuf → 行缓存）与 Akk 进一步重叠。

### 文件（`agent_run/scripts/`）

- `kda_cfbi6.cu`：oracle `rows6_kernel`。
- `kda_cfbi6_mma.cuh`：mma.sync 3×TF32 tile。
- `kda_cfbi6_pf4.cu`：最终 prefill。bv316/332 = K1s + K2s<16/32>；464 = K1s + K2r5<64>。
- `kda_cfbi6_pf.cu`、`kda_cfbi6_pf2.cu`、`kda_cfbi6_pf3.cu`：第 1–3 轮。
- `kda_cfbi6_dsync8.cu`：最终 decode。N·H·4 ≤ 148 走 dsplit<32>，否则走 dsync7。
- `kda_cfbi6_dsync*.cu`、`kda_cfbi6_dec.cu`：各轮 decode。
- 脚本：`kda_cfbi6_test.py`、`kda_cfbi6_decode.py`、`kda_cfbi6_dprof.py`、`kda_cfbi6_dncu.py`。

## v7 第 2 轮（GB200 集群 job 19807，GB200，同卡同作业对比第 1 轮）

`kda_cfbi7_pf2.cu` / `kda_cfbi7_dec2.cu` 全部门禁通过（P_out/P_E/P_pos/P_cache、decode 接 prefill、打包、fast 与 rows 150 步逐位）。接口与 v6 相同，可直接替换。

| 指标（ms） | 第 1 轮 | 第 2 轮 | FlashKDA / Triton |
|---|---|---|---|
| prefill H6 b1 | 0.535 | 0.527（1.48×） | 0.355 |
| prefill H6 b8 | 1.94 | 1.88（4.1×） | 0.456 |
| prefill H6 b64 | 12.49 | 11.78（4.75×） | 2.48 |
| prefill H3 b1 | 0.485 | 0.493（1.44×） | 0.342 |
| decode H6 N=1 每步 | 0.0106 | 0.0076 | 0.0021 |
| decode H6 N=8 每步 | 0.0113 | 0.0082 | 0.0038 |
| decode H6 N=64 每步 | 0.0327 | 0.0235 | 0.0094 |
| decode H6 N=256 每步 | 0.118 | 0.084 | 0.035 |

decode 去掉 k 循环内分支后快约 28%，N=1 达到 ≤0.01 ms；N=64 离 0.018 还差 30%。prefill 第 2 轮只小幅改进，b8/b64 仍是主要差距。
