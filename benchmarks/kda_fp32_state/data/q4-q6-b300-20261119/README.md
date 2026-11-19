# B300：Q4/Q6 FP32 state、FlashInfer 与 vLLM CF-BI

[English](README.en.md) · [性能入口](../../README.md)

B=64、H=12/16/32、D=128；Q4/Q6 对应 MTP3/MTP5。
双方直接索引状态池，计入全部 Q 个 checkpoint、设备端接受数选择及原地状态提交。
我们的状态/checkpoint 为 **FP32**，FlashInfer 为 **BF16**；CF-BI 的入口矩阵和行缓存为 FP32。
输入 Q/K/V/gate/beta 和输出均为 BF16，计算契约不要求跨实现逐位相同。

## 环境与源码

B300 SXM6 AC，双 GPU NV18，单 GPU 串行测量；driver 615.71.10，Python 3.12.3，
PyTorch 2.14.0a0+b2c75dd062.nv26.09，CUDA 13.4，CuTe DSL 4.8.1。
默认时钟，未锁频。详见 [environment.json](environment.json)。

| 实现 | 源码 |
|---|---|
| FP32 state | 可配置 Q 的候选实现，基于 [e652140](https://github.com/NVIDIA-JerryChen/kda-consistent/commit/e6521408691ca3b95e47a138670966f4192ab2ef)；实测文件 SHA256 见 [sources.json](sources.json) |
| FlashInfer | [flashinfer-ai/flashinfer @ c3af0c1](https://github.com/flashinfer-ai/flashinfer/tree/c3af0c13328fd23b256c53b9b4f94d9e160d118c)，原生 grouped Q4/Q6 |
| CF-BI C16/C64 | [Inferact/zero-mismatch-rl @ 5735778](https://github.com/Inferact/zero-mismatch-rl/tree/5735778f43f7d94108d432706dc5b17765bf723a)，`vllm/kimi-k3.1` 分支，C16=v6、C64=v5 |

CF-BI 使用该版本的 [公开 decode wrapper](https://github.com/Inferact/zero-mismatch-rl/blob/5735778f43f7d94108d432706dc5b17765bf723a/vllm/models/kimi_k3/nvidia/k31_kda_cfbi.py)
及未修改的 CUDA 实现。编译选项为 `-O3 --fmad=false -UNDEBUG -lineinfo`、`sm_103a`。
FlashInfer 保留原始 kernel 和编译函数，移除无关导入与分派；物理提交使用与本实现相同、保留 dtype 的 copy helper。
参考源码与文件哈希见 [sources.json](sources.json) 和 [upstream-provenance.json](upstream-provenance.json)。
本轮不包含 FlashKDA PR #5，也不测量框架、卷积或通信。

## 单次调用性能

42 个 case：Q4/Q6 × H12/16/32 × 7 组 KV 长度清单；每行是 7 个 case 延迟中位数的几何平均，单位 µs。
接受数为 `request_index % (Q + 1)`，覆盖接受 0～Q 个 token。

| Q | H | FP32 state | FlashInfer | CF-BI C16 适配 | CF-BI C64 适配 | FP32 / FlashInfer |
|---|---:|---:|---:|---:|---:|---:|
| 4 | 12 | 61.406 | 51.241 | 511.581 | 772.122 | 1.198× |
| 4 | 16 | 77.842 | 71.873 | 588.161 | 978.350 | 1.083× |
| 4 | 32 | 143.419 | 135.437 | 987.657 | 1803.975 | 1.059× |
| 6 | 12 | 77.833 | 75.781 | 712.728 | 1025.625 | 1.027× |
| 6 | 16 | 100.361 | 98.020 | 814.433 | 1299.032 | 1.024× |
| 6 | 32 | 188.120 | 185.391 | 1360.947 | 2385.270 | 1.015× |

跨 H 的几何平均：Q4 为 **88.175 / 79.306 / 667.327 / 1108.673 µs**，
Q6 为 **113.690 / 111.255 / 924.429 / 1470.219 µs**（依次为表中四个实现）。
我们的 FP32 state 相对 FlashInfer 延迟分别增加 **11.2% / 2.2%**。
[逐 case CSV](main.csv) · [全部原始样本](main.json) · [汇总](summary.json)。

**CF-BI 此版本没有原生多 token checkpoint API。** 本表的适配路径调用 Q 次公开 Q1，
每 token 保存完整入口矩阵、行缓存和位置，另存原始状态供接受 0 个时恢复，最后按接受数选择并物理写回。
复制整个缓存容量（含无效容量），wrapper 的布局转换也计时。因此这些数字是满足相同功能义务的
**完整适配耗时，不是原生 CF-BI Q4/Q6 kernel 延迟**；不据此宣称相同倍率的纯 kernel 加速。

## Chunk 位置与完整周期

384 个位置 case 覆盖 Q4/Q6、三个 H 和起始位置 0～63，包括跨 chunk 边界。
[位置 CSV](positions.csv) / [原始样本](positions.json) 保留每个位置的数据。

完整周期全部接受 Q 个 token：Q4 连续 16 次调用推进 64 token，Q6 连续 32 次推进 192 token，
覆盖 64 行契约的完整可达周期。下表为周期总耗时除以调用数后的中位数，单位 µs/调用。
每次调用仍输出全部 checkpoint 并提交状态。

| Q | H | FP32 state | FlashInfer | CF-BI C16 适配 | CF-BI C64 适配 |
|---|---:|---:|---:|---:|---:|
| 4 | 12 | 57.788 | 42.625 | 441.499 | 749.472 |
| 4 | 16 | 76.854 | 64.989 | 512.826 | 963.301 |
| 4 | 32 | 147.260 | 132.801 | 930.783 | 1783.552 |
| 6 | 12 | 74.526 | 65.377 | 610.192 | 997.681 |
| 6 | 16 | 98.784 | 89.293 | 704.048 | 1281.007 |
| 6 | 32 | 190.432 | 182.782 | 1273.071 | 2360.430 |

该口径的跨 H 延迟比几何平均为 Q4 **1.211×**、Q6 **1.095×**（FP32 / FlashInfer）。
接受数、Graph 内调用数及状态推进方式与单次调用表不同，结果分开报告；每 token 耗时为表值除以 Q。
[周期 CSV](cycles.csv) · [原始样本](cycles.json)。

## 数学精度与一致性

独立 FP64 递推参考；B=2、H=3、D=128，normal/open/saturated gate，非零 BF16 可表示初态。
总长度 64，先逐 token 执行 64−Q 个历史 token，再执行 Q4/Q6；比较最后 Q 个输出及终态。
CF-BI 在闭合 chunk 比较入口矩阵，与此时的逻辑递推状态相同。
每个实现 6 个 case，下面列各指标的实测最大值，不是误差上限。
NRMSE=RMS(error)/RMS(reference)，最大归一化误差=max(abs(error))/max(abs(reference))。

| 实现 | 输出 NRMSE | 输出最大归一化误差 | 状态 NRMSE | 状态最大归一化误差 |
|---|---:|---:|---:|---:|
| FP32 state | 1.688e-3 | 3.436e-3 | 2.033e-7 | 4.899e-7 |
| FlashInfer | 1.229e-2 | 1.695e-2 | 1.228e-2 | 3.049e-2 |
| CF-BI C16 | 1.688e-3 | 3.436e-3 | 3.725e-7 | 8.054e-7 |
| CF-BI C64 | 1.688e-3 | 3.436e-3 | 4.550e-7 | 8.014e-7 |

FlashInfer 的历史 Q1 每步写回 BF16，误差包含历史状态舍入的影响，**不能解释为单次 Q4/Q6 自身的误差**。
全部 [24 条诊断记录](accuracy.json) 均为有限值。

本轮检查 eager/Graph 输出、完整状态池及所有 checkpoint；FP32 state、CF-BI 的每个 checkpoint
与各自独立 Q1 前缀逐位比较，FlashInfer 则与自身紧凑索引 grouped 路径比较，不宣称其 BF16 Q1/Q多 token 逐位一致。
固定 Graph 下更新设备 slot 和接受数；未选 slot 保持原值，接受 0 个保留初态。
本实现代码与此前通过 586 项可配置 Q 验证的源码相同；本轮不是完整 vLLM 集成或新增多卡验收。
见 [validation.json](validation.json)。

## Workload 与计时边界

- [workloads.json](workloads.json) 沿用已确认 Q8 的七组 B64 KV 长度数组，不重抽样；
  KV 上限 8K/16K/32K/64K/128K/256K/512K，改为 Q4/Q6，入口位置为 `KV length − Q`。
- 历史由状态表示：在闭合 64-token 边界提供共同非零 anchor，各实现真实执行剩余 `context % 64`
  个历史 token 生成入口缓存和位置；未重放整个 8K～512K 历史，不用仅修改位置的伪造缓存。
- 状态池容量 129，64 个唯一逆序奇数 slot，原地写回。输入是合成值；没有按 case 特化 kernel。
- Hot-address CUDA Graph，图内 event 包围一次完整 prepared GPU API；状态 reset 在计时区间外，因而预热了状态。
  轮换实现顺序，每组每实现排队 8 次再同步读取事件。单次调用表丢弃 16 个 warmup，保留 80 个样本，取中位数。
- 位置表丢弃 4 个 warmup、保留 12 个样本；周期表丢弃 2 个 warmup 周期、保留 10 个周期。
- 包含 decode、全部 checkpoint、选择和状态提交及 CF-BI 适配所需 GPU 工作；不包含编译、分配、Graph capture、
  历史准备、接受决策与 metadata 生成、CPU 调度、框架通信。不是 cold-L2 或端到端模型性能。

## 复现

需要 CUDA 13.x 编译工具链和两张可见 B300 SXM6 AC，实际只在 GPU0 计时。
在已安装 CUDA 版 PyTorch 的仓库 checkout 中执行；`prepare.py` 下载固定 commit 的 CF-BI 文件并核验哈希。
`fi_q1.py` / `fi_q8.py` 保留 FlashInfer 的 [Apache-2.0](../../../../vllm/models/kimi_k3/nvidia/kda_fp32_state/licenses/FlashInfer-Apache-2.0.txt) 声明。

```bash
python -m pip install -r vllm/models/kimi_k3/nvidia/kda_fp32_state/requirements.txt pytest triton ninja
export KDA_BENCH_ROOT="$PWD/q46-reproduction"
python benchmarks/kda_fp32_state/data/q4-q6-b300-20261119/prepare.py "$KDA_BENCH_ROOT"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$KDA_BENCH_ROOT/source"
export RESULT_DIR="$KDA_BENCH_ROOT/results"
export TORCH_EXTENSIONS_DIR="$KDA_BENCH_ROOT/cache/torch"
export TRITON_CACHE_DIR="$KDA_BENCH_ROOT/cache/triton"
export CCACHE_DIR="$KDA_BENCH_ROOT/cache/ccache"
export SOURCE_DIGEST="$(cat "$KDA_BENCH_ROOT/source-digest.txt")"
python "$KDA_BENCH_ROOT/source/verify.py"
python "$KDA_BENCH_ROOT/source/benchmark.py" smoke
python "$KDA_BENCH_ROOT/source/benchmark.py" main
python "$KDA_BENCH_ROOT/source/benchmark.py" phases
python "$KDA_BENCH_ROOT/source/verify.py"
```

准备脚本从本分支复制 FP32 模块，在独立快照内命名为 `kda_consistent`，供冻结 runner 导入。
本表测量发生在模块加入 vLLM 分支之前；原测量源码哈希保留在 [sources.json](sources.json)。
Kernel 文件映射和本分支的独立回归验收见 [性能入口](../../README.md)。复现快照的模块路径和接口封装不同，整体摘要不相同。
