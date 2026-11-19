# FP32 state 训推一致 KDA kernel

[English](README.md)

面向 K3/K3.1 的 KDA 前向 kernel，输入/输出 BF16，续算状态 FP32。
Prefill 与 decode 执行同一固定顺序的逐 token FP32 程序：Q/K 归一化、gate、预测与 delta、
状态更新、输出投影。Prefill 并行生成系数，状态递推保持顺序；decode 支持任意正整数固定 Q，
包括 MTP3/MTP5/MTP7 对应的 Q4/Q6/Q8，可选输出每 token checkpoint。

同 GPU 型号、同软件栈、同契约内，prefill/decode 切分、batch 组合/重排、完整 head 分片的
有效输出、完整续算状态及 checkpoint 逐位一致。本实现与 CF-BI 是不同浮点契约；
训练端仍用 CF-BI 时，不能仅替换推理端。训推对齐要求两端采用相同契约。

本模块提供 kernel 和 prepared API，不改动模型路由、卷积、调度/cache 管理或通信。
不包含 backward，也不保证完整模型 logits。支持 UP 所需的完整 head 分片；不支持 Ring CP。

## 依赖

SM100/SM103（B200/B300）、CUDA 版 PyTorch、CuTe DSL 4.8.1。

```bash
python -m pip install -r vllm/models/kimi_k3/nvidia/kda_fp32_state/requirements.txt
```

## 示例

```python
import torch
from vllm.models.kimi_k3.nvidia.kda_fp32_state import prepare_decode

B, Q, H, D = 2, 4, 12, 128
q, k, v, g = [torch.randn(B, Q, H, D, device="cuda", dtype=torch.bfloat16)
               for _ in range(4)]
beta = torch.randn(B, Q, H, device="cuda", dtype=torch.bfloat16)
A_log = torch.zeros(H, device="cuda")
dt_bias = torch.zeros(H, D, device="cuda")
state = torch.zeros(B, H, D, D, device="cuda")
out = torch.empty_like(q)
checkpoints = torch.empty(B, Q, H, D, D, device="cuda")
accepted = torch.tensor([0, 3], device="cuda", dtype=torch.int32)
plan = prepare_decode(
    q, k, v, g, beta, A_log, dt_bias, state, out, state,
    checkpoint_states=checkpoints, num_accepted_tokens=accepted,
)
plan.run()
# Update accepted after the acceptance decision, then commit on the same stream.
plan.commit()
```

## API 契约

导出 `prepare_prefill`、`prepare_decode`、`PrefillPlan`、`DecodePlan`。
两个 prepare 的前几个参数均为 `q,k,v,g,beta,A_log,dt_bias,state_in,out`，随后为 state_out。
`state_dtype=torch.float32`，不接受其他状态 dtype。所有 buffer 须位于同一 CUDA 设备、连续，
非空时 32 字节对齐。状态轴为 **[V,K]**，D=128、H>0。

| 参数 | Prefill | Decode | dtype |
|---|---|---|---|
| q/k/v/g/out | dense `[B,L,H,D]` 或 packed `[1,T,H,D]` | `[B,Q,H,D]`，Q≥1 | BF16 |
| beta | 对应前三维 | `[B,Q,H]` | BF16 |
| A_log / dt_bias | `[H]` / `[H,D]` | 相同 | FP32 |
| state_in/state_out | `[N,H,D,D]` 或索引池 `[P,H,D,D]` | `[B,H,D,D]` 或索引池 `[P,H,D,D]` | FP32 |
| cu_seqlens | packed 设备端 `[N+1]`；dense 省略 | 不使用 | Int32 |
| state_indices | 可选 `[N]` | 可选 `[B]` | Int32 |
| checkpoint_states | 不支持 | Q>1 可选 `[B,Q,H,D,D]`；Q1 禁用 | FP32 |
| num_accepted_tokens | 不支持 | Q>1 commit 可选 `[B]`，取值 0..Q | Int32 |

默认 `scale=128**-0.5`、`eps=1e-6`、`lower_bound=-5.0`。
内部完成 Q/K 归一化及 gate 激活，g/beta 为激活前输入。标量须为有限 FP32 值，eps 为正，
lower_bound 位于 [-5,0]。中间系数与状态不舍入 BF16；输出舍入 BF16，初等函数使用 FP32 近似实现。

Prefill/Q1 必须提供 state_out，允许输入/输出状态池完全相同。
Q>1 不传 `checkpoint_states` 时必须提供 `state_out`，`run()` 输出全部 token 结果和最终状态，
允许输入/输出状态池完全相同；此模式不支持接受数选择或回退。
传入 `checkpoint_states` 时，Q>1 `run()` 不修改状态池，返回 `(out, checkpoint_states[:, -1])`；末态 view 可能不连续，
再次作为输入前须复制为连续 buffer。提供 state_out 和 num_accepted_tokens 后可调用 `commit()`：
a>0 复制 checkpoint a−1，a=0 保留/复制原始状态；返回目标池，也可通过 `accepted_state` 取得。

同次调用 slot 须唯一且在 `[0,P)`，`N≤P<2³¹`；未选 slot 保持不变。
**Slot 0 是普通 slot，不是 padding 标记。** 接入方须移除 padding/NULL_SLOT 行，或给其分配有效唯一的
scratch slot。禁止部分 buffer 重叠以及 checkpoint 与状态池重叠。

Packed offsets 从 0 开始、单调不减、末值不超过 T；无效输出尾部不改动。允许空请求/空 batch。
Prefill 最多 1023 请求，另受 Int32/grid 容量限制。Decode 各请求 Q 相同，Q 和 B×Q 均须小于 2³¹。
Metadata 内容保持设备端，由调用方保证有效，prepare 不读回 host。
历史 KV 仅由入口状态表示，非连续序列片段须使用各自正确的初态。

## Plan 与 CUDA Graph

Prepare 在 capture 外校验、编译并绑定地址/容量。Prefill 的 `workspace_limit_bytes` 默认 256 MiB，
decode 默认 64 MiB；本 FP32 decode 不需要内部 workspace。Prefill 空间不足时报错。
Run/commit 不分配设备内存、不编译，使用绑定设备的当前 stream。

Plan、buffer、workspace 须存活至 GPU 完成。Graph replay 可原地更新输入值、slot、offset 和接受数；
地址/shape/标量变化须重新 prepare。Run 与 commit 之间须保留 slot 身份、原始状态和 checkpoint。
不同调用/stream 访问相同 slot 时须同步，不得并发复用 plan/workspace。
原地 commit 后从已提交状态继续；非原地 commit 须把目标池绑定为下一次输入。

Prefill 的 B/H/长度为 runtime，不作为编译实例分派依据。Decode 可按 Q 选择实例，不能按精确 batch、
历史长度、设备指针或 metadata 内容特化。Host 可见的非法 shape/type/dtype/alignment/alias 或标量抛出
TypeError/ValueError；capture 内 prepare，或 autograd 开启且输入 requires-grad 时抛出 RuntimeError。
有效输入须有限，kernel 运行时不扫描 NaN/Inf。

## 测试和性能

```bash
python -m pip install pytest triton
python -m pytest -q tests/kernels/kda_fp32_state
```

完整 head 分片与 PD 续传检查需要两张同型号 GPU，单卡时跳过这些检查。
[测试](../../../../../tests/kernels/kda_fp32_state/README.md) 比较 prefill、Q1/可配置 Q、任意切分、
batch/重排、varlen、checkpoint 续算、动态 metadata、Graph 和编译缓存复用。
[性能数据](../../../../../benchmarks/kda_fp32_state/README.md) 包含 FlashInfer 和 vLLM CF-BI C16/C64 对照，注明精度与计时范围。

## 许可证

[MIT](LICENSE)，Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES。
FlashInfer 派生部分保留 [Apache-2.0](licenses/FlashInfer-Apache-2.0.txt) 与
[来源声明](licenses/FlashInfer-NOTICE.txt)。
