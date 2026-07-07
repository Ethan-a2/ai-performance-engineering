# HTP 与 CUDA 公平对比 Benchmark 设计

> 日期：2026-07-07  
> 目标：设计一个 HTP 版本与 CUDA 版本同 shape、同输入、同精度/误差预算、同端到端边界的公平 benchmark。  
> 参考：`/media/code/llm/llama/llama.cpp/ggml/src/ggml-hexagon` 与 Snapdragon Android 构建流程。

## 结论摘要

- 不再用当前 `htp_minimal` 与 `gpu_minimal` 的教学微基准直接横比；它们的 workload、输入类型、精度、runtime 边界都不同。
- 新 benchmark 应使用 **同一个 ggml graph**，分别跑 **ggml CUDA backend** 和 **ggml Hexagon/HTP backend**，这样 op 图、shape、输入、输出校验和端到端边界都能保持一致。
- 公平版本的 canonical 对比只接受 `status=ok` 且 `fallback_count=0` 的结果；任何 CPU fallback、unsupported op、shape 改写、dtype 替换都必须 `SKIPPED:` 或 `INVALID:`，不能发布为 HTP/CUDA 对比数字。
- v0 建议先做 **decode-step 端到端边界**：权重/graph/缓存初始化和 warmup 不计时；每轮计时包含输入写入目标 backend buffer、graph compute、backend synchronize、输出 sentinel/readback 和误差校验。
- v0 不建议直接跑完整 llama binary 的 tok/s，因为调度、tokenizer、sampling、IO、线程模型和 backend fallback 很难先验对齐；应先用可控 ggml 子图建立公平底座，再扩展到完整模型。

## 为什么现有 minimal 不公平

| 维度 | 当前 HTP minimal | 当前 GPU minimal | 不公平原因 |
| --- | --- | --- | --- |
| 图边界 | FastRPC 内部手写 HVX C loop | PyTorch CUDA eager / library op | 不是同一张计算图 |
| 数据类型 | `uint32_t` / `uint8_t` 为主 | FP16/FP32 tensor 为主 | 精度和误差预算不一致 |
| 输入 | 各自内部生成 | 各自内部生成 | 没有共享 input artifact/hash |
| shape | HTP `elements/repeats` | GPU `batch/m/k/n/tokens/width` | 维度语义不同 |
| runtime 边界 | ADB + host binary + FastRPC | local PyTorch CUDA event | host/device 提交边界不同 |
| fallback 可见性 | skel 内固定实现 | PyTorch 自动调度 | 不能证明同 op 都在目标 backend 上执行 |

## 公平性硬约束

| 约束 | 要求 | 失败处理 |
| --- | --- | --- |
| 同 shape | 每个 case 使用同一个 JSON case spec，维度完全一致 | `INVALID: shape mismatch` |
| 同输入 | 从同一个 seed 生成 FP32 master input，再按 case dtype 量化/转换；记录 SHA256 | `INVALID: input hash mismatch` |
| 同精度 | v0 使用 `f16_weight_f16_activation_f32_output`；CUDA 和 HTP 都必须执行同 dtype graph | `SKIPPED: dtype unsupported` |
| 同误差预算 | 与 CPU F32 reference 比较，逐 case 记录 `max_abs_err/max_rel_err/rmse` | `INVALID: error budget exceeded` |
| 同 E2E 边界 | canonical timed loop 包含 input update、graph compute、synchronize、output readback/sentinel | `INVALID: boundary mismatch` |
| 无 fallback | 所有 compute nodes 必须分配到目标 backend；CPU node count 必须为 0 | `INVALID: backend fallback` |
| 同 warmup/iterations | 同 case spec 指定 warmup、iterations、repeat policy | `INVALID: iteration mismatch` |
| 同输出 | 输出 tensor shape、dtype、hash、误差摘要一致 | `INVALID: output contract mismatch` |

## 选择 ggml 同图双后端

### HTP 侧参考

`llama.cpp` 的 Hexagon backend 已经提供接近目标的后端层：

- 构建入口：`ggml/src/ggml-hexagon/CMakeLists.txt`
- 后端实现：`ggml/src/ggml-hexagon/ggml-hexagon.cpp`
- HTP skel：`libggml-htp-v73.so`、`libggml-htp-v75.so`、`libggml-htp-v79.so`、`libggml-htp-v81.so`
- 支持 op 包括：`MUL_MAT`、`MUL_MAT_ID`、`ADD/MUL/SUB/DIV`、`RMS_NORM`、`SCALE`、`SOFT_MAX`、`ROPE`、`FLASH_ATTN_EXT`、`GET_ROWS/SET_ROWS`、`CPY/CONT`、`GLU` 等。

HTP `MUL_MAT` 关键约束：

- `dst` 必须是 `GGML_TYPE_F32`。
- `src1` 支持 `GGML_TYPE_F32` 或 `GGML_TYPE_F16`。
- `src0` 支持 `F16/F32` 以及部分量化权重类型：`Q4_0/Q4_1/Q8_0/IQ4_NL/MXFP4`。
- 量化权重必须放在 Hexagon repack buffer。
- `src0->ne[0]` 对量化类型需要 32 对齐。
- VTCM 需求超过 session budget 时必须 skip，而不是降级。

### CUDA 侧选择

CUDA 侧也使用 llama.cpp/ggml CUDA backend，而不是 PyTorch minimal：

- 同一个 benchmark binary 构造同一张 ggml graph。
- 后端选择参数为 `--backend cuda` 或 `--backend htp`。
- CUDA 结果作为同图 CUDA 实现，不混入 repo 现有章节里的 Blackwell/Triton/CUTLASS 特化目标。

## Benchmark case 集合

v0 先选择 HTP 和 CUDA 都可支持、且能代表 LLM 热路径的 3 个 case。所有 case 都用同一个 `fair_case_spec.json` 驱动。

| Case | 图边界 | Shape | Precision | 输出 | 目的 |
| --- | --- | --- | --- | --- | --- |
| `mm_decode_f16` | 单个 decode matmul：`Y = W @ X` | `batch=1, seq=1, in=1024, out=1024` | `W=f16, X=f16, Y=f32` | `Y[1024]` | 最小 GEMM 公平底座，验证同 dtype 与 VTCM 支持 |
| `mlp_swiglu_f16` | `down(silu(gate(x)) * up(x))` | `hidden=1024, intermediate=2816, batch=1, seq=1` | weights/activation f16，acc/output f32 | `Y[1024]` | 对齐 LLM MLP 热路径，覆盖 matmul + activation + elementwise |
| `attn_decode_f16_kv512` | 单 token decode attention | `hidden=1024, heads=8, head_dim=128, kv_len=512` | QKV/activation f16，scores/output f32 | `Y[1024]` | 对齐 decode attention + KV cache 边界，覆盖 RoPE/softmax/attention |

v1 扩展项：

| Case | 增加原因 | 前置条件 |
| --- | --- | --- |
| `mm_prefill_f16_s32` | 检查 prefill batch/seq 对吞吐的影响 | HTP `MUL_MAT` batched shape 支持明确 |
| `mm_decode_q8_0` | 对齐端侧量化权重常见路径 | HTP 和 CUDA 使用同一 ggml `Q8_0` 量化 artifact，HTP repack 计入 setup 而非 timed loop |
| `mlp_swiglu_q8_0` | 更接近端侧模型权重 | 量化误差预算单独定义 |
| `full_decode_block_f16` | 端到端 transformer block | 所有 compute nodes 均可无 fallback 跑在 HTP/CUDA |

## Shape 与 dtype 规范

### v0 canonical shape

```json
{
  "suite": "htp_cuda_fair_v0",
  "seed": 20260707,
  "precision": "f16_weight_f16_activation_f32_output",
  "warmup": 10,
  "iterations": 100,
  "cases": [
    {
      "name": "mm_decode_f16",
      "batch": 1,
      "seq": 1,
      "in_features": 1024,
      "out_features": 1024,
      "dtype_weight": "f16",
      "dtype_activation": "f16",
      "dtype_output": "f32",
      "abs_tol": 0.08,
      "rel_tol": 0.05
    },
    {
      "name": "mlp_swiglu_f16",
      "batch": 1,
      "seq": 1,
      "hidden": 1024,
      "intermediate": 2816,
      "dtype_weight": "f16",
      "dtype_activation": "f16",
      "dtype_output": "f32",
      "abs_tol": 0.20,
      "rel_tol": 0.08
    },
    {
      "name": "attn_decode_f16_kv512",
      "batch": 1,
      "seq": 1,
      "hidden": 1024,
      "heads": 8,
      "head_dim": 128,
      "kv_len": 512,
      "dtype_weight": "f16",
      "dtype_activation": "f16",
      "dtype_output": "f32",
      "abs_tol": 0.25,
      "rel_tol": 0.10
    }
  ]
}
```

### 为什么先选 `hidden=1024`

- 足够像 LLM decode 热路径，但不会一开始就把 HTP VTCM、Android memory、ADB artifact 流程压到极限。
- `head_dim=128`、`hidden=1024` 对 HTP/CUDA 都是常见对齐形状。
- `intermediate=2816` 接近 Llama 小模型比例，避免过小 shape 只测 launch 固定成本。
- 后续可以扩展到 `hidden=2048/4096`，但 v0 应先建立无 fallback、公平输入和误差闭环。

## 输入生成与 artifact 契约

### 输入生成

- 使用同一个 host-side generator 生成 FP32 master tensors。
- 推荐使用固定、跨平台可复现的 `xorshift128+` 或 PCG32，不使用 Python/torch 默认 RNG 作为 canonical 输入来源。
- FP32 master 输入范围：`uniform(-0.5, 0.5)`；attention logits 可额外缩放，避免 softmax 饱和。
- 按 case dtype 转换成 f16/量化后写入 artifact。

### Artifact 文件

```text
artifacts/fair_htp_cuda/<RUN_ID>/
  case_spec.json
  inputs/
    mm_decode_f16.weights.f16.bin
    mm_decode_f16.x.f16.bin
    mlp_swiglu_f16.w_gate.f16.bin
    mlp_swiglu_f16.w_up.f16.bin
    mlp_swiglu_f16.w_down.f16.bin
    attn_decode_f16_kv512.qkv_weights.f16.bin
    attn_decode_f16_kv512.k_cache.f16.bin
    attn_decode_f16_kv512.v_cache.f16.bin
  reference/
    <case>.cpu_f32_output.f32.bin
    <case>.reference_metrics.json
  results/
    <case>.cuda.json
    <case>.htp.json
    <case>.comparison.json
  raw/
    <case>.cuda.log
    <case>.htp.log
```

每个 binary artifact 必须记录：

- `path`
- `shape`
- `dtype`
- `numel`
- `sha256`
- `byte_size`
- `generator_seed`

## 端到端边界

### Canonical timed boundary

每个 iteration 计时必须包含：

1. 把本轮输入 activation/token/KV update 写入目标 backend buffer。
2. 执行同一个 ggml graph。
3. 调用 backend synchronize。
4. 读取输出 sentinel 或完整输出 tensor 到 host。
5. 记录本轮 latency。

不计入 canonical timed loop：

- backend 初始化
- graph 构建
- tensor allocation
- 权重上传
- HTP repack / CUDA upload
- warmup
- CPU reference 计算

### Diagnostic boundary

可以额外输出 `resident_compute_ms`，但只能作为诊断字段：

- 所有 tensors 已 resident。
- 每轮只计 graph compute + synchronize。
- 不作为 canonical HTP/CUDA 公平结论。

## 无 fallback 检查

公平对比必须显式证明所有 compute nodes 都在目标 backend 上执行。

### 必需字段

```json
{
  "backend": "htp",
  "case": "mlp_swiglu_f16",
  "graph_nodes_total": 12,
  "graph_compute_nodes": 7,
  "target_backend_nodes": 7,
  "cpu_fallback_nodes": 0,
  "unsupported_nodes": [],
  "fallback_count": 0,
  "status": "ok"
}
```

### 失败规则

- `fallback_count > 0`：结果标记为 `INVALID: backend fallback`。
- HTP VTCM 不足：`SKIPPED: htp vtcm budget exceeded`。
- CUDA capability 不足：`SKIPPED: cuda capability unsupported`。
- dtype 不支持：`SKIPPED: dtype unsupported`。
- graph op 不支持：`SKIPPED: unsupported op <op>`。

## 误差预算

### Reference

- CPU F32 reference 使用同一 ggml graph 或独立 scalar/reference path。
- reference 输出在 setup 阶段生成并保存 artifact。
- CUDA 和 HTP 都与同一个 CPU F32 reference 比较。

### 指标

| 指标 | 含义 |
| --- | --- |
| `max_abs_err` | 最大绝对误差 |
| `max_rel_err` | `abs(err) / max(abs(ref), 1e-6)` 的最大值 |
| `rmse` | 均方根误差 |
| `cosine_similarity` | 对 vector output 的方向一致性 |
| `topk_match` | 仅 full logits case 使用，v0 暂不要求 |

### v0 阈值

| Case | Abs Tol | Rel Tol | 说明 |
| --- | ---: | ---: | --- |
| `mm_decode_f16` | 0.08 | 0.05 | f16 input/weight、f32 output 的单 matmul |
| `mlp_swiglu_f16` | 0.20 | 0.08 | 多 matmul + activation，误差累积更大 |
| `attn_decode_f16_kv512` | 0.25 | 0.10 | softmax/attention 对局部误差更敏感 |

阈值必须在首轮实现后用 CPU/CUDA/HTP 三方输出复核；如果 HTP 使用不同近似函数，需把误差预算写入 case spec，不允许事后放宽而不记录原因。

## 运行产物 JSON 契约

每个 backend/case 输出一个 JSON：

```json
{
  "schema_version": 1,
  "run_id": "2026-07-07_htp_cuda_fair_v0",
  "case": "mm_decode_f16",
  "backend": "cuda",
  "status": "ok",
  "device": {
    "name": "...",
    "arch": "...",
    "driver": "..."
  },
  "shape": {
    "batch": 1,
    "seq": 1,
    "in_features": 1024,
    "out_features": 1024
  },
  "precision": "f16_weight_f16_activation_f32_output",
  "boundary": "input_update+graph_compute+synchronize+output_readback",
  "warmup": 10,
  "iterations": 100,
  "latency_ms": {
    "mean": 0.0,
    "median": 0.0,
    "p95": 0.0,
    "min": 0.0,
    "max": 0.0,
    "raw": []
  },
  "correctness": {
    "reference_sha256": "...",
    "output_sha256": "...",
    "max_abs_err": 0.0,
    "max_rel_err": 0.0,
    "rmse": 0.0,
    "cosine_similarity": 1.0,
    "abs_tol": 0.08,
    "rel_tol": 0.05,
    "passed": true
  },
  "backend_assignment": {
    "graph_compute_nodes": 1,
    "target_backend_nodes": 1,
    "cpu_fallback_nodes": 0,
    "unsupported_nodes": [],
    "fallback_count": 0
  },
  "artifacts": {
    "case_spec_sha256": "...",
    "input_hashes": {},
    "log": "raw/mm_decode_f16.cuda.log"
  }
}
```

Comparison JSON：

```json
{
  "case": "mm_decode_f16",
  "status": "ok",
  "cuda_mean_ms": 0.0,
  "htp_mean_ms": 0.0,
  "ratio_htp_over_cuda": 0.0,
  "ratio_cuda_over_htp": 0.0,
  "both_correct": true,
  "same_shape": true,
  "same_precision": true,
  "same_boundary": true,
  "both_no_fallback": true
}
```

## 构建流程

### Snapdragon / HTP build

用户给出的参考流程作为 HTP build baseline：

```bash
cd /media/code/llm/llama/llama.cpp
cmake --preset arm64-android-snapdragon-release -B build-snapdragon -DGGML_OPENCL=OFF
cmake --build build-snapdragon -j16
cmake --install build-snapdragon --prefix pkg-snapdragon/llama.cpp
```

需要确认 install 包含：

- benchmark binary，例如 `llama-fair-htp-cuda-bench` 或同等自定义 binary
- `libggml-hexagon.so`
- `libggml-htp-v73.so`
- `libggml-htp-v75.so`
- `libggml-htp-v79.so`
- `libggml-htp-v81.so`
- 所需 `libggml*`、`libllama*` shared libraries

### CUDA build

建议在同一 llama.cpp tree 构建 CUDA backend binary：

```bash
cd /media/code/llm/llama/llama.cpp
cmake -B build-cuda -DGGML_CUDA=ON -DGGML_OPENCL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-cuda -j16
```

CUDA binary 和 HTP binary 应由同一 benchmark 源文件构建，仅 backend init 与 platform packaging 不同。

## 运行流程

### 1. 生成 shared inputs/reference

```bash
python tools/fair_htp_cuda/generate_cases.py \
  --spec configs/fair_htp_cuda_v0.json \
  --out artifacts/fair_htp_cuda/2026-07-07_v0
```

### 2. CUDA 运行

```bash
/media/code/llm/llama/llama.cpp/build-cuda/bin/llama-fair-htp-cuda-bench \
  --backend cuda \
  --spec artifacts/fair_htp_cuda/2026-07-07_v0/case_spec.json \
  --inputs artifacts/fair_htp_cuda/2026-07-07_v0/inputs \
  --reference artifacts/fair_htp_cuda/2026-07-07_v0/reference \
  --out artifacts/fair_htp_cuda/2026-07-07_v0/results \
  --fail-on-fallback
```

### 3. HTP 运行

```bash
adb shell 'mkdir -p /data/local/tmp/fair_htp_cuda_v0'
adb push /media/code/llm/llama/llama.cpp/pkg-snapdragon/llama.cpp /data/local/tmp/fair_htp_cuda_v0/
adb push artifacts/fair_htp_cuda/2026-07-07_v0 /data/local/tmp/fair_htp_cuda_v0/artifacts

adb shell 'cd /data/local/tmp/fair_htp_cuda_v0/llama.cpp && \
  export LD_LIBRARY_PATH=$PWD/lib:$LD_LIBRARY_PATH && \
  export ADSP_LIBRARY_PATH="$PWD/lib;/dsp;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp" && \
  export DSP_LIBRARY_PATH="$PWD/lib;/dsp;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp" && \
  ./bin/llama-fair-htp-cuda-bench \
    --backend htp \
    --spec ../artifacts/case_spec.json \
    --inputs ../artifacts/inputs \
    --reference ../artifacts/reference \
    --out ../artifacts/results \
    --fail-on-fallback'

adb pull /data/local/tmp/fair_htp_cuda_v0/artifacts/results artifacts/fair_htp_cuda/2026-07-07_v0/results_htp
```

### 4. 汇总比较

```bash
python tools/fair_htp_cuda/compare_results.py \
  --cuda artifacts/fair_htp_cuda/2026-07-07_v0/results \
  --htp artifacts/fair_htp_cuda/2026-07-07_v0/results_htp \
  --out artifacts/fair_htp_cuda/2026-07-07_v0/summary.json
```

## 实现建议

### Benchmark binary

建议在 llama.cpp tree 增加一个小型 C++ binary，而不是在本 repo 重写 ggml backend 调度：

```text
/media/code/llm/llama/llama.cpp/examples/fair-htp-cuda-bench/
  CMakeLists.txt
  fair-htp-cuda-bench.cpp
```

职责：

1. 解析 `case_spec.json`。
2. 加载 shared binary inputs。
3. 初始化 backend：`cuda` 或 `hexagon`。
4. 为目标 backend 分配 tensor buffer。
5. 构建同一 ggml graph。
6. 强制检查所有 compute node 是否由目标 backend 支持。
7. 执行 warmup。
8. 执行 canonical timed loop。
9. 与 CPU F32 reference 对比。
10. 输出 JSON。

### Python wrapper

本 repo 可以只放 orchestrator/wrapper：

```text
tools/fair_htp_cuda/
  generate_cases.py
  compare_results.py
  run_cuda.sh
  run_htp_adb.sh
```

职责：生成 artifacts、调用 llama.cpp binary、拉取 Android 结果、做汇总。

## 验收标准

v0 只有同时满足下面条件才算公平 benchmark 可用：

| Gate | 要求 |
| --- | --- |
| 构建 | CUDA binary 和 Snapdragon binary 均来自同一 benchmark 源文件 |
| 输入 | `case_spec.json` 和所有 input artifact hash 在 CUDA/HTP 结果中一致 |
| dtype | 每个 tensor dtype 与 spec 一致；没有 backend 私自替换 dtype |
| fallback | CUDA/HTP 两边 `fallback_count=0` |
| correctness | 两边均通过 CPU F32 reference 误差预算 |
| boundary | 两边 JSON 的 `boundary` 字段完全一致 |
| repeatability | 至少 3 轮 run，mean/p50/p95 均输出；异常值不手动删除 |
| diagnostics | HTP 输出 DSP arch、HVX/HMX/VTCM 信息；CUDA 输出 GPU name、SM、driver/runtime 信息 |

## 首轮落地顺序

1. 在 llama.cpp 添加 `fair-htp-cuda-bench` binary，只实现 `mm_decode_f16`。
2. 实现 shared input/reference artifact 生成。
3. 跑 CUDA，确认 JSON、误差、backend assignment。
4. 用用户给出的 Snapdragon build 流程构建并推送。
5. 跑 HTP，先处理 `SKIPPED:` 和 unsupported op，不做 fallback。
6. 加入 `mlp_swiglu_f16`。
7. 加入 `attn_decode_f16_kv512`。
8. 只有三类 case 都无 fallback 且误差通过后，再考虑量化和 full decode block。

## 与现有章节文档的关系

- 这套 benchmark 不替代现有 `chXX/compare_htp_minimal.py`；后者仍是教学/覆盖入口。
- 这套 benchmark 也不替代各章默认 CUDA canonical 数字；它是一个新的 **HTP-vs-CUDA fair pair**，用来回答“同 shape/输入/精度/边界下，HTP 与 CUDA 的实际差异”。
- 如果后续把结果写入章节对比文档，必须明确标注为 `fair_htp_cuda_v0`，不能混入 `htp_minimal` 或 `gpu_minimal` speedup 表。

