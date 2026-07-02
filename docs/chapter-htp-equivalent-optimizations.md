# 各章节 HTP 等效/类似优化方法（Snapdragon Hexagon / llama.cpp ggml-hexagon）

更新时间：2026-07-02  
参考实现：`/media/code/llm/llama/llama.cpp/ggml/src/ggml-hexagon`  
适用范围：把 `code/ch01` 到 `code/ch20` 的 GPU/CPU 优化思想迁移到 Snapdragon Hexagon HTP（HVX/HMX/VTCM/DMA）时，可采用的等效或近似方法。

## 目录

- [核心结论](#核心结论)
- [HTP 能力模型](#htp-能力模型)
- [构建与运行参考](#构建与运行参考)
- [本仓库最小 HTP 实现](#本仓库最小-htp-实现)
- [GPU/CPU 到 HTP 的概念映射](#gpucpu-到-htp-的概念映射)
- [章节总览](#章节总览)
- [Chapter 1 - Performance Fundamentals](#chapter-1---performance-fundamentals)
- [Chapter 2 - GPU Hardware Architecture](#chapter-2---gpu-hardware-architecture)
- [Chapter 3 - System Tuning](#chapter-3---system-tuning)
- [Chapter 4 - Distributed Communication & Multi-GPU Distribution](#chapter-4---distributed-communication--multi-gpu-distribution)
- [Chapter 5 - Storage and IO Optimization](#chapter-5---storage-and-io-optimization)
- [Chapter 6 - CUDA Programming Fundamentals](#chapter-6---cuda-programming-fundamentals)
- [Chapter 7 - Memory Access Patterns](#chapter-7---memory-access-patterns)
- [Chapter 8 - Occupancy, Warp Efficiency & ILP](#chapter-8---occupancy-warp-efficiency--ilp)
- [Chapter 9 - Arithmetic Intensity & Kernel Fusion](#chapter-9---arithmetic-intensity--kernel-fusion)
- [Chapter 10 - Tensor Core Pipelines & Cluster Features](#chapter-10---tensor-core-pipelines--cluster-features)
- [Chapter 11 - Streams & Concurrency](#chapter-11---streams--concurrency)
- [Chapter 12 - CUDA Graphs & Dynamic Workloads](#chapter-12---cuda-graphs--dynamic-workloads)
- [Chapter 13 - PyTorch Profiling & Memory Tuning](#chapter-13---pytorch-profiling--memory-tuning)
- [Chapter 14 - Compiler & Triton Optimization](#chapter-14---compiler--triton-optimization)
- [Chapter 15 - Disaggregated Inference & KV Management](#chapter-15---disaggregated-inference--kv-management)
- [Chapter 16 - Production Inference Optimization](#chapter-16---production-inference-optimization)
- [Chapter 17 - Disaggregated Prefill/Decode & Routing](#chapter-17---disaggregated-prefilldecode--routing)
- [Chapter 18 - Advanced Attention & Decoding](#chapter-18---advanced-attention--decoding)
- [Chapter 19 - Dynamic & Adaptive Inference Precision/Memory Systems](#chapter-19---dynamic--adaptive-inference-precisionmemory-systems)
- [Chapter 20 - AI-Assisted Performance Optimization & Case Studies](#chapter-20---ai-assisted-performance-optimization--case-studies)
- [验证矩阵](#验证矩阵)

## 核心结论

CUDA/GPU 章节里的优化，大多可以在 HTP 上找到“同构思想”，但不能逐字平移 API：

- Tensor Core 对应 HMX；SIMT/vector memory 对应 HVX；shared memory / TMA staging 对应 VTCM + DMA；CUDA streams/graphs 对应 HTP op batching、dspqueue、worker pool、HMX queue 和 fused graph partition。
- GPU 的 kernel fusion、低精度、tiling、prefetch、overlap、routing、KV cache 管理，在 HTP 上主要落到 `ggml-hexagon` 的 supported ops、op fusion、HMX/HVX matmul selector、FlashAttention selector、VTCM scratchpad、DMA queue、host buffer 和多 HTP device 分片。
- HTP 的首要边界不是“所有算子都能上 HTP”，而是 supported op、buffer/session compatibility、VTCM 容量、shape/layout、量化类型和 HMX eligibility。unsupported op 必须留在 CPU/其他 backend，不能静默解释为 HTP 性能。

## HTP 能力模型

`ggml-hexagon` 由 Android/host 侧 `libggml-hexagon.so` 和 DSP/HTP 侧 skel `libggml-htp-v73/v75/v79/v81.so` 组成。参考实现里可观察到以下能力：

| 组件 | 类比 GPU/CPU 概念 | 参考实现线索 |
| --- | --- | --- |
| HVX | SIMD/vector lanes、标量/向量 kernel | `hvx-*.h`、`hvx-mm-kernels-*`、unary/binary/softmax/rope 等 ops |
| HMX | Tensor Core / matrix engine | `hmx-mm-kernels-tiled.h`、`hmx-fa-kernels.h`、`GGML_HEXAGON_USE_HMX` |
| VTCM | shared memory / scratchpad | `vtcm-utils.h`、`htp_spad`、FlashAttention VTCM tile buffers |
| DMA queue | async copy / TMA-like staging | `hex-dma.c`、`dma_queue`、trace event `DMA` |
| dspqueue | GPU command queue / stream submit | `dspqueue_t queue`、`GGML_HEXAGON_OPBATCH`、`GGML_HEXAGON_OPQUEUE` |
| worker pool | CPU/GPU thread block worker coordination | `worker-pool.c`、FlashAttention store/softmax parallel sections |
| HMX queue | async matrix engine queue | `hmx-queue.c`，用于 pipeline overlap |
| op fusion | kernel fusion / graph rewrite | `HTP_OP_MUL_MAT_QKV`、`HTP_OP_MUL_MAT_FFN`、`HTP_OP_MUL_MAT_ADD`、`HTP_OP_RMS_NORM_MUL` |
| profiling | Nsight/PMU/trace 的 HTP 对照 | `GGML_HEXAGON_PROFILE`，basic/PMU/trace；trace event 包含 DMA/HVX/HMX/FA 阶段 |

支持算子包括 `MUL_MAT`、`MUL_MAT_ID`、QKV/FFN fused matmul、binary/unary、RMSNorm、Softmax、RoPE、FlashAttention、set/get rows、copy、argsort、SSM conv、cumsum、concat、pad、triangular solve、gated delta net 等。数据类型包含 `F32`、`F16`、`Q4_0`、`Q4_1`、`Q8_0`、`IQ4_NL`、`MXFP4`，以及 tiled repack 内部类型。

## 构建与运行参考

### 环境变量

```bash
echo "$ANDROID_NDK_ROOT"
# /opt/Android/Ndk/android-ndk-r28c

echo "$HEXAGON_SDK_ROOT"
# /opt/qcom/Hexagon_SDK/6.6.0.0

echo "$HEXAGON_TOOLS_ROOT"
# /opt/qcom/Hexagon_SDK/6.6.0.0/tools/HEXAGON_Tools/19.0.07
```

### 编译

```bash
cd /media/code/llm/llama/llama.cpp/
cmake --preset arm64-android-snapdragon-release -B build-snapdragon -DGGML_OPENCL=OFF
cmake --build build-snapdragon -j16
cmake --install build-snapdragon --prefix pkg-snapdragon/llama.cpp
```

该 preset 会启用 `GGML_HEXAGON=ON`，并构建多代 HTP skel：`libggml-htp-v73.so`、`libggml-htp-v75.so`、`libggml-htp-v79.so`、`libggml-htp-v81.so`。

### 运行

```bash
M=functiongemma-270m-it-BF16.gguf D=HTP0 \
  ./scripts/snapdragon/adb/run-completion.sh \
  -p "what is the most popular cookie in the world?" --verbose
```

脚本默认会在设备端设置：

- `LD_LIBRARY_PATH` / `ADSP_LIBRARY_PATH` 指向安装包 lib 目录。
- `--no-mmap`，避免 mmap 与设备 buffer/session 管理冲突。
- `--ctx-size 8192 --ubatch-size 1024 -fa on -ngl 99 --device HTP0`。
- `-t 6 --cpu-mask 0xfc --cpu-strict 1`，固定 CPU 线程亲和性，减少 host 噪声。

### HTP 常用调优旋钮

| 旋钮 | 作用 |
| --- | --- |
| `D=HTP0` / `--device HTP0` | 选择 HTP backend device |
| `NDEV=4` / `GGML_HEXAGON_NDEV=4` | 多 HTP device/session |
| `NHVX=<n>` / `GGML_HEXAGON_NHVX` | HVX worker/thread 数 |
| `HMX=0/1` / `GGML_HEXAGON_USE_HMX` | 是否启用 HMX matrix engine |
| `MM=1/2/3` / `GGML_HEXAGON_MM_SELECT` | Matmul fallback 顺序：flat、tiled、HMX |
| `FA=0/1/2` / `GGML_HEXAGON_FA_SELECT` | FlashAttention fallback 顺序：CPU、HVX、HMX |
| `OB=<n>` / `GGML_HEXAGON_OPBATCH` | 每 batch 最大 op 数 |
| `OQ=<n>` / `GGML_HEXAGON_OPQUEUE` | pending op batch queue 深度 |
| `OP=<0/1>` / `GGML_HEXAGON_OPPOLL` | batch completion polling 策略 |
| `OC=0/1` / `GGML_HEXAGON_OPFUSION` | op fusion 开关 |
| `OF=<regex>` / `GGML_HEXAGON_OPFILTER` | 拒绝某些 op 上 HTP，用于消融/定位 |
| `HB=0/1` / `GGML_HEXAGON_HOSTBUF` | host buffer 路径 |
| `VM=<bytes>` / `GGML_HEXAGON_VMEM` | buffer mapping 虚拟地址上限 |
| `MB=<bytes>` / `GGML_HEXAGON_MBUF` | 最大 buffer 大小 |
| `PROF=1/2/3` / `GGML_HEXAGON_PROFILE` | basic / PMU / trace profiling |
| `V=1` / `GGML_HEXAGON_VERBOSE` | verbose op support/execution 日志 |

## 本仓库最小 HTP 实现

每章现在都有一组最小 HTP benchmark wrapper：

| 文件 | 作用 |
| --- | --- |
| `code/chXX/baseline_htp_minimal.py` | 调用 llama.cpp Snapdragon wrapper，使用 HVX/minimal baseline knobs：`HMX=0`、`MM=1`、`FA=1`、`OC=0`、`OB=1`、`OQ=1` |
| `code/chXX/optimized_htp_minimal.py` | 调用同一 prompt/model，使用 HMX/fused optimized knobs：`HMX=1`、`MM=3`、`FA=2`、`OC=1`、`OB=1024`、`OQ=16` |
| `code/chXX/compare_htp_minimal.py` | 直接运行 baseline/optimized 并打印 wall-clock speedup |
| `code/core/benchmark/htp_minimal.py` | 共享 runner、capability checks、ADB 调用、metrics parsing、strict verification payload |

运行方式：

```bash
cd /media/code/tools/ai-performance-engineering/code

# 直接跑某章最小 HTP 对比
AISP_HTP_N_PREDICT=4 python -m ch02.compare_htp_minimal

# 通过 benchmark CLI 跑同一目标，生成 manifest/report
AISP_HTP_N_PREDICT=4 \
  python -m cli.aisp bench run --targets ch02:htp_minimal --profile minimal --iterations 1 --warmup 0
```

可覆盖的环境变量：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `AISP_HTP_LLAMA_CPP_ROOT` | `/media/code/llm/llama/llama.cpp` | llama.cpp checkout 位置 |
| `AISP_HTP_MODEL` | `functiongemma-270m-it-BF16.gguf` | 设备端 `/data/local/tmp/gguf/` 下模型名 |
| `AISP_HTP_DEVICE` | `HTP0` | llama.cpp `--device` |
| `AISP_HTP_PROMPT` | `what is the most popular cookie in the world?` | completion prompt |
| `AISP_HTP_N_PREDICT` | `16` | 生成 token 数；smoke 建议 `4`，正式对比建议更大 |
| `AISP_HTP_TIMEOUT_SECONDS` | `120` | 单次 ADB wrapper 超时 |
| `AISP_HTP_PROFILE` | `1` | 映射到 `GGML_HEXAGON_PROFILE` |
| `AISP_HTP_VERBOSE` | `1` | 映射到 `GGML_HEXAGON_VERBOSE` |

能力门控：如果 `adb`、llama.cpp checkout、`scripts/snapdragon/adb/run-completion.sh` 或 `pkg-snapdragon/llama.cpp` 缺失，目标会显式返回 `SKIPPED:`；如果设备/模型不可用，运行阶段会失败或跳过，并在输出 tail 中保留 ADB/llama.cpp 诊断。

## GPU/CPU 到 HTP 的概念映射

| 原章节概念 | GPU/CPU 实现 | HTP 等效或类似实现 |
| --- | --- | --- |
| Tensor Core GEMM | CUDA WMMA/tcgen05/CUTLASS/cuBLASLt | HMX tiled matmul，`MM=3`，tiled repack `Q4_0_TILED`/`MXFP4_TILED` |
| SIMT elementwise | CUDA thread/block | HVX vector op，binary/unary/activation kernels |
| Shared memory tiling | `__shared__`、TMA staging | VTCM scratchpad + DMA queue + per-op `htp_spad` |
| Stream overlap | CUDA streams/events | dspqueue op batches、HMX queue、DMA/HVX/HMX trace overlap |
| CUDA Graph replay | graph capture/replay | ggml graph partition + HTP op batching/fusion，减少 host round trip |
| Kernel fusion | fused CUDA/Triton kernels | HTP fused opcodes：QKV、FFN、matmul+add、RMSNorm+mul |
| Quantization | FP8/FP4/NVFP4/INT8 kernels | Q4/Q8/IQ4_NL/MXFP4 data types，weight repack/tiled layout |
| Multi-GPU | NCCL/NVLink/NVSHMEM | 多 HTP device (`NDEV`) + host scheduler；不等价于 NCCL，但可类比多 accelerator 分片 |
| CPU affinity | NUMA/pinned host threads | Android CPU mask/strict affinity、host buffer、ADB run wrapper |
| Nsight profiling | NCU/NSYS/PMU | `GGML_HEXAGON_PROFILE=1/2/3`，HTP trace event/PMU counters |

## 章节总览

| Chapter | HTP 等效优化主线 |
| --- | --- |
| `ch01` | 建立 HTP baseline/optimized：CPU-only vs HTP offload、HVX vs HMX、fusion on/off |
| `ch02` | 识别 HTP arch/skel/device，测 HVX/HMX/VTCM/DMA/host buffer 能力 |
| `ch03` | Android host 线程亲和性、CPU mask、ubatch、host buffer 与提交队列调优 |
| `ch04` | 多 HTP device/session、op queue、host-side sharding，类比多 GPU 通信 |
| `ch05` | GGUF/ADB/storage staging、`--no-mmap`、host buffer、模型加载与预处理优化 |
| `ch06` | HVX 自定义算子、binary/unary/activation、shape/support gating |
| `ch07` | VTCM/DMA/tiled repack/contiguous layout，类比 coalescing/TMA/shared memory |
| `ch08` | HVX worker 数、HMX eligibility、unroll/vector lanes、VTCM 容量限制 |
| `ch09` | HTP op fusion、QKV/FFN/RMSNorm+mul、MXFP4/Q4/Q8 降低 bandwidth |
| `ch10` | HMX tiled matmul、FlashAttention HMX path、DMA/VTCM pipeline |
| `ch11` | dspqueue batching、HMX queue、opqueue/oppoll，类比 streams/concurrency |
| `ch12` | graph partition + op batching/fusion，类比 CUDA Graph/dynamic scheduling |
| `ch13` | HTP profiling、PMU/trace、opfilter 消融，类比 PyTorch profiler workflow |
| `ch14` | 编译/IDL/skel 生成、HTP-specific kernel authoring，类比 compiler/Triton |
| `ch15` | KV/get_rows/set_rows/RoPE/FlashAttention offload，类比 KV management/inference |
| `ch16` | HTP serving runtime knobs：ubatch、FA selector、scheduler、tail latency |
| `ch17` | prefill/decode 在 CPU/HTP/多 HTP 间路由，按 TTFT/TPOT 消融策略 |
| `ch18` | FlashAttention、RoPE、paged/KV ops、EOS/polling 类 decode 优化 |
| `ch19` | 动态量化/低精度 cache、MXFP4/Q4/Q8、prefetch/双缓冲 |
| `ch20` | 自动搜索 HTP env knobs 和 op fusion/selector 组合，端到端验证 |

## Chapter 1 - Performance Fundamentals

### HTP 等效方法

- 建立三层 baseline：CPU-only、HTP HVX-only、HTP HMX-enabled。用 `HMX=0` 与默认 `HMX=1` 区分 vector engine 和 matrix engine 收益。
- 对照 `cpu_minimal` 的“Python loop vs library call”，HTP 版本应比较“CPU ggml op vs HTP offloaded op”。
- 用 `OC=0/1` 比较 op fusion 是否改善 prompt/decode latency；用 `MM=1/2/3` 比较 flat HVX、tiled HVX、HMX matmul。
- 对低精度 MLP，优先比较 GGUF 量化类型和 HTP supported type，例如 `Q4_0`、`Q8_0`、`MXFP4`、`F16/BF16` 的吞吐和内存占用。

### 建议命令

```bash
# CPU/HTP 总体对照
M=functiongemma-270m-it-BF16.gguf D=HTP0 V=1 PROF=1 ./scripts/snapdragon/adb/run-completion.sh -p "hello" -n 32

# HMX 消融：HMX off vs on
M=functiongemma-270m-it-BF16.gguf D=HTP0 HMX=0 PROF=1 ./scripts/snapdragon/adb/run-completion.sh -p "hello" -n 32
M=functiongemma-270m-it-BF16.gguf D=HTP0 HMX=1 PROF=1 ./scripts/snapdragon/adb/run-completion.sh -p "hello" -n 32

# Fusion 消融
M=functiongemma-270m-it-BF16.gguf D=HTP0 OC=0 PROF=1 ./scripts/snapdragon/adb/run-completion.sh -p "hello" -n 32
M=functiongemma-270m-it-BF16.gguf D=HTP0 OC=1 PROF=1 ./scripts/snapdragon/adb/run-completion.sh -p "hello" -n 32
```

## Chapter 2 - GPU Hardware Architecture

### HTP 等效方法

- CUDA device capability 对应 HTP arch/skel capability：确认实际加载的是 `v73/v75/v79/v81` skel，必要时用 `GGML_HEXAGON_ARCH` 覆盖或日志确认 autodetect。
- GPU memory/fabric ceiling 对应 HTP VTCM、DMA、host buffer、DDR mapping ceiling；重点观察 `VTCM-TOO-SMALL`、buffer mapping、`GGML_HEXAGON_VMEM`、`GGML_HEXAGON_MBUF`。
- cuBLAS tuning 对应 `MM=1/2/3` matmul selector：flat HVX、tiled HVX、HMX fallback 顺序。
- NVLink/PCIe 拓扑意识对应 Android SoC 内 CPU/HTP/DDR 数据路径：减少 host bounce、固定 CPU affinity、使用 host buffer 或 HTP buffer。

### 验证重点

- `V=1` 看 op 是否被 HTP 支持与执行。
- `PROF=2` 看 PMU；`PROF=3` 看 DMA/HVX/HMX/FlashAttention trace 阶段。
- `MM=1/2/3` 应能区分 matmul path，不支持 shape/type 时不能把 fallback 当 HMX 结果。

## Chapter 3 - System Tuning

### HTP 等效方法

- NUMA pinning 对应 Android CPU affinity：运行脚本中的 `-t 6 --cpu-mask 0xfc --cpu-strict 1` 固定 host 线程，减少调度抖动。
- pinned prefetch 对应 host buffer 与 `--no-mmap` 下的显式 buffer 路径；评估 `HB=0/1` 对拷贝和映射的影响。
- double-buffered batch provisioning 对应增大/调节 `--ubatch-size`、`OB`、`OQ`，让 host 提交、DMA、HVX/HMX compute 尽量重叠。
- rack/container prep 对应 Android device thermal/power state、ADB 环境、`LD_LIBRARY_PATH`/`ADSP_LIBRARY_PATH`、模型路径和 skel 安装一致性。

### 消融建议

| 消融 | 观察 |
| --- | --- |
| `HB=0` vs `HB=1` | host buffer 是否减少拷贝/映射开销 |
| `--ubatch-size 256/512/1024` | 小 ubatch 是否提交过多，大 ubatch 是否触发 VTCM/latency 问题 |
| `OB=128/512/1024` | op batch 太小 host overhead 增大，太大可能排队延迟升高 |
| `OQ=1/4/16` | queue 深度对 overlap/tail latency 的影响 |

## Chapter 4 - Distributed Communication & Multi-GPU Distribution

### HTP 等效方法

- 多 GPU/NCCL 没有直接等价物；HTP 可用多 device/session (`NDEV`、`D=HTP0,HTP1,...`) 做近似分片，但通信通常由 host scheduler/ggml backend 管理。
- gradient fusion 类似于 HTP op batching/fusion：减少 host-DSP 往返和小 op 数量。
- NVSHMEM device-driven communication 类似于 HTP 侧 worker pool/HMX queue 推进计算，但不是跨 accelerator RDMA。
- symmetric memory 类似于同一 session 的 supported buffer requirement：所有 src/dst 必须映射到同一 session，buffer type/session 不匹配会拒绝上 HTP。

### HTP 迁移策略

- 先让单 HTP (`D=HTP0`) 稳定，再尝试 `NDEV>1`。
- 对多 HTP，优先切大矩阵/层/专家，不要切太细 token 小 op，否则 host scheduling overhead 会吞噬收益。
- 用 `GGML_SCHED_DEBUG=2` 和 `V=1` 查看 graph partition 是否频繁跨 CPU/HTP 迁移。

## Chapter 5 - Storage and IO Optimization

### HTP 等效方法

- GPUDirect Storage 没有直接等价；Android HTP 重点是 GGUF 推送、加载、buffer mapping 和避免 mmap 造成不可控路径。
- `--no-mmap` 是当前脚本默认选择，便于 backend 管理 buffer；可以作为 HTP canonical run 的固定条件。
- DataLoader/vectorized preprocessing 对应 tokenizer、prompt preparation、GGUF quantized model loading 和 CPU 侧预处理批量化。
- decompression/remote storage 对应模型文件拷贝、ADB push、设备本地存储吞吐，避免运行时边读边阻塞 HTP。

### 验证重点

- 分开测 cold load、prompt prefill、decode tokens/s，不要把模型加载 IO 误判为 HTP compute 性能。
- 记录模型位置、GGUF 类型、`--no-mmap`、`--ctx-size`、`--ubatch-size`。

## Chapter 6 - CUDA Programming Fundamentals

### HTP 等效方法

- CUDA custom kernel 对应新增 HTP op：在 `htp/*.c` 实现 op，在 `htp-ops.h` 加 opcode，在 `ggml-hexagon.cpp` 加 supports/dispatch。
- CUDA thread/block mapping 对应 HVX vector lanes + worker pool 分块；不要按 CUDA warp 思维逐字迁移。
- launch bounds 对应 `NHVX`、worker pool 粒度、VTCM scratchpad 大小和 HMX eligibility。
- bank conflict 对应 VTCM layout/alignment/stride；重点检查 tiled layout 和 per-thread scratchpad。

### 可类比目标

| CUDA 章节方法 | HTP 做法 |
| --- | --- |
| elementwise kernel | binary/unary HVX op |
| activation ILP | Silu/Gelu/Sigmoid HVX vector kernels |
| launch tuning | `NHVX`、`OB`、`OQ`、worker chunk size |
| shared memory tuning | VTCM `htp_spad`、alignment、stride |

## Chapter 7 - Memory Access Patterns

### HTP 等效方法

- coalesced/vectorized load 对应 HVX aligned vector load/store，避免非连续/奇怪 stride 让 op unsupported 或退回 CPU。
- shared-memory tiling/TMA 对应 VTCM tile + DMA queue；对 matmul/FlashAttention 重点看 tiled repack 和 VTCM buffer。
- async prefetch 对应 DMA staging 与 HMX queue overlap；用 `PROF=3` 看 `DMA`、`HMX_COMP`、`HVX_*` trace 是否重叠。
- lookup/gather 对应 `GET_ROWS`、`SET_ROWS`、`ADD_ID`、`MUL_MAT_ID`；应关注 index pattern、batch size 和 cache locality。

### HTP layout 建议

- 优先 contiguous 或 HTP supported stride；`CONT` 只支持部分同类型路径。
- 对 weight 使用 HTP repack/tiled buffer 类型，减少 runtime transpose/dequant。
- 对 KV/cache 类张量，避免 CPU/HTP 反复搬运小块。

## Chapter 8 - Occupancy, Warp Efficiency & ILP

### HTP 等效方法

- GPU occupancy 对应 HVX worker occupancy、HMX tile occupancy、VTCM 容量占用。
- warp divergence 对应 HVX lane mask/branch divergence；尽量用 vector predication/lookup table，而不是 per-element branch。
- loop unrolling 对应手写 HVX intrinsics 中展开 inner loop，但要控制 instruction cache 和 register pressure。
- tcgen05/custom vs cuBLAS 对应 HMX vs HVX tiled vs CPU fallback：`MM=3/2/1`。

### 消融命令

```bash
# HVX worker 数量
M=functiongemma-270m-it-BF16.gguf D=HTP0 NHVX=1 PROF=1 ./scripts/snapdragon/adb/run-completion.sh -p "hello" -n 32
M=functiongemma-270m-it-BF16.gguf D=HTP0 NHVX=4 PROF=1 ./scripts/snapdragon/adb/run-completion.sh -p "hello" -n 32

# Matmul path 选择
M=functiongemma-270m-it-BF16.gguf D=HTP0 MM=1 PROF=1 ./scripts/snapdragon/adb/run-completion.sh -p "hello" -n 32
M=functiongemma-270m-it-BF16.gguf D=HTP0 MM=2 PROF=1 ./scripts/snapdragon/adb/run-completion.sh -p "hello" -n 32
M=functiongemma-270m-it-BF16.gguf D=HTP0 MM=3 PROF=1 ./scripts/snapdragon/adb/run-completion.sh -p "hello" -n 32
```

## Chapter 9 - Arithmetic Intensity & Kernel Fusion

### HTP 等效方法

- arithmetic intensity 对应让更多算子留在 HTP、减少 DDR 往返、使用 tiled/HMX matmul。
- fused L2Norm/RMSNorm/activation 对应 HTP fused op：`RMS_NORM_MUL`、activation ops、GLU/SwiGLU/GELU。
- QKV/FFN fusion 对应 `HTP_OP_MUL_MAT_QKV`、`HTP_OP_MUL_MAT_FFN`；减少中间 tensor 和 host scheduling。
- matmul+add fusion 对应 `HTP_OP_MUL_MAT_ADD`。
- FP4/FP8/NVFP4 类低精度在 HTP 上近似为 Q4/Q8/IQ4_NL/MXFP4 与 tiled repack。

### 消融建议

- `OC=0` 禁用 fusion，比较 prompt/decode latency 和 op 数。
- `OF=.*RMS.*` 或针对某些 op filter，强制 fallback，验证某个 op 是否真贡献收益。
- 比较 `Q4_0`、`Q8_0`、`MXFP4`、`F16/BF16` 模型文件，观察 memory bandwidth 与数值质量。

## Chapter 10 - Tensor Core Pipelines & Cluster Features

### HTP 等效方法

- Tensor Core pipeline 对应 HMX tiled matmul pipeline；启用 `HMX=1` 和 `MM=3`。
- TMA pipeline 对应 DMA + VTCM staging；trace 中应能看到 DMA 和 compute 阶段。
- persistent kernel 对应 HTP op batching + HMX queue：减少 host round trip，让 DSP 侧持续消费 op batch。
- warp specialization 对应 HMX/HVX 分工：HMX 做 matrix core，HVX 做 quant/dequant、softmax、output processing。
- cluster/DSMEM 没有直接等价；最接近的是 VTCM scratchpad + worker pool 协作。
- FlashAttention TMA/pipeline 对应 `FLASH_ATTN_EXT` 的 HMX/HVX selector：`FA=2` HMX->HVX->CPU，`FA=1` HVX->CPU。

### 验证重点

- `PROF=3` 看 `HMX_COMP`、`HVX_FA_QK`、`HVX_FA_SFM`、`DMA`。
- `FA=0/1/2` 做 FlashAttention backend 消融。
- VTCM 不够时可能出现 `VTCM-TOO-SMALL`，需要调 shape、ubatch 或 fallback，不应冒充 HMX 成绩。

## Chapter 11 - Streams & Concurrency

### HTP 等效方法

- CUDA streams 对应 HTP dspqueue + op batch queue；`OB` 控制每批 op 数，`OQ` 控制 pending batch 深度。
- stream events/sync 对应 dspqueue completion；`OP=1` polling 可能降低延迟但增加 host active wait。
- stream-ordered KV update 对应 `SET_ROWS`/`GET_ROWS`/KV ops 的 graph dependency，不要用全局 CPU fallback 打断 pipeline。
- multistream warp specialization 对应 HMX queue 与 worker pool 在 matmul/FA 阶段的分工。

### 调优建议

- 小模型/短 prompt：降低 `OB` 和 `OQ` 可能改善 latency。
- 长 prompt/prefill：增大 `OB` 和 `OQ` 有机会提高吞吐。
- 用 `GGML_SCHED_DEBUG=2` 看 graph 是否被过度切碎。

## Chapter 12 - CUDA Graphs & Dynamic Workloads

### HTP 等效方法

- CUDA Graph capture/replay 对应 ggml graph 被 scheduler 切到 HTP backend，并通过 op batching/fusion 减少 repeated host submit。
- conditional graph 对应 supported op + fallback partition；动态 shape 不一定能完整上 HTP，需记录 fallback 边界。
- GPU-resident work queue 对应 HTP dspqueue/worker pool，但调度粒度是 backend op batch，不是 CUDA device launch。
- uneven partition 对应 prefill/decode 不同阶段的 CPU/HTP 切分和 ubatch 大小。

### HTP graph 消融

| 操作 | 目的 |
| --- | --- |
| `OC=0` | 去掉 graph-level op fusion |
| `OB=1` | 模拟无 batching/高 host submit overhead |
| `OF=<op regex>` | 强制某类 op 回 CPU，观察 partition 断裂成本 |
| `SCHED=1` | 打开 scheduler debug，确认实际 HTP placement |

## Chapter 13 - PyTorch Profiling & Memory Tuning

### HTP 等效方法

- PyTorch profiler 对应 `GGML_HEXAGON_PROFILE`：`1` basic、`2` PMU、`3` trace。
- memory profiling 对应 buffer mapping、host buffer、VTCM usage、DDR scratchpad、max vmem/mbuf。
- FP8/FP4 quantization 对应 GGUF quantized types 和 HTP internal tiled repack。
- DataLoader/KV-cache profiling 对应 tokenizer/model load/prompt processing 与 `GET_ROWS`/`SET_ROWS`/RoPE/FA ops 的阶段拆分。

### 推荐 profiling 流程

1. `V=1 SCHED=1`：确认 op placement 与 fallback。
2. `PROF=1`：得到 per-op latency 粗粒度结果。
3. `PROF=2`：采 PMU counters，看 HVX/HMX 是否饱和。
4. `PROF=3`：看 DMA/HVX/HMX/FA trace，定位 overlap 或同步问题。
5. `OF=<regex>`：做单 op 消融，证明瓶颈归因。

## Chapter 14 - Compiler & Triton Optimization

### HTP 等效方法

- Triton/CUDA kernel authoring 对应 HTP skel C/HVX intrinsic/HMX kernel authoring。
- `torch.compile` graph break 修复对应 ggml graph 支持算子和 buffer compatibility 修复：让更多 op 被 `supports_op` 接住。
- persistent Triton 对应 HTP op batching + HMX queue。
- CUTLASS template tuning 对应 HMX tiled kernel、weight repack、matmul eligibility 规则。
- compiler artifact 对应 IDL stub/skel、`build_htp_skel(v73/v75/v79/v81)`、Android package install。

### 开发 checklist

- 新 op 先定义 `HTP_OP_*` 和 supported shape/type。
- 实现 HVX baseline，再考虑 HMX/tiled path。
- 加 verbose support dump，unsupported 要显式返回，不要静默 fallback 后报告 HTP speedup。
- 用 `OPSTAGE` 或 `HTP_OPFLAGS_SKIP_COMPUTE` 类机制做 queue/compute 分离 profiling。

## Chapter 15 - Disaggregated Inference & KV Management

### HTP 等效方法

- KV cache management 对应 `GET_ROWS`、`SET_ROWS`、RoPE、FlashAttention 与 contiguous/paged-like layout 管理。
- prefill/decode disaggregation 对应阶段性 placement：prefill 更偏 HMX/HVX bulk compute，decode 更受 small batch、KV fetch、scheduler latency 影响。
- continuous batching 对应 `--ubatch-size`、op batching、prompt/decode mixed workload 的 host scheduler。
- MoE dispatch 对应 `MUL_MAT_ID`、`ADD_ID`、`GET_ROWS` 这类 indexed/expert-like 操作；重点避免专家选择导致碎片化 fallback。
- allreduce+rmsnorm fusion 在单 HTP 上没有 NCCL 等价；可类比为 RMSNorm/activation/mul fusion，减少中间 tensor。

### HTP 迁移注意

- decode 小 batch 容易被 host submit 和 unsupported small ops 主导。
- KV layout 必须围绕 HTP supported strides 和 VTCM/DDR traffic 设计。
- 用 TTFT/TPOT 分开报告，避免 prefill 优化掩盖 decode 回退。

## Chapter 16 - Production Inference Optimization

### HTP 等效方法

- Flash SDP/block sparse 对应 HTP `FLASH_ATTN_EXT` 和 `FA` selector；稀疏/特殊 mask 必须确认 supported。
- piece graphs/regional compilation 对应只让稳定子图上 HTP，动态外壳留在 CPU scheduler。
- runtime scheduler 对应 ubatch、opqueue、CPU mask、device selection、prompt/decode routing。
- telemetry hooks 对应收集 tokens/s、TTFT、TPOT、HTP per-op profile、CPU fallback ratio。

### 生产建议

- 固定模型、prompt length、ctx、ubatch、CPU affinity、thermal state 后再比较 HTP knobs。
- tail latency 更应关注 `OQ`、`OP`、small op fallback，而不是只看 HMX matmul 峰值。
- 对用户可见指标，必须分开报告 prefill tok/s 和 decode tok/s。

## Chapter 17 - Disaggregated Prefill/Decode & Routing

### HTP 等效方法

- dynamic routing 对应 CPU scheduler 在 CPU/HTP/多 HTP device 间选择 placement。
- topology-aware routing 对应选择 `HTP0`、多 device、host buffer 与模型/层分片，避免跨 session buffer 不兼容。
- prefill/decode TTFT/TPOT 策略：prefill 优先 HMX bulk；decode 可试更小 ubatch、polling、减少 fallback。
- static routing 是消融基线：固定所有请求 `D=HTP0`，再比较动态策略。

### 路由消融

| 策略 | 对照 |
| --- | --- |
| `D=HTP0` | 单 HTP baseline |
| `NDEV=2/4` | 多 HTP device 分片 |
| `FA=0/1/2` | attention backend 路由 |
| `MM=1/2/3` | matmul backend 路由 |
| `OF=<decode op>` | decode 某 op fallback 的影响 |

## Chapter 18 - Advanced Attention & Decoding

### HTP 等效方法

- FlexAttention/FlexDecoding 对应 HTP `FLASH_ATTN_EXT`、softmax、RoPE、mask/slopes 处理；复杂 mask 不一定 supported。
- paged attention layout 对应 KV tensor layout、`GET_ROWS`/`SET_ROWS`、contiguous/repack buffer。
- RoPE Q cache 对应 `ROPE` op 上 HTP，并避免每 token CPU 侧重复处理。
- CUDA graph bucketing 对应按 prompt/decode shape 选择稳定 ubatch 和 HTP graph partition。
- EOS early exit/polling 对应 host decode loop 减少无效 HTP submit，并调 `OP` completion polling。

### 验证重点

- `FA=2` 是否真的走 HMX FA；否则可能走 HVX 或 CPU。
- 对 attention 报告 QK、softmax、O store、DMA trace，而不是只报总 tokens/s。
- EOS early exit 要证明输出一致，不能因为提前停止而少生成。

## Chapter 19 - Dynamic & Adaptive Inference Precision/Memory Systems

### HTP 等效方法

- dynamic precision 对应不同 GGUF quantized model 或 runtime op type；HTP 支持 `Q4_0/Q4_1/Q8_0/IQ4_NL/MXFP4/F16/F32` 等。
- dynamic quantized cache 对应 KV/cache 张量是否能用低精度/packed layout 并保持 supported op。
- FP4 hardware kernel 对应 `MXFP4`、`Q4_*` tiled HTP path，而不是 CUDA NVFP4 原样复制。
- KV prefetch overlap 对应 DMA/VTCM staging 与 HMX/HVX trace overlap。
- memory double buffering 对应 VTCM ping-pong buffer 或 HMX queue pipeline。
- adaptive allocator 对应 `VM`、`MB`、host buffer、DDR scratchpad 和 VTCM capacity 管理。

### 消融建议

- 同一 prompt 比较 `Q4_0`、`Q8_0`、`BF16/F16`、`MXFP4` 模型。
- `MM=2` vs `MM=3` 区分 tiled HVX 与 HMX 对低精度 matmul 的收益。
- `PROF=3` 验证 dequant/prepare/output processing 是否成为瓶颈。

## Chapter 20 - AI-Assisted Performance Optimization & Case Studies

### HTP 等效方法

- autotuning 对应自动搜索 HTP env knob：`HMX`、`MM`、`FA`、`NHVX`、`OB`、`OQ`、`OP`、`OC`、`HB`、`VM`、`MB`。
- AI kernel generator 对应生成 HVX/HMX op 实现，但必须接入 `supports_op`、shape/type guard、profiling 与 correctness。
- end-to-end bandwidth 对应从 model load、prompt prefill、decode、KV cache 到 output 的全链路拆分。
- integrated KV cache 对应把 `GET_ROWS/SET_ROWS/ROPE/FLASH_ATTN_EXT` 合成稳定 HTP partition。
- proof/verification workflow 对应：CPU reference、same prompt deterministic decode、per-op profile、fallback ledger。

### 推荐自动搜索矩阵

| 维度 | 候选 |
| --- | --- |
| Matmul | `MM=1/2/3` |
| FlashAttention | `FA=0/1/2` |
| Fusion | `OC=0/1` |
| Queue | `OB=128/512/1024`、`OQ=1/4/16` |
| Workers | `NHVX=1/2/4/all` |
| Host buffer | `HB=0/1` |
| Model type | `Q4_0/Q8_0/MXFP4/F16/BF16` |

## 验证矩阵

### 基础可运行

```bash
cd /media/code/llm/llama/llama.cpp/
M=functiongemma-270m-it-BF16.gguf D=HTP0 V=1 \
  ./scripts/snapdragon/adb/run-completion.sh \
  -p "what is the most popular cookie in the world?" --verbose
```

通过标准：设备端能加载 `libggml-hexagon.so` 和对应 `libggml-htp-v*.so`，命令完成生成，verbose 日志能看到 HTP device/op placement。

### Profiling 与归因

```bash
# per-op/basic profile
M=functiongemma-270m-it-BF16.gguf D=HTP0 V=1 PROF=1 ./scripts/snapdragon/adb/run-completion.sh -p "hello" -n 32

# PMU profile
M=functiongemma-270m-it-BF16.gguf D=HTP0 V=1 PROF=2 ./scripts/snapdragon/adb/run-completion.sh -p "hello" -n 32

# trace profile
M=functiongemma-270m-it-BF16.gguf D=HTP0 V=1 PROF=3 ./scripts/snapdragon/adb/run-completion.sh -p "hello" -n 32
```

通过标准：能分辨 DMA、HVX、HMX、FlashAttention 子阶段；对每个优化结论都能说明收益来自 fewer ops、better matmul path、less fallback、lower memory traffic、better queue overlap 中的哪一个。

### 消融清单

| 问题 | 命令/旋钮 | 通过标准 |
| --- | --- | --- |
| HMX 是否有效 | `HMX=0` vs `HMX=1`，或 `MM=2` vs `MM=3` | HMX path 在 supported shape/type 上改善 prefill/matmul latency |
| Fusion 是否有效 | `OC=0` vs `OC=1` | op 数减少或中间 tensor/latency 降低 |
| FlashAttention 是否上 HTP | `FA=0/1/2` + `PROF=3` | trace 出现 HVX/HMX FA 阶段，且输出一致 |
| Queue 深度是否合适 | `OB/OQ` sweep | throughput/latency 有稳定改善，tail 不恶化 |
| 某 op 是否拖慢 | `OF=<regex>` | fallback 后性能变化能解释该 op 的贡献 |
| 多 HTP 是否有收益 | `NDEV=1/2/4` | 吞吐提升大于 host scheduling overhead |
| 量化是否值得 | 不同 GGUF 类型 | tokens/s、memory、质量三者有明确 tradeoff |

### 不能混淆的结论

- CPU fallback 的结果不能叫 HTP speedup。
- HMX disabled 的结果不能叫 HMX matmul 性能。
- 模型加载/ADB/storage 时间不能混入 decode tokens/s。
- unsupported op 的 fallback 需要在日志或报告中显式记录。
- `GGML_HEXAGON_PROFILE` 数据是 HTP backend profile，不等价于 Nsight，但可以承担同类归因职责。
