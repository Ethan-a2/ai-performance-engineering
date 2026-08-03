# GPU 算子与内存优化实战路线：从 Adreno OpenCL / Vulkan 到 CUDA 专家

> 目标：以“先跑、再测、再改、再解释”为主线，让没有 GPU 算子经验的开发者尽快形成可迁移到 CUDA、OpenCL、Vulkan Compute、Triton、TileLang 和大模型推理算子库的工程能力。  
> 资料与环境核验日期：**2026-08-02**。  
> 当前工作目录：`/media/code/tools/ai-performance-engineering`。  
> 当前手机实测：Xiaomi `2512BPNDAC`、Android 16 / API 36、Qualcomm `SM8850`、Adreno 840。  
> 当前主机限制：未安装或未检测到 `nvidia-smi`、`nvcc`，因此 CUDA 路线需要 NVIDIA 工作站、远程服务器或云 GPU；Adreno 路线可以立即开始。

---

## 1. 先说结论：最快成长路径

不要按照“先看完整本 CUDA 书，再开始写 kernel”的方式学习。最快的路线是：

1. **第一天就跑通一个真实 GPU baseline。**
2. **第二天只改一个变量，并用计时与正确性数据证明效果。**
3. **第一周掌握内存访问、launch 开销、融合与计时。**
4. **第二到第四周完成 reduction、softmax、LayerNorm、GEMM。**
5. **第五到第六周进入 online softmax、FlashAttention、KV cache。**
6. **第七周使用 Triton / TileLang 加速开发，但不跳过硬件原理。**
7. **第八周把实验代码变成带 dispatch、autotune、fallback、测试和性能回归的算子库。**

需要形成的核心闭环是：

```text
明确工作负载
  -> 建立正确的 CPU/库参考实现
  -> 固定 shape / dtype / layout / 精度门槛
  -> 预热和稳定设备状态
  -> 测绝对延迟、吞吐、带宽、launch 数量
  -> 用 profiler 找瓶颈
  -> 每次只改一个主要因素
  -> 验证正确性和性能
  -> 记录失败实验
  -> 扩展到多 shape、多 dtype、长时间运行
```

**领域专家与普通调参者的主要区别**不是会背多少技巧，而是能够：

- 判断瓶颈是 launch、内存、计算、同步、寄存器、occupancy、编译还是上层调度。
- 区分“真实算子变快”和“偷偷减少了工作量”。
- 用 profiler 证据解释速度变化，而不是只报告一个 speedup。
- 在不同 GPU 架构、shape、dtype 和功耗状态下保持方法有效。
- 把一次性的快 kernel 做成可维护、可回退、可验证的算子库。

---

## 2. 当前设备与仓库：你已经具备可开练的条件

### 2.1 手机端实测能力

通过 `adb`、OpenCL runtime 查询和仓库自带 benchmark，当前设备报告：

| 项目 | 实测值 | 训练意义 |
| --- | --- | --- |
| SoC | Qualcomm SM8850 | 必须按真实设备和驱动做 dispatch，不按营销名称猜能力 |
| GPU | QUALCOMM Adreno(TM) 840 | 当前移动 GPU 主练习设备 |
| Android | Android 16 / API 36 | 可使用现代 Android、Vulkan 与 AHardwareBuffer 路径 |
| OpenGL ES | 3.2 | 图形能力参考；本路线主要练 compute |
| OpenCL device version | OpenCL 3.0 | 最新 OpenCL 规范已到 3.1，但本机只能按设备实际 3.0 能力编程 |
| OpenCL C | OpenCL C 3.0 | 编译 kernel 时仍需查询可选 feature/extension |
| OpenCL driver | 0842.27.1 / Compiler E031.50.19.18 | 编译缓存必须把 driver version 放入 key |
| Compute Units | 12，驱动报告值 | 只用于容量估计，不能直接等价为 NVIDIA SM |
| Global memory | 7,860,164,608 bytes，约 7.32 GiB | 是统一系统内存视图，不等价于独立显存卡的专用 VRAM |
| Max allocation | 1,965,041,152 bytes，约 1.83 GiB | 单 buffer 必须控制大小；大模型需分块、复用和量化 |
| Global cache | 1 MiB，64-byte cache line | 布局、连续访问、重用与工作集大小仍然重要 |
| Local memory | 32 KiB / work-group 上限资源 | tiled kernel 必须把 tile、寄存器和并发 work-group 一起考虑 |
| Max work-group | 1024 | 只是合法上限，不是推荐值；通常从 64/128/256 sweep |
| Preferred/native vector | `float4`、`half8` | 可作为候选，但必须实测，不能认为向量类型自动获得等比例加速 |
| Host unified memory | true | CPU/GPU 共享物理内存并不等于没有 cache、同步、导入和页面迁移成本 |
| SVM capability | coarse-grain buffer + fine-grain buffer + atomics | 仍然要验证驱动行为；未报告 fine-grain system SVM |
| Subgroup | `cl_khr_subgroups` 与 shuffle/vote/arithmetic 扩展 | 可练习 CUDA warp-level primitive 的移动端对应技术 |
| 低精度能力 | FP16、integer dot product、QCOM bfloat16 product、dot-product8 | 适合练习 FP16、INT8/BF16 类推理 kernel，但能力和 Tensor Core 不等价 |
| 高级内存扩展 | AHardwareBuffer、dma-buf、ext host ptr、on-chip global memory | 可进一步练 zero-copy、跨 API buffer 和片上内存实验 |
| 重放/调度扩展 | `cl_qcom_recordable_queues` | 与 CUDA Graph / OpenCL command-buffer 的目标相似：摊薄重复提交开销 |

注意两个典型驱动陷阱：

- `CL_DEVICE_MAX_CLOCK_FREQUENCY` 当前报告为 `1 MHz`，明显不是可用频率数据。移动端驱动查询值必须做合理性检查。
- 仓库输出中的 `qcom_extension_advertised = 0` **只表示**当前未发现 `cl_qcom_subgroup_constant_load`，不代表所有 QCOM 扩展都不存在。当前设备实际暴露了大量其他 QCOM 扩展。

### 2.2 当前主机工具链

当前主机已具备：

- `adb`
- Android NDK r28c：`/opt/Android/Ndk/android-ndk-r28c`
- `clang++ 21`
- `cmake 4.2.3`
- `ninja 1.13.2`

当前主机未检测到：

- `nvidia-smi`
- `nvcc`

因此建议双轨并行：

- **轨道 A：Adreno/OpenCL 立即动手。**每天都能真机测量。
- **轨道 B：CUDA 使用远程 NVIDIA GPU。**等有 CUDA 环境后复刻同一个练习，比较两端优化方法。

### 2.3 当前仓库已跑通的 Adreno MVP

在设备温控状态正常、每项 50 次迭代的单次实测中：

| 练习 | Baseline | Optimized | Speedup | 主要优化来源 |
| --- | ---: | ---: | ---: | --- |
| `xmem_gemm` | 3.861685 ms | 0.458689 ms | 8.42x | 专用布局、图像对象、片上/局部重用、并行计算 |
| `copy_vectorized` | 1.400043 ms | 0.647188 ms | 2.16x | 减少一次拷贝并使用 `float4`；不是纯粹的向量化单因素对比 |
| `pipeline_fusion` | 0.337181 ms | 0.222296 ms | 1.52x | 两个 kernel 与中间 buffer 融合为一个 kernel |
| `kv_block` | 4.369839 ms | 0.123715 ms | 35.32x | 128 次 token kernel launch 合并成一次 block launch，主要是提交开销摊薄 |

这些结果是非常好的入门教材，但必须正确解释：

- `copy_vectorized` 的 baseline 做了两次 scalar copy，而 optimized 做一次 vector copy。它展示的是“减少数据移动 + 向量化”的组合收益，不能把 2.16x 全部归因于 `float4`。
- `kv_block` 的 35.32x 主要来自 128 次 launch 变 1 次 launch，而不是单个线程执行变快 35 倍。
- `xmem_gemm` 比较的是朴素标量 GEMM 与高度专用的多阶段实现，适合看系统级收益；若研究单项技术，需要拆分 prepack、source pack、GEMM 和 store 时间。
- 当前计时使用 host wall clock 包围多次 enqueue 并在末尾 `clFinish`。它适合测稳态端到端延迟；若要得到单 kernel 时间，必须打开 queue profiling 并读取 event timestamp。

---

## 3. 立即执行：前 48 小时 MVP

### 3.1 第 0 步：保存设备与环境快照

```bash
cd /media/code/tools/ai-performance-engineering

adb devices -l
adb shell getprop ro.product.model
adb shell getprop ro.build.version.release
adb shell getprop ro.build.version.sdk
adb shell getprop ro.soc.model
adb shell dumpsys SurfaceFlinger | grep -i GLES

printf 'ANDROID_NDK=%s\n' "$ANDROID_NDK"
cmake --version
ninja --version
clang++ --version
nvidia-smi || true
nvcc --version || true
```

把输出保存到实验目录，例如：

```bash
mkdir -p code/artifacts/adreno_bootcamp/day00
{
  date -Iseconds
  adb devices -l
  adb shell getprop
  adb shell dumpsys SurfaceFlinger
} > code/artifacts/adreno_bootcamp/day00/device_snapshot.txt 2>&1
```

验收标准：

- 能明确写出设备、Android 版本、GPU、OpenCL 版本和 driver 版本。
- 能解释为什么“OpenCL 3.0”不表示所有 3.0 feature 都必须存在。
- 能解释为什么移动端 global memory 与 NVIDIA 独立显存不是同一个物理概念。

### 3.2 第 1 步：跑通四个现成场景

```bash
cd /media/code/tools/ai-performance-engineering/code

export PYTHONPATH=.
export ANDROID_NDK=/opt/Android/Ndk/android-ndk-r28c
export AISP_ADRENO_XMEM_KERNEL=/media/code/llm/llama/llama.cpp/ggml/src/ggml-opencl/kernels/gemm_xmem_f16_f32_os8.cl
export AISP_ADRENO_MINIMAL_ITERATIONS=50

python ch01/compare_adreno_minimal.py
python ch05/compare_adreno_minimal.py
python ch06/compare_adreno_minimal.py
python ch11/compare_adreno_minimal.py
```

关键入口：

- `code/core/benchmark/adreno_minimal.py`
- `code/core/benchmark/adreno_minimal_opencl.cpp`
- `code/ch01/compare_adreno_minimal.py`
- `code/ch05/compare_adreno_minimal.py`
- `code/ch06/compare_adreno_minimal.py`
- `code/ch11/compare_adreno_minimal.py`
- `/media/code/llm/llama/llama.cpp/ggml/src/ggml-opencl/kernels/gemm_xmem_f16_f32_os8.cl`

验收标准：

- 四个场景都返回正确性误差与性能数据。
- 能指出每个 speedup 主要来自内存、计算、融合还是 launch 减少。
- 不把不同工作量的 baseline/optimized 当成单因素实验。

### 3.3 第 2 步：把一次跑分升级为可靠实验

至少重复 10 组，每组 50~200 次迭代，记录中位数、P90、最小值和温度。不要只保存平均值。

```bash
cd /media/code/tools/ai-performance-engineering/code
mkdir -p artifacts/adreno_bootcamp/day01

for run in $(seq -w 1 10); do
  adb shell dumpsys thermalservice \
    > "artifacts/adreno_bootcamp/day01/thermal_before_${run}.txt"

  python ch01/compare_adreno_minimal.py \
    | tee "artifacts/adreno_bootcamp/day01/ch01_${run}.txt"

  adb shell dumpsys thermalservice \
    > "artifacts/adreno_bootcamp/day01/thermal_after_${run}.txt"

  sleep 3
done
```

移动端基准稳定性要求：

- 固定充电状态、屏幕亮度和前台应用。
- 尽量避免系统更新、后台同步、动画和相机等 GPU 工作负载。
- 每项实验先预热；长实验同时报告前 10 秒与稳态 1~5 分钟数据。
- 温度、功耗状态或 GPU cooling device 发生变化时，不能把结果与冷机数据直接混合。
- 不 root 的情况下不要强行锁频；真实部署往往更需要理解 DVFS 下的持续性能。

### 3.4 第 3 步：做第一个公平对照实验

修改 `code/core/benchmark/adreno_minimal_opencl.cpp` 前先复制到独立实验目录，避免破坏所有章节共用的 minimal harness：

```bash
mkdir -p code/labs/adreno_bootcamp
cp code/core/benchmark/adreno_minimal_opencl.cpp \
   code/labs/adreno_bootcamp/adreno_opencl_lab.cpp
```

第一个实验只比较：

```text
一次 scalar float copy
vs
一次 float4 copy
```

保持以下条件完全一致：

- 相同字节数。
- 相同输入与输出。
- 相同 launch 次数。
- 相同 warmup 和 iteration。
- 相同 queue、buffer 和同步方式。
- 单独处理元素数不是 4 倍数的 tail。

需要 sweep：

- 元素数：4 KiB、64 KiB、1 MiB、4 MiB、16 MiB。
- local size：64、128、256、512。
- `float`、`float2`、`float4`、`float8`。
- 对齐与故意偏移 4/8/16 bytes 的地址。

输出至少包含：

```text
bytes
latency_us
effective_GBps
local_size
vector_width
alignment
max_abs_error
device_temperature_before_after
```

验收标准：你能回答“`float4` 在什么尺寸和对齐下更快、为什么小尺寸可能反而更慢”。

### 3.5 第 4 步：添加 OpenCL event 计时

当前 minimal harness 使用 host wall clock 与 `clFinish`。下一步应创建带 profiling 的 queue：

```cpp
const cl_queue_properties properties[] = {
    CL_QUEUE_PROPERTIES,
    CL_QUEUE_PROFILING_ENABLE,
    0,
};
```

enqueue 时保留 `cl_event`，完成后读取：

```text
CL_PROFILING_COMMAND_QUEUED
CL_PROFILING_COMMAND_SUBMIT
CL_PROFILING_COMMAND_START
CL_PROFILING_COMMAND_END
```

分别计算：

- queue 等待时间。
- submit 到 start 的调度时间。
- kernel device execution 时间。
- host 端端到端时间。

验收标准：能解释为什么 10 微秒级 kernel 的 host wall time 可能明显大于 device execution time。

---

## 4. CUDA 与 Adreno/OpenCL 的共同原理和术语映射

### 4.1 最重要的结论

**大多数性能原理相似，具体硬件机制、API、工具和最优参数不同。**

可迁移的核心原理包括：

- 大量独立工作并行化。
- 连续、对齐、可合并的内存访问。
- 数据重用与分块。
- 控制寄存器和片上内存占用。
- 减少分支发散。
- 提高 arithmetic intensity。
- 减少 kernel launch、同步和中间数据落地。
- 用低精度、向量指令或矩阵指令提高吞吐。
- 用多 queue/stream 与 event 构造依赖和重叠。
- 针对 shape、dtype、layout 和硬件做 dispatch/autotune。
- 把正确性、计时方法、温控和 profiler 证据放在 speedup 前面。

### 4.2 CUDA 与 OpenCL/Vulkan Compute 对照表

| CUDA / NVIDIA | OpenCL / Adreno | Vulkan Compute | 说明 |
| --- | --- | --- | --- |
| thread | work-item | invocation | 最小逻辑执行单元 |
| block / CTA | work-group | workgroup | 可共享片上数据并同步的一组线程 |
| grid | NDRange | dispatch grid | 全部工作范围 |
| warp，通常 32 threads | subgroup，宽度必须查询 | subgroup | 不要在 OpenCL/Vulkan 中硬编码 32 |
| `threadIdx/blockIdx` | `get_local_id/get_group_id` | `gl_LocalInvocationID/gl_WorkGroupID` | 索引模型对应 |
| global memory / device memory | `__global` buffer/image | storage buffer/image | Adreno 常处于 UMA，物理拓扑不同 |
| shared memory | `__local` memory | workgroup shared memory | **OpenCL local 对应 CUDA shared** |
| per-thread register/local spill | `__private` 与 spill | private values | **OpenCL private 才对应线程私有值** |
| constant memory | `__constant` | uniform/storage readonly | cache 与容量细节不同 |
| texture/surface | image/sampler | sampled/storage image | 移动 GPU 对 image 路径可能非常敏感 |
| stream | command queue | queue | 都用于异步提交和依赖组织 |
| CUDA event | OpenCL event | fence/semaphore/timestamp query | 用于依赖和计时 |
| CUDA Graph | `cl_khr_command_buffer` 或 QCOM recordable queue | reusable command buffer | 当前手机无 `cl_khr_command_buffer`，但有 QCOM recordable queue |
| warp shuffle/vote | subgroup shuffle/vote/arithmetic | subgroup operations | 当前 Adreno 暴露对应 Khronos 扩展 |
| `__syncthreads()` | `barrier()` / work-group barrier | `barrier()` | 只同步 work-group 内部 |
| atomic | OpenCL atomic | Vulkan atomic | 冲突和顺序仍需谨慎 |
| pinned host memory | QCOM ext host ptr、dma-buf、AHardwareBuffer | host-visible/imported memory | 目的相似，生命周期和 cache 同步完全不同 |
| Unified Memory | SVM / UMA / imported shared buffers | host-visible/shared allocations | 不能把“共享地址”误认为“免费访问” |
| `cudaMallocAsync` / memory pool | buffer pool、sub-buffer、SVM allocator | allocator + suballocation | OpenCL 标准中没有完全等价的一键方案 |
| `cp.async` / TMA | `async_work_group_copy`、预取、vendor path | cooperative load | 通常没有逐代 NVIDIA 功能的一比一对应 |
| Tensor Core MMA / WGMMA / TCGEN05 | integer dot、BF16 product、vendor ML ops | cooperative matrix / vendor extension | 数值格式、吞吐和编程模型不等价 |
| occupancy | 同驻 work-group / subgroup 数量 | 同驻 workgroup 数量 | 都受寄存器、片上内存和线程数限制 |
| Nsight Compute | Snapdragon Profiler、vendor counter、event timing | APA/AGI/vendor tools | 移动端公开 counter 和支持设备范围更受限制 |
| Nsight Systems | Android Performance Analyzer / Perfetto | APA/AGI/Perfetto | 适合找 CPU-GPU timeline、提交和同步问题 |

### 4.3 最容易混淆的差异

#### 差异 1：内存拓扑

离散 NVIDIA GPU 常见模型：

```text
CPU DRAM --PCIe/NVLink--> GPU VRAM
```

手机 SoC 常见模型：

```text
CPU / GPU / NPU / ISP 共享系统内存控制器和 DRAM
```

因此手机端通常没有传统意义上的 PCIe host-to-device copy，但仍存在：

- cache coherent 或 non-coherent 的区别。
- CPU/GPU ownership 与同步。
- buffer import/export 成本。
- 内存带宽竞争。
- 页面映射、分配和首次访问成本。
- image、buffer、压缩布局或硬件友好布局差异。

“统一内存”不表示 CPU 和 GPU 能无成本同时读写同一块数据。

#### 差异 2：warp 与 subgroup

CUDA 程序经常利用固定 warp 语义。OpenCL/Vulkan 应通过 runtime 查询 subgroup 支持和合适宽度，不要因为熟悉 CUDA 就硬编码 32。

当前 Adreno 支持 subgroup shuffle、vote、non-uniform arithmetic 等扩展，适合实现：

- reduction。
- softmax 的 max/sum。
- LayerNorm 的 mean/variance。
- ballot 与稀疏 mask。
- warp/subgroup 内数据交换。

#### 差异 3：occupancy 不是最终目标

两端都不能盲目追求 100% occupancy。可能出现：

- 降低寄存器后 occupancy 上升，但 spill 到内存导致更慢。
- 增大 work-group/block 后并行度上升，但 local/shared memory 或同步成本上升。
- 低 occupancy kernel 通过更高 ILP、更多数据重用或矩阵指令仍然更快。

目标应该是**最短时间或最高持续吞吐**，occupancy 是诊断指标。

#### 差异 4：移动端热与功耗是一级约束

桌面/服务器 GPU 也会功耗限制，但手机端更容易在几十秒到几分钟内因温度、整机功耗和前台渲染发生频率变化。必须同时测：

- 冷机 burst latency。
- 30 秒持续性能。
- 3~10 分钟稳态性能。
- 电池、skin、GPU 温度。
- UI 与其他 accelerator 干扰。

#### 差异 5：工具和可观测性

CUDA 生态中 Nsight Compute 可以提供非常细的 kernel counter。移动端 OpenCL 的公开 counter、驱动符号和工具支持更不一致，所以更需要：

- 精确 event timestamp。
- 构造单因素 microbenchmark。
- 读取程序 build log。
- 查询所有 device limit 与 extension。
- 使用 APA/Perfetto 找 host 提交、线程调度和系统级问题。
- Vulkan 路径使用 AGI/APA；OpenCL 深度 counter 视 Qualcomm 工具与设备支持情况而定。

---

## 5. 性能工程最佳实践：任何后端都必须遵守

### 5.1 正确性优先

每个算子必须有 CPU 或可信库参考实现，并至少检查：

- max absolute error。
- max relative error。
- NaN/Inf。
- 随机数据与结构化极值数据。
- 非对齐尺寸、非整 tile、空输入和极小输入。
- FP32、FP16/BF16、INT8 等不同 dtype 的误差门槛。
- 多次运行一致性；涉及 atomic 时检查非确定性范围。

不能只检查一个元素或一个 checksum。

### 5.2 先测绝对值，再谈 speedup

每个报告至少包含：

```text
device / driver / build options
shape / dtype / layout
warmup / iterations / repeat groups
p50 / p90 / min / max latency
effective bandwidth or achieved FLOP/s
launch count
temporary memory bytes
correctness tolerance
temperature and sustained duration
baseline and optimized absolute value
speedup
```

对于内存算子：

```text
effective bandwidth = logical bytes read and written / elapsed time
```

对于 GEMM：

```text
FLOPs ~= 2 * M * N * K
achieved FLOP/s = FLOPs / elapsed time
```

注意：如果算法融合后减少了实际内存流量，应同时报告“逻辑 tensor bytes”和“实际估计 DRAM bytes”，避免指标解释混乱。

### 5.3 分离不同阶段成本

以下成本应分别测量：

- 首次 context/runtime 初始化。
- 程序编译/JIT。
- 权重 prepack。
- buffer allocation。
- 数据上传/导入。
- kernel steady-state。
- 输出下载/同步。
- graph/command-buffer capture。
- graph/recorded queue replay。

大模型推理中，权重 prepack 可以是一次性成本，但请求级 activation pack 可能是每次成本，不能混为一谈。

### 5.4 一次只改一个主要变量

先做单因素归因，再做组合优化：

```text
baseline
  -> 只改访问顺序
  -> 只改向量宽度
  -> 只加 local/shared tile
  -> 只做 unroll/ILP
  -> 只改 work-group/block
  -> 只做 fusion
  -> 最后组合
```

组合版更接近生产性能，单因素版用于建立硬件直觉。两者都要保留。

### 5.5 不要忽视编译器输出

CUDA 侧关注：

- registers per thread。
- shared memory per block。
- spill load/store。
- SASS/PTX 中的 vector load、tensor instruction、barrier 和 memory instruction。
- 实际编译的 SM 架构。

OpenCL 侧关注：

- 完整 build log。
- `-cl-fast-relaxed-math` 等选项对精度的影响。
- device/driver/build-options 组合对应的 binary cache。
- extension 与 feature macro。
- kernel-specific work-group info。

### 5.6 建立失败实验日志

专家成长最快的材料通常不是成功结果，而是失败记录：

| 日期 | 假设 | 修改 | 结果 | Profiler 证据 | 失败原因 | 下一步 |
| --- | --- | --- | --- | --- | --- | --- |
| 例 | `float8` 会优于 `float4` | load/store 改成 `float8` | -7% | register 与 tail 增加 | 工作集/对齐/寄存器不合适 | 只在 128B 对齐大 tensor dispatch |

---

## 6. 分阶段练习：从小白到能够开发高性能算子

## 阶段 0：基准、计时和硬件调查

建议时间：2 天。

### 练习 0.1：设备能力清单

Adreno：

- 查询 OpenCL platform/device/version/extension。
- 查询 global/local memory、max work-group、vector width、SVM、image support。
- 输出 JSON，key 中包含 driver version。

CUDA：

- 运行 CUDA Samples 的 `deviceQuery` 与 `bandwidthTest`。
- 查询 compute capability、SM 数、warp size、shared memory、register、L2、memory bus。

验收标准：能从能力数据推导“哪些优化可以尝试”，同时明确“能力存在不等于实际更快”。

### 练习 0.2：三种计时

实现并比较：

1. 未同步 host timer。
2. 同步后的端到端 host timer。
3. GPU event/device timestamp。

验收标准：能演示未同步计时如何得到虚假的超低延迟。

### 练习 0.3：稳定性

对同一 kernel 连续运行 10 分钟，画出：

- 时间序列 latency。
- p50/p90。
- GPU/skin temperature。
- 前 10 秒、1 分钟、5 分钟、10 分钟吞吐。

验收标准：可以区分 warmup、DVFS、thermal throttling 和随机系统抖动。

---

## 阶段 1：线程映射、访存合并和向量化

建议时间：1 周。

### 练习 1.1：SAXPY / elementwise

实现：

```text
y[i] = a * x[i] + y[i]
```

版本：

- 朴素 scalar。
- grid-stride loop / 每线程多元素。
- `float4`。
- FP16/half vector。
- fused bias + activation。

要测：

- 4 KiB 到 64 MiB。
- 对齐与错位。
- 不同 block/work-group size。
- launch-bound 与 bandwidth-bound 分界。

### 练习 1.2：转置与 coalescing

当前 CUDA 参考：

- `code/ch06/cuda_extensions/coalescing_kernels.cu`
- `code/core/benchmark/cuda/memory_patterns_extension.cu`

重点对比：

- uncoalesced transpose。
- coalesced transpose。
- shared/local tiled transpose。
- padded tile 避免 bank conflict。

Adreno 版本使用 `__local` tile，sweep：

```text
tile: 8x8, 16x16, 32x8, 32x16
padding: 0, 1, 2
vector: scalar, float2, float4
```

验收标准：

- 能通过有效带宽证明访问模式改善。
- 能解释为什么 local memory 版本不一定在所有尺寸都更快。
- 能识别 local memory 过大导致并发 work-group 降低。

### 练习 1.3：真实移动端 buffer 与 image 对照

当前 Adreno xmem GEMM 已使用 image 对象。新增 microbenchmark 对比：

- linear buffer load。
- image load。
- `image2d_from_buffer`。
- 不同二维布局和行 pitch。

验收标准：只在 profiler 或多组尺寸数据证明 image 更好时使用 image；不得把“移动 GPU 喜欢 image”当成无条件规则。

---

## 阶段 2：local/shared memory、寄存器、occupancy 与 ILP

建议时间：1 周。

### 练习 2.1：block/work-group sweep

对 copy、reduction、transpose、GEMM 分别测试：

```text
32 / 64 / 128 / 256 / 512 / 1024 threads or work-items
```

记录：

- latency。
- registers/private pressure。
- local/shared bytes。
- achieved occupancy 或可替代的并发证据。
- spill。

验收标准：不能只给“256 最快”，必须解释为什么。

### 练习 2.2：低 occupancy 与高 ILP

当前参考：

- `code/core/profiling/cuda/occupancy_extension.cu`
- `code/ch06/optimized_ilp_low_occupancy_vec4_impl.cuh`
- `code/labs/occupancy_tuning/triton_matmul.py`
- `code/labs/occupancy_tuning/triton_matmul_schedules.py`

实现每线程 1、2、4、8 个独立 accumulator，对比：

- occupancy。
- instruction-level parallelism。
- register pressure。
- memory latency hiding。

验收标准：找到一个 occupancy 下降但总性能上升的例子，并用数据解释。

### 练习 2.3：subgroup/warp reduction

实现 reduction：

- global atomic baseline。
- local/shared tree reduction。
- subgroup/warp shuffle reduction。
- 两级 reduction。

Adreno 使用当前设备支持的 subgroup shuffle/arithmetic 扩展，并查询实际 subgroup size。

验收标准：支持非 2 次幂输入和任意尾部，误差可解释。

---

## 阶段 3：GPU 显存/内存管理与流水线

建议时间：1 周。

### 练习 3.1：分配成本与内存池

错误 baseline：每次调用都 allocation/free。

优化版本：

- 持久 buffer。
- size-class pool。
- suballocation。
- shape/dtype/layout-aware reuse。
- 有上限的 cache 与回收策略。

CUDA：比较 `cudaMalloc/cudaFree` 与 stream-ordered allocator / memory pool。

OpenCL：比较 `clCreateBuffer/clReleaseMemObject` 与复用 buffer；需要时使用 sub-buffer 或自定义 arena。

验收标准：分别报告 allocation latency、steady-state kernel latency 与峰值内存。

### 练习 3.2：host-device / shared-memory 数据路径

CUDA：

- pageable host memory。
- pinned memory。
- async copy。
- 双缓冲。
- H2D/compute/D2H overlap。

Adreno：

- `CL_MEM_COPY_HOST_PTR`。
- map/unmap。
- SVM buffer。
- `cl_qcom_ext_host_ptr`。
- dma-buf / AHardwareBuffer import。

验收标准：证明 zero-copy 或共享 buffer 在目标 workload 上真的减少端到端时间；同时验证 cache/ownership 同步正确。

### 练习 3.3：stream/queue 与 event DAG

构造三阶段 pipeline：

```text
prepare -> compute -> consume
```

比较：

- 每阶段 `finish/synchronize`。
- event dependency。
- 两个 buffer 的 ping-pong。
- out-of-order queue 或多 queue。

CUDA 当前参考：

- `code/ch11/baseline_gemm_streams.py`
- `code/ch11/optimized_gemm_streams.py`

验收标准：timeline 能显示真实 overlap，而不是只看到 API 异步但硬件仍串行。

### 练习 3.4：CUDA Graph 与移动端录制提交

CUDA 当前参考：

- `code/ch12/baseline_cuda_graphs.cu`
- `code/ch12/optimized_cuda_graphs.cu`

移动端对应路线：

- 标准 OpenCL `cl_khr_command_buffer`，若设备支持。
- 当前设备的 `cl_qcom_recordable_queues`，需按 Qualcomm 扩展文档实现。
- Vulkan reusable command buffer。

验收标准：分别报告 capture/record 成本、单次 replay 成本、达到盈亏平衡所需 replay 次数。

---

## 阶段 4：高性能 GEMM 与算子融合

建议时间：1~2 周。

### 练习 4.1：GEMM 优化阶梯

依次实现，不能直接跳到最终版本：

1. 一个 work-item/thread 计算一个输出。
2. 连续读取与布局调整。
3. local/shared tiled GEMM。
4. 每线程计算多个输出。
5. vector load/store。
6. double buffering / software pipelining。
7. FP16/BF16/INT8。
8. image/texture 路径。
9. 权重 prepack。
10. shape-specialized dispatch。
11. matrix/tensor instruction 或 vendor specialized kernel。

每一级必须保存：

- kernel source。
- shape 与 dtype。
- GFLOP/s。
- bandwidth。
- local/shared 与 register 使用。
- 精度。
- 为什么下一步有效。

当前 Adreno 参考：

- `code/core/benchmark/adreno_minimal_opencl.cpp`
- `/media/code/llm/llama/llama.cpp/ggml/src/ggml-opencl/kernels/gemm_xmem_f16_f32_os8.cl`

当前 CUDA 参考：

- `code/ch01/baseline_gemm.cu`
- `code/ch01/optimized_gemm_strided.cu`
- `code/ch01/optimized_gemm_batched.cu`
- `code/core/benchmark/cuda/cutlass_gemm_extension.cu`

### 练习 4.2：融合的收益模型

当前 Adreno `pipeline_fusion`：

```text
scale -> temporary buffer -> bias + ReLU
```

融合后：

```text
scale + bias + ReLU
```

需要计算理论减少的：

- kernel launch 数。
- temporary allocation。
- temporary write/read bytes。

再用实测验证。

进一步实现：

- bias + GELU。
- residual + LayerNorm。
- dequantize + GEMM epilogue。
- rotary embedding + Q/K layout transform。

验收标准：融合前后数学等价；若使用 fast math，误差门槛单独声明。

### 练习 4.3：什么时候不融合

构造以下反例：

- 融合后寄存器暴涨并 spill。
- 融合后 work-group 并发度下降。
- 一个中间结果被多个 consumer 重用，强行融合反而重复计算。
- shape 太大导致单 kernel watchdog/调度风险。

验收标准：能够根据数据决定拆分边界，而不是默认“kernel 越少越好”。

---

## 阶段 5：Softmax、LayerNorm、FlashAttention 与 KV Cache

建议时间：2 周。

### 练习 5.1：online softmax

先实现普通三阶段：

```text
max reduction
exp + sum reduction
normalize
```

再实现 online softmax，维护 running max 与 running sum。

必须覆盖：

- 长度 1、31、32、33、127、128、129。
- 大负数。
- 全相等值。
- FP16 输入 / FP32 accumulator。
- causal mask 或任意 mask。

验收标准：能解释 online softmax 为什么是 FlashAttention 的关键组成，而不只是背公式。

### 练习 5.2：FlashAttention forward MVP

MVP 限定：

```text
batch = 1
head = 1 或少量 heads
head_dim = 64
FP16 input + FP32 accumulation
causal=false 先做，再加 causal
仅 forward
```

步骤：

1. CPU/PyTorch 参考：`softmax(QK^T / sqrt(d))V`。
2. tiled Q/K/V。
3. tile 内计算 score。
4. online softmax 更新 running max/sum。
5. 不物化完整 attention matrix。
6. 验证误差与峰值内存。

CUDA 练习入口：

- `code/ch14/flash_attention_sdpa_bench.py`
- `code/ch16/baseline_dense_attention_flash.py`
- `code/ch16/optimized_dense_attention_flash.py`
- Triton 官方 fused attention tutorial。
- FlashAttention 官方实现。

Adreno 版本先追求：

- 正确的 tiled online-softmax。
- 明确 local memory 预算。
- 减少 score matrix DRAM 落地。
- subgroup reduction。
- 针对短序列与小 batch 的 launch/fusion 优势。

不要一开始追求与桌面 Tensor Core kernel 相同的结构。移动 GPU 的可用矩阵指令、片上内存、subgroup、带宽、热预算都不同。

### 练习 5.3：FlashAttention-4 的学习方式

截至核验日期，FlashAttention 官方项目已包含面向 Blackwell 的 FlashAttention-4 路线。它属于专家阶段材料，学习重点不是复制全部代码，而是理解：

- 异构硬件单元如何分工。
- 数据搬运与矩阵计算如何形成 pipeline。
- 为什么需要 tile scheduler。
- 为什么 epilogue、softmax 和矩阵乘高度耦合。
- 为什么架构专用 kernel 仍需要通用 fallback。

建议先完成上述简单 forward MVP，再阅读 FlashAttention-2/3/4 的演进。

### 练习 5.4：KV cache 更新与 paged KV

当前 Adreno `kv_block` 已展示 launch amortization。下一步拆成：

- 单 token update。
- 多 token contiguous update。
- paged/block table update。
- 非连续 page gather。
- quantized KV。
- active-window KV。

当前仓库参考：

- `code/ch13/baseline_kv_cache_naive.py`
- `code/ch13/optimized_kv_cache_naive.py`
- `code/ch13/optimized_kv_cache_naive_flash_blockwise.py`
- `code/ch15/baseline_kv_cache_management.py`
- `code/ch15/optimized_kv_cache_management.py`
- `code/ch18/kv_cache_integration_example.py`
- `code/ch19/baseline_dynamic_quantized_cache_coalesced.py`
- `code/ch19/optimized_dynamic_quantized_cache_coalesced.py`

验收标准：同时报告 update latency、decode latency、内存占用、fragmentation 与 launch 数量。

### 练习 5.5：FlashInfer 与 Transformer Engine 的定位

不要把这些库只当成“可直接调用的黑盒”。正确学习方式：

#### FlashInfer

- 先跑单请求 attention。
- 再跑 batch/paged KV。
- 观察 plan/run 分离。
- 找到 shape、page size、head dim、dtype 的 dispatch。
- 通过 profiler 分析 decode 与 prefill 不同瓶颈。
- 最后尝试添加一个小 shape specialization 或测试。

当前仓库参考：

- `code/ch16/baseline_flashinfer_block_sparse.py`
- `code/ch16/optimized_flashinfer_block_sparse.py`

#### Transformer Engine

- 重点学习 FP8/BF16 transformer layer、scaling metadata、fused attention/normalization 与框架集成。
- 先比较 PyTorch baseline 与 TE module。
- 再分析 cast、transpose、amax、scale update、GEMM、attention 的 timeline。
- 不要在没有误差分析的情况下只追求 FP8 speedup。

---

## 阶段 6：Triton、TileLang 与高性能 DSL

建议时间：1 周。

### 6.1 正确顺序

推荐顺序：

1. CUDA/OpenCL 手写 vector add、reduction、transpose。
2. 手写一个 tiled GEMM 或 softmax。
3. 用 Triton 重写同一算子。
4. 比较生成代码、tile、warps、stages、register 和性能。
5. 再用 TileLang 表达更复杂的 tile/pipeline。

如果一开始只会改 `BLOCK_SIZE`，很难成为真正的算子专家。

### 6.2 Triton MVP

当前仓库入口：

- `code/ch14/triton_examples.py`
- `code/ch14/baseline_triton_persistent.py`
- `code/ch14/optimized_triton_persistent.py`
- `code/ch14/triton_persistent_batched.py`
- `code/labs/occupancy_tuning/triton_matmul.py`
- `code/labs/occupancy_tuning/triton_matmul_schedules.py`

练习顺序：

- vector add。
- fused softmax。
- matmul。
- autotune config。
- persistent matmul。
- fused attention。

每个 Triton kernel 都要回答：

- program id 如何映射输出 tile。
- block pointer/layout 如何影响 coalescing。
- `num_warps`、`num_stages` 为什么有效。
- mask/tail 的成本。
- accumulator dtype。
- 编译后的 register 与 occupancy。

### 6.3 TileLang MVP

使用官方安装和示例，以固定 commit 或版本建立环境。建议重写同一个 matmul/attention MVP，重点观察：

- tile-level dataflow。
- shared memory layout。
- pipeline stage。
- layout/fragment 表达。
- autotuning。
- 与 Triton/CUDA baseline 的可读性、开发效率和性能差异。

TileLang 与 Triton 是 CUDA 主线技能；当前 Adreno 路线仍以 OpenCL/Vulkan、ncnn/MNN/llama.cpp 等移动后端为主。不要默认 Triton/TileLang 能直接部署到 Adreno。

---

## 阶段 7：高性能算子库工程化

建议时间：1~2 周，之后持续迭代。

建议在 `code/labs/` 下新建 `gpu_kernel_bootcamp/`，建立一个最小算子库：

```text
gpu_kernel_bootcamp/
  README.md
  include/
    operator_api.h
  cpu_reference/
    copy.cpp
    softmax.cpp
    gemm.cpp
    attention.cpp
  opencl/
    runtime.cpp
    device_info.cpp
    program_cache.cpp
    memory_pool.cpp
    copy.cl
    softmax.cl
    gemm.cl
    attention.cl
  cuda/
    runtime.cu
    memory_pool.cu
    copy.cu
    softmax.cu
    gemm.cu
    attention.cu
  benchmarks/
  tests/
  configs/
  results/
```

### 7.1 MVP 算子集合

只做五个：

1. copy/transform。
2. reduction/softmax。
3. LayerNorm/RMSNorm。
4. GEMM。
5. KV update 或 attention forward。

### 7.2 必须具备的库能力

- CPU reference。
- CUDA 和 OpenCL 至少一个可用后端；最终做双后端。
- shape/dtype/layout dispatch。
- extension/capability check。
- fallback kernel。
- program/JIT binary cache。
- memory pool。
- autotuning result cache。
- correctness test。
- benchmark JSON。
- profiler artifact 路径。
- 性能回归阈值。

### 7.3 Dispatch key 示例

```text
backend
device_vendor
device_name
driver_version
op
dtype
layout
shape_bucket
feature_extensions
build_options
```

移动端驱动更新后必须使 program binary 与 autotune cache 失效。

### 7.4 Autotune 原则

搜索空间示例：

```text
work-group/block size
tile M/N/K
vector width
unroll factor
stages
warps/subgroup mapping
local/shared memory layout
split-K
persistent vs non-persistent
buffer vs image
```

Autotune 必须：

- 先过滤不合法配置。
- 每个配置做正确性验证。
- 多次测量并使用稳健统计量。
- 限制调优时间。
- 以 device/driver/shape/dtype 为 key 缓存。
- 保留安全 fallback。

---

## 7. 8 周强化训练计划

| 周 | 主目标 | Adreno 实战 | CUDA 实战 | 周末交付物 |
| --- | --- | --- | --- | --- |
| 1 | 计时与基准可信度 | 跑通四个 minimal；event timing；温控曲线 | CUDA Samples deviceQuery/bandwidth；CUDA event | 一份可复现实验报告 |
| 2 | 内存访问 | scalar/vector、alignment、buffer/image、transpose | coalescing、shared transpose、bank conflict | 有效带宽 roofline 表 |
| 3 | 执行效率 | work-group sweep、subgroup reduction、local memory | occupancy、warp efficiency、ILP、register | 一份 profiler 证据链 |
| 4 | GEMM | 从朴素到 tiled/image/prepack | 从 naive 到 shared/CUTLASS/Triton | 多 shape GEMM dispatch |
| 5 | 融合与 normalization | fused elementwise、softmax、RMSNorm | fused softmax/LayerNorm、CUDA Graph | 三个可复用 fused op |
| 6 | Attention/KV | online softmax、attention MVP、KV block/page | FlashAttention、FlashInfer、paged KV | attention forward 对照报告 |
| 7 | DSL | 总结 OpenCL 手写经验 | Triton + TileLang 重写 GEMM/attention | DSL 与手写性能比较 |
| 8 | 算子库 | OpenCL runtime/cache/pool/dispatch | CUDA runtime/pool/dispatch | 双后端 MVP 算子库 |

每天 2~3 小时时推荐节奏：

```text
20 分钟：复现实验
40 分钟：读 profiler/代码
60 分钟：只改一个变量
30 分钟：正确性与多 shape
20 分钟：写实验日志
```

如果每天有 6~8 小时，可把 8 周压缩到 4~5 周，但不要跳过单因素实验和报告。

---

## 8. 面向岗位要求的能力拆解与 MVP

### 8.1 “CUDA 算子编程”

最小可行能力：

- 写 vector add、reduction、transpose、softmax、tiled GEMM。
- 正确处理 tail、dtype 和数值误差。
- 用 CUDA event 计时。
- 用 Nsight Compute 找到 memory/compute/occupancy 瓶颈。

专家能力：

- tensor/matrix instruction。
- async copy、software pipeline、persistent kernel。
- architecture-specific dispatch。
- CUTLASS/CuTe/Triton/TileLang 交叉验证。
- 从 profiler counter 反推代码问题。

### 8.2 “GPU 显存管理”

最小可行能力：

- 预分配与复用。
- pinned/async copy 或移动端共享 buffer。
- event 依赖。
- 正确分离 allocation 与 kernel 时间。

专家能力：

- stream-ordered memory pool。
- 多 stream lifetime。
- fragmentation 与 size-class。
- KV cache/page allocator。
- AHardwareBuffer/dma-buf 跨 API 导入。
- 内存压力与 OOM fallback。

### 8.3 “CUDA Kernel 优化”

最小可行能力：

- coalescing。
- shared memory tiling。
- block size sweep。
- divergence 与 ILP。
- fusion。

专家能力：

- roofline + instruction/counter 分析。
- register/shared/occupancy tradeoff。
- warp specialization。
- persistent scheduling。
- graph replay。
- 架构特化的 TMA/WGMMA/TCGEN05 等路径与通用 fallback。

当前仓库的 Blackwell 深入入口包括：

- `code/core/benchmark/cuda/tcgen05_probe.cu`
- `code/ch14/triton_tma_blackwell.py`
- `code/ch14/triton_fp8_advanced.py`

这些属于后期内容，不应替代基础 coalescing、tiling 和计时训练。

### 8.4 “设计高性能算子库”

最小可行能力：

- 5 个算子。
- CPU reference。
- 多 shape/dtype。
- dispatch + fallback。
- benchmark + test。

专家能力：

- autotune/cache。
- binary/JIT cache。
- versioned ABI/API。
- framework integration。
- graph capture compatibility。
- quantization metadata。
- performance CI。
- 多架构发布和回归分析。

### 8.5 “大模型推理优化和部署”

最小可行能力：

- 识别 prefill 与 decode 的不同瓶颈。
- 做 RMSNorm、RoPE、GEMM、softmax、KV update。
- 统计 token/s、TTFT、TPOT、峰值内存。

专家能力：

- FlashAttention/FlashInfer。
- paged/quantized KV。
- continuous batching。
- speculative decoding 配套算子。
- fused MoE routing/GEMM。
- 多 GPU/NVLink/NVSHMEM。
- 移动端低功耗持续推理。

### 8.6 “Triton / TileLang”

最小可行能力：

- 重写 softmax 与 matmul。
- 会 autotune。
- 会看生成代码和 profiler。

专家能力：

- persistent kernel。
- attention/MoE/quantized GEMM。
- 自定义 layout 与 pipeline。
- 多架构 config 与 fallback。
- 与 CUDA/CUTLASS 数值和性能对照。

---

## 9. Adreno 专项最佳实践

### 9.1 必须 runtime 查询，禁止按型号硬编码

检查：

- OpenCL version 与 OpenCL C version。
- extension string。
- max work-group/work-item。
- local memory。
- image limit。
- preferred work-group multiple。
- kernel-specific work-group info。
- subgroup size。
- SVM 与 external memory。

同为 Adreno，不同 Android 版本、OEM driver 和 GPU driver package 也可能行为不同。

### 9.2 向量化是候选，不是结论

当前设备报告 `float4` 和 `half8` 为 preferred/native width。练习时仍需验证：

- 地址对齐。
- tail。
- 寄存器压力。
- kernel 是否 memory-bound。
- 编译器是否真的产生有效向量 load/store。
- 小 tensor 的 launch/branch 成本。

### 9.3 local memory 要实测

Qualcomm 官方优化资料强调向量化、图像对象和访问模式，并提醒某些架构/工作负载中 local memory 未必比合理的 global/image 路径更快。当前设备虽有 32 KiB local memory 与 QCOM local-memory-control 扩展，也不能机械照搬 CUDA shared-memory 教程。

正确做法：

- 同时保留 direct-global、local-tiled、image 三个版本。
- sweep tile。
- 记录 local bytes 与并发工作组。
- 观察边界、barrier 和额外 copy 成本。

### 9.4 利用 image，但保留 buffer fallback

适合尝试 image 的场景：

- 2D/3D 数据。
- 固定格式向量读取。
- 权重/activation 的硬件友好布局。
- sampler 或专用 image cache 有价值。

不适合盲目使用的场景：

- 不规则 gather/scatter。
- 频繁格式转换。
- 设备 image limit 不满足。
- import、pitch 或 layout 转换成本超过收益。

### 9.5 充分利用 subgroup，但必须有 fallback

当前设备支持丰富 subgroup 扩展。推荐用在：

- reduction。
- scan。
- softmax。
- LayerNorm/RMSNorm。
- mask/vote。

生产代码必须：

- runtime check extension。
- 查询 subgroup size。
- 提供 local-memory fallback。
- 对非完整 subgroup/tail 做正确处理。

### 9.6 重放重复工作

当前设备暴露 QCOM recordable queue，可探索重复 decode step、固定 shape elementwise pipeline 或固定 KV update 的录制与 replay。

实验必须分开测：

- record 时间。
- 第一次 replay。
- 稳态 replay。
- 更新参数的成本。
- 多少次 replay 后回本。

### 9.7 program binary cache

OpenCL source 每次 build 会造成启动抖动。缓存至少使用：

```text
device vendor/name
driver version
OpenCL C version
extension set hash
kernel source hash
build options
specialization constants or shape bucket
```

build 失败必须保存完整 log，并回退到安全 kernel。

### 9.8 移动端热稳定优先于冷机峰值

一个 0.5 ms 冷机 kernel 若 3 分钟后变成 1.2 ms，可能不如始终 0.8 ms 的低功耗版本。专家报告至少同时给 burst 与 sustained。

---

## 10. CUDA 专项最佳实践

### 10.1 先与高质量库比较

在写自定义 kernel 前先比较：

- cuBLAS/cuBLASLt。
- cuDNN。
- CUTLASS/CuTe。
- PyTorch SDPA。
- FlashAttention。
- FlashInfer。
- Transformer Engine。

值得自定义的常见理由：

- 多算子融合。
- 特殊 layout/dtype。
- 很小或很不规则的 shape。
- 低延迟 decode。
- 库缺少目标功能。
- 框架调度开销过高。

### 10.2 CUDA 计时

- kernel-only 使用 CUDA event。
- timeline 使用 Nsight Systems。
- kernel counter 使用 Nsight Compute。
- 不在每个 kernel 后调用 `cudaDeviceSynchronize()`，除非实验目的就是测同步。
- capture/replay、allocation、prepack 分开报告。

### 10.3 显存管理

- 高频路径不反复 `cudaMalloc/cudaFree`。
- 使用 memory pool/stream-ordered allocation。
- pinned memory 只在传输确实是瓶颈时使用，并限制总量。
- 用 stream/event 明确 lifetime。
- Unified Memory 要结合 prefetch/advice 和访问模式，不是自动优化开关。
- 监控 fragmentation 与峰值，而不只看已分配 tensor 大小。

### 10.4 Kernel 优化优先级

通常按以下顺序更高效：

1. 正确性与公平 benchmark。
2. launch/同步问题。
3. coalescing 与数据布局。
4. 冗余内存流量与 fusion。
5. shared memory/data reuse。
6. block/warp mapping。
7. register/occupancy/ILP。
8. 低精度和矩阵指令。
9. async copy/pipeline。
10. persistent/warp-specialized/架构特化。

### 10.5 架构特化要有边界

- 以 compute capability dispatch。
- 编译正确的 `sm_XX`，不要只带 PTX fallback 后假设性能一致。
- Blackwell/Hopper 专用路径保留通用 CUDA/CUTLASS/Triton fallback。
- 把架构特化封装在 kernel/dispatch 层，不污染上层算子语义。

---

## 11. Profiler 与指标清单

### 11.1 CUDA：Nsight Systems 先，Nsight Compute 后

Nsight Systems 回答：

- GPU 是否空闲。
- kernel 是否太碎。
- stream 是否重叠。
- memcpy 和 kernel 是否串行。
- CPU 提交线程是否阻塞。
- graph replay 是否减少 API 开销。

Nsight Compute 回答：

- memory throughput 与 transaction。
- cache hit。
- warp stall reason。
- branch efficiency。
- achieved occupancy。
- register/shared usage。
- tensor/core utilization。
- roofline 中属于 memory-bound 还是 compute-bound。

示例：

```bash
nsys profile -t cuda,nvtx,osrt -o timeline python your_benchmark.py
ncu --set full -o kernel_report python your_benchmark.py
```

指标名称会随 GPU 架构和 Nsight 版本变化，优先使用当前 Nsight 文档与 `ncu --query-metrics`，不要从旧文章复制 metric 名称。

### 11.2 Android：APA/Perfetto/AGI

截至核验日期，Android 官方建议新的系统级采集优先考虑 Android Performance Analyzer；AGI 仍适合 Vulkan frame profiling 与支持设备上的相关分析。

使用层次：

1. OpenCL event timestamp：单 kernel/device 时间。
2. 自己的 NVTX 类似标记或日志：阶段边界。
3. APA/Perfetto：CPU 线程、调度、频率、系统 timeline。
4. AGI：Vulkan frame/queue/command 分析。
5. Qualcomm 工具：在设备和版本支持时读取 Adreno counter。

OpenCL 计算不应依赖只有图形 frame 才有意义的指标。

### 11.3 移动端必须额外记录

- battery/skin/GPU temperature。
- cooling device state。
- 充电状态。
- 屏幕与刷新率。
- 前后台状态。
- 是否同时运行 UI/Vulkan/OpenGL workload。
- 测试持续时间。

---

## 12. 当前仓库的推荐阅读与练习顺序

### 第一层：马上能跑

1. `code/core/benchmark/adreno_minimal.py`
2. `code/core/benchmark/adreno_minimal_opencl.cpp`
3. `code/ch01/compare_adreno_minimal.py`
4. `code/ch05/compare_adreno_minimal.py`
5. `code/ch06/compare_adreno_minimal.py`
6. `code/ch11/compare_adreno_minimal.py`

### 第二层：CUDA 核心优化

1. `code/ch06/cuda_extensions/coalescing_kernels.cu`
2. `code/core/benchmark/cuda/memory_patterns_extension.cu`
3. `code/core/profiling/cuda/occupancy_extension.cu`
4. `code/ch06/optimized_ilp_low_occupancy_vec4_impl.cuh`
5. `code/ch11/baseline_gemm_streams.py`
6. `code/ch11/optimized_gemm_streams.py`
7. `code/ch12/baseline_cuda_graphs.cu`
8. `code/ch12/optimized_cuda_graphs.cu`

### 第三层：Triton 与编译器

1. `code/ch14/triton_examples.py`
2. `code/ch14/baseline_triton_persistent.py`
3. `code/ch14/optimized_triton_persistent.py`
4. `code/labs/occupancy_tuning/triton_matmul.py`
5. `code/labs/occupancy_tuning/triton_matmul_schedules.py`

### 第四层：Attention 与推理

1. `code/ch14/flash_attention_sdpa_bench.py`
2. `code/ch16/baseline_dense_attention_flash.py`
3. `code/ch16/optimized_dense_attention_flash.py`
4. `code/ch16/baseline_flashinfer_block_sparse.py`
5. `code/ch16/optimized_flashinfer_block_sparse.py`
6. `code/ch13/baseline_kv_cache_naive.py`
7. `code/ch13/optimized_kv_cache_naive_flash_blockwise.py`
8. `code/ch19/optimized_dynamic_quantized_cache_coalesced.py`

### 配套文档

- `docs/chapter-optimization-methods.md`
- `docs/tooling-and-profiling.md`
- `code/README.md`

---

## 13. 优秀外部代码与文档：按优先级使用

版本号只是 2026-08-02 的核验快照；训练项目应固定 release/tag/commit，避免依赖 `main` 的每日变化。

### 13.1 CUDA 基础与工具

1. NVIDIA CUDA C++ Programming Guide  
   https://docs.nvidia.com/cuda/cuda-programming-guide/
2. NVIDIA CUDA C++ Best Practices Guide  
   https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
3. NVIDIA CUDA Samples，核验时 README 对应 CUDA 13.2  
   https://github.com/NVIDIA/cuda-samples
4. NVIDIA Nsight Compute Documentation  
   https://docs.nvidia.com/nsight-compute/
5. NVIDIA Nsight Systems Documentation  
   https://docs.nvidia.com/nsight-systems/

推荐阅读顺序：Programming Guide 的编程模型与 memory hierarchy -> Best Practices 的 bandwidth/coalescing/occupancy -> Samples 的 deviceQuery、bandwidth、transpose、reduction -> Nsight。

### 13.2 CUTLASS / CuTe

1. CUTLASS 官方仓库  
   https://github.com/NVIDIA/cutlass
2. CUTLASS 文档，核验时显示 4.5.2  
   https://docs.nvidia.com/cutlass/latest/

建议先读 `examples/00_basic_gemm`，再进入 CuTe layout、pipeline、collective 和架构特化。

### 13.3 OpenCL 与 Qualcomm Adreno

1. Khronos OpenCL Registry；最新规范核验为 OpenCL 3.1，2026-05-22 发布  
   https://registry.khronos.org/OpenCL/
2. Khronos OpenCL Guide  
   https://github.com/KhronosGroup/OpenCL-Guide
3. Khronos OpenCL SDK 与 Samples  
   https://github.com/KhronosGroup/OpenCL-SDK
4. Qualcomm Snapdragon Mobile Platform OpenCL General Programming and Optimization  
   https://docs.qualcomm.com/bundle/publicresource/80-NB295-11_REV_C_Qualcomm_Snapdragon_Mobile_Platform_Opencl_General_Programming_and_Optimization.pdf
5. Qualcomm Adreno GPU SDK  
   https://developer.qualcomm.com/software/adreno-gpu-sdk
6. Khronos `cl_khr_command_buffer`  
   https://registry.khronos.org/OpenCL/sdk/3.0/docs/man/html/cl_khr_command_buffer.html

Qualcomm PDF 较早，但其中的向量化、image、访问模式、work-group 和移动端内存注意事项仍可作为实验假设；最终结论必须由当前 Adreno 840 驱动实测决定。

### 13.4 Android 与 Vulkan Compute 工具

1. Android Performance Analyzer，Android 官方新的系统性能分析工具  
   https://developer.android.com/android-performance-analyzer
2. Android GPU Inspector  
   https://developer.android.com/agi
3. Android Vulkan tools and advanced features  
   https://developer.android.com/games/develop/vulkan/tools-and-advanced-features
4. Khronos Vulkan Samples  
   https://github.com/KhronosGroup/Vulkan-Samples

如果 OpenCL 路径缺少足够 profiler counter，可用 Vulkan compute 重做相同 microbenchmark，以获得更标准化的 Android/Vulkan 工具链体验。

### 13.5 Triton / TileLang

1. Triton 官方仓库  
   https://github.com/triton-lang/triton
2. Triton 官方教程，核验时文档版本 3.6.0  
   https://triton-lang.org/main/getting-started/tutorials/
3. TileLang 官方仓库  
   https://github.com/tile-ai/tilelang
4. TileLang 文档，核验时文档版本 0.1.12  
   https://www.tilelang.com/

### 13.6 FlashAttention / FlashInfer / Transformer Engine

1. FlashAttention 官方仓库，包含 FlashAttention-2/3 与 Blackwell FlashAttention-4 路线  
   https://github.com/Dao-AILab/flash-attention
2. FlashAttention-4 论文  
   https://arxiv.org/abs/2603.05451
3. FlashInfer 官方仓库  
   https://github.com/flashinfer-ai/flashinfer
4. FlashInfer 文档，核验时版本 0.6.15  
   https://docs.flashinfer.ai/
5. NVIDIA Transformer Engine 官方仓库  
   https://github.com/NVIDIA/TransformerEngine
6. Transformer Engine 文档，核验时版本 2.16.0  
   https://docs.nvidia.com/deeplearning/transformer-engine/

### 13.7 移动推理后端

1. llama.cpp；当前仓库 Adreno xmem kernel 已直接引用本地 llama.cpp OpenCL kernel  
   https://github.com/ggml-org/llama.cpp
2. ncnn，成熟的移动端 Vulkan 推理实现  
   https://github.com/Tencent/ncnn
3. MNN，包含 OpenCL/Vulkan/移动端推理后端  
   https://github.com/alibaba/MNN
4. Apache TVM，支持 OpenCL/Vulkan/Android 部署与自动调优  
   https://tvm.apache.org/docs/arch/runtime.html

阅读这些项目时，不要从顶层框架开始。优先定位：

```text
device capability
allocator/memory pool
program/pipeline cache
kernel source
shape/dtype dispatch
autotune
fallback
benchmark/test
```

---

## 14. CUDA 环境可用后的第一天命令

先跑官方样例：

```bash
git clone https://github.com/NVIDIA/cuda-samples.git
cd cuda-samples
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

find build -type f -name deviceQuery -executable -print -exec {} \;
find build -type f -name bandwidthTest -executable -print -exec {} \;
```

再进入当前仓库：

```bash
cd /media/code/tools/ai-performance-engineering/code
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements_latest.txt

python -m cli.aisp bench list-targets --chapter ch01
python -m cli.aisp bench run --targets ch01 --profile minimal
```

第一周不要直接跑全量 suite。优先选择一个短 kernel，依次使用：

```bash
nsys profile -t cuda,nvtx,osrt -o timeline python your_case.py
ncu --set full -o kernel_report python your_case.py
```

如果仓库目标依赖较新的 CUDA/PyTorch/Blackwell 环境，而你的 NVIDIA GPU 较旧，先用 CUDA Samples 和独立 `.cu` 练习建立基础，再按 `code/README.md` 的 portable/validity 说明选择兼容 target。

---

## 15. 常见错误清单

1. 只报告 speedup，不报告绝对时间。
2. baseline 和 optimized 工作量不同，却声称是某个单一技术的收益。
3. 未预热。
4. 用未同步 CPU timer 测异步 GPU kernel。
5. 把 JIT、allocation、prepack 混进稳态 kernel 时间，或反过来故意隐藏生产必付成本。
6. 只测一个 shape。
7. 不测 tail、非对齐和极小尺寸。
8. 盲目追求 occupancy。
9. 认为 `float4/half8` 一定更快。
10. 认为 local/shared memory 一定更快。
11. 认为移动端统一内存没有 copy/sync 成本。
12. 每个 kernel 后 `finish/synchronize`，然后抱怨无法 overlap。
13. 只看冷机峰值，不看持续性能。
14. 使用 fast math 却不报告精度变化。
15. 把 OpenCL `local` 当成 CUDA thread-local；它实际对应 CUDA shared。
16. 在 OpenCL/Vulkan 中硬编码 warp=32。
17. 看到某个 QCOM 扩展缺失，就认为所有 QCOM 能力都不可用。
18. 直接复制桌面 GPU tile 参数到手机。
19. 只会调用库，不会解释库为什么快。
20. 只会手写 kernel，不会优先判断标准库是否已经更快、更稳定。
21. 没有 fallback，导致换设备或驱动后直接失败。
22. autotune 不做正确性检查。
23. binary cache 不包含 driver/build option，驱动升级后加载错误缓存。
24. 性能回归只看平均值，不看方差和热状态。
25. 为了跑分固定不现实的输入分布，使生产数据退化。

---

## 16. 专家级毕业项目

完成一个“同一算子语义、双 GPU 后端”的推理算子包：

### 功能

- OpenCL/Adreno 后端。
- CUDA/NVIDIA 后端。
- CPU reference。
- RMSNorm。
- RoPE。
- GEMM 或 fused linear epilogue。
- online softmax。
- paged KV update。
- 简化 attention forward。

### 工程要求

- C++ API。
- Python benchmark wrapper。
- 多 shape/dtype/layout。
- capability/extension dispatch。
- program/kernel cache。
- memory pool。
- autotune cache。
- fallback。
- correctness tests。
- p50/p90/sustained benchmark。
- profiler artifact。
- JSON/Markdown 报告生成。

### 性能报告必须回答

1. 每个算子是 launch-bound、memory-bound 还是 compute-bound？
2. CUDA 与 Adreno 最优 tile 为什么不同？
3. 哪些优化在两端都有效？
4. 哪些 CUDA 技术在 Adreno 上没有直接对应？
5. 哪些移动端优化依赖 UMA、image 或 QCOM extension？
6. 冷机与持续运行差多少？
7. 对哪些 shape 使用自定义 kernel，哪些回退到库？
8. 精度和性能如何权衡？
9. 驱动升级后哪些 cache 需要失效？
10. 如何防止未来提交造成性能回归？

完成这个项目并能够独立回答上述问题，已经具备高性能 GPU 算子岗位所需的核心能力；后续成长重点将从“学技巧”转向“研究新架构、设计调度、维护生产算子库”。

---

## 17. 每次实验的最小报告模板

```markdown
# Experiment: <name>

## Hypothesis
只写一个主要假设。

## Environment
- device:
- driver:
- backend/API:
- compiler/build options:
- temperature/charging/display:

## Workload
- op:
- shape:
- dtype:
- layout:
- logical bytes/FLOPs:

## Baseline
- algorithm:
- launch count:
- temporary bytes:

## Change
- one primary change:

## Correctness
- reference:
- max abs error:
- max rel error:
- edge cases:

## Timing
- warmup:
- iterations:
- groups:
- device-event p50/p90:
- end-to-end p50/p90:
- sustained duration:

## Profiler Evidence
- bottleneck before:
- bottleneck after:
- register/local/shared:
- bandwidth/FLOP utilization:

## Result
- absolute delta:
- speedup:
- valid shapes:
- regressions:

## Explanation
为什么变快或变慢。

## Next Experiment
只列一个主要变量。
```

---

## 18. 最后的学习原则

- **先建立可靠 baseline，再优化。**
- **先处理数据移动和 launch，再处理复杂指令级技巧。**
- **先用库上限判断自定义 kernel 是否值得。**
- **每个性能结论都需要正确性、绝对时间和 profiler 证据。**
- **CUDA 与移动 GPU 的原理高度相似，但参数和硬件特性不能照搬。**
- **FlashAttention、Triton、TileLang、CUTLASS 都是硬件理解的放大器，不是替代品。**
- **真正的专家交付的是稳定算子库、可解释性能和持续回归能力，而不是一次漂亮跑分。**
