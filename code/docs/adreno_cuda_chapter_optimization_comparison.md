# Adreno 版本与默认 CUDA 版本各章节优化项差异对比

> 日期：2026-07-07  
> 范围：`ch01`–`ch20`  
> 口径：默认 CUDA 版本指各章 `README.md` 中的 canonical CUDA / PyTorch / Triton 主线；Adreno 版本指 `chXX/compare_adreno_minimal.py` 调用的 OpenCL minimal 路径。

## 结论摘要

- **默认 CUDA 版本**是完整章节主线，覆盖 cuBLAS、Triton/CUTLASS、Tensor Core、TMA、CUDA Graph、streams、NCCL/NVLink、`torch.compile`、Flash SDP、KV cache、serving scheduler、低精度等系统级与 kernel 级优化。
- **Adreno 版本**是移动 GPU minimal 类比路径，把每章思想压缩到 4 类 OpenCL 场景：`xmem_gemm`、`pipeline_fusion`、`copy_vectorized`、`kv_block`。
- **二者不能逐项横比 speedup**：CUDA canonical 是原章节 workload；Adreno minimal 是同类思想移植样例，验证“瓶颈模式是否能在 Adreno/OpenCL 上表达”，不代表完整移植后的章节性能。
- **最大差异在系统层**：CUDA 很多优化发生在 runtime、编译器、分布式、调度、serving policy；Adreno 当前几乎全部落在单设备 OpenCL kernel、enqueue 次数、memory layout 与数据块化。

## 源码入口

| 路径 | 作用 |
| --- | --- |
| `chXX/README.md` | 默认 CUDA / PyTorch / Triton 章节主线和 measured delta |
| `chXX/compare_adreno_minimal.py` | 每章 Adreno minimal 入口 |
| `core/benchmark/adreno_minimal.py` | 章节到 Adreno 场景的映射、Android NDK 构建、ADB 推送与运行 |
| `core/benchmark/adreno_minimal_opencl.cpp` | Adreno OpenCL minimal benchmark 的四类 kernel 与计时逻辑 |
| `docs/chapter_backend_optimization_map.md` | 已生成的多后端优化知识地图与指标对照 |
| `docs/_generated/backend_minimal_metrics.json` | Adreno/HTP/CPU minimal 采集指标 |

## Adreno 四类优化模型

| Adreno 场景 | Baseline | Optimized | 覆盖章节 | 捕捉的优化思想 |
| --- | --- | --- | --- | --- |
| `xmem_gemm` | naive OpenCL GEMM | xmem/image/pack/store GEMM | CH01、CH02、CH03、CH10、CH14、CH16 | GEMM tiling、local/xmem、数据打包、移动 GPU 内存层次 |
| `pipeline_fusion` | `scale` + `bias_relu` 两个 kernel + 临时 buffer | 单 fused kernel | CH04、CH06、CH08、CH09、CH20 | kernel fusion、减少 enqueue、减少中间内存往返 |
| `copy_vectorized` | 两次 scalar copy + 临时 buffer | 一次 `float4` vector copy | CH05、CH07、CH19 | vectorized memory movement、减少访存指令和提交次数 |
| `kv_block` | 逐 token KV update 循环 enqueue | 单 block update kernel | CH11、CH12、CH13、CH15、CH17、CH18 | launch amortization、KV/cache block 化、减少碎片化写入 |

## Adreno 每章 minimal 实测

| 章节 | Adreno 场景 | Baseline | Optimized | Speedup | Max Error |
| --- | --- | ---: | ---: | ---: | ---: |
| CH01 | `xmem_gemm` | 4.903385 ms | 1.358229 ms | 3.61x | 0.00009872 |
| CH02 | `xmem_gemm` | 4.857657 ms | 1.604948 ms | 3.03x | 0.00009872 |
| CH03 | `xmem_gemm` | 4.906875 ms | 1.543906 ms | 3.18x | 0.00009872 |
| CH04 | `pipeline_fusion` | 1.243802 ms | 0.927291 ms | 1.34x | 0.00000000 |
| CH05 | `copy_vectorized` | 2.616093 ms | 1.286562 ms | 2.03x | 0.00000000 |
| CH06 | `pipeline_fusion` | 0.993177 ms | 0.725990 ms | 1.37x | 0.00000000 |
| CH07 | `copy_vectorized` | 2.609427 ms | 1.411563 ms | 1.85x | 0.00000000 |
| CH08 | `pipeline_fusion` | 1.262604 ms | 0.845157 ms | 1.49x | 0.00000000 |
| CH09 | `pipeline_fusion` | 0.916666 ms | 0.710052 ms | 1.29x | 0.00000000 |
| CH10 | `xmem_gemm` | 5.017864 ms | 1.499896 ms | 3.35x | 0.00009872 |
| CH11 | `kv_block` | 5.414583 ms | 0.618854 ms | 8.75x | 0.00000000 |
| CH12 | `kv_block` | 4.326875 ms | 0.388697 ms | 11.13x | 0.00000000 |
| CH13 | `kv_block` | 6.228281 ms | 0.355208 ms | 17.53x | 0.00000000 |
| CH14 | `xmem_gemm` | 5.231771 ms | 1.635937 ms | 3.20x | 0.00009872 |
| CH15 | `kv_block` | 6.952032 ms | 0.609792 ms | 11.40x | 0.00000000 |
| CH16 | `xmem_gemm` | 4.845156 ms | 1.495364 ms | 3.24x | 0.00009872 |
| CH17 | `kv_block` | 5.836511 ms | 0.343073 ms | 17.01x | 0.00000000 |
| CH18 | `kv_block` | 6.465885 ms | 0.383854 ms | 16.84x | 0.00000000 |
| CH19 | `copy_vectorized` | 2.363281 ms | 1.342708 ms | 1.76x | 0.00000000 |
| CH20 | `pipeline_fusion` | 1.210833 ms | 0.741510 ms | 1.63x | 0.00000000 |

## 逐章差异详表

| 章节 | 默认 CUDA 版本优化项 | Adreno 版本优化项 | 核心差异 |
| --- | --- | --- | --- |
| CH01 | `gemm` 29.51x：strided/batched GEMM 减 launch；`performance` 4.82x：FP16 + fused microbatch；`nvfp4_mlp` 主要降内存 | `xmem_gemm` 3.61x：OpenCL xmem GEMM | CUDA 同时讲 precision、fusion、batched launch；Adreno 只保留 GEMM 数据打包/内存层次，不覆盖 FP16 training loop 与 NVFP4 MLP |
| CH02 | Grace coherent memory 23.14x、transfer 5.20x、cuBLAS tuning 5.17x | `xmem_gemm` 3.03x | CUDA 是硬件拓扑、coherent memory、链路与 cuBLAS；Adreno 只对应“硬件感知 GEMM”，没有 Grace/NVLink/cuBLAS 层 |
| CH03 | pinned prefetch 3.64x、host/runtime GEMM 2.90x、double-buffer provisioning 1.61x | `xmem_gemm` 3.18x | CUDA 重点是 NUMA/host staging/双缓冲；Adreno 体现较大 GEMM 的设备侧 provisioning/packing，未覆盖 NUMA、container、cluster 配置 |
| CH04 | gradient fusion 68.83x、DataParallel overhead removal 7.86x、locality 16.01x、communication cleanup 1.75x | `pipeline_fusion` 1.34x | CUDA 是多 GPU 通信、collective、拓扑与重叠；Adreno 仅用单设备 fused kernel 类比“减少通信/中间写回” |
| CH05 | preprocessing vectorization 72.64x、storage CPU 2.07x | `copy_vectorized` 2.03x | CUDA/默认路径偏数据加载、预处理、IO overlap；Adreno 是纯设备内存 copy vectorization，不覆盖 storage pipeline、worker、prefetch |
| CH06 | true CUDA add 3881.04x、attention ILP 265.82x、autotuning 3.92x | `pipeline_fusion` 1.37x | CUDA 讲线程、occupancy、ILP、launch bounds、autotune；Adreno 用融合表达“少 launch/少中间流量”，不等价于 CUDA kernel 微架构优化 |
| CH07 | TMA 2D bulk copy 3.44x、lookup locality 45.41x、shared-memory tiled matmul 3.18x | `copy_vectorized` 1.85x | CUDA 有 TMA/shared memory/tiling；Adreno 只做 `float4` copy，不覆盖 TMA、tensor map、shared-memory tiled matmul |
| CH08 | predication threshold 10.19x、loop unrolling 4.17x、occupancy-aware scheduling 2.68x | `pipeline_fusion` 1.49x | CUDA 是 warp efficiency、分支、unroll、occupancy；Adreno 是 branch-light fused elementwise，保留“少分支/少 kernel”但没有 warp 级调度细节 |
| CH09 | CUTLASS GEMM 3.95x、memory-bound 17.05x、SDPA attention 1.71x | `pipeline_fusion` 1.29x | CUDA 是库/自定义 kernel/attention compute-memory balance；Adreno 只用 fusion 降 DRAM 往返，不覆盖 CUTLASS/Triton/SDPA |
| CH10 | warp-specialized / persistent kernel：single CTA 71.42x、batch 54.44x | `xmem_gemm` 3.35x | CUDA 强依赖 warp specialization、persistent/TMA producer-consumer；Adreno 类比为 xmem GEMM pipeline，没有 CUDA warp/TMA/persistent 语义 |
| CH11 | streams overlap 1.86x、stream-ordered KV 1.50x、multistream warp specialization 1.67x | `kv_block` 8.75x | CUDA 是 stream 并发、有序缓存、多流重叠；Adreno 是把逐 token 提交合成 block update，更多体现 launch amortization |
| CH12 | CUDA Graph replay 4.21x、kernel fusion 2.72x、GPU work queue 4.75x | `kv_block` 11.13x | CUDA 是 graph replay/steady-state launch 消除；Adreno 没有 CUDA Graph，用 block KV 单次 enqueue 类比固定提交成本下降 |
| CH13 | paged KV 降内存 68.98%、memory profiling、optimized autograd 8.04x、TE FP8 5.17x | `kv_block` 17.53x | CUDA 覆盖 allocator、paged KV、autograd、FP8；Adreno 只覆盖 cache-aware block update，不覆盖 autograd/TE/量化策略 |
| CH14 | `torch.compile` + reduced precision 3.74x、regional Triton 2.25x、persistent Triton 9.68x | `xmem_gemm` 3.20x | CUDA 是编译器、Triton fusion/persistent；Adreno 是预编译 OpenCL GEMM + packing，不覆盖 `torch.compile` 动态图/区域编译 |
| CH15 | continuous batching 4.16x、NVLink pooled KV 6.11x、guided decoding 5.96x、greedy/spec decode | `kv_block` 11.40x | CUDA 是 serving scheduler、NVLink KV pool、decode policy；Adreno 只保留 KV block 写入/批处理，不覆盖跨 GPU pool 和解码策略 |
| CH16 | Flash SDP 1.63x、FlashInfer block-sparse 3.94x、runtime scheduler 1.78x | `xmem_gemm` 3.24x | CUDA 是生产推理 backend、attention kernel、scheduler；Adreno 用 GEMM specialization 类比 backend specialization，不覆盖 FlashAttention/FlashInfer |
| CH17 | static routing 7.07x、MoE topology routing 5.07x、prefill/decode disagg TTFT 2.85x | `kv_block` 17.01x | CUDA 是路由策略、MoE、prefill/decode 拆分；Adreno 是 KV handoff/block movement，未覆盖策略层与多服务编排 |
| CH18 | FlexDecoding active window 1.97x、EOS early exit 4.71x、tensor cores 15.65x、RoPE/Q cache 23.53x | `kv_block` 16.84x | CUDA 是 decode 算法、Tensor Core、active window、cache reuse；Adreno 只体现 active-window/cache-aware block update，不含 Tensor Core |
| CH19 | dynamic quantized cache 1.13x、memory double buffering 1.97x、MXFP8 MoE 7.71x | `copy_vectorized` 1.76x | CUDA 是低精度、量化、cache、double-buffer；Adreno 只表达“少搬/宽搬”的内存流量收益，不覆盖 FP8/MXFP8 数值路径 |
| CH20 | integrated KV cache 7.07x、BF16 MLP 2.63x，端到端组合验证 | `pipeline_fusion` 1.63x | CUDA 是多优化组合后的 E2E 收益；Adreno 只用 fused stages 类比组合优化，不证明完整推理流水线组合效果 |

## 按优化类型归类

| 优化类型 | CUDA 默认版本 | Adreno 版本 | 差异判断 |
| --- | --- | --- | --- |
| GEMM / 矩阵乘 | cuBLAS、Triton/CUTLASS、Tensor Core、shared memory、TMA、persistent kernel | `xmem_gemm`：OpenCL xmem、image、packing、store | Adreno 有内存层次/打包类比，但没有 Tensor Core、TMA、cuBLAS/Triton 生态 |
| Fusion / launch amortization | CUDA Graph、kernel fusion、graph-friendly execution、work queue、microbatch fusion | `pipeline_fusion`、`kv_block` | Adreno 能表达“减少 enqueue/减少中间 buffer”，但没有 CUDA Graph replay 和 GPU-resident queue 完整模型 |
| Memory movement | pinned memory、prefetch、double buffer、NVLink、coherent memory、quantized cache | `copy_vectorized`、`xmem_gemm` packing、`kv_block` | Adreno 主要是设备内 local/global/image 内存；CUDA 覆盖 host-device、GPU-GPU、allocator、cache footprint |
| 并发 / streams / overlap | streams、多流、NCCL、DataParallel、distributed communication、scheduler overlap | 通过 block 化减少 serialized enqueue | Adreno 当前不覆盖多 GPU、多 stream、collective 或 serving worker 编排 |
| 低精度 | FP16、BF16、FP8、NVFP4、MXFP8、Transformer Engine | xmem GEMM 使用 FP16 weights / FP32 output，但没有完整 dtype policy | Adreno minimal 有半精度输入形态，但不覆盖训练/推理低精度策略和误差预算体系 |
| Serving / KV cache | paged KV、NVLink KV pool、continuous batching、prefill/decode disaggregation、routing | `kv_block` | Adreno 只抽象 KV 写入块化；CUDA 覆盖端到端 serving 策略和拓扑 |
| 编译器 / autotune | `torch.compile`、regional compilation、Triton persistent、autotuning | OpenCL runtime build + precompiled kernel path | Adreno 更偏静态 OpenCL kernel；没有 PyTorch graph/compiler stack 对应物 |

## 容易误读的点

1. **不能说 Adreno 某章 speedup 比 CUDA 小，所以优化没价值**：Adreno minimal 只覆盖该章一个缩影场景，默认 CUDA 可能有多个不同 target。
2. **不能说 Adreno `kv_block` 比 CUDA KV 优化更强**：`kv_block` baseline 是逐 token enqueue，固定成本很高；CUDA KV cache 章节通常还包含 allocator、attention、serving policy、跨 GPU pool 等复杂因素。
3. **不能把 CUDA Graph 直接等同于 Adreno block update**：二者都降低提交固定成本，但 CUDA Graph 还包含 steady-state replay、捕获约束、图内依赖和 runtime 调度语义。
4. **不能把 `xmem_gemm` 等同于 Tensor Core GEMM**：Adreno xmem 是移动 GPU OpenCL 内存层次与 packing 优化；Tensor Core 是 NVIDIA 专用矩阵执行单元。
5. **不能只看 speedup**：CH01 `nvfp4_mlp` 延迟接近持平但显著降内存，CH13 KV cache 也以 memory footprint 为主要目标；Adreno 表格里的 speedup 不覆盖这些目标。

## 推荐阅读顺序

1. 先读 `docs/chapter_backend_optimization_map.md`，掌握全后端知识地图和指标口径。
2. 再读每章 `README.md` 的 `Measured Delta` 表，理解默认 CUDA 章节主线。
3. 然后看 `core/benchmark/adreno_minimal.py` 的 `CHAPTER_SCENARIOS`，确认每章映射到哪类 Adreno minimal 场景。
4. 最后看 `core/benchmark/adreno_minimal_opencl.cpp`，理解 baseline/optimized 在 OpenCL kernel 层具体差在哪里。

