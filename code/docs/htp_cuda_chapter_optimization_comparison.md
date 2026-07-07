# HTP 版本与默认 CUDA 版本各章节优化项差异对比

> 日期：2026-07-07  
> 范围：`ch01`–`ch20`  
> 口径：默认 CUDA 版本指各章 `README.md` 中的 canonical CUDA / PyTorch / Triton 主线；HTP 版本指 `chXX/compare_htp_minimal.py` 调用的 Hexagon HTP minimal 路径。

## 结论摘要

- **默认 CUDA 版本**是完整章节主线，覆盖 cuBLAS、Triton/CUTLASS、Tensor Core、TMA、CUDA Graph、streams、NCCL/NVLink、`torch.compile`、Flash SDP、KV cache、serving scheduler、低精度等系统级与 kernel 级优化。
- **HTP 版本**是 Hexagon/HTP minimal 类比路径，把每章思想压缩到 4 类 DSP/HVX 场景：`hvx_tile`、`pipeline_fusion`、`copy_vectorized`、`kv_block`。
- **二者不能逐项横比 speedup**：CUDA canonical 是原章节 workload；HTP minimal 是同类思想移植样例，验证“瓶颈模式是否能在 Hexagon HTP/HVX 上表达”，不代表完整移植后的章节性能。
- **最大差异在系统层和执行模型**：CUDA 很多优化发生在 GPU runtime、编译器、分布式、调度、serving policy；HTP 当前主要落在 FastRPC/skel 调用内的固定 shape、HVX vector lane、aligned buffer、低功耗 DSP 数据流。

## 源码入口

| 路径 | 作用 |
| --- | --- |
| `chXX/README.md` | 默认 CUDA / PyTorch / Triton 章节主线和 measured delta |
| `chXX/compare_htp_minimal.py` | 每章 HTP minimal 入口 |
| `core/benchmark/htp_minimal.py` | 章节到 HTP 场景的映射、Android/Hexagon 构建、ADB 推送与运行 |
| `core/benchmark/htp_minimal_project/src/aisp_htp_minimal_imp.c` | HTP 设备侧 skel 实现，包含四类 HVX 场景 |
| `core/benchmark/htp_minimal_project/src/aisp_htp_minimal_host.cpp` | Android host 侧 FastRPC 调用与计时 |
| `docs/chapter_backend_optimization_map.md` | 已生成的多后端优化知识地图与指标对照 |
| `docs/_generated/backend_minimal_metrics.json` | HTP/Adreno/CPU minimal 采集指标 |

## HTP 四类优化模型

| HTP 场景 | Baseline | Optimized | 覆盖章节 | 捕捉的优化思想 |
| --- | --- | --- | --- | --- |
| `hvx_tile` | 标量 `uint32` 循环逐元素加法 | 128B HVX vector load/add/store | CH01、CH02、CH03、CH10、CH14、CH16 | 标量到向量、tile 化、硬件感知 kernel、固定 shape DSP 加速 |
| `pipeline_fusion` | 两段循环 + `tmp` 中间 buffer | 单段 HVX vector fused loop | CH04、CH06、CH08、CH09、CH20 | pipeline fusion、减少中间写回、减少循环 passes、提高局部吞吐 |
| `copy_vectorized` | byte 级 scalar copy | HVX vector copy | CH05、CH07、CH19 | vectorized memory movement、aligned load/store、减少访存循环开销 |
| `kv_block` | scalar nested loop 更新 KV rows | HVX vector row/block update | CH11、CH12、CH13、CH15、CH17、CH18 | KV/cache block 化、固定 shape 批处理、减少碎片化更新 |

## HTP 每章 minimal 实测

| 章节 | HTP 场景 | Baseline | Optimized | Speedup | HVX Bytes | Max Error |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| CH01 | `hvx_tile` | 0.166406 ms | 0.147708 ms | 1.13x | 128 | 0 |
| CH02 | `hvx_tile` | 0.206459 ms | 0.180208 ms | 1.15x | 128 | 0 |
| CH03 | `hvx_tile` | 0.212813 ms | 0.195313 ms | 1.09x | 128 | 0 |
| CH04 | `pipeline_fusion` | 0.212135 ms | 0.202604 ms | 1.05x | 128 | 0 |
| CH05 | `copy_vectorized` | 0.191197 ms | 0.181511 ms | 1.05x | 128 | 0 |
| CH06 | `pipeline_fusion` | 0.220833 ms | 0.191458 ms | 1.15x | 128 | 0 |
| CH07 | `copy_vectorized` | 0.236927 ms | 0.179740 ms | 1.32x | 128 | 0 |
| CH08 | `pipeline_fusion` | 0.225886 ms | 0.189166 ms | 1.19x | 128 | 0 |
| CH09 | `pipeline_fusion` | 0.147969 ms | 0.121771 ms | 1.22x | 128 | 0 |
| CH10 | `hvx_tile` | 0.181302 ms | 0.160781 ms | 1.13x | 128 | 0 |
| CH11 | `kv_block` | 1.926458 ms | 1.596511 ms | 1.21x | 128 | 0 |
| CH12 | `kv_block` | 2.672187 ms | 2.111042 ms | 1.27x | 128 | 0 |
| CH13 | `kv_block` | 3.057031 ms | 1.934531 ms | 1.58x | 128 | 0 |
| CH14 | `hvx_tile` | 0.205782 ms | 0.183490 ms | 1.12x | 128 | 0 |
| CH15 | `kv_block` | 2.529167 ms | 1.886042 ms | 1.34x | 128 | 0 |
| CH16 | `hvx_tile` | 0.222240 ms | 0.175104 ms | 1.27x | 128 | 0 |
| CH17 | `kv_block` | 1.964688 ms | 1.430313 ms | 1.37x | 128 | 0 |
| CH18 | `kv_block` | 2.086146 ms | 1.867031 ms | 1.12x | 128 | 0 |
| CH19 | `copy_vectorized` | 0.195104 ms | 0.185885 ms | 1.05x | 128 | 0 |
| CH20 | `pipeline_fusion` | 0.216615 ms | 0.183802 ms | 1.18x | 128 | 0 |

## 逐章差异详表

| 章节 | 默认 CUDA 版本优化项 | HTP 版本优化项 | 核心差异 |
| --- | --- | --- | --- |
| CH01 | `gemm` 29.51x：strided/batched GEMM 减 launch；`performance` 4.82x：FP16 + fused microbatch；`nvfp4_mlp` 主要降内存 | `hvx_tile` 1.13x：scalar `uint32` loop → HVX vector tile update | CUDA 讲训练循环、FP16、batched GEMM、NVFP4 memory tradeoff；HTP 只保留“标量变向量”的低功耗 DSP tile 类比，不覆盖训练 loop、Tensor Core 或低精度 MLP |
| CH02 | Grace coherent memory 23.14x、memory transfer 5.20x、cuBLAS tuning 5.17x | `hvx_tile` 1.15x：HVX tiled update | CUDA 是平台拓扑、coherent memory、传输链路、cuBLAS 参数；HTP 是固定 shape 内部 vector lane 利用，没有 Grace/NVLink/cuBLAS 对应物 |
| CH03 | pinned prefetch 3.64x、host/runtime GEMM 2.90x、double-buffer provisioning 1.61x | `hvx_tile` 1.09x：较大 HTP tile work | CUDA 优化 host staging、NUMA、prefetch、双缓冲；HTP minimal 主要暴露 FastRPC/skel 固定成本下的 HVX tile 加速，不覆盖主机内存预取链路 |
| CH04 | gradient fusion 68.83x、DataParallel overhead removal 7.86x、locality 16.01x、bandwidth cleanup 1.75x | `pipeline_fusion` 1.05x：两段 HTP loop + tmp → 单 HVX fused loop | CUDA 是多 GPU 通信、collective、framework overhead、拓扑 locality；HTP 只是单 DSP 内 fused loop，能类比“减少中间写回”，不能类比 NCCL/collective |
| CH05 | vectorization 72.64x、storage CPU 2.07x | `copy_vectorized` 1.05x：scalar byte copy → HVX vector copy | CUDA/默认路径偏 Python preprocessing、IO 和 storage 不饿 GPU；HTP 是纯 DSP 内存 copy vectorization，不覆盖 dataloader、filesystem、worker/prefetch |
| CH06 | true CUDA add 3881.04x、attention ILP 265.82x、autotuning 3.92x | `pipeline_fusion` 1.15x：fused HVX vector loop | CUDA 讲 kernel 编程、occupancy、ILP、launch bounds、autotune；HTP 只表达“少循环/少 tmp + HVX vector”，没有 warp、SM occupancy 或 autotuner 搜索 |
| CH07 | TMA bulk tensor 3.44x、lookup locality 45.41x、shared-memory tiled matmul 3.18x | `copy_vectorized` 1.32x：HVX vector memory movement | CUDA 包含 TMA、tensor map、shared memory、matmul tiling；HTP 对应的是 128B HVX 宽搬运，不具备 NVIDIA TMA/shared-memory pipeline 语义 |
| CH08 | predication threshold 10.19x、loop unrolling 4.17x、occupancy-aware scheduling 2.68x | `pipeline_fusion` 1.19x：branch-light fused HVX work | CUDA 优化 warp divergence、unroll、resident work；HTP 只保留 branch-light/fused vector loop，调度单位和分支代价模型完全不同 |
| CH09 | CUTLASS GEMM 3.95x、memory-bound 17.05x、SDPA attention 1.71x | `pipeline_fusion` 1.22x：减少 HTP memory traffic | CUDA 有 CUTLASS/Triton/attention compute-memory balance；HTP 是固定整数向量 pipeline fusion，不能代表 SDPA 或 GEMM library schedule |
| CH10 | warp-specialized / persistent kernels：single CTA 71.42x、batch 54.44x | `hvx_tile` 1.13x：HVX tile specialization | CUDA 强依赖 warp specialization、persistent kernel、TMA producer-consumer；HTP 用 prebuilt skel + HVX tile 类比“专用硬件路径”，没有 warp/TMA/persistent 概念 |
| CH11 | streams overlap 1.86x、stream-ordered KV cache 1.50x、multistream warp specialization 1.67x | `kv_block` 1.21x：scalar KV row update → HVX vector row/block update | CUDA 是 stream 并发和有序缓存语义；HTP 是单 FastRPC 调用内的 DSP 向量化 KV 更新，不能表达 CUDA stream overlap |
| CH12 | CUDA Graph replay 4.21x、kernel fusion 2.72x、GPU-resident work queue 4.75x | `kv_block` 1.27x：block HTP replay 类比固定提交成本下降 | CUDA Graph 解决 repeated eager launch；HTP minimal 每次 host 仍经 FastRPC，主要收益来自 repeats 内的 HVX vector row update，不是 CUDA Graph 等价物 |
| CH13 | paged KV 降内存 68.98%、memory profiling、optimized autograd 8.04x、Transformer Engine FP8 5.17x | `kv_block` 1.58x：cache-aware HVX KV block update | CUDA 覆盖 allocator、paged KV、autograd、FP8；HTP 只覆盖 KV row/block update，不覆盖内存分配器、autograd 或 TE FP8 |
| CH14 | `torch.compile` + reduced precision 3.74x、regional Triton 2.25x、persistent Triton 9.68x | `hvx_tile` 1.12x：prebuilt HTP skel + HVX tile | CUDA 是 graph compiler/Triton persistent；HTP 是离线构建 skel + 固定 C/HVX 函数，编译模型、动态 shape 支持和优化空间完全不同 |
| CH15 | continuous batching 4.16x、NVLink pooled KV 6.11x、guided decoding 5.96x、greedy/spec decode | `kv_block` 1.34x：pooled/cache 类比的 HVX KV block update | CUDA 是 serving scheduler、跨 GPU KV pool、decode policy；HTP 只验证低功耗固定 KV 更新 kernel，不覆盖队列调度和跨设备池化 |
| CH16 | Flash SDP 1.63x、FlashInfer block-sparse 3.94x、runtime scheduler 1.78x | `hvx_tile` 1.27x：HTP skel specialization | CUDA 是生产推理 attention backend 和 scheduler；HTP 是固定 skel 的向量路径，适合端侧小固定算子，不覆盖 FlashAttention/FlashInfer |
| CH17 | static routing 7.07x、MoE topology routing 5.07x、prefill/decode disagg TTFT 2.85x | `kv_block` 1.37x：batched HTP handoff / KV movement reduction | CUDA 是路由策略、MoE placement、prefill/decode 服务拆分；HTP 只对应“handoff 数据块更新”，不包含策略层和多服务 orchestration |
| CH18 | FlexDecoding active window 1.97x、EOS early exit 4.71x、tensor cores 15.65x、RoPE/Q cache 23.53x | `kv_block` 1.12x：active-window 类比的 KV block update | CUDA 是 decode 算法、early exit、Tensor Core、RoPE/Q cache；HTP 只表达 cache-aware row update，不覆盖 attention math 或 Tensor Core kernel |
| CH19 | dynamic quantized cache 1.13x、memory double buffering 1.97x、MXFP8 MoE 7.71x | `copy_vectorized` 1.05x：HVX vector memory movement | CUDA 是低精度、量化、cache、double-buffer；HTP 只验证 vector copy/缓存流量下降，未覆盖 FP8/MXFP8 数值策略 |
| CH20 | integrated KV cache 7.07x、BF16 MLP 2.63x，端到端组合验证 | `pipeline_fusion` 1.18x：composed fused HTP stages | CUDA 是多优化组合后的 E2E 路径；HTP 是融合 stage 的 micro 类比，不证明完整推理/训练流水线组合收益 |

## 按优化类型归类

| 优化类型 | CUDA 默认版本 | HTP 版本 | 差异判断 |
| --- | --- | --- | --- |
| 向量化 / tile | CUDA thread/block、SIMT、Tensor Core、shared memory、TMA | HVX 128B vector lanes、aligned buffers、固定 repeats/elements | HTP 很适合固定形状整数/轻量向量任务；不等于 CUDA Tensor Core GEMM |
| Fusion | CUDA kernel fusion、CUDA Graph、Triton fusion、work queue | 单 FastRPC 内 fused HVX loop，减少 tmp 和 scalar passes | HTP 能表达“减少中间 buffer/循环次数”，但不能表达 CUDA Graph replay 的完整 runtime 语义 |
| Memory movement | pinned memory、prefetch、NVLink、coherent memory、allocator、cache footprint | HVX vector copy、aligned load/store、KV row update | HTP 主要在 DSP 本地 buffer 上优化；CUDA 覆盖 host-device、GPU-GPU、allocator 等更多层 |
| KV/cache | paged KV、NVLink KV pool、active window、RoPE/Q cache、serving-integrated KV | `kv_block` row/block update | HTP 是 KV 更新 kernel 缩影；CUDA 是完整 memory allocator + attention + serving 系统 |
| 低精度 | FP16/BF16/FP8/NVFP4/MXFP8/Transformer Engine | 当前 HTP minimal 基本是 `uint32_t`/`uint8_t` HVX 操作 | HTP 对低功耗定点/量化有潜力，但当前 minimal 没有覆盖 CUDA 低精度主线 |
| 分布式 / 调度 | NCCL/NVLink、DataParallel、streams、runtime scheduler、MoE routing | 无对应，只通过固定 shape/repeats 降低边际成本 | HTP minimal 不能证明分布式或服务调度类 CUDA 优化 |
| 编译器生态 | `torch.compile`、Triton、CUTLASS、cuBLAS、FlashInfer | Hexagon SDK skel + FastRPC + HVX intrinsics | HTP 更偏嵌入式部署链和专用 skel，动态 compiler/autotune 能力不是同一层级 |

## 容易误读的点

1. **不能说 HTP speedup 小，所以 HTP 没价值**：HTP minimal 的 baseline/optimized 都经过相同 FastRPC/host 调用路径，且 workload 是小型教学/验证场景；HTP 的价值经常体现在低功耗、固定 shape、端侧部署和稳定延迟。
2. **不能把 HVX vector tile 等同于 Tensor Core**：HVX 是宽向量 DSP lane，Tensor Core 是 NVIDIA 专用矩阵执行单元；二者适合的算子、数据类型、吞吐模型都不同。
3. **不能把 HTP `kv_block` 等同于 CUDA paged KV cache**：HTP 只做 row/block update；CUDA 章节还包含 allocator、attention、active window、serving scheduler 和跨 GPU KV pool。
4. **不能把 HTP fused loop 等同于 CUDA Graph**：二者都能降低固定成本或中间内存流量，但 CUDA Graph 还有 capture、dependency、replay、runtime scheduling 语义。
5. **不能只看 speedup**：CUDA 章节中有不少 memory footprint、latency budget、scheduler、throughput、低精度误差预算目标；HTP minimal 表格里的 speedup 不覆盖这些系统目标。

## 关键解读

- HTP speedup 普遍是 `1.05x`–`1.58x`，明显小于很多 CUDA canonical 数字，主要因为 HTP minimal 的 baseline/optimized 都经过相同 FastRPC/host 调用路径，且场景是固定 shape 的微型验证用例。
- HTP 优化的核心不是追服务端 GPU 极限吞吐，而是固定 shape、低功耗、低延迟、可部署的 DSP/HVX 路径。
- HTP 与 CUDA 最可比的是“标量到向量”“融合减少中间内存”“cache/KV block 更新”这些第一性原理；最不可比的是 Tensor Core、TMA、NCCL/NVLink、CUDA Graph、Triton/compile、FlashAttention、MoE/serving scheduler。
- 如果要公平比较，需要重新设计同 shape、同输入、同精度/误差预算、同端到端边界的 benchmark；当前 HTP minimal 更适合作为“章节思想在 Hexagon 上的最小落地样例”。

## 推荐阅读顺序

1. 先读 `docs/chapter_backend_optimization_map.md`，掌握全后端知识地图和指标口径。
2. 再读每章 `README.md` 的 `Measured Delta` 表，理解默认 CUDA 章节主线。
3. 然后看 `core/benchmark/htp_minimal.py` 的 `CHAPTER_SCENARIOS`，确认每章映射到哪类 HTP minimal 场景。
4. 最后看 `core/benchmark/htp_minimal_project/src/aisp_htp_minimal_imp.c`，理解 baseline/optimized 在 HVX/skel 层具体差在哪里。

