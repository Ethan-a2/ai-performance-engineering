# GPU Minimal 版本与默认 CUDA 版本各章节优化项差异对比

> 日期：2026-07-07  
> 范围：`ch01`–`ch20`  
> 口径：默认 CUDA 版本指各章 `README.md` 中的 canonical CUDA / PyTorch / Triton 主线；GPU minimal 版本指 `chXX/compare_gpu_minimal.py` 调用的 RTX-2060-safe PyTorch CUDA minimal 路径。

## 结论摘要

- **默认 CUDA 版本**是完整章节主线，覆盖 cuBLAS、Triton/CUTLASS、Tensor Core、TMA、CUDA Graph、streams、NCCL/NVLink、`torch.compile`、Flash SDP、KV cache、serving scheduler、低精度等系统级与 kernel 级优化。
- **GPU minimal 版本**是便携教学 CUDA 路径，把每章思想压缩到 4 类 PyTorch CUDA 场景：`torch_gemm`、`pipeline_fusion`、`copy_vectorized`、`kv_block`。
- **二者都是 CUDA，但不是同一个层级**：默认 CUDA 版本多为章节原 workload 和 publish-grade/canonical 目标；GPU minimal 版本是 Turing/RTX 2060 可跑的教学微基准，用来说明同类第一性原理，而不是完整替代 Blackwell/Grace/Triton/CUTLASS 等主线。
- **GPU minimal 的大 speedup 多来自极弱 baseline**：例如 looped small GEMMs → `torch.bmm`、chunked copy loop → whole-tensor copy、per-token KV loop → block update；这些数字不能直接代表默认 CUDA 章节真实优化幅度。
- **`pipeline_fusion` 在 GPU minimal 上接近 1x**：这很重要，说明普通 PyTorch 表达式未必真正融合；没有 `torch.compile`/Triton/kernel fusion 时，“写成一行”不等于 runtime 一定少 kernel 或少内存流量。

## 源码入口

| 路径 | 作用 |
| --- | --- |
| `chXX/README.md` | 默认 CUDA / PyTorch / Triton 章节主线和 measured delta |
| `chXX/compare_gpu_minimal.py` | 每章 GPU minimal 入口 |
| `core/benchmark/gpu_minimal.py` | 章节到 GPU minimal 场景的映射、PyTorch CUDA 计时、正确性检查 |
| `docs/chapter_backend_optimization_map.md` | 已生成的多后端优化知识地图与指标对照 |
| `docs/_generated/gpu_minimal_rtx2060_metrics.json` | RTX 2060 GPU minimal 采集指标 |

## 采集环境与口径限制

| 项目 | 值 |
| --- | --- |
| GPU | `NVIDIA GeForce RTX 2060` |
| Compute capability | `sm_75` |
| PyTorch | `2.12.1+cu129` |
| 数据口径 | `ssh mi` 上的标准 `gpu_minimal` helper |
| 有效性说明 | 非 canonical；未作为发布级锁频/完整 provenance 性能结论，只作为便携教学对照 |

## GPU Minimal 四类优化模型

| GPU minimal 场景 | Baseline | Optimized | 覆盖章节 | 捕捉的优化思想 |
| --- | --- | --- | --- | --- |
| `torch_gemm` | Python loop 逐个小矩阵 `a[index] @ b[index]` | 单次 `torch.bmm(a, b)` | CH01、CH02、CH03、CH10、CH14、CH16 | launch amortization、batched GEMM、把许多小任务合成一个大任务 |
| `pipeline_fusion` | `torch.mul` → `torch.add` → `torch.clamp` 三段 tensor op | 一个 PyTorch 表达式 `torch.clamp(x * 1.25 + 0.5)` | CH04、CH06、CH08、CH09、CH20 | 表达式级 fusion 类比；实际 PyTorch eager 下未必真正 fusion |
| `copy_vectorized` | 按 chunk 循环 `copy_` | 单次 whole-tensor `copy_` | CH05、CH07、CH19 | bulk memory movement、减少切片循环和 launch/dispatch 固定成本 |
| `kv_block` | per-token loop：逐行 `kv[index].add_(...)` | block tensor update：`kv2.add_(update * scale)` | CH11、CH12、CH13、CH15、CH17、CH18 | KV/cache block 化、减少 per-token 提交、批量更新 |

## GPU Minimal 每章实测

| 章节 | GPU minimal 场景 | Baseline | Optimized | Speedup | Max Error |
| --- | --- | ---: | ---: | ---: | ---: |
| CH01 | `torch_gemm` | 2.036330 ms | 0.029781 ms | 68.38x | 0 |
| CH02 | `torch_gemm` | 2.042666 ms | 0.030434 ms | 67.12x | 0 |
| CH03 | `torch_gemm` | 2.023883 ms | 0.030102 ms | 67.23x | 0 |
| CH04 | `pipeline_fusion` | 0.098804 ms | 0.100959 ms | 0.98x | 0 |
| CH05 | `copy_vectorized` | 10.845355 ms | 0.120022 ms | 90.36x | 0 |
| CH06 | `pipeline_fusion` | 0.101515 ms | 0.100858 ms | 1.01x | 0 |
| CH07 | `copy_vectorized` | 10.616170 ms | 0.120862 ms | 87.84x | 0 |
| CH08 | `pipeline_fusion` | 0.103693 ms | 0.101405 ms | 1.02x | 0 |
| CH09 | `pipeline_fusion` | 0.098741 ms | 0.100383 ms | 0.98x | 0 |
| CH10 | `torch_gemm` | 2.038838 ms | 0.029438 ms | 69.26x | 0 |
| CH11 | `kv_block` | 1.240781 ms | 0.028814 ms | 43.06x | 0 |
| CH12 | `kv_block` | 1.212330 ms | 0.029141 ms | 41.60x | 0 |
| CH13 | `kv_block` | 1.233930 ms | 0.033243 ms | 37.12x | 0 |
| CH14 | `torch_gemm` | 2.028454 ms | 0.029459 ms | 68.86x | 0 |
| CH15 | `kv_block` | 1.220435 ms | 0.029156 ms | 41.86x | 0 |
| CH16 | `torch_gemm` | 2.029125 ms | 0.029571 ms | 68.62x | 0 |
| CH17 | `kv_block` | 1.224431 ms | 0.028960 ms | 42.28x | 0 |
| CH18 | `kv_block` | 1.230403 ms | 0.029937 ms | 41.10x | 0 |
| CH19 | `copy_vectorized` | 10.781581 ms | 0.120570 ms | 89.42x | 0 |
| CH20 | `pipeline_fusion` | 0.103777 ms | 0.101018 ms | 1.03x | 0 |

## 逐章差异详表

| 章节 | 默认 CUDA 版本优化项 | GPU minimal 版本优化项 | 核心差异 |
| --- | --- | --- | --- |
| CH01 | `gemm` 29.51x：strided/batched GEMM 减 launch；`performance` 4.82x：FP16 + fused microbatch；`nvfp4_mlp` 主要降内存 | `torch_gemm` 68.38x：looped small GEMMs → `torch.bmm` | 二者都体现 batching/launch amortization，但默认 CUDA 还覆盖训练 loop、FP16、fusion、NVFP4 memory tradeoff；GPU minimal 只是 PyTorch eager 下小 GEMM 批处理样例 |
| CH02 | Grace coherent memory 23.14x、memory transfer 5.20x、cuBLAS tuning 5.17x | `torch_gemm` 67.12x：小 GEMM loop → batched matmul | 默认 CUDA 是硬件拓扑、coherent memory、链路与 cuBLAS 参数；GPU minimal 只保留“硬件上更合适的 batched GEMM 调用”，没有 Grace/NVLink/coherent memory 语义 |
| CH03 | pinned prefetch 3.64x、host/runtime GEMM 2.90x、double-buffer provisioning 1.61x | `torch_gemm` 67.23x：减少 Python loop 与多次 matmul dispatch | 默认 CUDA 重点是 host staging、prefetch、双缓冲和 provision；GPU minimal 只展示 launch/dispatch 固定成本，不覆盖 NUMA、container、host-device pipeline |
| CH04 | gradient fusion 68.83x、DataParallel overhead removal 7.86x、locality 16.01x、bandwidth cleanup 1.75x | `pipeline_fusion` 0.98x：三段 PyTorch tensor op → 一个表达式 | 默认 CUDA 是多 GPU 通信、collective、gradient fusion 和拓扑；GPU minimal 的 eager 表达式几乎不赢，说明这不是 NCCL/communication fusion 的替代物 |
| CH05 | preprocessing vectorization 72.64x、storage CPU 2.07x | `copy_vectorized` 90.36x：chunked copy loop → single bulk copy | 默认 CUDA/章节主线偏数据预处理、storage、worker/prefetch；GPU minimal 是设备 tensor copy 的固定成本极简演示，不覆盖文件系统、CPU preprocessing 或 dataloader |
| CH06 | true CUDA add 3881.04x、attention ILP 265.82x、autotuning 3.92x | `pipeline_fusion` 1.01x：三段 elementwise → 一个表达式 | 默认 CUDA 是 kernel 编程、ILP、occupancy、launch bounds、autotune；GPU minimal 没有自定义 CUDA kernel 或 ILP 调度，只说明 PyTorch eager 表达式不等于真实 kernel 优化 |
| CH07 | TMA 2D bulk copy 3.44x、lookup locality 45.41x、shared-memory tiled matmul 3.18x | `copy_vectorized` 87.84x：chunked copy → whole tensor copy | 默认 CUDA 覆盖 TMA、tensor map、shared memory、matmul tiling；GPU minimal 只是减少 copy 分块和 dispatch，没有 TMA/shared-memory pipeline |
| CH08 | predication threshold 10.19x、loop unrolling 4.17x、occupancy-aware scheduling 2.68x | `pipeline_fusion` 1.02x：elementwise 表达式级组合 | 默认 CUDA 优化 warp divergence、unroll、occupancy；GPU minimal 没有 warp 级控制，也没有 custom kernel 分支消除，只提供“减少中间 op”的浅层类比 |
| CH09 | CUTLASS GEMM 3.95x、memory-bound 17.05x、SDPA attention 1.71x | `pipeline_fusion` 0.98x：三段 tensor op → 一个表达式 | 默认 CUDA 是 CUTLASS/Triton/custom kernel、memory-bound schedule、SDPA；GPU minimal 不覆盖 GEMM/attention schedule，且 fusion 类比在 eager 下没有稳定收益 |
| CH10 | warp-specialized / persistent kernels：single CTA 71.42x、batch 54.44x | `torch_gemm` 69.26x：小 GEMM loop → `torch.bmm` | 默认 CUDA 强依赖 warp specialization、persistent kernel、TMA producer-consumer；GPU minimal 只有高层 PyTorch batched GEMM，没有 warp specialization 或 persistent pipeline |
| CH11 | streams overlap 1.86x、stream-ordered KV cache 1.50x、multistream warp specialization 1.67x | `kv_block` 43.06x：per-token row update → block vectorized update | 默认 CUDA 是 stream 并发、有序缓存、多流重叠；GPU minimal 是把逐 token Python/PyTorch loop 合成一个 tensor op，主要减少 dispatch 固定成本 |
| CH12 | CUDA Graph replay 4.21x、kernel fusion 2.72x、GPU-resident work queue 4.75x | `kv_block` 41.60x：per-token update → block update | 默认 CUDA Graph 解决 steady-state repeated launch；GPU minimal 没有 graph capture/replay，用 block tensor op 类比“提交次数少”，不能代表 CUDA Graph 或 work queue |
| CH13 | paged KV 降内存 68.98%、memory profiling、optimized autograd 8.04x、Transformer Engine FP8 5.17x | `kv_block` 37.12x：per-token KV row update → batched update | 默认 CUDA 覆盖 allocator、paged KV、autograd、FP8；GPU minimal 只覆盖 KV 更新粒度，不衡量 memory footprint、allocator fragmentation 或 TE FP8 |
| CH14 | `torch.compile` + reduced precision 3.74x、regional Triton 2.25x、persistent Triton 9.68x | `torch_gemm` 68.86x：looped GEMMs → `torch.bmm` | 默认 CUDA 是 compiler/Triton/persistent kernel；GPU minimal 是 PyTorch eager library op，不覆盖 `torch.compile` 区域编译或 Triton persistent 调度 |
| CH15 | continuous batching 4.16x、NVLink pooled KV 6.11x、guided decoding 5.96x、greedy/spec decode | `kv_block` 41.86x：per-token KV update → block update | 默认 CUDA 是 serving scheduler、跨 GPU KV pool、decode policy；GPU minimal 只展示 KV update batching，不覆盖 queueing、NVLink pool、sampler 或 speculative decoding |
| CH16 | Flash SDP 1.63x、FlashInfer block-sparse 3.94x、runtime scheduler 1.78x | `torch_gemm` 68.62x：batched GEMM microbenchmark | 默认 CUDA 是 production inference attention backend 和 scheduler；GPU minimal 的 GEMM batching 不能代表 FlashAttention/FlashInfer/block-sparse attention 或 runtime scheduler |
| CH17 | static routing 7.07x、MoE topology routing 5.07x、prefill/decode disagg TTFT 2.85x | `kv_block` 42.28x：KV handoff/update block 化 | 默认 CUDA 是路由策略、MoE placement、prefill/decode 服务拆分；GPU minimal 只表达数据块更新，不覆盖策略层、拓扑感知或 TTFT orchestration |
| CH18 | FlexDecoding active window 1.97x、EOS early exit 4.71x、tensor cores 15.65x、RoPE/Q cache 23.53x | `kv_block` 41.10x：per-token update → active-window 类比的 block update | 默认 CUDA 是 decode 算法、early exit、Tensor Core、RoPE/Q cache；GPU minimal 只覆盖 KV row/block update，不覆盖 attention math、Tensor Core 或 early-exit 控制流 |
| CH19 | dynamic quantized cache 1.13x、memory double buffering 1.97x、MXFP8 MoE 7.71x | `copy_vectorized` 89.42x：chunked copy loop → bulk copy | 默认 CUDA 是低精度、量化 cache、double buffer、MoE；GPU minimal 只是 bulk memory movement，不覆盖 FP8/MXFP8 数值策略或 quantized cache |
| CH20 | integrated KV cache 7.07x、BF16 MLP 2.63x，端到端组合验证 | `pipeline_fusion` 1.03x：三段 tensor op → 一个表达式 | 默认 CUDA 是多优化组合后的 E2E 路径；GPU minimal 的 elementwise 表达式几乎持平，只能作为组合优化的最小反例/提醒：组合是否有效必须测 |

## 按优化类型归类

| 优化类型 | 默认 CUDA 版本 | GPU minimal 版本 | 差异判断 |
| --- | --- | --- | --- |
| GEMM / matmul | cuBLAS、Triton/CUTLASS、Tensor Core、shared memory、TMA、persistent kernel | `torch_gemm`：looped small GEMMs → `torch.bmm` | minimal 能很好说明 batching/launch amortization；不能代表 Tensor Core/TMA/Triton schedule |
| Fusion | CUDA kernel fusion、CUDA Graph、Triton fusion、graph-friendly execution | `pipeline_fusion`：三段 PyTorch op → 一个表达式 | eager PyTorch 下收益接近 1x，说明真正 fusion 需要 compiler/custom kernel/profiler 证据 |
| Memory movement | pinned memory、prefetch、NVLink、coherent memory、allocator、cache footprint、TMA | `copy_vectorized`：chunked copy loop → whole tensor copy | minimal 主要消除 Python/dispatch 分块成本；默认 CUDA 覆盖 host-device/GPU-GPU/allocator 多层级 |
| KV/cache | paged KV、stream-ordered KV、NVLink KV pool、active window、RoPE/Q cache | `kv_block`：per-token row update → block update | minimal 是 KV 更新粒度缩影；默认 CUDA 是完整 cache/attention/serving 系统 |
| 低精度 | FP16/BF16/FP8/NVFP4/MXFP8/Transformer Engine | 主要用 FP16 GEMM 或默认 float tensor，不覆盖完整 dtype policy | minimal 不适合判断低精度策略，只能验证数值等价和基础吞吐方向 |
| 分布式 / 调度 | NCCL/NVLink、DataParallel、streams、runtime scheduler、MoE routing、prefill/decode disagg | 基本无对应；只通过 batching/block update 降提交成本 | minimal 不能证明通信、路由、服务调度类优化 |
| 编译器生态 | `torch.compile`、Triton、CUTLASS、cuBLAS、FlashInfer | PyTorch eager + library ops | minimal 是 portability/teaching layer；默认 CUDA 是 production/canonical optimization layer |

## 容易误读的点

1. **“GPU minimal 也是 CUDA，所以可以替代默认 CUDA”**：不可以。它是 RTX 2060-safe 教学路径，默认 CUDA 章节主线常依赖 Blackwell/Grace/Triton/CUTLASS/TMA/Graph 等能力。
2. **“GPU minimal speedup 更大，所以更先进”**：不一定。`torch_gemm`、`copy_vectorized`、`kv_block` 的 baseline 故意弱，主要暴露固定成本；默认 CUDA 的 baseline 通常已经更接近真实系统。
3. **“写成一个 PyTorch 表达式就是 fusion”**：不一定。`pipeline_fusion` 在 CH04/CH09 上低于 1x，说明 PyTorch eager 可能仍产生多个 kernel 或临时张量。
4. **“KV block 40x 证明 paged KV/cache 优化有 40x”**：不可以。minimal 只是 per-token loop 到 block update；默认 CUDA KV 章节还包含 allocator、memory footprint、attention、stream/order、serving integration。
5. **“RTX 2060 指标可作为 canonical 结论”**：不可以。当前指标是 portable/教学对照，未按 benchmark 稳定性要求锁频、profile、完整 provenance。

## 关键解读

- GPU minimal 的最强项是解释固定成本：多次小 GEMM、多次 chunk copy、多次 per-token update 变成一次 batched/block 操作，收益非常直观。
- GPU minimal 的弱项是解释系统级优化：它不覆盖 NCCL/NVLink、streams、TMA、CUDA Graph capture/replay、Triton persistent、FlashAttention、MoE routing、serving scheduler。
- 对学习者来说，GPU minimal 是“第一性原理的可运行小样本”；默认 CUDA 是“章节主线的工程化/硬件特化实现”。
- 公平比较需要重新设计同 shape、同输入、同精度/误差预算、同端到端边界、同锁频/provenance 的 benchmark；当前 GPU minimal 不应用来替代 canonical README 数字。

## 推荐阅读顺序

1. 先读 `docs/chapter_backend_optimization_map.md`，掌握全后端知识地图和指标口径。
2. 再读每章 `README.md` 的 `Measured Delta` 表，理解默认 CUDA 章节主线。
3. 然后看 `core/benchmark/gpu_minimal.py` 的 `CHAPTER_SCENARIOS` 和四个 builder，确认每章映射到哪类 GPU minimal 场景。
4. 最后看 `docs/_generated/gpu_minimal_rtx2060_metrics.json`，理解 RTX 2060 portable 指标和非 canonical 限制。

