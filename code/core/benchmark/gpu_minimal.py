"""Reusable CUDA GPU minimal benchmark helpers.

This path is intentionally smaller than the canonical chapter CUDA targets.  It
keeps the same optimization themes but uses PyTorch CUDA operations that run on
older teaching GPUs such as an RTX 2060 without requiring Blackwell-only
features, Triton kernels, or locked-clock harness preconditions.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GpuScenario:
    name: str
    summary: str
    iterations: int = 30
    warmup: int = 5
    elements: int = 1 << 20
    batch: int = 64
    m: int = 32
    k: int = 64
    n: int = 32
    chunk: int = 4096
    tokens: int = 64
    width: int = 2048


@dataclass(frozen=True)
class GpuTimedResult:
    mean_ms: float
    min_ms: float
    max_ms: float
    raw_ms: list[float]


CHAPTER_SCENARIOS: dict[str, GpuScenario] = {
    "ch01": GpuScenario("torch_gemm", "looped small GEMMs -> single batched matmul", iterations=20),
    "ch02": GpuScenario("torch_gemm", "looped small GEMMs -> single batched matmul", iterations=20),
    "ch03": GpuScenario("torch_gemm", "looped small GEMMs -> single batched matmul", iterations=20),
    "ch04": GpuScenario("pipeline_fusion", "three separate tensor ops -> one vectorized expression"),
    "ch05": GpuScenario("copy_vectorized", "chunked copy loop -> single bulk device copy", elements=1 << 22),
    "ch06": GpuScenario("pipeline_fusion", "three separate tensor ops -> one vectorized expression"),
    "ch07": GpuScenario("copy_vectorized", "chunked copy loop -> single bulk device copy", elements=1 << 22),
    "ch08": GpuScenario("pipeline_fusion", "three separate tensor ops -> one vectorized expression"),
    "ch09": GpuScenario("pipeline_fusion", "three separate tensor ops -> one vectorized expression"),
    "ch10": GpuScenario("torch_gemm", "looped small GEMMs -> single batched matmul", iterations=20),
    "ch11": GpuScenario("kv_block", "per-token KV row update -> block vectorized update"),
    "ch12": GpuScenario("kv_block", "per-token KV row update -> block vectorized update"),
    "ch13": GpuScenario("kv_block", "per-token KV row update -> block vectorized update"),
    "ch14": GpuScenario("torch_gemm", "looped small GEMMs -> single batched matmul", iterations=20),
    "ch15": GpuScenario("kv_block", "per-token KV row update -> block vectorized update"),
    "ch16": GpuScenario("torch_gemm", "looped small GEMMs -> single batched matmul", iterations=20),
    "ch17": GpuScenario("kv_block", "per-token KV row update -> block vectorized update"),
    "ch18": GpuScenario("kv_block", "per-token KV row update -> block vectorized update"),
    "ch19": GpuScenario("copy_vectorized", "chunked copy loop -> single bulk device copy", elements=1 << 22),
    "ch20": GpuScenario("pipeline_fusion", "three separate tensor ops -> one vectorized expression"),
}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer, got {value!r}") from exc
    if parsed < 0:
        raise RuntimeError(f"{name} must be non-negative, got {parsed}")
    return parsed


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _load_torch() -> Any:
    try:
        import torch  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - depends on host install
        raise RuntimeError(f"SKIPPED: PyTorch is required for GPU minimal benchmarks: {exc}") from exc
    return torch


def _requested_token(argv: Sequence[str] | None, token: str) -> bool:
    args = list(sys.argv[1:] if argv is None else argv)
    for arg in args:
        if token in str(arg).replace(",", " ").split():
            return True
    return False


def requested_gpu_minimal(argv: Sequence[str] | None = None) -> bool:
    return _requested_token(argv, "gpu_minimal")


def _has_explicit_non_gpu_examples(argv: Sequence[str] | None) -> bool:
    args = list(sys.argv[1:] if argv is None else argv)
    if requested_gpu_minimal(args):
        return False
    return any(arg == "--examples" or str(arg).startswith("--examples=") for arg in args)


def should_use_gpu_minimal(argv: Sequence[str] | None = None) -> bool:
    """Return True when compare.py should use the RTX-2060-safe CUDA path."""
    if requested_gpu_minimal(argv) or _env_bool("AISP_USE_GPU_MINIMAL", False):
        return True
    if _has_explicit_non_gpu_examples(argv) or not _env_bool("AISP_GPU_MINIMAL_AUTO", True):
        return False
    try:
        torch = _load_torch()
        if not torch.cuda.is_available():
            return False
        major, minor = torch.cuda.get_device_capability(0)
    except Exception:
        return False
    max_auto_sm = _env_int("AISP_GPU_MINIMAL_MAX_AUTO_SM", 75)
    return major * 10 + minor <= max_auto_sm


def _skipped_metrics(chapter: str, reason: str) -> dict[str, Any]:
    return {
        "metrics": {
            "chapter": chapter,
            "target": "gpu_minimal",
            "device": "cuda",
            "status": "skipped",
            "reason": reason,
        }
    }


def _time_cuda(torch: Any, fn: Callable[[], Any], *, iterations: int, warmup: int) -> GpuTimedResult:
    if iterations <= 0:
        raise RuntimeError("AISP_GPU_MINIMAL_ITERATIONS must be greater than zero")
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times_ms: list[float] = []
    for _ in range(iterations):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times_ms.append(float(start.elapsed_time(end)))
    return GpuTimedResult(
        mean_ms=sum(times_ms) / len(times_ms),
        min_ms=min(times_ms),
        max_ms=max(times_ms),
        raw_ms=times_ms,
    )


def _build_torch_gemm(torch: Any, device: Any, scenario: GpuScenario) -> tuple[Callable[[], Any], Callable[[], Any], Callable[[], float], dict[str, int]]:
    a = torch.randn(scenario.batch, scenario.m, scenario.k, device=device, dtype=torch.float16)
    b = torch.randn(scenario.batch, scenario.k, scenario.n, device=device, dtype=torch.float16)
    out_loop = torch.empty(scenario.batch, scenario.m, scenario.n, device=device, dtype=torch.float16)
    out_batch = torch.empty_like(out_loop)

    def baseline() -> Any:
        for index in range(scenario.batch):
            out_loop[index].copy_(a[index] @ b[index])
        return out_loop

    def optimized() -> Any:
        torch.bmm(a, b, out=out_batch)
        return out_batch

    def max_error() -> float:
        return float((out_loop - out_batch).abs().max().item())

    shape = {"batch": scenario.batch, "m": scenario.m, "k": scenario.k, "n": scenario.n}
    return baseline, optimized, max_error, shape


def _build_pipeline_fusion(torch: Any, device: Any, scenario: GpuScenario) -> tuple[Callable[[], Any], Callable[[], Any], Callable[[], float], dict[str, int]]:
    x = torch.randn(scenario.elements, device=device)
    t1 = torch.empty_like(x)
    t2 = torch.empty_like(x)
    out = torch.empty_like(x)
    out2 = torch.empty_like(x)

    def baseline() -> Any:
        torch.mul(x, 1.25, out=t1)
        torch.add(t1, 0.5, out=t2)
        torch.clamp(t2, min=0, out=out)
        return out

    def optimized() -> Any:
        torch.clamp(x * 1.25 + 0.5, min=0, out=out2)
        return out2

    def max_error() -> float:
        return float((out - out2).abs().max().item())

    return baseline, optimized, max_error, {"elements": scenario.elements}


def _build_copy_vectorized(torch: Any, device: Any, scenario: GpuScenario) -> tuple[Callable[[], Any], Callable[[], Any], Callable[[], float], dict[str, int]]:
    src = torch.randn(scenario.elements, device=device)
    dst = torch.empty_like(src)
    dst2 = torch.empty_like(src)

    def baseline() -> Any:
        for offset in range(0, scenario.elements, scenario.chunk):
            dst[offset : offset + scenario.chunk].copy_(src[offset : offset + scenario.chunk])
        return dst

    def optimized() -> Any:
        dst2.copy_(src)
        return dst2

    def max_error() -> float:
        return float((dst - dst2).abs().max().item())

    return baseline, optimized, max_error, {"elements": scenario.elements, "chunk": scenario.chunk}


def _build_kv_block(torch: Any, device: Any, scenario: GpuScenario) -> tuple[Callable[[], Any], Callable[[], Any], Callable[[], float], dict[str, int]]:
    kv = torch.randn(scenario.tokens, scenario.width, device=device)
    kv2 = kv.clone()
    update = torch.randn(scenario.tokens, scenario.width, device=device)
    scale = torch.linspace(0.9, 1.1, scenario.tokens, device=device).view(scenario.tokens, 1)

    def baseline() -> Any:
        for index in range(scenario.tokens):
            kv[index].add_(update[index] * scale[index])
        return kv

    def optimized() -> Any:
        kv2.add_(update * scale)
        return kv2

    def max_error() -> float:
        return float((kv - kv2).abs().max().item())

    return baseline, optimized, max_error, {"tokens": scenario.tokens, "width": scenario.width}


BUILDERS: dict[str, Callable[[Any, Any, GpuScenario], tuple[Callable[[], Any], Callable[[], Any], Callable[[], float], dict[str, int]]]] = {
    "torch_gemm": _build_torch_gemm,
    "pipeline_fusion": _build_pipeline_fusion,
    "copy_vectorized": _build_copy_vectorized,
    "kv_block": _build_kv_block,
}


def run_gpu_minimal_profile(chapter: str) -> dict[str, Any]:
    """Run the chapter's RTX-2060-compatible CUDA minimal scenario."""
    scenario = CHAPTER_SCENARIOS.get(chapter)
    if scenario is None:
        raise RuntimeError(f"No GPU minimal scenario registered for {chapter}")
    try:
        torch = _load_torch()
    except RuntimeError as exc:
        return _skipped_metrics(chapter, str(exc))
    if not torch.cuda.is_available():
        return _skipped_metrics(chapter, "SKIPPED: CUDA is required for GPU minimal benchmarks")

    device_name = os.environ.get("AISP_GPU_MINIMAL_DEVICE", "cuda:0")
    device = torch.device(device_name)
    if device.type != "cuda":
        return _skipped_metrics(chapter, f"SKIPPED: CUDA device is required, got {device_name!r}")
    torch.cuda.set_device(device)
    seed = _env_int("AISP_GPU_MINIMAL_SEED", 42)
    iterations = _env_int("AISP_GPU_MINIMAL_ITERATIONS", scenario.iterations)
    warmup = _env_int("AISP_GPU_MINIMAL_WARMUP", scenario.warmup)
    torch.manual_seed(seed)
    torch.cuda.empty_cache()

    builder = BUILDERS[scenario.name]
    baseline, optimized, max_error, shape = builder(torch, device, scenario)
    baseline()
    optimized()
    torch.cuda.synchronize()
    initial_max_error = max_error()

    baseline, optimized, max_error, shape = builder(torch, device, scenario)
    baseline_result = _time_cuda(torch, baseline, iterations=iterations, warmup=warmup)
    optimized_result = _time_cuda(torch, optimized, iterations=iterations, warmup=warmup)
    baseline()
    optimized()
    torch.cuda.synchronize()
    optimized_max_error = max_error()
    speedup = baseline_result.mean_ms / optimized_result.mean_ms if optimized_result.mean_ms else 0.0

    major, minor = torch.cuda.get_device_capability(device)
    return {
        "metrics": {
            "chapter": chapter,
            "target": "gpu_minimal",
            "device": "cuda",
            "status": "ok",
            "scenario": scenario.name,
            "scenario_summary": scenario.summary,
            "shape": shape,
            "iterations": iterations,
            "warmup": warmup,
            "seed": seed,
            "torch_version": torch.__version__,
            "cuda_available": True,
            "cuda_device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
            "compute_capability": f"sm_{major}{minor}",
            "baseline_ms": baseline_result.mean_ms,
            "optimized_ms": optimized_result.mean_ms,
            "speedup": speedup,
            "baseline_min_ms": baseline_result.min_ms,
            "optimized_min_ms": optimized_result.min_ms,
            "baseline_max_ms": baseline_result.max_ms,
            "optimized_max_ms": optimized_result.max_ms,
            "baseline_raw_ms": baseline_result.raw_ms,
            "optimized_raw_ms": optimized_result.raw_ms,
            "initial_max_abs_err": initial_max_error,
            "optimized_max_abs_err": optimized_max_error,
            "note": "RTX-2060-safe PyTorch CUDA minimal path; portable teaching data unless clocks and host validity are controlled",
        }
    }


def print_gpu_minimal_result(result: dict[str, Any]) -> None:
    metrics = result["metrics"]
    print(f"{metrics['chapter'].upper()} GPU minimal comparison")
    if metrics.get("status") != "ok":
        print(f"  {metrics.get('reason', 'GPU run did not complete')}")
        return
    print(f"  Scenario:              {metrics['scenario']}")
    print(f"  GPU:                   {metrics['gpu_name']} ({metrics['compute_capability']})")
    print(f"  Baseline:              {metrics['baseline_ms']:.6f} ms")
    print(f"  Optimized:             {metrics['optimized_ms']:.6f} ms")
    print(f"  Speedup:               {metrics['speedup']:.2f}x")
    err = metrics.get("optimized_max_abs_err")
    if err is not None:
        print(f"  Optimized max error:   {err:.8f}")
