"""Tiny CPU-only comparison path for Chapter 1."""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass

import torch

from ch01.baseline_cpu_minimal import get_benchmark as get_baseline_benchmark
from ch01.optimized_cpu_minimal import get_benchmark as get_optimized_benchmark


@dataclass(frozen=True)
class TimedResult:
    name: str
    mean_ms: float
    raw_ms: list[float]


def _time_benchmark(name: str, *, iterations: int = 5, warmup: int = 1) -> TimedResult:
    benchmark = (
        get_baseline_benchmark() if name == "baseline" else get_optimized_benchmark()
    )
    benchmark.setup()
    try:
        for _ in range(warmup):
            benchmark.benchmark_fn()
        times_ms = []
        for _ in range(iterations):
            start = time.perf_counter()
            benchmark.benchmark_fn()
            times_ms.append((time.perf_counter() - start) * 1000.0)
        validation_error = benchmark.validate_result()
        if validation_error:
            raise RuntimeError(validation_error)
        return TimedResult(name=name, mean_ms=statistics.mean(times_ms), raw_ms=times_ms)
    finally:
        benchmark.teardown()


def profile() -> dict:
    """Run the minimal CPU comparison and return machine-readable metrics."""
    baseline = _time_benchmark("baseline")
    optimized = _time_benchmark("optimized")
    speedup = baseline.mean_ms / optimized.mean_ms if optimized.mean_ms else 0.0
    return {
        "metrics": {
            "chapter": "ch01",
            "target": "cpu_minimal",
            "device": "cpu",
            "pytorch_version": torch.__version__,
            "baseline_ms": baseline.mean_ms,
            "optimized_ms": optimized.mean_ms,
            "speedup": speedup,
            "baseline_raw_ms": baseline.raw_ms,
            "optimized_raw_ms": optimized.raw_ms,
        }
    }


if __name__ == "__main__":
    result = profile()
    metrics = result["metrics"]
    print("Chapter 1 CPU minimal comparison")
    print(f"  Baseline scalar loop: {metrics['baseline_ms']:.3f} ms")
    print(f"  Optimized addmm:      {metrics['optimized_ms']:.3f} ms")
    print(f"  Speedup:             {metrics['speedup']:.2f}x")
