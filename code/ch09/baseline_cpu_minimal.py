"""CPU-only baseline for the CH09 minimal quick start."""

from __future__ import annotations

from core.benchmark.cpu_minimal import get_cpu_minimal_benchmark
from core.harness.benchmark_harness import BaseBenchmark


def get_benchmark() -> BaseBenchmark:
    return get_cpu_minimal_benchmark("ch09", vectorized=False)
