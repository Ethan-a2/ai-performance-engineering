"""CPU-only baseline for the Chapter 1 minimal quick start."""

from __future__ import annotations

from ch01.cpu_minimal_common import CpuMinimalBenchmark
from core.harness.benchmark_harness import BaseBenchmark


def get_benchmark() -> BaseBenchmark:
    return CpuMinimalBenchmark(vectorized=False)
