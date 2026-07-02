"""Chapter 20: Compare baseline vs optimized implementations using formal harness.

Uses the BaseBenchmark - benchmarks provide get_benchmark() function,
harness measures directly (no subprocess, no output parsing).
"""

from pathlib import Path
from typing import Any

from core.benchmark.cpu_minimal import should_use_cpu_minimal
from core.harness.benchmark_harness import (
    BenchmarkConfig,
)
from core.utils.chapter_compare_template import (
    profile_template,
)


def profile() -> dict[str, Any]:
    """Compare all baseline/optimized pairs using formal harness."""
    if should_use_cpu_minimal():
        from ch20.compare_cpu_minimal import profile as cpu_minimal_profile

        return cpu_minimal_profile()

    chapter_dir = Path(__file__).parent

    return profile_template(
        chapter='ch20',
        chapter_dir=chapter_dir,
        harness_config=BenchmarkConfig(iterations=20, warmup=5),
    )


if __name__ == '__main__':
    result = profile()
    print("\nMetrics:", result)
