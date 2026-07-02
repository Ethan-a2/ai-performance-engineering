"""Chapter 19: Compare baseline vs optimized implementations using formal harness."""

from pathlib import Path
from typing import Any

from core.benchmark.cpu_minimal import should_use_cpu_minimal
from core.harness.benchmark_harness import BenchmarkConfig
from core.utils.chapter_compare_template import profile_template


def profile() -> dict[str, Any]:
    """Compare all baseline/optimized pairs using formal harness."""
    if should_use_cpu_minimal():
        from ch19.compare_cpu_minimal import profile as cpu_minimal_profile

        return cpu_minimal_profile()

    chapter_dir = Path(__file__).parent

    # Reduced iterations for ch19 - Transformer Engine can be slow with large models
    # FP8/FP4 conversion overhead makes each iteration slower
    return profile_template(
        chapter="ch19",
        chapter_dir=chapter_dir,
        harness_config=BenchmarkConfig(iterations=10, warmup=5),  # Reduced from 20,5
    )


if __name__ == "__main__":
    result = profile()
    print("\nMetrics:", result)
