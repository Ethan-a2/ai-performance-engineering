"""Chapter 8: Compare baseline vs optimized implementations using formal harness."""

from contextlib import suppress
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

# Import arch_config early to set up torch inductor cache directory
# This prevents C++ compilation errors when torch.compile is used
with suppress(ImportError):
    from ch08 import arch_config  # noqa: F401 - triggers cache setup

from core.benchmark.cpu_minimal import should_use_cpu_minimal
from core.benchmark.gpu_minimal import should_use_gpu_minimal
from core.harness.benchmark_harness import BenchmarkConfig
from core.utils.chapter_compare_template import profile_template


def profile() -> dict[str, Any]:
    """Compare all baseline/optimized pairs using formal harness."""
    if should_use_cpu_minimal():
        from ch08.compare_cpu_minimal import profile as cpu_minimal_profile

        return cpu_minimal_profile()
    if should_use_gpu_minimal():
        from ch08.compare_gpu_minimal import profile as gpu_minimal_profile

        return gpu_minimal_profile()

    chapter_dir = Path(__file__).parent

    return profile_template(
        chapter="ch08",
        chapter_dir=chapter_dir,
        harness_config=BenchmarkConfig(iterations=20, warmup=5),
    )


if __name__ == "__main__":
    result = profile()
    print("\nMetrics:", result)
