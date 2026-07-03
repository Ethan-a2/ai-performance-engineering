"""Tiny CUDA GPU comparison path for CH05."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from core.benchmark.gpu_minimal import print_gpu_minimal_result, run_gpu_minimal_profile


def profile() -> dict:
    """Run the minimal GPU comparison and return machine-readable metrics."""
    return run_gpu_minimal_profile("ch05")


if __name__ == "__main__":
    print_gpu_minimal_result(profile())
