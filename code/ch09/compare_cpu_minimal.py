"""Tiny CPU-only comparison path for CH09."""

from __future__ import annotations

from core.benchmark.cpu_minimal import print_cpu_minimal_result, run_cpu_minimal_profile


def profile() -> dict:
    """Run the minimal CPU comparison and return machine-readable metrics."""
    return run_cpu_minimal_profile("ch09")


if __name__ == "__main__":
    print_cpu_minimal_result(profile())
