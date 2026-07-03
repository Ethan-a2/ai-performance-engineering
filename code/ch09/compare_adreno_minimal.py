"""Tiny Adreno OpenCL comparison path for CH09."""

from __future__ import annotations

from core.benchmark.adreno_minimal import print_adreno_minimal_result, run_adreno_minimal_profile


def profile() -> dict:
    """Run the minimal Adreno comparison and return machine-readable metrics."""
    return run_adreno_minimal_profile("ch09")


if __name__ == "__main__":
    print_adreno_minimal_result(profile())
