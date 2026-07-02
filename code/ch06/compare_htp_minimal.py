"""Tiny Snapdragon Hexagon HTP comparison path for CH06."""

from __future__ import annotations

from core.benchmark.htp_minimal import print_htp_minimal_result, run_htp_minimal_profile


def profile() -> dict:
    """Run the minimal HTP comparison and return machine-readable metrics."""
    return run_htp_minimal_profile("ch06")


if __name__ == "__main__":
    print_htp_minimal_result(profile())
