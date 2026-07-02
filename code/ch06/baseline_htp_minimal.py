"""Minimal Snapdragon Hexagon HTP baseline for CH06."""

from __future__ import annotations

from core.benchmark.htp_minimal import get_htp_minimal_benchmark


def get_benchmark():
    return get_htp_minimal_benchmark("ch06", optimized=False)
