"""Minimal Snapdragon Hexagon HTP optimized path for CH09."""

from __future__ import annotations

from core.benchmark.htp_minimal import get_htp_minimal_benchmark


def get_benchmark():
    return get_htp_minimal_benchmark("ch09", optimized=True)
