"""Minimal Snapdragon Hexagon HTP optimized path for CH10."""

from __future__ import annotations

from core.benchmark.htp_minimal import get_htp_minimal_benchmark


def get_benchmark():
    return get_htp_minimal_benchmark("ch10", optimized=True)
