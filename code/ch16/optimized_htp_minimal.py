"""Minimal Snapdragon Hexagon HTP optimized path for CH16."""

from __future__ import annotations

from core.benchmark.htp_minimal import get_htp_minimal_benchmark


def get_benchmark():
    return get_htp_minimal_benchmark("ch16", optimized=True)
