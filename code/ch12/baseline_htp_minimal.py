"""Minimal Snapdragon Hexagon HTP baseline for CH12."""

from __future__ import annotations

from core.benchmark.htp_minimal import get_htp_minimal_benchmark


def get_benchmark():
    return get_htp_minimal_benchmark("ch12", optimized=False)
