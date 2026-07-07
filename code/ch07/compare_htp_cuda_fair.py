"""Fair llama.cpp HTP/CUDA comparison path for CH07."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from core.benchmark.htp_cuda_fair import print_htp_cuda_fair_result, run_htp_cuda_fair_profile


def profile() -> dict:
    """Run the fair HTP/CUDA comparison and return machine-readable metrics."""
    return run_htp_cuda_fair_profile("ch07")


if __name__ == "__main__":
    print_htp_cuda_fair_result(profile())
