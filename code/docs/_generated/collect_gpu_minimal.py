from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from core.benchmark.gpu_minimal import CHAPTER_SCENARIOS, run_gpu_minimal_profile


def main() -> None:
    results: dict[str, object] = {
        "host": {},
        "scenarios": {},
        "chapters": {},
    }
    scenarios = results["scenarios"]
    chapters = results["chapters"]
    assert isinstance(scenarios, dict)
    assert isinstance(chapters, dict)

    for chapter in sorted(CHAPTER_SCENARIOS):
        metrics = run_gpu_minimal_profile(chapter)["metrics"]
        if not results["host"]:
            results["host"] = {
                "torch": metrics.get("torch_version"),
                "cuda_available": metrics.get("cuda_available", False),
                "gpu": metrics.get("gpu_name"),
                "compute_capability": metrics.get("compute_capability"),
                "device_count": 1 if metrics.get("status") == "ok" else 0,
                "note": "ssh mi standard gpu_minimal helper; RTX 2060 non-canonical unless clocks and host validity are controlled",
            }
        scenario_name = str(metrics.get("scenario", "unknown"))
        if metrics.get("status") == "ok" and scenario_name not in scenarios:
            scenarios[scenario_name] = {
                "summary": metrics["scenario_summary"],
                "baseline": {
                    "mean_ms": metrics["baseline_ms"],
                    "min_ms": metrics["baseline_min_ms"],
                    "max_ms": metrics["baseline_max_ms"],
                    "times_ms": metrics["baseline_raw_ms"],
                },
                "optimized": {
                    "mean_ms": metrics["optimized_ms"],
                    "min_ms": metrics["optimized_min_ms"],
                    "max_ms": metrics["optimized_max_ms"],
                    "times_ms": metrics["optimized_raw_ms"],
                },
                "speedup": metrics["speedup"],
                "max_abs_err": metrics["optimized_max_abs_err"],
            }
        chapters[chapter] = {
            "scenario": metrics.get("scenario"),
            "summary": metrics.get("scenario_summary"),
            "baseline_ms": metrics.get("baseline_ms"),
            "optimized_ms": metrics.get("optimized_ms"),
            "speedup": metrics.get("speedup"),
            "max_abs_err": metrics.get("optimized_max_abs_err"),
            "status": metrics.get("status"),
        }

    out = Path("docs/_generated/gpu_minimal_rtx2060_metrics.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("WROTE", out)


if __name__ == "__main__":
    main()
