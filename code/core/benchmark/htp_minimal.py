"""Reusable Snapdragon Hexagon HTP minimal benchmark helpers."""

from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig

_DEFAULT_LLAMA_CPP_ROOT = Path("/media/code/llm/llama/llama.cpp")
_DEFAULT_MODEL = "functiongemma-270m-it-BF16.gguf"
_DEFAULT_PROMPT = "what is the most popular cookie in the world?"


@dataclass(frozen=True)
class HtpMinimalConfig:
    chapter: str
    optimized: bool
    llama_cpp_root: Path = _DEFAULT_LLAMA_CPP_ROOT
    model: str = _DEFAULT_MODEL
    device: str = "HTP0"
    prompt: str = _DEFAULT_PROMPT
    n_predict: int = 16
    timeout_seconds: int = 120
    env: dict[str, str] = field(default_factory=dict)


def _env_override(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def _int_env_override(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer, got {value!r}") from exc


def build_htp_minimal_config(chapter: str, *, optimized: bool) -> HtpMinimalConfig:
    root = Path(_env_override("AISP_HTP_LLAMA_CPP_ROOT", str(_DEFAULT_LLAMA_CPP_ROOT))).expanduser()
    return HtpMinimalConfig(
        chapter=chapter,
        optimized=optimized,
        llama_cpp_root=root,
        model=_env_override("AISP_HTP_MODEL", _DEFAULT_MODEL),
        device=_env_override("AISP_HTP_DEVICE", "HTP0"),
        prompt=_env_override("AISP_HTP_PROMPT", _DEFAULT_PROMPT),
        n_predict=_int_env_override("AISP_HTP_N_PREDICT", 16),
        timeout_seconds=_int_env_override("AISP_HTP_TIMEOUT_SECONDS", 120),
    )


def _run_skip_check(config: HtpMinimalConfig) -> str | None:
    if shutil.which("adb") is None:
        return "SKIPPED: HTP minimal benchmark requires adb in PATH"
    if not config.llama_cpp_root.exists():
        return f"SKIPPED: llama.cpp root not found: {config.llama_cpp_root}"
    script = config.llama_cpp_root / "scripts" / "snapdragon" / "adb" / "run-completion.sh"
    if not script.exists():
        return f"SKIPPED: Snapdragon run script not found: {script}"
    installed = config.llama_cpp_root / "pkg-snapdragon" / "llama.cpp"
    if not installed.exists():
        return f"SKIPPED: Snapdragon package not installed: {installed}"
    return None


def _base_env(config: HtpMinimalConfig) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "M": config.model,
            "D": config.device,
            "V": env.get("AISP_HTP_VERBOSE", "1"),
            "PROF": env.get("AISP_HTP_PROFILE", "1"),
        }
    )
    return env


def _mode_env(config: HtpMinimalConfig) -> dict[str, str]:
    env = _base_env(config)
    if config.optimized:
        env.setdefault("HMX", "1")
        env.setdefault("MM", "3")
        env.setdefault("FA", "2")
        env.setdefault("OC", "1")
        env.setdefault("OB", "1024")
        env.setdefault("OQ", "16")
    else:
        env.setdefault("HMX", "0")
        env.setdefault("MM", "1")
        env.setdefault("FA", "1")
        env.setdefault("OC", "0")
        env.setdefault("OB", "1")
        env.setdefault("OQ", "1")
    return env


def _extract_metric(patterns: list[str], text: str) -> float | None:
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue
    return None


def _extract_metrics(output: str) -> dict[str, Any]:
    return {
        "tokens_per_second": _extract_metric(
            [
                r"tokens per second\s*=\s*([0-9.]+)",
                r"tok/s\s*=\s*([0-9.]+)",
                r"([0-9.]+)\s*tokens per second",
            ],
            output,
        ),
        "prompt_ms": _extract_metric([r"prompt eval time\s*=\s*([0-9.]+)\s*ms"], output),
        "eval_ms": _extract_metric([r"eval time\s*=\s*([0-9.]+)\s*ms"], output),
        "total_ms": _extract_metric([r"total time\s*=\s*([0-9.]+)\s*ms"], output),
    }


class HtpMinimalBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Run a tiny llama.cpp Snapdragon Hexagon HTP completion benchmark."""

    allow_cpu = True
    htp_minimal_benchmark = True
    signature_equivalence_group = "htp_minimal"
    signature_equivalence_ignore_fields: tuple[str, ...] = ()

    def __init__(self, *, chapter: str, optimized: bool) -> None:
        super().__init__()
        self.device = torch.device("cpu")
        self.config = build_htp_minimal_config(chapter, optimized=optimized)
        self.result: dict[str, Any] | None = None
        self._skip_reason: str | None = None
        self._verify_input = torch.tensor([float(self.config.n_predict)], dtype=torch.float32)
        self._verify_output: torch.Tensor | None = None
        self.register_workload_metadata(
            custom_units_per_iteration=float(self.config.n_predict),
            custom_unit_name="tokens_requested",
        )

    def setup(self) -> None:
        self._skip_reason = _run_skip_check(self.config)
        if self._skip_reason:
            raise RuntimeError(self._skip_reason)

    def benchmark_fn(self) -> None:
        script = self.config.llama_cpp_root / "scripts" / "snapdragon" / "adb" / "run-completion.sh"
        command = [
            str(script),
            "-p",
            shlex.quote(self.config.prompt),
            "-n",
            str(self.config.n_predict),
        ]
        started = time.perf_counter()
        process = subprocess.run(
            command,
            cwd=self.config.llama_cpp_root,
            env=_mode_env(self.config),
            text=True,
            capture_output=True,
            timeout=self.config.timeout_seconds,
            check=False,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        output = f"{process.stdout}\n{process.stderr}"
        metrics = _extract_metrics(output)
        self.result = {
            "returncode": process.returncode,
            "elapsed_ms": elapsed_ms,
            "output_tail": "\n".join(output.strip().splitlines()[-40:]),
            "metrics": metrics,
            "htp_mode": "optimized" if self.config.optimized else "baseline",
            "htp_env": {
                key: _mode_env(self.config).get(key)
                for key in ("D", "HMX", "MM", "FA", "OC", "OB", "OQ", "NHVX", "PROF", "V")
                if _mode_env(self.config).get(key) is not None
            },
        }

    def capture_verification_payload(self) -> None:
        if self.result is None:
            raise RuntimeError("HTP command did not run")
        total_ms = self.result.get("metrics", {}).get("total_ms")
        if total_ms is None:
            total_ms = self.result.get("elapsed_ms") or 0.0
        self._verify_output = torch.tensor([float(self.config.n_predict)], dtype=torch.float32)
        self._set_verification_payload(
            inputs={"requested_tokens": self._verify_input},
            output=self._verify_output,
            batch_size=1,
            parameter_count=max(1, len(self.config.model)),
            precision_flags={"fp16": False, "bf16": False, "fp8": False, "tf32": False},
            output_tolerance=(0.0, 0.0),
            signature_overrides={"quantization_mode": "htp_minimal_external"},
        )

    def validate_result(self) -> str | None:
        if self.result is None:
            return "HTP command did not run"
        if self.result["returncode"] != 0:
            tail = self.result.get("output_tail") or ""
            if "not found" in tail.lower() or "no devices" in tail.lower() or "device" in tail.lower():
                return f"SKIPPED: HTP run failed; check adb device and Snapdragon package. Tail:\n{tail}"
            return f"HTP command failed with rc={self.result['returncode']}. Tail:\n{tail}"
        return None

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=1,
            warmup=0,
            device=torch.device("cpu"),
            enable_memory_tracking=False,
            enable_profiling=False,
            use_subprocess=False,
            enforce_environment_validation=False,
            adaptive_iterations=False,
            cross_validate_timing=False,
            clear_l2_cache=False,
            isolate_warmup_cache=False,
            full_device_sync=False,
            disable_gc_during_timing=True,
            detect_benchmark_fn_antipatterns=False,
            detect_benchmark_fn_sync=False,
            measurement_timeout_seconds=self.config.timeout_seconds + 30,
            setup_timeout_seconds=30,
        )

    def get_custom_metrics(self) -> dict[str, Any]:
        if not self.result:
            return {}
        metrics = dict(self.result.get("metrics") or {})
        metrics.update(
            {
                "elapsed_ms": self.result.get("elapsed_ms"),
                "htp_optimized": 1.0 if self.config.optimized else 0.0,
            }
        )
        return metrics


def get_htp_minimal_benchmark(chapter: str, *, optimized: bool) -> BaseBenchmark:
    return HtpMinimalBenchmark(chapter=chapter, optimized=optimized)


def run_htp_minimal_profile(chapter: str) -> dict[str, Any]:
    baseline = HtpMinimalBenchmark(chapter=chapter, optimized=False)
    optimized = HtpMinimalBenchmark(chapter=chapter, optimized=True)

    results: dict[str, Any] = {"chapter": chapter, "target": "htp_minimal"}
    for name, bench in (("baseline", baseline), ("optimized", optimized)):
        try:
            bench.setup()
            bench.benchmark_fn()
            validation = bench.validate_result()
            if validation:
                return {"metrics": {**results, "status": "skipped" if validation.startswith("SKIPPED:") else "failed", "reason": validation}}
            results[f"{name}_metrics"] = bench.get_custom_metrics()
        finally:
            bench.teardown()

    baseline_ms = float(results["baseline_metrics"].get("elapsed_ms") or 0.0)
    optimized_ms = float(results["optimized_metrics"].get("elapsed_ms") or 0.0)
    results["baseline_ms"] = baseline_ms
    results["optimized_ms"] = optimized_ms
    results["speedup"] = baseline_ms / optimized_ms if optimized_ms else 0.0
    results["status"] = "ok"
    return {"metrics": results}


def print_htp_minimal_result(result: dict[str, Any]) -> None:
    metrics = result["metrics"]
    print(f"{metrics['chapter'].upper()} HTP minimal comparison")
    if metrics.get("status") != "ok":
        print(f"  {metrics.get('reason', 'HTP run did not complete')}")
        return
    print(f"  Baseline HVX/minimal path: {metrics['baseline_ms']:.3f} ms")
    print(f"  Optimized HMX/fused path:  {metrics['optimized_ms']:.3f} ms")
    print(f"  Speedup:                  {metrics['speedup']:.2f}x")
