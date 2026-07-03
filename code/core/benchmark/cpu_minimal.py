"""Reusable CPU-only minimal benchmark for chapter quick-start paths."""

from __future__ import annotations

import statistics
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig


@dataclass(frozen=True)
class CpuMinimalTimedResult:
    name: str
    mean_ms: float
    raw_ms: list[float]


class CpuMinimalBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Compare an intentionally scalar CPU matmul against a vectorized CPU matmul."""

    allow_cpu = True
    cpu_fallback_benchmark = True
    signature_equivalence_group = "cpu_minimal"
    signature_equivalence_ignore_fields: tuple[str, ...] = ()

    def __init__(self, *, chapter: str, vectorized: bool) -> None:
        super().__init__()
        self.device = torch.device("cpu")
        self.chapter = chapter
        self.vectorized = vectorized
        self.batch_size = 32
        self.input_dim = 64
        self.output_dim = 32
        self.x: torch.Tensor | None = None
        self.weight: torch.Tensor | None = None
        self.bias: torch.Tensor | None = None
        self.x_rows: list[list[float]] | None = None
        self.weight_rows: list[list[float]] | None = None
        self.bias_values: list[float] | None = None
        self.scalar_result: list[list[float]] | None = None
        self.result: torch.Tensor | None = None
        self._verify_input: torch.Tensor | None = None
        self._verify_output: torch.Tensor | None = None
        ops = self.batch_size * self.input_dim * self.output_dim * 2
        self.register_workload_metadata(custom_units_per_iteration=float(ops), custom_unit_name="ops")

    def setup(self) -> None:
        generator = torch.Generator(device="cpu").manual_seed(42)
        self.x = torch.randn(self.batch_size, self.input_dim, generator=generator)
        self.weight = torch.randn(self.output_dim, self.input_dim, generator=generator)
        self.bias = torch.randn(self.output_dim, generator=generator)
        self.x_rows = self.x.tolist()
        self.weight_rows = self.weight.tolist()
        self.bias_values = self.bias.tolist()
        self.scalar_result = [[0.0 for _ in range(self.output_dim)] for _ in range(self.batch_size)]
        self.result = torch.empty(self.batch_size, self.output_dim)
        self._verify_input = self.x.clone()
        self._verify_output = torch.empty_like(self.result)

    def _run_scalar(self) -> None:
        assert self.x_rows is not None
        assert self.weight_rows is not None
        assert self.bias_values is not None
        assert self.scalar_result is not None
        for row in range(self.batch_size):
            x_row = self.x_rows[row]
            out_row = self.scalar_result[row]
            for col in range(self.output_dim):
                weight_row = self.weight_rows[col]
                total = self.bias_values[col]
                for inner in range(self.input_dim):
                    total += x_row[inner] * weight_row[inner]
                out_row[col] = total

    def _run_vectorized(self) -> None:
        assert self.x is not None
        assert self.weight is not None
        assert self.bias is not None
        assert self.result is not None
        torch.addmm(self.bias, self.x, self.weight.t(), out=self.result)

    def benchmark_fn(self) -> None:
        if self.vectorized:
            self._run_vectorized()
        else:
            self._run_scalar()

    def _result_tensor(self) -> torch.Tensor:
        if self.vectorized:
            if self.result is None:
                raise RuntimeError("Output was not produced")
            return self.result
        if self.scalar_result is None:
            raise RuntimeError("Output was not produced")
        return torch.tensor(self.scalar_result, dtype=torch.float32)

    def capture_verification_payload(self) -> None:
        if self._verify_input is None or self._verify_output is None:
            raise RuntimeError("setup() and benchmark_fn() must run before capture_verification_payload()")
        self._verify_output.copy_(self._result_tensor())
        self._set_verification_payload(
            inputs={"x": self._verify_input},
            output=self._verify_output,
            batch_size=self.batch_size,
            parameter_count=self.output_dim * self.input_dim + self.output_dim,
            precision_flags={"fp16": False, "bf16": False, "fp8": False, "tf32": False},
            output_tolerance=(1e-4, 1e-4),
        )

    def teardown(self) -> None:
        self.x = None
        self.weight = None
        self.bias = None
        self.x_rows = None
        self.weight_rows = None
        self.bias_values = None
        self.scalar_result = None
        self.result = None
        self._verify_input = None
        self._verify_output = None

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=5,
            warmup=5,
            device=torch.device("cpu"),
            enable_memory_tracking=False,
            enable_profiling=False,
            use_subprocess=False,
            enforce_environment_validation=False,
            adaptive_iterations=False,
            cross_validate_timing=False,
            disable_gc_during_timing=False,
            track_memory_allocations=False,
        )

    def validate_result(self) -> str | None:
        if self.x is None or self.weight is None or self.bias is None:
            return "Inputs were not initialized"
        try:
            result = self._result_tensor()
        except RuntimeError as exc:
            return str(exc)
        if result.shape != (self.batch_size, self.output_dim):
            return f"Output shape mismatch: {tuple(result.shape)}"
        if not torch.isfinite(result).all():
            return "Output contains non-finite values"
        reference = torch.addmm(self.bias, self.x, self.weight.t())
        max_diff = (result - reference).abs().max().item()
        if max_diff > 1e-4:
            return f"Output mismatch: max_diff={max_diff:.6g}"
        return None


def get_cpu_minimal_benchmark(chapter: str, *, vectorized: bool) -> BaseBenchmark:
    return CpuMinimalBenchmark(chapter=chapter, vectorized=vectorized)


def requested_cpu_minimal(argv: Sequence[str] | None = None) -> bool:
    for arg in list(sys.argv[1:] if argv is None else argv):
        if "cpu_minimal" in str(arg).replace(",", " ").split():
            return True
    return False


def requested_htp_minimal(argv: Sequence[str] | None = None) -> bool:
    for arg in list(sys.argv[1:] if argv is None else argv):
        if "htp_minimal" in str(arg).replace(",", " ").split():
            return True
    return False


def requested_gpu_minimal(argv: Sequence[str] | None = None) -> bool:
    for arg in list(sys.argv[1:] if argv is None else argv):
        if "gpu_minimal" in str(arg).replace(",", " ").split():
            return True
    return False


def requested_adreno_minimal(argv: Sequence[str] | None = None) -> bool:
    for arg in list(sys.argv[1:] if argv is None else argv):
        if "adreno_minimal" in str(arg).replace(",", " ").split():
            return True
    return False


def should_use_cpu_minimal(argv: Sequence[str] | None = None) -> bool:
    requested_other_minimal = (
        requested_htp_minimal(argv) or requested_gpu_minimal(argv) or requested_adreno_minimal(argv)
    )
    return requested_cpu_minimal(argv) or (not requested_other_minimal and not torch.cuda.is_available())


def _time_benchmark(
    chapter: str,
    name: str,
    *,
    iterations: int = 5,
    warmup: int = 1,
) -> CpuMinimalTimedResult:
    benchmark = get_cpu_minimal_benchmark(chapter, vectorized=name != "baseline")
    benchmark.setup()
    try:
        for _ in range(warmup):
            benchmark.benchmark_fn()
        times_ms = []
        for _ in range(iterations):
            start = time.perf_counter()
            benchmark.benchmark_fn()
            times_ms.append((time.perf_counter() - start) * 1000.0)
        validation_error = benchmark.validate_result()
        if validation_error:
            raise RuntimeError(validation_error)
        return CpuMinimalTimedResult(
            name=name,
            mean_ms=statistics.mean(times_ms),
            raw_ms=times_ms,
        )
    finally:
        benchmark.teardown()


def run_cpu_minimal_profile(chapter: str) -> dict:
    baseline = _time_benchmark(chapter, "baseline")
    optimized = _time_benchmark(chapter, "optimized")
    speedup = baseline.mean_ms / optimized.mean_ms if optimized.mean_ms else 0.0
    return {
        "metrics": {
            "chapter": chapter,
            "target": "cpu_minimal",
            "device": "cpu",
            "pytorch_version": torch.__version__,
            "baseline_ms": baseline.mean_ms,
            "optimized_ms": optimized.mean_ms,
            "speedup": speedup,
            "baseline_raw_ms": baseline.raw_ms,
            "optimized_raw_ms": optimized.raw_ms,
        }
    }


def print_cpu_minimal_result(result: dict) -> None:
    metrics = result["metrics"]
    print(f"{metrics['chapter'].upper()} CPU minimal comparison")
    print(f"  Baseline scalar loop: {metrics['baseline_ms']:.3f} ms")
    print(f"  Optimized addmm:      {metrics['optimized_ms']:.3f} ms")
    print(f"  Speedup:             {metrics['speedup']:.2f}x")
