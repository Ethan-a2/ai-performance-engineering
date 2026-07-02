"""Small CPU-only benchmark used by the Chapter 1 quick-start path."""

from __future__ import annotations

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig


class CpuMinimalBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Compare an intentionally scalar CPU matmul against a vectorized CPU matmul."""

    allow_cpu = True
    signature_equivalence_group = "ch01_cpu_minimal"

    def __init__(self, *, vectorized: bool) -> None:
        super().__init__()
        self.device = torch.device("cpu")
        self.vectorized = vectorized
        self.batch_size = 32
        self.input_dim = 64
        self.output_dim = 32
        self.x: torch.Tensor | None = None
        self.weight: torch.Tensor | None = None
        self.bias: torch.Tensor | None = None
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
        self.result = torch.empty(self.batch_size, self.output_dim)
        self._verify_input = self.x.clone()
        self._verify_output = torch.empty_like(self.result)

    def _run_scalar(self) -> None:
        assert self.x is not None
        assert self.weight is not None
        assert self.bias is not None
        assert self.result is not None
        for row in range(self.batch_size):
            for col in range(self.output_dim):
                total = float(self.bias[col])
                for inner in range(self.input_dim):
                    total += float(self.x[row, inner]) * float(self.weight[col, inner])
                self.result[row, col] = total

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

    def capture_verification_payload(self) -> None:
        if self._verify_input is None or self._verify_output is None or self.result is None:
            raise RuntimeError("setup() and benchmark_fn() must run before capture_verification_payload()")
        self._verify_output.copy_(self.result)
        self._set_verification_payload(
            inputs={"x": self._verify_input},
            output=self._verify_output,
            batch_size=self.batch_size,
            parameter_count=int(self.weight.numel() + self.bias.numel()),
            precision_flags={"fp16": False, "bf16": False, "fp8": False, "tf32": False},
            output_tolerance=(1e-4, 1e-4),
        )

    def teardown(self) -> None:
        self.x = None
        self.weight = None
        self.bias = None
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
        if self.result is None:
            return "Output was not produced"
        if self.x is None or self.weight is None or self.bias is None:
            return "Inputs were not initialized"
        if self.result.shape != (self.batch_size, self.output_dim):
            return f"Output shape mismatch: {tuple(self.result.shape)}"
        if not torch.isfinite(self.result).all():
            return "Output contains non-finite values"
        reference = torch.addmm(self.bias, self.x, self.weight.t())
        max_diff = (self.result - reference).abs().max().item()
        if max_diff > 1e-4:
            return f"Output mismatch: max_diff={max_diff:.6g}"
        return None
