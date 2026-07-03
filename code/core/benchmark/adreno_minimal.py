"""Reusable Adreno OpenCL minimal benchmark helpers."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_NDK = Path("/opt/Android/Ndk/android-ndk-r28c")
DEFAULT_KERNEL = Path(
    "/media/code/llm/llama/llama.cpp/ggml/src/ggml-opencl/kernels/gemm_xmem_f16_f32_os8.cl"
)
SOURCE = Path(__file__).with_name("adreno_minimal_opencl.cpp")


@dataclass(frozen=True)
class AdrenoScenario:
    name: str
    summary: str
    iterations: int = 30


CHAPTER_SCENARIOS: dict[str, AdrenoScenario] = {
    "ch02": AdrenoScenario("xmem_gemm", "Adreno xmem GEMM mirrors tuned cuBLAS-style hardware-aware matmul"),
    "ch03": AdrenoScenario("xmem_gemm", "larger GEMM highlights runtime launch and provisioning overhead"),
    "ch04": AdrenoScenario("pipeline_fusion", "fused device-side work mirrors communication/gradient fusion"),
    "ch05": AdrenoScenario("copy_vectorized", "vectorized device copy mirrors vectorized preprocessing and IO staging"),
    "ch06": AdrenoScenario("pipeline_fusion", "fused OpenCL kernels replace scalar/poorly amortized work"),
    "ch07": AdrenoScenario("copy_vectorized", "float4 copy mirrors coalesced/vectorized memory movement"),
    "ch08": AdrenoScenario("pipeline_fusion", "fused branch-light work mirrors ILP and warp-efficiency tuning"),
    "ch09": AdrenoScenario("pipeline_fusion", "kernel fusion reduces memory traffic on a bandwidth-limited workload"),
    "ch10": AdrenoScenario("xmem_gemm", "Adreno constant-load xmem GEMM mirrors specialized tensor-core pipeline tuning"),
    "ch11": AdrenoScenario("kv_block", "batched device update mirrors stream/concurrency overlap by reducing serialized launches"),
    "ch12": AdrenoScenario("kv_block", "block replay mirrors graph-style launch amortization"),
    "ch13": AdrenoScenario("kv_block", "block KV update mirrors paged/cache-aware memory behavior"),
    "ch14": AdrenoScenario("xmem_gemm", "precompiled OpenCL GEMM mirrors compiler-specialized persistent kernels"),
    "ch15": AdrenoScenario("kv_block", "block KV update mirrors pooled cache and batching orchestration"),
    "ch16": AdrenoScenario("xmem_gemm", "xmem GEMM mirrors production backend specialization"),
    "ch17": AdrenoScenario("kv_block", "batched handoff mirrors prefill/decode routing and KV movement reduction"),
    "ch18": AdrenoScenario("kv_block", "active-window KV block update mirrors cache-aware decode"),
    "ch19": AdrenoScenario("copy_vectorized", "vectorized memory movement mirrors lower-precision/cache traffic reduction"),
    "ch20": AdrenoScenario("pipeline_fusion", "composed fused stages mirror end-to-end optimization composition"),
}


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else default


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer, got {value!r}") from exc


def _run(command: list[str], *, cwd: Path | None = None, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def _require_success(process: subprocess.CompletedProcess[str], command: list[str]) -> None:
    if process.returncode != 0:
        output = f"{process.stdout}\n{process.stderr}".strip()
        tail = "\n".join(output.splitlines()[-100:])
        raise RuntimeError(f"command failed rc={process.returncode}: {' '.join(command)}\n{tail}")


def _binary_path(chapter: str) -> Path:
    return Path(__file__).resolve().parents[2] / ".cache" / "adreno_minimal" / chapter / "adreno_minimal_android"


def build_adreno_minimal_binary(chapter: str, ndk: Path) -> Path:
    compiler = ndk / "toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android28-clang++"
    sysroot = ndk / "toolchains/llvm/prebuilt/linux-x86_64/sysroot"
    output = _binary_path(chapter)
    output.parent.mkdir(parents=True, exist_ok=True)

    if not SOURCE.exists():
        raise RuntimeError(f"Adreno OpenCL source not found: {SOURCE}")
    if not compiler.exists():
        raise RuntimeError(f"Android compiler not found: {compiler}")
    if not (sysroot / "usr/include/CL/cl.h").exists():
        raise RuntimeError(f"OpenCL headers not found under NDK sysroot: {sysroot}")
    if not (sysroot / "usr/lib/aarch64-linux-android/libOpenCL.so").exists():
        raise RuntimeError(f"OpenCL library not found under NDK sysroot: {sysroot}")

    command = [
        str(compiler),
        "-std=c++17",
        "-O3",
        "-Wall",
        "-Wextra",
        str(SOURCE),
        "-lOpenCL",
        "-o",
        str(output),
    ]
    _require_success(_run(command, cwd=SOURCE.parents[2]), command)
    return output


def _extract_float(name: str, text: str) -> float | None:
    match = re.search(rf"^{re.escape(name)}\s*=\s*([-+0-9.eE]+)", text, re.MULTILINE)
    if not match:
        return None
    return float(match.group(1))


def run_adreno_minimal_profile(chapter: str) -> dict[str, Any]:
    """Build, push, and run the chapter's Adreno minimal scenario."""
    if shutil.which("adb") is None:
        return {
            "metrics": {
                "chapter": chapter,
                "target": "adreno_minimal",
                "status": "skipped",
                "reason": "SKIPPED: adb is required for Adreno minimal benchmarks",
            }
        }

    scenario = CHAPTER_SCENARIOS.get(chapter)
    if scenario is None:
        raise RuntimeError(f"No Adreno minimal scenario registered for {chapter}")

    ndk = _env_path("ANDROID_NDK", DEFAULT_NDK)
    kernel = _env_path("AISP_ADRENO_XMEM_KERNEL", DEFAULT_KERNEL)
    iterations = _env_int("AISP_ADRENO_MINIMAL_ITERATIONS", scenario.iterations)
    timeout = _env_int("AISP_ADRENO_MINIMAL_TIMEOUT_SECONDS", 120)
    remote_dir = os.environ.get("AISP_ADRENO_MINIMAL_REMOTE_DIR", f"/data/local/tmp/{chapter}_adreno_minimal")

    if not kernel.exists():
        raise RuntimeError(f"Adreno xmem kernel not found: {kernel}")

    binary = build_adreno_minimal_binary(chapter, ndk)
    setup_cmd = ["adb", "shell", f"mkdir -p {remote_dir}"]
    _require_success(_run(setup_cmd, timeout=timeout), setup_cmd)

    for local in (binary, kernel):
        push_cmd = ["adb", "push", str(local), f"{remote_dir}/"]
        _require_success(_run(push_cmd, timeout=timeout), push_cmd)

    run_cmd = [
        "adb",
        "shell",
        f"cd {remote_dir} && chmod +x {binary.name} && ./{binary.name} {scenario.name} {kernel.name} {iterations}",
    ]
    process = _run(run_cmd, timeout=timeout)
    _require_success(process, run_cmd)
    output = f"{process.stdout}\n{process.stderr}"

    return {
        "metrics": {
            "chapter": chapter,
            "target": "adreno_minimal",
            "device": "adreno_opencl",
            "status": "ok",
            "scenario": scenario.name,
            "scenario_summary": scenario.summary,
            "iterations": iterations,
            "baseline_ms": _extract_float("baseline_ms", output),
            "optimized_ms": _extract_float("optimized_ms", output),
            "speedup": _extract_float("speedup", output),
            "baseline_max_abs_err": _extract_float("baseline_max_abs_err", output),
            "optimized_max_abs_err": _extract_float("optimized_max_abs_err", output),
            "output_tail": "\n".join(output.strip().splitlines()[-40:]),
        }
    }


def print_adreno_minimal_result(result: dict[str, Any]) -> None:
    metrics = result["metrics"]
    print(f"{metrics['chapter'].upper()} Adreno minimal comparison")
    if metrics.get("status") != "ok":
        print(f"  {metrics.get('reason', 'Adreno run did not complete')}")
        return
    print(f"  Scenario:              {metrics['scenario']}")
    print(f"  Baseline:              {metrics['baseline_ms']:.6f} ms")
    print(f"  Optimized:             {metrics['optimized_ms']:.6f} ms")
    print(f"  Speedup:               {metrics['speedup']:.2f}x")
    err = metrics.get("optimized_max_abs_err")
    if err is not None:
        print(f"  Optimized max error:   {err:.8f}")
