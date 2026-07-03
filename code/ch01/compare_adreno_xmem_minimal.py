"""Tiny Adreno OpenCL xmem comparison path for Chapter 1."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any


CHAPTER_DIR = Path(__file__).resolve().parent
DEFAULT_NDK = Path("/opt/Android/Ndk/android-ndk-r28c")
DEFAULT_KERNEL = Path(
    "/media/code/llm/llama/llama.cpp/ggml/src/ggml-opencl/kernels/gemm_xmem_f16_f32_os8.cl"
)


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
        tail = "\n".join(output.splitlines()[-80:])
        raise RuntimeError(f"command failed rc={process.returncode}: {' '.join(command)}\n{tail}")


def _build_binary(ndk: Path) -> Path:
    compiler = ndk / "toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android28-clang++"
    sysroot = ndk / "toolchains/llvm/prebuilt/linux-x86_64/sysroot"
    source = CHAPTER_DIR / "adreno_xmem_minimal.cpp"
    output = CHAPTER_DIR / "adreno_xmem_minimal_android"

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
        str(source),
        "-lOpenCL",
        "-o",
        str(output),
    ]
    process = _run(command, cwd=CHAPTER_DIR)
    _require_success(process, command)
    return output


def _extract_float(name: str, text: str) -> float | None:
    match = re.search(rf"^{re.escape(name)}\s*=\s*([-+0-9.eE]+)", text, re.MULTILINE)
    if not match:
        return None
    return float(match.group(1))


def profile() -> dict[str, Any]:
    """Build, push, and run the minimal Adreno xmem comparison."""
    if shutil.which("adb") is None:
        raise RuntimeError("adb is required for the Adreno xmem minimal path")

    ndk = _env_path("ANDROID_NDK", DEFAULT_NDK)
    kernel = _env_path("AISP_ADRENO_XMEM_KERNEL", DEFAULT_KERNEL)
    iterations = _env_int("AISP_ADRENO_XMEM_ITERATIONS", 30)
    m = _env_int("AISP_ADRENO_XMEM_M", 1024)
    n = _env_int("AISP_ADRENO_XMEM_N", 128)
    k = _env_int("AISP_ADRENO_XMEM_K", 1024)
    timeout = _env_int("AISP_ADRENO_XMEM_TIMEOUT_SECONDS", 120)
    remote_dir = os.environ.get("AISP_ADRENO_XMEM_REMOTE_DIR", "/data/local/tmp/ch01_adreno_xmem_minimal")

    if not kernel.exists():
        raise RuntimeError(f"Adreno xmem kernel not found: {kernel}")

    binary = _build_binary(ndk)

    setup_cmd = ["adb", "shell", f"mkdir -p {remote_dir}"]
    _require_success(_run(setup_cmd, timeout=timeout), setup_cmd)

    for local in (binary, kernel):
        push_cmd = ["adb", "push", str(local), f"{remote_dir}/"]
        _require_success(_run(push_cmd, timeout=timeout), push_cmd)

    run_cmd = [
        "adb",
        "shell",
        f"cd {remote_dir} && chmod +x {binary.name} && ./{binary.name} {kernel.name} {iterations} {m} {n} {k}",
    ]
    process = _run(run_cmd, timeout=timeout)
    _require_success(process, run_cmd)

    output = f"{process.stdout}\n{process.stderr}"
    return {
        "metrics": {
            "chapter": "ch01",
            "target": "adreno_xmem_minimal",
            "device": "adreno_opencl",
            "iterations": iterations,
            "M": m,
            "N": n,
            "K": k,
            "baseline_ms": _extract_float("baseline_ms", output),
            "optimized_ms": _extract_float("optimized_ms", output),
            "speedup": _extract_float("speedup", output),
            "baseline_max_abs_err": _extract_float("baseline_max_abs_err", output),
            "optimized_max_abs_err": _extract_float("optimized_max_abs_err", output),
            "output_tail": "\n".join(output.strip().splitlines()[-40:]),
        }
    }


if __name__ == "__main__":
    result = profile()
    metrics = result["metrics"]
    print("Chapter 1 Adreno xmem minimal comparison")
    print(f"  Baseline naive OpenCL: {metrics['baseline_ms']:.6f} ms")
    print(f"  Optimized xmem:        {metrics['optimized_ms']:.6f} ms")
    print(f"  Speedup:               {metrics['speedup']:.2f}x")
    print(f"  Optimized max error:   {metrics['optimized_max_abs_err']:.8f}")
