"""Reusable Hexagon HTP minimal benchmark helpers."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_NDK = Path("/opt/Android/Ndk/android-ndk-r28c")
DEFAULT_HEXAGON_SDK = Path("/opt/qcom/Hexagon_SDK/6.6.0.0")
DEFAULT_HEXAGON_TOOLS = DEFAULT_HEXAGON_SDK / "tools/HEXAGON_Tools/19.0.07"
PROJECT = Path(__file__).with_name("htp_minimal_project")


@dataclass(frozen=True)
class HtpScenario:
    name: str
    summary: str
    elements: int = 32768
    repeats: int = 64
    iterations: int = 8


CHAPTER_SCENARIOS: dict[str, HtpScenario] = {
    "ch01": HtpScenario("hvx_tile", "HTP HVX tile update mirrors the chapter's minimal scalar-to-vector acceleration"),
    "ch02": HtpScenario("hvx_tile", "HTP HVX tiled update mirrors hardware-aware matmul/vectorization"),
    "ch03": HtpScenario("hvx_tile", "larger HTP tile work highlights runtime launch and provisioning overhead"),
    "ch04": HtpScenario("pipeline_fusion", "fused HTP device-side work mirrors communication/gradient fusion"),
    "ch05": HtpScenario("copy_vectorized", "HVX vector copy mirrors vectorized preprocessing and IO staging", elements=131072, repeats=96),
    "ch06": HtpScenario("pipeline_fusion", "fused HTP loops replace scalar/poorly amortized work"),
    "ch07": HtpScenario("copy_vectorized", "HVX vector copy mirrors coalesced/vectorized memory movement", elements=131072, repeats=96),
    "ch08": HtpScenario("pipeline_fusion", "fused branch-light HTP work mirrors ILP and warp-efficiency tuning"),
    "ch09": HtpScenario("pipeline_fusion", "HTP fusion reduces memory traffic on a bandwidth-limited workload"),
    "ch10": HtpScenario("hvx_tile", "HTP HVX tile specialization mirrors compiler-guided tensor-core pipeline tuning"),
    "ch11": HtpScenario("kv_block", "batched HTP KV update mirrors stream/concurrency overlap by reducing serialized work", elements=2048, repeats=24),
    "ch12": HtpScenario("kv_block", "block HTP replay mirrors graph-style launch amortization", elements=2048, repeats=24),
    "ch13": HtpScenario("kv_block", "block KV update mirrors paged/cache-aware memory behavior", elements=2048, repeats=24),
    "ch14": HtpScenario("hvx_tile", "prebuilt HTP skel mirrors compiler-specialized persistent kernels"),
    "ch15": HtpScenario("kv_block", "block KV update mirrors pooled cache and batching orchestration", elements=2048, repeats=24),
    "ch16": HtpScenario("hvx_tile", "HTP skel specialization mirrors production backend specialization"),
    "ch17": HtpScenario("kv_block", "batched HTP handoff mirrors prefill/decode routing and KV movement reduction", elements=2048, repeats=24),
    "ch18": HtpScenario("kv_block", "active-window HTP KV block update mirrors cache-aware decode", elements=2048, repeats=24),
    "ch19": HtpScenario("copy_vectorized", "HVX vector memory movement mirrors lower-precision/cache traffic reduction", elements=131072, repeats=96),
    "ch20": HtpScenario("pipeline_fusion", "composed fused HTP stages mirror end-to-end optimization composition"),
}


def _repo_code_dir() -> Path:
    return Path(__file__).resolve().parents[2]


def _cache_dir() -> Path:
    return _repo_code_dir() / ".cache" / "htp_minimal"


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


def _run(
    command: list[str],
    *,
    cwd: Path | None = None,
    timeout: int = 300,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def _require_success(process: subprocess.CompletedProcess[str], command: list[str]) -> None:
    if process.returncode != 0:
        output = f"{process.stdout}\n{process.stderr}".strip()
        tail = "\n".join(output.splitlines()[-120:])
        raise RuntimeError(f"command failed rc={process.returncode}: {' '.join(command)}\n{tail}")


def _skip_metrics(chapter: str, reason: str) -> dict[str, Any]:
    return {
        "metrics": {
            "chapter": chapter,
            "target": "htp_minimal",
            "status": "skipped",
            "reason": reason,
        }
    }


def _preflight(chapter: str, ndk: Path, sdk: Path, tools: Path) -> dict[str, Any] | None:
    if shutil.which("adb") is None:
        return _skip_metrics(chapter, "SKIPPED: adb is required for HTP minimal benchmarks")
    if shutil.which("cmake") is None:
        return _skip_metrics(chapter, "SKIPPED: cmake is required for HTP minimal benchmarks")
    if shutil.which("ninja") is None:
        return _skip_metrics(chapter, "SKIPPED: ninja is required for HTP minimal benchmarks")
    if not PROJECT.exists():
        return _skip_metrics(chapter, f"SKIPPED: HTP minimal project is missing: {PROJECT}")
    if not ndk.exists():
        return _skip_metrics(chapter, f"SKIPPED: Android NDK not found: {ndk}")
    if not sdk.exists():
        return _skip_metrics(chapter, f"SKIPPED: Hexagon SDK not found: {sdk}")
    if not tools.exists():
        return _skip_metrics(chapter, f"SKIPPED: Hexagon tools not found: {tools}")
    return None


def _build_env(ndk: Path, sdk: Path, tools: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["ANDROID_NDK"] = str(ndk)
    env["HEXAGON_SDK_ROOT"] = str(sdk)
    env["HEXAGON_TOOLS_ROOT"] = str(tools)
    return env


def _configure_android(build_dir: Path, ndk: Path, sdk: Path, env: dict[str, str]) -> None:
    command = [
        "cmake",
        "-S",
        str(PROJECT),
        "-B",
        str(build_dir),
        "-G",
        "Ninja",
        f"-DCMAKE_TOOLCHAIN_FILE={sdk / 'build/cmake/android_toolchain.cmake'}",
        f"-DANDROID_NDK={ndk}",
        "-DANDROID_ABI=arm64-v8a",
        "-DANDROID_PLATFORM=android-28",
        "-DANDROID_NATIVE_API_LEVEL=28",
        "-DANDROID_STL=c++_shared",
        "-DOS_TYPE=HLOS",
        "-DPREBUILT_LIB_DIR=android_aarch64",
        f"-DHEXAGON_SDK_ROOT={sdk}",
        f"-DHEXAGON_CMAKE_ROOT={sdk / 'build/cmake'}",
        "-DDSP_TYPE=3",
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={build_dir / 'ship'}",
        f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY={build_dir / 'ship'}",
        f"-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY={build_dir / 'ship'}",
    ]
    _require_success(_run(command, timeout=300, env=env), command)


def _configure_hexagon(build_dir: Path, arch: str, sdk: Path, tools: Path, env: dict[str, str]) -> None:
    command = [
        "cmake",
        "-S",
        str(PROJECT),
        "-B",
        str(build_dir),
        "-G",
        "Ninja",
        f"-DCMAKE_TOOLCHAIN_FILE={sdk / 'build/cmake/hexagon_toolchain.cmake'}",
        f"-DHEXAGON_SDK_ROOT={sdk}",
        f"-DHEXAGON_TOOLS_ROOT={tools}",
        f"-DHEXAGON_CMAKE_ROOT={sdk / 'build/cmake'}",
        f"-DPREBUILT_LIB_DIR=hexagon_toolv19_{arch}",
        f"-DDSP_VERSION={arch}",
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={build_dir / 'ship'}",
        f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY={build_dir / 'ship'}",
        f"-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY={build_dir / 'ship'}",
        "-DQURT_OS=1",
    ]
    _require_success(_run(command, timeout=300, env=env), command)


def _cmake_build(build_dir: Path, target: str, env: dict[str, str]) -> None:
    command = ["cmake", "--build", str(build_dir), "--target", target, "-j2"]
    _require_success(_run(command, timeout=600, env=env), command)


def _arch_list() -> list[str]:
    raw = os.environ.get("AISP_HTP_MINIMAL_ARCHES", "v73,v75,v79,v81")
    arches = []
    for item in raw.split(","):
        arch = item.strip().lower()
        if not arch:
            continue
        if not arch.startswith("v"):
            arch = f"v{arch}"
        arches.append(arch)
    if not arches:
        raise RuntimeError("AISP_HTP_MINIMAL_ARCHES did not contain any DSP architectures")
    return arches


def build_htp_minimal_artifacts(ndk: Path, sdk: Path, tools: Path) -> list[Path]:
    build_root = _cache_dir()
    build_root.mkdir(parents=True, exist_ok=True)
    env = _build_env(ndk, sdk, tools)

    android_dir = build_root / "android"
    android_dir.mkdir(parents=True, exist_ok=True)
    _configure_android(android_dir, ndk, sdk, env)
    _cmake_build(android_dir, "aisp_htp_minimal_device", env)

    artifacts = [
        android_dir / "ship" / "aisp_htp_minimal",
        android_dir / "ship" / "libaisp_htp_minimal.so",
    ]
    libcxx = android_dir / "ship" / "libc++_shared.so"
    if libcxx.exists():
        artifacts.append(libcxx)

    for arch in _arch_list():
        hexagon_dir = build_root / f"hexagon_{arch}"
        hexagon_dir.mkdir(parents=True, exist_ok=True)
        _configure_hexagon(hexagon_dir, arch, sdk, tools, env)
        _cmake_build(hexagon_dir, "aisp_htp_minimal_skel", env)
        artifacts.append(hexagon_dir / "ship" / f"libaisp_htp_minimal-{arch}.so")

    missing = [str(path) for path in artifacts if not path.exists()]
    if missing:
        raise RuntimeError("HTP minimal build did not produce expected artifacts: " + ", ".join(missing))
    return artifacts


def _extract_float(name: str, text: str) -> float | None:
    match = re.search(rf"^{re.escape(name)}\s*=\s*([-+0-9.eE]+)", text, re.MULTILINE)
    return float(match.group(1)) if match else None


def _extract_int(name: str, text: str) -> int | None:
    match = re.search(rf"^{re.escape(name)}\s*=\s*([0-9]+)", text, re.MULTILINE)
    return int(match.group(1)) if match else None


def _extract_str(name: str, text: str) -> str | None:
    match = re.search(rf"^{re.escape(name)}\s*=\s*(\S+)", text, re.MULTILINE)
    return match.group(1) if match else None


def run_htp_minimal_profile(chapter: str) -> dict[str, Any]:
    """Build, push, and run the chapter's Hexagon HTP minimal scenario."""
    scenario = CHAPTER_SCENARIOS.get(chapter)
    if scenario is None:
        raise RuntimeError(f"No HTP minimal scenario registered for {chapter}")

    ndk = _env_path("ANDROID_NDK", DEFAULT_NDK)
    sdk = _env_path("HEXAGON_SDK_ROOT", DEFAULT_HEXAGON_SDK)
    tools = _env_path("HEXAGON_TOOLS_ROOT", DEFAULT_HEXAGON_TOOLS)
    skipped = _preflight(chapter, ndk, sdk, tools)
    if skipped is not None:
        return skipped

    iterations = _env_int("AISP_HTP_MINIMAL_ITERATIONS", scenario.iterations)
    repeats = _env_int("AISP_HTP_MINIMAL_REPEATS", scenario.repeats)
    elements = _env_int("AISP_HTP_MINIMAL_ELEMENTS", scenario.elements)
    timeout = _env_int("AISP_HTP_MINIMAL_TIMEOUT_SECONDS", 180)
    remote_dir = os.environ.get("AISP_HTP_MINIMAL_REMOTE_DIR", f"/data/local/tmp/{chapter}_htp_minimal")

    artifacts = build_htp_minimal_artifacts(ndk, sdk, tools)

    setup_cmd = ["adb", "shell", f"rm -rf {remote_dir} && mkdir -p {remote_dir}"]
    _require_success(_run(setup_cmd, timeout=timeout), setup_cmd)
    for local in artifacts:
        push_cmd = ["adb", "push", str(local), f"{remote_dir}/"]
        _require_success(_run(push_cmd, timeout=timeout), push_cmd)

    run_cmd = [
        "adb",
        "shell",
        " && ".join(
            [
                f"cd {remote_dir}",
                "chmod +x aisp_htp_minimal",
                "export LD_LIBRARY_PATH=$PWD:$LD_LIBRARY_PATH",
                "export ADSP_LIBRARY_PATH=\"$PWD;/dsp;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp\"",
                "export DSP_LIBRARY_PATH=\"$PWD;/dsp;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp\"",
                f"./aisp_htp_minimal {scenario.name} {iterations} {repeats} {elements}",
            ]
        ),
    ]
    process = _run(run_cmd, timeout=timeout)
    _require_success(process, run_cmd)
    output = f"{process.stdout}\n{process.stderr}"

    return {
        "metrics": {
            "chapter": chapter,
            "target": "htp_minimal",
            "device": "hexagon_htp",
            "status": "ok",
            "scenario": scenario.name,
            "scenario_summary": scenario.summary,
            "iterations": iterations,
            "repeats": repeats,
            "elements": elements,
            "arch": _extract_str("arch", output),
            "dsp_arch": _extract_str("dsp_arch", output),
            "hw_threads": _extract_int("hw_threads", output),
            "hvx_bytes": _extract_int("hvx_bytes", output),
            "baseline_ms": _extract_float("baseline_ms", output),
            "optimized_ms": _extract_float("optimized_ms", output),
            "speedup": _extract_float("speedup", output),
            "baseline_checksum": _extract_int("baseline_checksum", output),
            "optimized_checksum": _extract_int("optimized_checksum", output),
            "optimized_max_abs_err": _extract_float("optimized_max_abs_err", output),
            "output_tail": "\n".join(output.strip().splitlines()[-50:]),
        }
    }


def print_htp_minimal_result(result: dict[str, Any]) -> None:
    metrics = result["metrics"]
    print(f"{metrics['chapter'].upper()} HTP minimal comparison")
    if metrics.get("status") != "ok":
        print(f"  {metrics.get('reason', 'HTP run did not complete')}")
        return
    print(f"  Scenario:              {metrics['scenario']}")
    print(f"  HTP arch:              {metrics.get('arch')} (DSP reports {metrics.get('dsp_arch')})")
    print(f"  HVX bytes:             {metrics.get('hvx_bytes')}")
    print(f"  Baseline:              {metrics['baseline_ms']:.6f} ms")
    print(f"  Optimized:             {metrics['optimized_ms']:.6f} ms")
    print(f"  Speedup:               {metrics['speedup']:.2f}x")
    err = metrics.get("optimized_max_abs_err")
    if err is not None:
        print(f"  Optimized max error:   {err:.0f}")
