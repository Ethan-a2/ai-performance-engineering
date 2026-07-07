"""Fair llama.cpp HTP-vs-CUDA benchmark helpers."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_LLAMA_CPP = Path("/media/code/llm/llama/llama.cpp")
FAIR_TARGET = "llama-fair-htp-cuda-bench"
FAIR_SOURCE = Path("examples/fair-htp-cuda-bench/fair-htp-cuda-bench.cpp")


@dataclass(frozen=True)
class HtpCudaFairScenario:
    case_name: str
    summary: str
    cuda_optimization: str
    in_features: int
    out_features: int
    iterations: int = 8
    warmup: int = 2
    abs_tol: float = 0.08
    rel_tol: float = 0.05


CHAPTER_SCENARIOS: dict[str, HtpCudaFairScenario] = {
    "ch01": HtpCudaFairScenario("mm_decode_f16", "backend matmul mirrors the chapter's scalar-to-GEMM acceleration", "looped small GEMMs -> single backend matmul", 1024, 1024, iterations=5),
    "ch02": HtpCudaFairScenario("mm_decode_f16", "matmul shape keeps the CUDA hardware-aware vectorization theme", "looped small GEMMs -> single backend matmul", 1024, 1024, iterations=5),
    "ch03": HtpCudaFairScenario("mm_decode_f16", "same graph on both backends isolates runtime launch/provisioning overhead", "looped small GEMMs -> single backend matmul", 1024, 1024, iterations=5),
    "ch04": HtpCudaFairScenario("elementwise_fusion_f32", "fused binary chain mirrors CUDA kernel/operator fusion", "three separate tensor ops -> one fused expression", 512, 512),
    "ch05": HtpCudaFairScenario("copy_f16_to_f32", "bulk typed copy mirrors vectorized preprocessing and IO staging", "chunked copy loop -> single bulk device copy", 1024, 1024),
    "ch06": HtpCudaFairScenario("elementwise_fusion_f32", "fused HTP/CUDA graph replaces poorly amortized scalar work", "three separate tensor ops -> one fused expression", 512, 512),
    "ch07": HtpCudaFairScenario("copy_f16_to_f32", "contiguous typed copy mirrors coalesced memory movement", "chunked copy loop -> single bulk device copy", 1024, 1024),
    "ch08": HtpCudaFairScenario("elementwise_fusion_f32", "branch-light fusion mirrors ILP and warp-efficiency tuning", "three separate tensor ops -> one fused expression", 512, 512),
    "ch09": HtpCudaFairScenario("elementwise_fusion_f32", "fusion reduces memory traffic on the same tensor shape", "three separate tensor ops -> one fused expression", 512, 512),
    "ch10": HtpCudaFairScenario("mm_decode_f16", "backend matmul mirrors compiler-guided tensor pipeline tuning", "looped small GEMMs -> single backend matmul", 1024, 1024, iterations=5),
    "ch11": HtpCudaFairScenario("kv_set_rows_f16", "row-wise KV cache update mirrors CUDA stream/concurrency cache batching", "per-token KV row update -> block vectorized update", 1024, 64),
    "ch12": HtpCudaFairScenario("kv_set_rows_f16", "single graph KV row update mirrors launch amortization", "per-token KV row update -> block vectorized update", 1024, 64),
    "ch13": HtpCudaFairScenario("kv_set_rows_f16", "cache-row writeback mirrors paged/cache-aware memory behavior", "per-token KV row update -> block vectorized update", 1024, 64),
    "ch14": HtpCudaFairScenario("mm_decode_f16", "prebuilt backend matmul mirrors compiler-specialized persistent kernels", "looped small GEMMs -> single backend matmul", 1024, 1024, iterations=5),
    "ch15": HtpCudaFairScenario("kv_set_rows_f16", "batched KV writeback mirrors pooled cache orchestration", "per-token KV row update -> block vectorized update", 1024, 64),
    "ch16": HtpCudaFairScenario("mm_decode_f16", "backend-specialized matmul mirrors production runtime specialization", "looped small GEMMs -> single backend matmul", 1024, 1024, iterations=5),
    "ch17": HtpCudaFairScenario("kv_set_rows_f16", "KV row handoff mirrors prefill/decode movement reduction", "per-token KV row update -> block vectorized update", 1024, 64),
    "ch18": HtpCudaFairScenario("kv_set_rows_f16", "active-window KV writeback mirrors cache-aware decode", "per-token KV row update -> block vectorized update", 1024, 64),
    "ch19": HtpCudaFairScenario("copy_f16_to_f32", "typed copy mirrors lower-precision/cache traffic reduction", "chunked copy loop -> single bulk device copy", 1024, 1024),
    "ch20": HtpCudaFairScenario("elementwise_fusion_f32", "composed fusion case mirrors end-to-end optimization composition", "three separate tensor ops -> one fused expression", 512, 512),
}


class FairSkipped(RuntimeError):
    """Raised when a requested fair backend is not available on this host."""


def _repo_code_dir() -> Path:
    return Path(__file__).resolve().parents[2]


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else default


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer, got {value!r}") from exc
    if parsed < 0:
        raise RuntimeError(f"{name} must be non-negative, got {parsed}")
    return parsed


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    try:
        parsed = float(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a float, got {value!r}") from exc
    if parsed < 0.0:
        raise RuntimeError(f"{name} must be non-negative, got {parsed}")
    return parsed


def _env_list(name: str, default: str) -> list[str]:
    raw = os.environ.get(name, default)
    values = [item.strip().lower() for item in raw.split(",") if item.strip()]
    if not values:
        raise RuntimeError(f"{name} did not contain any backend names")
    allowed = {"htp", "hexagon", "cuda", "cpu"}
    unknown = sorted(set(values).difference(allowed))
    if unknown:
        raise RuntimeError(f"{name} contains unsupported backend(s): {', '.join(unknown)}")
    return ["htp" if value == "hexagon" else value for value in values]


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


def _skip_result(backend: str, reason: str) -> dict[str, Any]:
    return {"backend": backend, "status": "skipped", "reason": reason}


def _needs_rebuild(binary: Path, sources: list[Path]) -> bool:
    if os.environ.get("AISP_HTP_CUDA_FAIR_REBUILD", "").strip().lower() in {"1", "true", "yes", "on"}:
        return True
    if not binary.exists():
        return True
    binary_mtime = binary.stat().st_mtime
    return any(source.exists() and source.stat().st_mtime > binary_mtime for source in sources)


def _llama_cpp_root() -> Path:
    root = _env_path("AISP_HTP_CUDA_FAIR_LLAMA_CPP", DEFAULT_LLAMA_CPP)
    if not root.exists():
        raise FairSkipped(f"SKIPPED: llama.cpp root not found: {root}")
    return root


def _host_binary(root: Path, backend: str, timeout: int) -> Path:
    override = os.environ.get("AISP_HTP_CUDA_FAIR_HOST_BINARY")
    if override:
        binary = Path(override).expanduser()
        if not binary.exists():
            raise FairSkipped(f"SKIPPED: host fair binary not found: {binary}")
        return binary

    default_build = "build-fair-cuda" if backend == "cuda" else "build-fair-cpu"
    build_dir = _env_path("AISP_HTP_CUDA_FAIR_HOST_BUILD_DIR", root / default_build)
    binary = build_dir / "bin" / FAIR_TARGET
    sources = [root / FAIR_SOURCE, root / "examples/fair-htp-cuda-bench/CMakeLists.txt"]
    if not _needs_rebuild(binary, sources):
        return binary

    if shutil.which("cmake") is None:
        raise FairSkipped("SKIPPED: cmake is required to build the host fair benchmark")
    if backend == "cuda" and shutil.which("nvcc") is None:
        raise FairSkipped("SKIPPED: nvcc is required to build the CUDA fair benchmark")

    configure = [
        "cmake",
        "-B",
        str(build_dir),
        "-DGGML_NATIVE=OFF",
        f"-DGGML_CUDA={'ON' if backend == 'cuda' else 'OFF'}",
        "-DGGML_OPENCL=OFF",
        "-DGGML_HEXAGON=OFF",
        "-DLLAMA_BUILD_TESTS=OFF",
        "-DLLAMA_BUILD_SERVER=OFF",
        "-DLLAMA_CURL=OFF",
        "-DCMAKE_BUILD_TYPE=Release",
    ]
    _require_success(_run(configure, cwd=root, timeout=timeout), configure)
    jobs = str(_env_int("AISP_HTP_CUDA_FAIR_BUILD_JOBS", 16))
    build = ["cmake", "--build", str(build_dir), "--target", FAIR_TARGET, f"-j{jobs}"]
    _require_success(_run(build, cwd=root, timeout=timeout), build)
    if not binary.exists():
        raise RuntimeError(f"host fair binary was not produced: {binary}")
    return binary


def _snapdragon_binary(root: Path, timeout: int) -> Path:
    build_dir = _env_path("AISP_HTP_CUDA_FAIR_SNAPDRAGON_BUILD_DIR", root / "build-snapdragon")
    binary = build_dir / "bin" / FAIR_TARGET
    sources = [root / FAIR_SOURCE, root / "examples/fair-htp-cuda-bench/CMakeLists.txt"]
    if not _needs_rebuild(binary, sources):
        return binary
    if shutil.which("cmake") is None:
        raise FairSkipped("SKIPPED: cmake is required to build the Snapdragon fair benchmark")

    configure = ["cmake", "--preset", "arm64-android-snapdragon-release", "-B", str(build_dir), "-DGGML_OPENCL=OFF"]
    _require_success(_run(configure, cwd=root, timeout=timeout), configure)
    jobs = str(_env_int("AISP_HTP_CUDA_FAIR_BUILD_JOBS", 16))
    build = ["cmake", "--build", str(build_dir), "--target", FAIR_TARGET, f"-j{jobs}"]
    _require_success(_run(build, cwd=root, timeout=timeout), build)
    if not binary.exists():
        raise RuntimeError(f"Snapdragon fair binary was not produced: {binary}")
    return binary


def _snapdragon_artifacts(binary: Path) -> list[Path]:
    build_dir = binary.parents[1]
    artifacts = [
        binary,
        build_dir / "bin" / "libggml.so",
        build_dir / "bin" / "libggml-base.so",
        build_dir / "bin" / "libggml-cpu.so",
        build_dir / "bin" / "libggml-hexagon.so",
    ]
    artifacts.extend(sorted((build_dir / "ggml/src/ggml-hexagon").glob("libggml-htp-v*.so")))
    missing = [path for path in artifacts if not path.exists()]
    if missing:
        raise RuntimeError("missing Snapdragon artifact(s): " + ", ".join(str(path) for path in missing))
    return artifacts


def _bench_args(scenario: HtpCudaFairScenario, backend: str, seed: int, iterations: int, warmup: int, abs_tol: float, rel_tol: float) -> list[str]:
    return [
        "--backend",
        backend,
        "--case",
        scenario.case_name,
        "--in",
        str(scenario.in_features),
        "--out",
        str(scenario.out_features),
        "--warmup",
        str(warmup),
        "--iterations",
        str(iterations),
        "--seed",
        str(seed),
        "--abs-tol",
        str(abs_tol),
        "--rel-tol",
        str(rel_tol),
        "--fail-on-fallback",
    ]


def _parse_json_payload(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        tail = "\n".join(text.splitlines()[-80:])
        raise RuntimeError(f"fair benchmark did not produce JSON\n{tail}")
    return json.loads(text[start : end + 1])


def _run_host_backend(
    root: Path,
    backend: str,
    scenario: HtpCudaFairScenario,
    *,
    seed: int,
    iterations: int,
    warmup: int,
    abs_tol: float,
    rel_tol: float,
    timeout: int,
) -> dict[str, Any]:
    binary = _host_binary(root, backend, timeout)
    with tempfile.TemporaryDirectory(prefix="aisp_htp_cuda_fair_") as tmp:
        json_path = Path(tmp) / f"{backend}.json"
        command = [str(binary), *_bench_args(scenario, backend, seed, iterations, warmup, abs_tol, rel_tol), "--json", str(json_path)]
        process = _run(command, cwd=root, timeout=timeout)
        if json_path.exists():
            result = json.loads(json_path.read_text())
        else:
            result = _parse_json_payload(f"{process.stdout}\n{process.stderr}")
        result["runner_returncode"] = process.returncode
        result["runner"] = "host"
        result["binary"] = str(binary)
        if process.returncode != 0 and result.get("status") == "ok":
            result["status"] = "invalid"
            result["reason"] = f"host fair benchmark returned rc={process.returncode}"
        return result


def _push_android_artifacts(artifacts: list[Path], remote_dir: str, timeout: int) -> None:
    if shutil.which("adb") is None:
        raise FairSkipped("SKIPPED: adb is required for HTP fair benchmarks")
    mkdir_cmd = ["adb", "shell", f"mkdir -p {shlex.quote(remote_dir)}"]
    _require_success(_run(mkdir_cmd, timeout=timeout), mkdir_cmd)
    for artifact in artifacts:
        push_cmd = ["adb", "push", str(artifact), f"{remote_dir}/"]
        _require_success(_run(push_cmd, timeout=timeout), push_cmd)


def _run_android_htp_backend(
    root: Path,
    scenario: HtpCudaFairScenario,
    chapter: str,
    *,
    seed: int,
    iterations: int,
    warmup: int,
    abs_tol: float,
    rel_tol: float,
    timeout: int,
) -> dict[str, Any]:
    binary = _snapdragon_binary(root, timeout)
    remote_dir = os.environ.get("AISP_HTP_CUDA_FAIR_REMOTE_DIR", f"/data/local/tmp/{chapter}_htp_cuda_fair")
    _push_android_artifacts(_snapdragon_artifacts(binary), remote_dir, timeout)

    remote_args = [
        f"./{FAIR_TARGET}",
        *_bench_args(scenario, "htp", seed, iterations, warmup, abs_tol, rel_tol),
        "--json",
        "result.json",
    ]
    run_line = " ".join(shlex.quote(part) for part in remote_args)
    shell = " && ".join(
        [
            f"cd {shlex.quote(remote_dir)}",
            f"chmod +x {FAIR_TARGET}",
            "export LD_LIBRARY_PATH=$PWD:$LD_LIBRARY_PATH",
            "export ADSP_LIBRARY_PATH=\"$PWD;/dsp;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp\"",
            "export DSP_LIBRARY_PATH=\"$PWD;/dsp;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp\"",
            f"{run_line} > runner.log 2>&1; rc=$?; cat result.json 2>/dev/null; echo __AISP_RC__:$rc; echo __AISP_LOG__; cat runner.log 2>/dev/null; exit 0",
        ]
    )
    process = _run(["adb", "shell", shell], timeout=timeout)
    _require_success(process, ["adb", "shell", shell])
    result_text, _, trailer = process.stdout.partition("__AISP_RC__:")
    result = _parse_json_payload(result_text)
    rc_text = trailer.splitlines()[0].strip() if trailer else "0"
    try:
        runner_rc = int(rc_text)
    except ValueError:
        runner_rc = 0
    result["runner_returncode"] = runner_rc
    result["runner"] = "adb"
    result["binary"] = str(binary)
    result["remote_dir"] = remote_dir
    if runner_rc != 0 and result.get("status") == "ok":
        result["status"] = "invalid"
        result["reason"] = f"Android fair benchmark returned rc={runner_rc}"
    return result


def _latency_mean(result: dict[str, Any] | None) -> float | None:
    if not result or result.get("status") != "ok":
        return None
    latency = result.get("latency_ms")
    if not isinstance(latency, dict):
        return None
    value = latency.get("mean")
    return float(value) if value is not None else None


def _status_for_results(results: dict[str, dict[str, Any]]) -> str:
    if not results:
        return "skipped"
    statuses = {backend: result.get("status") for backend, result in results.items()}
    if all(status == "ok" for status in statuses.values()):
        return "ok"
    if any(status == "ok" for status in statuses.values()):
        return "partial"
    if any(status == "invalid" for status in statuses.values()):
        return "invalid"
    return "skipped"


def run_htp_cuda_fair_profile(chapter: str) -> dict[str, Any]:
    """Run the chapter's fair llama.cpp HTP-vs-CUDA comparison."""
    scenario = CHAPTER_SCENARIOS.get(chapter)
    if scenario is None:
        raise RuntimeError(f"No HTP/CUDA fair scenario registered for {chapter}")

    root = _llama_cpp_root()
    seed = _env_int("AISP_HTP_CUDA_FAIR_SEED", 20260707)
    iterations = _env_int("AISP_HTP_CUDA_FAIR_ITERATIONS", scenario.iterations)
    warmup = _env_int("AISP_HTP_CUDA_FAIR_WARMUP", scenario.warmup)
    abs_tol = _env_float("AISP_HTP_CUDA_FAIR_ABS_TOL", scenario.abs_tol)
    rel_tol = _env_float("AISP_HTP_CUDA_FAIR_REL_TOL", scenario.rel_tol)
    timeout = _env_int("AISP_HTP_CUDA_FAIR_TIMEOUT_SECONDS", 900)
    backends = _env_list("AISP_HTP_CUDA_FAIR_BACKENDS", "htp,cuda")

    results: dict[str, dict[str, Any]] = {}
    for backend in backends:
        try:
            if backend == "htp":
                results[backend] = _run_android_htp_backend(
                    root,
                    scenario,
                    chapter,
                    seed=seed,
                    iterations=iterations,
                    warmup=warmup,
                    abs_tol=abs_tol,
                    rel_tol=rel_tol,
                    timeout=timeout,
                )
            else:
                results[backend] = _run_host_backend(
                    root,
                    backend,
                    scenario,
                    seed=seed,
                    iterations=iterations,
                    warmup=warmup,
                    abs_tol=abs_tol,
                    rel_tol=rel_tol,
                    timeout=timeout,
                )
        except FairSkipped as exc:
            results[backend] = _skip_result(backend, str(exc))
        except Exception as exc:
            results[backend] = {"backend": backend, "status": "invalid", "reason": f"INVALID: {exc}"}

    htp_ms = _latency_mean(results.get("htp"))
    cuda_ms = _latency_mean(results.get("cuda"))
    speedup = (cuda_ms / htp_ms) if htp_ms and cuda_ms else None
    metrics: dict[str, Any] = {
        "chapter": chapter,
        "target": "htp_cuda_fair",
        "status": _status_for_results(results),
        "scenario": scenario.case_name,
        "scenario_summary": scenario.summary,
        "cuda_optimization": scenario.cuda_optimization,
        "shape": {"in": scenario.in_features, "out": scenario.out_features},
        "iterations": iterations,
        "warmup": warmup,
        "seed": seed,
        "abs_tol": abs_tol,
        "rel_tol": rel_tol,
        "backends_requested": backends,
        "backend_results": results,
        "htp_ms": htp_ms,
        "cuda_ms": cuda_ms,
        "baseline_label": "cuda",
        "optimized_label": "htp",
        "baseline_ms": cuda_ms,
        "optimized_ms": htp_ms,
        "speedup": speedup,
        "fair_contract": {
            "same_shape": True,
            "same_seed": True,
            "same_precision_budget": True,
            "same_boundary": "input_update+graph_compute+synchronize+output_readback",
            "fail_on_fallback": True,
        },
        "llama_cpp_root": str(root),
    }
    if metrics["status"] != "ok":
        metrics["reason"] = "; ".join(
            f"{backend}: {result.get('reason', result.get('status'))}" for backend, result in results.items() if result.get("status") != "ok"
        )
    return {"metrics": metrics}


def print_htp_cuda_fair_result(result: dict[str, Any]) -> None:
    metrics = result["metrics"]
    print(f"{metrics['chapter'].upper()} HTP/CUDA fair comparison")
    print(f"  Status:                {metrics.get('status')}")
    print(f"  Scenario:              {metrics['scenario']}")
    print(f"  CUDA-like optimization:{metrics['cuda_optimization']}")
    print(f"  Shape:                 in={metrics['shape']['in']} out={metrics['shape']['out']}")
    for backend in metrics.get("backends_requested", []):
        backend_result = metrics["backend_results"].get(backend, {})
        status = backend_result.get("status")
        latency = _latency_mean(backend_result)
        if latency is None:
            print(f"  {backend.upper():<6} {status}: {backend_result.get('reason', 'no latency')}")
        else:
            device = backend_result.get("backend_description", backend_result.get("backend_device", backend))
            print(f"  {backend.upper():<6} {latency:.6f} ms ({device})")
    if metrics.get("speedup") is not None:
        print(f"  HTP vs CUDA speedup:   {metrics['speedup']:.2f}x")
    elif metrics.get("reason"):
        print(f"  Note:                  {metrics['reason']}")
