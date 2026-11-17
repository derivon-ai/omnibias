# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Bounded rigorous Gaussian inverse queries at a shared enclosure accuracy.

Run after other benchmarks and test suites stop. ``--check-only`` performs
untimed construction and independent 80-digit validation without collecting
performance samples. The worker imports pure core and mpmath, not tensors.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from datetime import datetime, timezone
from fractions import Fraction as F
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import measure, rss_mb, source_provenance, write_json  # noqa: E402

REPEATS, WARMUP = 9, 2
ITERATIONS, DEGREE = 128, 24
TOLERANCE, BUILD_TOLERANCE = 1e-6, 1e-12
HALF_WIDTH, BASE_TARGET, DIRECTION = F(1, 4), F(1, 4), F(1)
QUERIES = tuple(F(k, 32) for k in range(-8, 9))
WALL_SECONDS, RSS_MIB = 60.0, 512.0
SOURCES = [
    "benchmarks/gaussian_inverse_approximation.py", "benchmarks/_common.py",
    "packages/omnibias-core/src/omnibias/core/verified/gaussian_chebyshev.py",
    "packages/omnibias-core/src/omnibias/core/verified/gaussian_tube.py",
    "packages/omnibias-core/src/omnibias/core/verified/inverse_gaussian_heat.py",
    "packages/omnibias-core/src/omnibias/core/verified/gaussian_heat.py",
    "packages/omnibias-core/src/omnibias/core/verified/mean_map.py",
    "packages/omnibias-core/src/omnibias/core/verified/interval.py",
    "packages/omnibias-core/src/omnibias/core/verified/transcend.py",
    "packages/omnibias-core/src/omnibias/core/proof/certificate.py",
    "packages/omnibias-core/src/omnibias/core/gaussian_jet.py",
    "packages/omnibias-core/src/omnibias/core/ridge_system.py",
    "packages/omnibias-core/src/omnibias/core/_rational_linear.py",
]


def _dump(path: Path, result: dict[str, Any]) -> None:
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _build(function: Callable[[], Any], check_only: bool) -> tuple[Any, float | None]:
    if check_only:
        return function(), None
    start = time.perf_counter()
    value = function()
    return value, (time.perf_counter() - start) * 1000


def _worker(output: Path, check_only: bool) -> None:
    import mpmath as mp
    from omnibias.core.proof.certificate import (
        certificate_is_unconditional,
        verify_certificate_digest,
    )
    from omnibias.core.verified.gaussian_chebyshev import approximate_gaussian_inverse
    from omnibias.core.verified.gaussian_heat import GaussianHeatMap
    from omnibias.core.verified.gaussian_tube import certify_gaussian_inverse_tube
    from omnibias.core.verified.inverse_gaussian_heat import inverse_gaussian_heat

    result: dict[str, Any] = {"status": "running", "phase": "construction", "methods": {}}
    _dump(output, result)
    model, model_ms = _build(lambda: GaussianHeatMap(((-1,), (1,)), (0, F(1, 3)), viscosity=1, time=2), check_only)
    tube, tube_ms = _build(lambda: certify_gaussian_inverse_tube(model), check_only)
    if tube is None:
        raise ArithmeticError("benchmark tube certification failed")
    approximation, approximation_ms = _build(lambda: approximate_gaussian_inverse(
        tube, (BASE_TARGET,), (DIRECTION,), half_width=HALF_WIDTH, degree=DEGREE,
        ellipse_parameter=2, max_iterations=ITERATIONS, absolute_tolerance=BUILD_TOLERANCE,
    ), check_only)
    if approximation is None:
        raise ArithmeticError("benchmark approximation construction failed")
    targets = tuple((BASE_TARGET + parameter * DIRECTION,) for parameter in QUERIES)

    def approximate_batch() -> Any:
        return tuple(approximation.enclose(parameter) for parameter in QUERIES)

    def inverse_batch() -> Any:
        boxes = []
        for target in targets:
            solved = inverse_gaussian_heat(model, target, initial=None, max_iterations=ITERATIONS,
                                           absolute_tolerance=TOLERANCE)
            if solved.certificate is None or solved.status != "certified":
                raise ArithmeticError("fresh inverse failed certification")
            boxes.append(solved.certificate.enclosure)
        return tuple(boxes)

    def mp_exact(value: Any) -> Any:
        exact = F(value)
        return mp.mpf(exact.numerator) / exact.denominator

    references = []
    with mp.workdps(80):
        for target in targets:
            wanted = mp_exact(target[0])
            # Independent closed scalar reduction: mean=tanh(x/4+1/6).
            root = mp.findroot(lambda x, wanted=wanted: x-mp.tanh(x/4+mp.mpf(1)/6)-2*wanted, 2*wanted)
            residual = abs(root-mp.tanh(root/4+mp.mpf(1)/6)-2*wanted)
            if residual > mp.mpf("1e-70"):
                raise AssertionError("independent root did not meet its residual threshold")
            references.append(root)

        accuracy_passed = True
        for method, function in (("approximation_enclose", approximate_batch), ("fresh_inverse", inverse_batch)):
            boxes = function()
            rows, maximum_width = [], F(0)
            maximum_error = mp.mpf(0)
            for parameter, target, box, reference in zip(QUERIES, targets, boxes, references, strict=True):
                interval = box[0]
                half_width = (F(interval.hi)-F(interval.lo))/2
                midpoint = (F(interval.hi)+F(interval.lo))/2
                contained = mp_exact(interval.lo) <= reference <= mp_exact(interval.hi)
                narrow = half_width <= F(1, 10**6)
                error = abs(mp_exact(midpoint)-reference)
                maximum_width, maximum_error = max(maximum_width, half_width), max(maximum_error, error)
                accuracy_passed &= contained and narrow
                rows.append({"parameter_exact": str(parameter), "target_exact": str(target[0]),
                             "enclosure": [interval.lo, interval.hi], "half_width_exact": str(half_width),
                             "reference_80_digits": mp.nstr(reference, 80), "reference_contained": contained,
                             "half_width_at_most_1e_minus_6": narrow, "midpoint_abs_error": float(error)})
            result["methods"][method] = {"accuracy_passed": all(row["reference_contained"] and row["half_width_at_most_1e_minus_6"] for row in rows),
                                         "max_coordinate_half_width": float(maximum_width),
                                         "max_midpoint_abs_error": float(maximum_error), "queries": rows}

    result.update(
        accuracy_passed=accuracy_passed, phase="validation", model_build_ms=model_ms,
        tube_build_ms=tube_ms, approximation_build_ms=approximation_ms,
        amortized_build_ms=None if check_only else tube_ms + approximation_ms,
        approximation_uniform_error_exact=[str(value) for value in approximation.uniform_error_bounds],
        approximation_certificate_digest=approximation.certificate["digest"],
        approximation_transcend_backend=approximation.certificate["meta"]["transcend_backend"],
        approximation_certificate_digest_valid=verify_certificate_digest(approximation.certificate),
        approximation_certificate_unconditional=certificate_is_unconditional(approximation.certificate),
        tube_certificate_digest=tube.certificate["digest"],
    )
    _dump(output, result)
    if not accuracy_passed:
        result.update(status="accuracy_failed", worker_max_rss_mib=rss_mb())
        _dump(output, result)
        return
    if not check_only:
        for method, function in (("approximation_enclose", approximate_batch), ("fresh_inverse", inverse_batch)):
            result["phase"] = method
            _dump(output, result)
            result["methods"][method].update(measure(function, repeats=REPEATS, warmup=WARMUP))
            result["methods"][method]["median_per_query_ms"] = result["methods"][method]["median_ms"] / len(QUERIES)
            _dump(output, result)
    if any(name in sys.modules for name in ("torch", "jax", "tensorflow")):
        raise AssertionError("pure-core benchmark unexpectedly imported a tensor backend")
    result.update(status="validated" if check_only else "ok", phase="complete", worker_max_rss_mib=rss_mb(),
                  tensor_backends_imported=False)
    _dump(output, result)


def _rss_mib(pid: int) -> float:
    try:
        for line in Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    except (FileNotFoundError, ProcessLookupError):
        pass
    return 0.0


def _isolated(check_only: bool, wall_seconds: float, rss_limit: float) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="omnibias-gaussian-inverse-approximation-") as directory:
        output = Path(directory) / "result.json"
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--result-file", str(output)]
        if check_only:
            command.append("--check-only")
        environment = os.environ.copy()
        environment.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        with (Path(directory) / "log.txt").open("w+", encoding="utf-8") as log:
            start = time.monotonic()
            process = subprocess.Popen(command, stdout=log, stderr=log, env=environment, start_new_session=True)
            peak, status = 0.0, None
            while process.poll() is None:
                peak = max(peak, _rss_mib(process.pid))
                if peak > rss_limit:
                    status = "memory_budget"
                elif time.monotonic()-start > wall_seconds:
                    status = "timeout"
                if status is not None:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    break
                time.sleep(0.025)
            process.wait()
            result = json.loads(output.read_text(encoding="utf-8")) if output.exists() else {}
            if status is not None:
                result.update(status=status, detail="parent terminated worker at configured resource limit")
            elif process.returncode:
                log.seek(0)
                result.update(status="error", detail=log.read()[-4000:])
            elif result.get("status") not in ("ok", "validated", "accuracy_failed"):
                result.update(status="error", detail="worker did not complete")
            result.update(observed_peak_rss_mib=peak, subprocess_wall_s=time.monotonic()-start,
                          process_returncode=process.returncode)
            return result


def _positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--wall-seconds", type=_positive_float, default=WALL_SECONDS)
    parser.add_argument("--rss-mib", type=_positive_float, default=RSS_MIB)
    parser.add_argument("--output-name")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--result-file", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        if args.result_file is None:
            parser.error("worker requires --result-file")
        _worker(args.result_file, args.check_only)
        return
    if not Path("/proc/self/status").exists():
        parser.error("bounded experiment requires Linux /proc RSS monitoring")
    name = args.output_name or ("gaussian_inverse_approximation_check.json" if args.check_only else "gaussian_inverse_approximation.json")
    if Path(name).name != name or not name.endswith(".json"):
        parser.error("output-name must be a .json filename without directory components")
    sources = source_provenance(SOURCES)
    result = _isolated(args.check_only, args.wall_seconds, args.rss_mib)
    unchanged = source_provenance(SOURCES)["source_sha256"] == sources["source_sha256"]
    methods = result.get("methods", {})
    eligible = unchanged and result.get("status") == "ok" and all(
        methods.get(method, {}).get("accuracy_passed") and len(methods[method].get("samples_ms", [])) == REPEATS
        for method in ("approximation_enclose", "fresh_inverse"))
    result["speedup_eligible"] = eligible
    if eligible:
        approximation_ms = methods["approximation_enclose"]["median_ms"]
        baseline_ms = methods["fresh_inverse"]["median_ms"]
        result["fresh_inverse_over_approximation_median_ratio"] = baseline_ms / approximation_ms
        saved_per_query = (baseline_ms-approximation_ms) / len(QUERIES)
        result["estimated_break_even_queries"] = (math.ceil(result["amortized_build_ms"] / saved_per_query)
                                                   if saved_per_query > 0 else None)
    payload: dict[str, Any] = {
        "schema": "omnibias/gaussian-inverse-approximation/v1",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "hardware_class": f"pure Python core CPU ({os.cpu_count() or 0} logical cores available; synchronous single process)",
        "versions": {"python": platform.python_version(), "platform": platform.platform(),
                     "mpmath": importlib.metadata.version("mpmath")},
        "config": {
            "mode": "correctness_only" if args.check_only else "measurement", "query_count": len(QUERIES),
            "degree": DEGREE, "ellipse_parameter_exact": "2", "max_iterations": ITERATIONS,
            "requested_max_coordinate_half_width_exact": "1/1000000", "baseline_absolute_tolerance": TOLERANCE,
            "build_sample_absolute_tolerance": BUILD_TOLERANCE, "reference_decimal_digits": 80,
            "warmup": 0 if args.check_only else WARMUP, "repeats": 0 if args.check_only else REPEATS,
            "wall_seconds": args.wall_seconds, "rss_limit_mib": args.rss_mib, "rss_poll_seconds": 0.025,
            "budget_scope": "one fresh subprocess including imports, construction, accuracy, warmups and all samples",
            "timing_scope": "17 exact rational runtime queries per batch; intervals returned by both methods",
            "first_execution_scope": "first timed batch after untimed independent correctness validation; not process cold-start",
            "construction_scope": "model common to both; tube plus approximation build charged separately to approximation",
            "baseline": "fresh inverse_gaussian_heat call and certificate per query; same fixed model, default zero initial state, no warm start",
            "approximation": "rigorous outward approximation.enclose for every runtime parameter; no tensor evaluator",
            "accuracy_gate": "both enclosures contain independent 80-digit scalar roots and every coordinate half-width <=1e-6",
            "limitations": "one fixed scalar family and target segment; no derivative, training, universal speedup, or fastest-possible-solver claim",
            "break_even_scope": "tube+approximation build divided by warmed median savings per query; excludes common model build and independent validation",
        },
        "operands": {"centers_exact": [["-1"], ["1"]], "biases_exact": ["0", "1/3"],
                     "viscosity_exact": "1", "time_exact": "2", "base_target_exact": str(BASE_TARGET),
                     "direction_exact": str(DIRECTION), "half_width_exact": str(HALF_WIDTH),
                     "query_parameters_exact": list(map(str, QUERIES)),
                     "reference_equation": "x-tanh(x/4+1/6)-2*(base_target+t*direction)=0"},
        **sources, "source_unchanged_during_run": unchanged, "result": result,
    }
    print(f"{result['status']}: wrote {write_json(name, payload)}")


if __name__ == "__main__":
    main()
