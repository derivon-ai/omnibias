# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Bounded one-pass versus iterative certified Gaussian inverse queries.

Every timed query constructs a fresh rigorous certificate, including posterior
moments or inverse iterations. Both methods must meet the same physical
coordinate half-width threshold and contain independent 80-digit roots.
Use --check-only while other workloads are active; it collects no timings.
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
from datetime import datetime, timezone
from fractions import Fraction as F
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import measure, rss_mb, source_provenance, write_json  # noqa: E402

REPEATS, WARMUP, ITERATIONS = 9, 2, 128
TOLERANCE, AGE = 1e-6, 1024
QUERIES = tuple(F(k, 32) for k in range(-8, 9))
WALL_SECONDS, RSS_MIB = 60.0, 512.0
METHODS = ("one_pass_predictor", "fresh_inverse")
SOURCES = [
    "benchmarks/gaussian_inverse_predictor.py", "benchmarks/_common.py",
    "packages/omnibias-core/src/omnibias/core/verified/gaussian_predictor.py",
    "packages/omnibias-core/src/omnibias/core/verified/_gaussian_moments.py",
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


def _worker(output: Path, check_only: bool) -> None:
    import mpmath as mp
    from omnibias.core.proof.certificate import (
        certificate_is_unconditional,
        verify_certificate_digest,
    )
    from omnibias.core.verified.gaussian_heat import GaussianHeatMap
    from omnibias.core.verified.gaussian_predictor import certify_gaussian_inverse_predictor
    from omnibias.core.verified.interval import Interval
    from omnibias.core.verified.inverse_gaussian_heat import inverse_gaussian_heat

    result: dict[str, Any] = {"status": "running", "phase": "construction", "methods": {}}
    _dump(output, result)
    model_build_ms = None
    start = None if check_only else time.perf_counter()
    model = GaussianHeatMap(((-1,), (1,)), (0, F(1, 3)), viscosity=1, time=AGE)
    if start is not None:
        model_build_ms = (time.perf_counter()-start)*1000
    targets = tuple((value,) for value in QUERIES)
    target_boxes = tuple((Interval.point(float(value)),) for value in QUERIES)

    def predictor_batch() -> Any:
        certificates = []
        for box in target_boxes:
            certificate = certify_gaussian_inverse_predictor(model, box, order=1)
            if certificate is None:
                raise ArithmeticError("one-pass predictor certification failed")
            certificates.append(certificate)
        return tuple(certificates)

    def inverse_batch() -> Any:
        certificates = []
        for target in targets:
            solved = inverse_gaussian_heat(model, target, initial=None, max_iterations=ITERATIONS,
                                           absolute_tolerance=TOLERANCE)
            if solved.certificate is None or solved.status != "certified":
                raise ArithmeticError("fresh inverse certification failed")
            certificates.append(solved.certificate)
        return tuple(certificates)

    def mp_exact(value: Any) -> Any:
        exact = F(value)
        return mp.mpf(exact.numerator)/exact.denominator

    def reference(target: F, age: int) -> Any:
        wanted, s = mp_exact(target), mp.mpf(age)
        # Direct scalar reduction of original Gaussian weights, independent
        # of limiting posterior moments and implementation proposal solvers.
        def residual(x: Any) -> Any:
            return x-mp.tanh(x/(2*s)+mp.mpf(1)/6)-s*wanted
        root = mp.findroot(residual, s*wanted)
        if abs(residual(root)) > mp.mpf("1e-70"):
            raise AssertionError("independent physical root missed its residual tolerance")
        return root

    functions = ((METHODS[0], predictor_batch), (METHODS[1], inverse_batch))
    with mp.workdps(80):
        references = tuple(reference(value, AGE) for value in QUERIES)
        accuracy_passed = True
        for method, function in functions:
            certificates = function()
            rows, maximum_width, maximum_error = [], F(0), mp.mpf(0)
            for target, certified, root in zip(QUERIES, certificates, references, strict=True):
                interval = certified.enclosure[0]
                half_width = (F(interval.hi)-F(interval.lo))/2
                midpoint = (F(interval.hi)+F(interval.lo))/2
                contained = mp_exact(interval.lo) <= root <= mp_exact(interval.hi)
                narrow = half_width <= F(1, 10**6)
                valid_digest = verify_certificate_digest(certified.certificate)
                unconditional = certificate_is_unconditional(certified.certificate)
                error = abs(mp_exact(midpoint)-root)
                maximum_width, maximum_error = max(maximum_width, half_width), max(maximum_error, error)
                passed = contained and narrow and valid_digest and unconditional
                accuracy_passed &= passed
                row = {
                    "target_exact": str(target), "enclosure": [interval.lo, interval.hi],
                    "half_width_exact": str(half_width), "reference_80_digits": mp.nstr(root, 80),
                    "reference_contained": contained, "half_width_at_most_1e_minus_6": narrow,
                    "midpoint_abs_error": float(error), "accuracy_passed": passed,
                    "certificate_digest": certified.certificate["digest"], "certificate_digest_valid": valid_digest,
                    "certificate_unconditional": unconditional,
                    "transcend_backend": certified.certificate["meta"]["transcend_backend"],
                }
                if method == METHODS[0]:
                    row["approximation_error_bound_exact"] = str(certified.approximation_error_bound)
                    proposal = certified.predictor_enclosure[0]
                    row["predictor_enclosure"] = [proposal.lo, proposal.hi]
                rows.append(row)
            result["methods"][method] = {
                "accuracy_passed": all(row["accuracy_passed"] for row in rows),
                "max_coordinate_half_width": float(maximum_width), "max_midpoint_abs_error": float(maximum_error),
                "queries": rows,
            }
        # Demonstrate the actual accuracy regime, without timing a looser
        # predictor against the strict baseline or claiming a speed ratio.
        diagnostic_model = GaussianHeatMap(((-1,), (1,)), (0, F(1, 3)), viscosity=1, time=16)
        diagnostic = certify_gaussian_inverse_predictor(diagnostic_model, (Interval.point(0.125),), order=1)
        if diagnostic is None:
            raise ArithmeticError("accuracy-regime diagnostic failed certification")
        box = diagnostic.enclosure[0]
        half_width = (F(box.hi)-F(box.lo))/2
        diagnostic_root = reference(F(1, 8), 16)
        diagnostic_contains = mp_exact(box.lo) <= diagnostic_root <= mp_exact(box.hi)
        if not diagnostic_contains:
            raise AssertionError("low-time predictor failed independent containment")
        result["insufficient_accuracy_diagnostic"] = {
            "time_exact": "16", "target_exact": "1/8", "order": 1,
            "approximation_error_bound_exact": str(diagnostic.approximation_error_bound),
            "enclosure": [box.lo, box.hi], "half_width_exact": str(half_width),
            "reference_80_digits": mp.nstr(diagnostic_root, 80), "reference_contained": diagnostic_contains,
            "qualifies_for_1e_minus_6_comparison": half_width <= F(1, 10**6),
            "timing_collected": False, "speedup_eligible": False,
        }

    result.update(accuracy_passed=accuracy_passed, phase="validation", model_build_ms=model_build_ms,
                  exact_contraction_bound=str(model.contraction_bound))
    _dump(output, result)
    if not accuracy_passed:
        result.update(status="accuracy_failed", worker_max_rss_mib=rss_mb())
        _dump(output, result)
        return
    if not check_only:
        for method, function in functions:
            result["phase"] = method
            _dump(output, result)
            result["methods"][method].update(measure(function, repeats=REPEATS, warmup=WARMUP))
            result["methods"][method]["median_per_query_ms"] = result["methods"][method]["median_ms"]/len(QUERIES)
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
                return int(line.split()[1])/1024
    except (FileNotFoundError, ProcessLookupError):
        pass
    return 0.0


def _isolated(check_only: bool, wall_seconds: float, rss_limit: float) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="omnibias-gaussian-inverse-predictor-") as directory:
        output = Path(directory)/"result.json"
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--result-file", str(output)]
        if check_only:
            command.append("--check-only")
        environment = os.environ.copy()
        environment.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        with (Path(directory)/"log.txt").open("w+", encoding="utf-8") as log:
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
    name = args.output_name or ("gaussian_inverse_predictor_check.json" if args.check_only else "gaussian_inverse_predictor.json")
    if Path(name).name != name or not name.endswith(".json"):
        parser.error("output-name must be a .json filename without directory components")
    sources = source_provenance(SOURCES)
    result = _isolated(args.check_only, args.wall_seconds, args.rss_mib)
    unchanged = source_provenance(SOURCES)["source_sha256"] == sources["source_sha256"]
    methods = result.get("methods", {})
    eligible = unchanged and result.get("status") == "ok" and all(
        methods.get(method, {}).get("accuracy_passed") and len(methods[method].get("samples_ms", [])) == REPEATS
        for method in METHODS)
    result["speedup_eligible"] = eligible
    if eligible:
        result["fresh_inverse_over_predictor_median_ratio"] = methods[METHODS[1]]["median_ms"]/methods[METHODS[0]]["median_ms"]
    payload: dict[str, Any] = {
        "schema": "omnibias/gaussian-inverse-predictor/v1",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "hardware_class": f"pure Python core CPU ({os.cpu_count() or 0} logical cores available; synchronous single process)",
        "versions": {"python": platform.python_version(), "platform": platform.platform(),
                     "mpmath": importlib.metadata.version("mpmath")},
        "config": {
            "mode": "correctness_only" if args.check_only else "measurement", "query_count": len(QUERIES),
            "predictor_order": 1, "max_iterations": ITERATIONS, "baseline_absolute_tolerance": TOLERANCE,
            "requested_max_coordinate_half_width_exact": "1/1000000", "reference_decimal_digits": 80,
            "warmup": 0 if args.check_only else WARMUP, "repeats": 0 if args.check_only else REPEATS,
            "wall_seconds": args.wall_seconds, "rss_limit_mib": args.rss_mib, "rss_poll_seconds": 0.025,
            "budget_scope": "one fresh subprocess including imports, model construction, accuracy, warmups and all samples",
            "timing_scope": "17 exact binary rational runtime targets per batch; fresh rigorous certificate returned for each query in both methods",
            "first_execution_scope": "first timed batch after untimed correctness validation; not process cold-start",
            "construction_scope": "same model reused by both methods; common exact model construction recorded separately and excluded equally",
            "baseline": "fresh inverse_gaussian_heat and certificate, zero default initial state, no predictor warm start or query handoff",
            "predictor": "fresh certify_gaussian_inverse_predictor(order=1) including all posterior moments and certificate sealing each query",
            "accuracy_gate": "both physical inverse enclosures contain independent 80-digit roots and every coordinate half-width <=1e-6",
            "limitations": "one fixed scalar family in a large-time accuracy regime; no tensor, derivative, training, or universal speedup claim",
            "amortization": "none; no predictor artifact or posterior is reused between queries",
        },
        "operands": {"centers_exact": [["-1"], ["1"]], "biases_exact": ["0", "1/3"],
                     "viscosity_exact": "1", "time_exact": str(AGE), "query_targets_exact": list(map(str, QUERIES)),
                     "reference_equation": "x-tanh(x/(2*time)+1/6)-time*target=0"},
        **sources, "source_unchanged_during_run": unchanged, "result": result,
    }
    print(f"{result['status']}: wrote {write_json(name, payload)}")


if __name__ == "__main__":
    main()
