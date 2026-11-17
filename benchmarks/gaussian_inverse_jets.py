# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Bounded Torch CPU inverse Taylor jets versus nested implicit differentiation.

Run ``uv run python benchmarks/gaussian_inverse_jets.py`` after other benchmarks
and test suites stop. Each method/order runs in a fresh, monitored subprocess.
Use ``--check-only --orders 1,2`` for a small untimed correctness smoke.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from fractions import Fraction as F
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import provenance, rss_mb, source_provenance, write_json  # noqa: E402

METHODS = ("forward_jet", "nested_ad")
ORDERS = (1, 2, 4, 6, 8)
REPEATS, WARMUP = 9, 2
ITERATIONS, TOLERANCE = 128, 1e-13
RTOL, ATOL = 2e-10, 2e-12
WALL_SECONDS, RSS_MIB = 60.0, 1536.0
SOURCES = [
    "benchmarks/gaussian_inverse_jets.py", "benchmarks/_common.py",
    "packages/omnibias-torch/src/omnibias/torch/gaussian_jet.py",
    "packages/omnibias-torch/src/omnibias/torch/gaussian_heat.py",
    "packages/omnibias-core/src/omnibias/core/gaussian_jet.py",
    "packages/omnibias-core/src/omnibias/core/_rational_linear.py",
    "packages/omnibias-core/src/omnibias/core/ridge_system.py",
    "packages/omnibias-core/src/omnibias/core/verified/gaussian_heat.py",
    "packages/omnibias-core/src/omnibias/core/verified/mean_map.py",
]
CENTERS = ((-F(1, 2),), (-F(1, 4),), (F(3, 4),))
BIASES = (F(9, 64), F(1, 16), F(1, 16))
BASE_POINT = (F(1, 4),)
PATH = (F(1, 4), F(3, 8), -F(1, 16), F(1, 32))


def _dump(path: Path, payload: dict[str, Any]) -> None:
    """Checkpoint outside measured intervals, using an atomic local replacement."""
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _accuracy(actual: Any, expected: Any) -> dict[str, Any]:
    import torch

    finite = bool(torch.isfinite(actual).all())
    difference = (actual - expected).abs()
    passed = finite and bool((difference <= ATOL + RTOL * expected.abs()).all())
    return {
        "passed": passed, "all_finite": finite,
        "max_abs_error": float(difference.max()) if finite else None,
        "max_tolerance_ratio": float((difference / (ATOL + RTOL * expected.abs())).max()) if finite else None,
        "ordinary_coefficients": [float(value) if math.isfinite(float(value)) else str(float(value))
                                  for value in actual.reshape(-1)],
    }


def _worker(method: str, order: int, output: Path, check_only: bool) -> None:
    import torch
    from omnibias.core.gaussian_jet import GaussianInverseSeries
    from omnibias.core.verified.gaussian_heat import GaussianHeatMap
    from omnibias.torch.gaussian_heat import gaussian_heat_inverse
    from omnibias.torch.gaussian_jet import gaussian_heat_inverse_jet

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    dtype = torch.float64
    target = tuple((PATH[n] if n < len(PATH) else F(0),) for n in range(order + 1))
    model = GaussianHeatMap(CENTERS, BIASES, viscosity=1, time=1)
    reference = GaussianInverseSeries.from_equal_logits(model, BASE_POINT, target[1:])
    if reference.target_coefficients != target or any(row == (0,) for row in reference.coefficients):
        raise AssertionError("benchmark must use the declared nonzero, nonsymmetric exact coefficients")
    coefficients = torch.tensor([[float(value) for value in row] for row in target], dtype=dtype)
    centers = torch.tensor([[float(value) for value in row] for row in CENTERS], dtype=dtype)
    biases = torch.tensor([float(value) for value in BIASES], dtype=dtype)
    path = torch.tensor([float(value) for value in PATH], dtype=dtype)
    expected = torch.tensor([[float(value) for value in row] for row in reference.coefficients], dtype=dtype)

    def forward_jet() -> Any:
        with torch.no_grad():
            result = gaussian_heat_inverse_jet(coefficients, centers, biases, max_iterations=ITERATIONS,
                                               tolerance=TOLERANCE, adjoint="auto")
        return result.coefficients

    def nested_ad() -> Any:
        parameter = torch.zeros((), dtype=dtype, requires_grad=True)
        goal = path[-1]
        for coefficient in reversed(path[:-1]):
            goal = goal * parameter + coefficient
        value = gaussian_heat_inverse(goal.reshape(1), centers, biases, max_iterations=ITERATIONS,
                                      tolerance=TOLERANCE, adjoint="auto").value
        result = [value.detach()]
        for n in range(1, order + 1):
            value = torch.autograd.grad(value.sum(), parameter, create_graph=n < order)[0]
            result.append((value / math.factorial(n)).detach().reshape(1))
        return torch.stack(result)

    function: Callable[[], Any] = forward_jet if method == "forward_jet" else nested_ad
    result: dict[str, Any] = {
        "status": "running", "method": method, "order": order, "phase": "first_execution",
        "torch_version": torch.__version__, "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "exact_reference": [str(row[0]) for row in reference.coefficients],
        "exact_base_target": str(reference.target_coefficients[0][0]),
        "exact_posterior": [str(value) for value in reference.posterior],
        "exact_contraction_bound": str(reference.contraction_bound),
        "samples_ms": [], "warmup_completed": 0,
    }
    _dump(output, result)
    if check_only:
        actual = function()
        result["accuracy"] = _accuracy(actual, expected)
        result.update(status="validated" if result["accuracy"]["passed"] else "accuracy_failed", phase="complete")
        result["worker_max_rss_mib"] = rss_mb()
        _dump(output, result)
        return

    start = time.perf_counter()
    actual = function()
    result["first_execution_ms"] = (time.perf_counter() - start) * 1000
    result["accuracy"] = _accuracy(actual, expected)
    if not result["accuracy"]["passed"]:
        result.update(status="accuracy_failed", phase="complete", worker_max_rss_mib=rss_mb())
        _dump(output, result)
        return
    result["phase"] = "warmup"
    _dump(output, result)
    for n in range(WARMUP):
        function()
        result["warmup_completed"] = n + 1
        _dump(output, result)
    result["phase"] = "samples"
    _dump(output, result)
    for _ in range(REPEATS):
        start = time.perf_counter()
        actual = function()
        result["samples_ms"].append((time.perf_counter() - start) * 1000)
        accuracy = _accuracy(actual, expected)
        if not accuracy["passed"]:
            result.update(status="accuracy_failed", accuracy=accuracy, phase="complete", worker_max_rss_mib=rss_mb())
            _dump(output, result)
            return
        result["accuracy"] = accuracy
        _dump(output, result)
    result.update(status="ok", phase="complete", median_ms=statistics.median(result["samples_ms"]),
                  worker_max_rss_mib=rss_mb())
    _dump(output, result)


def _rss_mib(pid: int) -> float:
    try:
        for line in Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    except (FileNotFoundError, ProcessLookupError):
        pass
    return 0.0


def _isolated(method: str, order: int, wall_seconds: float, rss_limit: float, check_only: bool) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="omnibias-gaussian-inverse-jets-") as directory:
        output = Path(directory) / "result.json"
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", method,
                   "--order", str(order), "--result-file", str(output)]
        if check_only:
            command.append("--check-only")
        environment = os.environ.copy()
        environment.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                           OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
        with (Path(directory) / "log.txt").open("w+", encoding="utf-8") as log:
            start = time.monotonic()
            process = subprocess.Popen(command, stdout=log, stderr=log, env=environment, start_new_session=True)
            peak, status = 0.0, None
            while process.poll() is None:
                peak = max(peak, _rss_mib(process.pid))
                if peak > rss_limit:
                    status = "memory_budget"
                elif time.monotonic() - start > wall_seconds:
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
                result.update(status=status, detail="parent terminated the subprocess at its configured budget")
            elif process.returncode != 0:
                log.seek(0)
                result.update(status="error", detail=log.read()[-4000:])
            elif result.get("status") not in ("ok", "validated", "accuracy_failed"):
                result.update(status="error", detail="worker did not produce a completed result")
            result.update(observed_peak_rss_mib=peak, subprocess_wall_s=time.monotonic() - start,
                          process_returncode=process.returncode)
            return result


def _positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orders", default=",".join(map(str, ORDERS)))
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--wall-seconds", type=_positive_float, default=WALL_SECONDS)
    parser.add_argument("--rss-mib", type=_positive_float, default=RSS_MIB)
    parser.add_argument("--check-only", action="store_true", help="one untimed accuracy evaluation per method/order")
    parser.add_argument("--output-name", help="JSON filename under OMNIBIAS_SCRATCH or artifacts/")
    parser.add_argument("--worker", choices=METHODS, help=argparse.SUPPRESS)
    parser.add_argument("--order", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--result-file", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        if args.order not in ORDERS or args.result_file is None:
            parser.error("worker requires a supported --order and --result-file")
        _worker(args.worker, args.order, args.result_file, args.check_only)
        return
    if not Path("/proc/self/status").exists():
        parser.error("the bounded benchmark requires Linux /proc RSS monitoring")
    try:
        orders = tuple(int(value) for value in args.orders.split(","))
    except ValueError:
        parser.error("orders must be a comma-separated subset of 1,2,4,6,8")
    methods = tuple(args.methods.split(","))
    if not orders or len(set(orders)) != len(orders) or any(order not in ORDERS for order in orders):
        parser.error("orders must be a comma-separated subset of 1,2,4,6,8 without duplicates")
    if not methods or len(set(methods)) != len(methods) or any(method not in METHODS for method in methods):
        parser.error("methods must be a comma-separated subset of forward_jet,nested_ad without duplicates")
    name = args.output_name or ("gaussian_inverse_jets_check.json" if args.check_only else "gaussian_inverse_jets.json")
    if Path(name).name != name or not name.endswith(".json"):
        parser.error("output-name must be a .json filename, without directory components")
    sources = source_provenance(SOURCES)
    rows = []
    for order in orders:
        row: dict[str, Any] = {"order": order, "methods": {}}
        for method in methods:
            result = _isolated(method, order, args.wall_seconds, args.rss_mib, args.check_only)
            row["methods"][method] = result
            print(f"N={order} {method}: {result['status']} {result.get('median_ms', '')}", flush=True)
        rows.append(row)
    source_unchanged = source_provenance(SOURCES)["source_sha256"] == sources["source_sha256"]
    for row in rows:
        jet, nested = (row["methods"].get(method, {}) for method in METHODS)
        eligible = source_unchanged and all(result.get("status") == "ok" and len(result["samples_ms"]) == REPEATS
                                            for result in (jet, nested))
        row["speedup_eligible"] = eligible
        if eligible:
            row["nested_over_forward_median_ratio"] = nested["median_ms"] / jet["median_ms"]
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    payload = provenance(schema="omnibias/gaussian-inverse-jets/v1", config={
        "orders": orders, "methods": methods, "mode": "correctness_only" if args.check_only else "measurement",
        "dtype": "float64", "device": "cpu", "torch_threads": 1, "torch_interop_threads": 1,
        "repeats": 0 if args.check_only else REPEATS, "warmup": 0 if args.check_only else WARMUP,
        "max_iterations": ITERATIONS, "tolerance": TOLERANCE, "adjoint": "auto",
        "wall_seconds": args.wall_seconds, "rss_limit_mib": args.rss_mib, "rss_poll_seconds": 0.025,
        "budget_scope": "per fresh subprocess; imports, exact reference, accuracy checks, warmups and all samples",
        "timing_scope": "all ordinary coefficients 0..N, including a fresh base inverse; derivative evaluation only",
        "excluded": "operand/reference setup, accuracy validation, JSON checkpoints, parameter backward and training",
        "execution": "Torch eager; methods/orders run sequentially; first execution then 2 warmups then 9 raw samples",
        "nested_ad": "same implicit inverse API; recursive torch.autograd.grad along the nonlinear scalar path; divide by n!",
        "forward_jet": "same implicit base API followed by its forward coefficient recurrence; no parameter autograd tape",
        "reference": "exact rational GaussianInverseSeries.from_equal_logits, with every order checked",
        "accuracy_rtol": RTOL, "accuracy_atol": ATOL,
        "limitation": "one small scalar family; numerical implicit derivatives; no interval or training-performance claim",
        "rss_limit_note": "observed process RSS; the 25ms sampling interval permits brief overshoot",
    })
    payload["hardware_class"] = f"Torch eager CPU ({os.cpu_count() or 0} logical cores available; one active Torch thread; float64)"
    payload.update(sources, source_unchanged_during_run=source_unchanged)
    payload["operands"] = {
        "centers_exact": [[str(value) for value in row] for row in CENTERS],
        "biases_exact": [str(value) for value in BIASES], "base_point_exact": [str(value) for value in BASE_POINT],
        "viscosity_exact": "1", "time_exact": "1", "target_path_exact": [str(value) for value in PATH],
        "target_path_convention": "v(t)=sum(path[n]*t**n); higher coefficients beyond the stored path are zero",
        "all_runtime_operands_exactly_representable_in_binary64": True,
    }
    payload["rows"] = rows
    print(f"wrote {write_json(name, payload)}")


if __name__ == "__main__":
    main()
