# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Budget, metadata and independent-oracle checks; no Inductor compilation."""

import importlib.util
import sys
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "benchmarks" / "jet_compiler.py"
MODULE_SPEC = importlib.util.spec_from_file_location("jet_compiler_benchmark", SOURCE)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
benchmark = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(benchmark)


def test_defaults_and_shared_deterministic_operands():
    args = benchmark.parser().parse_args([])
    benchmark.validate(args)
    assert args.orders == [2, 4, 8, 16, 32]
    assert (args.seed, args.batch, args.hidden, args.repeats, args.threads) == (1, 16, 8, 5, 1)
    assert benchmark.model_data(1, 8, 16) == benchmark.model_data(1, 8, 16)
    assert benchmark.model_data(1, 8, 16) != benchmark.model_data(2, 8, 16)
    for relative in benchmark.SOURCE_PATHS:
        assert (SOURCE.parents[1] / relative).is_file()


def test_nested_over_budget_is_refused_in_isolated_worker():
    args = benchmark.parser().parse_args(["--timeout", "10"])
    result = benchmark.run_case(args, "nested", 16)
    assert result["status"] == "refused"
    assert "budget 8" in result["reason"]


def test_timeout_terminates_worker(monkeypatch):
    args = benchmark.parser().parse_args(["--timeout", "0.02"])
    monkeypatch.setattr(benchmark, "_worker_command", lambda *args: [sys.executable, "-c", "import time; time.sleep(5)"])
    result = benchmark.run_case(args, "compiler", 2)
    assert result["status"] == "timeout"
    assert result["timeout_seconds"] == 0.02


def test_worker_failure_is_structured(monkeypatch):
    args = benchmark.parser().parse_args([])
    monkeypatch.setattr(benchmark, "_worker_command", lambda *args: [sys.executable, "-c", "raise RuntimeError('deliberate')"])
    result = benchmark.run_case(args, "compiler", 2)
    assert result["status"] == "error"
    assert "deliberate" in result["stderr_tail"]


def test_independent_reference_and_accuracy():
    # One affine layer: y=3x+2 has exact derivative three, no jet oracle.
    value, derivative = benchmark.reference([([[3.0]], [2.0])], 0.25, 1)
    assert float(value) == 2.75
    assert float(derivative) == 3.0
    assert benchmark.accuracy(3.0, derivative)["absolute_error"] == 0
    assert benchmark.accuracy(float("inf"), derivative)["status"] == "nonfinite"


def test_timing_resets_before_every_iteration():
    state = {"value": 10}
    seen = []

    def prepare():
        state["value"] = 10

    def action(_):
        seen.append(state["value"])
        state["value"] += 1

    result = benchmark.timing(prepare, action, repeats=2, warmup=1)
    assert seen == [10, 10, 10, 10]
    assert len(result["samples_ms"]) == 2


@pytest.mark.parametrize("flags", [["--orders", "0"], ["--batch", "0"], ["--output", "../outside.json"]])
def test_invalid_configuration(flags):
    with pytest.raises(ValueError):
        benchmark.validate(benchmark.parser().parse_args(flags))
