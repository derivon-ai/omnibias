# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Bounded operator benchmark's independent oracle and subprocess contract."""

import importlib.util
import math
import sys
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "benchmarks" / "biharmonic_compiler.py"
MODULE_SPEC = importlib.util.spec_from_file_location("biharmonic_compiler_benchmark", SOURCE)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
benchmark = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(benchmark)


def test_defaults_counts_and_shared_data():
    args = benchmark.parser().parse_args([])
    benchmark.validate(args)
    assert (args.dims, args.hidden, args.batch, args.timeout) == ([2, 4], 4, 2, 20)
    assert benchmark.model_data(1, 2, 4, 2) == benchmark.model_data(1, 2, 4, 2)
    assert benchmark.state_counts(2)["packed_components_per_scalar"] == 9
    assert benchmark.state_counts(4)["packed_components_per_scalar"] == 20
    assert benchmark.state_counts(4)["full_total_degree_four_coefficients"] == 70
    assert benchmark.state_counts(4)["support_groups"] == 10
    for path in benchmark.SOURCE_PATHS:
        assert (SOURCE.parents[1] / path).is_file()


def test_independent_ridge_reference_matches_closed_fourth_derivative():
    # f(x)=3*tanh(2x+0.1)+7; the output offset has zero fourth derivative.
    actual = benchmark.reference(([[2.0]], [0.1], [[3.0]], [7.0]), [[0.25]])[0]
    t = math.tanh(0.6)
    expected = 3 * 2**4 * (16*t - 40*t**3 + 24*t**5)
    assert actual == pytest.approx(expected, rel=3e-15)
    assert benchmark.agreement([actual], [actual])["max_absolute_error"] == 0
    assert benchmark.agreement([float("inf")], [0])["status"] == "nonfinite"


def test_timeout_preserves_completed_forward_checkpoint(monkeypatch):
    args = benchmark.parser().parse_args(["--timeout", "0.1"])
    script = "import time; print('BIHARMONIC_PROGRESS={\"evaluation\":{\"median_ms\":1.25}}', flush=True); time.sleep(5)"
    monkeypatch.setattr(benchmark, "worker_command", lambda *args: [sys.executable, "-c", script])
    result = benchmark.run_case(args, "packed", 2)
    assert result["status"] == "timeout"
    assert result["evaluation"]["median_ms"] == 1.25


def test_worker_failure_returns_data(monkeypatch):
    args = benchmark.parser().parse_args([])
    monkeypatch.setattr(benchmark, "worker_command", lambda *args: [sys.executable, "-c", "raise ValueError('deliberate')"])
    result = benchmark.run_case(args, "folx", 2)
    assert result["status"] == "error"
    assert "deliberate" in result["stderr_tail"]


@pytest.mark.parametrize("flags", [["--dims", "0"], ["--repeats", "0"], ["--output", "../outside.json"]])
def test_invalid_configuration(flags):
    with pytest.raises(ValueError):
        benchmark.validate(benchmark.parser().parse_args(flags))
